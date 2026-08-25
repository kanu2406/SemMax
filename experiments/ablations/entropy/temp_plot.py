import os
import re
import glob
import json
import math
import argparse

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from evaluation.tools.text_quality_analyzer import PPLCalculator
from transformers import AutoModelForCausalLM, AutoTokenizer
from evaluation.tools.success_rate_calculator import DynamicThresholdSuccessRateCalculator
from experiments.common.io import load_lines, last_by_idx, strip_prompt, sent_split, akey, load_kv, append
from experiments.common.detect import score_from_detect
from experiments.common.metrics import auroc, tpr_at_fpr, safe_ppl

DIR = "results/temp_var"
ORACLE_PATH = "facebook/opt-2.7b"
METHODS = ["SemMax", "Watermax", "KSEMSTAMP"]


def get_scores(recs, key):
    finite, n_fail = [], 0
    for r in recs.values():
        if r.get("error"):
            continue
        s = score_from_detect(r.get(key))
        if s is None:
            n_fail += 1
        else:
            finite.append(s)
    return finite, n_fail





def discover(dir_):
    """{method: {temp: path}} from filenames method_temp{T}.jsonl."""
    out = {}
    for path in glob.glob(os.path.join(dir_, "*_temp*.jsonl")):
        m = re.match(r"(.+)_temp([0-9.]+)\.jsonl$", os.path.basename(path))
        if not m:
            continue
        method, temp = m.group(1), float(m.group(2))
        out.setdefault(method, {})[temp] = path
    return out


# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=DIR)
    ap.add_argument("--oracle", default=ORACLE_PATH)
    args = ap.parse_args()

    files = discover(args.dir)
    if not files:
        print(f"!! no *_temp*.jsonl in {args.dir} — run temp_variation.py first")
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cache_path = os.path.join(args.dir, "quality_cache.jsonl")
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    cache[f"{r['method']}:{r['temp']}:{r['role']}:{r['idx']}"] = r.get("ppl")

    need = []
    recs_cache = {}
    for method, temps in files.items():
        for temp, path in temps.items():
            recs = last_by_idx(path)
            recs_cache[(method, temp)] = recs
            for idx, r in recs.items():
                if r.get("error"):
                    continue
                for role, key in (("wm", "watermarked_text"), ("unwm", "unwatermarked_text")):
                    ck = f"{method}:{temp}:{role}:{idx}"
                    if ck not in cache and r.get(key):
                        need.append((method, temp, role, idx, r[key], ck))

    if need:
        print(f"loading PPL oracle {args.oracle} ... ({len(need)} texts)")
        model = AutoModelForCausalLM.from_pretrained(args.oracle).to(device)
        tok = AutoTokenizer.from_pretrained(args.oracle)
        ppl = PPLCalculator(model=model, tokenizer=tok, device=device)
        with open(cache_path, "a") as fout:
            for k, (method, temp, role, idx, text, ck) in enumerate(need, 1):
                v = safe_ppl(ppl, text)
                fout.write(json.dumps({"method": method, "temp": temp, "role": role,
                                       "idx": idx, "ppl": v}) + "\n")
                fout.flush(); os.fsync(fout.fileno())
                cache[ck] = v
                if k % 25 == 0 or k == len(need):
                    print(f"  ppl {k}/{len(need)}")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --- per (method, temp): AUROC + rel PPL ---
    results = {}   # method -> list of (temp, auroc, rel_ppl)
    for method, temps in files.items():
        pts = []
        for temp in sorted(temps):
            recs = recs_cache[(method, temp)]
            pos_f, n_fail = get_scores(recs, "detect_watermarked")
            neg_f, _ = get_scores(recs, "detect_unwatermarked")
            au = auroc(pos_f, neg_f, n_fail)
            tpr1 = tpr_at_fpr(pos_f, n_fail, neg_f, 0.01)
            tpr5 = tpr_at_fpr(pos_f, n_fail, neg_f, 0.05)
            wm = {i: cache.get(f"{method}:{temp}:wm:{i}") for i in recs}
            un = {i: cache.get(f"{method}:{temp}:unwm:{i}") for i in recs}
            ratios = [wm[i] / un[i] for i in recs
                      if wm.get(i) and un.get(i) and un[i] > 0]
            rp = float(np.mean(ratios)) if ratios else float("nan")
            pts.append((temp, au, rp, tpr1, tpr5))
            print(f"  {method:10s} T={temp}: AUROC={au:.3f} relPPL={rp:.3f}")
        results[method] = pts

    # --- CSV ---
    import csv
    with open(os.path.join(args.dir, "temp_var_summary.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["method", "temp", "AUROC", "rel_ppl", "tpr1", "tpr5"])
        for method, pts in results.items():
            for temp, au, rp, tp1, tp5 in pts:
                w.writerow([method, temp, round(au, 4), round(rp, 4), round(tp1, 4), round(tp5, 4)])

    # --- plot ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    markers = {"SemMax": "o", "Watermax": "s", "KSEMSTAMP": "^"}

   

    # ============================================================
    # 1. AUROC vs. Temperature
    # ============================================================
    fig, ax = plt.subplots(figsize=(5.5, 4.4))

    for method, pts in results.items():
        pts = sorted(pts, key=lambda x: x[0])
        temps = [p[0] for p in pts]
        aus = [p[1] for p in pts]

        mk = markers.get(method, "d")
        ax.plot(temps, aus, mk + "-", lw=2, label=method)

    ax.set_xlabel(r"Temperature $\theta$")
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.4, 1.02)
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title("AUROC vs. Temperature")

    fig.tight_layout()
    p = os.path.join(args.dir, "auroc_vs_temp.png")
    fig.savefig(p, dpi=150)
    plt.close()
    print(f"wrote {p}")


    # ============================================================
    # 2. AUROC vs. Quality
    # ============================================================
    fig, ax = plt.subplots(figsize=(5.5, 4.4))

    for method, pts in results.items():
        pts = sorted(pts, key=lambda x: x[0])
        rps = [p[2] for p in pts]
        aus = [p[1] for p in pts]

        mk = markers.get(method, "d")
        ax.plot(rps, aus, mk + "-", lw=2, label=method)

    ax.axvline(1.0, color="gray", ls=":", lw=0.8)
    ax.set_xlabel("Relative perplexity")
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.4, 1.02)
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title("AUROC vs. Quality")

    fig.tight_layout()
    p = os.path.join(args.dir, "auroc_vs_quality.png")
    fig.savefig(p, dpi=150)
    plt.close()
    print(f"wrote {p}")


    # ============================================================
    # 3. TPR @ 1% FPR vs. Temperature
    # ============================================================
    fig, ax = plt.subplots(figsize=(5.5, 4.4))

    for method, pts in results.items():
        pts = sorted(pts, key=lambda x: x[0])
        temps = [p[0] for p in pts]
        tpr1s = [p[3] for p in pts]

        mk = markers.get(method, "d")
        ax.plot(temps, tpr1s, mk + "-", lw=2, label=method)

    ax.set_xlabel(r"Temperature $\theta$")
    ax.set_ylabel("TPR @ 1% FPR")
    ax.set_ylim(0.0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title("TPR @ 1% FPR vs. Temperature")

    fig.tight_layout()
    p = os.path.join(args.dir, "tpr1_vs_temp.png")
    fig.savefig(p, dpi=150)
    plt.close()
    print(f"wrote {p}")


    # ============================================================
    # 4. TPR @ 1% FPR vs. Quality
    # ============================================================
    fig, ax = plt.subplots(figsize=(5.5, 4.4))

    for method, pts in results.items():
        pts = sorted(pts, key=lambda x: x[0])
        rps = [p[2] for p in pts]
        tpr1s = [p[3] for p in pts]

        mk = markers.get(method, "d")
        ax.plot(rps, tpr1s, mk + "-", lw=2, label=method)

    ax.axvline(1.0, color="gray", ls=":", lw=0.8)
    ax.set_xlabel("Relative perplexity")
    ax.set_ylabel("TPR @ 1% FPR")
    ax.set_ylim(0.0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title("TPR @ 1% FPR vs. Quality")

    fig.tight_layout()
    p = os.path.join(args.dir, "tpr1_vs_quality.png")
    fig.savefig(p, dpi=150)
    plt.close()
    print(f"wrote {p}")



    # ============================================================
    # 3. TPR @ 1% FPR vs. Temperature
    # ============================================================
    fig, ax = plt.subplots(figsize=(5.5, 4.4))

    for method, pts in results.items():
        pts = sorted(pts, key=lambda x: x[0])
        temps = [p[0] for p in pts]
        tpr5s = [p[4] for p in pts]

        mk = markers.get(method, "d")
        ax.plot(temps, tpr5s, mk + "-", lw=2, label=method)

    ax.set_xlabel(r"Temperature $\theta$")
    ax.set_ylabel("TPR @ 5% FPR")
    ax.set_ylim(0.0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title("TPR @ 5% FPR vs. Temperature")

    fig.tight_layout()
    p = os.path.join(args.dir, "tpr5_vs_temp.png")
    fig.savefig(p, dpi=150)
    plt.close()
    print(f"wrote {p}")


    # ============================================================
    # 4. TPR @ 5% FPR vs. Quality
    # ============================================================
    fig, ax = plt.subplots(figsize=(5.5, 4.4))

    for method, pts in results.items():
        pts = sorted(pts, key=lambda x: x[0])
        rps = [p[2] for p in pts]
        tpr5s = [p[4] for p in pts]

        mk = markers.get(method, "d")
        ax.plot(rps, tpr5s, mk + "-", lw=2, label=method)

    ax.axvline(1.0, color="gray", ls=":", lw=0.8)
    ax.set_xlabel("Relative perplexity")
    ax.set_ylabel("TPR @ 5% FPR")
    ax.set_ylim(0.0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title("TPR @ 5% FPR vs. Quality")

    fig.tight_layout()
    p = os.path.join(args.dir, "tpr5_vs_quality.png")
    fig.savefig(p, dpi=150)
    plt.close()
    print(f"wrote {p}")


    # ============================================================
    # 4. Temp vs. Quality
    # ============================================================
    fig, ax = plt.subplots(figsize=(5.5, 4.4))

    for method, pts in results.items():
        pts = sorted(pts, key=lambda x: x[0])
        temps = [p[0] for p in pts]
        rps = [p[2] for p in pts]

        mk = markers.get(method, "d")
        ax.plot(temps, rps, mk + "-", lw=2, label=method)

    ax.axvline(1.0, color="gray", ls=":", lw=0.8)
    ax.set_xlabel("Temperature")
    ax.set_ylabel("Relative Perplexity")
    ax.set_ylim(0.0, 1.5)
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title("Temperature vs. Quality")

    fig.tight_layout()
    p = os.path.join(args.dir, "temp_vs_quality.png")
    fig.savefig(p, dpi=150)
    plt.close()
    print(f"wrote {p}")




























    # fig, (axa, axb) = plt.subplots(1, 2, figsize=(11, 4.4))
    # fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    # (ax1, ax2), (ax3, ax4) = axes

    # for method, pts in results.items():
    #     pts = sorted(pts, key=lambda x: x[0])
    #     temps = [p[0] for p in pts]
    #     aus   = [p[1] for p in pts]
    #     rps   = [p[2] for p in pts]
    #     tpr1s  = [p[3] for p in pts]
    #     tpr5s  = [p[4] for p in pts]

    #     mk = markers.get(method, "d")
    #     # axa.plot(temps, aus, mk + "-", label=method)
    #     # axb.plot(rps, aus, mk + "-", label=method)


    #     # AUROC
    #     ax1.plot(temps, aus, mk + "-", lw=2, label=method)
    #     ax2.plot(rps, aus, mk + "-", lw=2, label=method)

    #     # TPR @ 1% FPR
    #     ax3.plot(temps, tpr1s, mk + "-", lw=2, label=method)
    #     ax4.plot(rps, tpr1s, mk + "-", lw=2, label=method)

    
    # # ---------- AUROC ----------
    # ax1.set_xlabel(r"Temperature $\theta$")
    # ax1.set_ylabel("AUROC")
    # ax1.set_ylim(0.4, 1.02)
    # ax1.grid(alpha=0.3)
    # ax1.legend()
    # ax1.set_title("(a) AUROC vs. Temperature")

    # ax2.axvline(1.0, color="gray", ls=":", lw=0.8)
    # ax2.set_xlabel("Relative perplexity")
    # ax2.set_ylabel("AUROC")
    # ax2.set_ylim(0.4, 1.02)
    # ax2.grid(alpha=0.3)
    # ax2.legend()
    # ax2.set_title("(b) AUROC vs. Quality")

    # # ---------- TPR ----------
    # ax3.set_xlabel(r"Temperature $\theta$")
    # ax3.set_ylabel("TPR @ 1% FPR")
    # ax3.set_ylim(0.0, 1.02)
    # ax3.grid(alpha=0.3)
    # ax3.legend()
    # ax3.set_title("(c) TPR @ 1% FPR vs. Temperature")

    # ax4.axvline(1.0, color="gray", ls=":", lw=0.8)
    # ax4.set_xlabel("Relative perplexity")
    # ax4.set_ylabel("TPR @ 1% FPR")
    # # ax4.set_ylim(0.0, 1.02)
    # ax4.grid(alpha=0.3)
    # ax4.legend()
    # ax4.set_title("(d) TPR @ 1% FPR vs. Quality")

    # fig.tight_layout()
    # p = os.path.join(args.dir, "detectability_summary.png")
    # fig.savefig(p, dpi=150)
    # plt.close()

    # print(f"\nwrote {p}")



if __name__ == "__main__":
    main()