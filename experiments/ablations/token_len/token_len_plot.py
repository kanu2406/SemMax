import os
import re
import glob
import json
import math
import argparse

import numpy as np
import torch

from evaluation.tools.text_quality_analyzer import PPLCalculator
from transformers import AutoModelForCausalLM, AutoTokenizer
from experiments.common.io import load_lines, last_by_idx
from experiments.common.detect import score_from_detect
from experiments.common.metrics import auroc, tpr_at_fpr, safe_ppl

DIR = "results/token_len"
ORACLE_PATH = "facebook/opt-2.7b"
METHODS = ["SemMax", "Watermax", "KSEMSTAMP"]
METRICS = ["AUROC", "TPR@1%", "TPR@5%"]     # detectability metrics to plot

# --------------------------------------------------------------------------- #

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


def mean_tokens(recs):
    vals = [r["n_tokens_wm"] for r in recs.values()
            if not r.get("error") and r.get("n_tokens_wm")]
    return float(np.mean(vals)) if vals else float("nan")


def discover(dir_):
    """{method: {length: path}} from filenames method_len{L}.jsonl."""
    out = {}
    for path in glob.glob(os.path.join(dir_, "*_len*.jsonl")):
        m = re.match(r"(.+)_len(\d+)\.jsonl$", os.path.basename(path))
        if not m:
            continue
        method, length = m.group(1), int(m.group(2))
        out.setdefault(method, {})[length] = path
    return out


# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=DIR)
    ap.add_argument("--oracle", default=ORACLE_PATH)
    ap.add_argument("--metric", choices=METRICS + ["all"], default="all",
                    help="which detectability metric(s) to plot")
    ap.add_argument("--no_ppl", action="store_true")
    args = ap.parse_args()

    files = discover(args.dir)
    if not files:
        print(f"!! no *_len*.jsonl in {args.dir} — run token_len_generate.py first")
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- PPL cache (method:length:role:idx) ----
    cache_path = os.path.join(args.dir, "quality_cache.jsonl")
    cache = {}
    if os.path.exists(cache_path):
        for r in load_lines(cache_path):
            cache[f"{r['method']}:{r['length']}:{r['role']}:{r['idx']}"] = r.get("ppl")

    recs_cache, need = {}, []
    for method, lengths in files.items():
        for length, path in lengths.items():
            recs = last_by_idx(path)
            recs_cache[(method, length)] = recs
            if args.no_ppl:
                continue
            for idx, r in recs.items():
                if r.get("error"):
                    continue
                for role, key in (("wm", "watermarked_text"), ("unwm", "unwatermarked_text")):
                    ck = f"{method}:{length}:{role}:{idx}"
                    if ck not in cache and r.get(key):
                        need.append((method, length, role, idx, r[key], ck))

    if need:
        print(f"loading PPL oracle {args.oracle} ... ({len(need)} texts)")
        model = AutoModelForCausalLM.from_pretrained(args.oracle).to(device)
        tok = AutoTokenizer.from_pretrained(args.oracle)
        ppl = PPLCalculator(model=model, tokenizer=tok, device=device)
        with open(cache_path, "a") as fout:
            for k, (method, length, role, idx, text, ck) in enumerate(need, 1):
                v = safe_ppl(ppl, text)
                fout.write(json.dumps({"method": method, "length": length, "role": role,
                                       "idx": idx, "ppl": v}) + "\n")
                fout.flush(); os.fsync(fout.fileno())
                cache[ck] = v
                if k % 25 == 0 or k == len(need):
                    print(f"  ppl {k}/{len(need)}")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- per (method, length): detectability metrics + rel PPL + actual tokens ----
    results = {}   # method -> list of dict rows
    for method, lengths in files.items():
        pts = []
        for length in sorted(lengths):
            recs = recs_cache[(method, length)]
            pos_f, n_fail = get_scores(recs, "detect_watermarked")
            neg_f, _ = get_scores(recs, "detect_unwatermarked")
            if not pos_f and n_fail == 0:
                print(f"  {method} len={length}: no positives, skipping")
                continue
            wm = {i: cache.get(f"{method}:{length}:wm:{i}") for i in recs}
            un = {i: cache.get(f"{method}:{length}:unwm:{i}") for i in recs}
            ratios = [wm[i] / un[i] for i in recs if wm.get(i) and un.get(i) and un[i] > 0]
            row = {
                "length": length,
                "tokens": mean_tokens(recs),
                "AUROC": auroc(pos_f, neg_f, n_fail),
                "TPR@1%": tpr_at_fpr(pos_f, n_fail, neg_f, 0.01),
                "TPR@5%": tpr_at_fpr(pos_f, n_fail, neg_f, 0.05),
                "rel_ppl": float(np.mean(ratios)) if ratios else float("nan"),
                "n_pos": len(pos_f) + n_fail, "n_fail": n_fail,
            }
            pts.append(row)
            print(f"  {method:10s} len={length:>3} (~{row['tokens']:.0f} tok): "
                  f"AUROC={row['AUROC']:.3f} TPR@1%={row['TPR@1%']:.3f} "
                  f"TPR@5%={row['TPR@5%']:.3f} relPPL={row['rel_ppl']:.3f}")
        if pts:
            results[method] = pts

    if not results:
        print("!! nothing to plot")
        return

    # ---- CSV ----
    import csv
    csv_path = os.path.join(args.dir, "token_len_summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "length", "tokens", "AUROC", "TPR@1%", "TPR@5%",
                    "rel_ppl", "n_pos", "n_fail"])
        for method, pts in results.items():
            for r in pts:
                w.writerow([method, r["length"], round(r["tokens"], 1),
                            round(r["AUROC"], 4), round(r["TPR@1%"], 4), round(r["TPR@5%"], 4),
                            round(r["rel_ppl"], 4), r["n_pos"], r["n_fail"]])
    print(f"\nwrote {csv_path}")

    # ---- plots ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    markers = {"SemMax": "o", "Watermax": "s", "KSEMSTAMP": "^"}

    def plot_metric(metric, fname, ylabel, ylim=None):
        fig, ax = plt.subplots(figsize=(5.5, 4.4))
        for method, pts in results.items():
            pts = sorted(pts, key=lambda x: x["tokens"])
            xs = [p["tokens"] for p in pts]
            ys = [p[metric] for p in pts]
            mk = markers.get(method, "d")
            ax.plot(xs, ys, mk + "-", lw=2, label=method)
        ax.set_xlabel("Mean tokens generated")
        ax.set_ylabel(ylabel)
        if ylim:
            ax.set_ylim(*ylim)
        ax.grid(alpha=0.3); ax.legend()
        ax.set_title(f"{ylabel} vs. length")
        fig.tight_layout()
        p = os.path.join(args.dir, fname)
        fig.savefig(p, dpi=150); plt.close(); print(f"wrote {p}")

    # detectability: one plot per requested metric
    to_plot = METRICS if args.metric == "all" else [args.metric]
    fname_of = {"AUROC": "auroc_vs_len.png",
                "TPR@1%": "tpr1_vs_len.png",
                "TPR@5%": "tpr5_vs_len.png"}
    for m in to_plot:
        plot_metric(m, fname_of[m], m, ylim=(0.0, 1.02))

    # quality: relative PPL only
    fig, ax = plt.subplots(figsize=(5.5, 4.4))
    for method, pts in results.items():
        pts = sorted(pts, key=lambda x: x["tokens"])
        xs = [p["tokens"] for p in pts]
        ys = [p["rel_ppl"] for p in pts]
        mk = markers.get(method, "d")
        ax.plot(xs, ys, mk + "-", lw=2, label=method)
    ax.axhline(1.0, color="gray", ls=":", lw=0.8)
    ax.set_xlabel("Mean tokens generated")
    ax.set_ylabel("Relative perplexity (↓ better)")
    ax.grid(alpha=0.3); ax.legend()
    ax.set_title("Text quality vs. length")
    fig.tight_layout()
    p = os.path.join(args.dir, "ppl_vs_len.png")
    fig.savefig(p, dpi=150); plt.close(); print(f"wrote {p}")


if __name__ == "__main__":
    main()