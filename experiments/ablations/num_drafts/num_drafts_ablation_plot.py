import os
import re
import glob
import json
import math
import argparse

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve

# MarkLLM's own quality analyzers (as requested)
from evaluation.tools.text_quality_analyzer import PPLCalculator, LogDiversityAnalyzer
from transformers import AutoModelForCausalLM, AutoTokenizer
from evaluation.tools.success_rate_calculator import DynamicThresholdSuccessRateCalculator
from experiments.common.io import load_lines, last_by_idx, strip_prompt, sent_split, akey, load_kv, append
from experiments.common. detect import score_from_detect
from experiments.common.metrics import auroc, tpr_at_fpr, safe_ppl, safe_div
from experiments.common.attacks import CropAttack, MarianTranslator, SmallParaphraser, build_attacks, apply_attack

DIR = "results/ablation_num_seq"
ORACLE_PATH = "facebook/opt-2.7b"     # oracle LM for perplexity (paper uses opt-2.7b)




def load_positive_scores(path):
    finite, n_fail, times = [], 0, []
    for r in last_by_idx(path).values():
        if r.get("error"):
            continue
        s = score_from_detect(r.get("detect_watermarked"))
        if s is None:
            n_fail += 1
        else:
            finite.append(s)
        if r.get("gen_time_wm") is not None:
            times.append(float(r["gen_time_wm"]))
    return finite, n_fail, times


def load_negative_scores(path, which):
    key = "detect_unwatermarked" if which == "unwatermarked" else "detect_natural"
    finite = []
    for r in last_by_idx(path).values():
        if r.get("error"):
            continue
        s = score_from_detect(r.get(key))
        if s is not None:
            finite.append(s)
    return finite


def sanitize(pos_finite, n_fail, neg):
    all_fin = list(pos_finite) + list(neg)
    if not all_fin:
        return [], []
    floor = min(all_fin) - 1.0
    return list(pos_finite) + [floor] * n_fail, list(neg)


def load_texts(path, key):
    return {i: r.get(key) for i, r in last_by_idx(path).items()
            if not r.get("error") and r.get(key)}


def build_analyzers(device, use_ppl, oracle_path):
    logdiv = LogDiversityAnalyzer()
    ppl = None
    model = None
    if use_ppl:
        print(f"loading PPL oracle: {oracle_path} ...")
        model = AutoModelForCausalLM.from_pretrained(oracle_path).to(device)
        tok = AutoTokenizer.from_pretrained(oracle_path)
        ppl = PPLCalculator(model=model, tokenizer=tok, device=device)
    return ppl, logdiv, model





def compute_quality(sources, ppl, logdiv, cache_path):
    """sources: list of datasets for each num_seq (source_name, {idx: text}). Cache keyed 'source:idx'."""
    cache = {}
    for r in load_lines(cache_path):
        if "source" in r and "idx" in r:
            cache[f"{r['source']}:{r['idx']}"] = r
    with open(cache_path, "a") as fout:
        for source, texts in sources:
            todo = [(i, t) for i, t in texts.items() if f"{source}:{i}" not in cache]
            if todo:
                print(f"  quality[{source}]: {len(cache)} cached, {len(todo)} to score")
            for k, (i, t) in enumerate(todo, 1):
                rec = {"source": source, "idx": i,
                       "ppl": safe_ppl(ppl, t), "logdiv": safe_div(logdiv, t)}
                fout.write(json.dumps(rec) + "\n")
                fout.flush()
                os.fsync(fout.fileno())
                cache[f"{source}:{i}"] = rec
                if k % 25 == 0 or k == len(todo):
                    print(f"    {source} {k}/{len(todo)}")
    return cache


def quality_for(cache, source, field):
    out = {}
    for k, r in cache.items():
        if r.get("source") == source and r.get(field) is not None:
            out[r["idx"]] = float(r[field])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=DIR)
    ap.add_argument("--neg", choices=["unwatermarked", "natural"], default="unwatermarked")
    ap.add_argument("--oracle", default=ORACLE_PATH)
    ap.add_argument("--no_ppl", action="store_true", help="skip perplexity (log-diversity only)")
    args = ap.parse_args()

    neg_path = os.path.join(args.dir, "SemMax_negatives.jsonl")
    nseq_paths = sorted(glob.glob(os.path.join(args.dir, "SemMax_nseq*.jsonl")))
    if not nseq_paths:
        print("!! no SemMax_nseq*.jsonl found")
        return

    # ---- quality (MarkLLM analyzers on saved texts) ----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ppl, logdiv, oracle_model = build_analyzers(device, use_ppl=not args.no_ppl, oracle_path=args.oracle)

    neg_key = "unwatermarked_text" if args.neg == "unwatermarked" else "natural_text"
    sources = [("negatives", load_texts(neg_path, neg_key))]
    file_by_n = {}
    for path in nseq_paths:
        m = re.search(r"nseq(\d+)", os.path.basename(path))
        if not m:
            continue
        n = int(m.group(1))
        file_by_n[n] = path
        sources.append((f"nseq{n}", load_texts(path, "watermarked_text")))

    qcache = compute_quality(sources, ppl, logdiv,
                             os.path.join(args.dir, "quality_cache.jsonl"))
    if oracle_model is not None:
        del oracle_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    unwm_ppl = quality_for(qcache, "negatives", "ppl")

    # ---- detectability + cost + quality per num_seq ----
    neg_scores = load_negative_scores(neg_path, args.neg)
    rows = []
    for n in sorted(file_by_n):
        path = file_by_n[n]
        finite, n_fail, times = load_positive_scores(path)
        pos, neg = sanitize(finite, n_fail, neg_scores)
        if not pos or not neg:
            print(f"  num_seq={n}: missing scores, skipping")
            continue

        wm_ppl = quality_for(qcache, f"nseq{n}", "ppl")
        wm_div = quality_for(qcache, f"nseq{n}", "logdiv")
        shared = set(wm_ppl) & set(unwm_ppl)
        ratios = [wm_ppl[i] / unwm_ppl[i] for i in shared if unwm_ppl[i] > 0]
        baseline_ppl = ( float(np.mean(list(unwm_ppl.values()))) if unwm_ppl else float("nan"))

        rows.append({
            "num_seq": n,
            "n_pos": len(pos), "n_neg": len(neg), "n_fail": n_fail,
            "AUROC": auroc(pos, neg),
            "TPR@FPR=1%": tpr_at_fpr(pos,n_fail, neg, 0.01),
            "TPR@FPR=5%": tpr_at_fpr(pos,n_fail, neg, 0.05),
            "mean_gen_time_s": round(float(np.mean(times)), 3) if times else float("nan"),
            "rel_ppl": round(float(np.mean(ratios)), 4) if ratios else float("nan"),
            "mean_ppl_wm": round(float(np.mean(list(wm_ppl.values()))), 3) if wm_ppl else float("nan"),
            "mean_logdiv_wm": round(float(np.mean(list(wm_div.values()))), 4) if wm_div else float("nan"),
        })
        r = rows[-1]
        print(f"  num_seq={n:>3}: AUROC={r['AUROC']:.3f} TPR@5%={r['TPR@FPR=5%']:.3f} "
              f"time={r['mean_gen_time_s']}s PPL={r['mean_ppl_wm']} logdiv={r['mean_logdiv_wm']}")

    if not rows:
        print("!! nothing to plot")
        return
    rows.sort(key=lambda x: x["num_seq"])

    # ---- CSV ----
    import csv
    csv_path = os.path.join(args.dir, "ablation_summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {csv_path}")

    # ---- plots ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs  = [r["num_seq"] for r in rows]
    au  = [r["AUROC"] for r in rows]
    t5  = [r["TPR@FPR=5%"] for r in rows]
    t1  = [r["TPR@FPR=1%"] for r in rows]
    tim = [r["mean_gen_time_s"] for r in rows]
    ppl = [r["mean_ppl_wm"] for r in rows]
    r_ppl = [r["rel_ppl"] for r in rows]
    div = [r["mean_logdiv_wm"] for r in rows]

    # 1) detectability vs num_seq
    plt.figure(figsize=(5.4, 4))
    plt.plot(xs, au, "o-", label="AUROC")
    plt.plot(xs, t5, "s-", label="TPR@FPR=5%")
    plt.plot(xs, t1, "^--", label="TPR@FPR=1%", alpha=0.7)
    plt.xlabel("Number of drafts (num_seq)"); plt.ylabel("Detectability")
    plt.ylim(0.4, 1.02); plt.grid(alpha=0.3); plt.legend()
    plt.title("SemMax detectability vs. num_seq"); plt.tight_layout()
    p = os.path.join(args.dir, "detectability_vs_numseq.png")
    plt.savefig(p, dpi=150); plt.close(); print(f"wrote {p}")

    # 2) time vs num_seq 
    plt.figure(figsize=(5.4, 4))
    plt.plot(xs, tim, "o-", color="tab:red")
    plt.xlabel("Number of drafts (num_seq)"); plt.ylabel("Mean generation time (s)")
    plt.grid(alpha=0.3); plt.title("SemMax generation cost vs. num_seq"); plt.tight_layout()
    p = os.path.join(args.dir, "time_vs_numseq.png")
    plt.savefig(p, dpi=150); plt.close(); print(f"wrote {p}")


    # 3) quality vs num_seq (relative PPL + log-diversity)
    fig, ax1 = plt.subplots(figsize=(5.4, 4))
    have_ppl = any(not math.isnan(v) for v in r_ppl)
    if have_ppl:
        ax1.plot(xs, r_ppl, "o-", color="tab:purple", label=" PPL")
        ax1.axhline(1.0, color="gray", ls=":", lw=0.8)
        ax1.set_ylabel("Relative perplexity (lower is better)", color="tab:purple")
        ax1.tick_params(axis="y", labelcolor="tab:purple")
    ax1.set_xlabel("Number of drafts (num_seq)")
    ax2 = ax1.twinx()
    ax2.plot(xs, div, "s--", color="tab:blue", label="log-diversity")
    ax2.set_ylabel("Log-diversity ↑", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")
    plt.title("SemMax text quality vs. num_seq"); fig.tight_layout()
    p = os.path.join(args.dir, "quality_vs_numseq.png")
    plt.savefig(p, dpi=150); plt.close(); print(f"wrote {p}")


   
    # PPL vs num_seq
    plt.figure(figsize=(5.4, 4))
    plt.plot(xs,r_ppl,"o-",linewidth=2,label="Watermarked")
    # plt.axhline(baseline_ppl,linestyle="--",linewidth=2,color="gray",label="Unwatermarked baseline")
    # plt.ylim(min(min(r_ppl), baseline_ppl) - 0.002,
    #      max(max(r_ppl), baseline_ppl) + 0.002)
    plt.xlabel("Number of drafts (num_seq)")
    plt.ylabel("Mean Relative perplexity")
    plt.title("SemMax perplexity vs. num_seq")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    p = os.path.join(args.dir, "ppl_vs_numseq.png")
    plt.savefig(p, dpi=150)
    plt.close()
    print(f"wrote {p}")

    # ---------------------------------------------------------
    # Log-diversity vs num_seq
    # ---------------------------------------------------------
    plt.figure(figsize=(5.4, 4))
    plt.plot(xs,div,"s-",linewidth=2,label="Watermarked")
    plt.xlabel("Number of drafts (num_seq)")
    plt.ylabel("Mean log diversity")
    plt.title("SemMax log diversity vs. num_seq")
    plt.grid(alpha=0.3)

    plt.tight_layout()

    p = os.path.join(args.dir, "logdiv_vs_numseq.png")
    plt.savefig(p, dpi=150)
    plt.close()

    print(f"wrote {p}")


        


if __name__ == "__main__":
    main()