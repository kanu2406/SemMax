
import os
import json
import math
import argparse

import numpy as np
import torch

from evaluation.tools.text_quality_analyzer import (
    PPLCalculator, LogDiversityAnalyzer,
    BLEUCalculator, ROUGE1Calculator, ROUGE2Calculator, ROUGELCalculator,
    BERTScoreCalculator,
)
from transformers import AutoModelForCausalLM, AutoTokenizer
from experiments.common.metrics import relative_ppl, build_analyzers, score_text
from experiments.common.io import last_by_idx, load_cache

GEN_DIR         = "results/generations"
ORACLE_PATH     = "facebook/opt-2.7b"      # PPL oracle (different, stronger than generator)
BERTSCORE_MODEL = "bert-base-uncased"
METHODS         = ["SemMax", "Watermax", "KSEMSTAMP"]
ROLES           = [("watermarked", "watermarked_text"), ("unwatermarked", "unwatermarked_text")]

DIRECT_METRICS     = ["ppl", "log_diversity"]
REFERENCED_METRICS = ["bleu", "rouge1", "rouge2", "rougeL", "bertscore"]
ALL_METRICS        = DIRECT_METRICS + REFERENCED_METRICS




def safe(fn, *args):
    try:
        v = fn(*args)
        return float(v) if v is not None else None
    except Exception:
        return None





def mean_ignore_none(vals):
    vs = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return (float(np.mean(vs)), len(vs)) if vs else (float("nan"), 0)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=GEN_DIR)
    ap.add_argument("--methods", nargs="+", default=METHODS)
    ap.add_argument("--oracle", default=ORACLE_PATH)
    ap.add_argument("--bertscore_model", default=BERTSCORE_MODEL)
    ap.add_argument("--no_ppl", action="store_true")
    ap.add_argument("--no_bertscore", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    records = {}
    for m in args.methods:
        recs = last_by_idx(os.path.join(args.dir, f"{m}.jsonl"))
        if recs:
            records[m] = recs
        else:
            print(f"!! no generations for {m}, skipping")
    if not records:
        return

    cache_path = os.path.join(args.dir, "quality_all_cache.jsonl")
    cache = load_cache(cache_path)

    # what still needs scoring?
    todo = []
    for method, recs in records.items():
        for idx, r in recs.items():
            if r.get("error"):
                continue
            ref = r.get("natural_text")
            for role, key in ROLES:
                ck = f"{method}:{role}:{idx}"
                if ck not in cache and r.get(key):
                    todo.append((method, role, idx, r[key], ref, ck))

    if todo:
        A, heavy = build_analyzers(device, not args.no_ppl, not args.no_bertscore,
                                   args.oracle, args.bertscore_model)
        print(f"scoring {len(todo)} texts with: {sorted(A.keys())}")
        with open(cache_path, "a") as fout:
            for k, (method, role, idx, text, ref, ck) in enumerate(todo, 1):
                with torch.no_grad():
                    scores = score_text(A, text, ref)
                fout.write(json.dumps({"method": method, "role": role, "idx": idx,
                                       "scores": scores}) + "\n")
                fout.flush(); os.fsync(fout.fileno())
                cache[ck] = scores
                if k % 20 == 0 or k == len(todo):
                    print(f"  {k}/{len(todo)}  ({method}/{role}/{idx})")
        for m in heavy:
            del m
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        print("all texts already cached.")

    # --- aggregate per method ---
    import csv
    cols = ["method", "n"]
    for m in ALL_METRICS:
        cols += [f"{m}_wm", f"{m}_unwm"]
    cols += ["rel_ppl"]

    rows = []
    for method in records:
        row = {"method": method}
        n_wm = 0
        for metric in ALL_METRICS:
            wm_vals = [cache.get(f"{method}:watermarked:{i}", {}).get(metric) for i in records[method]]
            un_vals = [cache.get(f"{method}:unwatermarked:{i}", {}).get(metric) for i in records[method]]
            wm_mean, c = mean_ignore_none(wm_vals)
            un_mean, _ = mean_ignore_none(un_vals)
            row[f"{metric}_wm"] = round(wm_mean, 4) if not math.isnan(wm_mean) else ""
            row[f"{metric}_unwm"] = round(un_mean, 4) if not math.isnan(un_mean) else ""
            n_wm = max(n_wm, c)
        row["rel_ppl"] = round(relative_ppl(cache, method), 4)
        row["n"] = n_wm
        rows.append(row)

    csv_path = os.path.join(args.dir, "quality_all_summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {csv_path}")

    # readable table (watermarked scores + relative PPL)
    print("\n=== watermarked-text quality (means) ===")
    hdr = f"{'method':12s}" + "".join(f"{m:>12s}" for m in ALL_METRICS) + f"{'rel_ppl':>10s}"
    print(hdr)
    for r in rows:
        line = f"{r['method']:12s}"
        for m in ALL_METRICS:
            v = r.get(f"{m}_wm", "")
            line += f"{(v if v != '' else '-'):>12}"
        line += f"{r['rel_ppl']:>10}"
        print(line)


if __name__ == "__main__":
    main()