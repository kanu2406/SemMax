
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

GEN_DIR         = "generations"
ORACLE_PATH     = "facebook/opt-2.7b"      # PPL oracle (different, stronger than generator)
BERTSCORE_MODEL = "bert-base-uncased"
METHODS         = ["SemMax", "Watermax", "KSEMSTAMP"]
ROLES           = [("watermarked", "watermarked_text"), ("unwatermarked", "unwatermarked_text")]

DIRECT_METRICS     = ["ppl", "log_diversity"]
REFERENCED_METRICS = ["bleu", "rouge1", "rouge2", "rougeL", "bertscore"]
ALL_METRICS        = DIRECT_METRICS + REFERENCED_METRICS

def last_by_idx(path):
    last = {}
    if not os.path.exists(path):
        return last
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("idx") is not None:
                last[r["idx"]] = r
    return last


def load_cache(path):
    cache = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                cache[f"{r['method']}:{r['role']}:{r['idx']}"] = r.get("scores", {})
    return cache


def safe(fn, *args):
    try:
        v = fn(*args)
        return float(v) if v is not None else None
    except Exception:
        return None


def build_analyzers(device, use_ppl, use_bertscore, oracle_path, bertscore_model):
    A = {
        "log_diversity": LogDiversityAnalyzer(),
        "bleu": BLEUCalculator(),
        "rouge1": ROUGE1Calculator(),
        "rouge2": ROUGE2Calculator(),
        "rougeL": ROUGELCalculator(),
    }
    heavy = []
    if use_ppl:
        print(f"loading PPL oracle {oracle_path} ...")
        m = AutoModelForCausalLM.from_pretrained(oracle_path).to(device)
        t = AutoTokenizer.from_pretrained(oracle_path)
        A["ppl"] = PPLCalculator(model=m, tokenizer=t, device=device)
        heavy.append(m)
    if use_bertscore:
        print(f"loading BERTScore model {bertscore_model} ...")
        A["bertscore"] = BERTScoreCalculator(model_path=bertscore_model)
    return A, heavy


def score_text(A, text, reference):
    """Run every applicable analyzer on one text. Returns {metric: value_or_None}."""
    out = {}
    if not text or not text.strip():
        return out
    # direct
    if "ppl" in A and len(text.split()) >= 2:
        out["ppl"] = safe(A["ppl"].analyze, text)
    if "log_diversity" in A:
        out["log_diversity"] = safe(A["log_diversity"].analyze, text)
    # referenced (need a non-empty reference)
    if reference and reference.strip():
        for m in ("bleu", "rouge1", "rouge2", "rougeL", "bertscore"):
            if m in A:
                out[m] = safe(A[m].analyze, text, reference)
    return out


def mean_ignore_none(vals):
    vs = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return (float(np.mean(vs)), len(vs)) if vs else (float("nan"), 0)


def relative_ppl(cache, method):
    wm, un = {}, {}
    for k, sc in cache.items():
        mth, role, idx = k.split(":")
        if mth != method:
            continue
        p = sc.get("ppl")
        if p is None:
            continue
        (wm if role == "watermarked" else un)[int(idx)] = p
    shared = set(wm) & set(un)
    ratios = [wm[i] / un[i] for i in shared if un[i] > 0]
    return float(np.mean(ratios)) if ratios else float("nan")


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