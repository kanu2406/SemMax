import os
import re
import glob
import json
import math
import random
import argparse

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from watermark.auto_watermark import AutoWatermark
from utils.transformers_config import TransformersConfig
from evaluation.tools.text_editor import (
    WordDeletion, SynonymSubstitution, ContextAwareSynonymSubstitution,
    BackTranslationTextEditor, DipperParaphraser,
)
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM,
    BertTokenizer, BertForMaskedLM, T5Tokenizer, T5ForConditionalGeneration,
)
from utils.transformers_config import TransformersConfig
from evaluation.tools.text_quality_analyzer import PPLCalculator
from evaluation.tools.success_rate_calculator import DynamicThresholdSuccessRateCalculator
from transformers import (AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM,
                          BertTokenizer, BertForMaskedLM)
from experiments.common.io import load_lines, last_by_idx, strip_prompt, sent_split, akey, load_kv, append
from experiments.common. detect import det_score
from experiments.common.metrics import auroc, tpr_at_fpr, safe_ppl
from experiments.common.attacks import CropAttack, MarianTranslator, SmallParaphraser, build_attacks, apply_attack


DIR         = "results/alpha_ablation"
MODEL_PATH  = "facebook/opt-1.3b"
VOCAB_SIZE  = 50272
ORACLE_PATH = "facebook/opt-2.7b"
BERT_PATH   = "bert-large-uncased"
PARA_MODEL  = "humarin/chatgpt_paraphraser_on_T5_base"
PARA_PREFIX = "paraphrase: "
MT_FWD = "Helsinki-NLP/opus-mt-en-de"
MT_BWD = "Helsinki-NLP/opus-mt-de-en"


DIPPER_TOK_PATH   = "google/t5-v1_1-xxl"
DIPPER_MODEL_PATH = "kalpeshk2011/dipper-paraphraser-xxl"


DEFAULT_ATTACKS = ["clean", "Word-D", "Word-S", "Crop-0.25", "Crop-0.5", "Crop-0.75","Doc-P-Dipper","Translation"]
DEFAULT_ATTACKS = ["clean", "Word-D", "Word-S","Doc-P-Dipper","Translation"]
# --------------------------------------------------------------------------- #




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=DIR)
    ap.add_argument("--attacks", nargs="+", default=DEFAULT_ATTACKS)
    ap.add_argument("--oracle", default=ORACLE_PATH)
    ap.add_argument("--metric", default="AUROC", choices=["TPR@1%", "TPR@5%", "AUROC"])
    ap.add_argument("--no_ppl", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # discover alpha files (decimals!) -> {alpha_float: path}
    alpha_files = {}
    for p in glob.glob(os.path.join(args.dir, "SemMax_a*.jsonl")):
        m = re.search(r"_a([0-9.]+)\.jsonl$", os.path.basename(p))
        if m:
            alpha_files[akey(m.group(1))] = p
    if not alpha_files:
        print(f"!! no SemMax_a*.jsonl in {args.dir}")
        return
    alphas = sorted(alpha_files)

    neg_recs = last_by_idx(os.path.join(args.dir, "SemMax_negatives.jsonl"))
    unwm_text = {i: r.get("unwatermarked_text") for i, r in neg_recs.items()
                 if not r.get("error") and r.get("unwatermarked_text")}

    filler_pool = []
    for r in neg_recs.values():
        nt = r.get("natural_text")
        if nt:
            filler_pool.extend(sent_split(nt))
    filler_pool = [s for s in filler_pool if len(s.split()) >= 4]
    if not filler_pool:
        filler_pool = ["This is an unrelated sentence added by the editor."]

    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(device)
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tconf = TransformersConfig(model=model, tokenizer=tok, vocab_size=VOCAB_SIZE,
                               device=device, max_new_tokens=200, do_sample=True,
                               top_p=0.95, temperature=0.85, no_repeat_ngram_size=4)
    wm = AutoWatermark.load("SemMax", algorithm_config="config/SemMax.json",
                            transformers_config=tconf, max_gen_len=200)
    editors = build_attacks(args.attacks, device, filler_pool)

    ppl = om = None
    if not args.no_ppl:
        print(f"loading PPL oracle {args.oracle} ...")
        om = AutoModelForCausalLM.from_pretrained(args.oracle).to(device)
        ot = AutoTokenizer.from_pretrained(args.oracle)
        ppl = PPLCalculator(model=om, tokenizer=ot, device=device)

    neg_path = os.path.join(args.dir, "neg_scores.jsonl")
    pos_path = os.path.join(args.dir, "pos_scores.jsonl")
    ppl_path = os.path.join(args.dir, "ppl_cache.jsonl")
    neg_cache = load_kv(neg_path, lambda r: (akey(r["alpha"]), r["idx"]), lambda r: r.get("score"))
    pos_cache = load_kv(pos_path, lambda r: (akey(r["alpha"]), r["attack"], r["idx"]), lambda r: r.get("score"))
    ppl_cache = load_kv(ppl_path, lambda r: (r["source"], r["idx"]), lambda r: r.get("ppl"))

    if ppl is not None:
        for i, t in unwm_text.items():
            if ("unwm", i) not in ppl_cache:
                v = safe_ppl(ppl, t)
                append(ppl_path, {"source": "unwm", "idx": i, "ppl": v})
                ppl_cache[("unwm", i)] = v

    rows = []
    for a in alphas:
        wm.detector.alpha = a          # detection alpha MUST match generation alpha
        print(f"\n=== alpha {a} ===")

        # negatives for this alpha (re-detect shared unwatermarked text at alpha a)
        for i, t in unwm_text.items():
            if (a, i) not in neg_cache:
                s = det_score(wm, t)
                append(neg_path, {"alpha": a, "idx": i, "score": s})
                neg_cache[(a, i)] = s
        neg = [v for (a_, i), v in neg_cache.items()
               if a_ == a and v is not None and math.isfinite(v)]

        wrecs = last_by_idx(alpha_files[a])
        wm_text = {i: r.get("watermarked_text") for i, r in wrecs.items()
                   if not r.get("error") and r.get("watermarked_text")}

        # relative PPL of the clean watermarked text (alpha-dependent, attack-independent)
        rel = float("nan")
        if ppl is not None:
            src = f"a{a}"
            for i, t in wm_text.items():
                if (src, i) not in ppl_cache:
                    v = safe_ppl(ppl, t)
                    append(ppl_path, {"source": src, "idx": i, "ppl": v})
                    ppl_cache[(src, i)] = v
            ratios = [ppl_cache[(src, i)] / ppl_cache[("unwm", i)]
                      for i in wm_text
                      if ppl_cache.get((src, i)) and ppl_cache.get(("unwm", i))
                      and ppl_cache[("unwm", i)] > 0]
            rel = float(np.mean(ratios)) if ratios else float("nan")

        for attack in args.attacks:
            todo = [(i, t) for i, t in wm_text.items() if (a, attack, i) not in pos_cache]
            if todo:
                print(f"  [{attack}] scoring {len(todo)} (alpha {a})")
            for i, t in todo:
                prompt = wrecs[i].get("prompt")
                try:
                    atk = apply_attack(t, prompt, editors[attack])
                    s = det_score(wm, atk)
                except Exception:
                    s = None
                append(pos_path, {"alpha": a, "attack": attack, "idx": i, "score": s})
                pos_cache[(a, attack, i)] = s              # (a, ...) not (W, ...)

            finite = [v for (a_, a2, i), v in pos_cache.items()
                      if a_ == a and a2 == attack and v is not None and math.isfinite(v)]
            n_attempt = sum(1 for (a_, a2, i) in pos_cache if a_ == a and a2 == attack)  # a_, not a
            n_fail = n_attempt - len(finite)

            rows.append({
                "alpha": a, "attack": attack, "rel_ppl": round(rel, 4),
                "AUROC": round(auroc(finite, neg, n_fail), 4),
                "TPR@1%": round(tpr_at_fpr(finite, n_fail, neg, 0.01), 4),
                "TPR@5%": round(tpr_at_fpr(finite, n_fail, neg, 0.05), 4),
                "n_pos": len(finite) + n_fail, "n_neg": len(neg), "n_fail": n_fail,
            })
            r = rows[-1]
            print(f"    {attack:12s} {args.metric}={r[args.metric]} AUROC={r['AUROC']} relPPL={r['rel_ppl']}")

    del wm, model
    if om is not None:
        del om
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    import csv
    csv_path = os.path.join(args.dir, "robustness_quality.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {csv_path}")

    # ----------------------------- plots ----------------------------------- #
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    M = args.metric

    # (1) PRIMARY: alpha vs detectability, one line per attack
    plt.figure(figsize=(6.4, 4.6))
    for attack in args.attacks:
        pts = sorted((r["alpha"], r[M]) for r in rows
                     if r["attack"] == attack and not (isinstance(r[M], float) and math.isnan(r[M])))
        if pts:
            plt.plot([p[0] for p in pts], [p[1] for p in pts], "o-", label=attack)
    plt.xlabel("EMA decay alpha  (higher = shorter memory)")
    plt.ylabel(f"{M} (→ better)")
    plt.ylim(0, 1.02); plt.grid(alpha=0.3); plt.legend(fontsize=8)
    plt.title(f"SemMax alpha vs. {M}")
    plt.tight_layout()
    p = os.path.join(args.dir, "alpha_vs_metric.png")
    plt.savefig(p, dpi=150); plt.close(); print(f"wrote {p}")

    # (2) alpha vs relative PPL — single line
    seen, pts = set(), []
    for r in rows:
        if r["alpha"] in seen or (isinstance(r["rel_ppl"], float) and math.isnan(r["rel_ppl"])):
            continue
        seen.add(r["alpha"]); pts.append((r["alpha"], r["rel_ppl"]))
    pts.sort()
    if pts:
        plt.figure(figsize=(6, 4.4))
        plt.plot([p[0] for p in pts], [p[1] for p in pts], "o-", color="tab:purple")
        plt.axhline(1.0, color="gray", ls=":", lw=0.8)
        plt.xlabel("EMA decay alpha"); plt.ylabel("Relative perplexity (↓ better)")
        plt.grid(alpha=0.3); plt.title("SemMax alpha vs. text quality")
        plt.tight_layout()
        p = os.path.join(args.dir, "alpha_vs_ppl.png")
        plt.savefig(p, dpi=150); plt.close(); print(f"wrote {p}")

    # (3) tradeoff plane: detectability vs relative PPL, one line per attack (points = alphas)
    plt.figure(figsize=(6.4, 4.6))
    metric_choices = ["TPR@1%", "TPR@5%", "AUROC"]
    for c in metric_choices:
        pts = [(r["rel_ppl"], r[c], r["alpha"]) for r in rows
               if r["attack"] == "clean"
               and not (isinstance(r["rel_ppl"], float) and math.isnan(r["rel_ppl"]))
               and not (isinstance(r[M], float) and math.isnan(r[M]))]
        pts.sort(key=lambda x: x[2])
        if pts:
            plt.plot([p[0] for p in pts], [p[1] for p in pts], "o-", label=attack)
            for x, y, alpha in pts:
                plt.annotate(
                    f"α={alpha:g}",
                    (x, y),
                    xytext=(0, 6),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8
                )
        plt.axvline(1.0, color="gray", ls=":", lw=0.8)
        plt.xlabel("Relative perplexity (← better)"); plt.ylabel(f"{c} (→ better)")
        plt.grid(alpha=0.3); plt.legend(fontsize=8)
        plt.title("SemMax alpha: robustness vs. quality")
        plt.tight_layout()
        # p = os.path.join(args.dir, "robustness_quality.png")
        metric_name = c.replace("@", "_at_").replace("%", "pct")
        p = os.path.join(
            args.dir,
            f"robustness_quality_{c}.png"
        )
        plt.savefig(p, dpi=150); plt.close(); print(f"wrote {p}")


if __name__ == "__main__":
    main()