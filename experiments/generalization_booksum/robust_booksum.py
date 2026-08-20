
import os
import re
import gc
import csv
import json
import math
import argparse

import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM,
    BertTokenizer, BertForMaskedLM, T5Tokenizer, T5ForConditionalGeneration,
)

from watermark.auto_watermark import AutoWatermark
from utils.transformers_config import TransformersConfig
from evaluation.tools.text_editor import (
    WordDeletion, SynonymSubstitution, ContextAwareSynonymSubstitution,
    BackTranslationTextEditor, DipperParaphraser,
)
from evaluation.tools.success_rate_calculator import DynamicThresholdSuccessRateCalculator
from sklearn.metrics import roc_auc_score
from translate import Translator
from experiments.common.io import to_jsonable, append_line, load_outcome, strip_prompt, _cleanup, write_summary_csv, load_generations
from experiments.common.detect import detect
from experiments.common.attacks import MarianTranslator, SmallParaphraser, build_attacks, apply_attack, run_attack
from experiments.common.metrics import compute_all_metrics, get_pos_neg


MODEL_PATH = "facebook/opt-1.3b"
VOCAB_SIZE = 50272
GEN_DIR    = "results/generations_booksum_in_domain"
OUT_DIR    = "results/robustness_booksum_in_domain"
BERT_PATH  = "bert-large-uncased"

MT_FWD = "Helsinki-NLP/opus-mt-en-de"
MT_BWD = "Helsinki-NLP/opus-mt-de-en"

PARA_MODEL  = "humarin/chatgpt_paraphraser_on_T5_base"
PARA_PREFIX = "paraphrase: "

DIPPER_TOK_PATH   = "google/t5-v1_1-xxl"
DIPPER_MODEL_PATH = "kalpeshk2011/dipper-paraphraser-xxl"

METHODS = ["SemMax", "Watermax", "KSEMSTAMP"]

LOAD_KWARGS = {
    
    "SemMax":    {"config": "config/SemMax_booksum.json", "load_kwargs": {"max_gen_len": 200}},
    "KSEMSTAMP": {"config": "config/KSEMSTAMP_booksum.json",       "load_kwargs": {"max_gen_len": 200}},

}
DEFAULT_ATTACKS = ["clean", "Word-D", "Word-S", "Word-S-Context", "Translation", "Paraphrase-Small"]

ALL_ATTACKS     = DEFAULT_ATTACKS + ["Doc-P-Dipper"]

THRESHOLD_LABELS = ["TPR", "TNR", "FPR", "FNR", "P", "R", "F1", "ACC"]
RULES = [("best", {"rule": "best"}),
         ("tpr@0.01fpr", {"rule": "target_fpr", "target_fpr": 0.01}),
         ("tpr@0.05fpr", {"rule": "target_fpr", "target_fpr": 0.05})]





def load_watermark(method, device):
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tconf = TransformersConfig(
        model=model, tokenizer=tokenizer, vocab_size=VOCAB_SIZE, device=device,
        max_new_tokens=200, do_sample=True, top_p=0.95, temperature=0.85,
        no_repeat_ngram_size=4)
    wm = AutoWatermark.load(method, algorithm_config=LOAD_KWARGS[method]["config"],
                            transformers_config=tconf, **LOAD_KWARGS[method]["load_kwargs"])
    return wm, model





def compute_negatives(method, wm, recs,output_path):
    path = os.path.join(output_path, f"{method}__negative.jsonl")
    done, _ = load_outcome(path)
    todo = [r for r in recs if r["idx"] not in done and r.get("unwatermarked_text")]
    if todo:
        print(f"  negatives: {len(done)} done, {len(todo)} to score")
    for r in todo:
        text = strip_prompt(r["unwatermarked_text"], r["prompt"])
        d, s, err = detect(wm, text)
        append_line(path, {"idx": r["idx"], "text": text, "detect": d,
                           "score": s, "error": err})


def plot_roc(methods, attacks, out_dir=OUT_DIR):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, roc_auc_score

    for method in methods:
        plt.figure(figsize=(5.2, 5))
        plotted = 0
        for attack in attacks:
            pos, neg = get_pos_neg(method, attack, out_dir)   # same sanitization as metrics
            if not pos or not neg:
                continue
            y_true = [1] * len(pos) + [0] * len(neg)
            y_score = list(pos) + list(neg)
            try:
                fpr, tpr, _ = roc_curve(y_true, y_score)
                auc = roc_auc_score(y_true, y_score)
            except Exception as e:
                print(f"  [plot] {method}/{attack} roc failed: {e}")
                continue
            plt.plot(fpr, tpr, lw=1.6, label=f"{attack} (AUC={auc:.3f})")
            plotted += 1
        if plotted == 0:
            plt.close()
            print(f"  [plot] nothing to plot for {method}")
            continue
        plt.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
        plt.xlim(0, 1); plt.ylim(0, 1.02)
        plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
        plt.title(f"ROC — {method}")
        plt.legend(fontsize=8, loc="lower right")
        plt.tight_layout()
        p = os.path.join(out_dir, f"roc_{method}.png")
        plt.savefig(p, dpi=150)
        plt.close()
        print(f"  wrote {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=METHODS)
    ap.add_argument("--attacks", nargs="+", default=ALL_ATTACKS,
                    help=f"choose from {ALL_ATTACKS}")
    ap.add_argument("--generations_path",type=str,default=GEN_DIR,help="path to the generations JSONL file")
    ap.add_argument("--output_path",type=str,default=OUT_DIR,help="path to the output folder")
    ap.add_argument("--plot_only", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    if args.plot_only:
        plot_roc(args.methods, args.attacks, args.output_path)
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    editors = build_attacks(args.attacks, device)

    summary_path = os.path.join(args.output_path, "summary.jsonl")
    for method in args.methods:
        recs = load_generations(method,args.generations_path)
        if not recs:
            continue
        print(f"\n=== {method} ===  {len(recs)} generations")
        wm, model = load_watermark(method, device)
        try:
            compute_negatives(method, wm, recs, args.output_path)
            for attack in args.attacks:
                run_attack(method, attack, editors[attack], wm, recs, args.output_path)
                pos, neg = get_pos_neg(method, attack, args.output_path)
                row = {"method": method, "attack": attack}
                row.update(compute_all_metrics(pos, neg))
                append_line(summary_path, row)
                n_miss = row["n_pos"] - len([1 for _ in pos])  # 0; kept for clarity
                print(f"  >> {method}/{attack}  AUROC={row.get('AUROC')}  "
                      f"n_pos={row['n_pos']} n_neg={row['n_neg']}  "
                      f"TPR(best)={row.get('best', {}).get('TPR') if isinstance(row.get('best'), dict) else None}")
        finally:
            _cleanup(wm, model)

    write_summary_csv(summary_path, os.path.join(args.output_path, "summary.csv"))
    plot_roc(args.methods, args.attacks,args.output_path)

    print("\n============== SUMMARY (AUROC | TPR) ==============")
    print(f"{'method':12s} {'attack':16s} {'AUROC':>7s} {'best':>7s} {'tpr@1fpr':>7s} {'tpr@5fpr':>7s}")
    seen = {}
    with open(summary_path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                seen[(r["method"], r["attack"])] = r
    for (m, a), r in sorted(seen.items()):
        def tpr(rule):
            v = r.get(rule)
            return f"{v['TPR']:.3f}" if isinstance(v, dict) and "TPR" in v else "  -  "
        au = f"{r['AUROC']:.3f}" if isinstance(r.get("AUROC"), float) else "  -  "
        print(f"{m:12s} {a:16s} {au:>7s} {tpr('best'):>7s} {tpr('tpr@0.01fpr'):>7s} {tpr('tpr@0.05fpr'):>7s}")


if __name__ == "__main__":
    main()