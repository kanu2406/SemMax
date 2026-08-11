
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

MODEL_PATH = "facebook/opt-1.3b"
VOCAB_SIZE = 50272
GEN_DIR    = "generations"
OUT_DIR    = "robustness"
BERT_PATH  = "bert-large-uncased"

MT_FWD = "Helsinki-NLP/opus-mt-en-de"
MT_BWD = "Helsinki-NLP/opus-mt-de-en"

PARA_MODEL  = "humarin/chatgpt_paraphraser_on_T5_base"
PARA_PREFIX = "paraphrase: "

DIPPER_TOK_PATH   = "google/t5-v1_1-xxl"
DIPPER_MODEL_PATH = "kalpeshk2011/dipper-paraphraser-xxl"

METHODS = ["SemMax", "Watermax", "KSEMSTAMP"]
LOAD_KWARGS = {"SemMax": {"max_gen_len": 200}, "Watermax": {"max_gen_len": 200}, "KSEMSTAMP": {"max_gen_len": 200}}

DEFAULT_ATTACKS = ["clean", "Word-D", "Word-S", "Word-S-Context", "Translation", "Paraphrase-Small"]
DEFAULT_ATTACKS_2 = ["clean", "Word-D", "Word-S", "Word-S-Context"]
ALL_ATTACKS     = DEFAULT_ATTACKS + ["Doc-P-Dipper"]

THRESHOLD_LABELS = ["TPR", "TNR", "FPR", "FNR", "P", "R", "F1", "ACC"]
RULES = [("best", {"rule": "best"}),
         ("tpr@0.01fpr", {"rule": "target_fpr", "target_fpr": 0.01}),
         ("tpr@0.05fpr", {"rule": "target_fpr", "target_fpr": 0.05})]


def to_jsonable(x):
    if isinstance(x, dict):
        return {k: to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, np.generic):
        x = x.item()
    elif isinstance(x, torch.Tensor):
        x = x.item() if x.numel() == 1 else x.tolist()
    if isinstance(x, float):
        return x if math.isfinite(x) else None   # NaN/inf -> null (valid JSON)
    if isinstance(x, list):
        return [to_jsonable(v) for v in x]
    return x


def append_line(path, obj):
    with open(path, "a") as f:
        f.write(json.dumps(to_jsonable(obj), ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def load_outcome(path):
    """Return (done_idx, finite_scores)"""
    last = {}
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
                if r.get("idx") is not None:
                    last[r["idx"]] = r

    done, finite = set(), {}
    for i, r in last.items():
        val = None
        if r.get("error") is None and r.get("score") is not None:
            try:
                v = float(r["score"])
                if math.isfinite(v):
                    val = v
            except (TypeError, ValueError):
                pass
        if val is not None:
            finite[i] = val
            done.add(i)
        else:
            err = str(r.get("error") or "")
            if not err.startswith("attack "):   # deterministic failure -> done (a miss)
                done.add(i)
    return done, finite


def load_generations(method):
    path = os.path.join(GEN_DIR, f"{method}.jsonl")
    recs = []
    if not os.path.exists(path):
        print(f"!! missing {path} — run generate.py for {method} first")
        return recs
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("error") is None and r.get("watermarked_text"):
                recs.append(r)
    by_idx = {r["idx"]: r for r in recs}
    return [by_idx[i] for i in sorted(by_idx)]


# --------------------------------------------------------------------------- #
# Text ops
# --------------------------------------------------------------------------- #

def strip_prompt(text, prompt):
    if prompt and text.startswith(prompt):
        return text[len(prompt):].lstrip()
    return text


def detect(wm, text):
    """Return (detect_dict_or_None, finite_score_or_None, error_or_None).
    A non-finite detector score is reported as a deterministic error (a miss)."""
    if not text or not text.strip():
        return None, None, "empty text"
    try:
        d = wm.detect_watermark(text, return_dict=True)
    except Exception as e:
        return None, None, f"detect {type(e).__name__}: {e}"
    s = d.get("score") if isinstance(d, dict) else None
    try:
        s = float(s)
    except (TypeError, ValueError):
        return d, None, "no score in result"
    if not math.isfinite(s):
        return d, None, "non-finite detector score"
    return d, s, None


# --------------------------------------------------------------------------- #
# Attacks
# --------------------------------------------------------------------------- #

class MarianTranslator:
    def __init__(self, model_name, device):
        from transformers import MarianMTModel, MarianTokenizer
        self.tok = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name).to(device)
        self.device = device

    def translate(self, text):
        if not text or not text.strip():
            return text
        batch = self.tok([text], return_tensors="pt", truncation=True,
                         max_length=512, padding=True).to(self.device)
        with torch.no_grad():
            gen = self.model.generate(**batch, max_length=512, num_beams=4)
        return self.tok.batch_decode(gen, skip_special_tokens=True)[0]


class SmallParaphraser:
    """Local sentence-wise paraphrase attack (small T5, no API)."""
    def __init__(self, model_name, device, max_new_tokens=64, num_beams=4):
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams

    def _para(self, sentence):
        ids = self.tok(PARA_PREFIX + sentence, return_tensors="pt",
                       truncation=True, max_length=256).to(self.device)
        with torch.no_grad():
            out = self.model.generate(**ids, max_new_tokens=self.max_new_tokens,
                                      num_beams=self.num_beams, do_sample=False)
        return self.tok.decode(out[0], skip_special_tokens=True).strip()

    def edit(self, text, reference=None):
        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        if not sents:
            return text
        return " ".join(self._para(s) for s in sents)


def build_attacks(names, device):
    editors = {}
    for name in names:
        if name == "clean":
            editors[name] = None
        elif name == "Word-D":
            editors[name] = WordDeletion(ratio=0.3)
        elif name == "Word-S":
            editors[name] = SynonymSubstitution(ratio=0.5)
        elif name == "Word-S-Context":
            editors[name] = ContextAwareSynonymSubstitution(
                ratio=0.5,
                tokenizer=BertTokenizer.from_pretrained(BERT_PATH),
                model=BertForMaskedLM.from_pretrained(BERT_PATH).to(device))
        elif name == "Translation":
            fwd = MarianTranslator(MT_FWD, device)
            bwd = MarianTranslator(MT_BWD, device)
            editors[name] = BackTranslationTextEditor(
                translate_to_intermediary=fwd.translate,
                translate_to_source=bwd.translate)
            
        elif name == "Paraphrase-Small":
            print("  loading small paraphraser...")
            editors[name] = SmallParaphraser(PARA_MODEL, device)
        elif name == "Doc-P-Dipper":
            print("  loading Dipper XXL (this is large)...")
            editors[name] = DipperParaphraser(
                tokenizer=T5Tokenizer.from_pretrained(DIPPER_TOK_PATH),
                model=T5ForConditionalGeneration.from_pretrained(
                    DIPPER_MODEL_PATH, device_map="auto"),
                lex_diversity=60, order_diversity=0, sent_interval=1)
                # max_new_tokens=100, do_sample=True, top_p=0.75, top_k=None)
        else:
            raise ValueError(f"unknown attack {name}")
    return editors


def apply_attack(text, prompt, editor):
    text = strip_prompt(text, prompt)
    if editor is None:
        return text
    try:
        return editor.edit(text, prompt)
    except TypeError:
        return editor.edit(text)


# --------------------------------------------------------------------------- #
# Watermark loading
# --------------------------------------------------------------------------- #

def load_watermark(method, device):
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tconf = TransformersConfig(
        model=model, tokenizer=tokenizer, vocab_size=VOCAB_SIZE, device=device,
        max_new_tokens=200, do_sample=True, top_p=0.95, temperature=0.85,
        no_repeat_ngram_size=4)
    wm = AutoWatermark.load(method, algorithm_config=f"config/{method}.json",
                            transformers_config=tconf, **LOAD_KWARGS.get(method, {}))
    return wm, model


def cleanup(*objs):
    for o in objs:
        try:
            del o
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()



def compute_negatives(method, wm, recs):
    path = os.path.join(OUT_DIR, f"{method}__negative.jsonl")
    done, _ = load_outcome(path)
    todo = [r for r in recs if r["idx"] not in done and r.get("unwatermarked_text")]
    if todo:
        print(f"  negatives: {len(done)} done, {len(todo)} to score")
    for r in todo:
        text = strip_prompt(r["unwatermarked_text"], r["prompt"])
        d, s, err = detect(wm, text)
        append_line(path, {"idx": r["idx"], "text": text, "detect": d,
                           "score": s, "error": err})


def run_attack(method, attack, editor, wm, recs):
    path = os.path.join(OUT_DIR, f"{method}__{attack}.jsonl")
    done, _ = load_outcome(path)
    todo = [r for r in recs if r["idx"] not in done]
    print(f"  [{attack}] {len(done)} done, {len(todo)} to attack+score")

    for k, r in enumerate(todo, 1):
        original = strip_prompt(r["watermarked_text"], r["prompt"])
        try:
            attacked = apply_attack(r["watermarked_text"], r["prompt"], editor)
            d, s, err = detect(wm, attacked)
        except Exception as e:
            # transient (e.g. translation) -> 'attack ...' prefix -> retried next run
            attacked, d, s, err = None, None, None, f"attack {type(e).__name__}: {e}"

        append_line(path, {"idx": r["idx"], "original_text": original,
                           "attacked_text": attacked, "detect": d, "score": s, "error": err})
        if k % 10 == 0 or k == len(todo):
            print(f"    {attack} {k}/{len(todo)} (idx {r['idx']}) score={s}")


def get_pos_neg(method, attack):
    """Sanitized score arrays. Positives: finite scores + floored misses
    (deterministic detection failures). Negatives: finite only."""
    pos_done, pos_fin = load_outcome(os.path.join(OUT_DIR, f"{method}__{attack}.jsonl"))
    neg_done, neg_fin = load_outcome(os.path.join(OUT_DIR, f"{method}__negative.jsonl"))

    finite_all = list(pos_fin.values()) + list(neg_fin.values())
    if not finite_all:
        return [], []
    floor = min(finite_all) - 1.0

    n_pos_miss = len(pos_done - set(pos_fin))          # positives that failed detection
    pos = list(pos_fin.values()) + [floor] * n_pos_miss
    neg = list(neg_fin.values())                       # negatives: drop failures
    return pos, neg


def compute_all_metrics(pos, neg):
    out = {"n_pos": len(pos), "n_neg": len(neg)}
    if not pos or not neg:
        out["error"] = "empty score set"
        return out
    y_true = [1] * len(pos) + [0] * len(neg)
    y_score = list(pos) + list(neg)
    try:
        out["AUROC"] = float(roc_auc_score(y_true, y_score))
    except Exception as e:
        out["AUROC"] = None
        print(f"    AUROC failed: {e}")
    for name, kw in RULES:
        try:
            calc = DynamicThresholdSuccessRateCalculator(labels=THRESHOLD_LABELS, **kw)
            out[name] = calc.calculate([float(v) for v in pos], [float(v) for v in neg])
        except Exception as e:
            out[name] = {"error": f"{type(e).__name__}: {e}"}
    return out


def write_summary_csv(summary_path, csv_path):
    seen = {}
    with open(summary_path) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                seen[(r["method"], r["attack"])] = r
    cols = ["method", "attack", "n_pos", "n_neg", "AUROC"]
    for rule, _ in RULES:
        for lab in THRESHOLD_LABELS:
            cols.append(f"{rule}_{lab}")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for (m, a), r in sorted(seen.items()):
            row = [m, a, r.get("n_pos"), r.get("n_neg"), r.get("AUROC")]
            for rule, _ in RULES:
                met = r.get(rule, {})
                for lab in THRESHOLD_LABELS:
                    row.append(met.get(lab) if isinstance(met, dict) else None)
            w.writerow(row)
    print(f"\nwrote {csv_path}")


def plot_roc(methods, attacks, out_dir=OUT_DIR):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, roc_auc_score

    for method in methods:
        plt.figure(figsize=(5.2, 5))
        plotted = 0
        for attack in attacks:
            pos, neg = get_pos_neg(method, attack)   # same sanitization as metrics
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
    ap.add_argument("--attacks", nargs="+", default=DEFAULT_ATTACKS_2,
                    help=f"choose from {ALL_ATTACKS}")
    ap.add_argument("--plot_only", action="store_true")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    if args.plot_only:
        plot_roc(args.methods, args.attacks)
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    editors = build_attacks(args.attacks, device)

    summary_path = os.path.join(OUT_DIR, "summary.jsonl")
    for method in args.methods:
        recs = load_generations(method)
        if not recs:
            continue
        print(f"\n=== {method} ===  {len(recs)} generations")
        wm, model = load_watermark(method, device)
        try:
            compute_negatives(method, wm, recs)
            for attack in args.attacks:
                run_attack(method, attack, editors[attack], wm, recs)
                pos, neg = get_pos_neg(method, attack)
                row = {"method": method, "attack": attack}
                row.update(compute_all_metrics(pos, neg))
                append_line(summary_path, row)
                n_miss = row["n_pos"] - len([1 for _ in pos])  # 0; kept for clarity
                print(f"  >> {method}/{attack}  AUROC={row.get('AUROC')}  "
                      f"n_pos={row['n_pos']} n_neg={row['n_neg']}  "
                      f"TPR(best)={row.get('best', {}).get('TPR') if isinstance(row.get('best'), dict) else None}")
        finally:
            cleanup(wm, model)

    write_summary_csv(summary_path, os.path.join(OUT_DIR, "summary.csv"))
    plot_roc(args.methods, args.attacks)

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