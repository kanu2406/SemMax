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


from watermark.auto_watermark import AutoWatermark
from utils.transformers_config import TransformersConfig
from evaluation.tools.text_editor import (
    WordDeletion, SynonymSubstitution, ContextAwareSynonymSubstitution,
    BackTranslationTextEditor, DipperParaphraser,
)
from evaluation.tools.success_rate_calculator import DynamicThresholdSuccessRateCalculator
from sklearn.metrics import roc_auc_score
from experiments.common.io import load_outcome, to_jsonable, append_line


THRESHOLD_LABELS = ["TPR", "TNR", "FPR", "FNR", "P", "R", "F1", "ACC"]
RULES = [("best", {"rule": "best"}),
         ("tpr@0.01fpr", {"rule": "target_fpr", "target_fpr": 0.01}),
         ("tpr@0.05fpr", {"rule": "target_fpr", "target_fpr": 0.05})]



def describe(vals):
    a = np.asarray(vals, float)
    return dict(n=len(a), mean=float(a.mean()), std=float(a.std(ddof=1) if len(a) > 1 else 0.0),
               median=float(np.median(a)))


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




def get_pos_neg(method, attack, out_dir):
    """Sanitized score arrays. Positives: finite scores + floored misses
    (deterministic detection failures). Negatives: finite only."""
    pos_done, pos_fin = load_outcome(os.path.join(out_dir, f"{method}__{attack}.jsonl"))
    neg_done, neg_fin = load_outcome(os.path.join(out_dir, f"{method}__negative.jsonl"))

    finite_all = list(pos_fin.values()) + list(neg_fin.values())
    if not finite_all:
        return [], []
    floor = min(finite_all) - 1.0

    n_pos_miss = len(pos_done - set(pos_fin))          # positives that failed detection
    pos = list(pos_fin.values()) + [floor] * n_pos_miss
    neg = list(neg_fin.values())                       # negatives: drop failures
    return pos, neg




def safe_ppl(ppl, text):
    if ppl is None or not text or len(text.split()) < 2:
        return None
    try:
        with torch.no_grad():
            return float(ppl.analyze(text))
    except Exception:
        return None







def safe_div(logdiv, text):
    if not text or not text.strip():
        return None
    try:
        return float(logdiv.analyze(text))
    except Exception:
        return None


def safe(fn, *args):
    try:
        v = fn(*args)
        return float(v) if v is not None else None
    except Exception:
        return None


def auroc(pos_finite, neg, n_fail=0):
    all_fin = list(pos_finite) + list(neg)
    if not all_fin or not neg:
        return float("nan")
    floor = min(all_fin) - 1.0
    pos = list(pos_finite) + [floor] * n_fail
    try:
        return float(roc_auc_score([1] * len(pos) + [0] * len(neg), pos + list(neg)))
    except Exception:
        return float("nan")


def tpr_at_fpr(pos_finite, n_fail, neg, target):
    all_fin = list(pos_finite) + list(neg)
    if not all_fin or not neg:
        return float("nan")
    floor = min(all_fin) - 1.0
    pos = list(pos_finite) + [floor] * n_fail
    try:
        res = DynamicThresholdSuccessRateCalculator(
            rule="target_fpr", target_fpr=target, reverse=False).calculate(
                watermarked_result=[float(v) for v in pos],
                non_watermarked_result=[float(v) for v in neg])
        return float(res.get("TPR", 0.0))
    except Exception:
        return float("nan")















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
