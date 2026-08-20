import os
import json
import math
import argparse

import numpy as np
from scipy import stats


def score_from_detect(d):
    if not isinstance(d, dict) or d.get("error"):
        return None
    try:
        s = float(d.get("score"))
    except (TypeError, ValueError):
        return None
    return s if math.isfinite(s) else None



def safe_detect(wm, text: str):
    if not text or not text.strip():
        return {"error": "empty text"}
    try:
        return wm.detect_watermark(text, return_dict=True)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


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





def det_score(wm, text):
    if not text or not text.strip():
        return None
    try:
        d = wm.detect_watermark(text, return_dict=True)
        s = float(d.get("score"))
        return s if math.isfinite(s) else None
    except Exception:
        return None
