

import os
import gc
import json
import time
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from watermark.auto_watermark import AutoWatermark
from utils.transformers_config import TransformersConfig

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

MODEL_PATH   = "facebook/opt-1.3b"        # ALL methods use this one model
VOCAB_SIZE   = 50272                       # OPT-1.3b
DATASET_PATH = "dataset/c4/processed_c4.json"
OUT_DIR      = "generations"
NUM_PROMPTS  = 100
TARGET_TOKENS = 200                        # common generation length target
SEED = 29

# Shared decoding kwargs. Used verbatim by generate_unwatermarked_text and by
# token-level methods; the SemMax/RobustGauss wrappers also copy
# no_repeat_ngram_size / repetition_penalty from here onto generation_config so
# watermarked decoding doesn't degenerate.
GEN_KWARGS = dict(
    max_new_tokens=TARGET_TOKENS,
    do_sample=True,
    top_p=0.95,
    temperature=0.85,
    no_repeat_ngram_size=4,
)

# Per-method load-time overrides. AutoWatermark.load forwards **kwargs into the
# config (BaseConfig updates config_dict with them), so this pins each method's
# internal length to TARGET_TOKENS without editing the JSON files.
#   - SemMax / RobustGauss control length via their own max_gen_len -> override it.
#   - KSemStamp uses gen_kwargs -> nothing to override here.
METHODS = {
    "SemMax":      {"load_kwargs": {"max_gen_len": TARGET_TOKENS}},
    "Watermax": {"load_kwargs": {"max_gen_len": TARGET_TOKENS}},
    "KSEMSTAMP":   {"load_kwargs": {"max_new_tokens" : TARGET_TOKENS}}
}

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_prompts(path: str, n: int):
    items = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            items.append({"prompt": obj["prompt"], "natural_text": obj["natural_text"]})
            if len(items) >= n:
                break
    return items


def done_indices(path: str):
    """Indices already successfully written (parse-robust against a torn last line)."""
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # skip a partially-written trailing line
            if rec.get("error") is None and "idx" in rec:
                done.add(rec["idx"])
    return done


def build_transformers_config(device: str):
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:                # WaterMax uses padding=True
        tokenizer.pad_token = tokenizer.eos_token
    return TransformersConfig(
        model=model,
        tokenizer=tokenizer,
        vocab_size=VOCAB_SIZE,
        device=device,
        **GEN_KWARGS,
    ), model, tokenizer


def ntokens(tokenizer, text: str) -> int:
    if not text:
        return 0
    return len(tokenizer.encode(text, add_special_tokens=False))


def safe_detect(wm, text: str):
    if not text or not text.strip():
        return {"error": "empty text"}
    try:
        return wm.detect_watermark(text, return_dict=True)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


# --------------------------------------------------------------------------- #
# Per-method run
# --------------------------------------------------------------------------- #

def run_method(method: str, prompts, device: str):
    out_path = os.path.join(OUT_DIR, f"{method}.jsonl")
    done = done_indices(out_path)
    remaining = [i for i in range(len(prompts)) if i not in done]
    print(f"\n=== {method} ===  {len(done)} done, {len(remaining)} to go -> {out_path}")
    if not remaining:
        return

    transformers_config, model, tokenizer = build_transformers_config(device)
    try:
        wm = AutoWatermark.load(
            method,
            algorithm_config=f"config/{method}.json",
            transformers_config=transformers_config,
            **METHODS[method]["load_kwargs"],
        )
    except Exception as e:
        print(f"!! could not load {method}: {type(e).__name__}: {e}")
        _cleanup(model)
        return

    # append mode = never clobber earlier progress
    with open(out_path, "a") as fout:
        for count, i in enumerate(remaining, 1):
            item = prompts[i]
            prompt, natural = item["prompt"], item["natural_text"]
            set_seed(SEED + i)  # reproducible per-sample sampling

            rec = {"idx": i, "method": method, "prompt": prompt,
                   "natural_text": natural, "error": None}
            try:
                t0 = time.time()
                wm_text = wm.generate_watermarked_text(prompt)
                rec["gen_time_wm"] = round(time.time() - t0, 3)

                unwm_text = wm.generate_unwatermarked_text(prompt)

                rec["watermarked_text"] = wm_text
                rec["unwatermarked_text"] = unwm_text
                rec["n_tokens_wm"] = ntokens(tokenizer, wm_text)
                rec["n_tokens_unwm"] = ntokens(tokenizer, unwm_text)

                # clean-text detection baseline (attacks are applied downstream)
                rec["detect_watermarked"]   = safe_detect(wm, wm_text)
                rec["detect_unwatermarked"] = safe_detect(wm, unwm_text)
                rec["detect_natural"]       = safe_detect(wm, natural)
            except Exception as e:
                rec["error"] = f"{type(e).__name__}: {e}"
                print(f"  [{method} idx={i}] ERROR: {rec['error']}")

            
            rec = convert(rec)
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            os.fsync(fout.fileno())  # durable across a hard kill

            if count % 10 == 0 or count == len(remaining):
                wmd = rec.get("detect_watermarked", {})
                print(f"  {method} {count}/{len(remaining)} (idx {i})  "
                      f"wm_detect={wmd}")

    _cleanup(model, wm)

import numpy as np

def convert(obj):
    if isinstance(obj, dict):
        return {k: convert(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert(v) for v in obj]
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def _cleanup(*objs):
    for o in objs:
        try:
            del o
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=list(METHODS.keys()),
                    help="subset of methods to run")
    ap.add_argument("--num_prompts", type=int, default=NUM_PROMPTS)
    args = ap.parse_args()

    for m in args.methods:
        if m not in METHODS:
            raise ValueError(f"unknown method {m}; known: {list(METHODS)}")

    os.makedirs(OUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    prompts = load_prompts(DATASET_PATH, args.num_prompts)
    print(f"Loaded {len(prompts)} prompts | model={MODEL_PATH} | device={device} "
          f"| target={TARGET_TOKENS} toks")

    for method in args.methods:
        run_method(method, prompts, device)

    print("\nAll done.")


if __name__ == "__main__":
    main()