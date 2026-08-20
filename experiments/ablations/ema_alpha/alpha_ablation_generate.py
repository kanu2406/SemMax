

import os
import gc
import json
import time
import argparse

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from watermark.auto_watermark import AutoWatermark
from utils.transformers_config import TransformersConfig
from experiments.common.io import akey, set_seed, convert, load_prompts, done_indices, write_rec, ntokens,_cleanup

MODEL_PATH    = "facebook/opt-1.3b"
VOCAB_SIZE    = 50272
DATASET_PATH  = "dataset/c4/processed_c4.json"
OUT_DIR       = "results/alpha_ablation"
NUM_PROMPTS   = 100
TARGET_TOKENS = 200
SEED          = 29
DEFAULT_ALPHAS = [0.1, 0.2, 0.4, 0.6, 0.7, 0.8, 0.95]

GEN_KWARGS = dict(max_new_tokens=TARGET_TOKENS, do_sample=True, top_p=0.95,
                  temperature=0.85, no_repeat_ngram_size=4)




# def convert(obj):
#     if isinstance(obj, dict):
#         return {k: convert(v) for k, v in obj.items()}
#     if isinstance(obj, list):
#         return [convert(v) for v in obj]
#     if isinstance(obj, (np.bool_,)):
#         return bool(obj)
#     if isinstance(obj, np.integer):
#         return int(obj)
#     if isinstance(obj, np.floating):
#         return float(obj)
#     if isinstance(obj, np.ndarray):
#         return obj.tolist()
#     return obj







def gen_negatives(wm, prompts):
    path = os.path.join(OUT_DIR, "SemMax_negatives.jsonl")
    done = done_indices(path)
    remaining = [i for i in range(len(prompts)) if i not in done]
    print(f"\n[negatives] {len(done)} done, {len(remaining)} to go")
    if not remaining:
        return
    with open(path, "a") as fout:
        for c, i in enumerate(remaining, 1):
            it = prompts[i]
            set_seed(SEED + i)
            rec = {"idx": i, "prompt": it["prompt"], "natural_text": it["natural_text"], "error": None}
            try:
                rec["unwatermarked_text"] = wm.generate_unwatermarked_text(it["prompt"])
            except Exception as e:
                rec["error"] = f"{type(e).__name__}: {e}"
            write_rec(fout, rec)
            if c % 10 == 0 or c == len(remaining):
                print(f"  negatives {c}/{len(remaining)}")


def gen_alpha(wm, tokenizer, prompts, a):
    wm.generator.alpha = a          # generation uses the generator's EMA decay
    path = os.path.join(OUT_DIR, f"SemMax_a{akey(a)}.jsonl")
    done = done_indices(path)
    remaining = [i for i in range(len(prompts)) if i not in done]
    print(f"\n[alpha={akey(a)}] {len(done)} done, {len(remaining)} to go")
    if not remaining:
        return
    with open(path, "a") as fout:
        for c, i in enumerate(remaining, 1):
            it = prompts[i]
            set_seed(SEED + i)
            rec = {"idx": i, "alpha": a, "prompt": it["prompt"], "error": None}
            try:
                t0 = time.time()
                txt = wm.generate_watermarked_text(it["prompt"])
                rec["gen_time_wm"] = round(time.time() - t0, 3)
                rec["watermarked_text"] = txt
                rec["n_tokens_wm"] = ntokens(tokenizer, txt)
            except Exception as e:
                rec["error"] = f"{type(e).__name__}: {e}"
            write_rec(fout, rec)
            if c % 10 == 0 or c == len(remaining):
                print(f"  a={akey(a)} {c}/{len(remaining)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alphas", nargs="+", type=float, default=DEFAULT_ALPHAS)  # FLOAT, not int
    ap.add_argument("--num_prompts", type=int, default=NUM_PROMPTS)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    prompts = load_prompts(DATASET_PATH, args.num_prompts)

    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tconf = TransformersConfig(model=model, tokenizer=tokenizer, vocab_size=VOCAB_SIZE,
                               device=device, **GEN_KWARGS)
    wm = AutoWatermark.load("SemMax", algorithm_config="config/SemMax.json",
                            transformers_config=tconf, max_gen_len=TARGET_TOKENS)

    alphas = sorted(set(round(a, 2) for a in args.alphas))
    print(f"{len(prompts)} prompts | alphas={alphas}")
    try:
        gen_negatives(wm, prompts)
        for a in alphas:
            gen_alpha(wm, tokenizer, prompts, a)
    finally:
        del wm, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print("\nAll done.")


if __name__ == "__main__":
    main()