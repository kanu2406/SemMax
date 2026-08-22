
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


from experiments.common.io import akey, set_seed, convert, load_prompts, done_indices, write_rec, ntokens, _cleanup
from experiments.common.detect import safe_detect


MODEL_PATH    = "facebook/opt-1.3b"
VOCAB_SIZE    = 50272
DATASET_PATH  = "dataset/c4/processed_c4.json"
OUT_DIR       = "results/ablation_num_seq"
NUM_PROMPTS   = 100
TARGET_TOKENS = 200
SEED          = 29

DEFAULT_NUM_SEQ = [1, 5, 10, 20, 30, 50, 75, 100]
# DEFAULT_NUM_SEQ = [1, 5, 10]

GEN_KWARGS = dict(
    max_new_tokens=TARGET_TOKENS,
    do_sample=True,
    top_p=0.95,
    temperature=0.85,
    no_repeat_ngram_size=4,
)






def build_transformers_config(device: str):
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tconf = TransformersConfig(model=model, tokenizer=tokenizer,
                               vocab_size=VOCAB_SIZE, device=device, **GEN_KWARGS)
    return tconf, model, tokenizer


def load_semmax(tconf):
    return AutoWatermark.load("SemMax", algorithm_config="config/SemMax.json",
                              transformers_config=tconf, max_gen_len=TARGET_TOKENS)


def set_num_seq(wm, n: int):
    
    wm.generator.num_seq = n


def gen_negatives(wm, tokenizer, prompts):
    path = os.path.join(OUT_DIR, "SemMax_negatives.jsonl")
    done = done_indices(path)
    remaining = [i for i in range(len(prompts)) if i not in done]
    print(f"\n[negatives] {len(done)} done, {len(remaining)} to go -> {path}")
    if not remaining:
        return
    with open(path, "a") as fout:
        for count, i in enumerate(remaining, 1):
            item = prompts[i]
            prompt, natural = item["prompt"], item["natural_text"]
            set_seed(SEED + i)
            rec = {"idx": i, "prompt": prompt, "natural_text": natural, "error": None}
            try:
                unwm = wm.generate_unwatermarked_text(prompt)
                rec["unwatermarked_text"] = unwm
                rec["n_tokens_unwm"] = ntokens(tokenizer, unwm)
                rec["detect_unwatermarked"] = safe_detect(wm, unwm)
                rec["detect_natural"] = safe_detect(wm, natural)
            except Exception as e:
                rec["error"] = f"{type(e).__name__}: {e}"
                print(f"  [neg idx={i}] ERROR: {rec['error']}")
            write_rec(fout, rec)
            if count % 10 == 0 or count == len(remaining):
                print(f"  negatives {count}/{len(remaining)} (idx {i})")


def gen_watermarked(wm, tokenizer, prompts, n):
    set_num_seq(wm, n)
    path = os.path.join(OUT_DIR, f"SemMax_nseq{n}.jsonl")
    done = done_indices(path)
    remaining = [i for i in range(len(prompts)) if i not in done]
    print(f"\n[num_seq={n}] {len(done)} done, {len(remaining)} to go -> {path}")
    if not remaining:
        return
    with open(path, "a") as fout:
        for count, i in enumerate(remaining, 1):
            item = prompts[i]
            prompt = item["prompt"]
            set_seed(SEED + i)  # same per-sample seed as negatives -> reproducible
            rec = {"idx": i, "num_seq": n, "prompt": prompt, "error": None}
            try:
                t0 = time.time()
                wm_text = wm.generate_watermarked_text(prompt)
                rec["gen_time_wm"] = round(time.time() - t0, 3)
                rec["watermarked_text"] = wm_text
                rec["n_tokens_wm"] = ntokens(tokenizer, wm_text)
                rec["detect_watermarked"] = safe_detect(wm, wm_text)
            except Exception as e:
                rec["error"] = f"{type(e).__name__}: {e}"
                print(f"  [nseq={n} idx={i}] ERROR: {rec['error']}")
            write_rec(fout, rec)
            if count % 10 == 0 or count == len(remaining):
                dt = rec.get("gen_time_wm")
                wmd = rec.get("detect_watermarked", {})
                print(f"  nseq={n} {count}/{len(remaining)} (idx {i}) "
                      f"t={dt}s detect={wmd}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_seq", nargs="+", type=int, default=DEFAULT_NUM_SEQ,
                    help="num_seq values")
    ap.add_argument("--num_prompts", type=int, default=NUM_PROMPTS)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    prompts = load_prompts(DATASET_PATH, args.num_prompts)
    list_num_seq = sorted(set(args.num_seq))
    print(f"Loaded {len(prompts)} prompts | model={MODEL_PATH} | device={device} "
          f"| num_seq ={list_num_seq}")

    tconf, model, tokenizer = build_transformers_config(device)
    wm = load_semmax(tconf)
    try:
        gen_negatives(wm, tokenizer, prompts)     # once, shared across all num_seq
        for n in list_num_seq:
            gen_watermarked(wm, tokenizer, prompts, n)
    finally:
        _cleanup(wm, model)

    print("\nAll done. Pair SemMax_nseq{N} positives vs SemMax_negatives for metrics.")


if __name__ == "__main__":
    main()