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
OUT_DIR       = "results/temp_var"
NUM_PROMPTS   = 100
TARGET_TOKENS = 200
SEED          = 29
TEMPS         = [0.8, 0.9, 1.0, 1.1, 1.2]    
# TEMPS         = [0.8, 1.1, 1.2]     

BASE_GEN_KWARGS = dict(max_new_tokens=TARGET_TOKENS, do_sample=True,
                       top_p=0.95, no_repeat_ngram_size=4)   # temperature added per T

METHODS = {
    "SemMax":    {"load_kwargs": {"max_gen_len": TARGET_TOKENS}},
    "Watermax":  {"load_kwargs": {"max_gen_len": TARGET_TOKENS}},
    "KSEMSTAMP": {"load_kwargs": {"max_new_tokens": TARGET_TOKENS}},
}



# --------------------------------------------------------------------------- #

def run(method, temp, model, tokenizer, prompts, device):
    out_path = os.path.join(OUT_DIR, f"{method}_temp{temp}.jsonl")
    done = done_indices(out_path)
    remaining = [i for i in range(len(prompts)) if i not in done]
    print(f"\n=== {method} @ T={temp} ===  {len(done)} done, {len(remaining)} to go -> {out_path}")
    if not remaining:
        return

    gen_kwargs = dict(BASE_GEN_KWARGS, temperature=temp)
    tconf = TransformersConfig(model=model, tokenizer=tokenizer, vocab_size=VOCAB_SIZE,
                               device=device, **gen_kwargs)
    load_kwargs = dict(METHODS[method]["load_kwargs"], temperature=temp)
    try:
        wm = AutoWatermark.load(method, algorithm_config=f"config/{method}.json",
                                transformers_config=tconf, **load_kwargs)
    except Exception as e:
        print(f"!! could not load {method}@T={temp}: {type(e).__name__}: {e}")
        return

    with open(out_path, "a") as fout:
        for count, i in enumerate(remaining, 1):
            item = prompts[i]
            prompt, natural = item["prompt"], item["natural_text"]
            set_seed(SEED + i)
            rec = {"idx": i, "method": method, "temp": temp,
                   "prompt": prompt, "natural_text": natural, "error": None}
            try:
                t0 = time.time()
                wm_text = wm.generate_watermarked_text(prompt)
                rec["gen_time_wm"] = round(time.time() - t0, 3)
                unwm_text = wm.generate_unwatermarked_text(prompt)
                rec["watermarked_text"] = wm_text
                rec["unwatermarked_text"] = unwm_text
                rec["n_tokens_wm"] = ntokens(tokenizer, wm_text)
                rec["detect_watermarked"] = safe_detect(wm, wm_text)
                rec["detect_unwatermarked"] = safe_detect(wm, unwm_text)
            except Exception as e:
                rec["error"] = f"{type(e).__name__}: {e}"
                print(f"  [{method}@{temp} idx={i}] ERROR: {rec['error']}")
            fout.write(json.dumps(convert(rec), ensure_ascii=False) + "\n")
            fout.flush(); os.fsync(fout.fileno())
            if count % 10 == 0 or count == len(remaining):
                print(f"  {method}@{temp} {count}/{len(remaining)} (idx {i})")

    _cleanup(wm)   # free embedder/detector before next (method,temp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=list(METHODS.keys()))
    ap.add_argument("--temps", nargs="+", type=float, default=TEMPS)
    ap.add_argument("--num_prompts", type=int, default=NUM_PROMPTS)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    prompts = load_prompts(DATASET_PATH, args.num_prompts)

    # base model loaded once, reused across methods and temperatures
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"{len(prompts)} prompts | model={MODEL_PATH} | temps={args.temps}")

    for method in args.methods:
        for temp in args.temps:
            run(method, temp, model, tokenizer, prompts, device)

    _cleanup(model)
    print("\nAll done.")


if __name__ == "__main__":
    main()