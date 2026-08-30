import os
import json
import time
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from watermark.auto_watermark import AutoWatermark
from utils.transformers_config import TransformersConfig

from experiments.common.io import (set_seed, convert, load_prompts, done_indices,
                                    ntokens, _cleanup)
from experiments.common.detect import safe_detect

MODEL_PATH   = "facebook/opt-1.3b"
VOCAB_SIZE   = 50272
DATASET_PATH = "dataset/c4/processed_c4.json"
OUT_DIR      = "results/token_len"
NUM_PROMPTS  = 100
SEED         = 29
TEMPERATURE  = 0.85                       
LENGTHS      = [100, 150, 200, 300, 400, 500, 600]
# LENGTHS      = [50, 100]

BASE_GEN_KWARGS = dict(do_sample=True, top_p=0.95, temperature=TEMPERATURE,
                       no_repeat_ngram_size=4)   # max_new_tokens added per length

METHODS = ["SemMax", "Watermax", "KSEMSTAMP"]

# --------------------------------------------------------------------------- #

def set_token_len(wm, method, n):
    if method in ("SemMax", "Watermax"):
        wm.config.max_gen_len = n          
    elif method == "KSEMSTAMP":
        wm.config.max_new_tokens = n       
    elif method == "Watermax":
        wm.config.max_gen_len = n
        if hasattr(wm, "split_len") and hasattr(wm, "n_splits"):
            wm.split_len = max(1, n // max(1, wm.n_splits))
    else:
        raise ValueError(f"unknown method {method}")


def run(method, length, model, tokenizer, prompts, device):
    out_path = os.path.join(OUT_DIR, f"{method}_len{length}.jsonl")
    done = done_indices(out_path)
    remaining = [i for i in range(len(prompts)) if i not in done]
    print(f"\n=== {method} @ len={length} ===  {len(done)} done, {len(remaining)} to go -> {out_path}")
    if not remaining:
        return

    # length enters the generation config (how many tokens the model produces)
    gen_kwargs = dict(BASE_GEN_KWARGS, max_new_tokens=length)
    tconf = TransformersConfig(model=model, tokenizer=tokenizer, vocab_size=VOCAB_SIZE,
                               device=device, **gen_kwargs)
   
    try:
        wm = AutoWatermark.load(method, algorithm_config=f"config/{method}.json",
                                transformers_config=tconf)
    except Exception as e:
        print(f"!! could not load {method}@len={length}: {type(e).__name__}: {e}")
        return

    set_token_len(wm, method, length)

    with open(out_path, "a") as fout:
        for count, i in enumerate(remaining, 1):
            item = prompts[i]
            prompt, natural = item["prompt"], item["natural_text"]
            set_seed(SEED + i)
            rec = {"idx": i, "method": method, "length": length,
                   "prompt": prompt, "natural_text": natural, "error": None}
            try:
                t0 = time.time()
                wm_text = wm.generate_watermarked_text(prompt)
                rec["gen_time_wm"] = round(time.time() - t0, 3)
                unwm_text = wm.generate_unwatermarked_text(prompt)
                rec["watermarked_text"] = wm_text
                rec["unwatermarked_text"] = unwm_text
                rec["n_tokens_wm"] = ntokens(tokenizer, wm_text)
                rec["n_tokens_unwm"] = ntokens(tokenizer, unwm_text)
                rec["detect_watermarked"] = safe_detect(wm, wm_text)
                rec["detect_unwatermarked"] = safe_detect(wm, unwm_text)
                rec["detect_natural"] = safe_detect(wm, natural)
            except Exception as e:
                rec["error"] = f"{type(e).__name__}: {e}"
                print(f"  [{method}@len{length} idx={i}] ERROR: {rec['error']}")
            fout.write(json.dumps(convert(rec), ensure_ascii=False) + "\n")
            fout.flush(); os.fsync(fout.fileno())
            if count % 10 == 0 or count == len(remaining):
                print(f"  {method}@len{length} {count}/{len(remaining)} (idx {i})")

    _cleanup(wm)   # free embedder/detector before the next (method, length)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=METHODS)
    ap.add_argument("--lengths", nargs="+", type=int, default=LENGTHS)
    ap.add_argument("--num_prompts", type=int, default=NUM_PROMPTS)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    prompts = load_prompts(DATASET_PATH, args.num_prompts)

    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"{len(prompts)} prompts | model={MODEL_PATH} | lengths={args.lengths}")

    for method in args.methods:
        for length in args.lengths:
            run(method, length, model, tokenizer, prompts, device)

    _cleanup(model)
    print("\nAll done.")


if __name__ == "__main__":
    main()