

import os
import gc
import json
import time
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from watermark.auto_watermark import AutoWatermark
from utils.transformers_config import TransformersConfig
from experiments.common.io import convert, _cleanup, done_indices, set_seed, load_prompts, ntokens
from experiments.common.detect import safe_detect

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

MODEL_PATH   = "facebook/opt-1.3b"        # ALL methods use this one model
VOCAB_SIZE   = 50272                       # OPT-1.3b
DATASET_PATH = "dataset/c4/processed_c4.json"
OUT_DIR      = "results/generations"
NUM_PROMPTS  = 100
TARGET_TOKENS = 200                        # common generation length target
SEED = 29

GEN_KWARGS = dict(
    max_new_tokens=TARGET_TOKENS,
    do_sample=True,
    top_p=0.95,
    temperature=0.85,
    no_repeat_ngram_size=4,
)

METHODS = {
    "SemMax":      {"load_kwargs": {"max_gen_len": TARGET_TOKENS}},
    "Watermax": {"load_kwargs": {"max_gen_len": TARGET_TOKENS}},
    "KSEMSTAMP":   {"load_kwargs": {"max_new_tokens" : TARGET_TOKENS}}
}




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



# --------------------------------------------------------------------------- #
# Per-method run
# --------------------------------------------------------------------------- #

def run_method(method: str, prompts, device: str, output_path = OUT_DIR):
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





# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=list(METHODS.keys()),
                    help="subset of methods to run")
    ap.add_argument("--num_prompts", type=int, default=NUM_PROMPTS)
    ap.add_argument("--dataset_path",type=str,default=DATASET_PATH,help="path to the processed dataset JSONL file")
    ap.add_argument("--output_path",type=str,default=OUT_DIR,help="path to the output folder")
    args = ap.parse_args()

    for m in args.methods:
        if m not in METHODS:
            raise ValueError(f"unknown method {m}; known: {list(METHODS)}")

    os.makedirs(OUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    prompts = load_prompts(args.dataset_path, args.num_prompts)
    print(f"Loaded {len(prompts)} prompts | model={MODEL_PATH} | device={device} "
          f"| target={TARGET_TOKENS} toks")

    for method in args.methods:
        run_method(method, prompts, device, args.output_path)

    print("\nAll done.")


if __name__ == "__main__":
    main()