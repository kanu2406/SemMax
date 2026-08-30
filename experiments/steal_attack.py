import os
import json
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json, glob, os
import numpy as np
from sklearn.metrics import roc_auc_score

from watermark.auto_watermark import AutoWatermark
from utils.transformers_config import TransformersConfig
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


from experiments.common.io import set_seed, convert, load_prompts, done_indices
from experiments.common.detect import safe_detect

MODEL_PATH    = "facebook/opt-1.3b"
VOCAB_SIZE    = 50272
DATASET_PATH  = "dataset/c4/processed_c4.json"
OUT_DIR       = "results/steal"
NUM_PROMPTS   = 100
TARGET_TOKENS = 200
SEED          = 29
STEAL_NAME    = "STEAL"
METHODS       = ["SemMax", "Watermax", "KSEMSTAMP"]

GEN_KWARGS = dict(max_new_tokens=TARGET_TOKENS, min_length=230, do_sample=True,
                  top_p=0.95, temperature=0.85, no_repeat_ngram_size=4)

# --------------------------------------------------------------------------- #

def build_tconf(device):
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(device)
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tconf = TransformersConfig(model=model, tokenizer=tok, vocab_size=VOCAB_SIZE,
                               device=device, **GEN_KWARGS)
    return tconf, model


def kde_curve(x, grid):
    x = x[~np.isnan(x)]
    return gaussian_kde(x)(grid)


def run_method(method, steal_wm, tconf, prompts, device):
    """Load one watermark method, generate its genuine text + detect on
    STEAL/genuine/natural for every prompt."""
    out_path = os.path.join(OUT_DIR, f"{method}.jsonl")
    done = done_indices(out_path)
    remaining = [i for i in range(len(prompts)) if i not in done]
    print(f"\n=== {method} vs STEAL ===  {len(done)} done, {len(remaining)} to go -> {out_path}")
    if not remaining:
        return

    try:
        wm = AutoWatermark.load(method, algorithm_config=f"config/{method}.json",
                                transformers_config=tconf)
    except Exception as e:
        print(f"!! could not load {method}: {type(e).__name__}: {e}")
        return

    with open(out_path, "a") as fout:
        for count, i in enumerate(remaining, 1):
            item = prompts[i]
            prompt, natural = item["prompt"], item["natural_text"]
            set_seed(SEED + i)
            rec = {"idx": i, "method": method, "prompt": prompt,
                   "natural_text": natural, "error": None}
            try:
                # STEAL's spoofed text for this prompt (same across methods, but the
                # seed is reset per prompt so it's reproducible)
                steal_text = steal_wm.generate_watermarked_text(prompt)
                # this method's genuine watermarked text (positive control)
                genuine_text = wm.generate_watermarked_text(prompt)

                rec["steal_text"] = steal_text
                rec["genuine_text"] = genuine_text
                rec["detect_steal"]   = safe_detect(wm, steal_text)     # expect ~natural
                rec["detect_genuine"] = safe_detect(wm, genuine_text)   # positive control
                rec["detect_natural"] = safe_detect(wm, natural)        # negative control
            except Exception as e:
                rec["error"] = f"{type(e).__name__}: {e}"
                print(f"  [{method} idx={i}] ERROR: {rec['error']}")
            fout.write(json.dumps(convert(rec), ensure_ascii=False) + "\n")
            fout.flush(); os.fsync(fout.fileno())
            if count % 10 == 0 or count == len(remaining):
                print(f"  {method} {count}/{len(remaining)} (idx {i})")

def score(d):
    # ADJUST: return the z-score (or -p, so higher = more watermarked)
    if d is None: return np.nan
    return d.get("z_score", d.get("score"))   # <- confirm the real key

def load(path):
    g, n, s = [], [], []
    for line in open(path):
        r = json.loads(line)
        if r.get("error"): continue
        g.append(score(r.get("detect_genuine")))
        n.append(score(r.get("detect_natural")))
        s.append(score(r.get("detect_steal")))
    f = lambda x: np.array([v for v in x if v is not None and not np.isnan(v)])
    return f(g), f(n), f(s)

def flag_rate_at_fpr(pos, null, fpr=0.05):
    thr = np.quantile(null, 1 - fpr)      # threshold from the human/natural null
    return (pos > thr).mean()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=METHODS)
    ap.add_argument("--num_prompts", type=int, default=NUM_PROMPTS)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    prompts = load_prompts(DATASET_PATH, args.num_prompts)

    tconf, base_model = build_tconf(device)

    # load STEAL once, reuse across methods
    try:
        steal_wm = AutoWatermark.load(STEAL_NAME, algorithm_config=f"config/{STEAL_NAME}.json",
                                      transformers_config=tconf)
    except Exception as e:
        print(f"!! could not load STEAL: {type(e).__name__}: {e}")
        print("   (check config/STEAL.json exists and its stealing model is set up)")
        return

    print(f"{len(prompts)} prompts | STEAL vs {args.methods}")
    for method in args.methods:
        run_method(method, steal_wm, tconf, prompts, device)


    

    for path in glob.glob("results/steal/*.jsonl"):
        method = os.path.basename(path).replace(".jsonl", "")
        g, n, s = load(path)
        genuine_flag = flag_rate_at_fpr(g, n)
        spoof_flag   = flag_rate_at_fpr(s, n)
        labels = np.r_[np.ones_like(s), np.zeros_like(n)]
        scores = np.r_[s, n]
        auroc  = roc_auc_score(labels, scores)   # ~0.5 means spoof ≈ human
        print(f"{method:12s} genuine@5%={genuine_flag:.2f}  spoof@5%={spoof_flag:.2f}  AUROC(steal|nat)={auroc:.3f}")

    
    g, n, s = load("results/steal/SemMax.jsonl")
    lo, hi = np.nanmin(np.r_[g,n,s]), np.nanmax(np.r_[g,n,s])
    grid = np.linspace(lo, hi, 300)
    for arr, lab in [(n,"natural"), (s,"steal"), (g,"genuine")]:
        plt.plot(grid, kde_curve(arr, grid), label=lab)
    plt.fill_between(grid, kde_curve(n, grid), alpha=0.2)
    plt.xlabel("detection score (→ more watermarked)"); plt.ylabel("density")
    plt.legend(); plt.title("SemMax: genuine vs natural vs spoof")
    plt.tight_layout(); plt.savefig("steal_semmax.png", dpi=200)

    del steal_wm, base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("\nAll done. NOTE: detect_steal ~ detect_natural means the spoof carries no "
          "signal for that method (expected for non-token-level watermarks).")

    


if __name__ == "__main__":
    main()