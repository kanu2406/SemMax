

import os
import re
import glob
import json
import math
import argparse

import numpy as np
from experiments.common.io import load_lines
from experiments.common.metrics import describe
from experiments.common.detect import score_from_detect
from experiments.common.metrics import auroc

GEN_DIR = "results/generations"
ROB_DIR = "results/robustness"
METHODS = ["SemMax", "Watermax", "KSEMSTAMP"]

ATTACK_ORDER = ["clean", "Word-D", "Word-S", "Word-S-Context",
                "Translation", "Paraphrase-Small", "Crop-0.25", "Crop-0.5", "Crop-0.75"]



def gen_scores(method):
    """Clean scores from generations: {watermarked, unwatermarked, natural}."""
    recs = {}
    for r in load_lines(os.path.join(GEN_DIR, f"{method}.jsonl")):
        if r.get("idx") is not None:
            recs[r["idx"]] = r
    out = {"watermarked": [], "unwatermarked": [], "natural": []}
    for r in recs.values():
        if r.get("error"):
            continue
        for grp, key in (("watermarked", "detect_watermarked"),
                         ("unwatermarked", "detect_unwatermarked"),
                         ("natural", "detect_natural")):
            s = score_from_detect(r.get(key))
            if s is not None:
                out[grp].append(s)
    return {k: v for k, v in out.items() if v}


def rob_scores(method):
    """From robustness folder: {null: [...], clean: [...], <attack>: [...]}.
    Robustness files store a top-level 'score' (already the detect score)."""
    out = {}
    neg_path = os.path.join(ROB_DIR, f"{method}__negative.jsonl")
    null = [float(r["score"]) for r in load_lines(neg_path)
            if r.get("error") is None and r.get("score") is not None
            and math.isfinite(float(r["score"]))]
    if null:
        out["__null__"] = null
    for path in glob.glob(os.path.join(ROB_DIR, f"{method}__*.jsonl")):
        base = os.path.basename(path)
        m = re.match(rf"{re.escape(method)}__(.+)\.jsonl$", base)
        if not m:
            continue
        grp = m.group(1)
        if grp == "negative":
            continue
        vals = [float(r["score"]) for r in load_lines(path)
                if r.get("error") is None and r.get("score") is not None
                and math.isfinite(float(r["score"]))]
        if vals:
            out[grp] = vals
    return out





# --------------------------------------------------------------------------- #
# plotting
# --------------------------------------------------------------------------- #

def density_plot(groups, null_key, title, path, order=None):
    """groups: {label: scores}. null_key filled gray as reference. Others = lines."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        from scipy.stats import gaussian_kde
    except Exception:
        gaussian_kde = None

    labels = list(groups.keys())
    if order:
        labels = [null_key] if null_key in groups else []
        labels += [l for l in order if l in groups and l != null_key]
        labels += [l for l in groups if l not in labels]

    allv = np.concatenate([np.asarray(groups[l], float) for l in labels]) if labels else np.array([])
    if allv.size == 0:
        print(f"[skip] {title}: no data"); return
    lo, hi = np.percentile(allv, 0.5), np.percentile(allv, 99.5)
    pad = 0.05 * (hi - lo + 1e-9)
    grid = np.linspace(lo - pad, hi + pad, 400)

    plt.figure(figsize=(6.6, 4.4))
    for lab in labels:
        v = np.asarray([x for x in groups[lab] if np.isfinite(x)], float)
        if v.size == 0:
            continue
        is_null = (lab == null_key)
        dens = None
        if gaussian_kde is not None and v.size >= 3 and v.std() > 1e-9:
            try:
                dens = gaussian_kde(v)(grid)
            except Exception:
                dens = None
        if dens is not None:
            if is_null:
                plt.fill_between(grid, dens, color="gray", alpha=0.35, label=f"{lab} (n={v.size})")
                plt.plot(grid, dens, color="gray", lw=1)
            else:
                plt.plot(grid, dens, lw=1.8, label=f"{lab} (n={v.size})")
        else:  # fallback: normalized histogram
            plt.hist(v, bins=15, density=True, histtype="stepfilled" if is_null else "step",
                     alpha=0.35 if is_null else 1.0,
                     color="gray" if is_null else None, label=f"{lab} (n={v.size})")
        plt.axvline(v.mean(), color="gray" if is_null else None, ls=":", lw=0.8, alpha=0.6)

    plt.xlabel("Detection score (→ more watermarked)")
    plt.ylabel("Density")
    plt.title(title)
    plt.legend(fontsize=8)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"wrote {path}")


# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=METHODS)
    ap.add_argument("--out", default="results/score_analysis")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    stats_rows = []
    for method in args.methods:
        g = gen_scores(method)
        r = rob_scores(method)

        # (1) base distributions from generations
        if g:
            null = g.get("unwatermarked", [])
            density_plot(g, "unwatermarked",
                         f"{method}: score distribution (clean)",
                         os.path.join(args.out, f"dist_scores_{method}.png"),
                         order=["unwatermarked", "watermarked", "natural"])
            for grp, vals in g.items():
                d = describe(vals); d.update(method=method, source="generations", group=grp,
                                             auroc_vs_null=round(auroc(vals, null), 4) if grp != "unwatermarked" else "")
                stats_rows.append(d)
        else:
            print(f"[skip] {method}: no generations/{method}.jsonl")

        # (2) distributions under attacks from robustness folder
        if r and "__null__" in r:
            groups = {"unwatermarked (null)": r["__null__"]}
            for k, v in r.items():
                if k != "__null__":
                    groups[k] = v
            density_plot(groups, "unwatermarked (null)",
                         f"{method}: score distribution under attacks",
                         os.path.join(args.out, f"dist_under_attacks_{method}.png"),
                         order=["unwatermarked (null)"] + ATTACK_ORDER)
            null = r["__null__"]
            for grp, vals in r.items():
                if grp == "__null__":
                    continue
                d = describe(vals); d.update(method=method, source="robustness", group=grp,
                                             auroc_vs_null=round(auroc(vals, null), 4))
                stats_rows.append(d)
        elif r:
            print(f"[note] {method}: robustness files present but no __negative.jsonl null set")
        else:
            print(f"[skip] {method}: no robustness/{method}__*.jsonl")

    # stats CSV
    if stats_rows:
        import csv
        cols = ["method", "source", "group", "n", "mean", "std", "median", "auroc_vs_null"]
        for d in stats_rows:
            for k in ("mean", "std", "median"):
                d[k] = round(d[k], 4)
        csv_path = os.path.join(args.out, "score_stats.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for d in stats_rows:
                w.writerow({c: d.get(c, "") for c in cols})
        print(f"\nwrote {csv_path}")
        print("\n=== separation from null (AUROC) ===")
        for d in stats_rows:
            if d.get("auroc_vs_null") not in ("", None):
                print(f"{d['method']:12s} {d['source']:11s} {d['group']:22s} "
                      f"mean={d['mean']:>8}  AUROC_vs_null={d['auroc_vs_null']}")


if __name__ == "__main__":
    main()