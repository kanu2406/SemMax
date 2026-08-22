"""
normality_check.py — is the detection statistic's NULL distribution normal (and
standard-normal)? This validates the analytic p-value / fixed-threshold FPR.

Tests the scores on UNWATERMARKED (and natural/human) text — the null — since that
is what norm.sf(z) assumes is N(0,1). Watermarked scores are shifted by design and
are NOT expected to be normal.

Per (method, null group) it writes a 2-panel figure:
  (left)  histogram + fitted-normal PDF + N(0,1) reference
  (right) Q-Q plot vs standard normal, with the fit line and the y=x (N(0,1)) line
and a stats row: n, mean, std, skew, excess kurtosis, Shapiro p (shape),
KS-vs-N(0,1) p (calibration).

Reading it (for a z-statistic like SemMax):
  * mean ~ 0 and std ~ 1  -> calibrated; analytic p-values valid.
  * std > 1               -> p-values anti-conservative (real FPR ABOVE nominal).
  * std < 1               -> conservative.
  * mean != 0             -> systematic bias (e.g. embedding anisotropy).
  * Q-Q curved / low Shapiro p -> non-normal; the Gaussian tail (hence low-FPR
    p-values) is unreliable even after recentering/scaling.
NOTE: methods whose score is not a z (e.g. Watermax = -log10 p, ~exponential null)
will correctly reject normality — that's expected, not a bug.

Sources (no models, no re-detection):
  generations/{method}.jsonl  -> detect_unwatermarked / detect_natural
  robustness/{method}__negative.jsonl -> stripped unwatermarked (with --source robustness)

Usage:
  python normality_check.py
  python normality_check.py --methods SemMax --source robustness
"""

import os
import json
import math
import argparse

import numpy as np
from scipy import stats
from experiments.common.io import load_lines
from experiments.common.detect import score_from_detect

GEN_DIR = "results/generations"
ROB_DIR = "results/robustness"
METHODS = ["SemMax", "Watermax", "KSEMSTAMP"]

# --------------------------------------------------------------------------- #




def null_groups(method, source):
    """Return {group_label: [scores]} for the null distribution(s)."""
    out = {}
    if source == "generations":
        recs = load_lines(os.path.join(GEN_DIR, f"{method}.jsonl"))
        for grp, key in (("unwatermarked", "detect_unwatermarked"),
                         ("natural", "detect_natural")):
            vals = [score_from_detect(r.get(key)) for r in recs if not r.get("error")]
            vals = [v for v in vals if v is not None]
            if vals:
                out[grp] = vals
    else:  # robustness negative file (stripped unwatermarked)
        vals = [float(r["score"]) for r in load_lines(os.path.join(ROB_DIR, f"{method}__negative.jsonl"))
                if r.get("error") is None and r.get("score") is not None
                and math.isfinite(float(r["score"]))]
        if vals:
            out["unwatermarked"] = vals
    return out


# --------------------------------------------------------------------------- #

def normality_stats(vals):
    a = np.asarray(vals, float)
    n = len(a)
    mean = float(a.mean())
    std = float(a.std(ddof=1)) if n > 1 else 0.0
    row = dict(n=n, mean=round(mean, 4), std=round(std, 4),
               skew=round(float(stats.skew(a)), 4),
               excess_kurtosis=round(float(stats.kurtosis(a)), 4))  # 0 = normal
    # Shapiro-Wilk: shape only (location/scale-free). p<0.05 => reject normality.
    try:
        row["shapiro_p"] = round(float(stats.shapiro(a).pvalue), 5) if 3 <= n <= 5000 else None
    except Exception:
        row["shapiro_p"] = None
    # KS vs standard normal N(0,1): tests calibration (mean 0, std 1). p<0.05 => reject.
    try:
        row["ks_vs_N01_p"] = round(float(stats.kstest(a, "norm").pvalue), 5)
    except Exception:
        row["ks_vs_N01_p"] = None
    return row


def plot_normality(vals, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    a = np.asarray([v for v in vals if np.isfinite(v)], float)
    mean, std = a.mean(), (a.std(ddof=1) if len(a) > 1 else 1.0)

    fig, (axh, axq) = plt.subplots(1, 2, figsize=(11, 4.4))

    # left: histogram + fitted normal + N(0,1)
    axh.hist(a, bins=min(25, max(8, len(a) // 4)), density=True, alpha=0.5,
             color="tab:blue", label=f"null (n={len(a)})")
    lo, hi = min(a.min(), -3.5), max(a.max(), 3.5)
    grid = np.linspace(lo, hi, 400)
    if std > 1e-9:
        axh.plot(grid, stats.norm.pdf(grid, mean, std), "r-", lw=1.8,
                 label=f"fit  N({mean:.2f}, {std:.2f}²)")
    axh.plot(grid, stats.norm.pdf(grid, 0, 1), "k--", lw=1.2, label="N(0, 1)")
    axh.axvline(0, color="gray", ls=":", lw=0.8)
    axh.set_xlabel("Detection score"); axh.set_ylabel("Density")
    axh.set_title("Null vs. normal"); axh.legend(fontsize=8); axh.grid(alpha=0.25)

    # right: Q-Q vs standard normal
    (osm, osr), (slope, intercept, r) = stats.probplot(a, dist="norm")
    axq.scatter(osm, osr, s=14, alpha=0.7, label="sample quantiles")
    axq.plot(osm, slope * osm + intercept, "r-", lw=1.5,
             label=f"fit (slope={slope:.2f}, int={intercept:.2f})")
    lim = [min(osm.min(), osr.min()), max(osm.max(), osr.max())]
    axq.plot(lim, lim, "k--", lw=1.0, label="y = x  (N(0,1))")
    axq.set_xlabel("Theoretical N(0,1) quantiles"); axq.set_ylabel("Ordered scores")
    axq.set_title(f"Q-Q plot (R²={r**2:.3f})"); axq.legend(fontsize=8); axq.grid(alpha=0.25)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"wrote {path}")


# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=METHODS)
    ap.add_argument("--source", choices=["results/generations", "results/robustness"], default="generations")
    ap.add_argument("--out", default="results/score_analysis")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rows = []
    for method in args.methods:
        groups = null_groups(method, args.source)
        if not groups:
            print(f"[skip] {method}: no null scores from {args.source}")
            continue
        for grp, vals in groups.items():
            if len(vals) < 3:
                print(f"[skip] {method}/{grp}: only {len(vals)} points")
                continue
            plot_normality(vals, f"{method} — {grp} null normality",
                           os.path.join(args.out, f"normality_{method}_{grp}.png"))
            st = normality_stats(vals)
            st.update(method=method, group=grp)
            rows.append(st)

    if rows:
        import csv
        cols = ["method", "group", "n", "mean", "std", "skew", "excess_kurtosis",
                "shapiro_p", "ks_vs_N01_p"]
        csv_path = os.path.join(args.out, "normality_stats.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow({c: r.get(c, "") for c in cols})
        print(f"\nwrote {csv_path}")
        print("\n=== null normality (want mean~0, std~1, p's not tiny) ===")
        print(f"{'method':12s} {'group':13s} {'mean':>7s} {'std':>7s} {'skew':>7s} "
              f"{'kurt':>7s} {'shapiro':>9s} {'KS_N01':>9s}")
        for r in rows:
            print(f"{r['method']:12s} {r['group']:13s} {r['mean']:>7} {r['std']:>7} "
                  f"{r['skew']:>7} {r['excess_kurtosis']:>7} "
                  f"{str(r['shapiro_p']):>9} {str(r['ks_vs_N01_p']):>9}")


if __name__ == "__main__":
    main()