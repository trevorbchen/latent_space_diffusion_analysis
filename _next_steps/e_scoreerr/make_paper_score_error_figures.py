"""Regenerate the paper's synthetic-MLP score-error panels from the corrected CSV.

The main-text panels (`synthetic_mlp_score_error{,_per_dim}.pdf`) cover the
d_latent = 5..40 range quoted in Section 4.  The appendix `_full` panels cover
the complete fixed-width sweep, d_latent = 5..240.

Input:  corrected_score_error_paper_h256_5seed_sn05_5M.csv, written by
        compute_corrected_score_error.py (DSM-identity correction; the logged
        `score_error` column is invalid, see that script's docstring).

Usage:  python3 make_paper_score_error_figures.py --csv <csv> --outdir <figures/>
        --range full     -> d_latent 5..240, writes the *_full.pdf pair
        --range main     -> d_latent 5..40,  writes the main-text pair
"""
from __future__ import annotations
import argparse
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# House style, matching theory_notes/e_scoreerr/make_figures_and_checks.py
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e6e3", "grid.linewidth": 0.6,
                     "axes.edgecolor": "#8a8a86", "legend.frameon": False})

PANELS = {
    "synthetic_mlp_score_error":        ("corrected_error_per_dim", False,
                                         "score error",
                                         "Corrected score error over training ($t = 0.1$)"),
    "synthetic_mlp_score_error_per_dim": ("corrected_error_per_dim", True,
                                          "score error per latent dimension",
                                          "Corrected dimension-normalized score error over training ($t = 0.1$)"),
}


# 95% t-interval over 5 seeds, matching the convention used elsewhere in the
# paper: mean +/- t_{0.975,4} * SEM.
T_CRIT_4DF = 2.776445


def build(df: pd.DataFrame, per_dim: bool) -> pd.DataFrame:
    """Five-seed mean and 95% CI of the corrected error, by (d_lat, step)."""
    col = "corrected_error_per_dim"
    d = df.copy()
    # the CSV stores the per-dim value; undo the division for the total panel
    d["y"] = d[col] if per_dim else d[col] * d["d_lat"]
    g = d.groupby(["d_lat", "step"])["y"]
    out = g.agg(mean="mean", sd=lambda v: v.std(ddof=1), n="count").reset_index()
    out["ci95"] = T_CRIT_4DF * out["sd"] / out["n"] ** 0.5
    return out


def draw(agg: pd.DataFrame, ylabel: str, title: str, out: Path) -> None:
    dims = sorted(agg.d_lat.unique())
    cmap = plt.cm.viridis
    fig, ax = plt.subplots(figsize=(5.6, 3.4), dpi=150)
    for i, d in enumerate(dims):
        sub = agg[agg.d_lat == d].sort_values("step")
        c = cmap(i / max(len(dims) - 1, 1))
        x = sub.step / 1e6
        ax.fill_between(x, sub["mean"] - sub.ci95, sub["mean"] + sub.ci95,
                        color=c, alpha=0.20, lw=0)
        ax.plot(x, sub["mean"], lw=1.4, color=c, label=f"{d}")
    ax.set_xlabel("training steps (millions)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=8)
    # legend outside the axes so it never covers the curves
    ncol = 2 if len(dims) > 10 else 1
    ax.legend(title=r"$d_{\mathrm{latent}}$", fontsize=6, title_fontsize=7,
              ncol=ncol, loc="upper left", bbox_to_anchor=(1.01, 1.0),
              borderaxespad=0.0, handlelength=1.4, columnspacing=1.0)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}  ({len(dims)} curves, d_latent {dims[0]}..{dims[-1]})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--range", choices=("main", "full"), required=True)
    a = ap.parse_args()

    df = pd.read_csv(a.csv)
    if a.range == "main":
        df = df[df.d_lat <= 40]
    suffix = "_full" if a.range == "full" else ""
    outdir = Path(a.outdir)
    for stem, (_, per_dim, ylabel, title) in PANELS.items():
        draw(build(df, per_dim), ylabel, title, outdir / f"{stem}{suffix}.pdf")
