"""Stitch p=64*d_latent RFNN spectra into one theory-index histogram figure."""

from pathlib import Path
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).parent
OUT = ROOT / "clean_four_bulk_candidates" / "pratio64_stitched"
OUT.mkdir(parents=True, exist_ok=True)

D_INT = 5
N = 500


def eig_path(dlat):
    if dlat in {5, 10, 20, 40}:
        return ROOT / "results_rfnn_exp2v3" / f"di5_d{dlat}_n500_s42" / "eigenvalues_pre.npy"
    if dlat in {100, 200}:
        return ROOT / "results_rfnn_exp2_wide" / f"di5_d{dlat}_n500_s42" / "eigenvalues_pre.npy"
    if dlat == 500:
        stable = ROOT / "clean_four_bulk_candidates" / "pratio64_stitched" / "eigenvalues_d500.npy"
        if stable.exists():
            return stable
        return ROOT / "clean_four_bulk_candidates" / "pratio64_width_compare_d500" / "eigenvalues_d500.npy"
    if dlat == 1000:
        full = ROOT / "clean_four_bulk_candidates" / "pratio64_width_compare_d1000" / "eigenvalues_d1000.npy"
        if full.exists():
            return full
        return ROOT / "clean_four_bulk_candidates" / "pratio64_d1000_top_regions" / "top_eigenvalues_d1000.npy"
    raise ValueError(dlat)


def region_log_values(eigs, dlat, top_only=False):
    p = len(eigs)
    if top_only:
        regions = [
            ("sample", eigs[dlat:min(dlat + N, p)], "#1971c2"),
            ("noise-dim", eigs[D_INT:dlat], "#2f9e44"),
            ("signal", eigs[:D_INT], "#e03131"),
        ]
    else:
        regions = [
            ("rank-null", eigs[min(dlat + N, p):], "#5f3dc4"),
            ("sample", eigs[dlat:min(dlat + N, p)], "#1971c2"),
            ("noise-dim", eigs[D_INT:dlat], "#2f9e44"),
            ("signal", eigs[:D_INT], "#e03131"),
        ]
    out = []
    for label, vals, color in regions:
        if len(vals) == 0:
            continue
        vals = np.log10(np.maximum(vals, 1e-30))
        out.append((label, vals, color))
    return out


def main():
    dlats = [5, 10, 20, 40, 100, 200, 500]
    if eig_path(1000).exists():
        dlats.append(1000)

    cached = []
    all_logs = []
    for dlat in dlats:
        eigs = np.sort(np.load(eig_path(dlat)))[::-1]
        top_only = dlat == 1000 and len(eigs) <= dlat + N
        regions = region_log_values(eigs, dlat, top_only=top_only)
        p = 64 * dlat if top_only else len(eigs)
        cached.append((dlat, p, regions, top_only))
        all_logs.extend(vals for _label, vals, _color in regions)

    n_cols = 4
    n_rows = math.ceil(len(cached) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.4 * n_cols, 7.2 * n_rows),
                             sharex=True, sharey=False, squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)

    for ax, (dlat, p, regions, top_only) in zip(axes.flat, cached):
        ax.set_visible(True)
        panel_vals = np.concatenate([vals for _label, vals, _color in regions])
        lo, hi = float(panel_vals.min()), float(panel_vals.max())
        pad = 0.04 * max(hi - lo, 1e-6)
        bins = np.linspace(lo - pad, hi + pad, 70)
        for label, vals, color in regions:
            alpha = 0.72 if label == "signal" else 0.34
            linewidth = 2.0 if label == "signal" else 0.25
            zorder = 5 if label == "signal" else 2
            ax.hist(vals, bins=bins, histtype="stepfilled", alpha=alpha,
                    color=color, label=label, edgecolor=color, linewidth=linewidth,
                    zorder=zorder)
            if label == "signal":
                ax.hist(vals, bins=bins, histtype="step", color=color, linewidth=2.5,
                        zorder=zorder + 1)
            ax.axvline(float(np.median(vals)), color=color, lw=0.9,
                       ls="--", alpha=0.85)
        ax.set_yscale("log")
        ax.set_title(rf"$d_{{lat}}={dlat}$", pad=3)
        ax.text(0.03, 0.94, rf"$p={p}$", transform=ax.transAxes,
                fontsize=7, ha="left", va="top")
        if top_only:
            ax.text(0.03, 0.82, "top only, no rank-null", transform=ax.transAxes,
                    fontsize=7, ha="left", va="top", color="0.25")
        ax.grid(True, which="both", alpha=0.16, lw=0.4)

    for ax in axes[-1, :]:
        if ax.get_visible():
            ax.set_xlabel(r"$\log_{10}\lambda_i(U)$")
    for ax in axes[:, 0]:
        if ax.get_visible():
            ax.set_ylabel("count")

    from matplotlib.patches import Patch
    legend_items = [
        Patch(facecolor="#e03131", alpha=0.33, label="signal"),
        Patch(facecolor="#2f9e44", alpha=0.33, label="noise-dim"),
        Patch(facecolor="#1971c2", alpha=0.33, label="sample"),
        Patch(facecolor="#5f3dc4", alpha=0.33, label="rank-null"),
    ]
    fig.suptitle(r"Stitched $p=64d_{lat}$ GMM spectra, $\sigma_\perp=0.5$", y=1.01)
    fig.legend(handles=legend_items, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(OUT / "hist_by_index_region.png", dpi=170)
    fig.savefig(OUT / "hist_by_index_region.pdf")
    print(OUT / "hist_by_index_region.png")


if __name__ == "__main__":
    main()
