"""Spectral-only RFNN sweeps for finding cleaner four-bulk figures.

This script does not train. It samples anisotropic data, builds the RFNN
feature-correlation matrix U with tanh random features, eigendecomposes U,
and writes candidate full-rank sorted-spectrum figures.

Run examples:
  python clean_four_bulk_sweep.py --preset quick
  python clean_four_bulk_sweep.py --preset sigma
  python clean_four_bulk_sweep.py --preset scale
  python clean_four_bulk_sweep.py --preset data-kind
  python clean_four_bulk_sweep.py --preset extended-dlat
"""

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).parent
OUT_ROOT = ROOT / "clean_four_bulk_candidates"


@dataclass(frozen=True)
class SpectralConfig:
    name: str
    sweep: str = "exp2"              # exp2: vary d_lat, exp3: vary d_int
    data_kind: str = "gaussian"      # gaussian or gmm
    n: int = 500
    k: int = 10
    d_intrinsic: int = 5
    d_latent: int = 20
    sigma_signal: float = 1.0
    sigma_noise: float = 0.3
    center_scale: float = 1.5
    t: float = 0.01
    mc_samples: int = 500
    rank_null: int = 300             # p = d_latent + n + rank_null
    width_mode: str = "controlled"   # controlled or p_ratio
    p_ratio: int = 64
    seed: int = 42


def sweep_values(config):
    if config.sweep == "exp2":
        return [(5, dlat) for dlat in [5, 10, 20, 40]]
    if config.sweep == "exp2_extended":
        return [(5, dlat) for dlat in [5, 10, 20, 40, 100, 200, 500, 1000]]
    if config.sweep == "exp2_wide_only":
        return [(5, dlat) for dlat in [500, 1000]]
    if config.sweep == "exp3":
        return [(dint, 20) for dint in [2, 5, 12, 20]]
    raise ValueError(f"unknown sweep: {config.sweep}")


def random_rotation(rng, d):
    q, r = np.linalg.qr(rng.standard_normal((d, d)))
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1
    return q * signs


def sample_anisotropic_gaussian(rng, n, d_intrinsic, d_latent,
                                sigma_signal, sigma_noise):
    data = np.zeros((n, d_latent), dtype=np.float64)
    data[:, :d_intrinsic] = rng.standard_normal((n, d_intrinsic)) * sigma_signal
    if d_latent > d_intrinsic:
        data[:, d_intrinsic:] = (
            rng.standard_normal((n, d_latent - d_intrinsic)) * sigma_noise
        )
    return data


def sample_anisotropic_gmm(rng, n, d_intrinsic, d_latent, k,
                           sigma_signal, sigma_noise, center_scale):
    raw = rng.standard_normal((k, d_intrinsic))
    raw_norm = np.linalg.norm(raw, axis=1, keepdims=True)
    raw_norm[raw_norm == 0] = 1
    centers = raw / raw_norm * center_scale
    labels = rng.integers(0, k, size=n)
    data = np.zeros((n, d_latent), dtype=np.float64)
    data[:, :d_intrinsic] = (
        centers[labels] + rng.standard_normal((n, d_intrinsic)) * sigma_signal
    )
    if d_latent > d_intrinsic:
        data[:, d_intrinsic:] = (
            rng.standard_normal((n, d_latent - d_intrinsic)) * sigma_noise
        )
    return data


def sample_data(config, d_intrinsic, d_latent, seed):
    rng = np.random.default_rng(seed)
    if config.data_kind == "gaussian":
        data = sample_anisotropic_gaussian(
            rng, config.n, d_intrinsic, d_latent,
            config.sigma_signal, config.sigma_noise,
        )
    elif config.data_kind == "gmm":
        data = sample_anisotropic_gmm(
            rng, config.n, d_intrinsic, d_latent, config.k,
            config.sigma_signal, config.sigma_noise, config.center_scale,
        )
    else:
        raise ValueError(f"unknown data_kind: {config.data_kind}")
    q = random_rotation(rng, d_latent)
    return data @ q.T


def compute_eigs(config, d_intrinsic, d_latent, seed):
    rng = np.random.default_rng(seed)
    data = sample_data(config, d_intrinsic, d_latent, seed + 1000)
    if config.width_mode == "controlled":
        p = d_latent + config.n + config.rank_null
    elif config.width_mode == "p_ratio":
        p = config.p_ratio * d_latent
    else:
        raise ValueError(f"unknown width_mode: {config.width_mode}")
    w = rng.standard_normal((p, d_latent)) / math.sqrt(d_latent)
    delta_t = 1 - math.exp(-2 * config.t)
    e_neg_t = math.exp(-config.t)

    U = np.zeros((p, p), dtype=np.float64)
    for _ in range(config.mc_samples):
        noise = rng.standard_normal((config.n, d_latent))
        x_t = e_neg_t * data + math.sqrt(delta_t) * noise
        phi = np.tanh(x_t @ w.T)
        U += phi.T @ phi / config.n
    U /= config.mc_samples
    return np.sort(np.linalg.eigvalsh(U))[::-1]


def shade_regions(ax, d_intrinsic, d_latent, n, p):
    regions = [
        (1, d_intrinsic, "signal", "#e03131"),
        (d_intrinsic + 1, d_latent, "noise-dim", "#2f9e44"),
        (d_latent + 1, min(d_latent + n, p), "sample", "#1971c2"),
        (min(d_latent + n, p) + 1, p, "rank-null", "#5f3dc4"),
    ]
    for lo, hi, _label, color in regions:
        if hi < lo:
            continue
        ax.axvspan(lo, hi, color=color, alpha=0.12, lw=0)
        ax.axvline(hi + 0.5, color=color, lw=0.75, ls="--", alpha=0.65)


def write_figure(config, run_dir):
    cells = sweep_values(config)
    fig, axes = plt.subplots(1, len(cells), figsize=(2.35 * len(cells), 2.35),
                             sharey=True, squeeze=False)
    for i, (dint, dlat) in enumerate(cells):
        eig_path = run_dir / f"di{dint}_d{dlat}" / "eigenvalues.npy"
        eigs = np.load(eig_path)
        p = len(eigs)
        x = np.arange(1, p + 1)
        positive = eigs[eigs > 0]
        floor = max(np.percentile(positive, 1) * 0.25, 1e-12)
        ax = axes[0, i]
        ax.loglog(x, np.maximum(eigs, floor), color="black", lw=0.9)
        shade_regions(ax, dint, dlat, config.n, p)
        title = rf"$d_{{lat}}={dlat}$" if config.sweep == "exp2" else rf"$d_{{int}}={dint}$"
        ax.set_title(title, pad=3)
        ax.set_xlabel("sorted eigenvalue index")
        ax.grid(True, which="both", alpha=0.18, lw=0.4)
    axes[0, 0].set_ylabel("eigenvalue")

    from matplotlib.patches import Patch
    legend_items = [
        Patch(facecolor="#e03131", alpha=0.18, label="signal"),
        Patch(facecolor="#2f9e44", alpha=0.18, label="noise-dim buffer"),
        Patch(facecolor="#1971c2", alpha=0.18, label="sample"),
        Patch(facecolor="#5f3dc4", alpha=0.18, label="rank-null"),
    ]
    title = (
        f"{config.name}: {config.data_kind}, "
        f"sigma_noise={config.sigma_noise}, "
        f"sigma_signal={config.sigma_signal}, "
        f"center_scale={config.center_scale if config.data_kind == 'gmm' else 'n/a'}"
    )
    fig.suptitle(title, y=1.02, fontsize=9)
    fig.legend(handles=legend_items, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.075, 1, 1))
    fig.savefig(run_dir / "spectrum.png", dpi=170)
    fig.savefig(run_dir / "spectrum.pdf")
    plt.close(fig)


def region_log_values(eigs, d_intrinsic, d_latent, n):
    p = len(eigs)
    regions = [
        ("rank-null", eigs[min(d_latent + n, p):], "#5f3dc4"),
        ("sample", eigs[d_latent:min(d_latent + n, p)], "#1971c2"),
        ("noise-dim", eigs[d_intrinsic:d_latent], "#2f9e44"),
        ("signal", eigs[:d_intrinsic], "#e03131"),
    ]
    out = []
    for label, vals, color in regions:
        if len(vals) == 0:
            continue
        vals = np.log10(np.maximum(vals, 1e-30))
        out.append((label, vals, color))
    return out


def write_hist_figure(config, run_dir):
    cells = sweep_values(config)
    n_cols = 4
    n_rows = math.ceil(len(cells) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.25 * n_cols, 2.15 * n_rows),
                             sharex=True, sharey=True, squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)

    cached = []
    for dint, dlat in cells:
        eig_path = run_dir / f"di{dint}_d{dlat}" / "eigenvalues.npy"
        eigs = np.sort(np.load(eig_path))[::-1]
        regions = region_log_values(eigs, dint, dlat, config.n)
        cached.append((dint, dlat, regions))

    for ax, (dint, dlat, regions) in zip(axes.flat, cached):
        ax.set_visible(True)
        panel_vals = np.concatenate([vals for _label, vals, _color in regions])
        lo, hi = float(panel_vals.min()), float(panel_vals.max())
        pad = 0.04 * max(hi - lo, 1e-6)
        bins = np.linspace(lo - pad, hi + pad, 70)
        for label, vals, color in regions:
            alpha = 0.72 if label == "signal" else 0.33
            linewidth = 2.0 if label == "signal" else 0.25
            zorder = 5 if label == "signal" else 2
            ax.hist(vals, bins=bins, histtype="stepfilled", alpha=alpha,
                    color=color, label=label, edgecolor=color,
                    linewidth=linewidth, zorder=zorder)
            if label == "signal":
                ax.hist(vals, bins=bins, histtype="step", color=color,
                        linewidth=2.5, zorder=zorder + 1)
            ax.axvline(float(np.median(vals)), color=color, lw=0.8,
                       ls="--", alpha=0.8)
        ax.set_yscale("log")
        ax.set_title(rf"$d_{{lat}}={dlat}$", pad=3)
        ax.grid(True, which="both", alpha=0.16, lw=0.4)
        if config.width_mode == "controlled":
            p_label = dlat + config.n + config.rank_null
        else:
            p_label = config.p_ratio * dlat
        ax.text(0.03, 0.94, rf"$p={p_label}$",
                transform=ax.transAxes, fontsize=7, ha="left", va="top")

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
    title = (
        f"{config.name}: theory-index histograms, "
        f"{config.data_kind}, sigma_noise={config.sigma_noise}, "
        f"MC={config.mc_samples}, rank_null={config.rank_null}"
    )
    fig.suptitle(title, y=1.01, fontsize=9)
    fig.legend(handles=legend_items, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(run_dir / "hist_by_index_region.png", dpi=170)
    fig.savefig(run_dir / "hist_by_index_region.pdf")
    plt.close(fig)


def candidate_configs(preset):
    base = dict(n=500, mc_samples=500, rank_null=300, sigma_signal=1.0,
                sigma_noise=0.3, center_scale=1.5)
    if preset == "quick":
        return [
            SpectralConfig(name="quick_gaussian_exp2", sweep="exp2", data_kind="gaussian", **base),
            SpectralConfig(name="quick_gmm_exp2", sweep="exp2", data_kind="gmm", **base),
            SpectralConfig(name="quick_gaussian_exp3", sweep="exp3", data_kind="gaussian", **base),
            SpectralConfig(name="quick_gmm_exp3", sweep="exp3", data_kind="gmm", **base),
        ]
    if preset == "sigma":
        return [
            SpectralConfig(name=f"sigma_{sn:g}_gaussian_exp2", sweep="exp2",
                           data_kind="gaussian", **{**base, "sigma_noise": sn})
            for sn in [0.1, 0.2, 0.3, 0.5]
        ] + [
            SpectralConfig(name=f"sigma_{sn:g}_gmm_exp2", sweep="exp2",
                           data_kind="gmm", **{**base, "sigma_noise": sn})
            for sn in [0.1, 0.2, 0.3, 0.5]
        ]
    if preset == "scale":
        return [
            SpectralConfig(name=f"sigscale_{ss:g}_gaussian_exp2", sweep="exp2",
                           data_kind="gaussian", **{**base, "sigma_signal": ss})
            for ss in [0.5, 0.75, 1.0, 1.5]
        ] + [
            SpectralConfig(name=f"centerscale_{cs:g}_gmm_exp2", sweep="exp2",
                           data_kind="gmm", **{**base, "center_scale": cs})
            for cs in [0.75, 1.0, 1.5, 2.0]
        ]
    if preset == "data-kind":
        return [
            SpectralConfig(name=f"{kind}_exp{sweep[-1]}", sweep=sweep,
                           data_kind=kind, **base)
            for kind in ["gaussian", "gmm"]
            for sweep in ["exp2", "exp3"]
        ]
    if preset == "extended-dlat":
        return [
            SpectralConfig(name="extended_dlat_gaussian", sweep="exp2_extended",
                           data_kind="gaussian", **base),
            SpectralConfig(name="extended_dlat_gmm", sweep="exp2_extended",
                           data_kind="gmm", **base),
        ]
    if preset == "width-compare":
        compare_base = dict(n=500, mc_samples=500, rank_null=300,
                            sigma_signal=1.0, sigma_noise=0.5,
                            center_scale=3.0)
        return [
            SpectralConfig(name="width_controlled_gmm_original_scale",
                           sweep="exp2_extended", data_kind="gmm",
                           width_mode="controlled", **compare_base),
            SpectralConfig(name="width_pratio64_gmm_original_scale",
                           sweep="exp2_wide_only", data_kind="gmm",
                           width_mode="p_ratio", p_ratio=64, **compare_base),
        ]
    if preset == "width-controlled":
        compare_base = dict(n=500, mc_samples=500, rank_null=300,
                            sigma_signal=1.0, sigma_noise=0.5,
                            center_scale=3.0)
        return [
            SpectralConfig(name="width_controlled_gmm_original_scale",
                           sweep="exp2_extended", data_kind="gmm",
                           width_mode="controlled", **compare_base),
        ]
    if preset == "width-pratio64":
        compare_base = dict(n=500, mc_samples=500, rank_null=300,
                            sigma_signal=1.0, sigma_noise=0.5,
                            center_scale=3.0)
        return [
            SpectralConfig(name="width_pratio64_gmm_original_scale",
                           sweep="exp2_wide_only", data_kind="gmm",
                           width_mode="p_ratio", p_ratio=64, **compare_base),
        ]
    raise ValueError(f"unknown preset: {preset}")


def run_config(config):
    run_dir = OUT_ROOT / config.name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(asdict(config), indent=2))
    for i, (dint, dlat) in enumerate(sweep_values(config)):
        cell_dir = run_dir / f"di{dint}_d{dlat}"
        cell_dir.mkdir(exist_ok=True)
        eig_path = cell_dir / "eigenvalues.npy"
        if not eig_path.exists():
            eigs = compute_eigs(config, dint, dlat, config.seed + 97 * i)
            np.save(eig_path, eigs)
        print(f"{config.name}: di{dint}_d{dlat}")
    write_figure(config, run_dir)
    write_hist_figure(config, run_dir)
    print(f"wrote {run_dir / 'spectrum.png'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=["quick", "sigma", "scale", "data-kind", "extended-dlat", "width-compare", "width-controlled", "width-pratio64"],
                        default="quick")
    args = parser.parse_args()

    OUT_ROOT.mkdir(exist_ok=True)
    for config in candidate_configs(args.preset):
        run_config(config)


if __name__ == "__main__":
    main()
