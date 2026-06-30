"""RFNN Exp-3 over-intrinsic sweep.

This probes the regime where the synthetic source dimension can exceed the
observed latent dimension.  Data are regenerated for each d_intrinsic:

    source GMM in d_intrinsic dims -> random map into fixed d_latent=20

For d_intrinsic <= d_latent this reduces to the usual anisotropic GMM in the
first d_intrinsic coordinates plus null noise.  For d_intrinsic > d_latent,
the source GMM is compressed into the fixed latent space.

Outputs are intentionally diagnostic, not written to ``clean figures``.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


OUT_ROOT = Path("diagnostic figures") / "exp3_overintrinsic_rfnn"


@dataclass(frozen=True)
class Config:
    name: str
    n: int = 500
    k: int = 10
    d_latent: int = 20
    sigma_signal: float = 1.0
    sigma_noise: float = 0.5
    center_scale: float = 3.0
    t: float = 0.01
    mc_samples: int = 500
    rank_null: int = 300
    p_mode: str = "n_plus_d_plus_r"
    p_ratio: int = 64
    p_fixed: int = 1800
    seed: int = 42


def random_rotation(rng, d):
    q, r = np.linalg.qr(rng.standard_normal((d, d)))
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1
    return q * signs


def feature_width(cfg: Config):
    if cfg.p_mode == "n_plus_d_plus_r":
        return cfg.d_latent + cfg.n + cfg.rank_null
    if cfg.p_mode == "p64":
        return cfg.p_ratio * cfg.d_latent
    if cfg.p_mode == "fixed":
        return cfg.p_fixed
    raise ValueError(cfg.p_mode)


def sample_overintrinsic_gmm(cfg: Config, d_intrinsic: int, seed: int):
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal((cfg.k, d_intrinsic))
    raw = raw / np.linalg.norm(raw, axis=1, keepdims=True).clip(1e-12)
    centers = raw * cfg.center_scale
    labels = rng.integers(0, cfg.k, size=cfg.n)
    source = centers[labels] + rng.standard_normal((cfg.n, d_intrinsic)) * cfg.sigma_signal

    if d_intrinsic <= cfg.d_latent:
        x = np.zeros((cfg.n, cfg.d_latent), dtype=np.float64)
        x[:, :d_intrinsic] = source
        if cfg.d_latent > d_intrinsic:
            x[:, d_intrinsic:] = (
                rng.standard_normal((cfg.n, cfg.d_latent - d_intrinsic))
                * cfg.sigma_noise
            )
    else:
        # Energy-preserving random compression from source dims into latent dims.
        # Entries ~ N(0, 1/d_latent) preserve expected squared norm.
        A = rng.standard_normal((d_intrinsic, cfg.d_latent)) / math.sqrt(cfg.d_latent)
        x = source @ A

    q = random_rotation(rng, cfg.d_latent)
    return x @ q.T


def compute_eigs_numpy(cfg: Config, d_intrinsic: int, seed: int):
    rng = np.random.default_rng(seed)
    x = sample_overintrinsic_gmm(cfg, d_intrinsic, seed + 1000)
    p = feature_width(cfg)
    w = rng.standard_normal((p, cfg.d_latent)) / math.sqrt(cfg.d_latent)
    delta_t = 1 - math.exp(-2 * cfg.t)
    e_neg_t = math.exp(-cfg.t)
    U = np.zeros((p, p), dtype=np.float64)
    for _ in range(cfg.mc_samples):
        noise = rng.standard_normal((cfg.n, cfg.d_latent))
        x_t = e_neg_t * x + math.sqrt(delta_t) * noise
        phi = np.tanh(x_t @ w.T)
        U += phi.T @ phi / cfg.n
    U /= cfg.mc_samples
    return np.linalg.eigvalsh(U)[::-1]


def compute_eigs_torch(cfg: Config, d_intrinsic: int, seed: int):
    if torch is None or not torch.cuda.is_available():
        return compute_eigs_numpy(cfg, d_intrinsic, seed)

    rng = np.random.default_rng(seed)
    x_np = sample_overintrinsic_gmm(cfg, d_intrinsic, seed + 1000)
    p = feature_width(cfg)
    w_np = rng.standard_normal((p, cfg.d_latent)) / math.sqrt(cfg.d_latent)

    device = torch.device("cuda:0")
    dtype = torch.float64
    gen = torch.Generator(device=device)
    gen.manual_seed(seed + 2000)
    x = torch.as_tensor(x_np, dtype=dtype, device=device)
    w = torch.as_tensor(w_np, dtype=dtype, device=device)
    U = torch.zeros((p, p), dtype=dtype, device=device)
    delta_t = 1 - math.exp(-2 * cfg.t)
    e_neg_t = math.exp(-cfg.t)
    sqrt_delta_t = math.sqrt(delta_t)
    for _ in range(cfg.mc_samples):
        noise = torch.randn(
            (cfg.n, cfg.d_latent), dtype=dtype, device=device, generator=gen
        )
        x_t = e_neg_t * x + sqrt_delta_t * noise
        phi = torch.tanh(x_t @ w.T)
        U.add_(phi.T @ phi, alpha=1.0 / cfg.n)
    U.div_(cfg.mc_samples)
    eigs = torch.linalg.eigvalsh(U).flip(0)
    return eigs.detach().cpu().numpy()


def bulk_regions(eigs, d_intrinsic, d_latent, n):
    p = len(eigs)
    active_dim = min(d_intrinsic, d_latent)
    regions = [
        ("rank-null", eigs[min(d_latent + n, p):], "#5f3dc4"),
        ("sample", eigs[d_latent:min(d_latent + n, p)], "#1971c2"),
        ("noise-dim", eigs[active_dim:d_latent], "#2f9e44"),
        ("signal/source", eigs[:active_dim], "#e03131"),
    ]
    out = []
    for label, vals, color in regions:
        if len(vals) == 0:
            continue
        out.append((label, np.log10(np.maximum(vals, 1e-30)), color))
    return out


def shade_regions(ax, d_intrinsic, d_latent, n, p):
    active_dim = min(d_intrinsic, d_latent)
    regions = [
        (1, active_dim, "signal/source", "#e03131"),
        (active_dim + 1, d_latent, "noise-dim", "#2f9e44"),
        (d_latent + 1, min(d_latent + n, p), "sample", "#1971c2"),
        (min(d_latent + n, p) + 1, p, "rank-null", "#5f3dc4"),
    ]
    for lo, hi, _label, color in regions:
        if hi < lo:
            continue
        ax.axvspan(lo, hi, color=color, alpha=0.12, lw=0)
        ax.axvline(hi + 0.5, color=color, lw=0.75, ls="--", alpha=0.65)


def plot_hist(out_dir: Path, cfg: Config, dints: list[int]):
    n_cols = 5
    n_rows = math.ceil(len(dints) / n_cols)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(2.5 * n_cols, 2.35 * n_rows),
        sharex=False, sharey=True, squeeze=False,
    )
    for ax in axes.flat:
        ax.set_visible(False)

    for ax, dint in zip(axes.flat, dints):
        ax.set_visible(True)
        eigs = np.sort(np.load(out_dir / f"di{dint}_d{cfg.d_latent}" / "eigenvalues.npy"))[::-1]
        regions = bulk_regions(eigs, dint, cfg.d_latent, cfg.n)
        panel_vals = np.concatenate([vals for _label, vals, _color in regions])
        lo, hi = float(panel_vals.min()), float(panel_vals.max())
        pad = 0.04 * max(hi - lo, 1e-6)
        bins = np.linspace(lo - pad, hi + pad, 70)
        for label, vals, color in regions:
            alpha = 0.72 if label == "signal/source" else 0.33
            lw = 2.0 if label == "signal/source" else 0.25
            zorder = 5 if label == "signal/source" else 2
            ax.hist(
                vals, bins=bins, histtype="stepfilled", alpha=alpha,
                color=color, edgecolor=color, linewidth=lw, zorder=zorder,
            )
            if label == "signal/source":
                ax.hist(vals, bins=bins, histtype="step", color=color,
                        linewidth=2.4, zorder=zorder + 1)
            ax.axvline(float(np.median(vals)), color=color, lw=0.8,
                       ls="--", alpha=0.82)
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.16, lw=0.4)
        ax.set_title(rf"$d_{{int}}={dint}$", pad=3)
        ax.text(0.03, 0.93, rf"$p={len(eigs)}$", transform=ax.transAxes,
                fontsize=7, ha="left", va="top")

    for ax in axes[-1, :]:
        if ax.get_visible():
            ax.set_xlabel(r"$\log_{10}\lambda_i(U)$")
    for ax in axes[:, 0]:
        if ax.get_visible():
            ax.set_ylabel("count")

    from matplotlib.patches import Patch
    legend_items = [
        Patch(facecolor="#e03131", alpha=0.45, label="signal/source"),
        Patch(facecolor="#2f9e44", alpha=0.34, label="noise-dim"),
        Patch(facecolor="#1971c2", alpha=0.34, label="sample"),
        Patch(facecolor="#5f3dc4", alpha=0.34, label="rank-null"),
    ]
    fig.suptitle(f"Exp 3 over-intrinsic RFNN: {cfg.name}", y=1.01)
    fig.legend(handles=legend_items, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out_dir / "hist_by_index_region.png", dpi=180)
    fig.savefig(out_dir / "hist_by_index_region.pdf")
    plt.close(fig)


def plot_spectrum(out_dir: Path, cfg: Config, dints: list[int]):
    n_cols = 5
    n_rows = math.ceil(len(dints) / n_cols)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(2.5 * n_cols, 2.25 * n_rows),
        sharey=True, squeeze=False,
    )
    for ax in axes.flat:
        ax.set_visible(False)
    for ax, dint in zip(axes.flat, dints):
        ax.set_visible(True)
        eigs = np.sort(np.load(out_dir / f"di{dint}_d{cfg.d_latent}" / "eigenvalues.npy"))[::-1]
        p = len(eigs)
        x = np.arange(1, p + 1)
        floor = max(np.percentile(eigs[eigs > 0], 1) * 0.25, 1e-12)
        ax.loglog(x, np.maximum(eigs, floor), color="black", lw=0.9)
        shade_regions(ax, dint, cfg.d_latent, cfg.n, p)
        ax.set_title(rf"$d_{{int}}={dint}$", pad=3)
        ax.set_xlabel("index")
        ax.grid(True, which="both", alpha=0.18, lw=0.4)
    for ax in axes[:, 0]:
        if ax.get_visible():
            ax.set_ylabel("eigenvalue")
    fig.suptitle(f"Exp 3 over-intrinsic RFNN spectrum: {cfg.name}", y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / "spectrum.png", dpi=180)
    fig.savefig(out_dir / "spectrum.pdf")
    plt.close(fig)


def run_one(root: Path, cfg: Config, dints: list[int]):
    out_dir = root / cfg.name
    out_dir.mkdir(parents=True, exist_ok=True)
    for j, dint in enumerate(dints):
        run_dir = out_dir / f"di{dint}_d{cfg.d_latent}"
        run_dir.mkdir(exist_ok=True)
        eig_path = run_dir / "eigenvalues.npy"
        if not eig_path.exists():
            seed = cfg.seed + 97 * j
            eigs = compute_eigs_torch(cfg, dint, seed)
            np.save(eig_path, eigs)
        (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
        print(f"{cfg.name}: di{dint}_d{cfg.d_latent}", flush=True)
    plot_hist(out_dir, cfg, dints)
    plot_spectrum(out_dir, cfg, dints)
    (out_dir / "README.md").write_text(f"""# {cfg.name}

RFNN Exp-3 over-intrinsic sweep.

Fixed observed `d_latent={cfg.d_latent}`. Data are regenerated for each
`d_intrinsic`. For `d_intrinsic > d_latent`, the source GMM is sampled in
`d_intrinsic` dimensions and compressed into the fixed latent space with a
random energy-preserving map.

Width mode: `{cfg.p_mode}`.
Feature width: `{feature_width(cfg)}`.
MC samples: `{cfg.mc_samples}`.

Index regions use `active_dim = min(d_intrinsic, d_latent)`:

- signal/source: `eigs[:active_dim]`
- noise-dim: `eigs[active_dim:d_latent]`
- sample: `eigs[d_latent:d_latent+n]`
- rank-null: `eigs[d_latent+n:]`
""")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--d-intrinsics", type=int, nargs="+",
                        default=[2, 5, 8, 12, 16, 20, 25, 30, 35, 40])
    args = parser.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    configs = [
        Config(name="p_n_plus_d_plus_r", p_mode="n_plus_d_plus_r"),
        Config(name="p64", p_mode="p64"),
        Config(name="p_fixed", p_mode="fixed", p_fixed=1800),
    ]
    for cfg in configs:
        run_one(args.out_root, cfg, args.d_intrinsics)
    (args.out_root / "README.md").write_text("""# Exp 3 over-intrinsic RFNN diagnostics

This folder is not part of `clean figures`.

The experiment fixes observed `d_latent=20` and regenerates the synthetic GMM
for each source `d_intrinsic`, including over-intrinsic cases
`d_intrinsic > 20`.

Subfolders:

- `p_n_plus_d_plus_r`: `p=d_latent+n+300`
- `p64`: `p=64*d_latent`
- `p_fixed`: `p=1800`
""")
    print(f"wrote {args.out_root}")


if __name__ == "__main__":
    main()
