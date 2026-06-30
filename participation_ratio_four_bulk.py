"""Participation-ratio diagnostics for four-bulk RFNN spectra.

This is intentionally separate from ``clean figures``.  It recomputes the
same RFNN feature covariance matrices as ``clean_figure_suite.py``, keeps the
eigenvectors, and summarizes eigenvector localization by the four index
regions used in the paper candidate histograms.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

from clean_figure_suite import (
    Config,
    compute_eigs_numpy,
    effective_sigma_noise,
    exp2_cells,
    exp3_cells,
    feature_width,
    sample_data,
)


OUT_ROOT = Path("diagnostic figures") / "four_bulk_pr"
BULK_COLORS = {
    "signal": "#e03131",
    "noise-dim": "#2f9e44",
    "sample": "#1971c2",
    "rank-null": "#5f3dc4",
}


def bulk_slices(dint: int, dlat: int, n: int, p: int):
    return [
        ("signal", slice(0, min(dint, p))),
        ("noise-dim", slice(min(dint, p), min(dlat, p))),
        ("sample", slice(min(dlat, p), min(dlat + n, p))),
        ("rank-null", slice(min(dlat + n, p), p)),
    ]


def compute_eigh_numpy(cfg: Config, dint: int, dlat: int, seed: int):
    rng = np.random.default_rng(seed)
    x = sample_data(cfg, dint, dlat, seed + 1000)
    p = feature_width(cfg, dlat)
    w = rng.standard_normal((p, dlat)) / math.sqrt(dlat)
    delta_t = 1 - math.exp(-2 * cfg.t)
    e_neg_t = math.exp(-cfg.t)
    U = np.zeros((p, p), dtype=np.float64)
    for _ in range(cfg.mc_samples):
        noise = rng.standard_normal((cfg.n, dlat))
        x_t = e_neg_t * x + math.sqrt(delta_t) * noise
        phi = np.tanh(x_t @ w.T)
        U += phi.T @ phi / cfg.n
    U /= cfg.mc_samples
    vals, vecs = np.linalg.eigh(U)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:, order]


def compute_eigh_torch(cfg: Config, dint: int, dlat: int, seed: int):
    if torch is None or not torch.cuda.is_available():
        return compute_eigh_numpy(cfg, dint, dlat, seed)

    rng = np.random.default_rng(seed)
    x_np = sample_data(cfg, dint, dlat, seed + 1000)
    p = feature_width(cfg, dlat)
    w_np = rng.standard_normal((p, dlat)) / math.sqrt(dlat)

    device = torch.device("cuda:0")
    dtype = torch.float64
    gen = torch.Generator(device=device)
    gen.manual_seed(seed + 2000)

    x = torch.as_tensor(x_np, dtype=dtype, device=device)
    w = torch.as_tensor(w_np, dtype=dtype, device=device)
    delta_t = 1 - math.exp(-2 * cfg.t)
    e_neg_t = math.exp(-cfg.t)
    sqrt_delta_t = math.sqrt(delta_t)

    U = torch.zeros((p, p), dtype=dtype, device=device)
    for _ in range(cfg.mc_samples):
        noise = torch.randn(
            (cfg.n, dlat), dtype=dtype, device=device, generator=gen
        )
        x_t = e_neg_t * x + sqrt_delta_t * noise
        phi = torch.tanh(x_t @ w.T)
        U.add_(phi.T @ phi, alpha=1.0 / cfg.n)
    U.div_(cfg.mc_samples)
    vals, vecs = torch.linalg.eigh(U)
    vals = vals.flip(0).detach().cpu().numpy()
    vecs = vecs.flip(1).detach().cpu().numpy()
    return vals, vecs


def summarize_run(width: str, experiment: str, cfg: Config, dint: int, dlat: int,
                  seed: int, out_dir: Path):
    p = feature_width(cfg, dlat)
    run_dir = out_dir / width / experiment / f"{cfg.name}_di{dint}_d{dlat}"
    run_dir.mkdir(parents=True, exist_ok=True)
    eig_path = run_dir / "eigenvalues.npy"
    pr_path = run_dir / "feature_pr.npy"
    maxabs_path = run_dir / "feature_maxabs.npy"

    if eig_path.exists() and pr_path.exists() and maxabs_path.exists():
        eigs = np.load(eig_path)
        pr = np.load(pr_path)
        maxabs = np.load(maxabs_path)
    else:
        eigs, vecs = compute_eigh_torch(cfg, dint, dlat, seed)
        pr = 1.0 / np.sum(vecs ** 4, axis=0).clip(1e-300)
        maxabs = np.max(np.abs(vecs), axis=0)
        np.save(eig_path, eigs)
        np.save(pr_path, pr)
        np.save(maxabs_path, maxabs)
        (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))

    rows = []
    for bulk, sl in bulk_slices(dint, dlat, cfg.n, p):
        vals = eigs[sl]
        prs = pr[sl]
        maxabs_vals = maxabs[sl]
        if len(vals) == 0:
            continue
        rows.append({
            "width": width,
            "experiment": experiment,
            "config": cfg.name,
            "d_intrinsic": dint,
            "d_latent": dlat,
            "p": p,
            "seed": seed,
            "bulk": bulk,
            "count": len(vals),
            "eig_log10_median": float(np.median(np.log10(np.maximum(vals, 1e-30)))),
            "pr_mean": float(np.mean(prs)),
            "pr_median": float(np.median(prs)),
            "pr_q10": float(np.quantile(prs, 0.10)),
            "pr_q90": float(np.quantile(prs, 0.90)),
            "pr_over_p_median": float(np.median(prs) / p),
            "maxabs_median": float(np.median(maxabs_vals)),
        })
    return rows


def plot_exp2(summary_rows: list[dict], out_dir: Path):
    for width in sorted({r["width"] for r in summary_rows}):
        rows = [
            r for r in summary_rows
            if r["width"] == width
            and r["experiment"] == "exp2_main_gmm"
            and r["config"] == "gmm_main"
        ]
        if not rows:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
        for bulk, color in BULK_COLORS.items():
            br = sorted([r for r in rows if r["bulk"] == bulk],
                        key=lambda r: r["d_latent"])
            if not br:
                continue
            dvals = [r["d_latent"] for r in br]
            axes[0].plot(
                dvals, [r["pr_median"] for r in br],
                marker="o", color=color, label=bulk,
            )
            axes[1].plot(
                dvals, [r["pr_over_p_median"] for r in br],
                marker="o", color=color, label=bulk,
            )
        for ax in axes:
            ax.set_xscale("log")
            ax.grid(True, alpha=0.22)
            ax.set_xlabel(r"$d_{latent}$")
        axes[0].set_ylabel("median feature PR")
        axes[1].set_ylabel("median feature PR / p")
        axes[0].legend(frameon=False, fontsize=8)
        fig.suptitle(f"Feature-eigenvector participation ratio: {width}")
        fig.tight_layout()
        fig.savefig(out_dir / f"exp2_feature_pr_{width}.png", dpi=180)
        fig.savefig(out_dir / f"exp2_feature_pr_{width}.pdf")
        plt.close(fig)


def build_configs():
    main_null_energy = (20 - 5) * 0.5 ** 2
    base_gmm = Config(
        name="gmm_main",
        data_kind="gmm",
        sweep="exp2",
        row_label="GMM",
        sigma_noise=0.5,
        center_scale=3.0,
    )
    return [
        (
            "p_n_plus_d_plus_r",
            replace(base_gmm, p_mode="n_plus_d_plus_r"),
            exp2_cells(),
            exp3_cells(),
            4096,
        ),
        (
            "p64",
            replace(base_gmm, p_mode="p64"),
            exp2_cells(),
            exp3_cells(),
            4096,
        ),
        (
            "p_fixed_energy_fixed",
            replace(base_gmm, p_mode="fixed", p_fixed=1800,
                    null_energy=main_null_energy),
            exp2_cells(),
            exp3_cells(),
            4096,
        ),
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=OUT_ROOT)
    parser.add_argument("--max-p", type=int, default=4096)
    parser.add_argument("--only-exp2", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for width, cfg, exp2, exp3, default_max_p in build_configs():
        max_p = min(args.max_p, default_max_p)
        jobs = [("exp2_main_gmm", cfg, exp2)]
        if not args.only_exp2:
            jobs.append(("exp3_main_gmm", replace(cfg, sweep="exp3"), exp3))
        for experiment, ecfg, cells in jobs:
            for j, (dint, dlat) in enumerate(cells):
                p = feature_width(ecfg, dlat)
                if p > max_p:
                    print(f"skip {width}/{experiment} di{dint} d{dlat}: p={p} > {max_p}")
                    continue
                seed = ecfg.seed + 97 * j
                print(f"run {width}/{experiment} di{dint} d{dlat} p={p}", flush=True)
                all_rows.extend(
                    summarize_run(width, experiment, ecfg, dint, dlat, seed,
                                  args.out_dir)
                )

    csv_path = args.out_dir / "feature_pr_summary.csv"
    fieldnames = [
        "width", "experiment", "config", "d_intrinsic", "d_latent", "p",
        "seed", "bulk", "count", "eig_log10_median", "pr_mean",
        "pr_median", "pr_q10", "pr_q90", "pr_over_p_median",
        "maxabs_median",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    plot_exp2(all_rows, args.out_dir)
    (args.out_dir / "README.md").write_text("""# Four-bulk participation-ratio diagnostics

This folder is not part of `clean figures`.

The script recomputes the RFNN feature covariance `U`, keeps eigenvectors,
and computes feature-coordinate participation ratio

`PR(v) = 1 / sum_j v_j^4`

for each eigenvector.  Rows are grouped using the same index convention as
the clean four-bulk histograms:

- signal: `eigs[:d_intrinsic]`
- noise-dim: `eigs[d_intrinsic:d_latent]`
- sample: `eigs[d_latent:d_latent+n]`
- rank-null: `eigs[d_latent+n:]`

Important caveat: these are PRs across RFNN feature coordinates, not sample
coordinates. They diagnose whether eigenmodes are feature-localized. A
separate sample-Gram/eigenvector diagnostic would be needed for literal
sample localization.
""")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
