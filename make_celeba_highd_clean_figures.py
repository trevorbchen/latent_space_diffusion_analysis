from __future__ import annotations

import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRICS_ROOT = Path("tmp_celeba_highd_metrics/results/celeba_diverse1k_bigmlp_sgd_lr001_m08_10k_5m")
OUT_DIR = Path("clean figures/celeba_highd_multiseed_5m")
SEEDS = [42, 43, 44, 45, 46]
DIMS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200]
T95_N5 = 2.7764451051977987


def read_metric_rows() -> pd.DataFrame:
    rows = []
    for path in METRICS_ROOT.rglob("metrics.jsonl"):
        rel = path.relative_to(METRICS_ROOT)
        parts = rel.parts
        if parts[0].startswith("seed"):
            seed = int(parts[0].replace("seed", ""))
            d_part = parts[1]
        else:
            seed = 42
            d_part = parts[0]
        match = re.fullmatch(r"d(\d+)", d_part)
        if not match:
            continue
        d = int(match.group(1))
        if seed not in SEEDS or d not in DIMS:
            continue
        with path.open() as f:
            for line in f:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "step" not in row:
                    continue
                rows.append(
                    {
                        "seed": seed,
                        "d": d,
                        "step": int(row["step"]),
                        "mem": row.get("memorization_fraction_pixel"),
                        "fid": row.get("fid"),
                        "loss": row.get("train_loss_step"),
                    }
                )
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(f"No metric rows found under {METRICS_ROOT}")
    return df


def mean_ci(values: pd.Series) -> tuple[float, float, int]:
    vals = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    n = len(vals)
    if n == 0:
        return math.nan, math.nan, 0
    mean = float(vals.mean())
    if n == 1:
        return mean, 0.0, 1
    return mean, float(T95_N5 * vals.std(ddof=1) / math.sqrt(n)), n


def savefig(name: str) -> None:
    plt.tight_layout()
    for ext in ("png", "pdf"):
        plt.savefig(OUT_DIR / f"{name}.{ext}", dpi=220, bbox_inches="tight")
    plt.close()


def savefig_at(path_no_ext: Path) -> None:
    path_no_ext.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        plt.savefig(path_no_ext.with_suffix(f".{ext}"), dpi=220, bbox_inches="tight")
    plt.close()


def plot_final(df: pd.DataFrame, metric: str, ylabel: str, name: str, with_ci: bool) -> None:
    final = df[df["step"] == 5_000_000]
    stats = []
    for d in DIMS:
        mean, ci, n = mean_ci(final.loc[final["d"] == d, metric])
        stats.append({"d": d, "mean": mean, "ci": ci, "n": n})
    stats_df = pd.DataFrame(stats)

    plt.figure(figsize=(6.0, 4.0))
    if with_ci:
        plt.errorbar(
            stats_df["d"],
            stats_df["mean"],
            yerr=stats_df["ci"],
            marker="o",
            capsize=4,
            linewidth=2,
            color="#2a6fbb",
        )
    else:
        plt.plot(stats_df["d"], stats_df["mean"], marker="o", linewidth=2, color="#2a6fbb")
    plt.xlabel(r"$d_{\mathrm{latent}}$")
    plt.ylabel(ylabel)
    plt.xticks(DIMS)
    plt.grid(True, alpha=0.22)
    savefig(name)


def plot_trajectory(df: pd.DataFrame, metric: str, ylabel: str, name: str) -> None:
    metric_df = df.dropna(subset=[metric]).copy()
    metric_df[metric] = pd.to_numeric(metric_df[metric], errors="coerce")
    metric_df = metric_df.replace([np.inf, -np.inf], np.nan).dropna(subset=[metric])

    plt.figure(figsize=(7.2, 4.6))
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(DIMS)))
    for d, color in zip(DIMS, colors):
        sub = metric_df[metric_df["d"] == d]
        grouped = []
        for step, g in sub.groupby("step"):
            mean, ci, n = mean_ci(g[metric])
            if n:
                grouped.append((step, mean, ci))
        if not grouped:
            continue
        arr = np.array(grouped, dtype=float)
        order = np.argsort(arr[:, 0])
        x = arr[order, 0] / 1_000_000
        y = arr[order, 1]
        ci = arr[order, 2]
        plt.plot(x, y, label=f"d={d}", linewidth=2, color=color)
        plt.fill_between(x, y - ci, y + ci, color=color, alpha=0.13, linewidth=0)
    plt.xlabel("training steps (millions)")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.22)
    plt.legend(ncol=3, fontsize=9, frameon=False)
    savefig(name)


def plot_individual_by_d(df: pd.DataFrame, metric: str, ylabel: str, name: str) -> None:
    metric_df = df.dropna(subset=[metric]).copy()
    metric_df[metric] = pd.to_numeric(metric_df[metric], errors="coerce")
    metric_df = metric_df.replace([np.inf, -np.inf], np.nan).dropna(subset=[metric])

    ncols = 3
    nrows = int(math.ceil(len(DIMS) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(10.2, 2.75 * nrows), sharex=True, sharey=True)
    axes = axes.ravel()
    seeds = sorted(int(seed) for seed in metric_df["seed"].unique())
    seed_colors = plt.cm.tab10(np.linspace(0.0, 0.9, max(len(seeds), 1)))
    seed_color_lookup = {seed: seed_colors[i % len(seed_colors)] for i, seed in enumerate(seeds)}

    for panel_idx, (ax, d) in enumerate(zip(axes, DIMS)):
        sub = metric_df[metric_df["d"] == d].copy()
        for seed, run in sub.groupby("seed"):
            run = run.sort_values("step")
            ax.plot(
                run["step"] / 1_000_000,
                run[metric],
                color=seed_color_lookup[int(seed)],
                linewidth=0.95,
                alpha=0.42,
                label=f"seed {seed}" if d == DIMS[0] else None,
            )

        grouped = []
        for step, g in sub.groupby("step"):
            mean, _, n = mean_ci(g[metric])
            if n:
                grouped.append((step, mean))
        if grouped:
            arr = np.asarray(grouped, dtype=float)
            order = np.argsort(arr[:, 0])
            ax.plot(arr[order, 0] / 1_000_000, arr[order, 1], color="black", linewidth=1.9, label="mean" if d == DIMS[0] else None)

        ax.set_title(f"d={d}", fontsize=10)
        ax.grid(True, alpha=0.18)
        ax.set_xlim(0, 5.05)
        if panel_idx % ncols == 0:
            ax.set_ylabel(ylabel)

    for ax in axes[len(DIMS):]:
        ax.axis("off")
    for ax in axes[-ncols:]:
        if ax.axison:
            ax.set_xlabel("steps (M)")
    axes[0].legend(frameon=False, fontsize=8, ncol=2)
    savefig(name)


def plot_individual_per_d(df: pd.DataFrame, metric: str, ylabel: str, subfolder: str) -> None:
    metric_df = df.dropna(subset=[metric]).copy()
    metric_df[metric] = pd.to_numeric(metric_df[metric], errors="coerce")
    metric_df = metric_df.replace([np.inf, -np.inf], np.nan).dropna(subset=[metric])
    seeds = sorted(int(seed) for seed in metric_df["seed"].unique())
    seed_colors = plt.cm.tab10(np.linspace(0.0, 0.9, max(len(seeds), 1)))
    seed_color_lookup = {seed: seed_colors[i % len(seed_colors)] for i, seed in enumerate(seeds)}

    for d in DIMS:
        sub = metric_df[metric_df["d"] == d].copy()
        if sub.empty:
            continue

        plt.figure(figsize=(5.8, 3.6))
        for seed, run in sub.groupby("seed"):
            run = run.sort_values("step")
            plt.plot(
                run["step"] / 1_000_000,
                run[metric],
                color=seed_color_lookup[int(seed)],
                linewidth=1.15,
                alpha=0.55,
                label=f"seed {seed}",
            )

        grouped = []
        for step, g in sub.groupby("step"):
            mean, _, n = mean_ci(g[metric])
            if n:
                grouped.append((step, mean))
        if grouped:
            arr = np.asarray(grouped, dtype=float)
            order = np.argsort(arr[:, 0])
            plt.plot(arr[order, 0] / 1_000_000, arr[order, 1], color="black", linewidth=2.2, label="mean")

        plt.title(f"CelebA d={d}")
        plt.xlabel("training steps (M)")
        plt.ylabel(ylabel)
        plt.xlim(0, 5.05)
        plt.grid(True, alpha=0.22)
        plt.legend(frameon=False, fontsize=8, ncol=2)
        savefig_at(OUT_DIR / "individual_by_d" / subfolder / f"d{d:03d}_{subfolder}_over_steps")


def write_summary(df: pd.DataFrame) -> None:
    final = df[df["step"] == 5_000_000]
    rows = []
    for d in DIMS:
        row = {"d": d}
        for metric in ("mem", "fid", "loss"):
            mean, ci, n = mean_ci(final.loc[final["d"] == d, metric])
            row[f"{metric}_mean"] = mean
            row[f"{metric}_ci95"] = ci
            row[f"{metric}_n"] = n
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT_DIR / "celeba_highd_final_summary.csv", index=False)

    readme = """# CelebA high-d multiseed figures

Source: `results/celeba_diverse1k_bigmlp_sgd_lr001_m08_10k_5m` on ML6.

Sweep: CIFAR-style big MLP diffusion on the diverse CelebA 1k subset, seeds 42-46, `d_latent = 10, 20, ..., 100, 120, 140, 160, 180, 200`, 5M training steps.

Confidence intervals are 95% t-intervals over five seeds: `mean +/- t_0.975,4 * SEM`.

Generated files:
- `final_mem_mean_ci.*`, `final_mem_mean_no_ci.*`
- `final_fid_mean_ci.*`, `final_fid_mean_no_ci.*`
- `train_mem_over_steps_ci.*`
- `train_fid_over_steps_ci.*`
- `individual_mem_by_d_over_steps.*`
- `individual_fid_by_d_over_steps.*`
- `individual_by_d/mem/d*_mem_over_steps.*`
- `individual_by_d/fid/d*_fid_over_steps.*`
- `train_loss_over_steps_ci.*`
- `celeba_highd_final_summary.csv`
"""
    (OUT_DIR / "README.md").write_text(readme)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = read_metric_rows()
    df.to_csv(OUT_DIR / "celeba_highd_all_metrics.csv", index=False)

    plot_final(df, "mem", "memorization fraction", "final_mem_mean_ci", with_ci=True)
    plot_final(df, "mem", "memorization fraction", "final_mem_mean_no_ci", with_ci=False)
    plot_final(df, "fid", "FID", "final_fid_mean_ci", with_ci=True)
    plot_final(df, "fid", "FID", "final_fid_mean_no_ci", with_ci=False)

    plot_trajectory(df, "mem", "memorization fraction", "train_mem_over_steps_ci")
    plot_trajectory(df, "fid", "FID", "train_fid_over_steps_ci")
    plot_individual_by_d(df, "mem", "memorization fraction", "individual_mem_by_d_over_steps")
    plot_individual_by_d(df, "fid", "FID", "individual_fid_by_d_over_steps")
    plot_individual_per_d(df, "mem", "memorization fraction", "mem")
    plot_individual_per_d(df, "fid", "FID", "fid")
    plot_trajectory(df, "loss", "training loss", "train_loss_over_steps_ci")
    write_summary(df)
    print(f"Wrote figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
