from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRICS_ROOT = Path("results/cifar10_beta005_diverse1k_bigmlp_sgd_lr001_m08_10k_5m")
OUT_DIR = Path("diagnostics/cifar10_highd_multiseed_5m_figures")
DIMS = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260]
T95_N5 = 2.7764451051977987

SELECTED_RUNS = {
    20: [(42, "d20"), (43, "d20"), (44, "d20"), (45, "d20"), (46, "d20")],
    40: [(42, "d40"), (43, "d40"), (44, "d40"), (45, "d40"), (46, "d40")],
    60: [(42, "d60"), (43, "d60"), (44, "d60"), (45, "d60"), (46, "d60")],
    80: [(42, "d80"), (43, "d80"), (44, "d80"), (45, "d80"), (46, "d80")],
    100: [(42, "d100"), (43, "d100"), (44, "d100"), (45, "d100"), (46, "d100")],
    120: [(42, "d120"), (43, "d120"), (44, "d120"), (45, "d120"), (46, "d120")],
    140: [(42, "d140"), (43, "d140"), (44, "d140"), (45, "d140"), (46, "d140")],
    160: [(42, "d160"), (43, "d160"), (44, "d160"), (45, "d160"), (46, "d160")],
    180: [(42, "d180"), (43, "d180"), (44, "d180"), (45, "d180"), (46, "d180")],
    200: [(42, "d200"), (43, "d200"), (44, "d200"), (45, "d200"), (46, "d200")],
    220: [(42, "d220"), (43, "d220"), (44, "d220"), (45, "d220"), (47, "d220")],
    240: [(42, "d240"), (43, "d240"), (45, "d240"), (46, "d240"), (47, "d240")],
    260: [(43, "d260"), (47, "d260"), (48, "d260"), (49, "d260"), (50, "d260")],
}


def finite_or_nan(value):
    if value is None:
        return math.nan
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out if math.isfinite(out) else math.nan


def fid_retry_value(run_dir: Path) -> tuple[float, str]:
    path = run_dir / "fid_retry.json"
    if not path.exists():
        return math.nan, ""
    data = json.loads(path.read_text())
    value = data.get("mean_fid")
    if value is None:
        value = data.get("best_fid")
        source = "fid_retry_best"
    else:
        source = "fid_retry_mean"
    return finite_or_nan(value), source


def read_metric_rows() -> pd.DataFrame:
    rows = []
    for d in DIMS:
        for seed, folder in SELECTED_RUNS[d]:
            run_dir = METRICS_ROOT / f"seed{seed}" / folder
            path = run_dir / "metrics.jsonl"
            if not path.exists():
                raise FileNotFoundError(path)
            retry_fid, retry_source = fid_retry_value(run_dir)
            with path.open() as f:
                for line in f:
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if "step" not in row:
                        continue
                    step = int(row["step"])
                    fid = finite_or_nan(row.get("fid"))
                    fid_source = "metrics"
                    if step == 5_000_000 and not math.isfinite(fid) and math.isfinite(retry_fid):
                        fid = retry_fid
                        fid_source = retry_source
                    rows.append(
                        {
                            "seed": seed,
                            "folder": folder,
                            "d": d,
                            "step": step,
                            "mem": finite_or_nan(row.get("memorization_fraction_pixel")),
                            "fid": fid,
                            "fid_source": fid_source,
                            "loss": finite_or_nan(row.get("train_loss_step")),
                            "score_loss_per_dim": finite_or_nan(row.get("train_loss")),
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

    plt.figure(figsize=(7.0, 4.2))
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
    plt.xticks(DIMS, rotation=30)
    plt.grid(True, alpha=0.22)
    savefig(name)


def plot_trajectory(df: pd.DataFrame, metric: str, ylabel: str, name: str) -> None:
    metric_df = df.dropna(subset=[metric]).copy()
    metric_df[metric] = pd.to_numeric(metric_df[metric], errors="coerce")
    metric_df = metric_df.replace([np.inf, -np.inf], np.nan).dropna(subset=[metric])

    plt.figure(figsize=(8.4, 5.2))
    colors = plt.cm.viridis(np.linspace(0.06, 0.94, len(DIMS)))
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
        plt.plot(x, y, label=f"d={d}", linewidth=1.75, color=color)
        plt.fill_between(x, y - ci, y + ci, color=color, alpha=0.10, linewidth=0)
    plt.xlabel("training steps (millions)")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.22)
    plt.legend(ncol=4, fontsize=8, frameon=False)
    savefig(name)


def plot_individual_by_d(df: pd.DataFrame, metric: str, ylabel: str, name: str) -> None:
    metric_df = df.dropna(subset=[metric]).copy()
    metric_df[metric] = pd.to_numeric(metric_df[metric], errors="coerce")
    metric_df = metric_df.replace([np.inf, -np.inf], np.nan).dropna(subset=[metric])

    ncols = 3
    nrows = int(math.ceil(len(DIMS) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(10.6, 2.35 * nrows), sharex=True, sharey=True)
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
                linewidth=0.9,
                alpha=0.38,
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
            ax.plot(arr[order, 0] / 1_000_000, arr[order, 1], color="black", linewidth=1.8, label="mean" if d == DIMS[0] else None)

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
    axes[0].legend(frameon=False, fontsize=7, ncol=2)
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

        plt.title(f"CIFAR-10 d={d}")
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
        row = {"d": d, "selected_runs": ";".join(f"seed{s}/{folder}" for s, folder in SELECTED_RUNS[d])}
        for metric in ("mem", "fid", "loss", "score_loss_per_dim"):
            mean, ci, n = mean_ci(final.loc[final["d"] == d, metric])
            row[f"{metric}_mean"] = mean
            row[f"{metric}_ci95"] = ci
            row[f"{metric}_n"] = n
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT_DIR / "cifar10_highd_final_summary.csv", index=False)

    selection_rows = []
    for d in DIMS:
        for seed, folder in SELECTED_RUNS[d]:
            selection_rows.append({"d": d, "seed": seed, "folder": folder, "clipped": False})
    pd.DataFrame(selection_rows).to_csv(OUT_DIR / "cifar10_selected_runs.csv", index=False)

    readme = """# CIFAR-10 high-d multiseed figures

Source: `results/cifar10_beta005_diverse1k_bigmlp_sgd_lr001_m08_10k_5m` on ML6.

Sweep: CIFAR-10 diverse-1k VAE-latent MLP diffusion, 5M training steps,
`d_latent = 20, 40, ..., 260`.

The clean selected set intentionally excludes all gradient-clipped diagnostic
runs. Replacement seeds are used for unstable or contaminated runs:

- d220: seeds 42, 43, 44, 45, 47
- d240: seeds 42, 43, 45, 46, 47
- d260: seeds 43, 47, 48, 49, 50

All selected runs are unclipped and use the same optimizer settings:
SGD, lr=0.001, momentum=0.80.

If the final metrics row had a nonfinite FID but a `fid_retry.json` with
successful finite samples existed, the final FID summary uses the mean retry
FID. Training-trajectory plots still drop nonfinite evaluation rows.

Confidence intervals are 95% t-intervals over five seeds:
`mean +/- t_0.975,4 * SEM`.

Generated files:

- `final_mem_mean_ci.*`, `final_mem_mean_no_ci.*`
- `final_fid_mean_ci.*`, `final_fid_mean_no_ci.*`
- `final_loss_mean_ci.*`, `final_loss_mean_no_ci.*`
- `train_mem_over_steps_ci.*`
- `train_fid_over_steps_ci.*`
- `individual_mem_by_d_over_steps.*`
- `individual_fid_by_d_over_steps.*`
- `individual_by_d/mem/d*_mem_over_steps.*`
- `individual_by_d/fid/d*_fid_over_steps.*`
- `train_loss_over_steps_ci.*`
- `train_score_loss_per_dim_over_steps_ci.*`
- `cifar10_highd_final_summary.csv`
- `cifar10_selected_runs.csv`
- `cifar10_highd_all_metrics.csv`
"""
    (OUT_DIR / "README.md").write_text(readme)


def main() -> None:
    global METRICS_ROOT, OUT_DIR
    p = argparse.ArgumentParser()
    p.add_argument("--metrics_root", default=str(METRICS_ROOT))
    p.add_argument("--out", default=str(OUT_DIR))
    args = p.parse_args()
    METRICS_ROOT = Path(args.metrics_root)
    OUT_DIR = Path(args.out)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = read_metric_rows()
    df.to_csv(OUT_DIR / "cifar10_highd_all_metrics.csv", index=False)

    plot_final(df, "mem", "memorization fraction", "final_mem_mean_ci", with_ci=True)
    plot_final(df, "mem", "memorization fraction", "final_mem_mean_no_ci", with_ci=False)
    plot_final(df, "fid", "FID", "final_fid_mean_ci", with_ci=True)
    plot_final(df, "fid", "FID", "final_fid_mean_no_ci", with_ci=False)
    plot_final(df, "loss", "raw training loss", "final_loss_mean_ci", with_ci=True)
    plot_final(df, "loss", "raw training loss", "final_loss_mean_no_ci", with_ci=False)

    plot_trajectory(df, "mem", "memorization fraction", "train_mem_over_steps_ci")
    plot_trajectory(df, "fid", "FID", "train_fid_over_steps_ci")
    plot_individual_by_d(df, "mem", "memorization fraction", "individual_mem_by_d_over_steps")
    plot_individual_by_d(df, "fid", "FID", "individual_fid_by_d_over_steps")
    plot_individual_per_d(df, "mem", "memorization fraction", "mem")
    plot_individual_per_d(df, "fid", "FID", "fid")
    plot_trajectory(df, "loss", "raw training loss", "train_loss_over_steps_ci")
    plot_trajectory(df, "score_loss_per_dim", "training loss per latent dim", "train_score_loss_per_dim_over_steps_ci")
    write_summary(df)
    print(f"Wrote figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
