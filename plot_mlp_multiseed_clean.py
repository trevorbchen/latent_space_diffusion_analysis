"""Clean multi-seed MLP figures for the 5M-step d_latent sweep."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RUN_RE = re.compile(r"di(?P<dint>\d+)_d(?P<dlat>\d+)_n(?P<n>\d+)_s(?P<seed>\d+)")


def ci95(vals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vals = np.asarray(vals, dtype=float)
    mean = np.nanmean(vals, axis=0)
    counts = np.sum(~np.isnan(vals), axis=0)
    std = np.nanstd(vals, axis=0, ddof=1)
    sem = np.divide(std, np.sqrt(counts), out=np.zeros_like(std), where=counts > 1)
    ci = 1.96 * sem
    ci[counts <= 1] = 0.0
    return mean, ci


def scalar_ci95(vals: list[float]) -> tuple[float, float, float]:
    arr = np.asarray(vals, dtype=float)
    mean = float(arr.mean())
    if len(arr) <= 1:
        return mean, 0.0, 0.0
    sem = float(arr.std(ddof=1) / math.sqrt(len(arr)))
    return mean, 1.96 * sem, sem


def read_metrics(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if "step" in row:
            rows.append(row)
    rows.sort(key=lambda row: row["step"])
    return rows


def load_runs(root: Path):
    runs = []
    for metrics_path in sorted(root.glob("di*_d*_n*_s*/metrics.jsonl")):
        match = RUN_RE.fullmatch(metrics_path.parent.name)
        if match is None:
            continue
        rows = read_metrics(metrics_path)
        if not rows:
            continue
        info = {key: int(val) for key, val in match.groupdict().items()}
        runs.append({**info, "path": metrics_path.parent, "rows": rows})
    return runs


def aligned_metric(runs: list[dict], metric: str, transform=None):
    by_d = defaultdict(list)
    for run in runs:
        if run["rows"][-1]["step"] < 5_000_000:
            continue
        by_d[run["dlat"]].append(run)

    out = {}
    for dlat, group in by_d.items():
        common_steps = set(group[0]["rows"][i]["step"] for i in range(len(group[0]["rows"])))
        for run in group[1:]:
            common_steps &= {row["step"] for row in run["rows"]}
        steps = np.asarray(sorted(common_steps), dtype=int)
        vals = []
        seeds = []
        for run in sorted(group, key=lambda item: item["seed"]):
            row_by_step = {row["step"]: row for row in run["rows"]}
            series = []
            for step in steps:
                value = row_by_step[step].get(metric, np.nan)
                if transform is not None and not np.isnan(value):
                    value = transform(value, run)
                series.append(value)
            vals.append(series)
            seeds.append(run["seed"])
        vals = np.asarray(vals, dtype=float)
        mean, ci = ci95(vals)
        out[dlat] = {"steps": steps, "values": vals, "mean": mean, "ci": ci, "seeds": seeds}
    return out


def summarize_runs(runs: list[dict]):
    rows = []
    for run in runs:
        if run["rows"][-1]["step"] < 5_000_000:
            continue
        mem = np.asarray([row["memorization_fraction"] for row in run["rows"]], dtype=float)
        score = np.asarray([row["score_error"] for row in run["rows"]], dtype=float)
        rows.append({
            "d_latent": run["dlat"],
            "d_intrinsic": run["dint"],
            "n": run["n"],
            "seed": run["seed"],
            "final_step": run["rows"][-1]["step"],
            "final_mem_ratio": float(mem[-1]),
            "max_mem_ratio": float(mem.max()),
            "max_mem_step": int(run["rows"][int(mem.argmax())]["step"]),
            "final_score_error": float(score[-1]),
            "min_score_error": float(score.min()),
            "min_score_step": int(run["rows"][int(score.argmin())]["step"]),
        })
    rows.sort(key=lambda row: (row["d_latent"], row["seed"]))
    return rows


def write_csvs(seed_rows: list[dict], out_dir: Path):
    seed_keys = [
        "d_latent", "d_intrinsic", "n", "seed", "final_step",
        "final_mem_ratio", "max_mem_ratio", "max_mem_step",
        "final_score_error", "min_score_error", "min_score_step",
    ]
    with (out_dir / "mlp_seed_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=seed_keys)
        writer.writeheader()
        writer.writerows(seed_rows)

    by_d = defaultdict(list)
    for row in seed_rows:
        by_d[row["d_latent"]].append(row)

    summary_keys = [
        "d_latent", "num_seeds", "seeds",
        "max_mem_mean", "max_mem_ci95", "max_mem_sem",
        "final_mem_mean", "final_mem_ci95", "final_mem_sem",
        "final_score_error_mean", "final_score_error_ci95",
        "final_score_error_sem",
    ]
    summary_rows = []
    for dlat in sorted(by_d):
        group = by_d[dlat]
        max_mean, max_ci, max_sem = scalar_ci95([row["max_mem_ratio"] for row in group])
        final_mean, final_ci, final_sem = scalar_ci95([row["final_mem_ratio"] for row in group])
        score_mean, score_ci, score_sem = scalar_ci95([row["final_score_error"] for row in group])
        summary_rows.append({
            "d_latent": dlat,
            "num_seeds": len(group),
            "seeds": " ".join(str(row["seed"]) for row in sorted(group, key=lambda row: row["seed"])),
            "max_mem_mean": max_mean,
            "max_mem_ci95": max_ci,
            "max_mem_sem": max_sem,
            "final_mem_mean": final_mean,
            "final_mem_ci95": final_ci,
            "final_mem_sem": final_sem,
            "final_score_error_mean": score_mean,
            "final_score_error_ci95": score_ci,
            "final_score_error_sem": score_sem,
        })
    with (out_dir / "mlp_d_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_keys)
        writer.writeheader()
        writer.writerows(summary_rows)
    return summary_rows


def palette(dvals):
    cmap = plt.get_cmap("tab10")
    return {dlat: cmap(i % 10) for i, dlat in enumerate(sorted(dvals))}


def plot_curve(metric_data, out_dir: Path, filename: str, ylabel: str,
               title: str, percent: bool = False):
    dvals = sorted(metric_data)
    colors = palette(dvals)
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    for dlat in dvals:
        rec = metric_data[dlat]
        x = rec["steps"] / 1e6
        mean = rec["mean"] * (100.0 if percent else 1.0)
        ci = rec["ci"] * (100.0 if percent else 1.0)
        color = colors[dlat]
        ax.plot(x, mean, color=color, lw=1.8, label=rf"$d_{{lat}}={dlat}$")
        ax.fill_between(x, mean - ci, mean + ci, color=color, alpha=0.14, linewidth=0)
    ax.set_xlabel("training steps (millions)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.22)
    ax.legend(ncol=2, frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f"{filename}.png", dpi=180)
    fig.savefig(out_dir / f"{filename}.pdf")
    plt.close(fig)


def plot_max_mem(summary_rows: list[dict], seed_rows: list[dict], out_dir: Path):
    dvals = [row["d_latent"] for row in summary_rows]
    means = np.asarray([row["max_mem_mean"] for row in summary_rows]) * 100.0
    cis = np.asarray([row["max_mem_ci95"] for row in summary_rows]) * 100.0
    colors = palette(dvals)

    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    x = np.arange(len(dvals))
    ax.bar(
        x, means, yerr=cis, capsize=3, width=0.72,
        color=[colors[d] for d in dvals], alpha=0.78, edgecolor="black",
        linewidth=0.45,
    )
    rng = np.random.default_rng(7)
    by_d = defaultdict(list)
    for row in seed_rows:
        by_d[row["d_latent"]].append(row["max_mem_ratio"] * 100.0)
    for i, dlat in enumerate(dvals):
        vals = by_d[dlat]
        jitter = rng.uniform(-0.18, 0.18, size=len(vals))
        ax.scatter(
            np.full(len(vals), i) + jitter, vals, s=18,
            color="white", edgecolor="black", linewidth=0.55, zorder=5,
        )
    ax.set_xticks(x, [str(d) for d in dvals])
    ax.set_xlabel(r"$d_{latent}$")
    ax.set_ylabel("max memorization ratio (%)")
    ax.set_title("Maximum memorization ratio across training, 5-seed mean")
    ax.grid(True, axis="y", alpha=0.22)
    fig.tight_layout()
    fig.savefig(out_dir / "max_mem_ratio_bar_ci.png", dpi=180)
    fig.savefig(out_dir / "max_mem_ratio_bar_ci.pdf")
    plt.close(fig)


def plot_final_mem(summary_rows: list[dict], seed_rows: list[dict], out_dir: Path):
    dvals = [row["d_latent"] for row in summary_rows]
    means = np.asarray([row["final_mem_mean"] for row in summary_rows]) * 100.0
    cis = np.asarray([row["final_mem_ci95"] for row in summary_rows]) * 100.0
    colors = palette(dvals)

    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    x = np.arange(len(dvals))
    ax.errorbar(
        x, means, yerr=cis, fmt="o-", color="#202020",
        ecolor="#202020", capsize=3, lw=1.5, ms=4.5,
    )
    for i, dlat in enumerate(dvals):
        ax.scatter(i, means[i], s=46, color=colors[dlat], edgecolor="black",
                   linewidth=0.45, zorder=4)
    ax.set_xticks(x, [str(d) for d in dvals])
    ax.set_xlabel(r"$d_{latent}$")
    ax.set_ylabel("final memorization ratio (%)")
    ax.set_title("Final memorization ratio at 5M steps, 5-seed mean")
    ax.grid(True, alpha=0.22)
    fig.tight_layout()
    fig.savefig(out_dir / "final_mem_ratio_line_ci.png", dpi=180)
    fig.savefig(out_dir / "final_mem_ratio_line_ci.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path,
                        default=Path("multiseed_runs/exp2_mlp_dlat_sn05_5m_combined"))
    parser.add_argument("--out-dir", type=Path,
                        default=Path("clean figures") / "mlp_multiseed_5m")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    runs = load_runs(args.root)
    if not runs:
        raise SystemExit(f"No MLP runs found under {args.root}")

    seed_rows = summarize_runs(runs)
    summary_rows = write_csvs(seed_rows, args.out_dir)
    score = aligned_metric(runs, "score_error")
    score_per_dim = aligned_metric(
        runs, "score_error",
        transform=lambda value, run: value / run["dlat"],
    )
    mem = aligned_metric(runs, "memorization_fraction")

    plot_curve(
        score, args.out_dir, "score_error_over_steps_ci",
        "score error", "Score error over training, 5-seed mean with 95% CI",
    )
    plot_curve(
        score_per_dim, args.out_dir, "score_error_per_dim_over_steps_ci",
        r"score error / $d_{latent}$",
        "Dimension-normalized score error over training, 5-seed mean with 95% CI",
    )
    plot_curve(
        mem, args.out_dir, "mem_ratio_over_steps_ci",
        "memorization ratio (%)",
        "Memorization ratio over training, 5-seed mean with 95% CI",
        percent=True,
    )
    plot_max_mem(summary_rows, seed_rows, args.out_dir)
    plot_final_mem(summary_rows, seed_rows, args.out_dir)

    (args.out_dir / "README.md").write_text("""# MLP multi-seed 5M figures

Clean MLP section figures from the 5M-step sweep.

Runs:

- `d_intrinsic=5`
- `d_latent in {5, 8, 10, 12, 15, 20, 25, 30, 35, 40}`
- `n=500`
- `sigma_noise=0.5`
- seeds `42, 43, 44, 45, 46`
- pure MLP, not RFNN

Figures:

- `score_error_over_steps_ci`: mean score error over training with 95% CI.
- `score_error_per_dim_over_steps_ci`: mean score error divided by
  `d_latent` over training with 95% CI.
- `mem_ratio_over_steps_ci`: mean memorization ratio over training with 95% CI.
- `max_mem_ratio_bar_ci`: bar plot of each seed's maximum memorization ratio,
  averaged over seeds with 95% CI.
- `final_mem_ratio_line_ci`: final memorization ratio at 5M steps with 95% CI.

Tables:

- `mlp_seed_summary.csv`: one row per seed/run.
- `mlp_d_summary.csv`: one row per `d_latent`, averaged over seeds.
""")
    print(f"wrote {args.out_dir}")


if __name__ == "__main__":
    main()
