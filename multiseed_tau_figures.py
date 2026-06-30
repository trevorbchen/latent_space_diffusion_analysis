"""Aggregate 5-seed tau_gen/tau_mem runs with confidence intervals."""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_rows(run_dir):
    path = run_dir / "metrics.jsonl"
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text().splitlines()
            if line.strip()]
    rows = [row for row in rows if "step" in row]
    rows.sort(key=lambda row: row["step"])
    return rows


def tau_gen(rows, key="test_loss", frac=1.05):
    vals = [row[key] for row in rows if row.get(key) is not None]
    if not vals:
        return None
    target = min(vals) * frac
    for row in rows:
        if row.get(key) is not None and row[key] <= target:
            return row["step"]
    return None


def tau_mem(rows, thresh=0.01):
    for row in rows:
        if row.get("memorization_fraction", 0.0) > thresh:
            return row["step"]
    return None


def ci95(values):
    vals = np.asarray([v for v in values if v is not None], dtype=float)
    if vals.size == 0:
        return math.nan, math.nan, math.nan, 0
    mean = float(vals.mean())
    if vals.size == 1:
        return mean, 0.0, 0.0, 1
    sem = float(vals.std(ddof=1) / math.sqrt(vals.size))
    return mean, 1.96 * sem, sem, int(vals.size)


def grouped_runs(root):
    groups = defaultdict(list)
    for run_dir in sorted(Path(root).glob("**/metrics.jsonl")):
        run = run_dir.parent
        cfg_path = run / "config.json"
        if not cfg_path.exists():
            continue
        cfg = json.loads(cfg_path.read_text())
        key = (
            cfg.get("d_intrinsic"),
            cfg.get("d_latent"),
            cfg.get("n"),
            float(cfg.get("sigma_noise", 0.0)),
        )
        groups[key].append((cfg.get("seed"), run, cfg))
    return groups


def aggregate(root):
    records = []
    for (dint, dlat, n, sigma_noise), runs in grouped_runs(root).items():
        seed_rows = []
        for seed, run, cfg in runs:
            rows = load_rows(run)
            if not rows:
                continue
            last = rows[-1]["step"]
            tg = tau_gen(rows)
            tm = tau_mem(rows)
            seed_rows.append({
                "seed": seed,
                "run": str(run),
                "tau_gen": tg,
                "tau_mem": tm,
                "tau_mem_censored": tm is None,
                "last_step": last,
            })
        if not seed_rows:
            continue
        tg_vals = [row["tau_gen"] for row in seed_rows]
        tm_vals = [
            row["tau_mem"] if row["tau_mem"] is not None else row["last_step"]
            for row in seed_rows
        ]
        tg_mean, tg_ci, tg_sem, tg_n = ci95(tg_vals)
        tm_mean, tm_ci, tm_sem, tm_n = ci95(tm_vals)
        records.append({
            "d_intrinsic": dint,
            "d_latent": dlat,
            "ratio": dlat / dint,
            "n": n,
            "sigma_noise": sigma_noise,
            "num_seeds": len(seed_rows),
            "tau_gen_mean": tg_mean,
            "tau_gen_ci95": tg_ci,
            "tau_gen_sem": tg_sem,
            "tau_gen_n": tg_n,
            "tau_mem_mean": tm_mean,
            "tau_mem_ci95": tm_ci,
            "tau_mem_sem": tm_sem,
            "tau_mem_n": tm_n,
            "tau_mem_censored_seeds": sum(row["tau_mem_censored"]
                                          for row in seed_rows),
            "effect_mem_minus_gen": tm_mean - tg_mean,
            "seed_rows": seed_rows,
        })
    records.sort(key=lambda rec: rec["d_latent"])
    return records


def write_csv(records, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = [
        "d_intrinsic", "d_latent", "ratio", "n", "sigma_noise", "num_seeds",
        "tau_gen_mean", "tau_gen_ci95", "tau_gen_sem", "tau_gen_n",
        "tau_mem_mean", "tau_mem_ci95", "tau_mem_sem", "tau_mem_n",
        "tau_mem_censored_seeds", "effect_mem_minus_gen",
    ]
    with (out_dir / "tau_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for rec in records:
            writer.writerow({key: rec[key] for key in keys})
    (out_dir / "tau_summary.json").write_text(json.dumps(records, indent=2))


def plot_timescales(records, out_dir):
    fig, ax = plt.subplots(figsize=(4.7, 3.1))
    x = np.asarray([rec["ratio"] for rec in records], dtype=float)
    tg = np.asarray([rec["tau_gen_mean"] for rec in records], dtype=float) / 1e6
    tm = np.asarray([rec["tau_mem_mean"] for rec in records], dtype=float) / 1e6
    tgerr = np.asarray([rec["tau_gen_ci95"] for rec in records], dtype=float) / 1e6
    tmerr = np.asarray([rec["tau_mem_ci95"] for rec in records], dtype=float) / 1e6
    cens = np.asarray([rec["tau_mem_censored_seeds"] > 0 for rec in records])

    ax.errorbar(x, tg, yerr=tgerr, fmt="o-", color="tab:blue",
                capsize=3, lw=1.4, ms=5, label=r"$\tau_{\rm gen}$")
    ax.errorbar(x[~cens], tm[~cens], yerr=tmerr[~cens], fmt="o-",
                color="tab:red", capsize=3, lw=1.4, ms=5,
                label=r"$\tau_{\rm mem}$")
    if np.any(cens):
        ax.errorbar(x[cens], tm[cens], yerr=tmerr[cens], fmt="^-",
                    color="tab:red", alpha=0.55, capsize=3, lw=1.2, ms=6,
                    label=r"$\tau_{\rm mem}$ censored seed(s)")
    ax.set_xlabel(r"$d_{\rm latent}/d_{\rm intrinsic}$")
    ax.set_ylabel("training steps (millions)")
    ax.set_title("5-seed mean timescales with 95% CI")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "tau_timescales_5seed.pdf")
    fig.savefig(out_dir / "tau_timescales_5seed.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="multiseed_runs/exp2_mlp_dlat_sn05")
    parser.add_argument("--out-dir", default="clean figures/multiseed_tau")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    records = aggregate(args.root)
    if not records:
        raise SystemExit(f"No complete metric runs found under {args.root}")
    write_csv(records, out_dir)
    plot_timescales(records, out_dir)
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
