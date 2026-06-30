from __future__ import annotations

import csv
import json
import math
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
INPUT_ROOT = ROOT / "tmp_real_vae_clean_inputs"
CLEAN_ROOT = ROOT / "clean figures"


DATASETS = {
    "celeba": {
        "title": "CelebA",
        "out_dir": CLEAN_ROOT / "celeba_vae_diagnostics",
        "checkpoint_prefix": "celeba_resnet_modernloss_d",
        "recon_mem_dir": INPUT_ROOT
        / "diagnostics"
        / "celeba_resnet_modernloss_diverse_reconstruction_mem_full",
        "dims": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200],
        "contact_dims": [10, 30, 60, 100, 160, 200],
        "note": (
            "CelebA uses the modern ResNet VAE objective "
            "(beta warmup, capacity warmup, free bits, EMA, and early stopping)."
        ),
    },
    "cifar10": {
        "title": "CIFAR-10",
        "out_dir": CLEAN_ROOT / "cifar10_vae_diagnostics",
        "checkpoint_prefix": "cifar10_resnet_beta005_d",
        "recon_mem_dir": INPUT_ROOT / "diagnostics" / "cifar10_beta005_diverse_reconstruction_mem",
        "dims": [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260],
        "contact_dims": [20, 60, 100, 160, 220, 260],
        "note": (
            "CIFAR-10 uses the beta=0.05 ResNet VAE family retrained on "
            "20-step latent intervals through d=260."
        ),
    },
}


def read_jsonl(path: Path) -> tuple[dict, list[dict]]:
    config: dict = {}
    rows: list[dict] = []
    if not path.exists():
        return config, rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("event") == "config":
                config = record
            elif "epoch" in record and "test_loss" in record:
                rows.append(record)
    return config, rows


def checkpoint_dir(prefix: str, d: int) -> Path:
    return INPUT_ROOT / "vae_checkpoints" / f"{prefix}{d}"


def flatten_config(config: dict) -> dict:
    cfg = config.get("cfg", {})
    return {
        "dataset": config.get("dataset", ""),
        "arch": cfg.get("arch", ""),
        "image_channels": cfg.get("image_channels", ""),
        "image_size": cfg.get("image_size", ""),
        "hidden_dims": json.dumps(cfg.get("hidden_dims", [])),
        "n_train": config.get("n_train", ""),
        "n_val": config.get("n_val", ""),
        "epochs_requested": config.get("epochs", ""),
        "lr": config.get("lr", ""),
        "beta": config.get("beta", ""),
        "beta_warmup_frac": config.get("beta_warmup_frac", ""),
        "max_capacity": config.get("max_capacity", ""),
        "capacity_warmup_frac": config.get("capacity_warmup_frac", ""),
        "mse_weight": config.get("mse_weight", ""),
        "l1_weight": config.get("l1_weight", ""),
        "free_bits": config.get("free_bits", ""),
        "ema_decay": config.get("ema_decay", ""),
        "grad_clip": config.get("grad_clip", ""),
        "early_stop_patience": config.get("early_stop_patience", ""),
        "early_stop_tol": config.get("early_stop_tol", ""),
    }


def training_summary(dataset_cfg: dict) -> tuple[list[dict], dict[int, list[dict]]]:
    prefix = dataset_cfg["checkpoint_prefix"]
    summaries: list[dict] = []
    curves: dict[int, list[dict]] = {}
    for d in dataset_cfg["dims"]:
        log_path = checkpoint_dir(prefix, d) / "training.log"
        config, rows = read_jsonl(log_path)
        if not rows:
            continue
        curves[d] = rows
        final = rows[-1]
        best = min(rows, key=lambda r: r.get("test_loss", math.inf))
        row = {
            "d_latent": d,
            **flatten_config(config),
            "final_epoch": final.get("epoch"),
            "final_step": final.get("step"),
            "final_lr": final.get("lr"),
            "final_beta": final.get("beta"),
            "final_capacity": final.get("capacity"),
            "final_train_loss": final.get("train_loss"),
            "final_train_recon": final.get("train_recon"),
            "final_train_mse": final.get("train_mse"),
            "final_train_l1": final.get("train_l1"),
            "final_train_kl": final.get("train_kl"),
            "final_test_loss": final.get("test_loss"),
            "final_test_recon": final.get("test_recon"),
            "final_test_mse": final.get("test_mse"),
            "final_test_l1": final.get("test_l1"),
            "final_test_kl": final.get("test_kl"),
            "best_epoch": best.get("epoch"),
            "best_step": best.get("step"),
            "best_test_loss": best.get("test_loss"),
            "best_test_recon": best.get("test_recon"),
            "best_test_mse": best.get("test_mse"),
            "best_test_l1": best.get("test_l1"),
            "best_test_kl": best.get("test_kl"),
            "epochs_since_improve_at_stop": final.get("epochs_since_improve"),
        }
        summaries.append(row)
    return summaries, curves


def read_reconstruction_mem(dataset_cfg: dict) -> list[dict]:
    path = dataset_cfg["recon_mem_dir"] / "reconstruction_mem_summary.json"
    if not path.exists():
        return []
    rows = json.loads(path.read_text(encoding="utf-8"))
    dims = set(dataset_cfg["dims"])
    return [r for r in rows if int(r.get("d_latent", -1)) in dims]


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def savefig(out_dir: Path, stem: str) -> None:
    for suffix in ("png", "pdf"):
        plt.savefig(out_dir / f"{stem}.{suffix}", dpi=220, bbox_inches="tight")
    plt.close()


def plot_final_metrics(title: str, out_dir: Path, summary_rows: list[dict]) -> None:
    rows = sorted(summary_rows, key=lambda r: r["d_latent"])
    d = np.array([r["d_latent"] for r in rows])
    metrics = [
        ("best_test_loss", "Best Val Loss"),
        ("best_test_mse", "Best Val MSE"),
        ("best_test_l1", "Best Val L1"),
        ("best_test_kl", "Best Val KL"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    fig.suptitle(f"{title} VAE Metrics vs Latent Dimension", fontsize=14)
    for ax, (key, ylabel) in zip(axes.flat, metrics):
        y = np.array([float(r[key]) for r in rows])
        ax.plot(d, y, marker="o", linewidth=2)
        ax.set_xlabel(r"$d_{\mathrm{latent}}$")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
    savefig(out_dir, "vae_metrics_vs_d")


def plot_training_curves(title: str, out_dir: Path, curves: dict[int, list[dict]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, max(len(curves), 1)))
    for color, d in zip(colors, sorted(curves)):
        rows = curves[d]
        epochs = [r["epoch"] for r in rows]
        axes[0].plot(epochs, [r["test_mse"] for r in rows], color=color, linewidth=1.6, label=str(d))
        axes[1].plot(epochs, [r["test_kl"] for r in rows], color=color, linewidth=1.6, label=str(d))
    axes[0].set_title("Validation Reconstruction MSE")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("MSE")
    axes[1].set_title("Validation KL")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("KL")
    for ax in axes:
        ax.grid(True, alpha=0.25)
    axes[1].legend(title=r"$d_{\mathrm{latent}}$", ncols=2, fontsize=8, title_fontsize=9)
    fig.suptitle(f"{title} VAE Training Curves", fontsize=14)
    savefig(out_dir, "vae_training_curves")


def plot_reconstruction_mem(title: str, out_dir: Path, mem_rows: list[dict]) -> None:
    rows = sorted(mem_rows, key=lambda r: int(r["d_latent"]))
    d = np.array([int(r["d_latent"]) for r in rows])
    mem = np.array([float(r["memorized_fraction"]) for r in rows]) * 100.0
    ratio = np.array([float(r["mean_nn_ratio"]) for r in rows])
    mse = np.array([float(r["mean_pixel_mse"]) for r in rows])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    axes[0].plot(d, mem, marker="o", linewidth=2, color="#d62728")
    axes[0].set_ylabel("reconstruction memorized (%)")
    axes[0].set_ylim(-2, 102)

    axes[1].plot(d, ratio, marker="o", linewidth=2, color="#1f77b4")
    axes[1].axhline(1 / 3, color="black", linestyle="--", linewidth=1, alpha=0.7)
    axes[1].set_ylabel("mean NN1 / NN2 ratio")

    axes[2].plot(d, mse, marker="o", linewidth=2, color="#2ca02c")
    axes[2].set_ylabel("mean pixel MSE")

    for ax in axes:
        ax.set_xlabel(r"$d_{\mathrm{latent}}$")
        ax.grid(True, alpha=0.25)
    fig.suptitle(f"{title} Encode-Decode Reconstruction Memorization", fontsize=14)
    savefig(out_dir, "reconstruction_mem_vs_d")


def plot_metric_join(title: str, out_dir: Path, summary_rows: list[dict], mem_rows: list[dict]) -> None:
    by_d = {int(r["d_latent"]): r for r in mem_rows}
    rows = [r for r in sorted(summary_rows, key=lambda r: r["d_latent"]) if int(r["d_latent"]) in by_d]
    d = np.array([int(r["d_latent"]) for r in rows])
    val_mse = np.array([float(r["best_test_mse"]) for r in rows])
    recon_mem = np.array([float(by_d[int(r["d_latent"])]["memorized_fraction"]) for r in rows]) * 100
    recon_mse = np.array([float(by_d[int(r["d_latent"])]["mean_pixel_mse"]) for r in rows])

    fig, ax1 = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax2 = ax1.twinx()
    ax1.plot(d, val_mse, marker="o", linewidth=2, color="#1f77b4", label="VAE val MSE")
    ax1.plot(d, recon_mse, marker="s", linewidth=2, color="#2ca02c", label="subset recon MSE")
    ax2.plot(d, recon_mem, marker="^", linewidth=2, color="#d62728", label="recon mem")
    ax1.set_xlabel(r"$d_{\mathrm{latent}}$")
    ax1.set_ylabel("MSE")
    ax2.set_ylabel("reconstruction memorized (%)")
    ax2.set_ylim(-2, 102)
    ax1.grid(True, alpha=0.25)
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines], loc="center right")
    ax1.set_title(f"{title}: VAE Quality vs Reconstruction Memorization")
    savefig(out_dir, "vae_quality_and_reconstruction_mem")


def copy_recon_grids(dataset_cfg: dict, out_dir: Path) -> None:
    prefix = dataset_cfg["checkpoint_prefix"]
    grid_dir = out_dir / "recon_grids"
    grid_dir.mkdir(parents=True, exist_ok=True)
    for d in dataset_cfg["dims"]:
        src = checkpoint_dir(prefix, d) / "recon_grid.png"
        if src.exists():
            shutil.copy2(src, grid_dir / f"d{d}_recon_grid.png")

    # Copy any one-off encode/decode comparison grids we have for the same dataset.
    if dataset_cfg["title"] == "CIFAR-10":
        diag_dir = INPUT_ROOT / "diagnostics" / "cifar10_vae_recon_compare_beta005"
        for src in diag_dir.glob("*.png"):
            shutil.copy2(src, grid_dir / src.name)


def plot_contact_sheet(dataset_cfg: dict, out_dir: Path) -> None:
    prefix = dataset_cfg["checkpoint_prefix"]
    dims = [d for d in dataset_cfg["contact_dims"] if (checkpoint_dir(prefix, d) / "recon_grid.png").exists()]
    if not dims:
        return
    ncols = 2
    nrows = math.ceil(len(dims) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4.2 * nrows), constrained_layout=True)
    axes_arr = np.atleast_1d(axes).reshape(nrows, ncols)
    for ax in axes_arr.flat:
        ax.axis("off")
    for ax, d in zip(axes_arr.flat, dims):
        img = mpimg.imread(checkpoint_dir(prefix, d) / "recon_grid.png")
        ax.imshow(img)
        ax.set_title(rf"$d_{{latent}}={d}$", fontsize=12)
    fig.suptitle(f"{dataset_cfg['title']} VAE Reconstruction Grids", fontsize=15)
    savefig(out_dir, "vae_reconstruction_contact_sheet")


def write_readme(dataset_cfg: dict, out_dir: Path, n_summary: int, n_mem: int) -> None:
    title = dataset_cfg["title"]
    dims = ", ".join(str(d) for d in dataset_cfg["dims"])
    text = f"""# {title} VAE Diagnostics

Clean VAE-side diagnostics for the real-data latent diffusion experiments.

Included latent dimensions: `{dims}`.

What is here:

- `vae_training_summary.csv`: final and best validation metrics parsed from each VAE training log.
- `reconstruction_mem_summary.csv`: encode/decode reconstruction memorization on the same diverse 1k subset used for diffusion training.
- `vae_metrics_vs_d.png/pdf`: best validation loss, MSE, L1, and KL against latent dimension.
- `vae_training_curves.png/pdf`: validation reconstruction MSE and KL over VAE training epochs.
- `reconstruction_mem_vs_d.png/pdf`: reconstruction memorization, NN1/NN2 ratio, and pixel MSE against latent dimension.
- `vae_quality_and_reconstruction_mem.png/pdf`: compact comparison of VAE quality and reconstruction-memorization behavior.
- `vae_reconstruction_contact_sheet.png/pdf`: representative reconstruction grids copied from the VAE checkpoints.
- `recon_grids/`: per-dimension reconstruction grids and any extra encode/decode comparison grids.

Reconstruction memorization uses the Bonnaire/Somepalli nearest-neighbor ratio
test in pixel space on the 1k subset:
`NN1(reconstruction, training subset) / NN2(training image, training subset) < 1/3`.
The VAE was trained on the full available training split; this diagnostic asks
whether encoding and decoding the 1k diffusion subset returns images close
enough to their original subset examples to trigger the memorization criterion.

{dataset_cfg["note"]}

Parsed `{n_summary}` VAE training logs and `{n_mem}` reconstruction-mem rows.
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def build_dataset(name: str, dataset_cfg: dict) -> None:
    out_dir = dataset_cfg["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "recon_grids").mkdir(exist_ok=True)

    summary_rows, curves = training_summary(dataset_cfg)
    mem_rows = read_reconstruction_mem(dataset_cfg)
    write_csv(out_dir / "vae_training_summary.csv", summary_rows)
    write_csv(out_dir / "reconstruction_mem_summary.csv", mem_rows)

    if summary_rows:
        plot_final_metrics(dataset_cfg["title"], out_dir, summary_rows)
        plot_training_curves(dataset_cfg["title"], out_dir, curves)
    if mem_rows:
        plot_reconstruction_mem(dataset_cfg["title"], out_dir, mem_rows)
    if summary_rows and mem_rows:
        plot_metric_join(dataset_cfg["title"], out_dir, summary_rows, mem_rows)
    copy_recon_grids(dataset_cfg, out_dir)
    plot_contact_sheet(dataset_cfg, out_dir)
    write_readme(dataset_cfg, out_dir, len(summary_rows), len(mem_rows))


def main() -> None:
    for name, cfg in DATASETS.items():
        build_dataset(name, cfg)


if __name__ == "__main__":
    main()
