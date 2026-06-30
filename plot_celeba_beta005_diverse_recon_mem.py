from __future__ import annotations

import json
from pathlib import Path
import argparse

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir",
        default="diagnostic figures/celeba_beta005_diverse_reconstruction_mem",
        help="Directory containing reconstruction_mem_summary.json.",
    )
    args = parser.parse_args()
    in_dir = Path(args.dir)
    summary = in_dir / "reconstruction_mem_summary.json"
    out = in_dir / "reconstruction_mem_curve.png"

    rows = json.loads(summary.read_text())
    d = [row["d_latent"] for row in rows]
    mem = [row["memorized_fraction"] for row in rows]
    ratio = [row["mean_nn_ratio"] for row in rows]

    fig, ax1 = plt.subplots(figsize=(7.0, 4.4))
    ax1.plot(d, mem, marker="o", color="#d73027", linewidth=2.2, label="memorized fraction")
    ax1.set_xlabel(r"$d_{latent}$")
    ax1.set_ylabel("memorized fraction", color="#d73027")
    ax1.tick_params(axis="y", labelcolor="#d73027")
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(d, ratio, marker="s", color="#4575b4", linewidth=2.0, label="mean NN ratio")
    ax2.axhline(1 / 3, color="#4575b4", linestyle="--", linewidth=1.2, alpha=0.65)
    ax2.set_ylabel("mean NN ratio", color="#4575b4")
    ax2.tick_params(axis="y", labelcolor="#4575b4")

    fig.suptitle("CelebA-HQ VAE reconstruction memorization on farthest-first 1k")
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    print(out)


if __name__ == "__main__":
    main()
