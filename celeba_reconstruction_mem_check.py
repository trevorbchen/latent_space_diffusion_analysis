from __future__ import annotations

import json
import random
from pathlib import Path

import torch

from lib import metrics
from lib.data_real import load_vae
from train_vae_celeba_tar import TarImageDataset


SEED = 42
N = 1000
D_LATENTS = [10, 15, 20, 25, 30, 35, 40, 45, 50]
TRAIN_TAR = "/overflow/data/cerberus/celeba_hq_256_partitioned_ws/train/celeba_hq_256-train-0000.tar"
OUT_DIR = Path("diagnostics/celeba_reconstruction_mem")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset = TarImageDataset(TRAIN_TAR, image_size=32)
    rng = random.Random(SEED)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    indices = indices[:N]

    images = torch.stack([dataset[i][0] for i in indices], dim=0)
    flat = images.flatten(1)
    train_dists = torch.cdist(flat, flat)
    train_dists.fill_diagonal_(float("inf"))

    rows = []
    print("dataset", "celeba_hq_tar")
    print("train_tar", TRAIN_TAR)
    print("subset_seed", SEED)
    print("n", N)
    print("threshold", 1.0 / 3.0)
    print("d_latent,memorized_count,memorized_fraction,mean_nn_ratio")

    for d_latent in D_LATENTS:
        vae = load_vae(f"vae_checkpoints/celeba_d{d_latent}/vae.pt", torch.device("cpu"))
        with torch.no_grad():
            mu, _ = vae.encode(images)
            recon = vae.decode(mu)
        result = metrics.nn_ratio_memorization(
            recon.flatten(1),
            flat,
            train_dists=train_dists,
        )
        count = int(round(result.memorization_fraction * N))
        row = {
            "d_latent": d_latent,
            "memorized_count": count,
            "memorized_fraction": result.memorization_fraction,
            "mean_nn_ratio": result.mean_nn_ratio,
        }
        rows.append(row)
        print(
            f"{d_latent},"
            f"{count},"
            f"{result.memorization_fraction:.6f},"
            f"{result.mean_nn_ratio:.6f}"
        )

    (OUT_DIR / "subset_indices.json").write_text(json.dumps(indices))
    (OUT_DIR / "reconstruction_mem_summary.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
