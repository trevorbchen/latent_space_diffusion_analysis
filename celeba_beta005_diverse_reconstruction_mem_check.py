from __future__ import annotations

import json
from pathlib import Path

import torch

from lib import metrics
from standard_vae import StandardConvVAE, StandardVAEConfig
from train_vae_celeba_standard_tar import TarImageDataset


N = 1000
D_LATENTS = [10, 15, 20, 25, 30, 35, 40, 45, 50]
TRAIN_TAR = "/overflow/data/cerberus/celeba_hq_256_partitioned_ws/train/celeba_hq_256-train-0000.tar"
SUBSET_PATH = Path("diagnostics/celeba_diverse_1k_mean_plus_sd/subset_indices.json")
OUT_DIR = Path("diagnostics/celeba_beta005_diverse_reconstruction_mem")


def load_standard_vae(path: str) -> StandardConvVAE:
    ckpt = torch.load(path, map_location="cpu")
    cfg = StandardVAEConfig(
        image_channels=ckpt["cfg"]["image_channels"],
        image_size=ckpt["cfg"]["image_size"],
        hidden_dims=tuple(ckpt["cfg"]["hidden_dims"]),
        d_latent=ckpt["cfg"]["d_latent"],
        arch=ckpt["cfg"].get("arch", "standard"),
    )
    model = StandardConvVAE(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset = TarImageDataset(TRAIN_TAR, image_size=32)
    indices = json.loads(SUBSET_PATH.read_text())[:N]

    images = torch.stack([dataset[i][0] for i in indices], dim=0)
    flat = images.flatten(1)
    train_dists = torch.cdist(flat, flat)
    train_dists.fill_diagonal_(float("inf"))

    rows = []
    print("dataset", "celeba_hq_tar")
    print("vae_family", "celeba_standard_beta005")
    print("subset", str(SUBSET_PATH))
    print("n", N)
    print("threshold", 1.0 / 3.0)
    print("d_latent,memorized_count,memorized_fraction,mean_nn_ratio")

    for d_latent in D_LATENTS:
        vae = load_standard_vae(f"vae_checkpoints/celeba_standard_beta005_d{d_latent}/vae.pt")
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
