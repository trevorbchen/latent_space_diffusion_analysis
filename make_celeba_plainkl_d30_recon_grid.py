from __future__ import annotations

import json
from pathlib import Path

import torch
from torchvision.utils import save_image

from standard_vae import StandardConvVAE, StandardVAEConfig
from train_vae_celeba_standard_tar import TarImageDataset


TRAIN_TAR = "/overflow/data/cerberus/celeba_hq_256_partitioned_ws/train/celeba_hq_256-train-0000.tar"
CKPT = "vae_checkpoints/celeba_standard_plainkl_d30/vae.pt"
OUT = Path("diagnostics/celeba_standard_plainkl_d30_recon_grid.png")
N = 10


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
    dataset = TarImageDataset(TRAIN_TAR, image_size=32)
    if Path("diagnostics/celeba_reconstruction_mem/subset_indices.json").exists():
        indices = json.loads(Path("diagnostics/celeba_reconstruction_mem/subset_indices.json").read_text())[:N]
    else:
        indices = list(range(N))
    images = torch.stack([dataset[i][0] for i in indices], dim=0)
    model = load_standard_vae(CKPT)
    with torch.no_grad():
        mu, _ = model.encode(images)
        recon = model.decode(mu).clamp(-1, 1)
    grid = torch.cat([images, recon], dim=0)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    save_image((grid + 1) / 2, OUT, nrow=N, padding=3)
    print("indices", indices)
    print(OUT)


if __name__ == "__main__":
    main()
