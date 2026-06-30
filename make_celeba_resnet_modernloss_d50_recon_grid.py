from __future__ import annotations

import torch
from torchvision.utils import save_image

from standard_vae import StandardConvVAE, StandardVAEConfig
from train_vae_celeba_standard_tar import TarImageDataset


TRAIN_TAR = "/overflow/data/cerberus/celeba_hq_256_partitioned_ws/train/celeba_hq_256-train-0000.tar"
CKPT = "vae_checkpoints/celeba_resnet_modernloss_d50/vae.pt"
OUT = "diagnostics/celeba_resnet_modernloss_d50_recon_grid.png"
INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


def load_vae(path: str) -> StandardConvVAE:
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
    images = torch.stack([dataset[i][0] for i in INDICES], dim=0)
    vae = load_vae(CKPT)
    with torch.no_grad():
        mu, _ = vae.encode(images)
        recon = vae.decode(mu)
    grid = torch.stack([images, recon], dim=1).flatten(0, 1)
    save_image((grid * 0.5 + 0.5).clamp(0, 1), OUT, nrow=2, padding=2)
    print(OUT)


if __name__ == "__main__":
    main()
