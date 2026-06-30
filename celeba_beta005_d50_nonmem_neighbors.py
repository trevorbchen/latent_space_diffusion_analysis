from __future__ import annotations

import json
from pathlib import Path

import torch
from torchvision.utils import save_image

from standard_vae import StandardConvVAE, StandardVAEConfig
from train_vae_celeba_standard_tar import TarImageDataset


TRAIN_TAR = "/overflow/data/cerberus/celeba_hq_256_partitioned_ws/train/celeba_hq_256-train-0000.tar"
SUBSET_PATH = Path("diagnostics/celeba_beta005_reconstruction_mem/subset_indices.json")
CKPT = "vae_checkpoints/celeba_standard_beta005_d50/vae.pt"
OUT = Path("diagnostics/celeba_beta005_reconstruction_mem/d50_nonmem_neighbors.png")
THRESHOLD = 1.0 / 3.0
N_SHOW = 6


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
    subset_indices = json.loads(SUBSET_PATH.read_text())
    images = torch.stack([dataset[i][0] for i in subset_indices], dim=0)
    flat = images.flatten(1)
    train_dists = torch.cdist(flat, flat)
    train_dists.fill_diagonal_(float("inf"))

    model = load_standard_vae(CKPT)
    with torch.no_grad():
        mu, _ = model.encode(images)
        recon = model.decode(mu).clamp(-1, 1)

    dists = torch.cdist(recon.flatten(1), flat)
    nn1_dist, nn1_idx = dists.min(dim=1)
    nn2_dist, nn2_idx = train_dists[nn1_idx].min(dim=1)
    ratio = nn1_dist / (nn2_dist + 1e-10)
    nonmem = torch.where(ratio >= THRESHOLD)[0]

    # Show near-threshold non-mem examples first: visually close, but not enough.
    order = nonmem[torch.argsort(ratio[nonmem])[:N_SHOW]]
    rows = []
    print("columns: source_original, reconstruction, nearest_train, nearest_other_of_nearest")
    for pos_t in order:
        pos = int(pos_t)
        row = torch.stack(
            [
                images[pos],
                recon[pos],
                images[int(nn1_idx[pos])],
                images[int(nn2_idx[pos])],
            ],
            dim=0,
        )
        rows.append(row)
        print(
            "subset_pos", pos,
            "global_idx", subset_indices[pos],
            "nn1_subset_pos", int(nn1_idx[pos]),
            "nn1_global_idx", subset_indices[int(nn1_idx[pos])],
            "nn1_dist", f"{float(nn1_dist[pos]):.6f}",
            "nn2_subset_pos", int(nn2_idx[pos]),
            "nn2_global_idx", subset_indices[int(nn2_idx[pos])],
            "nn2_dist", f"{float(nn2_dist[pos]):.6f}",
            "ratio", f"{float(ratio[pos]):.6f}",
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    save_image((torch.cat(rows, dim=0) + 1) / 2, OUT, nrow=4, padding=4)
    print(OUT)


if __name__ == "__main__":
    main()
