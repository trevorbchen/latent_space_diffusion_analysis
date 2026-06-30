from __future__ import annotations

import json
from pathlib import Path

import torch
from torchvision.utils import save_image

from lib.data_real import load_vae
from train_vae_celeba_tar import TarImageDataset


TRAIN_TAR = "/overflow/data/cerberus/celeba_hq_256_partitioned_ws/train/celeba_hq_256-train-0000.tar"
SUBSET_PATH = Path("diagnostics/celeba_reconstruction_mem/subset_indices.json")
D_LATENT = 30
THRESHOLD = 1.0 / 3.0


def main() -> None:
    dataset = TarImageDataset(TRAIN_TAR, image_size=32)
    subset_indices = json.loads(SUBSET_PATH.read_text())
    images = torch.stack([dataset[i][0] for i in subset_indices], dim=0)
    flat = images.flatten(1)

    train_dists = torch.cdist(flat, flat)
    train_dists.fill_diagonal_(float("inf"))

    vae = load_vae(f"vae_checkpoints/celeba_d{D_LATENT}/vae.pt", torch.device("cpu"))
    with torch.no_grad():
        mu, _ = vae.encode(images)
        recon = vae.decode(mu).clamp(-1, 1)

    dists = torch.cdist(recon.flatten(1), flat)
    nn1_dist, nn1_idx = dists.min(dim=1)
    nn2_dist, nn2_idx = train_dists[nn1_idx].min(dim=1)
    ratio = nn1_dist / (nn2_dist + 1e-10)
    flags = ratio < THRESHOLD

    mem_pos = int(torch.where(flags)[0][0].item())
    nonmem_pos = int(torch.where(~flags)[0][0].item())

    rows = []
    for pos in [mem_pos, nonmem_pos]:
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

    grid = torch.cat(rows, dim=0)
    out = Path("diagnostics/celeba_reconstruction_mem/celeba_d30_mem_vs_nonmem_examples.png")
    save_image((grid + 1) / 2, out, nrow=4, padding=4)

    print("columns: source_original, reconstruction, nearest_train, nearest_other_of_nearest")
    for name, pos in [("memorized", mem_pos), ("not_memorized", nonmem_pos)]:
        print(
            name,
            "subset_pos", pos,
            "global_idx", subset_indices[pos],
            "nn1_subset_pos", int(nn1_idx[pos]),
            "nn1_global_idx", subset_indices[int(nn1_idx[pos])],
            "nn1_dist", f"{float(nn1_dist[pos]):.6f}",
            "nn2_subset_pos", int(nn2_idx[pos]),
            "nn2_global_idx", subset_indices[int(nn2_idx[pos])],
            "nn2_dist", f"{float(nn2_dist[pos]):.6f}",
            "ratio", f"{float(ratio[pos]):.6f}",
            "memorized", bool(flags[pos]),
        )
    print(out)


if __name__ == "__main__":
    main()
