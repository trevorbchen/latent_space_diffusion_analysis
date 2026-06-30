from __future__ import annotations

import random
from pathlib import Path

import torch
from torchvision import datasets, transforms
from torchvision.utils import save_image

from lib.data_real import load_vae


SEED = 42
PER_CLASS = 100
EXAMPLE_POS = 1
D_LATENTS = [5, 10]
TOPK = 8
THRESHOLD = 1.0 / 3.0


def balanced_indices(dataset, per_class: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    by_class = {k: [] for k in range(10)}
    for idx in range(len(dataset)):
        by_class[int(dataset[idx][1])].append(idx)

    chosen: list[int] = []
    for label in range(10):
        pool = by_class[label][:]
        rng.shuffle(pool)
        chosen.extend(pool[:per_class])

    rng.shuffle(chosen)
    return chosen


def main() -> None:
    transform = transforms.Compose(
        [
            transforms.Pad(2),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
    subset_idxs = balanced_indices(dataset, PER_CLASS, SEED)
    images = torch.stack([dataset[i][0] for i in subset_idxs], dim=0)
    labels = [int(dataset[i][1]) for i in subset_idxs]
    flat = images.flatten(1)

    train_dists = torch.cdist(flat, flat)
    train_dists.fill_diagonal_(float("inf"))

    print("subset_seed", SEED)
    print(
        "example_pos,global_idx,label",
        EXAMPLE_POS,
        subset_idxs[EXAMPLE_POS],
        labels[EXAMPLE_POS],
    )

    rows = []
    blank = torch.full((TOPK, *images.shape[1:]), -1.0)
    rows.append(torch.cat([images[EXAMPLE_POS : EXAMPLE_POS + 1], blank], dim=0))

    for d_latent in D_LATENTS:
        vae = load_vae(f"vae_checkpoints/mnist_d{d_latent}/vae.pt", torch.device("cpu"))
        with torch.no_grad():
            mu, _ = vae.encode(images[EXAMPLE_POS : EXAMPLE_POS + 1])
            recon = vae.decode(mu).clamp(-1, 1)

        dists = torch.cdist(recon.flatten(1), flat).squeeze(0)
        vals, idxs = torch.topk(dists, TOPK, largest=False)
        nn1_idx = int(idxs[0])
        nn1_dist = float(vals[0])
        nn2_idx = int(train_dists[nn1_idx].argmin())
        nn2_dist = float(train_dists[nn1_idx, nn2_idx])
        ratio = nn1_dist / (nn2_dist + 1e-10)

        print(f"\nd_latent={d_latent}")
        print(
            "nn1_subset_idx,nn1_global_idx,label,dist,"
            "nn2_subset_idx,nn2_global_idx,label,nn2_dist,ratio,memorized",
            nn1_idx,
            subset_idxs[nn1_idx],
            labels[nn1_idx],
            f"{nn1_dist:.6f}",
            nn2_idx,
            subset_idxs[nn2_idx],
            labels[nn2_idx],
            f"{nn2_dist:.6f}",
            f"{ratio:.6f}",
            bool(ratio < THRESHOLD),
        )
        print("top_neighbors subset_idx,global_idx,label,dist")
        for idx, val in zip(idxs.tolist(), vals.tolist()):
            print(idx, subset_idxs[idx], labels[idx], f"{val:.6f}")

        rows.append(torch.cat([recon, images[idxs]], dim=0))

    out = Path("diagnostics/vae_balanced_1k_second_example_neighbors.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    save_image((torch.cat(rows, dim=0) + 1) / 2, out, nrow=TOPK + 1, padding=2)
    print("\nimage_grid", out)


if __name__ == "__main__":
    main()
