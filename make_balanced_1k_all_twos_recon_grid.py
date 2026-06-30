from __future__ import annotations

import random
from pathlib import Path

import torch
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image

from lib.data_real import load_vae


SEED = 42
PER_CLASS = 100
TARGET_LABEL = 2
D_LATENTS = [5, 10, 15, 20, 25, 30, 40]
NROW = 10


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
    two_positions = [
        pos for pos, idx in enumerate(subset_idxs) if int(dataset[idx][1]) == TARGET_LABEL
    ]
    images = torch.stack([dataset[subset_idxs[pos]][0] for pos in two_positions], dim=0)

    blocks = [images]
    for d_latent in D_LATENTS:
        vae = load_vae(f"vae_checkpoints/mnist_d{d_latent}/vae.pt", torch.device("cpu"))
        with torch.no_grad():
            mu, _ = vae.encode(images)
            recon = vae.decode(mu).clamp(-1, 1)
        blocks.append(recon)

    grids = [make_grid((block + 1) / 2, nrow=NROW, padding=2) for block in blocks]
    contact = torch.cat(grids, dim=1)

    out = Path("diagnostics/vae_balanced_1k_all_twos_recon_grid.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    save_image(contact, out)

    print("subset_seed", SEED)
    print("target_label", TARGET_LABEL)
    print("count", len(two_positions))
    print("blocks", ["original"] + [f"d{d}" for d in D_LATENTS])
    print(out)


if __name__ == "__main__":
    main()
