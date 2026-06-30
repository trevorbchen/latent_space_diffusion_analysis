from __future__ import annotations

import random
from pathlib import Path

import torch
from torchvision import datasets, transforms
from torchvision.utils import save_image


SEED = 42
PER_CLASS = 100
TARGET_LABEL = 2


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
    subset = balanced_indices(dataset, PER_CLASS, SEED)
    chosen = [idx for idx in subset if int(dataset[idx][1]) == TARGET_LABEL]
    images = torch.stack([dataset[idx][0] for idx in chosen], dim=0)

    out = Path("diagnostics/vae_balanced_1k_all_twos_originals.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    save_image((images + 1) / 2, out, nrow=10, padding=2)
    print(out)


if __name__ == "__main__":
    main()
