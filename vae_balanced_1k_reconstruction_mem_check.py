from __future__ import annotations

import random

import torch
from torchvision import datasets, transforms

from lib import metrics
from lib.data_real import load_vae


SEED = 42
PER_CLASS = 100
D_LATENTS = [5, 10, 15, 20, 25, 30, 40]


def mnist_transform():
    return transforms.Compose(
        [
            transforms.Pad(2),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )


def balanced_indices(dataset, per_class: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    by_class: dict[int, list[int]] = {k: [] for k in range(10)}
    for idx in range(len(dataset)):
        label = int(dataset[idx][1])
        by_class[label].append(idx)

    chosen: list[int] = []
    for label in range(10):
        pool = by_class[label][:]
        rng.shuffle(pool)
        chosen.extend(pool[:per_class])

    rng.shuffle(chosen)
    return chosen


def main() -> None:
    device = torch.device("cpu")
    dataset = datasets.MNIST(
        "./data", train=True, download=True, transform=mnist_transform()
    )
    idxs = balanced_indices(dataset, PER_CLASS, SEED)
    images = torch.stack([dataset[i][0] for i in idxs], dim=0)
    labels = torch.tensor([int(dataset[i][1]) for i in idxs])
    flat = images.flatten(1)
    train_dists = torch.cdist(flat, flat)
    train_dists.fill_diagonal_(float("inf"))

    print("subset_seed", SEED)
    print("n", len(idxs))
    print("class_counts", torch.bincount(labels, minlength=10).tolist())
    print("threshold", 1.0 / 3.0)
    print("d_latent,memorized_count,memorized_fraction,mean_nn_ratio")

    for d_latent in D_LATENTS:
        vae = load_vae(f"vae_checkpoints/mnist_d{d_latent}/vae.pt", device)
        with torch.no_grad():
            mu, _ = vae.encode(images)
            recon = vae.decode(mu)
        result = metrics.nn_ratio_memorization(
            recon.flatten(1),
            flat,
            train_dists=train_dists,
        )
        count = int(round(result.memorization_fraction * len(idxs)))
        print(
            f"{d_latent},"
            f"{count},"
            f"{result.memorization_fraction:.6f},"
            f"{result.mean_nn_ratio:.6f}"
        )


if __name__ == "__main__":
    main()
