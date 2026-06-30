from __future__ import annotations

import json
from pathlib import Path

import torch
from torchvision import datasets, transforms

from lib import metrics
from lib.data_real import load_vae


D_LATENTS = [5, 10, 15, 20, 25, 30, 40]
SUBSET_PATH = Path("diagnostics/diverse_balanced_1k/subset_indices.json")


def main() -> None:
    transform = transforms.Compose(
        [
            transforms.Pad(2),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
    subset_idxs = json.loads(SUBSET_PATH.read_text())

    keep = [idx for idx in subset_idxs if int(dataset[idx][1]) not in (0, 1)]
    images = torch.stack([dataset[idx][0] for idx in keep], dim=0)
    labels = torch.tensor([int(dataset[idx][1]) for idx in keep])
    flat = images.flatten(1)
    train_dists = torch.cdist(flat, flat)
    train_dists.fill_diagonal_(float("inf"))

    print("base_subset", str(SUBSET_PATH))
    print("eval_filter", "labels != 0,1")
    print("n_eval", len(keep))
    print("class_counts", torch.bincount(labels, minlength=10).tolist())
    print("d_latent,memorized_count,memorized_fraction,mean_nn_ratio")

    for d_latent in D_LATENTS:
        vae = load_vae(f"vae_checkpoints/mnist_d{d_latent}/vae.pt", torch.device("cpu"))
        with torch.no_grad():
            mu, _ = vae.encode(images)
            recon = vae.decode(mu)
        result = metrics.nn_ratio_memorization(
            recon.flatten(1),
            flat,
            train_dists=train_dists,
        )
        count = int(round(result.memorization_fraction * len(keep)))
        print(
            f"{d_latent},"
            f"{count},"
            f"{result.memorization_fraction:.6f},"
            f"{result.mean_nn_ratio:.6f}"
        )


if __name__ == "__main__":
    main()
