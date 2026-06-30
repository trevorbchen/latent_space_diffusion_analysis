from __future__ import annotations

from pathlib import Path

import torch
from torchvision import datasets, transforms

from lib import metrics
from lib.data_real import load_vae


N = 500
D_LATENTS = [5, 10, 15, 20, 25, 30, 40]


def mnist_transform():
    return transforms.Compose(
        [
            transforms.Pad(2),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )


def main() -> None:
    device = torch.device("cpu")
    dataset = datasets.MNIST(
        "./data", train=True, download=True, transform=mnist_transform()
    )
    images = torch.stack([dataset[i][0] for i in range(N)], dim=0)
    flat = images.flatten(1)
    train_dists = torch.cdist(flat, flat)
    train_dists.fill_diagonal_(float("inf"))

    print("n,threshold,d_latent,memorized_count,memorized_fraction,mean_nn_ratio")
    for d_latent in D_LATENTS:
        ckpt = Path(f"vae_checkpoints/mnist_d{d_latent}/vae.pt")
        vae = load_vae(str(ckpt), device)
        with torch.no_grad():
            mu, _ = vae.encode(images)
            recon = vae.decode(mu)
        result = metrics.nn_ratio_memorization(
            recon.flatten(1),
            flat,
            train_dists=train_dists,
        )
        count = int(round(result.memorization_fraction * N))
        print(
            f"{N},0.3333333333,{d_latent},"
            f"{count},{result.memorization_fraction:.6f},"
            f"{result.mean_nn_ratio:.6f}"
        )


if __name__ == "__main__":
    main()
