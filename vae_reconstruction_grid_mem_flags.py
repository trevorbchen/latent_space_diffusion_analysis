from __future__ import annotations

import torch
from torchvision import datasets, transforms

from lib.data_real import load_vae


D_LATENTS = [5, 10, 20, 40]
N_EXAMPLES = 8
THRESHOLD = 1.0 / 3.0


def main() -> None:
    transform = transforms.Compose(
        [
            transforms.Pad(2),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
    images = torch.stack([dataset[i][0] for i in range(N_EXAMPLES)], dim=0)
    flat = images.flatten(1)

    train_dists = torch.cdist(flat, flat)
    train_dists.fill_diagonal_(float("inf"))

    print("d_latent,count,flags,ratios,nearest_train_idx")
    for d_latent in D_LATENTS:
        vae = load_vae(f"vae_checkpoints/mnist_d{d_latent}/vae.pt", torch.device("cpu"))
        with torch.no_grad():
            mu, _ = vae.encode(images)
            recon = vae.decode(mu)

        dists = torch.cdist(recon.flatten(1), flat)
        nn1, idx = dists.min(dim=1)
        nn2 = train_dists[idx].min(dim=1).values
        ratio = nn1 / (nn2 + 1e-10)
        flags = ratio < THRESHOLD

        print(
            f"{d_latent},"
            f"{int(flags.sum().item())},"
            f"{flags.int().tolist()},"
            f"{[round(float(x), 3) for x in ratio]},"
            f"{idx.tolist()}"
        )


if __name__ == "__main__":
    main()
