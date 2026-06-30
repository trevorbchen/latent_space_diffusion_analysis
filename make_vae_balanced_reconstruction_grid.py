from __future__ import annotations

import random
from pathlib import Path

import torch
from torchvision import datasets, transforms
from torchvision.utils import save_image

from lib.data_real import load_vae


SEED = 42
PER_CLASS = 100
D_LATENTS = [5, 10, 20, 40]
N_EXAMPLES = 8


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
    idxs = balanced_indices(dataset, PER_CLASS, SEED)
    originals = torch.stack([dataset[i][0] for i in idxs[:N_EXAMPLES]], dim=0)
    labels = [int(dataset[i][1]) for i in idxs[:N_EXAMPLES]]

    rows = [originals]
    for d_latent in D_LATENTS:
        vae = load_vae(f"vae_checkpoints/mnist_d{d_latent}/vae.pt", torch.device("cpu"))
        with torch.no_grad():
            mu, _ = vae.encode(originals)
            recon = vae.decode(mu)
        rows.append(recon.clamp(-1, 1))

    grid = torch.cat(rows, dim=0)
    out = Path("diagnostics/vae_balanced_1k_reconstruction_pairs_mnist.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    save_image((grid + 1) / 2, out, nrow=N_EXAMPLES, padding=2)
    print("labels", labels)
    print(out)


if __name__ == "__main__":
    main()
