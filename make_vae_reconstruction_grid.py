from __future__ import annotations

from pathlib import Path

import torch
from torchvision import datasets, transforms
from torchvision.utils import save_image

from lib.data_real import load_vae


D_LATENTS = [5, 10, 20, 40]
N_EXAMPLES = 8


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
    originals = torch.stack([dataset[i][0] for i in range(N_EXAMPLES)], dim=0)

    rows = [originals]
    for d_latent in D_LATENTS:
        vae = load_vae(f"vae_checkpoints/mnist_d{d_latent}/vae.pt", device)
        with torch.no_grad():
            mu, _ = vae.encode(originals)
            recon = vae.decode(mu)
        rows.append(recon.clamp(-1, 1))

    grid = torch.cat(rows, dim=0)
    out = Path("diagnostics/vae_reconstruction_pairs_mnist.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    save_image((grid + 1) / 2, out, nrow=N_EXAMPLES, padding=2)
    print(out)


if __name__ == "__main__":
    main()
