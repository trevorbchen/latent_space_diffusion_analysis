from __future__ import annotations

from pathlib import Path

import torch
from torchvision import datasets, transforms
from torchvision.utils import save_image

from lib.data_real import load_vae


N_TRAIN = 500
EXAMPLE_IDX = 1  # second image in the reconstruction grid
D_LATENTS = [5, 10]
TOPK = 8
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
    images = torch.stack([dataset[i][0] for i in range(N_TRAIN)], dim=0)
    labels = [int(dataset[i][1]) for i in range(N_TRAIN)]
    flat = images.flatten(1)

    train_dists = torch.cdist(flat, flat)
    train_dists.fill_diagonal_(float("inf"))

    print("example_idx,label", EXAMPLE_IDX, labels[EXAMPLE_IDX])
    rows = [images[EXAMPLE_IDX : EXAMPLE_IDX + 1]]

    for d_latent in D_LATENTS:
        vae = load_vae(f"vae_checkpoints/mnist_d{d_latent}/vae.pt", torch.device("cpu"))
        with torch.no_grad():
            mu, _ = vae.encode(images[EXAMPLE_IDX : EXAMPLE_IDX + 1])
            recon = vae.decode(mu).clamp(-1, 1)

        dists = torch.cdist(recon.flatten(1), flat).squeeze(0)
        vals, idxs = torch.topk(dists, TOPK, largest=False)
        nn1_idx = int(idxs[0])
        nn1_dist = float(vals[0])
        nn2_dist = float(train_dists[nn1_idx].min())
        ratio = nn1_dist / (nn2_dist + 1e-10)

        other_idx = int(train_dists[nn1_idx].argmin())
        print(f"\nd_latent={d_latent}")
        print(
            "nn1_idx,label,dist,nn2_idx,label,nn2_dist,ratio,memorized",
            nn1_idx,
            labels[nn1_idx],
            f"{nn1_dist:.6f}",
            other_idx,
            labels[other_idx],
            f"{nn2_dist:.6f}",
            f"{ratio:.6f}",
            bool(ratio < THRESHOLD),
        )
        print("top_neighbors idx,label,dist")
        for idx, val in zip(idxs.tolist(), vals.tolist()):
            print(idx, labels[idx], f"{val:.6f}")

        row = torch.cat([recon, images[idxs]], dim=0)
        rows.append(row)

    width = TOPK + 1
    blank = torch.full((width - 1, *images.shape[1:]), -1.0)
    grid = torch.cat([torch.cat([images[EXAMPLE_IDX : EXAMPLE_IDX + 1], blank], dim=0), *rows[1:]], dim=0)
    out = Path("diagnostics/vae_second_example_neighbors.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    save_image((grid + 1) / 2, out, nrow=width, padding=2)
    print("\nimage_grid", out)


if __name__ == "__main__":
    main()
