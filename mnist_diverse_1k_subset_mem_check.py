from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from torchvision import datasets, transforms
from torchvision.utils import save_image

from lib import metrics
from lib.data_real import load_vae


SEED = 42
PER_CLASS = 100
D_LATENTS = [5, 10, 15, 20, 25, 30, 40]
OUT_DIR = Path("diagnostics/diverse_balanced_1k")


def mnist_transform():
    return transforms.Compose(
        [
            transforms.Pad(2),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )


def farthest_point_indices(x: torch.Tensor, k: int, seed: int) -> list[int]:
    """Greedy max-min selection within one class, using pixel L2."""
    rng = random.Random(seed)
    n = x.shape[0]
    first = rng.randrange(n)
    selected = [first]

    min_dists = torch.cdist(x[first : first + 1], x).squeeze(0)
    min_dists[first] = -1.0

    for _ in range(1, k):
        nxt = int(torch.argmax(min_dists).item())
        selected.append(nxt)
        d = torch.cdist(x[nxt : nxt + 1], x).squeeze(0)
        min_dists = torch.minimum(min_dists, d)
        min_dists[selected] = -1.0
    return selected


def diverse_balanced_subset(dataset, images: torch.Tensor) -> list[int]:
    chosen: list[int] = []
    for label in range(10):
        class_idxs = [idx for idx in range(len(dataset)) if int(dataset[idx][1]) == label]
        class_flat = images[class_idxs].flatten(1)
        local = farthest_point_indices(class_flat, PER_CLASS, SEED + label)
        chosen.extend(class_idxs[i] for i in local)
    random.Random(SEED).shuffle(chosen)
    return chosen


def nn_spacing_summary(flat: torch.Tensor) -> dict[str, float]:
    dists = torch.cdist(flat, flat)
    dists.fill_diagonal_(float("inf"))
    nn = dists.min(dim=1).values
    return {
        "mean_nn_dist": float(nn.mean()),
        "median_nn_dist": float(nn.median()),
        "min_nn_dist": float(nn.min()),
        "max_nn_dist": float(nn.max()),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset = datasets.MNIST("./data", train=True, download=True, transform=mnist_transform())
    all_images = torch.stack([dataset[i][0] for i in range(len(dataset))], dim=0)

    subset_idxs = diverse_balanced_subset(dataset, all_images)
    subset_images = torch.stack([dataset[i][0] for i in subset_idxs], dim=0)
    labels = torch.tensor([int(dataset[i][1]) for i in subset_idxs])
    flat = subset_images.flatten(1)

    train_dists = torch.cdist(flat, flat)
    train_dists.fill_diagonal_(float("inf"))

    save_image((subset_images + 1) / 2, OUT_DIR / "diverse_1k_contact_sheet.png", nrow=25, padding=2)
    for label in range(10):
        class_imgs = subset_images[labels == label]
        save_image((class_imgs + 1) / 2, OUT_DIR / f"digit_{label}_diverse_100.png", nrow=10, padding=2)

    rows = []
    print("subset,seed,n,class_counts")
    print("diverse_l2", SEED, len(subset_idxs), torch.bincount(labels, minlength=10).tolist())
    print("spacing", json.dumps(nn_spacing_summary(flat), sort_keys=True))
    print("d_latent,memorized_count,memorized_fraction,mean_nn_ratio")
    for d_latent in D_LATENTS:
        vae = load_vae(f"vae_checkpoints/mnist_d{d_latent}/vae.pt", torch.device("cpu"))
        with torch.no_grad():
            mu, _ = vae.encode(subset_images)
            recon = vae.decode(mu)
        result = metrics.nn_ratio_memorization(
            recon.flatten(1),
            flat,
            train_dists=train_dists,
        )
        count = int(round(result.memorization_fraction * len(subset_idxs)))
        row = {
            "d_latent": d_latent,
            "memorized_count": count,
            "memorized_fraction": result.memorization_fraction,
            "mean_nn_ratio": result.mean_nn_ratio,
        }
        rows.append(row)
        print(
            f"{d_latent},"
            f"{count},"
            f"{result.memorization_fraction:.6f},"
            f"{result.mean_nn_ratio:.6f}"
        )

    (OUT_DIR / "subset_indices.json").write_text(json.dumps(subset_idxs))
    (OUT_DIR / "reconstruction_mem_summary.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
