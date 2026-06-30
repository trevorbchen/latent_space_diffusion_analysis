from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from torchvision.utils import save_image

from train_vae_celeba_standard_tar import TarImageDataset


SEED = 42
N_SELECT = 1000
N_PAIR_ESTIMATE = 250_000
TRAIN_TAR = "/overflow/data/cerberus/celeba_hq_256_partitioned_ws/train/celeba_hq_256-train-0000.tar"
OUT_DIR = Path("diagnostics/celeba_diverse_1k_mean_plus_sd")


def estimate_pairwise_stats(flat: torch.Tensor, n_pairs: int, seed: int) -> dict[str, float]:
    gen = torch.Generator().manual_seed(seed)
    n = flat.shape[0]
    i = torch.randint(0, n, (n_pairs,), generator=gen)
    j = torch.randint(0, n, (n_pairs,), generator=gen)
    same = i == j
    while same.any():
        j[same] = torch.randint(0, n, (int(same.sum()),), generator=gen)
        same = i == j
    d = (flat[i] - flat[j]).pow(2).sum(dim=1).sqrt()
    return {
        "n_pairs": int(n_pairs),
        "mean": float(d.mean()),
        "std": float(d.std(unbiased=True)),
        "min": float(d.min()),
        "max": float(d.max()),
        "median": float(d.median()),
    }


def farthest_first(flat: torch.Tensor, k: int, seed: int) -> tuple[list[int], torch.Tensor]:
    rng = random.Random(seed)
    n = flat.shape[0]
    first = rng.randrange(n)
    selected = [first]
    min_dists = torch.cdist(flat[first : first + 1], flat).squeeze(0)
    min_dists[first] = -1.0

    for _ in range(1, k):
        nxt = int(torch.argmax(min_dists).item())
        selected.append(nxt)
        d = torch.cdist(flat[nxt : nxt + 1], flat).squeeze(0)
        min_dists = torch.minimum(min_dists, d)
        min_dists[selected] = -1.0
    return selected, min_dists


def selected_stats(flat: torch.Tensor, selected: list[int]) -> dict[str, float]:
    sel = flat[selected]
    d = torch.cdist(sel, sel)
    d.fill_diagonal_(float("inf"))
    nn = d.min(dim=1).values
    finite = d[torch.isfinite(d)]
    return {
        "pairwise_mean": float(finite.mean()),
        "pairwise_std": float(finite.std(unbiased=True)),
        "pairwise_min": float(finite.min()),
        "pairwise_max": float(finite.max()),
        "pairwise_median": float(finite.median()),
        "nearest_mean": float(nn.mean()),
        "nearest_std": float(nn.std(unbiased=True)),
        "nearest_min": float(nn.min()),
        "nearest_max": float(nn.max()),
        "nearest_median": float(nn.median()),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset = TarImageDataset(TRAIN_TAR, image_size=32)
    images = torch.stack([dataset[i][0] for i in range(len(dataset))], dim=0)
    flat = images.flatten(1)

    global_stats = estimate_pairwise_stats(flat, N_PAIR_ESTIMATE, SEED)
    threshold = global_stats["mean"] + global_stats["std"]
    selected, _ = farthest_first(flat, N_SELECT, SEED)
    stats = selected_stats(flat, selected)

    save_image((images[selected] + 1) / 2, OUT_DIR / "celeba_diverse_1k_contact_sheet.png", nrow=25, padding=2)
    (OUT_DIR / "subset_indices.json").write_text(json.dumps(selected))
    summary = {
        "seed": SEED,
        "train_tar": TRAIN_TAR,
        "n_train": len(dataset),
        "n_select": N_SELECT,
        "global_pairwise_estimate": global_stats,
        "target_threshold_mean_plus_sd": threshold,
        "selected_stats": stats,
        "all_selected_nearest_above_threshold": stats["nearest_min"] >= threshold,
        "selected_nearest_min_minus_threshold": stats["nearest_min"] - threshold,
        "selected_pairwise_mean_minus_threshold": stats["pairwise_mean"] - threshold,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
