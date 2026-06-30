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
OUT_DIR = Path("diagnostics/celeba_random_remove_nearest_1k")


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


def random_keep_remove_nearest(flat: torch.Tensor, k: int, seed: int) -> tuple[list[int], list[dict[str, float]]]:
    rng = random.Random(seed)
    active = list(range(flat.shape[0]))
    selected: list[int] = []
    removed_pairs: list[dict[str, float]] = []

    while len(selected) < k and len(active) > 1:
        pos = rng.randrange(len(active))
        idx = active.pop(pos)
        selected.append(idx)

        x = flat[idx : idx + 1]
        active_flat = flat[active]
        d = torch.cdist(x, active_flat).squeeze(0)
        nearest_pos = int(torch.argmin(d).item())
        nearest_idx = active.pop(nearest_pos)
        removed_pairs.append(
            {
                "kept_index": int(idx),
                "removed_nearest_index": int(nearest_idx),
                "distance": float(d[nearest_pos]),
            }
        )

    return selected, removed_pairs


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


def removal_stats(removed_pairs: list[dict[str, float]]) -> dict[str, float]:
    d = torch.tensor([x["distance"] for x in removed_pairs])
    return {
        "removed_nearest_mean": float(d.mean()),
        "removed_nearest_std": float(d.std(unbiased=True)),
        "removed_nearest_min": float(d.min()),
        "removed_nearest_max": float(d.max()),
        "removed_nearest_median": float(d.median()),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset = TarImageDataset(TRAIN_TAR, image_size=32)
    images = torch.stack([dataset[i][0] for i in range(len(dataset))], dim=0)
    flat = images.flatten(1)

    global_stats = estimate_pairwise_stats(flat, N_PAIR_ESTIMATE, SEED)
    threshold = global_stats["mean"] + global_stats["std"]
    selected, removed_pairs = random_keep_remove_nearest(flat, N_SELECT, SEED)
    stats = selected_stats(flat, selected)

    save_image((images[selected] + 1) / 2, OUT_DIR / "celeba_random_remove_nearest_1k_contact_sheet.png", nrow=25, padding=2)
    (OUT_DIR / "subset_indices.json").write_text(json.dumps(selected))
    (OUT_DIR / "removed_nearest_pairs.json").write_text(json.dumps(removed_pairs, indent=2))

    summary = {
        "seed": SEED,
        "train_tar": TRAIN_TAR,
        "n_train": len(dataset),
        "n_select": len(selected),
        "method": "Repeatedly sample one random active image, keep it, then remove its nearest remaining neighbor.",
        "global_pairwise_estimate": global_stats,
        "target_threshold_mean_plus_sd": threshold,
        "selected_stats": stats,
        "removed_neighbor_stats": removal_stats(removed_pairs),
        "all_selected_nearest_above_threshold": stats["nearest_min"] >= threshold,
        "selected_nearest_min_minus_threshold": stats["nearest_min"] - threshold,
        "selected_pairwise_mean_minus_threshold": stats["pairwise_mean"] - threshold,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
