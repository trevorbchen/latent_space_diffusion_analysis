from __future__ import annotations

import json
import random
from pathlib import Path

import torch

from train_vae_celeba_standard_tar import TarImageDataset


SEED = 42
TRAIN_TAR = "/overflow/data/cerberus/celeba_hq_256_partitioned_ws/train/celeba_hq_256-train-0000.tar"
SUMMARY_PATH = Path("diagnostics/celeba_diverse_1k_mean_plus_sd/summary.json")
OUT_DIR = Path("diagnostics/celeba_diverse_1k_mean_plus_sd")
TRIALS = 8


def greedy_threshold(flat: torch.Tensor, threshold: float, seed: int) -> list[int]:
    order = list(range(flat.shape[0]))
    random.Random(seed).shuffle(order)
    selected: list[int] = []
    selected_flat = None
    for idx in order:
        x = flat[idx : idx + 1]
        if selected_flat is None:
            selected.append(idx)
            selected_flat = x.clone()
            continue
        d = torch.cdist(x, selected_flat).squeeze(0)
        if bool((d >= threshold).all()):
            selected.append(idx)
            selected_flat = torch.cat([selected_flat, x], dim=0)
    return selected


def main() -> None:
    summary = json.loads(SUMMARY_PATH.read_text())
    threshold = float(summary["target_threshold_mean_plus_sd"])
    dataset = TarImageDataset(TRAIN_TAR, image_size=32)
    images = torch.stack([dataset[i][0] for i in range(len(dataset))], dim=0)
    flat = images.flatten(1)

    results = []
    best: list[int] = []
    for t in range(TRIALS):
        selected = greedy_threshold(flat, threshold, SEED + t)
        results.append({"seed": SEED + t, "count": len(selected)})
        if len(selected) > len(best):
            best = selected

    out = {
        "threshold": threshold,
        "trials": results,
        "best_count": len(best),
        "best_indices": best,
    }
    (OUT_DIR / "strict_threshold_subset.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "best_indices"}, indent=2))


if __name__ == "__main__":
    main()
