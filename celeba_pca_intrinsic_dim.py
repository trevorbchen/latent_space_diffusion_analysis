from __future__ import annotations

import json
from pathlib import Path

import torch

from train_vae_celeba_standard_tar import TarImageDataset


TRAIN_TAR = "/overflow/data/cerberus/celeba_hq_256_partitioned_ws/train/celeba_hq_256-train-0000.tar"
OUT_DIR = Path("diagnostics/celeba_pca_intrinsic_dim")


def threshold_rank(cum: torch.Tensor, threshold: float) -> int:
    return int(torch.searchsorted(cum, torch.tensor(threshold)).item() + 1)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset = TarImageDataset(TRAIN_TAR, image_size=32)
    images = torch.stack([dataset[i][0] for i in range(len(dataset))], dim=0)
    x = images.flatten(1).float()
    x = x - x.mean(dim=0, keepdim=True)

    cov = x.T @ x / (x.shape[0] - 1)
    evals = torch.linalg.eigvalsh(cov).flip(0).clamp_min(0)
    frac = evals / evals.sum()
    cum = torch.cumsum(frac, dim=0)

    participation = float(evals.sum().pow(2) / evals.pow(2).sum())
    entropy_dim = float(torch.exp(-(frac[frac > 0] * torch.log(frac[frac > 0])).sum()))

    thresholds = {str(t): threshold_rank(cum, t) for t in [0.5, 0.8, 0.9, 0.95, 0.99]}
    first_dims = {str(k): float(cum[k - 1]) for k in [5, 10, 20, 30, 40, 50, 75, 100, 150, 200, 300, 500]}
    out = {
        "dataset": TRAIN_TAR,
        "n_images": x.shape[0],
        "ambient_dim": x.shape[1],
        "variance_threshold_dims": thresholds,
        "variance_explained_by_dim": first_dims,
        "participation_ratio_dim": participation,
        "entropy_effective_dim": entropy_dim,
        "top_eigenvalues": [float(v) for v in evals[:50]],
    }
    (OUT_DIR / "pca_intrinsic_dim_summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "top_eigenvalues"}, indent=2))


if __name__ == "__main__":
    main()
