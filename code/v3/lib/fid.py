"""Frechet Inception Distance for real-data diffusion runs.

Standard FID pipeline:
1. Pass real images and generated images through the InceptionV3 backbone
2. Extract pool3 features (2048-d, from the layer just before the classifier)
3. Fit Gaussians to each set of features and compute the Frechet distance

Reference implementation choices that match the canonical Heusel et al. FID:
- Use ImageNet-pretrained Inception V3 (torchvision.models.inception_v3)
- Resize inputs to 299x299 with bilinear interpolation
- Map [-1, 1] inputs (our VAE convention) -> [0, 1] -> ImageNet normalize
- Replicate single-channel images to 3 channels (MNIST)
- 2048-d features taken from the Mixed_7c block via global average pooling

The real-feature side is computed exactly once per run via
`cached_real_features`, then reused at every eval step.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ---------------------------------------------------------------------------
# Inception V3 wrapper
# ---------------------------------------------------------------------------

class InceptionFeatures(nn.Module):
    """Inception V3 backbone returning 2048-d pool3 features per image.

    Lazily loads the torchvision model on first call to forward(). The
    model weights are downloaded once into the torch hub cache.
    """

    def __init__(self):
        super().__init__()
        self._inception: nn.Module | None = None
        self.register_buffer('imnet_mean', _IMAGENET_MEAN)
        self.register_buffer('imnet_std',  _IMAGENET_STD)

    def _ensure_loaded(self, device: torch.device) -> None:
        if self._inception is not None:
            return
        from torchvision.models import inception_v3, Inception_V3_Weights
        net = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1,
                            aux_logits=True)
        net.fc = nn.Identity()           # drop classifier
        net.eval()
        for p in net.parameters():
            p.requires_grad_(False)
        self._inception = net.to(device)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W) in [-1, 1]. Returns (B, 2048) features."""
        self._ensure_loaded(x.device)
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)     # MNIST -> 3ch
        x = (x + 1) / 2                   # [-1, 1] -> [0, 1]
        x = F.interpolate(x, size=(299, 299), mode='bilinear',
                          align_corners=False)
        x = (x - self.imnet_mean) / self.imnet_std
        # torchvision Inception in eval mode returns (logits) only;
        # because we replaced fc with Identity, those "logits" are the
        # 2048-d pool3 features.
        return self._inception(x)


# ---------------------------------------------------------------------------
# Frechet distance
# ---------------------------------------------------------------------------

def frechet_distance(mu_r: np.ndarray,
                     cov_r: np.ndarray,
                     mu_g: np.ndarray,
                     cov_g: np.ndarray) -> float:
    """Standard FID formula. cov_* are 2048x2048 numpy arrays."""
    from scipy import linalg
    diff = mu_r - mu_g
    covmean, _ = linalg.sqrtm(cov_r @ cov_g, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(cov_r + cov_g - 2 * covmean))


# ---------------------------------------------------------------------------
# High-level helpers
# ---------------------------------------------------------------------------

@dataclass
class RealFeatures:
    mean: np.ndarray
    cov: np.ndarray
    n: int


def _iter_batches(images: torch.Tensor, batch_size: int):
    for i in range(0, images.size(0), batch_size):
        yield images[i:i + batch_size]


@torch.no_grad()
def extract_features(extractor: InceptionFeatures,
                     images: torch.Tensor,
                     *,
                     batch_size: int = 64,
                     device: torch.device = torch.device('cpu')) -> np.ndarray:
    """Run Inception over a stack of (C, H, W) images in [-1, 1]."""
    extractor.to(device).eval()
    feats: list[np.ndarray] = []
    for chunk in _iter_batches(images, batch_size):
        feats.append(extractor(chunk.to(device)).cpu().numpy())
    return np.concatenate(feats, axis=0)


def cached_real_features(images: torch.Tensor,
                         *,
                         cache_path: Path,
                         extractor: InceptionFeatures,
                         device: torch.device,
                         batch_size: int = 64) -> RealFeatures:
    """Compute (or load from cache) the real-side feature statistics.

    The cache key is the cache_path itself; callers should incorporate the
    dataset name and split into the path so different runs don't collide.
    """
    cache_path = Path(cache_path)
    if cache_path.exists():
        blob = np.load(cache_path)
        return RealFeatures(mean=blob['mean'], cov=blob['cov'],
                            n=int(blob['n']))
    feats = extract_features(extractor, images,
                             batch_size=batch_size, device=device)
    mean = feats.mean(0)
    cov = np.cov(feats, rowvar=False)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, mean=mean, cov=cov, n=feats.shape[0])
    return RealFeatures(mean=mean, cov=cov, n=feats.shape[0])


def fid_against_real(real: RealFeatures,
                     gen_images: torch.Tensor,
                     *,
                     extractor: InceptionFeatures,
                     device: torch.device,
                     batch_size: int = 64) -> float:
    """Compute FID between cached real stats and a fresh batch of gen images."""
    feats = extract_features(extractor, gen_images,
                             batch_size=batch_size, device=device)
    mu_g = feats.mean(0)
    cov_g = np.cov(feats, rowvar=False)
    return frechet_distance(real.mean, real.cov, mu_g, cov_g)
