"""Evaluation metrics: score error, NN-ratio memorization, MMD, FID.

Score error and MMD are usable on synthetic data only (we have a true
score and a known population sampler). Memorization fraction works on any
generative process; for real data we ask the caller to pass already-decoded
pixel-space samples.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Score-matching loss + analytic score error
# ---------------------------------------------------------------------------

def score_matching_loss(model,
                        x: torch.Tensor,
                        t,
                        *,
                        per_dim: bool = True) -> float:
    """Mean over batch of ||sqrt(delta_t) s + eps||^2, optionally / d_latent."""
    from .diffusion import forward_noise, score_loss
    x_t, noise, _, sqrt_dt = forward_noise(x, t)
    pred = model(x_t, t if torch.is_tensor(t) else
                  torch.full((x.shape[0],), float(t), device=x.device))
    loss = score_loss(pred, noise, sqrt_dt)
    return loss.item() / (x.shape[-1] if per_dim else 1)


def score_error(model,
                test_x: torch.Tensor,
                test_noise: torch.Tensor,
                t: float,
                true_score_fn) -> float:
    """||s_theta(x_t) - s_true(x_t)||^2 on fixed test data, divided by d."""
    e_neg_t = math.exp(-t)
    delta_t = 1 - math.exp(-2 * t)
    x_t = e_neg_t * test_x + math.sqrt(delta_t) * test_noise
    t_batch = torch.full((test_x.shape[0],), t, device=test_x.device)
    pred = model(x_t, t_batch)
    true_s = true_score_fn(x_t, t)
    return ((pred - true_s) ** 2).sum(-1).mean().item() / test_x.shape[-1]


# ---------------------------------------------------------------------------
# Memorization (Bonnaire / Somepalli NN-ratio)
# ---------------------------------------------------------------------------

@dataclass
class MemorizationResult:
    mean_nn_ratio: float
    memorization_fraction: float
    nn1_idx: torch.Tensor


def nn_ratio_memorization(generated: torch.Tensor,
                          train_data: torch.Tensor,
                          *,
                          train_dists: torch.Tensor | None = None,
                          threshold: float = 1.0 / 3.0) -> MemorizationResult:
    """d(gen, NN1_train) / d(NN1_train, NN2_train).

    `train_dists` may be precomputed via torch.cdist(train, train) with the
    diagonal set to inf. If not provided we compute it once here.
    """
    if train_dists is None:
        train_dists = torch.cdist(train_data, train_data)
        train_dists.fill_diagonal_(float('inf'))

    dists = torch.cdist(generated, train_data)
    nn1_dists, nn1_idx = dists.min(dim=1)
    nn2_dists = train_dists[nn1_idx].min(dim=1).values
    ratio = nn1_dists / (nn2_dists + 1e-10)
    return MemorizationResult(
        mean_nn_ratio=ratio.mean().item(),
        memorization_fraction=(ratio < threshold).float().mean().item(),
        nn1_idx=nn1_idx,
    )


# ---------------------------------------------------------------------------
# MMD with Gaussian kernel
# ---------------------------------------------------------------------------

def compute_mmd(x: torch.Tensor, y: torch.Tensor, bandwidth: float = 1.0) -> float:
    def k(a, b):
        return torch.exp(-torch.cdist(a, b) ** 2 / (2 * bandwidth ** 2))
    return (k(x, x).mean() + k(y, y).mean() - 2 * k(x, y).mean()).item()


# ---------------------------------------------------------------------------
# FID
# ---------------------------------------------------------------------------

def fid_from_features(real_feats: np.ndarray,
                      gen_feats: np.ndarray) -> float:
    """Standard Frechet distance between two Gaussians fit to the features.

    Caller is responsible for having extracted features (e.g. InceptionV3
    pool3) from a consistent number of samples per side.
    """
    from scipy import linalg

    mu_r = real_feats.mean(0)
    mu_g = gen_feats.mean(0)
    cov_r = np.cov(real_feats, rowvar=False)
    cov_g = np.cov(gen_feats, rowvar=False)

    diff = mu_r - mu_g
    covmean, _ = linalg.sqrtm(cov_r @ cov_g, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(cov_r + cov_g - 2 * covmean))
