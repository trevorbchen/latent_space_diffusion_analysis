"""Anisotropic Gaussian-mixture data on a low-dim manifold in R^d_latent.

Matches the v2 generator (METHODS.md / experiment_v2.py:90) exactly:
orthogonal equal-magnitude cluster centers in d_intrinsic dims, signal
variance sigma_signal in those dims, sigma_noise in the (d_latent -
d_intrinsic) null directions, fixed orthogonal rotation Q.
"""
from __future__ import annotations

import numpy as np
import torch


def generate_data(n: int,
                  d_intrinsic: int,
                  d_latent: int,
                  k: int = 10,
                  sigma_noise: float = 0.5,
                  sigma_signal: float = 1.0,
                  scale: float = 3.0,
                  seed: int = 42):
    """Returns (data, labels, means_full, Q) as torch tensors.

    data:        (n, d_latent)
    labels:      (n,)
    means_full:  (k, d_latent)  — cluster centers in the rotated frame
    Q:           (d_latent, d_latent)  — rotation applied to data and means
    """
    rng = np.random.default_rng(seed)

    if k <= d_intrinsic:
        raw = rng.standard_normal((k, d_intrinsic))
        Q_centers, _ = np.linalg.qr(raw.T)        # (d_intrinsic, k)
        means_intrinsic = Q_centers.T * scale     # rows have norm == scale
    else:
        raw = rng.standard_normal((d_intrinsic, d_intrinsic))
        Q_centers, _ = np.linalg.qr(raw)
        means_intrinsic = np.zeros((k, d_intrinsic))
        means_intrinsic[:d_intrinsic] = Q_centers * scale
        for i in range(d_intrinsic, k):
            v = rng.standard_normal(d_intrinsic)
            means_intrinsic[i] = v / np.linalg.norm(v) * scale

    means_full = np.zeros((k, d_latent))
    means_full[:, :d_intrinsic] = means_intrinsic

    labels = rng.integers(0, k, size=n)
    data = np.zeros((n, d_latent))
    data[:, :d_intrinsic] = (
        means_intrinsic[labels] + rng.standard_normal((n, d_intrinsic)) * sigma_signal
    )
    if d_latent > d_intrinsic:
        data[:, d_intrinsic:] = (
            rng.standard_normal((n, d_latent - d_intrinsic)) * sigma_noise
        )

    Q, _ = np.linalg.qr(rng.standard_normal((d_latent, d_latent)))
    data = data @ Q.T
    means_full = means_full @ Q.T

    return (
        torch.tensor(data, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.long),
        torch.tensor(means_full, dtype=torch.float32),
        torch.tensor(Q, dtype=torch.float32),
    )


def generate_test_samples(means: torch.Tensor,
                          k: int,
                          d_intrinsic: int,
                          d_latent: int,
                          Q: torch.Tensor,
                          sigma_noise: float = 0.5,
                          sigma_signal: float = 1.0,
                          n: int = 2048,
                          seed: int | None = 9999) -> torch.Tensor:
    """Fresh samples from the population distribution. Matches v2 seed=9999."""
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, k, size=n)
    means_orig = means @ Q  # un-rotate
    data = np.zeros((n, d_latent))
    data[:, :d_intrinsic] = (
        means_orig.numpy()[labels, :d_intrinsic]
        + rng.standard_normal((n, d_intrinsic)) * sigma_signal
    )
    if d_latent > d_intrinsic:
        data[:, d_intrinsic:] = (
            rng.standard_normal((n, d_latent - d_intrinsic)) * sigma_noise
        )
    data = data @ Q.T.numpy()
    return torch.tensor(data, dtype=torch.float32)
