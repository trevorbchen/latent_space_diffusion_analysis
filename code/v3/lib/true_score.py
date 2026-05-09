"""Analytic score for an anisotropic Gaussian-mixture P_0.

P_t(x) = (1/k) sum_j N(e^{-t} mu_j, Sigma_t),
Sigma_t = delta_t I + e^{-2t} Sigma_data.

Sigma_data is diagonal in the un-rotated frame (sigma_signal^2 in the first
d_intrinsic dims, sigma_noise^2 in the rest), then carried into the data
frame by the orthogonal Q. The early v2 bug of treating Sigma_t as
isotropic is avoided by always passing Q.
"""
from __future__ import annotations

import math

import torch


def true_score(x: torch.Tensor,
               t: float,
               means: torch.Tensor,
               *,
               sigma_t_inv: torch.Tensor | None = None,
               log_det_sigma_t: torch.Tensor | None = None) -> torch.Tensor:
    """nabla log P_t(x). When sigma_t_inv is None we fall back to Sigma_t = I."""
    e_neg_t = math.exp(-t)
    delta_t = 1 - math.exp(-2 * t)

    shifted = x.unsqueeze(1) - e_neg_t * means.unsqueeze(0)  # (B, k, d)

    if sigma_t_inv is not None:
        shifted_prec = shifted @ sigma_t_inv               # (B, k, d)
        log_w = -0.5 * (shifted * shifted_prec).sum(-1)
        if log_det_sigma_t is not None:
            log_w = log_w - 0.5 * log_det_sigma_t
        log_w = log_w - log_w.max(dim=1, keepdim=True).values
        w = torch.softmax(log_w, dim=1)
        return -(w.unsqueeze(-1) * shifted_prec).sum(1)

    sigma_t = delta_t + math.exp(-2 * t)
    log_w = -0.5 * (shifted ** 2).sum(-1) / sigma_t
    log_w = log_w - log_w.max(dim=1, keepdim=True).values
    w = torch.softmax(log_w, dim=1)
    return (w.unsqueeze(-1) * (-shifted / sigma_t)).sum(1)


def precompute_sigma_t(d_intrinsic: int,
                       d_latent: int,
                       Q: torch.Tensor,
                       sigma_noise: float,
                       t: float,
                       sigma_signal: float = 1.0):
    """Returns (Sigma_t^{-1}, log det Sigma_t) in the data frame."""
    delta_t = 1 - math.exp(-2 * t)
    e_neg_2t = math.exp(-2 * t)

    sigma_data_diag = torch.empty(d_latent)
    sigma_data_diag[:d_intrinsic] = sigma_signal ** 2
    if d_latent > d_intrinsic:
        sigma_data_diag[d_intrinsic:] = sigma_noise ** 2

    sigma_t_diag = delta_t + e_neg_2t * sigma_data_diag
    Q_t = Q.float()
    sigma_t_inv = Q_t.T @ torch.diag(1.0 / sigma_t_diag) @ Q_t
    log_det = torch.log(sigma_t_diag).sum()
    return sigma_t_inv, log_det
