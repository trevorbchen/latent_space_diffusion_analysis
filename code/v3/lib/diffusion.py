"""OU forward and reverse-SDE sampling for the variance-preserving diffusion.

x_t = e^{-t} x_0 + sqrt(delta_t) eps,   delta_t = 1 - e^{-2t},   eps ~ N(0, I)
"""
from __future__ import annotations

import math

import torch


def forward_noise(x0: torch.Tensor, t):
    """Sample x_t given x0. Returns (x_t, noise, e_neg_t, sqrt_delta_t)."""
    if isinstance(t, (int, float)):
        e_neg_t = math.exp(-t)
        delta_t = 1 - math.exp(-2 * t)
        noise = torch.randn_like(x0)
        x_t = e_neg_t * x0 + math.sqrt(delta_t) * noise
        return x_t, noise, e_neg_t, math.sqrt(delta_t)

    e_neg_t = torch.exp(-t)
    delta_t = 1 - torch.exp(-2 * t)
    noise = torch.randn_like(x0)
    x_t = e_neg_t.unsqueeze(-1) * x0 + delta_t.sqrt().unsqueeze(-1) * noise
    return x_t, noise, e_neg_t, delta_t.sqrt()


@torch.no_grad()
def euler_maruyama(model,
                   n_gen: int,
                   d: int,
                   *,
                   n_steps: int = 500,
                   t_max: float = 3.0,
                   t_min: float = 0.01,
                   device=torch.device('cpu')) -> torch.Tensor:
    """Standard reverse-SDE sampler used in v2."""
    dt = (t_max - t_min) / n_steps
    x = torch.randn(n_gen, d, device=device)
    for i in range(n_steps):
        t_curr = t_max - i * dt
        t_batch = torch.full((n_gen,), t_curr, device=device)
        score = model(x, t_batch)
        noise = torch.randn_like(x) * math.sqrt(2 * dt)
        x = x + (x + 2 * score) * dt + noise
    return x


def score_loss(pred_score: torch.Tensor,
               noise: torch.Tensor,
               sqrt_delta_t) -> torch.Tensor:
    """||sqrt(delta_t) * s + eps||^2 summed over dims, mean over batch."""
    if torch.is_tensor(sqrt_delta_t):
        residual = sqrt_delta_t.unsqueeze(-1) * pred_score + noise
    else:
        residual = sqrt_delta_t * pred_score + noise
    return (residual ** 2).sum(dim=-1).mean()
