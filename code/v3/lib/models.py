"""Score networks: trainable MLP and Bonnaire-style RFNN.

Both expose s_theta(x_t, t) -> R^d. The RFNN is trained at fixed t, so its
forward signature accepts t but ignores it — keeping a single training-loop
interface across the two models.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class MLPScore(nn.Module):
    """3-layer GELU MLP with sinusoidal time embedding (matches v2)."""

    def __init__(self, d_latent: int, hidden: int = 256, n_freq: int = 32):
        super().__init__()
        self.d_latent = d_latent
        self.n_freq = n_freq
        self.register_buffer(
            'freqs', torch.exp(torch.linspace(0, math.log(1000), n_freq))
        )
        d_input = d_latent + 2 * n_freq
        self.net = nn.Sequential(
            nn.Linear(d_input, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_latent),
        )

    def forward(self, x_t: torch.Tensor, t) -> torch.Tensor:
        if isinstance(t, (int, float)):
            t = torch.full((x_t.shape[0],), float(t), device=x_t.device)
        if t.dim() == 0:
            t = t.expand(x_t.shape[0])
        t_emb = torch.cat([
            torch.sin(t.unsqueeze(-1) * self.freqs),
            torch.cos(t.unsqueeze(-1) * self.freqs),
        ], dim=-1)
        return self.net(torch.cat([x_t, t_emb], dim=-1))


class RFNNScore(nn.Module):
    """Bonnaire RFNN: s_A(x) = (1/sqrt(p)) A * tanh(W x / sqrt(d_latent)).

    W is frozen at N(0, 1/d_latent). A is learnable, zero-init.
    Trained at fixed t; forward ignores its `t` arg for a uniform interface.
    """

    def __init__(self, d_latent: int, p: int, t_fixed: float = 0.01):
        super().__init__()
        self.d_latent = d_latent
        self.p = p
        self.t_fixed = t_fixed
        self.register_buffer(
            'W', torch.randn(p, d_latent) / math.sqrt(d_latent)
        )
        self.A = nn.Parameter(torch.zeros(d_latent, p))

    def forward(self, x_t: torch.Tensor, t=None) -> torch.Tensor:
        features = torch.tanh(x_t @ self.W.T)               # (B, p)
        return features @ self.A.T / math.sqrt(self.p)


DEFAULT_MLP_HIDDEN = 256


def build_model(model_kind: str, d_latent: int, *,
                hidden: int | None = None,
                p_ratio: int = 64,
                t_fixed: float = 0.01) -> nn.Module:
    """Build score network. MLP default is fixed-256 (deviates from the v2
    project's 8*d_latent rule per user request, 2026-05-09); pass
    `hidden=8 * d_latent` to recover the paper's capacity scaling.
    """
    if model_kind == 'mlp':
        h = hidden if hidden is not None else DEFAULT_MLP_HIDDEN
        return MLPScore(d_latent, hidden=h)
    if model_kind == 'rfnn':
        return RFNNScore(d_latent, p=p_ratio * d_latent, t_fixed=t_fixed)
    raise ValueError(f"Unknown model kind: {model_kind!r}")
