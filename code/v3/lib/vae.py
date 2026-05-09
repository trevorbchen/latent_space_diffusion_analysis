"""Compact convolutional VAE with configurable bottleneck `d_latent`.

Used to compress real images into a low-dimensional latent on which the
diffusion experiments run. The architecture is intentionally small and
identical for MNIST (1x32x32 after pad) and CelebA (3x64x64) — only the
input channel count and image size change. We keep the same encoder
trunk depth so that capacity scales primarily with `d_latent`.

This is a standard Gaussian VAE (mean + log-variance heads, KL regularizer,
reparameterization trick). It is *not* meant to be SoTA; just clean enough
to give a usable latent for the diffusion sweep.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def _conv_block(c_in: int, c_out: int, *, down: bool = True) -> nn.Sequential:
    stride = 2 if down else 1
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, stride=stride, padding=1),
        nn.GroupNorm(min(8, c_out), c_out),
        nn.SiLU(),
        nn.Conv2d(c_out, c_out, kernel_size=3, stride=1, padding=1),
        nn.GroupNorm(min(8, c_out), c_out),
        nn.SiLU(),
    )


def _deconv_block(c_in: int, c_out: int, *, up: bool = True) -> nn.Sequential:
    layers: list[nn.Module] = []
    if up:
        layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
    layers += [
        nn.Conv2d(c_in, c_out, kernel_size=3, stride=1, padding=1),
        nn.GroupNorm(min(8, c_out), c_out),
        nn.SiLU(),
        nn.Conv2d(c_out, c_out, kernel_size=3, stride=1, padding=1),
        nn.GroupNorm(min(8, c_out), c_out),
        nn.SiLU(),
    ]
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# VAE
# ---------------------------------------------------------------------------

@dataclass
class VAEConfig:
    image_channels: int = 1     # 1 for MNIST, 3 for CelebA
    image_size: int = 32        # power-of-2 (MNIST padded 28 -> 32)
    base_channels: int = 32
    n_down: int = 3             # 32 -> 4, 64 -> 8
    d_latent: int = 16


class ConvVAE(nn.Module):
    def __init__(self, cfg: VAEConfig):
        super().__init__()
        self.cfg = cfg

        chs = [cfg.base_channels * (2 ** i) for i in range(cfg.n_down + 1)]
        # Encoder
        enc_layers: list[nn.Module] = [
            nn.Conv2d(cfg.image_channels, chs[0], kernel_size=3, padding=1),
        ]
        for i in range(cfg.n_down):
            enc_layers.append(_conv_block(chs[i], chs[i + 1], down=True))
        self.encoder = nn.Sequential(*enc_layers)

        feat_size = cfg.image_size // (2 ** cfg.n_down)
        feat_dim = chs[-1] * feat_size * feat_size
        self.feat_size = feat_size
        self.feat_chs = chs[-1]
        self.fc_mu = nn.Linear(feat_dim, cfg.d_latent)
        self.fc_logvar = nn.Linear(feat_dim, cfg.d_latent)

        # Decoder
        self.fc_dec = nn.Linear(cfg.d_latent, feat_dim)
        dec_layers: list[nn.Module] = []
        for i in range(cfg.n_down, 0, -1):
            dec_layers.append(_deconv_block(chs[i], chs[i - 1], up=True))
        dec_layers.append(
            nn.Conv2d(chs[0], cfg.image_channels, kernel_size=3, padding=1)
        )
        self.decoder = nn.Sequential(*dec_layers)

    # ---- encode / decode ----
    def encode(self, x: torch.Tensor):
        h = self.encoder(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_dec(z).view(-1, self.feat_chs, self.feat_size, self.feat_size)
        return self.decoder(h)

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def vae_loss(x: torch.Tensor,
             x_recon: torch.Tensor,
             mu: torch.Tensor,
             logvar: torch.Tensor,
             *,
             kl_weight: float = 1.0) -> tuple[torch.Tensor, dict]:
    """Negative ELBO with Gaussian likelihood (sum-of-squares recon)."""
    recon = F.mse_loss(x_recon, x, reduction='none').flatten(1).sum(-1).mean()
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(-1).mean()
    loss = recon + kl_weight * kl
    return loss, {'recon': recon.item(), 'kl': kl.item()}
