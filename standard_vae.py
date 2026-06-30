from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class StandardVAEConfig:
    image_channels: int = 1
    image_size: int = 32
    hidden_dims: tuple[int, ...] = (32, 64, 128, 256, 512)
    d_latent: int = 32
    arch: str = "standard"


def _group_count(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        groups = _group_count(channels)
        self.net = nn.Sequential(
            nn.GroupNorm(groups, channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class ResDownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.down = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.res = ResidualBlock(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.res(self.down(x))


class ResUpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )
        self.res = ResidualBlock(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.res(self.up(x))


class StandardConvVAE(nn.Module):
    """AntixK-style convolutional VAE for 64x64 images.

    Encoder blocks use stride-2 Conv2d + BatchNorm + LeakyReLU.
    Decoder mirrors them with ConvTranspose2d blocks and Tanh output.
    """

    def __init__(self, cfg: StandardVAEConfig):
        super().__init__()
        self.cfg = cfg

        modules: list[nn.Module] = []
        in_channels = cfg.image_channels
        for h_dim in cfg.hidden_dims:
            if cfg.arch == "resnet":
                modules.append(ResDownBlock(in_channels, h_dim))
            else:
                modules.append(
                    nn.Sequential(
                        nn.Conv2d(
                            in_channels,
                            h_dim,
                            kernel_size=3,
                            stride=2,
                            padding=1,
                        ),
                        nn.BatchNorm2d(h_dim),
                        nn.LeakyReLU(0.2, inplace=True),
                    )
                )
            in_channels = h_dim
        self.encoder = nn.Sequential(*modules)

        self.final_spatial = cfg.image_size // (2 ** len(cfg.hidden_dims))
        if self.final_spatial < 1:
            raise ValueError("image_size too small for hidden_dims depth")
        final_dim = cfg.hidden_dims[-1] * self.final_spatial * self.final_spatial
        self.fc_mu = nn.Linear(final_dim, cfg.d_latent)
        self.fc_logvar = nn.Linear(final_dim, cfg.d_latent)

        self.decoder_input = nn.Linear(cfg.d_latent, final_dim)
        hidden_rev = list(cfg.hidden_dims[::-1])
        decoder_blocks: list[nn.Module] = []
        for i in range(len(hidden_rev) - 1):
            if cfg.arch == "resnet":
                decoder_blocks.append(ResUpBlock(hidden_rev[i], hidden_rev[i + 1]))
            else:
                decoder_blocks.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(
                            hidden_rev[i],
                            hidden_rev[i + 1],
                            kernel_size=3,
                            stride=2,
                            padding=1,
                            output_padding=1,
                        ),
                        nn.BatchNorm2d(hidden_rev[i + 1]),
                        nn.LeakyReLU(0.2, inplace=True),
                    )
                )
        self.decoder = nn.Sequential(*decoder_blocks)
        if cfg.arch == "resnet":
            self.final_layer = nn.Sequential(
                ResUpBlock(hidden_rev[-1], hidden_rev[-1]),
                nn.GroupNorm(_group_count(hidden_rev[-1]), hidden_rev[-1]),
                nn.SiLU(inplace=True),
                nn.Conv2d(hidden_rev[-1], cfg.image_channels, kernel_size=3, padding=1),
                nn.Tanh(),
            )
        else:
            self.final_layer = nn.Sequential(
                nn.ConvTranspose2d(
                    hidden_rev[-1],
                    hidden_rev[-1],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                nn.BatchNorm2d(hidden_rev[-1]),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(hidden_rev[-1], cfg.image_channels, kernel_size=3, padding=1),
                nn.Tanh(),
            )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.decoder_input(z)
        h = h.view(z.size(0), self.cfg.hidden_dims[-1], self.final_spatial, self.final_spatial)
        h = self.decoder(h)
        return self.final_layer(h)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def standard_vae_loss(
    x: torch.Tensor,
    x_recon: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    *,
    beta: float = 1.0,
    gamma: float = 0.0,
    capacity: float = 0.0,
    mse_weight: float = 1.0,
    l1_weight: float = 0.0,
    free_bits: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    mse = F.mse_loss(x_recon, x, reduction="none").flatten(1).sum(-1).mean()
    l1 = F.l1_loss(x_recon, x, reduction="none").flatten(1).sum(-1).mean()
    recon = mse_weight * mse + l1_weight * l1
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    kl = kl_per_dim.sum(-1).mean()
    if free_bits > 0:
        kl_objective = kl_per_dim.mean(0).clamp_min(free_bits).sum()
    else:
        kl_objective = kl
    if gamma > 0:
        loss = recon + gamma * (kl - capacity).abs()
    else:
        loss = recon + beta * kl_objective
    return loss, {
        "loss": float(loss.detach()),
        "recon": float(recon.detach()),
        "mse": float(mse.detach()),
        "l1": float(l1.detach()),
        "kl": float(kl.detach()),
        "kl_objective": float(kl_objective.detach()),
        "capacity": float(capacity),
    }
