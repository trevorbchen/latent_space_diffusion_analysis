from __future__ import annotations

import glob

import torch

from lib.vae import ConvVAE, VAEConfig


for path in sorted(glob.glob("vae_checkpoints/mnist_d*/vae.pt")):
    ckpt = torch.load(path, map_location="cpu")
    cfg = VAEConfig(**ckpt["cfg"])
    model = ConvVAE(cfg)
    params = sum(p.numel() for p in model.parameters())
    print(path, ckpt["cfg"], params)
