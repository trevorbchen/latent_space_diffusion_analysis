"""Real-data adapters: load MNIST/CelebA, encode through a trained VAE,
expose an `(n_train, d_latent)` tensor that the diffusion training loop
treats exactly like synthetic data.

The encoded latents are cached on disk so the VAE forward pass is paid
exactly once per (dataset, vae_checkpoint) pair across all diffusion runs.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .vae import ConvVAE, VAEConfig

# torchvision imports are lazy so the synthetic path doesn't require it.


# ---------------------------------------------------------------------------
# Dataset transforms (must match train_vae.py)
# ---------------------------------------------------------------------------

def _mnist_transform():
    from torchvision import transforms
    return transforms.Compose([
        transforms.Pad(2),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])


def _celeba_transform(image_size: int = 32):
    from torchvision import transforms
    return transforms.Compose([
        transforms.CenterCrop(178),
        transforms.Resize(image_size),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])


def _load_dataset(name: str, data_root: str, split: str, image_size: int):
    from torchvision import datasets
    if name == 'mnist':
        train_flag = (split == 'train')
        return datasets.MNIST(data_root, train=train_flag, download=True,
                              transform=_mnist_transform())
    if name == 'celeba':
        return datasets.CelebA(data_root, split=split, download=True,
                               transform=_celeba_transform(image_size))
    raise ValueError(f"Unknown dataset: {name!r}")


# ---------------------------------------------------------------------------
# Encoding / caching
# ---------------------------------------------------------------------------

def _cache_key(dataset: str, vae_checkpoint: str, split: str,
               image_size: int, d_latent: int) -> str:
    h = hashlib.sha1()
    h.update(dataset.encode())
    h.update(vae_checkpoint.encode())
    h.update(split.encode())
    h.update(f"{image_size}-{d_latent}".encode())
    return h.hexdigest()[:16]


def load_vae(checkpoint_path: str, device: torch.device) -> ConvVAE:
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = VAEConfig(**ckpt['cfg'])
    model = ConvVAE(cfg).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    return model


@torch.no_grad()
def encode_split(vae: ConvVAE,
                 dataset_name: str,
                 split: str,
                 *,
                 data_root: str,
                 batch_size: int = 256,
                 device: torch.device = torch.device('cpu'),
                 cache_dir: str = './data/encoded') -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (latents, images). Latents in (n, d_latent), images in pixel space.

    The pixel-space images are needed for the memorization-fraction NN ratio
    on real data (Somepalli/Bonnaire compute distances in pixel space).
    """
    cache_root = Path(cache_dir); cache_root.mkdir(parents=True, exist_ok=True)
    key = _cache_key(dataset_name,
                     str(getattr(vae, '_ckpt_path', 'inline')),
                     split,
                     vae.cfg.image_size,
                     vae.cfg.d_latent)
    cache_path = cache_root / f'{dataset_name}_{split}_{key}.pt'
    if cache_path.exists():
        blob = torch.load(cache_path, map_location='cpu')
        return blob['latents'], blob['images']

    ds = _load_dataset(dataset_name, data_root, split, vae.cfg.image_size)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)

    latents_chunks: list[torch.Tensor] = []
    images_chunks: list[torch.Tensor] = []
    for batch in loader:
        x = batch[0].to(device)
        mu, _ = vae.encode(x)
        latents_chunks.append(mu.cpu())
        images_chunks.append(x.cpu())

    latents = torch.cat(latents_chunks, 0)
    images = torch.cat(images_chunks, 0)
    torch.save({'latents': latents, 'images': images}, cache_path)
    return latents, images


# ---------------------------------------------------------------------------
# High-level entry point
# ---------------------------------------------------------------------------

@dataclass
class RealDataBundle:
    train_latents: torch.Tensor
    test_latents: torch.Tensor
    train_images: torch.Tensor   # for memorization NN ratio
    test_images: torch.Tensor
    vae: ConvVAE
    d_latent: int


def load_real_dataset(dataset_name: str,
                      vae_checkpoint: str,
                      *,
                      data_root: str = './data',
                      device: torch.device = torch.device('cpu'),
                      n_train: int | None = None) -> RealDataBundle:
    vae = load_vae(vae_checkpoint, device)
    setattr(vae, '_ckpt_path', vae_checkpoint)
    train_latents, train_images = encode_split(
        vae, dataset_name, 'train', data_root=data_root, device=device,
    )
    test_split = 'test' if dataset_name == 'mnist' else 'valid'
    test_latents, test_images = encode_split(
        vae, dataset_name, test_split, data_root=data_root, device=device,
    )
    if n_train is not None:
        train_latents = train_latents[:n_train]
        train_images = train_images[:n_train]
    return RealDataBundle(
        train_latents=train_latents,
        test_latents=test_latents,
        train_images=train_images,
        test_images=test_images,
        vae=vae,
        d_latent=vae.cfg.d_latent,
    )
