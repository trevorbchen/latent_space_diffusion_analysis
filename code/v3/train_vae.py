"""Train a ConvVAE on MNIST or CelebA at a configurable bottleneck.

Saves:
    {out_dir}/vae.pt          — state dict + config
    {out_dir}/recon_grid.png  — sanity-check image grid
    {out_dir}/training.log    — per-epoch loss

Usage:
    python train_vae.py --dataset mnist  --d_latent 10
    python train_vae.py --dataset celeba --d_latent 64 --epochs 30
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from lib.vae import ConvVAE, VAEConfig, vae_loss


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def _mnist_loaders(root: str, batch_size: int):
    tfm = transforms.Compose([
        transforms.Pad(2),               # 28 -> 32
        transforms.ToTensor(),           # [0, 1]
        transforms.Normalize((0.5,), (0.5,)),  # -> [-1, 1]
    ])
    train = datasets.MNIST(root, train=True, download=True, transform=tfm)
    test = datasets.MNIST(root, train=False, download=True, transform=tfm)
    return (DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=2),
            DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=2))


def _celeba_loaders(root: str, batch_size: int, image_size: int = 64):
    tfm = transforms.Compose([
        transforms.CenterCrop(178),      # CelebA's standard pre-crop
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * 3, (0.5,) * 3),
    ])
    train = datasets.CelebA(root, split='train', download=True, transform=tfm)
    test = datasets.CelebA(root, split='valid', download=True, transform=tfm)
    return (DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=4),
            DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=4))


def make_loaders(dataset: str, root: str, batch_size: int):
    if dataset == 'mnist':
        return _mnist_loaders(root, batch_size), VAEConfig(
            image_channels=1, image_size=32, base_channels=32, n_down=3,
        )
    if dataset == 'celeba':
        return _celeba_loaders(root, batch_size), VAEConfig(
            image_channels=3, image_size=64, base_channels=32, n_down=4,
        )
    raise ValueError(f"Unknown dataset: {dataset!r}")


# ---------------------------------------------------------------------------
# Train loop
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    (train_loader, test_loader), base_cfg = make_loaders(
        args.dataset, args.data_root, args.batch_size,
    )
    cfg = VAEConfig(
        image_channels=base_cfg.image_channels,
        image_size=base_cfg.image_size,
        base_channels=base_cfg.base_channels,
        n_down=base_cfg.n_down,
        d_latent=args.d_latent,
    )
    model = ConvVAE(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    log_path = out / 'training.log'
    log = open(log_path, 'w')
    log.write(json.dumps({'event': 'config', 'cfg': cfg.__dict__,
                          'lr': args.lr, 'epochs': args.epochs}) + '\n')

    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        running = {'loss': 0.0, 'recon': 0.0, 'kl': 0.0, 'n': 0}
        for batch in train_loader:
            x = batch[0].to(device, non_blocking=True)
            x_recon, mu, logvar = model(x)
            loss, parts = vae_loss(x, x_recon, mu, logvar, kl_weight=args.kl_weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
            bs = x.size(0)
            running['loss']  += loss.item() * bs
            running['recon'] += parts['recon'] * bs
            running['kl']    += parts['kl'] * bs
            running['n']     += bs

        # eval
        model.eval()
        test_loss = 0.0; n_test = 0
        with torch.no_grad():
            for batch in test_loader:
                x = batch[0].to(device)
                x_recon, mu, logvar = model(x)
                l, _ = vae_loss(x, x_recon, mu, logvar, kl_weight=args.kl_weight)
                test_loss += l.item() * x.size(0)
                n_test += x.size(0)

        msg = {
            'epoch': epoch,
            'wall_time': time.time() - t0,
            'train_loss': running['loss'] / running['n'],
            'train_recon': running['recon'] / running['n'],
            'train_kl': running['kl'] / running['n'],
            'test_loss': test_loss / n_test,
        }
        log.write(json.dumps(msg) + '\n')
        log.flush()
        print(f"  epoch {epoch:3d} | train={msg['train_loss']:.3f} "
              f"recon={msg['train_recon']:.3f} kl={msg['train_kl']:.3f} "
              f"test={msg['test_loss']:.3f}", flush=True)

    log.close()

    # Save model + sanity grid
    torch.save({'state_dict': model.state_dict(), 'cfg': cfg.__dict__}, out / 'vae.pt')
    with torch.no_grad():
        sample_batch = next(iter(test_loader))[0][:32].to(device)
        x_recon, _, _ = model(sample_batch)
        grid = torch.cat([sample_batch, x_recon], dim=0)
        save_image((grid * 0.5 + 0.5).clamp(0, 1), out / 'recon_grid.png', nrow=8)
    print(f"Saved VAE to {out / 'vae.pt'}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset',   choices=['mnist', 'celeba'], required=True)
    p.add_argument('--d_latent',  type=int, required=True)
    p.add_argument('--data_root', default='./data')
    p.add_argument('--out',       default=None,
                   help='Defaults to vae_checkpoints/{dataset}_d{d_latent}/')
    p.add_argument('--epochs',    type=int, default=20)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--lr',        type=float, default=1e-3)
    p.add_argument('--kl_weight', type=float, default=1.0)
    p.add_argument('--device',    default=None)
    args = p.parse_args()

    if args.out is None:
        args.out = f'vae_checkpoints/{args.dataset}_d{args.d_latent}'

    train(args)


if __name__ == '__main__':
    main()
