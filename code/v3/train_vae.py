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


def make_loaders(dataset: str, root: str, batch_size: int,
                 n_train_subset: int | None = None):
    if dataset == 'mnist':
        train_loader, test_loader = _mnist_loaders(root, batch_size)
        cfg = VAEConfig(image_channels=1, image_size=32,
                        base_channels=32, n_down=3)
    elif dataset == 'celeba':
        train_loader, test_loader = _celeba_loaders(root, batch_size)
        cfg = VAEConfig(image_channels=3, image_size=64,
                        base_channels=32, n_down=4)
    else:
        raise ValueError(f"Unknown dataset: {dataset!r}")

    # Optionally subset the train set for fast smoke runs (test set is
    # left full so test_loss remains a meaningful held-out signal).
    if n_train_subset is not None and n_train_subset < len(train_loader.dataset):
        from torch.utils.data import Subset, DataLoader
        ds_sub = Subset(train_loader.dataset, range(n_train_subset))
        train_loader = DataLoader(ds_sub, batch_size=batch_size, shuffle=True,
                                   num_workers=train_loader.num_workers)
    return (train_loader, test_loader), cfg


# ---------------------------------------------------------------------------
# Train loop
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    (train_loader, test_loader), base_cfg = make_loaders(
        args.dataset, args.data_root, args.batch_size,
        n_train_subset=args.n_train_subset,
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
                          'lr': args.lr, 'epochs': args.epochs,
                          'early_stop_patience': args.early_stop_patience,
                          'early_stop_tol': args.early_stop_tol,
                          'target_loss': args.target_loss}) + '\n')

    # Best-so-far tracking — used for both early stopping decisions and
    # for saving the best checkpoint (not the final-epoch one).
    best_test_loss = float('inf')
    best_state: dict | None = None
    best_epoch = -1
    epochs_since_improve = 0
    epochs_diverging = 0
    stop_reason = None

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
        test_loss = test_loss / n_test

        # ---- Best-so-far + early-stopping bookkeeping ----
        # First epoch always becomes the new best (avoids inf - inf NaN trap).
        # Subsequent epochs need to beat best by --early_stop_tol fraction.
        if best_state is None:
            is_improvement = True
        else:
            is_improvement = test_loss < best_test_loss * (1 - args.early_stop_tol)

        if is_improvement:
            best_test_loss = test_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_since_improve = 0
            epochs_diverging = 0
        else:
            epochs_since_improve += 1
            # Diverging = current loss is meaningfully worse than the best
            if test_loss > best_test_loss * (1 + args.early_stop_tol):
                epochs_diverging += 1
            else:
                epochs_diverging = 0

        msg = {
            'epoch': epoch,
            'wall_time': time.time() - t0,
            'train_loss': running['loss'] / running['n'],
            'train_recon': running['recon'] / running['n'],
            'train_kl': running['kl'] / running['n'],
            'test_loss': test_loss,
            'best_test_loss': best_test_loss,
            'best_epoch': best_epoch,
            'epochs_since_improve': epochs_since_improve,
            'epochs_diverging': epochs_diverging,
        }
        log.write(json.dumps(msg) + '\n')
        log.flush()
        print(f"  epoch {epoch:3d} | train={msg['train_loss']:.3f} "
              f"recon={msg['train_recon']:.3f} kl={msg['train_kl']:.3f} "
              f"test={msg['test_loss']:.3f} "
              f"best={best_test_loss:.3f}@{best_epoch}", flush=True)

        # ---- Early-stopping triggers (patience > 0 to enable) ----
        if args.target_loss is not None and test_loss <= args.target_loss:
            stop_reason = f'target_loss<={args.target_loss}'
            break
        if args.early_stop_patience > 0:
            if epochs_diverging >= args.early_stop_patience:
                stop_reason = (f'diverging for {epochs_diverging} epochs '
                                f'(test={test_loss:.3f} > best={best_test_loss:.3f})')
                break
            if epochs_since_improve >= args.early_stop_patience:
                stop_reason = (f'plateau: no improvement for '
                                f'{epochs_since_improve} epochs')
                break

    if stop_reason is not None:
        log.write(json.dumps({'event': 'early_stop', 'reason': stop_reason,
                              'epoch': epoch}) + '\n')
        print(f"  stopped early at epoch {epoch}: {stop_reason}", flush=True)
    log.close()

    # Save best (not final) model + sanity grid
    save_state = best_state if best_state is not None else \
                 {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    torch.save({'state_dict': save_state, 'cfg': cfg.__dict__,
                'best_epoch': best_epoch,
                'best_test_loss': best_test_loss},
               out / 'vae.pt')
    # Restore best weights into the live model for the recon grid
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    with torch.no_grad():
        sample_batch = next(iter(test_loader))[0][:32].to(device)
        x_recon, _, _ = model(sample_batch)
        grid = torch.cat([sample_batch, x_recon], dim=0)
        save_image((grid * 0.5 + 0.5).clamp(0, 1), out / 'recon_grid.png', nrow=8)
    print(f"Saved VAE (best epoch {best_epoch}, "
          f"test_loss={best_test_loss:.3f}) to {out / 'vae.pt'}")


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
    p.add_argument('--epochs',    type=int, default=20,
                   help='Hard upper bound on epochs; early stopping may end '
                        'training before this if --early_stop_patience > 0.')
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--lr',        type=float, default=1e-3)
    p.add_argument('--kl_weight', type=float, default=1.0)
    p.add_argument('--early_stop_patience', type=int, default=0,
                   help='Stop after this many consecutive epochs without '
                        'improvement OR with divergence. 0 = disabled.')
    p.add_argument('--early_stop_tol', type=float, default=0.005,
                   help='Relative test-loss tolerance for "improved" / '
                        '"diverging" judgments (default 0.5%).')
    p.add_argument('--target_loss', type=float, default=None,
                   help='Stop as soon as test_loss <= this absolute value. '
                        'None = disabled.')
    p.add_argument('--device',    default=None)
    p.add_argument('--n_train_subset', type=int, default=None,
                   help='If set, train the VAE on only the first N images. '
                        'Test set stays full. Useful for fast smoke runs '
                        'on big datasets like CelebA.')
    args = p.parse_args()

    if args.out is None:
        args.out = f'vae_checkpoints/{args.dataset}_d{args.d_latent}'

    train(args)


if __name__ == '__main__':
    main()
