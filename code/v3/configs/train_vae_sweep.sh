#!/usr/bin/env bash
# Train all VAEs at the d_latent values needed by the real-data sweep.
# Run on a single GPU; each VAE takes <30 min for MNIST and ~2h for CelebA.

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

MNIST_DS=(2 4 8 16 32 64 128)
CELEBA_DS=(8 16 32 64 128 256)

for d in "${MNIST_DS[@]}"; do
    python train_vae.py --dataset mnist --d_latent "$d" --epochs 20
done

for d in "${CELEBA_DS[@]}"; do
    python train_vae.py --dataset celeba --d_latent "$d" --epochs 30
done
