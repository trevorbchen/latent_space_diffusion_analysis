#!/usr/bin/env bash
# Train all VAEs at the d_latent values needed by the real-data sweep.
# Per VAE: hard --epochs cap + early stopping (plateau or divergence) +
# target test_loss. Whichever fires first wins. Best-epoch weights saved.
#
# Tuning notes for the targets:
# - MNIST: ELBO of ~70 is "good enough digits" at 32x32 grayscale; below
#   that the recon term has stopped improving and KL is essentially flat.
# - CelebA: ELBO of ~200 is "good enough faces" at 64x64 RGB. CelebA is
#   3-channel and 4x the spatial resolution, so the absolute number is
#   roughly 12x MNIST.
#
# Run on a single GPU; the loop below dispatches them sequentially so you
# can ctrl-C cleanly without orphaning subprocesses.

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

MNIST_DS=(2 4 8 16 32 64 128)
CELEBA_DS=(8 16 32 64 128 256)

# ---- MNIST ----
for d in "${MNIST_DS[@]}"; do
    echo "=== MNIST d_latent=${d} ==="
    python train_vae.py \
        --dataset mnist \
        --d_latent "$d" \
        --epochs 30 \
        --early_stop_patience 5 \
        --early_stop_tol 0.005 \
        --target_loss 70
done

# ---- CelebA ----
for d in "${CELEBA_DS[@]}"; do
    echo "=== CelebA d_latent=${d} ==="
    python train_vae.py \
        --dataset celeba \
        --d_latent "$d" \
        --epochs 80 \
        --early_stop_patience 8 \
        --early_stop_tol 0.005 \
        --target_loss 200
done
