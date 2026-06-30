#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/latent_space_diffusion_analysis/code/v3"
mkdir -p runner_logs

if [ ! -f vae_checkpoints/mnist_d5/vae.pt ]; then
  echo "[$(date)] training missing MNIST VAE d=5"
  python train_vae.py \
    --dataset mnist \
    --d_latent 5 \
    --epochs 30 \
    --early_stop_patience 5 \
    --early_stop_tol 0.005 \
    --target_loss 70
else
  echo "[$(date)] MNIST VAE d=5 exists; skipping VAE training"
fi

for seed in 42 43 44 45 46; do
  echo "[$(date)] starting MNIST fast packed seed ${seed}"
  python runner.py "configs/mnist_n1k_fast_seedmajor/seed${seed}.json" --gpus 0,0,0,0,0,0,0
  echo "[$(date)] finished MNIST fast packed seed ${seed}"
done
