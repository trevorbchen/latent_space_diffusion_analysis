#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/latent_space_diffusion_analysis/code/v3"

while pgrep -f "run_cifar_diverse_diffusion.py .*seed43/d220|run_cifar_diverse_diffusion.py .*seed43/d240" >/dev/null; do
  sleep 120
done

env CUDA_VISIBLE_DEVICES=1 python3 retry_cifar_fid.py \
  --run_dir results/cifar10_beta005_diverse1k_bigmlp_sgd_lr001_m08_10k_5m/seed42/d260 \
  --attempts 3
