#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/latent_space_diffusion_analysis/code/v3"
mkdir -p runner_logs

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

for seed in 42 43 44 45 46; do
  echo "[$(date)] starting MNIST FID-100k seed ${seed}"
  python runner.py "configs/mnist_n1k_fid100k_seedmajor/seed${seed}.json" --gpus 2,2,2,3,3,3,3
  echo "[$(date)] finished MNIST FID-100k seed ${seed}"
done
