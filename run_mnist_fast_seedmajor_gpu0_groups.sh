#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/latent_space_diffusion_analysis/code/v3"
mkdir -p runner_logs

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

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

run_seed() {
  local seed="$1"
  echo "[$(date)] starting MNIST fast packed seed ${seed}"
  python runner.py "configs/mnist_n1k_fast_seedmajor/seed${seed}.json" --gpus 0,0,0,0,0,0,0
  echo "[$(date)] finished MNIST fast packed seed ${seed}"
}

run_seed 42 &
pid_a=$!
run_seed 43 &
pid_b=$!
wait "$pid_a"
wait "$pid_b"

run_seed 44 &
pid_a=$!
run_seed 45 &
pid_b=$!
wait "$pid_a"
wait "$pid_b"

run_seed 46
