#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/latent_space_diffusion_analysis/code/v3"
mkdir -p runner_logs

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

run_d() {
  local gpu="$1"
  local d="$2"
  echo "[$(date)] starting CelebA-HQ VAE d=${d} on gpu ${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" python3 train_vae_celeba_tar.py \
    --d_latent "${d}" \
    --epochs 200 \
    --early_stop_patience 15 \
    --early_stop_tol 0.002 \
    --batch_size 256 \
    --num_workers 4 \
    --out "vae_checkpoints/celeba_d${d}" \
    > "vae_checkpoints/celeba_d${d}/train_stdout.log" 2>&1
  echo "[$(date)] finished CelebA-HQ VAE d=${d}"
}

mkdir -p vae_checkpoints/celeba_d10 vae_checkpoints/celeba_d15
run_d 0 10 &
pid_a=$!
run_d 1 15 &
pid_b=$!
wait "$pid_a"
wait "$pid_b"

mkdir -p vae_checkpoints/celeba_d20 vae_checkpoints/celeba_d25
run_d 0 20 &
pid_a=$!
run_d 1 25 &
pid_b=$!
wait "$pid_a"
wait "$pid_b"

mkdir -p vae_checkpoints/celeba_d30 vae_checkpoints/celeba_d35
run_d 0 30 &
pid_a=$!
run_d 1 35 &
pid_b=$!
wait "$pid_a"
wait "$pid_b"

mkdir -p vae_checkpoints/celeba_d40 vae_checkpoints/celeba_d45
run_d 0 40 &
pid_a=$!
run_d 1 45 &
pid_b=$!
wait "$pid_a"
wait "$pid_b"

mkdir -p vae_checkpoints/celeba_d50
run_d 0 50
