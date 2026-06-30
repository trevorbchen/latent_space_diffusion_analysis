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
  local out="vae_checkpoints/celeba_standard_d${d}"
  mkdir -p "${out}"
  echo "[$(date)] starting standard CelebA-HQ VAE d=${d} on gpu ${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" python3 train_vae_celeba_standard_tar.py \
    --d_latent "${d}" \
    --epochs 250 \
    --early_stop_patience 25 \
    --early_stop_tol 0.001 \
    --batch_size 256 \
    --num_workers 4 \
    --image_size 32 \
    --hidden_dims 32,64,128,256,512 \
    --gamma 30.0 \
    --max_capacity 25.0 \
    --capacity_warmup_frac 0.35 \
    --grad_clip 1.5 \
    --out "${out}" \
    > "${out}/train_stdout.log" 2>&1
  echo "[$(date)] finished standard CelebA-HQ VAE d=${d}"
}

run_d 0 10 &
pid_a=$!
run_d 1 15 &
pid_b=$!
wait "$pid_a"
wait "$pid_b"

run_d 0 20 &
pid_a=$!
run_d 1 25 &
pid_b=$!
wait "$pid_a"
wait "$pid_b"

run_d 0 30 &
pid_a=$!
run_d 1 35 &
pid_b=$!
wait "$pid_a"
wait "$pid_b"

run_d 0 40 &
pid_a=$!
run_d 1 45 &
pid_b=$!
wait "$pid_a"
wait "$pid_b"

run_d 0 50
