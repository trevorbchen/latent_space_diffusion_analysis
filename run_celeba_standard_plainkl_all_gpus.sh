#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/latent_space_diffusion_analysis/code/v3"
mkdir -p runner_logs

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2

run_d() {
  local gpu="$1"
  local d="$2"
  local out="vae_checkpoints/celeba_standard_plainkl_d${d}"
  mkdir -p "${out}"
  if [ -f "${out}/vae.pt" ]; then
    echo "[$(date)] skipping standard plain-KL CelebA-HQ VAE d=${d}; checkpoint exists"
    return 0
  fi
  echo "[$(date)] starting standard plain-KL CelebA-HQ VAE d=${d} on gpu ${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" python3 train_vae_celeba_standard_tar.py \
    --d_latent "${d}" \
    --epochs 250 \
    --early_stop_patience 25 \
    --early_stop_tol 0.001 \
    --batch_size 256 \
    --num_workers 2 \
    --image_size 32 \
    --hidden_dims 32,64,128,256,512 \
    --gamma 0.0 \
    --beta 1.0 \
    --beta_warmup_frac 0.10 \
    --grad_clip 1.5 \
    --out "${out}" \
    > "${out}/train_stdout.log" 2>&1
  echo "[$(date)] finished standard plain-KL CelebA-HQ VAE d=${d}"
}

run_d 0 10 &
run_d 1 15 &
run_d 2 20 &
run_d 3 25 &
run_d 0 30 &
run_d 1 35 &
run_d 2 40 &
run_d 3 45 &
run_d 0 50 &

wait
echo "[$(date)] all standard plain-KL CelebA-HQ VAE jobs complete"
