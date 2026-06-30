#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/latent_space_diffusion_analysis/code/v3"
mkdir -p runner_logs

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2

run_vae() {
  local gpu="$1"
  local d="$2"
  local out="vae_checkpoints/cifar10_resnet_beta005_d${d}"
  mkdir -p "${out}"
  if [ -f "${out}/vae.pt" ]; then
    echo "[$(date)] skipping CIFAR-10 beta005 VAE d=${d}; checkpoint exists"
    return 0
  fi
  echo "[$(date)] starting CIFAR-10 beta005 VAE d=${d} on gpu ${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" python3 train_vae_cifar.py \
    --d_latent "${d}" \
    --out "${out}" \
    --epochs 200 \
    --batch_size 256 \
    --arch resnet \
    --beta 0.05 \
    --gamma 0.0 \
    --max_capacity 25.0 \
    --capacity_warmup_frac 0.35 \
    --grad_clip 1.5 \
    --early_stop_patience 20 \
    --early_stop_tol 0.002 \
    > "${out}/train_stdout.log" 2>&1
  echo "[$(date)] finished CIFAR-10 beta005 VAE d=${d}"
}

run_vae 2 10 &
run_vae 2 20 &
run_vae 2 30 &
run_vae 2 40 &
run_vae 2 50 &

run_vae 3 60 &
run_vae 3 70 &
run_vae 3 80 &
run_vae 3 90 &
run_vae 3 100 &

wait

echo "[$(date)] all CIFAR-10 beta005 VAE d10-100 jobs complete"
