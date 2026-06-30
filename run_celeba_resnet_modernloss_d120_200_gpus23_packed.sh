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
  local out="vae_checkpoints/celeba_resnet_modernloss_d${d}"
  mkdir -p "${out}"
  if [ -f "${out}/vae.pt" ]; then
    echo "[$(date)] skipping ResNet modern-loss CelebA-HQ VAE d=${d}; checkpoint exists"
    return 0
  fi
  echo "[$(date)] starting ResNet modern-loss CelebA-HQ VAE d=${d} on gpu ${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" python3 train_vae_celeba_standard_tar.py \
    --d_latent "${d}" \
    --epochs 300 \
    --early_stop_patience 30 \
    --early_stop_tol 0.0008 \
    --batch_size 256 \
    --num_workers 2 \
    --image_size 32 \
    --hidden_dims 32,64,128,256,512 \
    --arch resnet \
    --gamma 0.0 \
    --beta 0.05 \
    --beta_warmup_frac 0.15 \
    --mse_weight 0.5 \
    --l1_weight 1.0 \
    --free_bits 0.05 \
    --ema_decay 0.999 \
    --grad_clip 1.5 \
    --out "${out}" \
    > "${out}/train_stdout.log" 2>&1
  echo "[$(date)] finished ResNet modern-loss CelebA-HQ VAE d=${d}"
}

# Pack high-d VAE training onto the currently idle GPUs.
run_d 2 120 &
run_d 2 140 &
run_d 2 160 &

run_d 3 180 &
run_d 3 200 &

wait
echo "[$(date)] all d120-200 ResNet modern-loss CelebA-HQ VAE jobs complete"
