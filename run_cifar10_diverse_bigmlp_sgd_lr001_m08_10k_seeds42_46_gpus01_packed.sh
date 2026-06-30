#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/latent_space_diffusion_analysis/code/v3"
mkdir -p runner_logs

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2

OUT_ROOT="results/cifar10_diverse1k_bigmlp_sgd_lr001_m08_10k_5m"
SUBSET="diagnostics/cifar10_diverse_1k_mean_plus_sd/subset_indices.json"

wait_for_cifar_vaes() {
  while true; do
    local missing=0
    for d in 10 20 30 40 50 60 70 80 90 100; do
      if [ ! -f "vae_checkpoints/cifar10_resnet_modernloss_d${d}/vae.pt" ]; then
        missing=1
      fi
    done
    if [ "${missing}" -eq 0 ]; then
      return 0
    fi
    echo "[$(date)] waiting for CIFAR-10 VAE d10-100 checkpoints"
    sleep 300
  done
}

wait_for_subset() {
  while [ ! -f "${SUBSET}" ]; do
    echo "[$(date)] waiting for CIFAR-10 diverse subset ${SUBSET}"
    sleep 300
  done
}

is_complete() {
  local out="$1"
  python3 - "$out" <<'PY'
import json
import math
import sys
from pathlib import Path

metrics = Path(sys.argv[1]) / "metrics.jsonl"
last = None
if metrics.exists():
    for line in metrics.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "step" in row:
            last = row
if last is None or last.get("step", 0) < 5_000_000:
    sys.exit(1)
vals = [last.get("train_loss_step"), last.get("train_loss")]
sys.exit(0 if all(v is not None and math.isfinite(v) for v in vals) else 1)
PY
}

run_d() {
  local seed="$1"
  local gpu="$2"
  local d="$3"
  local out="${OUT_ROOT}/seed${seed}/d${d}"
  mkdir -p "${out}"
  if is_complete "${out}"; then
    echo "[$(date)] skipping CIFAR-10 seed=${seed} d=${d}; already complete"
    return 0
  fi
  echo "[$(date)] starting CIFAR-10 seed=${seed} d=${d} on gpu ${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" python3 run_cifar_diverse_diffusion.py \
    --vae_checkpoint "vae_checkpoints/cifar10_resnet_modernloss_d${d}/vae.pt" \
    --out "${out}" \
    --subset_path "${SUBSET}" \
    --n_train 1000 \
    --total_steps 5000000 \
    --eval_interval 10000 \
    --mem_interval 100000 \
    --fid_interval 100000 \
    --fid_n_real 1000 \
    --fid_n_gen 10000 \
    --n_gen_samples 10000 \
    --n_sde_steps 500 \
    --batch_size 512 \
    --hidden 1024 \
    --depth 5 \
    --optimizer sgd \
    --lr 0.001 \
    --momentum 0.80 \
    --seed "${seed}" \
    > "${out}/runner.log" 2>&1
  echo "[$(date)] finished CIFAR-10 seed=${seed} d=${d}"
}

run_seed() {
  local seed="$1"
  echo "[$(date)] launching packed CIFAR-10 seed=${seed} on GPUs 0/1"
  run_d "${seed}" 0 10 &
  run_d "${seed}" 0 20 &
  run_d "${seed}" 0 30 &
  run_d "${seed}" 0 40 &
  run_d "${seed}" 0 50 &

  run_d "${seed}" 1 60 &
  run_d "${seed}" 1 70 &
  run_d "${seed}" 1 80 &
  run_d "${seed}" 1 90 &
  run_d "${seed}" 1 100 &

  wait
  echo "[$(date)] complete packed CIFAR-10 seed=${seed}"
}

wait_for_subset
wait_for_cifar_vaes

for seed in 42 43 44 45 46; do
  run_seed "${seed}"
done

echo "[$(date)] all CIFAR-10 seeds 42-46 complete"
