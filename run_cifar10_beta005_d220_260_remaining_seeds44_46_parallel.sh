#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/latent_space_diffusion_analysis/code/v3"
mkdir -p runner_logs

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2

OUT_ROOT="results/cifar10_beta005_diverse1k_bigmlp_sgd_lr001_m08_10k_5m"
SUBSET="diagnostics/cifar10_diverse_1k_mean_plus_sd/subset_indices.json"

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
    echo "[$(date)] skipping CIFAR beta005 seed=${seed} d=${d}; already complete"
    return 0
  fi
  if pgrep -f "run_cifar_diverse_diffusion.py .*seed${seed}/d${d}( |$)" >/dev/null; then
    echo "[$(date)] waiting: CIFAR beta005 seed=${seed} d=${d} already running"
    while pgrep -f "run_cifar_diverse_diffusion.py .*seed${seed}/d${d}( |$)" >/dev/null; do
      sleep 120
    done
    if is_complete "${out}"; then
      echo "[$(date)] completed elsewhere: CIFAR beta005 seed=${seed} d=${d}"
      return 0
    fi
  fi
  echo "[$(date)] starting CIFAR beta005 seed=${seed} d=${d} on gpu ${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" python3 run_cifar_diverse_diffusion.py \
    --vae_checkpoint "vae_checkpoints/cifar10_resnet_beta005_d${d}/vae.pt" \
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
  echo "[$(date)] finished CIFAR beta005 seed=${seed} d=${d}"
}

run_seed_on_gpu() {
  local seed="$1"
  local gpu="$2"
  echo "[$(date)] launching CIFAR beta005 high-d extension seed=${seed} on GPU ${gpu}"
  run_d "${seed}" "${gpu}" 220 &
  run_d "${seed}" "${gpu}" 240 &
  run_d "${seed}" "${gpu}" 260 &
  wait
  echo "[$(date)] complete CIFAR beta005 high-d extension seed=${seed}"
}

for d in 220 240 260; do
  if [ ! -f "vae_checkpoints/cifar10_resnet_beta005_d${d}/vae.pt" ]; then
    echo "Missing required VAE checkpoint: d=${d}" >&2
    exit 1
  fi
done

run_seed_on_gpu 44 0 &
run_seed_on_gpu 45 2 &
run_seed_on_gpu 46 3 &

wait

echo "[$(date)] all remaining CIFAR beta005 d220/d240/d260 seeds 44-46 complete"
