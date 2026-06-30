#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/latent_space_diffusion_analysis/code/v3"
mkdir -p runner_logs

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2

OUT_ROOT="results/celeba_diverse1k_bigmlp_sgd_lr001_m08_10k_5m"
SUBSET="diagnostics/celeba_diverse_1k_mean_plus_sd/subset_indices.json"

wait_for_existing_diffusion_queue() {
  while pgrep -f "run_celeba_diverse_bigmlp_sgd_lr001_m08_10k_gpus01_seeds43_46_packed.sh" >/dev/null \
     || pgrep -f "run_celeba_diverse_diffusion.py --vae_checkpoint vae_checkpoints/celeba_resnet_modernloss_d" >/dev/null; do
    echo "[$(date)] waiting for current d10-100 CelebA diffusion queue to finish"
    sleep 300
  done
}

wait_for_vaes() {
  while true; do
    local missing=()
    for d in 120 140 160 180 200; do
      if [ ! -f "vae_checkpoints/celeba_resnet_modernloss_d${d}/vae.pt" ]; then
        missing+=("${d}")
      fi
    done
    if [ "${#missing[@]}" -eq 0 ]; then
      echo "[$(date)] all high-d VAEs are present"
      return 0
    fi
    echo "[$(date)] waiting for high-d VAEs: missing d=${missing[*]}"
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
  local out
  if [ "${seed}" = "42" ]; then
    out="${OUT_ROOT}/d${d}"
  else
    out="${OUT_ROOT}/seed${seed}/d${d}"
  fi
  mkdir -p "${out}"
  if is_complete "${out}"; then
    echo "[$(date)] skipping seed=${seed} d=${d}; already complete"
    return 0
  fi
  echo "[$(date)] starting high-d CelebA seed=${seed} d=${d} on gpu ${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" python3 run_celeba_diverse_diffusion.py \
    --vae_checkpoint "vae_checkpoints/celeba_resnet_modernloss_d${d}/vae.pt" \
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
  echo "[$(date)] finished high-d CelebA seed=${seed} d=${d}"
}

run_seed() {
  local seed="$1"
  echo "[$(date)] launching high-d CelebA seed=${seed} on GPUs 0/1"
  run_d "${seed}" 0 120 &
  run_d "${seed}" 0 140 &
  run_d "${seed}" 0 160 &
  run_d "${seed}" 1 180 &
  run_d "${seed}" 1 200 &
  wait
  echo "[$(date)] complete high-d CelebA seed=${seed}"
}

wait_for_vaes
wait_for_existing_diffusion_queue

run_seed 42
run_seed 43
run_seed 44
run_seed 45
run_seed 46

echo "[$(date)] all high-d CelebA d120-200 seeds complete"
