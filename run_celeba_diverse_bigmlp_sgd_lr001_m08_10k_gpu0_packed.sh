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
GPU=0

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
  local d="$1"
  local out="${OUT_ROOT}/d${d}"
  mkdir -p "${out}"
  if is_complete "${out}"; then
    echo "[$(date)] skipping d=${d}; already complete"
    return 0
  fi
  echo "[$(date)] starting CelebA diverse1k big-MLP SGD lr=0.001 m=0.80 d=${d} on gpu ${GPU}"
  CUDA_VISIBLE_DEVICES="${GPU}" python3 run_celeba_diverse_diffusion.py \
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
    --seed 42 \
    > "${out}/runner.log" 2>&1
  echo "[$(date)] finished CelebA diverse1k big-MLP SGD lr=0.001 m=0.80 d=${d}"
}

# Pack multiple independent d_latent runs onto one A100.
# Each run uses about 8-9GB, so four-at-a-time fits comfortably on an 80GB GPU.
run_d 10 &
run_d 20 &
run_d 30 &
run_d 40 &
wait

run_d 50 &
run_d 60 &
run_d 70 &
run_d 80 &
wait

run_d 90 &
run_d 100 &
wait

echo "[$(date)] all packed GPU0 CelebA diverse1k big-MLP SGD lr=0.001 m=0.80 jobs complete"
