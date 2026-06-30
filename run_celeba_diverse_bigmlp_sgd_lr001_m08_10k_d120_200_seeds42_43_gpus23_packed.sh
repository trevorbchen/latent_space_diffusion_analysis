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
  local vae="vae_checkpoints/celeba_resnet_modernloss_d${d}/vae.pt"
  if [ ! -f "${vae}" ]; then
    echo "[$(date)] missing VAE checkpoint for d=${d}: ${vae}" >&2
    return 1
  fi
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
    --vae_checkpoint "${vae}" \
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

# Seed 42 on GPU2, seed 43 on GPU3. These will be skipped by the later watcher.
for d in 120 140 160 180 200; do
  run_d 42 2 "${d}" &
done

for d in 120 140 160 180 200; do
  run_d 43 3 "${d}" &
done

wait
echo "[$(date)] complete high-d CelebA seeds 42 and 43 on GPUs 2/3"
