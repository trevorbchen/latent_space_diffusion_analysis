#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/latent_space_diffusion_analysis/code/v3"
mkdir -p runner_logs

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2

is_complete() {
  local out="$1"
  python3 - "$out" <<'PY'
import json
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
            last = row["step"]
sys.exit(0 if last is not None and last >= 5_000_000 else 1)
PY
}

run_d() {
  local gpu="$1"
  local d="$2"
  local out="results/celeba_hq_diverse1k_resnet_modernloss_5m/d${d}"
  mkdir -p "${out}"
  if is_complete "${out}"; then
    echo "[$(date)] skipping d=${d}; already complete"
    return 0
  fi
  if pgrep -af "run_celeba_diverse_diffusion.py .*d${d} " >/dev/null; then
    echo "[$(date)] skipping d=${d}; already running"
    return 0
  fi
  echo "[$(date)] starting packed CelebA-HQ diverse1k diffusion d=${d} on gpu ${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" python3 run_celeba_diverse_diffusion.py \
    --vae_checkpoint "vae_checkpoints/celeba_resnet_modernloss_d${d}/vae.pt" \
    --out "${out}" \
    --subset_path "diagnostics/celeba_diverse_1k_mean_plus_sd/subset_indices.json" \
    --n_train 1000 \
    --total_steps 5000000 \
    --eval_interval 10000 \
    --mem_interval 100000 \
    --fid_interval 100000 \
    --fid_n_real 1000 \
    --fid_n_gen 1000 \
    --n_gen_samples 1000 \
    --n_sde_steps 500 \
    --batch_size 256 \
    --hidden 256 \
    --lr 1e-4 \
    --seed 42 \
    > "${out}/runner.log" 2>&1
  echo "[$(date)] finished packed CelebA-HQ diverse1k diffusion d=${d}"
}

run_d 0 50 &
run_d 1 60 &
run_d 2 70 &
run_d 3 80 &
run_d 0 90 &
run_d 1 100 &

wait
echo "[$(date)] remaining packed CelebA-HQ diverse1k diffusion jobs complete"
