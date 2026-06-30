#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/latent_space_diffusion_analysis/code/v3"
mkdir -p runner_logs

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2

OUT_ROOT="results/celeba_diverse1k_bigmlp_sgd_10k_5m"
SUBSET="diagnostics/celeba_diverse_1k_mean_plus_sd/subset_indices.json"

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

wait_for_synthetic_mlp() {
  while pgrep -u "$USER" -f "run_mlp_multiseed.py.*exp2_mlp_dlat_sn05_5m_bonnaire_hidden256" >/dev/null; do
    echo "[$(date)] waiting for corrected synthetic MLP rerun to finish before starting CelebA big-MLP jobs"
    sleep 300
  done
}

run_d() {
  local gpu="$1"
  local d="$2"
  local out="${OUT_ROOT}/d${d}"
  mkdir -p "${out}"
  if is_complete "${out}"; then
    echo "[$(date)] skipping d=${d}; already complete"
    return 0
  fi
  echo "[$(date)] starting CelebA diverse1k big-MLP SGD d=${d} on gpu ${gpu}"
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
    --lr 0.01 \
    --momentum 0.95 \
    --seed 42 \
    > "${out}/runner.log" 2>&1
  echo "[$(date)] finished CelebA diverse1k big-MLP SGD d=${d}"
}

wait_for_synthetic_mlp

run_d 2 10 &
run_d 3 20 &
wait

run_d 2 30 &
run_d 3 40 &
wait

run_d 2 50 &
run_d 3 60 &
wait

run_d 2 70 &
run_d 3 80 &
wait

run_d 2 90 &
run_d 3 100 &
wait

echo "[$(date)] all CelebA diverse1k big-MLP SGD 10k-sample jobs complete"
