"""Multi-GPU runner: dispatch a list of run_experiment.py invocations
across all visible CUDA devices, one config per GPU at a time.

The configs file is a JSON (or YAML if pyyaml is installed) list. Each
entry is a dict of CLI flags for run_experiment.py, with required keys
`out` and `data`. Example (configs/exp_synth_d_latent_sweep.json):

    [
      {"data": "synthetic", "model": "mlp",
       "d_intrinsic": 5, "d_latent": 5,  "out": "results/synth_mlp_d5"},
      {"data": "synthetic", "model": "mlp",
       "d_intrinsic": 5, "d_latent": 8,  "out": "results/synth_mlp_d8"},
      ...
    ]

Run:
    python runner.py configs/exp_synth_d_latent_sweep.json
    python runner.py configs/...json --gpus 0,1,2,3
    python runner.py configs/...json --dry_run
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

try:
    import yaml  # type: ignore
except Exception:
    yaml = None


def _load_configs(path: Path) -> list[dict]:
    text = path.read_text()
    if path.suffix in ('.yaml', '.yml'):
        if yaml is None:
            raise SystemExit("PyYAML not installed; use a .json configs file.")
        cfgs = yaml.safe_load(text)
    else:
        cfgs = json.loads(text)
    if not isinstance(cfgs, list):
        raise SystemExit(f"Configs file {path} must contain a top-level list.")
    return cfgs


def _detect_gpus(spec: str | None) -> list[int]:
    if spec is not None:
        return [int(x) for x in spec.split(',') if x.strip()]
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible:
        return [int(x) for x in visible.split(',') if x.strip()]
    try:
        import torch
        return list(range(torch.cuda.device_count())) if torch.cuda.is_available() else [-1]
    except Exception:
        return [-1]


def _format_cmd(cfg: dict) -> list[str]:
    cmd = [sys.executable, 'run_experiment.py']
    for k, v in cfg.items():
        if v is None or v is False:
            continue
        flag = f"--{k}"
        if v is True:
            cmd.append(flag)
        elif isinstance(v, list):
            cmd.append(flag); cmd.extend(str(x) for x in v)
        else:
            cmd += [flag, str(v)]
    return cmd


def _run_one(cfg: dict, gpu_id: int, log_root: Path,
             dry_run: bool) -> tuple[int, dict]:
    out_dir = Path(cfg.get('out', '.'))
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / 'runner.log'

    env = os.environ.copy()
    if gpu_id >= 0:
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    cmd = _format_cmd(cfg)
    if dry_run:
        print(f"[gpu {gpu_id}] DRY: {' '.join(cmd)}")
        return 0, cfg

    t0 = time.time()
    with open(log_path, 'w') as logf:
        logf.write(f"# gpu_id={gpu_id}\n# cmd={' '.join(cmd)}\n")
        logf.flush()
        proc = subprocess.run(
            cmd, env=env, stdout=logf, stderr=subprocess.STDOUT,
            cwd=Path(__file__).parent,
        )
    elapsed = time.time() - t0
    status = 'ok' if proc.returncode == 0 else f'FAIL({proc.returncode})'
    print(f"[gpu {gpu_id}] {status} in {elapsed/60:.1f}m  out={out_dir}")
    return proc.returncode, cfg


def main():
    p = argparse.ArgumentParser()
    p.add_argument('configs', type=Path)
    p.add_argument('--gpus', type=str, default=None,
                   help='Comma-separated GPU IDs (default: all visible).')
    p.add_argument('--log_root', type=Path, default=Path('runner_logs'))
    p.add_argument('--dry_run', action='store_true')
    args = p.parse_args()

    cfgs = _load_configs(args.configs)
    gpus = _detect_gpus(args.gpus)
    print(f"Dispatching {len(cfgs)} configs across GPUs {gpus}")

    args.log_root.mkdir(parents=True, exist_ok=True)
    work: queue.Queue = queue.Queue()
    for c in cfgs:
        work.put(c)

    failures: list[dict] = []
    lock = threading.Lock()

    def worker(gpu_id: int):
        while True:
            try:
                cfg = work.get_nowait()
            except queue.Empty:
                return
            rc, cfg = _run_one(cfg, gpu_id, args.log_root, args.dry_run)
            if rc != 0:
                with lock:
                    failures.append(cfg)
            work.task_done()

    threads = [threading.Thread(target=worker, args=(g,), daemon=True)
               for g in gpus]
    for th in threads: th.start()
    for th in threads: th.join()

    if failures:
        print(f"\n{len(failures)} run(s) failed:")
        for cfg in failures:
            print(f"  - {cfg.get('out', cfg)}")
        sys.exit(1)
    print("All runs complete.")


if __name__ == '__main__':
    main()
