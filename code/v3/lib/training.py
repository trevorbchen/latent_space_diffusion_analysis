"""Unified training loop for synthetic-and-real, MLP-and-RFNN runs.

The loop is organized around an EvalSuite that gathers callables of varying
cost; the trainer fires the cheap ones every `eval_interval`, the expensive
ones every `mem_interval` (memorization, FID, eigenvalue snapshots).

The trainer is data-agnostic: it accepts a `train_data` tensor and an
EvalSuite. The caller wires up the suite differently for synthetic
(analytic score error, NN ratio in latent space) vs real (FID against
test images, NN ratio in pixel space via VAE decoder).
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import torch
import torch.nn as nn

from .diffusion import forward_noise, score_loss


# ---------------------------------------------------------------------------
# Eval suite
# ---------------------------------------------------------------------------

@dataclass
class EvalSpec:
    name: str
    fn: Callable[[nn.Module, int], dict[str, Any]]
    interval: int                # in training steps
    needs_eval_mode: bool = True


@dataclass
class EvalSuite:
    """A bundle of EvalSpec entries. Cadence is per-entry."""
    specs: list[EvalSpec] = field(default_factory=list)

    def add(self, name: str, fn, interval: int, needs_eval_mode: bool = True):
        self.specs.append(EvalSpec(name, fn, interval, needs_eval_mode))

    def run(self, step: int, model: nn.Module) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for spec in self.specs:
            if step != 1 and step % spec.interval != 0:
                continue
            if spec.needs_eval_mode:
                model.eval()
            with torch.no_grad():
                result = spec.fn(model, step)
            if spec.needs_eval_mode:
                model.train()
            for k, v in result.items():
                out[k] = v
        return out


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    total_steps: int = 300_000
    eval_interval: int = 1_000          # cheap metrics
    mem_interval: int = 10_000          # mem fraction, FID
    eigvals_interval: int = 25_000      # RFNN U eigvals + bulk absorption
    batch_size: int = 256
    t_min: float = 0.01
    t_max: float = 3.0
    fixed_t: float | None = None         # set for RFNN training
    log_every: int = 1_000               # stdout cadence
    checkpoint_every: int = 0            # 0 = save at every eval; >0 = step interval


def train_loop(model: nn.Module,
               train_data: torch.Tensor,
               optimizer: torch.optim.Optimizer,
               *,
               cfg: TrainConfig,
               eval_suite: EvalSuite,
               device: torch.device,
               out_dir: Path,
               run_meta: dict[str, Any] | None = None) -> None:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / 'metrics.jsonl'
    metrics_file = open(metrics_path, 'w')
    if run_meta is not None:
        metrics_file.write(json.dumps({'event': 'meta', **run_meta}) + '\n')
        metrics_file.flush()

    train_data = train_data.to(device)
    n = train_data.shape[0]
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    t0 = time.time()
    is_full_batch = (cfg.batch_size >= n)

    for step in range(1, cfg.total_steps + 1):
        if is_full_batch:
            batch = train_data
        else:
            idx = torch.randint(0, n, (cfg.batch_size,), device=device)
            batch = train_data[idx]

        # Sample t
        if cfg.fixed_t is not None:
            t = cfg.fixed_t
        else:
            t = (torch.rand(batch.shape[0], device=device)
                 * (cfg.t_max - cfg.t_min) + cfg.t_min)

        x_t, noise, _, sqrt_dt = forward_noise(batch, t)
        if isinstance(t, float):
            t_arg = torch.full((batch.shape[0],), float(t), device=device)
        else:
            t_arg = t
        pred = model(x_t, t_arg)
        loss = score_loss(pred, noise, sqrt_dt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step == 1 or step % cfg.eval_interval == 0:
            eval_results = eval_suite.run(step, model)
            wall = time.time() - t0
            row = {
                'step': step,
                'wall_time': wall,
                'train_loss_step': loss.item(),
                'n_params': n_params,
                'total_flops': step * 6 * n_params * batch.shape[0],
                **eval_results,
            }
            metrics_file.write(json.dumps(row) + '\n')
            metrics_file.flush()
            if step == 1 or step % cfg.log_every == 0:
                _stdout_summary(step, wall, row)

            # Periodic model snapshot. Saves to last_model.pt (overwritten each
            # time) so a killed run leaves behind a usable checkpoint at the
            # most recent eval boundary. Default cadence (checkpoint_every=0)
            # is "every eval"; set >0 to checkpoint less often if disk I/O
            # matters.
            should_ckpt = (
                cfg.checkpoint_every == 0
                or step % cfg.checkpoint_every == 0
            )
            if should_ckpt:
                tmp = out_dir / 'last_model.pt.tmp'
                torch.save({
                    'step': step,
                    'state_dict': {k: v.detach().cpu()
                                   for k, v in model.state_dict().items()},
                }, tmp)
                tmp.replace(out_dir / 'last_model.pt')   # atomic on POSIX

    metrics_file.close()


def _stdout_summary(step: int, wall: float, row: dict) -> None:
    parts = [f"step {step:7d}", f"{wall:6.0f}s"]
    for key in ('train_loss', 'test_loss', 'score_error',
                'memorization_fraction', 'fid'):
        if key in row:
            parts.append(f"{key}={row[key]:.4f}")
    print('  ' + ' | '.join(parts), flush=True)
