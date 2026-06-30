"""
Experiment v2 Track 2: RFNN (mechanistic)

RFNN score model with frozen W, learned A, trained at fixed t.
Computes eigenvalue spectrum of feature correlation matrix U.
Same data generation as Track 1.
"""

import math
import json
import os
import time
import argparse
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Import shared code from experiment_v2
from experiment_v2 import (
    generate_data, generate_test_samples, true_score,
    precompute_score_params, get_device, opt_step,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class RFNNConfig:
    # Data
    d_intrinsic: int = 5
    d_latent: int = 20
    k: int = 10
    sigma_noise: float = 0.5
    sigma_signal: float = 1.0
    scale: float = 3.0
    n: int = 500

    # RFNN model
    p_ratio: int = 64               # p = p_ratio * d_latent (matches Bonnaire)
    t_fixed: float = 0.01

    # Training (lr auto-scaled like Bonnaire: 0.01 * d / delta_t)
    lr: float = -1                  # auto
    momentum: float = 0.0
    total_steps: int = 2000000
    eval_interval: int = 5000

    # Eigenvalue analysis
    n_noise_samples: int = 50       # MC samples for U computation

    # Run
    seed: int = 42
    results_dir: str = "results_rfnn"


# ---------------------------------------------------------------------------
# RFNN Score Model
# ---------------------------------------------------------------------------

class RFNNScore(nn.Module):
    def __init__(self, d_latent, p, t_fixed):
        super().__init__()
        self.d = d_latent
        self.p = p
        self.t_fixed = t_fixed
        self.register_buffer('W', torch.randn(p, d_latent) / math.sqrt(d_latent))
        self.A = nn.Parameter(torch.zeros(d_latent, p))

    def forward(self, x_t):
        features = torch.tanh(x_t @ self.W.T)  # (batch, p)
        return features @ self.A.T / math.sqrt(self.p)


# ---------------------------------------------------------------------------
# Training step (fixed t, full-batch)
# ---------------------------------------------------------------------------

def train_step_rfnn(model, data, optimizer, device):
    t = model.t_fixed
    delta_t = 1 - math.exp(-2 * t)
    e_neg_t = math.exp(-t)

    noise = torch.randn_like(data)
    x_t = e_neg_t * data + math.sqrt(delta_t) * noise

    n, d = data.shape
    pred_score = model(x_t)
    residual = math.sqrt(delta_t) * pred_score + noise
    loss = (residual ** 2).sum() / (d * n)  # Bonnaire scaling

    optimizer.zero_grad()
    loss.backward()
    opt_step(optimizer, device)
    return loss.item()


# ---------------------------------------------------------------------------
# Compute U(t) and eigenvalues
# ---------------------------------------------------------------------------

def compute_U(W, data, t, n_noise_samples=50):
    """Feature correlation matrix U = (1/n) sum E[phi(x_t) phi(x_t)^T]"""
    p, d = W.shape
    n = data.shape[0]
    delta_t = 1 - math.exp(-2 * t)
    e_neg_t = math.exp(-t)

    U = torch.zeros(p, p)
    for _ in range(n_noise_samples):
        noise = torch.randn(n, d)
        x_t = e_neg_t * data.cpu() + math.sqrt(delta_t) * noise
        phi = torch.tanh(x_t @ W.cpu().T)  # (n, p)
        U += phi.T @ phi / n
    U /= n_noise_samples
    return U


def analyze_spectrum(U):
    """Eigendecompose U and return sorted eigenvalues."""
    eigenvalues = torch.linalg.eigvalsh(U)
    return eigenvalues.flip(0).cpu().numpy()  # descending


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_rfnn(model, train_data_dev, train_data_cpu, means, Q, config, device,
                  sigma_t_inv=None, log_det_sigma_t=None,
                  test_data_fixed=None, test_noise_fixed=None, train_dists=None):
    d = config.d_latent
    t = config.t_fixed
    results = {}

    with torch.no_grad():
        e_neg_t = math.exp(-t)
        delta_t = 1 - math.exp(-2 * t)

        # Train loss
        noise = torch.randn_like(train_data_dev)
        x_t = e_neg_t * train_data_dev + math.sqrt(delta_t) * noise
        pred = model(x_t)
        results['train_loss'] = (
            (math.sqrt(delta_t) * pred + noise) ** 2
        ).sum(-1).mean().item() / d

        # Test loss (fixed test data)
        test_dev = test_data_fixed.to(device)
        test_noise_dev = test_noise_fixed.to(device)
        x_t_test = e_neg_t * test_dev + math.sqrt(delta_t) * test_noise_dev
        pred_test = model(x_t_test)
        results['test_loss'] = (
            (math.sqrt(delta_t) * pred_test + test_noise_dev) ** 2
        ).sum(-1).mean().item() / d

        results['gen_gap'] = results['test_loss'] - results['train_loss']

        # Score error
        means_dev = means.to(device)
        si = sigma_t_inv.to(device) if sigma_t_inv is not None else None
        ld = log_det_sigma_t.to(device) if log_det_sigma_t is not None else None
        true_s = true_score(x_t_test, t, means_dev, sigma_t_inv=si, log_det_sigma_t=ld)
        pred_score_test = model(x_t_test)
        results['score_error'] = (
            (pred_score_test - true_s) ** 2
        ).sum(-1).mean().item() / d

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_experiment(config: RFNNConfig):
    device = get_device()
    print(f"Device: {device}")
    print(f"RFNN: d_intrinsic={config.d_intrinsic}, d_latent={config.d_latent}, "
          f"n={config.n}, p={config.p_ratio * config.d_latent}, t_fixed={config.t_fixed}")

    save_dir = Path(config.results_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    with open(save_dir / "config.json", "w") as f:
        json.dump(asdict(config), f, indent=2)

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Generate data
    data, labels, means, Q = generate_data(
        config.n, config.d_intrinsic, config.d_latent,
        config.k, config.sigma_noise, config.scale, config.seed,
        sigma_signal=config.sigma_signal,
    )
    print(f"Data shape: {data.shape}")

    # Precompute score params
    sigma_t_inv, log_det_sigma_t = precompute_score_params(
        config.d_intrinsic, config.d_latent, Q, config.sigma_noise, config.t_fixed,
        sigma_signal=config.sigma_signal,
    )

    # Fixed test data
    test_data_fixed = generate_test_samples(
        means.cpu(), config.k, config.d_intrinsic, config.d_latent,
        Q, config.sigma_noise, n=2048, seed=9999,
        sigma_signal=config.sigma_signal,
    )
    test_noise_fixed = torch.randn_like(test_data_fixed)

    # Precompute train NN dists
    train_dists = torch.cdist(data, data)
    train_dists.fill_diagonal_(float('inf'))

    # Diffusion params
    t = config.t_fixed
    delta_t = 1 - math.exp(-2 * t)
    e_neg_t = math.exp(-t)

    # Model
    p = config.p_ratio * config.d_latent
    model = RFNNScore(config.d_latent, p, config.t_fixed).to(device)
    n_params = config.d_latent * p  # only A is learned

    # Learning rate: Bonnaire scaling
    lr = config.lr if config.lr > 0 else 0.01 * config.d_latent / delta_t
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=config.momentum)
    print(f"p={p}, learnable params: {n_params:,}, lr={lr:.2f}")

    # Compute U eigenvalues BEFORE training
    print("Computing U eigenvalues (pre-training)...")
    U = compute_U(model.W, data, config.t_fixed, config.n_noise_samples)
    eigenvalues_pre = analyze_spectrum(U)
    np.save(save_dir / "eigenvalues_pre.npy", eigenvalues_pre)
    print(f"  Top 10 eigenvalues: {eigenvalues_pre[:10].round(3)}")
    print(f"  Bottom 10 eigenvalues: {eigenvalues_pre[-10:].round(3)}")

    # Training
    data_dev = data.to(device)
    metrics_file = open(save_dir / "metrics.jsonl", "w")
    t0 = time.time()

    for step in range(1, config.total_steps + 1):
        loss = train_step_rfnn(model, data_dev, optimizer, device)

        if step % config.eval_interval == 0 or step == 1:
            model.eval()
            eval_results = evaluate_rfnn(
                model, data_dev, data, means, Q, config, device,
                sigma_t_inv, log_det_sigma_t,
                test_data_fixed, test_noise_fixed, train_dists,
            )
            model.train()

            wall_time = time.time() - t0
            tau = step * lr  # Bonnaire rescaled time
            batch_sz = config.n  # full batch
            metrics = {
                'step': step,
                'tau': tau,
                'wall_time': wall_time,
                'epochs': step,
                'total_flops': step * 6 * n_params * batch_sz,
                'train_loss_step': loss,
                **eval_results,
            }

            metrics_file.write(json.dumps(metrics) + "\n")
            metrics_file.flush()

            print(f"  step {step:6d} | {wall_time:6.0f}s | "
                  f"train={eval_results['train_loss']:.4f} | "
                  f"test={eval_results['test_loss']:.4f} | "
                  f"score_err={eval_results['score_error']:.4f} | "
                  f"gap={eval_results['gen_gap']:.4f}",
                  flush=True)

    metrics_file.close()

    # Compute U eigenvalues AFTER training
    print("Computing U eigenvalues (post-training)...")
    U_post = compute_U(model.W, data, config.t_fixed, config.n_noise_samples)
    eigenvalues_post = analyze_spectrum(U_post)
    np.save(save_dir / "eigenvalues_post.npy", eigenvalues_post)

    print("Done!")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--d_latent", type=int, nargs="+", default=None)
    parser.add_argument("--d_intrinsic", type=int, nargs="+", default=None)
    parser.add_argument("--n", type=int, nargs="+", default=None)
    parser.add_argument("--p_ratio", type=int, default=64)
    parser.add_argument("--sigma_noise", type=float, default=0.5)
    parser.add_argument("--sigma_signal", type=float, default=1.0)
    parser.add_argument("--scale", type=float, default=3.0)
    parser.add_argument("--steps", type=int, default=300000)
    parser.add_argument("--eval_interval", type=int, default=1000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--base_dir", type=str, default="results_rfnn")
    args = parser.parse_args()

    d_latents = args.d_latent or [20]
    d_intrinsics = args.d_intrinsic or [5]
    ns = args.n or [500]

    for d_intr in d_intrinsics:
        for d_lat in d_latents:
            if d_lat < d_intr:
                print(f"Skipping d_latent={d_lat} < d_intrinsic={d_intr}")
                continue
            for n_val in ns:
                for seed in args.seeds:
                    config = RFNNConfig(
                        d_intrinsic=d_intr,
                        d_latent=d_lat,
                        n=n_val,
                        p_ratio=args.p_ratio,
                        sigma_noise=args.sigma_noise,
                        sigma_signal=args.sigma_signal,
                        scale=args.scale,
                        total_steps=args.steps,
                        eval_interval=args.eval_interval,
                        seed=seed,
                        results_dir=f"{args.base_dir}/di{d_intr}_d{d_lat}_n{n_val}_s{seed}",
                    )
                    print(f"\n{'='*60}")
                    print(f"RFNN: d_intrinsic={d_intr}, d_latent={d_lat}, n={n_val}, p={d_lat * args.p_ratio}, seed={seed}")
                    print(f"{'='*60}")
                    run_experiment(config)


if __name__ == "__main__":
    main()
