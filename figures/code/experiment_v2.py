"""
Experiment v2: Latent dimensionality and the generalization-memorization transition.

No VAE. Diffusion directly on Gaussian mixture data with known intrinsic dimension.
Analytic true score for clean tau_gen measurement.
OU process (continuous time) diffusion.
"""

import math
import json
import os
import time
import argparse
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    # Data
    d_intrinsic: int = 5
    d_latent: int = 20
    k: int = 10                     # number of clusters
    sigma_noise: float = 0.5        # noise in null-space dims
    sigma_signal: float = 1.0       # noise in signal dims (around cluster centers)
    scale: float = 3.0              # cluster center spread
    n: int = 500

    # Model
    hidden: int = 256
    n_freq: int = 32

    # Training
    lr: float = 1e-4
    batch_size: int = 256
    total_steps: int = 300000
    t_min: float = 0.01
    t_max: float = 3.0

    # Eval
    eval_interval: int = 1000
    n_gen_samples: int = 10000
    n_sde_steps: int = 500
    t_eval: float = 0.1             # fixed t for score error

    # Run
    seed: int = 42
    results_dir: str = "results_v2"


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def get_device():
    try:
        import torch_xla.core.xla_model as xm
        return xm.xla_device()
    except ImportError:
        pass
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def opt_step(optimizer, device):
    optimizer.step()
    if 'xla' in str(device):
        import torch_xla.core.xla_model as xm
        xm.mark_step()


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def generate_data(n, d_intrinsic, d_latent, k=10, sigma_noise=10.0,
                  scale=100.0, seed=42, sigma_signal=30.0):
    rng = np.random.default_rng(seed)

    # Orthogonal equal-magnitude cluster centers in d_intrinsic dims
    # Use QR on random matrix to get orthogonal directions, scale to equal magnitude
    if k <= d_intrinsic:
        raw = rng.standard_normal((k, d_intrinsic))
        Q_centers, _ = np.linalg.qr(raw.T)  # (d_intrinsic, k)
        means_intrinsic = Q_centers.T * scale  # (k, d_intrinsic), each row has norm=scale
    else:
        # More clusters than dims: first d_intrinsic are orthogonal, rest random on sphere
        raw = rng.standard_normal((d_intrinsic, d_intrinsic))
        Q_centers, _ = np.linalg.qr(raw)
        means_intrinsic = np.zeros((k, d_intrinsic))
        means_intrinsic[:d_intrinsic] = Q_centers * scale
        for i in range(d_intrinsic, k):
            v = rng.standard_normal(d_intrinsic)
            v = v / np.linalg.norm(v) * scale
            means_intrinsic[i] = v

    means_full = np.zeros((k, d_latent))
    means_full[:, :d_intrinsic] = means_intrinsic

    # Sample assignments
    labels = rng.integers(0, k, size=n)
    data = np.zeros((n, d_latent))

    # Signal dimensions: sigma_signal variance around cluster centers
    data[:, :d_intrinsic] = (
        means_intrinsic[labels] + rng.standard_normal((n, d_intrinsic)) * sigma_signal
    )

    # Null-space dimensions: sigma_noise
    if d_latent > d_intrinsic:
        data[:, d_intrinsic:] = (
            rng.standard_normal((n, d_latent - d_intrinsic)) * sigma_noise
        )

    # Random orthogonal rotation
    Q, _ = np.linalg.qr(rng.standard_normal((d_latent, d_latent)))
    data = data @ Q.T
    means_full = means_full @ Q.T

    return (
        torch.tensor(data, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.long),
        torch.tensor(means_full, dtype=torch.float32),
        torch.tensor(Q, dtype=torch.float32),
    )


def generate_test_samples(means, k, d_intrinsic, d_latent, Q,
                          sigma_noise=10.0, n=2048, seed=None, sigma_signal=30.0):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, k, size=n)
    means_orig = means @ Q  # un-rotate
    data = np.zeros((n, d_latent))
    data[:, :d_intrinsic] = (
        means_orig.numpy()[labels, :d_intrinsic]
        + rng.standard_normal((n, d_intrinsic)) * sigma_signal
    )
    if d_latent > d_intrinsic:
        data[:, d_intrinsic:] = (
            rng.standard_normal((n, d_latent - d_intrinsic)) * sigma_noise
        )
    data = data @ Q.T.numpy()
    return torch.tensor(data, dtype=torch.float32)


# ---------------------------------------------------------------------------
# True score (analytic)
# ---------------------------------------------------------------------------

def true_score(x, t, means, sigma_data_inv=None, sigma_t_inv=None, log_det_sigma_t=None):
    """Analytically compute nabla log P_t(x) for Gaussian mixture.

    P_t(x) = (1/k) sum_j N(x; e^{-t} mu_j, Sigma_t)
    where Sigma_t = delta_t * I + e^{-2t} * Sigma_data

    If sigma_t_inv not provided, assumes Sigma_data = I (isotropic clusters).
    """
    delta_t = 1 - math.exp(-2 * t)
    e_neg_t = math.exp(-t)

    shifted = x.unsqueeze(1) - e_neg_t * means.unsqueeze(0)  # (batch, k, d)

    if sigma_t_inv is not None:
        # Anisotropic: use full precision matrix
        # shifted @ Sigma_t_inv gives (batch, k, d)
        shifted_prec = shifted @ sigma_t_inv  # (batch, k, d)
        log_w = -0.5 * (shifted * shifted_prec).sum(-1)  # (batch, k)
        if log_det_sigma_t is not None:
            log_w = log_w - 0.5 * log_det_sigma_t
        log_w = log_w - log_w.max(dim=1, keepdim=True).values
        w = torch.softmax(log_w, dim=1)
        score = -(w.unsqueeze(-1) * shifted_prec).sum(1)
    else:
        # Isotropic fallback: Sigma_t = I
        sigma_t = delta_t + math.exp(-2 * t)  # = 1
        log_w = -0.5 * (shifted ** 2).sum(-1) / sigma_t
        log_w = log_w - log_w.max(dim=1, keepdim=True).values
        w = torch.softmax(log_w, dim=1)
        score = (w.unsqueeze(-1) * (-shifted / sigma_t)).sum(1)

    return score


def precompute_score_params(d_intrinsic, d_latent, Q, sigma_noise, t, sigma_signal=30.0):
    """Precompute Sigma_t inverse and log det for true_score at time t."""
    delta_t = 1 - math.exp(-2 * t)
    e_neg_2t = math.exp(-2 * t)

    # Sigma_data in original coords: diag(sigma_signal^2,..., sigma_noise^2,...)
    sigma_data_diag = torch.ones(d_latent)
    sigma_data_diag[:d_intrinsic] = sigma_signal ** 2
    if d_latent > d_intrinsic:
        sigma_data_diag[d_intrinsic:] = sigma_noise ** 2

    # Sigma_t in original coords: delta_t * I + e^{-2t} * Sigma_data (diagonal)
    sigma_t_diag = delta_t + e_neg_2t * sigma_data_diag

    # In rotated coords: Sigma_t = Q diag(sigma_t_diag) Q^T
    # Sigma_t_inv = Q diag(1/sigma_t_diag) Q^T
    Q_t = Q.float()
    sigma_t_inv = Q_t.T @ torch.diag(1.0 / sigma_t_diag) @ Q_t
    log_det = torch.log(sigma_t_diag).sum()

    return sigma_t_inv, log_det


# ---------------------------------------------------------------------------
# MLP Score Network
# ---------------------------------------------------------------------------

class MLPScore(nn.Module):
    def __init__(self, d_latent, hidden=256, n_freq=32):
        super().__init__()
        self.n_freq = n_freq
        self.register_buffer(
            'freqs', torch.exp(torch.linspace(0, math.log(1000), n_freq))
        )
        d_input = d_latent + 2 * n_freq
        self.net = nn.Sequential(
            nn.Linear(d_input, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_latent),
        )

    def forward(self, x_t, t):
        if isinstance(t, (int, float)):
            t = torch.full((x_t.shape[0],), t, device=x_t.device)
        if t.dim() == 0:
            t = t.expand(x_t.shape[0])
        t_emb = torch.cat([
            torch.sin(t.unsqueeze(-1) * self.freqs),
            torch.cos(t.unsqueeze(-1) * self.freqs),
        ], dim=-1)
        inp = torch.cat([x_t, t_emb], dim=-1)
        return self.net(inp)


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def train_step(model, data, optimizer, device, t_min=0.01, t_max=3.0):
    batch_size = data.shape[0]
    t = torch.rand(batch_size, device=device) * (t_max - t_min) + t_min
    delta_t = 1 - torch.exp(-2 * t)
    e_neg_t = torch.exp(-t)

    noise = torch.randn_like(data)
    x_t = e_neg_t.unsqueeze(-1) * data + delta_t.sqrt().unsqueeze(-1) * noise

    pred_score = model(x_t, t)
    residual = delta_t.sqrt().unsqueeze(-1) * pred_score + noise
    loss = (residual ** 2).sum(dim=-1).mean()

    optimizer.zero_grad()
    loss.backward()
    opt_step(optimizer, device)
    return loss.item()


# ---------------------------------------------------------------------------
# Sample generation (Euler-Maruyama reverse SDE)
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_samples(model, n_gen, d, n_steps=500, t_max=3.0, t_min=0.01,
                     device=torch.device('cpu')):
    dt = (t_max - t_min) / n_steps
    x = torch.randn(n_gen, d, device=device)

    for i in range(n_steps):
        t_curr = t_max - i * dt
        t_batch = torch.full((n_gen,), t_curr, device=device)
        score = model(x, t_batch)
        noise = torch.randn_like(x) * math.sqrt(2 * dt)
        x = x + (x + 2 * score) * dt + noise

    return x


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def compute_mmd(x, y, bandwidth=1.0):
    def kernel(a, b):
        dists = torch.cdist(a, b) ** 2
        return torch.exp(-dists / (2 * bandwidth ** 2))
    xx = kernel(x, x).mean()
    yy = kernel(y, y).mean()
    xy = kernel(x, y).mean()
    return (xx + yy - 2 * xy).item()


def evaluate(model, train_data_dev, train_data_cpu, means, Q, config, device,
             sigma_t_inv=None, log_det_sigma_t=None,
             test_data_fixed=None, test_noise_fixed=None, train_dists=None):
    d = config.d_latent
    t_eval = config.t_eval
    results = {}

    with torch.no_grad():
        e_neg_t = math.exp(-t_eval)
        delta_t = 1 - math.exp(-2 * t_eval)

        # --- Train loss (fixed noise per step for consistency) ---
        train_noise = torch.randn_like(train_data_dev)
        x_t_train = e_neg_t * train_data_dev + math.sqrt(delta_t) * train_noise
        t_batch = torch.full((len(train_data_dev),), t_eval, device=device)
        pred_train = model(x_t_train, t_batch)
        results['train_loss'] = (
            (math.sqrt(delta_t) * pred_train + train_noise) ** 2
        ).sum(-1).mean().item() / d

        # --- Test loss (fixed test data + noise across all eval steps) ---
        test_dev = test_data_fixed.to(device)
        test_noise_dev = test_noise_fixed.to(device)
        x_t_test = e_neg_t * test_dev + math.sqrt(delta_t) * test_noise_dev
        t_batch_test = torch.full((len(test_dev),), t_eval, device=device)
        pred_test = model(x_t_test, t_batch_test)
        results['test_loss'] = (
            (math.sqrt(delta_t) * pred_test + test_noise_dev) ** 2
        ).sum(-1).mean().item() / d

        results['gen_gap'] = results['test_loss'] - results['train_loss']

        # --- Score error (E_score) on fixed test points ---
        means_dev = means.to(device)
        si = sigma_t_inv.to(device) if sigma_t_inv is not None else None
        ld = log_det_sigma_t.to(device) if log_det_sigma_t is not None else None
        true_s = true_score(x_t_test, t_eval, means_dev, sigma_t_inv=si, log_det_sigma_t=ld)
        pred_score_test = model(x_t_test, t_batch_test)
        results['score_error'] = (
            (pred_score_test - true_s) ** 2
        ).sum(-1).mean().item() / d

        # --- Generate samples for memorization metrics ---
        cpu_model = MLPScore(config.d_latent, config.hidden, config.n_freq)
        cpu_model.load_state_dict({k: v.cpu() for k, v in model.state_dict().items()})
        cpu_model.eval()

        generated = generate_samples(
            cpu_model, config.n_gen_samples, d,
            n_steps=config.n_sde_steps, t_max=config.t_max, t_min=config.t_min,
        )

        # NN ratio: d(gen, NN1_train) / d(NN1_train, NN2_train)
        dists_to_train = torch.cdist(generated, train_data_cpu)
        nn1_dists, nn1_idx = dists_to_train.min(dim=1)
        nn2_dists = train_dists[nn1_idx].min(dim=1).values

        nn_ratio = nn1_dists / (nn2_dists + 1e-10)
        results['mean_nn_ratio'] = nn_ratio.mean().item()
        results['memorization_fraction'] = (nn_ratio < 1/3).float().mean().item()

    return results


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def run_experiment(config: Config):
    device = get_device()
    print(f"Device: {device}")
    print(f"d_intrinsic={config.d_intrinsic}, d_latent={config.d_latent}, "
          f"n={config.n}, k={config.k}")

    save_dir = Path(config.results_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(save_dir / "config.json", "w") as f:
        json.dump(asdict(config), f, indent=2)

    # Fix seeds
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Generate data
    data, labels, means, Q = generate_data(
        config.n, config.d_intrinsic, config.d_latent,
        config.k, config.sigma_noise, config.scale, config.seed,
        sigma_signal=config.sigma_signal,
    )
    print(f"Data shape: {data.shape}")
    svs = torch.linalg.svdvals(data).numpy()
    print(f"Data SVD (top 10): {svs[:10].round(2)}")

    # Precompute score params for eval
    sigma_t_inv, log_det_sigma_t = precompute_score_params(
        config.d_intrinsic, config.d_latent, Q, config.sigma_noise, config.t_eval,
        sigma_signal=config.sigma_signal,
    )

    # Precompute fixed test data + noise for consistent eval across steps
    test_data_fixed = generate_test_samples(
        means.cpu(), config.k, config.d_intrinsic, config.d_latent,
        Q, config.sigma_noise, n=2048, seed=9999,
        sigma_signal=config.sigma_signal,
    )
    test_noise_fixed = torch.randn_like(test_data_fixed)

    # Precompute training NN distances (for memorization metric)
    train_dists = torch.cdist(data, data)
    train_dists.fill_diagonal_(float('inf'))

    # Move to device
    data_dev = data.to(device)
    means_dev = means.to(device)

    # Model
    model = MLPScore(config.d_latent, config.hidden, config.n_freq).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    # OVERWRITE metrics file (not append) to avoid duplicates
    metrics_file = open(save_dir / "metrics.jsonl", "w")

    t0 = time.time()

    for step in range(1, config.total_steps + 1):
        # Sample batch
        idx = torch.randint(0, config.n, (min(config.batch_size, config.n),))
        batch = data_dev[idx]

        loss = train_step(model, batch, optimizer, device,
                         config.t_min, config.t_max)

        if step % config.eval_interval == 0 or step == 1:
            model.eval()
            eval_results = evaluate(
                model, data_dev, data, means, Q, config, device,
                sigma_t_inv, log_det_sigma_t,
                test_data_fixed, test_noise_fixed, train_dists,
            )
            model.train()

            wall_time = time.time() - t0
            batch_sz = min(config.batch_size, config.n)
            samples_seen = step * batch_sz
            epochs = samples_seen / config.n
            flops_per_step = 6 * n_params * batch_sz  # ~2x fwd + 4x bwd
            total_flops = step * flops_per_step
            metrics = {
                'step': step,
                'wall_time': wall_time,
                'samples_seen': samples_seen,
                'epochs': epochs,
                'total_flops': total_flops,
                'train_loss_step': loss,
                **eval_results,
            }

            metrics_file.write(json.dumps(metrics) + "\n")
            metrics_file.flush()

            print(f"  step {step:6d} | {wall_time:6.0f}s | "
                  f"train={eval_results['train_loss']:.4f} | "
                  f"test={eval_results['test_loss']:.4f} | "
                  f"score_err={eval_results['score_error']:.4f} | "
                  f"mem={eval_results['memorization_fraction']:.3f} | "
                  f"nn={eval_results['mean_nn_ratio']:.3f}",
                  flush=True)

    metrics_file.close()
    print("Done!")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--d_latent", type=int, nargs="+", default=None)
    parser.add_argument("--d_intrinsic", type=int, nargs="+", default=None)
    parser.add_argument("--n", type=int, nargs="+", default=None)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--sigma_noise", type=float, default=0.5)
    parser.add_argument("--sigma_signal", type=float, default=1.0)
    parser.add_argument("--scale", type=float, default=3.0)
    parser.add_argument("--steps", type=int, default=300000)
    parser.add_argument("--eval_interval", type=int, default=1000)
    parser.add_argument("--n_gen", type=int, default=10000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--base_dir", type=str, default="results_v2")
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
                    config = Config(
                        d_intrinsic=d_intr,
                        d_latent=d_lat,
                        n=n_val,
                        hidden=args.hidden,
                        sigma_noise=args.sigma_noise,
                        sigma_signal=args.sigma_signal,
                        scale=args.scale,
                        total_steps=args.steps,
                        eval_interval=args.eval_interval,
                        n_gen_samples=args.n_gen,
                        seed=seed,
                        results_dir=f"{args.base_dir}/di{d_intr}_d{d_lat}_n{n_val}_s{seed}",
                    )
                    print(f"\n{'='*60}")
                    print(f"d_intrinsic={d_intr}, d_latent={d_lat}, n={n_val}, seed={seed}")
                    print(f"{'='*60}")
                    run_experiment(config)


if __name__ == "__main__":
    main()
