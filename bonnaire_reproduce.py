"""
Reproduce Bonnaire et al. RFNN results.

Data: isotropic Gaussian N(0, I_d), d=100
Model: s_A(x) = (1/sqrt(p)) A tanh(Wx/sqrt(d)), W frozen, A learned
p = 64d = 6400
t_fixed = 0.01
n/d ∈ {4, 8, 16, 32}
SGD, A(0) = 0
"""

import math
import json
import time
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


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


@dataclass
class Config:
    d: int = 100
    psi_p: int = 64          # p = psi_p * d
    psi_n: int = 8           # n = psi_n * d
    t_fixed: float = 0.01
    lr: float = -1               # auto: 0.01 * d / delta_t
    momentum: float = 0.0        # Bonnaire theory uses plain GD
    total_steps: int = 2000000
    eval_interval: int = 1000
    n_noise_samples: int = 50  # for U computation
    seed: int = 42
    results_dir: str = "results_bonnaire"


class RFNNScore(nn.Module):
    def __init__(self, d, p, t_fixed):
        super().__init__()
        self.d = d
        self.p = p
        self.t_fixed = t_fixed
        self.register_buffer('W', torch.randn(p, d) / math.sqrt(d))
        self.A = nn.Parameter(torch.zeros(d, p))

    def forward(self, x_t):
        features = torch.tanh(x_t @ self.W.T)
        return features @ self.A.T / math.sqrt(self.p)


def compute_U(W, data, t, n_noise_samples=50):
    p, d = W.shape
    n = data.shape[0]
    delta_t = 1 - math.exp(-2 * t)
    e_neg_t = math.exp(-t)

    U = torch.zeros(p, p)
    for _ in range(n_noise_samples):
        noise = torch.randn(n, d)
        x_t = e_neg_t * data.cpu() + math.sqrt(delta_t) * noise
        phi = torch.tanh(x_t @ W.cpu().T)
        U += phi.T @ phi / n
    U /= n_noise_samples
    return U


def true_score_isotropic(x, t):
    """True score for N(0, I_d): nabla log p_t(x) = -x / (delta_t + e^{-2t})"""
    delta_t = 1 - math.exp(-2 * t)
    sigma_t = delta_t + math.exp(-2 * t)  # = 1 for unit variance
    return -x / sigma_t


def run(config: Config):
    device = get_device()
    d = config.d
    p = config.psi_p * d
    n = config.psi_n * d

    print(f"Device: {device}")
    print(f"d={d}, p={p} (psi_p={config.psi_p}), n={n} (psi_n={config.psi_n}), t={config.t_fixed}")

    save_dir = Path(config.results_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "config.json", "w") as f:
        json.dump(asdict(config), f, indent=2)

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Data: isotropic Gaussian
    data = torch.randn(n, d)
    test_data = torch.randn(2048, d)
    test_noise = torch.randn_like(test_data)

    # Precompute diffusion params
    t = config.t_fixed
    delta_t = 1 - math.exp(-2 * t)
    e_neg_t = math.exp(-t)

    # Model
    model = RFNNScore(d, p, config.t_fixed).to(device)

    # Learning rate: Bonnaire uses lr = 0.01 * d / delta_t
    lr = config.lr if config.lr > 0 else 0.01 * d / delta_t
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=config.momentum)
    print(f"Learnable params: {d * p:,}, lr={lr:.2f}")

    # Compute U before training
    print("Computing U eigenvalues...")
    U = compute_U(model.W, data, config.t_fixed, config.n_noise_samples)
    eigenvalues = torch.linalg.eigvalsh(U).flip(0).cpu().numpy()
    np.save(save_dir / "eigenvalues.npy", eigenvalues)
    print(f"  Top 10: {eigenvalues[:10].round(3)}")
    print(f"  Eigenvalues > 0.01: {(eigenvalues > 0.01).sum()}")

    # Training
    data_dev = data.to(device)
    metrics_file = open(save_dir / "metrics.jsonl", "w")
    t0 = time.time()

    for step in range(1, config.total_steps + 1):
        noise = torch.randn_like(data_dev)
        x_t = e_neg_t * data_dev + math.sqrt(delta_t) * noise

        pred = model(x_t)
        residual = math.sqrt(delta_t) * pred + noise
        loss = (residual ** 2).sum() / (d * n)  # Bonnaire scaling

        optimizer.zero_grad()
        loss.backward()
        opt_step(optimizer, device)

        if step % config.eval_interval == 0 or step == 1:
            model.eval()
            with torch.no_grad():
                # Train loss
                train_noise = torch.randn_like(data_dev)
                train_xt = e_neg_t * data_dev + math.sqrt(delta_t) * train_noise
                train_pred = model(train_xt)
                train_loss = ((math.sqrt(delta_t) * train_pred + train_noise) ** 2).sum(-1).mean().item() / d

                # Test loss
                test_dev = test_data.to(device)
                test_noise_dev = test_noise.to(device)
                test_xt = e_neg_t * test_dev + math.sqrt(delta_t) * test_noise_dev
                test_pred = model(test_xt)
                test_loss = ((math.sqrt(delta_t) * test_pred + test_noise_dev) ** 2).sum(-1).mean().item() / d

                # Score error (isotropic Gaussian: true score = -x / sigma_t)
                true_s = true_score_isotropic(test_xt, t)
                score_err = ((test_pred - true_s) ** 2).sum(-1).mean().item() / d

            model.train()

            wall_time = time.time() - t0
            tau = step * lr  # Bonnaire's rescaled training time
            metrics = {
                'step': step,
                'tau': tau,
                'wall_time': wall_time,
                'train_loss': train_loss,
                'test_loss': test_loss,
                'gen_gap': test_loss - train_loss,
                'score_error': score_err,
            }
            metrics_file.write(json.dumps(metrics) + "\n")
            metrics_file.flush()

            print(f"  step {step:6d} | {wall_time:5.0f}s | "
                  f"train={train_loss:.4f} test={test_loss:.4f} "
                  f"gap={test_loss - train_loss:.4f} score_err={score_err:.4f}",
                  flush=True)

    metrics_file.close()
    print("Done!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--psi_n", type=int, nargs="+", default=[4, 8, 16, 32])
    parser.add_argument("--d", type=int, default=100)
    parser.add_argument("--psi_p", type=int, default=64)
    parser.add_argument("--steps", type=int, default=300000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--base_dir", type=str, default="results_bonnaire")
    args = parser.parse_args()

    for psi_n in args.psi_n:
        for seed in args.seeds:
            config = Config(
                d=args.d,
                psi_p=args.psi_p,
                psi_n=psi_n,
                total_steps=args.steps,
                seed=seed,
                results_dir=f"{args.base_dir}/psin{psi_n}_s{seed}",
            )
            print(f"\n{'='*60}")
            print(f"psi_n={psi_n} (n={psi_n * args.d}), seed={seed}")
            print(f"{'='*60}")
            run(config)


if __name__ == "__main__":
    main()
