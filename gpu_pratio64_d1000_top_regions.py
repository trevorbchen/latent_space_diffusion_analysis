"""Approximate top signal/noise/sample eigenvalues for d_latent=1000, p=64*d.

This avoids full 64k x 64k eigendecomposition. It builds U on GPU, then uses a
randomized subspace projection to estimate the top d_latent+n eigenvalues.
The rank-null tail is intentionally omitted.
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def random_rotation(d, device):
    q, r = torch.linalg.qr(torch.randn(d, d, device=device))
    signs = torch.sign(torch.diag(r))
    signs[signs == 0] = 1
    return q * signs


def sample_gmm(args, device):
    torch.manual_seed(args.seed)
    raw = torch.randn(args.k, args.d_intrinsic, device=device)
    raw = raw / raw.norm(dim=1, keepdim=True).clamp_min(1e-12) * args.center_scale
    labels = torch.randint(0, args.k, (args.n,), device=device)
    data = torch.zeros(args.n, args.d_latent, device=device)
    data[:, :args.d_intrinsic] = (
        raw[labels] + torch.randn(args.n, args.d_intrinsic, device=device) * args.sigma_signal
    )
    if args.d_latent > args.d_intrinsic:
        data[:, args.d_intrinsic:] = (
            torch.randn(args.n, args.d_latent - args.d_intrinsic, device=device) * args.sigma_noise
        )
    q = random_rotation(args.d_latent, device)
    return data @ q.T


def build_U(args, device):
    p = args.p_ratio * args.d_latent
    data = sample_gmm(args, device)
    torch.manual_seed(args.seed + 1)
    W = torch.randn(p, args.d_latent, device=device) / math.sqrt(args.d_latent)
    U = torch.zeros(p, p, device=device)
    delta_t = 1 - math.exp(-2 * args.t)
    e_neg_t = math.exp(-args.t)
    scale = 1.0 / (args.mc_samples * args.n)
    for i in range(args.mc_samples):
        noise = torch.randn(args.n, args.d_latent, device=device)
        x_t = e_neg_t * data + math.sqrt(delta_t) * noise
        phi = torch.tanh(x_t @ W.T)
        U.addmm_(phi.T, phi, beta=1.0, alpha=scale)
        if (i + 1) % args.log_every == 0:
            torch.cuda.synchronize()
            print(f"built U MC {i + 1}/{args.mc_samples}", flush=True)
    return U


def approximate_top_eigs(U, args):
    p = U.shape[0]
    target = args.d_latent + args.n
    sketch = target + args.oversample
    torch.manual_seed(args.seed + 2)
    omega = torch.randn(p, sketch, device=U.device)
    print(f"randomized range multiply p={p}, sketch={sketch}", flush=True)
    Y = U @ omega
    Q, _ = torch.linalg.qr(Y, mode="reduced")
    for i in range(args.power_iters):
        print(f"power iteration {i + 1}/{args.power_iters}", flush=True)
        Y = U @ Q
        Q, _ = torch.linalg.qr(Y, mode="reduced")
    print("projected matrix", flush=True)
    B = Q.T @ (U @ Q)
    print("small eigendecomposition", flush=True)
    evals = torch.linalg.eigvalsh(B).flip(0)
    return evals[:target].detach().cpu().numpy()


def write_plot(eigs, args, out_dir):
    d = args.d_latent
    n = args.n
    regions = [
        ("sample", eigs[d:d + n], "#1971c2"),
        ("noise-dim", eigs[args.d_intrinsic:d], "#2f9e44"),
        ("signal", eigs[:args.d_intrinsic], "#e03131"),
    ]
    vals_all = np.concatenate([np.log10(np.maximum(v, 1e-30)) for _, v, _ in regions if len(v)])
    bins = np.linspace(np.percentile(vals_all, 0.2), np.percentile(vals_all, 99.8), 80)
    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    for label, vals, color in regions:
        vals = np.log10(np.maximum(vals, 1e-30))
        ax.hist(vals, bins=bins, histtype="stepfilled", alpha=0.35, color=color, label=label)
        ax.axvline(float(np.median(vals)), color=color, lw=0.9, ls="--")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\log_{10}\lambda_i(U)$")
    ax.set_ylabel("count")
    ax.set_title(rf"Approx top regions, $d_{{lat}}=1000$, $p=64000$")
    ax.grid(True, which="both", alpha=0.18)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "top_regions_hist.png", dpi=170)
    fig.savefig(out_dir / "top_regions_hist.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="pratio64_d1000_top_regions")
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--d-latent", type=int, default=1000)
    parser.add_argument("--d-intrinsic", type=int, default=5)
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--sigma-signal", type=float, default=1.0)
    parser.add_argument("--sigma-noise", type=float, default=0.5)
    parser.add_argument("--center-scale", type=float, default=3.0)
    parser.add_argument("--t", type=float, default=0.01)
    parser.add_argument("--mc-samples", type=int, default=500)
    parser.add_argument("--p-ratio", type=int, default=64)
    parser.add_argument("--oversample", type=int, default=128)
    parser.add_argument("--power-iters", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=25)
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2))

    U = build_U(args, device)
    eigs = approximate_top_eigs(U, args)
    np.save(out_dir / "top_eigenvalues_d1000.npy", eigs)
    write_plot(eigs, args, out_dir)
    print(f"wrote {out_dir / 'top_regions_hist.png'}", flush=True)


if __name__ == "__main__":
    main()
