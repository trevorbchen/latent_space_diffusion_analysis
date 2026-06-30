"""GPU spectral run for p = 64 * d_latent at d_latent = 500, 1000.

Designed for ml-login7 A100 80GB. It computes the RFNN feature correlation
matrix U using tanh features, eigendecomposes U, and writes histograms colored
by theoretical index regions.
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


def random_rotation(d, device, dtype):
    q, r = torch.linalg.qr(torch.randn(d, d, device=device, dtype=dtype))
    signs = torch.sign(torch.diag(r))
    signs[signs == 0] = 1
    return q * signs


def sample_gmm(n, d_intrinsic, d_latent, k, sigma_signal, sigma_noise,
               center_scale, device, dtype, seed):
    gen = torch.Generator(device=device).manual_seed(seed)
    raw = torch.randn(k, d_intrinsic, generator=gen, device=device, dtype=dtype)
    raw = raw / raw.norm(dim=1, keepdim=True).clamp_min(1e-12) * center_scale
    labels = torch.randint(0, k, (n,), generator=gen, device=device)
    data = torch.zeros(n, d_latent, device=device, dtype=dtype)
    data[:, :d_intrinsic] = (
        raw[labels]
        + torch.randn(n, d_intrinsic, generator=gen, device=device, dtype=dtype) * sigma_signal
    )
    if d_latent > d_intrinsic:
        data[:, d_intrinsic:] = (
            torch.randn(n, d_latent - d_intrinsic, generator=gen, device=device, dtype=dtype)
            * sigma_noise
        )
    q = random_rotation(d_latent, device, dtype)
    return data @ q.T


def compute_eigs(d_latent, args):
    device = torch.device(args.device)
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    torch.manual_seed(args.seed + d_latent)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = args.tf32

    p = args.p_ratio * d_latent
    data = sample_gmm(
        args.n, args.d_intrinsic, d_latent, args.k,
        args.sigma_signal, args.sigma_noise, args.center_scale,
        device, dtype, args.seed + 1000 + d_latent,
    )
    gen = torch.Generator(device=device).manual_seed(args.seed + 2000 + d_latent)
    W = torch.randn(p, d_latent, generator=gen, device=device, dtype=dtype) / math.sqrt(d_latent)
    U = torch.zeros(p, p, device=device, dtype=dtype)

    delta_t = 1 - math.exp(-2 * args.t)
    e_neg_t = math.exp(-args.t)
    scale = 1.0 / (args.mc_samples * args.n)
    for i in range(args.mc_samples):
        noise = torch.randn(args.n, d_latent, generator=gen, device=device, dtype=dtype)
        x_t = e_neg_t * data + math.sqrt(delta_t) * noise
        phi = torch.tanh(x_t @ W.T)
        U.addmm_(phi.T, phi, beta=1.0, alpha=scale)
        if (i + 1) % max(1, args.log_every) == 0:
            if device.type == "cuda":
                torch.cuda.synchronize()
            print(f"d={d_latent} MC {i + 1}/{args.mc_samples}", flush=True)

    print(f"d={d_latent} eigvalsh p={p}", flush=True)
    if not torch.isfinite(U).all():
        bad = U.numel() - torch.isfinite(U).sum().item()
        raise RuntimeError(f"U contains {bad} non-finite entries")
    if args.eig_backend == "scipy-cpu":
        import scipy.linalg
        print("copy U.T to CPU for scipy.linalg.eigvalsh Fortran view", flush=True)
        U_cpu_base = U.T.contiguous().detach().cpu().numpy()
        del U
        U = None
        if device.type == "cuda":
            torch.cuda.empty_cache()
        U_fortran = U_cpu_base.T
        print(f"CPU matrix F-contiguous={U_fortran.flags.f_contiguous}", flush=True)
        print(f"scipy eigvalsh driver={args.scipy_driver}", flush=True)
        eigs = scipy.linalg.eigvalsh(
            U_fortran,
            overwrite_a=True,
            check_finite=False,
            driver=args.scipy_driver,
        )[::-1]
        del U_cpu_base, U_fortran
    else:
        eigs = torch.linalg.eigvalsh(U).flip(0).detach().cpu().numpy()
    if U is not None:
        del U
    del W, data
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return eigs


def region_log_values(eigs, d_intrinsic, d_latent, n):
    p = len(eigs)
    regions = [
        ("rank-null", eigs[min(d_latent + n, p):], "#5f3dc4"),
        ("sample", eigs[d_latent:min(d_latent + n, p)], "#1971c2"),
        ("noise-dim", eigs[d_intrinsic:d_latent], "#2f9e44"),
        ("signal", eigs[:d_intrinsic], "#e03131"),
    ]
    out = []
    for label, vals, color in regions:
        if len(vals) == 0:
            continue
        vals = np.log10(np.maximum(vals, 1e-30))
        out.append((label, vals, color))
    return out


def write_hist(out_dir, d_latents, args):
    fig, axes = plt.subplots(1, len(d_latents), figsize=(3.2 * len(d_latents), 2.8),
                             sharey=True, squeeze=False)
    all_logs = []
    cached = []
    for dlat in d_latents:
        eigs = np.sort(np.load(out_dir / f"eigenvalues_d{dlat}.npy"))[::-1]
        regions = region_log_values(eigs, args.d_intrinsic, dlat, args.n)
        cached.append((dlat, regions))
        all_logs.extend(vals for _label, vals, _color in regions)
    all_logs = np.concatenate(all_logs)
    lo, hi = np.percentile(all_logs, [0.2, 99.8])
    bins = np.linspace(lo, hi, 90)

    for ax, (dlat, regions) in zip(axes.flat, cached):
        for label, vals, color in regions:
            ax.hist(vals, bins=bins, histtype="stepfilled", alpha=0.33,
                    color=color, label=label)
            ax.axvline(float(np.median(vals)), color=color, lw=0.9,
                       ls="--", alpha=0.85)
        ax.set_yscale("log")
        ax.set_title(rf"$d_{{lat}}={dlat}$, $p={args.p_ratio * dlat}$")
        ax.set_xlabel(r"$\log_{10}\lambda_i(U)$")
        ax.grid(True, which="both", alpha=0.16, lw=0.4)
    axes[0, 0].set_ylabel("count")

    from matplotlib.patches import Patch
    legend_items = [
        Patch(facecolor="#e03131", alpha=0.33, label="signal"),
        Patch(facecolor="#2f9e44", alpha=0.33, label="noise-dim"),
        Patch(facecolor="#1971c2", alpha=0.33, label="sample"),
        Patch(facecolor="#5f3dc4", alpha=0.33, label="rank-null"),
    ]
    fig.suptitle(
        rf"$p={args.p_ratio}d_{{lat}}$, GMM, $\sigma_\perp={args.sigma_noise}$, MC={args.mc_samples}",
        y=1.03,
    )
    fig.legend(handles=legend_items, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out_dir / "hist_by_index_region.png", dpi=170)
    fig.savefig(out_dir / "hist_by_index_region.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="pratio64_d500_d1000")
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--tf32", action="store_true")
    parser.add_argument("--linalg-backend", choices=["default", "cusolver", "magma"],
                        default="default")
    parser.add_argument("--eig-backend", choices=["torch-gpu", "scipy-cpu"],
                        default="torch-gpu")
    parser.add_argument("--scipy-driver", choices=["ev", "evd", "evr", "evx"],
                        default="evr")
    parser.add_argument("--d-latents", nargs="+", type=int, default=[500, 1000])
    parser.add_argument("--d-intrinsic", type=int, default=5)
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--sigma-signal", type=float, default=1.0)
    parser.add_argument("--sigma-noise", type=float, default=0.5)
    parser.add_argument("--center-scale", type=float, default=3.0)
    parser.add_argument("--t", type=float, default=0.01)
    parser.add_argument("--mc-samples", type=int, default=500)
    parser.add_argument("--p-ratio", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()
    if args.linalg_backend != "default":
        torch.backends.cuda.preferred_linalg_library(args.linalg_backend)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2))

    for dlat in args.d_latents:
        eig_path = out_dir / f"eigenvalues_d{dlat}.npy"
        if eig_path.exists():
            print(f"skip existing {eig_path}", flush=True)
            continue
        eigs = compute_eigs(dlat, args)
        np.save(eig_path, eigs)
        print(f"wrote {eig_path}", flush=True)
    write_hist(out_dir, args.d_latents, args)
    print(f"wrote {out_dir / 'hist_by_index_region.png'}", flush=True)


if __name__ == "__main__":
    main()
