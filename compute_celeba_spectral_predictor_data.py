from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch

from run_celeba_diverse_diffusion import load_diverse_subset, load_standard_vae


DIMS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200]
TRAIN_TAR = "/overflow/data/cerberus/celeba_hq_256_partitioned_ws/train/celeba_hq_256-train-0000.tar"


def t_label(t: float) -> str:
    return str(t).replace(".", "p")


def effective_rank(vals: np.ndarray) -> float:
    vals = np.asarray(vals, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    vals = vals[vals > 0]
    if vals.size == 0:
        return 0.0
    denom = float(np.sum(vals * vals))
    if denom <= 0:
        return 0.0
    return float(np.sum(vals) ** 2 / denom)


def eta_star_mc(z: torch.Tensor, *, t: float, n_mc: int, seed: int) -> float:
    gen = torch.Generator(device=z.device)
    gen.manual_seed(seed)
    e_neg = math.exp(-t)
    delta = 1.0 - math.exp(-2.0 * t)
    vals = []
    for _ in range(n_mc):
        eps = torch.randn(z.shape, generator=gen, device=z.device, dtype=z.dtype)
        x_t = e_neg * z + math.sqrt(delta) * eps
        vals.append(((x_t.square().sum(dim=1) / z.shape[1]) ** 3).mean().item())
    return float(np.mean(vals))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="diagnostics/celeba_spectral_mem_predictor")
    p.add_argument("--subset_path", default="diagnostics/celeba_diverse_1k_mean_plus_sd/subset_indices.json")
    p.add_argument("--train_tar", default=TRAIN_TAR)
    p.add_argument("--n_train", type=int, default=1000)
    p.add_argument("--dims", default=",".join(str(d) for d in DIMS))
    p.add_argument("--times", default="0.01,0.1,1.0")
    p.add_argument("--primary_t", type=float, default=0.1)
    p.add_argument("--eta_mc", type=int, default=64)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    dims = [int(x) for x in args.dims.split(",") if x.strip()]
    times = [float(x) for x in args.times.split(",") if x.strip()]
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    rows: list[dict[str, float | int | str]] = []
    spectra: dict[str, np.ndarray] = {}
    latent_summary: dict[str, np.ndarray] = {}
    subset_indices = json.loads(Path(args.subset_path).read_text())[: args.n_train]

    for d in dims:
        ckpt_path = f"vae_checkpoints/celeba_resnet_modernloss_d{d}/vae.pt"
        if not Path(ckpt_path).exists():
            raise FileNotFoundError(ckpt_path)
        print(f"[spectral] d={d}: loading {ckpt_path}", flush=True)
        vae = load_standard_vae(ckpt_path, device)
        z_cpu, _, indices = load_diverse_subset(
            vae=vae,
            subset_path=Path(args.subset_path),
            train_tar=args.train_tar,
            n_train=args.n_train,
            device=device,
        )
        if indices != subset_indices:
            raise RuntimeError(f"Subset mismatch for d={d}")
        z = z_cpu.to(device=device, dtype=torch.float64)
        n = z.shape[0]
        if z.shape[1] != d:
            raise RuntimeError(f"Expected latent dim {d}, got {z.shape[1]}")

        gram = (z.T @ z) / float(n)
        z_centered = z - z.mean(dim=0, keepdim=True)
        cov_centered = (z_centered.T @ z_centered) / float(n)

        latent_summary[f"z_normsq_d{d}"] = z.square().sum(dim=1).cpu().numpy()

        for t in times:
            delta = 1.0 - math.exp(-2.0 * t)
            scale = math.exp(-2.0 * t)
            m_t = scale * gram + delta * torch.eye(d, device=device, dtype=torch.float64)
            centered_m_t = scale * cov_centered + delta * torch.eye(d, device=device, dtype=torch.float64)
            eig = torch.linalg.eigvalsh(m_t).cpu().numpy()
            eig_centered = torch.linalg.eigvalsh(centered_m_t).cpu().numpy()
            eig_desc = eig[::-1].copy()
            eig_centered_desc = eig_centered[::-1].copy()

            if not np.all(np.isfinite(eig_desc)):
                raise RuntimeError(f"Non-finite eigenvalues for d={d}, t={t}")
            if eig_desc[-1] < -1e-8:
                raise RuntimeError(f"Negative eigenvalue below tolerance for d={d}, t={t}: {eig_desc[-1]}")

            bottom_k = max(1, int(math.ceil(0.30 * d)))
            beta = float(np.median(eig[:bottom_k]))
            eig_excess = np.maximum(eig_desc - beta, 0.0)
            eig_centered_excess = np.maximum(eig_centered_desc - beta, 0.0)
            r_eff = effective_rank(eig_excess)
            r_eff_centered = effective_rank(eig_centered_excess)
            eta = eta_star_mc(z, t=t, n_mc=args.eta_mc, seed=1729 + d + int(round(1000 * t)))

            key = f"d{d}_t{t_label(t)}"
            spectra[f"eig_{key}"] = eig_desc
            spectra[f"eig_centered_{key}"] = eig_centered_desc
            spectra[f"weights_excess_{key}"] = eig_excess

            rows.append(
                {
                    "d": d,
                    "t": t,
                    "n": n,
                    "lambda_max": float(eig_desc[0]),
                    "lambda_min": float(eig_desc[-1]),
                    "lambda_median": float(np.median(eig_desc)),
                    "lambda_mean": float(np.mean(eig_desc)),
                    "beta_floor": beta,
                    "effective_rank_excess": r_eff,
                    "effective_rank_centered_excess": r_eff_centered,
                    "buffer_proxy": max(float(d - round(r_eff)), 0.0),
                    "eta_star": eta,
                    "z_normsq_over_d_mean": float((z.square().sum(dim=1) / d).mean().item()),
                    "z_normsq_over_d_std": float((z.square().sum(dim=1) / d).std(unbiased=True).item()),
                }
            )

        del vae, z, z_cpu, gram, cov_centered
        if device.type == "cuda":
            torch.cuda.empty_cache()

    fieldnames = list(rows[0].keys())
    with (out_dir / "spectral_features.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(out_dir / "spectra.npz", **spectra)
    np.savez_compressed(out_dir / "latent_norms.npz", **latent_summary)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2) + "\n")
    print(f"[spectral] wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
