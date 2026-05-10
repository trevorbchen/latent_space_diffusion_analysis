"""Unified experiment runner.

Examples:
    # Synthetic, MLP score
    python run_experiment.py --data synthetic --model mlp \
        --d_intrinsic 5 --d_latent 20 --sigma_noise 0.5 \
        --total_steps 300000 --out results/synth_mlp_d20

    # Synthetic, RFNN
    python run_experiment.py --data synthetic --model rfnn \
        --d_intrinsic 5 --d_latent 20 --total_steps 300000 \
        --out results/synth_rfnn_d20

    # Real, MNIST through a pretrained VAE (d_latent set by VAE bottleneck)
    python run_experiment.py --data mnist --model mlp \
        --vae_checkpoint vae_checkpoints/mnist_d10/vae.pt \
        --total_steps 300000 --out results/mnist_mlp_d10
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from lib import (
    data_synthetic,
    data_real,
    diffusion,
    eigenvalues as eigmod,
    metrics,
    models,
    true_score,
    training,
)
# fid is imported lazily inside build_real_suite to avoid pulling torchvision
# into synthetic-only runs.


# ---------------------------------------------------------------------------
# Synthetic eval suite
# ---------------------------------------------------------------------------

def build_synthetic_suite(model,
                          *,
                          train_data: torch.Tensor,
                          test_data: torch.Tensor,
                          test_noise: torch.Tensor,
                          means: torch.Tensor,
                          Q: torch.Tensor,
                          d_intrinsic: int,
                          d_latent: int,
                          sigma_noise: float,
                          sigma_signal: float,
                          t_eval: float,
                          n_gen_samples: int,
                          n_sde_steps: int,
                          t_min: float,
                          t_max: float,
                          eval_interval: int,
                          mem_interval: int,
                          eigvals_interval: int,
                          model_kind: str,
                          device: torch.device,
                          train_dists: torch.Tensor) -> training.EvalSuite:
    suite = training.EvalSuite()

    # RFNN is trained at fixed t_fixed; evaluate score error there too. MLP
    # samples t uniformly in [t_min, t_max], so we evaluate at the requested
    # t_eval (matches v2 / paper Appendix A).
    if model_kind == 'rfnn':
        t_eval = float(model.t_fixed)

    sigma_t_inv, log_det = true_score.precompute_sigma_t(
        d_intrinsic, d_latent, Q, sigma_noise, t_eval, sigma_signal=sigma_signal,
    )
    sigma_t_inv = sigma_t_inv.to(device)
    log_det = log_det.to(device)
    means_dev = means.to(device)
    test_dev = test_data.to(device)
    test_noise_dev = test_noise.to(device)

    def cheap_metrics(model, step):
        e_neg_t = math.exp(-t_eval)
        delta_t = 1 - math.exp(-2 * t_eval)
        sqrt_dt = math.sqrt(delta_t)

        # Train loss
        n_train_eval = train_data.shape[0]
        tn = torch.randn_like(train_data, device=device)
        x_t_train = e_neg_t * train_data.to(device) + sqrt_dt * tn
        t_b = torch.full((n_train_eval,), t_eval, device=device)
        pred_tr = model(x_t_train, t_b)
        train_loss = ((sqrt_dt * pred_tr + tn) ** 2).sum(-1).mean().item() / d_latent

        # Test loss
        x_t_test = e_neg_t * test_dev + sqrt_dt * test_noise_dev
        t_b_test = torch.full((test_dev.shape[0],), t_eval, device=device)
        pred_te = model(x_t_test, t_b_test)
        test_loss = (
            (sqrt_dt * pred_te + test_noise_dev) ** 2
        ).sum(-1).mean().item() / d_latent

        # Score error against analytic ground truth
        true_s = true_score.true_score(
            x_t_test, t_eval, means_dev,
            sigma_t_inv=sigma_t_inv, log_det_sigma_t=log_det,
        )
        score_err = ((pred_te - true_s) ** 2).sum(-1).mean().item() / d_latent

        return {
            'train_loss': train_loss,
            'test_loss': test_loss,
            'gen_gap': test_loss - train_loss,
            'score_error': score_err,
        }
    suite.add('cheap', cheap_metrics, interval=eval_interval)

    # Memorization is only meaningful for the trainable MLP (RFNN at fixed
    # t can't generate). Skip generation-based metrics for RFNN.
    if model_kind == 'mlp':
        def memorization(model, step):
            cpu_model = models.MLPScore(
                d_latent,
                hidden=model.net[0].out_features,
                n_freq=model.n_freq,
            )
            cpu_model.load_state_dict({k: v.cpu() for k, v in model.state_dict().items()})
            cpu_model.eval()
            generated = diffusion.euler_maruyama(
                cpu_model, n_gen_samples, d_latent,
                n_steps=n_sde_steps, t_max=t_max, t_min=t_min,
            )
            mem = metrics.nn_ratio_memorization(
                generated, train_data, train_dists=train_dists,
            )
            return {
                'memorization_fraction': mem.memorization_fraction,
                'mean_nn_ratio': mem.mean_nn_ratio,
            }
        suite.add('memorization', memorization, interval=mem_interval)

    # Eigenvalue snapshots + bulk-absorption tracking for RFNN
    if model_kind == 'rfnn':
        n = train_data.shape[0]
        p = model.p
        layout = eigmod.bulk_indices(d_intrinsic, d_latent, n, p)

        # U is fixed (depends only on W and data) — compute once, cache.
        U_init = eigmod.compute_U(model.W, train_data.cpu(),
                                  t=model.t_fixed, n_noise_samples=50)
        eigvals_U, eigvecs_U = eigmod.eigendecompose_U(U_init)

        def eigenvalues_snapshot(model, step):
            row = {
                'eigvals_U_summary': eigmod.bulk_summary(eigvals_U, layout),
                'absorption': eigmod.absorption_per_bulk(
                    model.A, eigvecs_U, layout, eigvals_U,
                ),
            }
            return row
        suite.add('eigvals', eigenvalues_snapshot, interval=eigvals_interval)

    return suite


# ---------------------------------------------------------------------------
# Real-data eval suite
# ---------------------------------------------------------------------------

def build_real_suite(model,
                     *,
                     train_latents: torch.Tensor,
                     test_latents: torch.Tensor,
                     train_images: torch.Tensor,
                     test_images: torch.Tensor,
                     vae,
                     d_latent: int,
                     n_gen_samples: int,
                     n_sde_steps: int,
                     t_min: float,
                     t_max: float,
                     t_eval: float,
                     eval_interval: int,
                     mem_interval: int,
                     fid_interval: int,
                     fid_cache_path,
                     fid_n_real: int,
                     fid_n_gen: int,
                     model_kind: str,
                     device: torch.device,
                     train_dists_pixels: torch.Tensor) -> training.EvalSuite:
    """Real data eval: no analytic score, so we use test-loss-plateau for
    \tau_gen (matching the paper's Appendix A definition). FID against the
    test set is computed every fid_interval steps.
    """
    suite = training.EvalSuite()

    test_dev = test_latents.to(device)
    test_noise_dev = torch.randn_like(test_dev)

    # ---- Precompute Inception features for the real (test) side ----
    # Skip everything FID-related if disabled (fid_interval == 0). Inception
    # V3 alone is ~3 s/image on CPU, so the smoke test must be able to skip it.
    fid_enabled = fid_interval > 0
    extractor = None
    real_feats = None
    if fid_enabled:
        from lib import fid as fid_mod
        extractor = fid_mod.InceptionFeatures().to(device)
        real_subset = test_images[:fid_n_real]
        real_feats = fid_mod.cached_real_features(
            real_subset, cache_path=fid_cache_path,
            extractor=extractor, device=device,
        )
        print(f"  FID real-side stats cached over {real_feats.n} test images")
    else:
        print("  FID disabled (fid_interval=0)")

    def cheap_metrics(model, step):
        e_neg_t = math.exp(-t_eval)
        delta_t = 1 - math.exp(-2 * t_eval)
        sqrt_dt = math.sqrt(delta_t)

        train_dev = train_latents.to(device)
        tn = torch.randn_like(train_dev)
        x_t_train = e_neg_t * train_dev + sqrt_dt * tn
        t_b = torch.full((train_dev.shape[0],), t_eval, device=device)
        pred_tr = model(x_t_train, t_b)
        train_loss = ((sqrt_dt * pred_tr + tn) ** 2).sum(-1).mean().item() / d_latent

        x_t_test = e_neg_t * test_dev + sqrt_dt * test_noise_dev
        t_b_test = torch.full((test_dev.shape[0],), t_eval, device=device)
        pred_te = model(x_t_test, t_b_test)
        test_loss = (
            (sqrt_dt * pred_te + test_noise_dev) ** 2
        ).sum(-1).mean().item() / d_latent

        return {
            'train_loss': train_loss,
            'test_loss': test_loss,
            'gen_gap': test_loss - train_loss,
        }
    suite.add('cheap', cheap_metrics, interval=eval_interval)

    if model_kind == 'mlp':
        # Memorization fraction (uses generated pixel-space samples)
        def memorization(model, step):
            cpu_model = models.MLPScore(
                d_latent,
                hidden=model.net[0].out_features,
                n_freq=model.n_freq,
            )
            cpu_model.load_state_dict({k: v.cpu() for k, v in model.state_dict().items()})
            cpu_model.eval()
            gen_latents = diffusion.euler_maruyama(
                cpu_model, n_gen_samples, d_latent,
                n_steps=n_sde_steps, t_max=t_max, t_min=t_min,
            )
            with torch.no_grad():
                gen_pixels = vae.decode(gen_latents.to(next(vae.parameters()).device)).cpu()
            mem = metrics.nn_ratio_memorization(
                gen_pixels.flatten(1),
                train_images.flatten(1),
                train_dists=train_dists_pixels,
            )
            return {
                'memorization_fraction_pixel': mem.memorization_fraction,
                'mean_nn_ratio_pixel': mem.mean_nn_ratio,
            }
        suite.add('memorization', memorization, interval=mem_interval)

        # FID gets its own (heavier) cadence and a fresh, larger generation.
        # Unlike the memorization hook (which mirrors v2's CPU sampling
        # pattern for direct numerical comparability), FID samples on the
        # main training device for speed. Inception V3 is already on
        # `device`; matching the SDE step keeps everything on GPU when
        # device == cuda.
        if fid_enabled:
            def fid_eval(model, step):
                model.eval()
                gen_latents = diffusion.euler_maruyama(
                    model, fid_n_gen, d_latent,
                    n_steps=n_sde_steps, t_max=t_max, t_min=t_min,
                    device=device,
                )
                with torch.no_grad():
                    gen_pixels = vae.decode(
                        gen_latents.to(next(vae.parameters()).device)
                    )
                score = fid_mod.fid_against_real(
                    real_feats, gen_pixels,
                    extractor=extractor, device=device,
                )
                model.train()
                return {'fid': score}
            suite.add('fid', fid_eval, interval=fid_interval)

    return suite


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def get_device(spec: str | None) -> torch.device:
    if spec is not None:
        return torch.device(spec)
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', choices=['synthetic', 'mnist', 'celeba'], required=True)
    p.add_argument('--model', choices=['mlp', 'rfnn'], required=True)

    # Synthetic-only knobs
    p.add_argument('--d_intrinsic', type=int, default=5)
    p.add_argument('--d_latent',    type=int, default=20)
    p.add_argument('--n',           type=int, default=500)
    p.add_argument('--k',           type=int, default=10)
    p.add_argument('--sigma_noise', type=float, default=0.5)
    p.add_argument('--sigma_signal', type=float, default=1.0)
    p.add_argument('--scale',       type=float, default=3.0)

    # Real-only knobs
    p.add_argument('--vae_checkpoint', type=str, default=None,
                   help='Path to vae.pt produced by train_vae.py')
    p.add_argument('--data_root',      type=str, default='./data')
    p.add_argument('--n_train',        type=int, default=None,
                   help='If set, take only the first n_train training samples.')

    # Model knobs
    p.add_argument('--hidden',   type=int, default=None)        # MLP only; auto = 8 * d_latent
    p.add_argument('--p_ratio',  type=int, default=64)          # RFNN only
    p.add_argument('--t_fixed',  type=float, default=0.01)      # RFNN only
    p.add_argument('--lr',       type=float, default=None)
    p.add_argument('--momentum', type=float, default=0.0)

    # Training knobs
    p.add_argument('--total_steps',     type=int,   default=300_000)
    p.add_argument('--eval_interval',   type=int,   default=1_000)
    p.add_argument('--mem_interval',    type=int,   default=10_000)
    p.add_argument('--eigvals_interval', type=int,  default=25_000)
    p.add_argument('--fid_interval',    type=int,   default=25_000)
    p.add_argument('--fid_n_real',      type=int,   default=10_000)
    p.add_argument('--fid_n_gen',       type=int,   default=10_000)
    p.add_argument('--batch_size',      type=int,   default=256)
    p.add_argument('--t_min',           type=float, default=0.01)
    p.add_argument('--t_max',           type=float, default=3.0)
    p.add_argument('--t_eval',          type=float, default=0.1)
    p.add_argument('--n_gen_samples',   type=int,   default=5_000)
    p.add_argument('--n_sde_steps',     type=int,   default=500)
    p.add_argument('--save_samples_grid', type=int, default=0,
                   help='If >0, dump a sqrt(N)xsqrt(N) grid of decoded '
                        'samples at end of training (real-data MLP only).')

    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out',  type=str, required=True)
    p.add_argument('--device', type=str, default=None)
    args = p.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    device = get_device(args.device)
    print(f"device={device} model={args.model} data={args.data} out={out_dir}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.data == 'synthetic':
        run_synthetic(args, out_dir, device)
    else:
        run_real(args, out_dir, device)


# ---------------------------------------------------------------------------
# Synthetic dispatch
# ---------------------------------------------------------------------------

def run_synthetic(args, out_dir: Path, device: torch.device) -> None:
    data, labels, means, Q = data_synthetic.generate_data(
        args.n, args.d_intrinsic, args.d_latent,
        k=args.k, sigma_noise=args.sigma_noise,
        sigma_signal=args.sigma_signal, scale=args.scale,
        seed=args.seed,
    )
    test = data_synthetic.generate_test_samples(
        means, args.k, args.d_intrinsic, args.d_latent, Q,
        sigma_noise=args.sigma_noise, sigma_signal=args.sigma_signal,
        n=2048, seed=9999,
    )
    test_noise = torch.randn_like(test)
    train_dists = torch.cdist(data, data); train_dists.fill_diagonal_(float('inf'))

    model = models.build_model(
        args.model, args.d_latent,
        hidden=args.hidden, p_ratio=args.p_ratio, t_fixed=args.t_fixed,
    ).to(device)
    optimizer, fixed_t, batch_size = _make_optimizer(args, model)

    suite = build_synthetic_suite(
        model, train_data=data, test_data=test, test_noise=test_noise,
        means=means, Q=Q,
        d_intrinsic=args.d_intrinsic, d_latent=args.d_latent,
        sigma_noise=args.sigma_noise, sigma_signal=args.sigma_signal,
        t_eval=args.t_eval, n_gen_samples=args.n_gen_samples,
        n_sde_steps=args.n_sde_steps, t_min=args.t_min, t_max=args.t_max,
        eval_interval=args.eval_interval, mem_interval=args.mem_interval,
        eigvals_interval=args.eigvals_interval,
        model_kind=args.model, device=device, train_dists=train_dists,
    )
    cfg = training.TrainConfig(
        total_steps=args.total_steps,
        eval_interval=args.eval_interval,
        mem_interval=args.mem_interval,
        eigvals_interval=args.eigvals_interval,
        batch_size=batch_size,
        t_min=args.t_min, t_max=args.t_max, fixed_t=fixed_t,
    )
    training.train_loop(
        model, data, optimizer, cfg=cfg, eval_suite=suite, device=device,
        out_dir=out_dir,
        run_meta={'data': 'synthetic', 'model': args.model,
                  'd_latent': args.d_latent, 'd_intrinsic': args.d_intrinsic,
                  'n': args.n, 'sigma_noise': args.sigma_noise,
                  'seed': args.seed},
    )


# ---------------------------------------------------------------------------
# Real dispatch
# ---------------------------------------------------------------------------

def run_real(args, out_dir: Path, device: torch.device) -> None:
    if args.vae_checkpoint is None:
        raise SystemExit("--vae_checkpoint is required for real data.")
    bundle = data_real.load_real_dataset(
        args.data, args.vae_checkpoint, data_root=args.data_root,
        device=device, n_train=args.n_train,
    )
    args.d_latent = bundle.d_latent  # locked by the VAE
    print(f"  d_latent (VAE bottleneck) = {bundle.d_latent}; "
          f"n_train={bundle.train_latents.shape[0]}")

    train_imgs_flat = bundle.train_images.flatten(1)
    train_dists_pixels = torch.cdist(train_imgs_flat, train_imgs_flat)
    train_dists_pixels.fill_diagonal_(float('inf'))

    model = models.build_model(
        args.model, args.d_latent,
        hidden=args.hidden, p_ratio=args.p_ratio, t_fixed=args.t_fixed,
    ).to(device)
    optimizer, fixed_t, batch_size = _make_optimizer(args, model)

    fid_cache_path = Path('./data/fid_cache') / (
        f'{args.data}_test_real_n{args.fid_n_real}.npz'
    )
    suite = build_real_suite(
        model, train_latents=bundle.train_latents,
        test_latents=bundle.test_latents,
        train_images=bundle.train_images,
        test_images=bundle.test_images,
        vae=bundle.vae, d_latent=bundle.d_latent,
        n_gen_samples=args.n_gen_samples, n_sde_steps=args.n_sde_steps,
        t_min=args.t_min, t_max=args.t_max, t_eval=args.t_eval,
        eval_interval=args.eval_interval, mem_interval=args.mem_interval,
        fid_interval=args.fid_interval,
        fid_cache_path=fid_cache_path,
        fid_n_real=args.fid_n_real,
        fid_n_gen=args.fid_n_gen,
        model_kind=args.model, device=device,
        train_dists_pixels=train_dists_pixels,
    )
    cfg = training.TrainConfig(
        total_steps=args.total_steps,
        eval_interval=args.eval_interval,
        mem_interval=args.mem_interval,
        eigvals_interval=args.eigvals_interval,
        batch_size=batch_size,
        t_min=args.t_min, t_max=args.t_max, fixed_t=fixed_t,
    )
    training.train_loop(
        model, bundle.train_latents, optimizer,
        cfg=cfg, eval_suite=suite, device=device,
        out_dir=out_dir,
        run_meta={'data': args.data, 'model': args.model,
                  'd_latent': bundle.d_latent,
                  'n': bundle.train_latents.shape[0],
                  'vae_checkpoint': args.vae_checkpoint,
                  'seed': args.seed},
    )

    # Post-training: generate a grid of samples decoded back to pixel space,
    # so we can eyeball whether the pipeline produces recognizable images.
    if args.model == 'mlp' and args.save_samples_grid:
        _save_final_samples_grid(model, bundle.vae, args, out_dir, device)


def _save_final_samples_grid(model, vae, args, out_dir: Path,
                             device: torch.device) -> None:
    from torchvision.utils import save_image
    n = args.save_samples_grid
    cpu_model = models.MLPScore(
        args.d_latent,
        hidden=model.net[0].out_features,
        n_freq=model.n_freq,
    )
    cpu_model.load_state_dict({k: v.cpu() for k, v in model.state_dict().items()})
    cpu_model.eval()
    gen_latents = diffusion.euler_maruyama(
        cpu_model, n, args.d_latent,
        n_steps=args.n_sde_steps, t_max=args.t_max, t_min=args.t_min,
    )
    with torch.no_grad():
        gen_pixels = vae.decode(gen_latents.to(next(vae.parameters()).device)).cpu()
    grid_path = out_dir / 'final_samples.png'
    n_row = int(round(n ** 0.5))
    save_image((gen_pixels * 0.5 + 0.5).clamp(0, 1), grid_path, nrow=n_row)
    print(f"  saved {n} sample grid to {grid_path}")


# ---------------------------------------------------------------------------
# Optimizer + RFNN-vs-MLP knobs
# ---------------------------------------------------------------------------

def _make_optimizer(args, model):
    """Returns (optimizer, fixed_t_or_None, effective_batch_size)."""
    if args.model == 'mlp':
        lr = args.lr if args.lr is not None else 1e-4
        return torch.optim.Adam(model.parameters(), lr=lr), None, args.batch_size

    # RFNN: SGD + Bonnaire LR scaling, full-batch
    delta_t = 1 - math.exp(-2 * args.t_fixed)
    lr = args.lr if args.lr is not None else 0.01 * args.d_latent / delta_t
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=args.momentum)
    return opt, args.t_fixed, args.n if args.data == 'synthetic' else args.batch_size


if __name__ == '__main__':
    main()
