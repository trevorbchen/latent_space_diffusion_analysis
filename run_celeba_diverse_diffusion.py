from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from lib import diffusion, fid as fid_mod, metrics, models, training
from standard_vae import StandardConvVAE, StandardVAEConfig
from train_vae_celeba_standard_tar import TarImageDataset


TRAIN_TAR = "/overflow/data/cerberus/celeba_hq_256_partitioned_ws/train/celeba_hq_256-train-0000.tar"


class FlexibleMLPScore(nn.Module):
    """Configurable MLP score net for capacity ablations."""

    def __init__(self, d_latent: int, hidden: int, depth: int, n_freq: int = 32):
        super().__init__()
        if depth < 1:
            raise ValueError("--depth must be at least 1")
        self.d_latent = d_latent
        self.n_freq = n_freq
        self.register_buffer(
            "freqs", torch.exp(torch.linspace(0, math.log(1000), n_freq))
        )
        d_input = d_latent + 2 * n_freq
        layers: list[nn.Module] = []
        prev = d_input
        for _ in range(depth):
            layers.append(nn.Linear(prev, hidden))
            layers.append(nn.GELU())
            prev = hidden
        layers.append(nn.Linear(prev, d_latent))
        self.net = nn.Sequential(*layers)

    def forward(self, x_t: torch.Tensor, t) -> torch.Tensor:
        if isinstance(t, (int, float)):
            t = torch.full((x_t.shape[0],), float(t), device=x_t.device)
        if t.dim() == 0:
            t = t.expand(x_t.shape[0])
        t_emb = torch.cat(
            [
                torch.sin(t.unsqueeze(-1) * self.freqs),
                torch.cos(t.unsqueeze(-1) * self.freqs),
            ],
            dim=-1,
        )
        return self.net(torch.cat([x_t, t_emb], dim=-1))


def load_standard_vae(path: str, device: torch.device) -> StandardConvVAE:
    ckpt = torch.load(path, map_location=device)
    cfg = StandardVAEConfig(
        image_channels=ckpt["cfg"]["image_channels"],
        image_size=ckpt["cfg"]["image_size"],
        hidden_dims=tuple(ckpt["cfg"]["hidden_dims"]),
        d_latent=ckpt["cfg"]["d_latent"],
        arch=ckpt["cfg"].get("arch", "standard"),
    )
    model = StandardConvVAE(cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


@torch.no_grad()
def load_diverse_subset(
    *,
    vae: StandardConvVAE,
    subset_path: Path,
    train_tar: str,
    n_train: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    indices = json.loads(subset_path.read_text())[:n_train]
    dataset = TarImageDataset(train_tar, image_size=vae.cfg.image_size)
    images = torch.stack([dataset[i][0] for i in indices], dim=0)
    latents = []
    for chunk in images.split(256):
        mu, _ = vae.encode(chunk.to(device))
        latents.append(mu.cpu())
    return torch.cat(latents, dim=0), images, indices


def build_suite(
    *,
    model_kind: str,
    train_latents: torch.Tensor,
    train_images: torch.Tensor,
    vae: StandardConvVAE,
    d_latent: int,
    eval_interval: int,
    mem_interval: int,
    fid_interval: int,
    fid_cache_path: Path,
    fid_n_real: int,
    fid_n_gen: int,
    n_gen_samples: int,
    n_sde_steps: int,
    t_min: float,
    t_max: float,
    t_eval: float,
    device: torch.device,
) -> training.EvalSuite:
    suite = training.EvalSuite()
    train_dev = train_latents.to(device)
    fixed_noise = torch.randn_like(train_dev)
    train_imgs_flat = train_images.flatten(1)
    train_dists_pixels = torch.cdist(train_imgs_flat, train_imgs_flat)
    train_dists_pixels.fill_diagonal_(float("inf"))

    def cheap_metrics(model, step):
        e_neg_t = math.exp(-t_eval)
        delta_t = 1 - math.exp(-2 * t_eval)
        sqrt_dt = math.sqrt(delta_t)
        x_t = e_neg_t * train_dev + sqrt_dt * fixed_noise
        t_b = torch.full((train_dev.shape[0],), t_eval, device=device)
        pred = model(x_t, t_b)
        eval_loss = ((sqrt_dt * pred + fixed_noise) ** 2).sum(-1).mean().item() / d_latent
        return {
            "train_loss": eval_loss,
            "subset_score_loss": eval_loss,
        }

    suite.add("cheap", cheap_metrics, interval=eval_interval)

    if model_kind != "mlp":
        return suite

    def memorization(model, step):
        gen_latents = diffusion.euler_maruyama(
            model,
            n_gen_samples,
            d_latent,
            n_steps=n_sde_steps,
            t_max=t_max,
            t_min=t_min,
            device=device,
        )
        with torch.no_grad():
            gen_pixels = vae.decode(gen_latents.to(next(vae.parameters()).device)).cpu()
        mem = metrics.nn_ratio_memorization(
            gen_pixels.flatten(1),
            train_images.flatten(1),
            train_dists=train_dists_pixels,
        )
        return {
            "memorization_fraction_pixel": mem.memorization_fraction,
            "mean_nn_ratio_pixel": mem.mean_nn_ratio,
        }

    suite.add("memorization", memorization, interval=mem_interval)

    if fid_interval > 0:
        extractor = fid_mod.InceptionFeatures().to(device)
        real_feats = fid_mod.cached_real_features(
            train_images[:fid_n_real],
            cache_path=fid_cache_path,
            extractor=extractor,
            device=device,
        )
        print(f"  FID real-side stats cached over {real_feats.n} diverse train images")

        def fid_eval(model, step):
            gen_latents = diffusion.euler_maruyama(
                model,
                fid_n_gen,
                d_latent,
                n_steps=n_sde_steps,
                t_max=t_max,
                t_min=t_min,
                device=device,
            )
            with torch.no_grad():
                gen_pixels = vae.decode(gen_latents.to(next(vae.parameters()).device))
            score = fid_mod.fid_against_real(
                real_feats,
                gen_pixels.cpu(),
                extractor=extractor,
                device=device,
            )
            return {"fid": score}

        suite.add("fid", fid_eval, interval=fid_interval)

    return suite


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--vae_checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--subset_path", default="diagnostics/celeba_diverse_1k_mean_plus_sd/subset_indices.json")
    p.add_argument("--train_tar", default=TRAIN_TAR)
    p.add_argument("--n_train", type=int, default=1000)
    p.add_argument("--total_steps", type=int, default=5_000_000)
    p.add_argument("--eval_interval", type=int, default=10_000)
    p.add_argument("--mem_interval", type=int, default=100_000)
    p.add_argument("--fid_interval", type=int, default=100_000)
    p.add_argument("--fid_n_real", type=int, default=1000)
    p.add_argument("--fid_n_gen", type=int, default=1000)
    p.add_argument("--n_gen_samples", type=int, default=1000)
    p.add_argument("--n_sde_steps", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--optimizer", choices=("adam", "sgd"), default="adam")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--momentum", type=float, default=0.95)
    p.add_argument("--t_min", type=float, default=0.01)
    p.add_argument("--t_max", type=float, default=3.0)
    p.add_argument("--t_eval", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2) + "\n")

    vae = load_standard_vae(args.vae_checkpoint, device)
    train_latents, train_images, indices = load_diverse_subset(
        vae=vae,
        subset_path=Path(args.subset_path),
        train_tar=args.train_tar,
        n_train=args.n_train,
        device=device,
    )
    d_latent = vae.cfg.d_latent
    if args.depth == 3:
        model = models.build_model("mlp", d_latent, hidden=args.hidden).to(device)
    else:
        model = FlexibleMLPScore(d_latent, hidden=args.hidden, depth=args.depth).to(device)
    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    fid_cache_path = Path("data/fid_cache") / f"celeba_hq_diverse1k_d{d_latent}_n{args.fid_n_real}.npz"
    suite = build_suite(
        model_kind="mlp",
        train_latents=train_latents,
        train_images=train_images,
        vae=vae,
        d_latent=d_latent,
        eval_interval=args.eval_interval,
        mem_interval=args.mem_interval,
        fid_interval=args.fid_interval,
        fid_cache_path=fid_cache_path,
        fid_n_real=args.fid_n_real,
        fid_n_gen=args.fid_n_gen,
        n_gen_samples=args.n_gen_samples,
        n_sde_steps=args.n_sde_steps,
        t_min=args.t_min,
        t_max=args.t_max,
        t_eval=args.t_eval,
        device=device,
    )
    cfg = training.TrainConfig(
        total_steps=args.total_steps,
        eval_interval=args.eval_interval,
        mem_interval=args.mem_interval,
        batch_size=args.batch_size,
        t_min=args.t_min,
        t_max=args.t_max,
        checkpoint_every=100_000,
        log_every=args.eval_interval,
    )
    training.train_loop(
        model,
        train_latents,
        optimizer,
        cfg=cfg,
        eval_suite=suite,
        device=device,
        out_dir=out_dir,
        run_meta={
            "data": "celeba_hq_diverse1k",
            "model": "mlp",
            "hidden": args.hidden,
            "depth": args.depth,
            "optimizer": args.optimizer,
            "lr": args.lr,
            "momentum": args.momentum if args.optimizer == "sgd" else None,
            "d_latent": d_latent,
            "n": train_latents.shape[0],
            "vae_checkpoint": args.vae_checkpoint,
            "subset_path": args.subset_path,
            "subset_indices": indices,
            "fid_reference": "same farthest-first 1k training subset",
            "seed": args.seed,
        },
    )


if __name__ == "__main__":
    main()
