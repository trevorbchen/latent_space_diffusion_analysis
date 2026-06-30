from __future__ import annotations

import json
from pathlib import Path

import torch

from lib import diffusion, metrics, models
from standard_vae import StandardConvVAE, StandardVAEConfig
from train_vae_celeba_standard_tar import TarImageDataset


D_LATENTS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
TRAIN_TAR = "/overflow/data/cerberus/celeba_hq_256_partitioned_ws/train/celeba_hq_256-train-0000.tar"
SUBSET_PATH = Path("diagnostics/celeba_diverse_1k_mean_plus_sd/subset_indices.json")
RESULT_ROOT = Path("results/celeba_hq_diverse1k_resnet_modernloss_5m")
OUT_DIR = Path("diagnostics/celeba_diverse_final_mem_bonnaire_10k_latent")


def load_vae(path: str, device: torch.device) -> StandardConvVAE:
    ckpt = torch.load(path, map_location=device)
    cfg = StandardVAEConfig(
        image_channels=ckpt["cfg"]["image_channels"],
        image_size=ckpt["cfg"]["image_size"],
        hidden_dims=tuple(ckpt["cfg"]["hidden_dims"]),
        d_latent=ckpt["cfg"]["d_latent"],
        arch=ckpt["cfg"].get("arch", "standard"),
    )
    vae = StandardConvVAE(cfg).to(device)
    vae.load_state_dict(ckpt["state_dict"])
    vae.eval()
    return vae


@torch.no_grad()
def load_subset(vae: StandardConvVAE, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    indices = json.loads(SUBSET_PATH.read_text())[:1000]
    dataset = TarImageDataset(TRAIN_TAR, image_size=vae.cfg.image_size)
    images = torch.stack([dataset[i][0] for i in indices], dim=0)
    latents = []
    for chunk in images.split(256):
        mu, _ = vae.encode(chunk.to(device))
        latents.append(mu.cpu())
    return images, torch.cat(latents, dim=0)


def load_diffusion_model(path: Path, d_latent: int, device: torch.device) -> models.MLPScore:
    model = models.MLPScore(d_latent, hidden=256).to(device)
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


@torch.no_grad()
def decode_in_chunks(vae: StandardConvVAE, z: torch.Tensor, chunk_size: int = 512) -> torch.Tensor:
    outs = []
    for chunk in z.split(chunk_size):
        outs.append(vae.decode(chunk).cpu())
    return torch.cat(outs, dim=0)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(12345)

    rows = []
    for d in D_LATENTS:
        vae = load_vae(f"vae_checkpoints/celeba_resnet_modernloss_d{d}/vae.pt", device)
        train_images, train_latents = load_subset(vae, device)
        model = load_diffusion_model(RESULT_ROOT / f"d{d}" / "last_model.pt", d, device)
        gen_latents = diffusion.euler_maruyama(
            model,
            10_000,
            d,
            n_steps=500,
            t_max=3.0,
            t_min=0.01,
            device=device,
        )
        gen_pixels = decode_in_chunks(vae, gen_latents)

        pixel_mem = metrics.nn_ratio_memorization(
            gen_pixels.flatten(1),
            train_images.flatten(1),
        )
        latent_mem = metrics.nn_ratio_memorization(
            gen_latents.cpu(),
            train_latents,
        )
        row = {
            "d_latent": d,
            "n_generated": 10_000,
            "pixel_mean_nn_ratio": pixel_mem.mean_nn_ratio,
            "pixel_mem_fraction": pixel_mem.memorization_fraction,
            "pixel_mem_count": int(round(10_000 * pixel_mem.memorization_fraction)),
            "latent_mean_nn_ratio": latent_mem.mean_nn_ratio,
            "latent_mem_fraction": latent_mem.memorization_fraction,
            "latent_mem_count": int(round(10_000 * latent_mem.memorization_fraction)),
        }
        rows.append(row)
        print(
            f"d={d:3d} pixel={row['pixel_mem_count']:5d}/10000 "
            f"mean={row['pixel_mean_nn_ratio']:.6f} | "
            f"latent={row['latent_mem_count']:5d}/10000 "
            f"mean={row['latent_mean_nn_ratio']:.6f}",
            flush=True,
        )

    (OUT_DIR / "final_mem_bonnaire_10k_latent_summary.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
