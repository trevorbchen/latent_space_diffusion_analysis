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
OUT_DIR = Path("diagnostics/celeba_diverse_final_mem_bonnaire")


def old_anchor_ratio_memorization(generated: torch.Tensor, train_data: torch.Tensor) -> tuple[float, float]:
    train_dists = torch.cdist(train_data, train_data)
    train_dists.fill_diagonal_(float("inf"))
    dists = torch.cdist(generated, train_data)
    nn1_dists, nn1_idx = dists.min(dim=1)
    nn2_dists = train_dists[nn1_idx].min(dim=1).values
    ratio = nn1_dists / (nn2_dists + 1e-10)
    return ratio.mean().item(), (ratio < 1 / 3).float().mean().item()


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


def load_subset_images(image_size: int, n: int = 1000) -> torch.Tensor:
    indices = json.loads(SUBSET_PATH.read_text())[:n]
    dataset = TarImageDataset(TRAIN_TAR, image_size=image_size)
    return torch.stack([dataset[i][0] for i in indices], dim=0)


def load_diffusion_model(path: Path, d_latent: int, device: torch.device) -> models.MLPScore:
    model = models.MLPScore(d_latent, hidden=256).to(device)
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(12345)

    rows = []
    for d in D_LATENTS:
        vae = load_vae(f"vae_checkpoints/celeba_resnet_modernloss_d{d}/vae.pt", device)
        train_images = load_subset_images(vae.cfg.image_size)
        train_flat = train_images.flatten(1)
        model = load_diffusion_model(RESULT_ROOT / f"d{d}" / "last_model.pt", d, device)
        with torch.no_grad():
            gen_latents = diffusion.euler_maruyama(
                model,
                1000,
                d,
                n_steps=500,
                t_max=3.0,
                t_min=0.01,
                device=device,
            )
            gen_pixels = vae.decode(gen_latents).cpu()
        bonnaire = metrics.nn_ratio_memorization(gen_pixels.flatten(1), train_flat)
        old_mean, old_frac = old_anchor_ratio_memorization(gen_pixels.flatten(1), train_flat)
        row = {
            "d_latent": d,
            "bonnaire_mean_nn_ratio": bonnaire.mean_nn_ratio,
            "bonnaire_mem_fraction": bonnaire.memorization_fraction,
            "bonnaire_mem_count": int(round(1000 * bonnaire.memorization_fraction)),
            "old_anchor_mean_nn_ratio": old_mean,
            "old_anchor_mem_fraction": old_frac,
            "old_anchor_mem_count": int(round(1000 * old_frac)),
        }
        rows.append(row)
        print(
            f"d={d:3d} bonnaire={row['bonnaire_mem_count']:4d}/1000 "
            f"mean={row['bonnaire_mean_nn_ratio']:.6f} | "
            f"old={row['old_anchor_mem_count']:4d}/1000 "
            f"mean={row['old_anchor_mean_nn_ratio']:.6f}",
            flush=True,
        )

    (OUT_DIR / "final_mem_bonnaire_summary.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
