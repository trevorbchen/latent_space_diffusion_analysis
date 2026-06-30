from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from lib import metrics
from standard_vae import StandardConvVAE, StandardVAEConfig
from train_vae_celeba_standard_tar import TarImageDataset
from train_vae_cifar import CIFAR10ImageDataset


def load_vae(path: str, device: torch.device) -> StandardConvVAE:
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


def load_images(dataset_name: str, indices: list[int], train_tar: str) -> torch.Tensor:
    if dataset_name == "celeba":
        dataset = TarImageDataset(train_tar, image_size=32)
    elif dataset_name == "cifar10":
        dataset = CIFAR10ImageDataset("data", train=True, download=True)
    else:
        raise ValueError(f"unknown dataset {dataset_name}")
    return torch.stack([dataset[i][0] for i in indices], dim=0)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["celeba", "cifar10"], required=True)
    p.add_argument("--vae_prefix", required=True)
    p.add_argument("--dims", required=True, help="comma-separated latent dimensions")
    p.add_argument("--subset_path", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=1000)
    p.add_argument(
        "--train_tar",
        default="/overflow/data/cerberus/celeba_hq_256_partitioned_ws/train/celeba_hq_256-train-0000.tar",
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    dims = [int(x) for x in args.dims.split(",") if x.strip()]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    indices = json.loads(Path(args.subset_path).read_text())[: args.n]
    images = load_images(args.dataset, indices, args.train_tar)
    flat = images.flatten(1)
    train_dists = torch.cdist(flat, flat)
    train_dists.fill_diagonal_(float("inf"))

    rows = []
    for d_latent in dims:
        ckpt_path = f"{args.vae_prefix}{d_latent}/vae.pt"
        vae = load_vae(ckpt_path, device)
        recon_chunks = []
        with torch.no_grad():
            for chunk in images.split(128):
                mu, _ = vae.encode(chunk.to(device))
                recon_chunks.append(vae.decode(mu).cpu())
        recon = torch.cat(recon_chunks, dim=0)
        result = metrics.nn_ratio_memorization(
            recon.flatten(1),
            flat,
            train_dists=train_dists,
        )
        diff = (recon - images).flatten(1)
        mse = diff.square().mean(dim=1)
        l2 = diff.square().sum(dim=1).sqrt()
        row = {
            "dataset": args.dataset,
            "vae_prefix": args.vae_prefix,
            "d_latent": d_latent,
            "n": len(indices),
            "threshold": 1.0 / 3.0,
            "memorized_count": int(round(result.memorization_fraction * len(indices))),
            "memorized_fraction": float(result.memorization_fraction),
            "mean_nn_ratio": float(result.mean_nn_ratio),
            "mean_pixel_mse": float(mse.mean()),
            "std_pixel_mse": float(mse.std(unbiased=True)),
            "mean_pixel_l2": float(l2.mean()),
            "std_pixel_l2": float(l2.std(unbiased=True)),
        }
        rows.append(row)
        print(
            f"d={d_latent} mem={row['memorized_fraction']:.4f} "
            f"nn_ratio={row['mean_nn_ratio']:.4f} mse={row['mean_pixel_mse']:.5f}"
        )

    (out_dir / "subset_indices.json").write_text(json.dumps(indices))
    (out_dir / "reconstruction_mem_summary.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
