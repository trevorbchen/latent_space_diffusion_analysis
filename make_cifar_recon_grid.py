from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torchvision.utils import save_image

from run_cifar_diverse_diffusion import load_standard_vae
from train_vae_cifar import CIFAR10ImageDataset


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--d_latent", type=int, required=True)
    p.add_argument("--vae_checkpoint", default=None)
    p.add_argument("--subset_path", default="diagnostics/cifar10_diverse_1k_mean_plus_sd/subset_indices.json")
    p.add_argument("--out_dir", default="diagnostics/cifar10_vae_recon_compare")
    p.add_argument("--n", type=int, default=16)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = args.vae_checkpoint or f"vae_checkpoints/cifar10_resnet_modernloss_d{args.d_latent}/vae.pt"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vae = load_standard_vae(ckpt, device)
    dataset = CIFAR10ImageDataset("data", train=True, download=True)
    indices = json.loads(Path(args.subset_path).read_text())[: args.n]
    images = torch.stack([dataset[i][0] for i in indices], dim=0).to(device)
    labels = [int(dataset[i][1]) for i in indices]

    with torch.no_grad():
        mu, logvar = vae.encode(images)
        recon = vae.decode(mu)

    diff = (recon - images).detach().cpu()
    per_sample_mse = diff.pow(2).flatten(1).mean(dim=1)
    per_sample_l2 = diff.flatten(1).norm(dim=1)
    grid = torch.empty((2 * args.n, *images.shape[1:]), device="cpu")
    grid[0::2] = images.detach().cpu()
    grid[1::2] = recon.detach().cpu()

    png = out_dir / f"cifar10_d{args.d_latent}_encode_decode_pairs.png"
    save_image((grid * 0.5 + 0.5).clamp(0, 1), png, nrow=8, padding=2)
    summary = {
        "d_latent": args.d_latent,
        "vae_checkpoint": ckpt,
        "subset_path": args.subset_path,
        "indices": indices,
        "labels": labels,
        "n": args.n,
        "mean_pixel_mse": float(per_sample_mse.mean()),
        "std_pixel_mse": float(per_sample_mse.std(unbiased=True)),
        "mean_pixel_l2": float(per_sample_l2.mean()),
        "std_pixel_l2": float(per_sample_l2.std(unbiased=True)),
        "per_sample_mse": [float(x) for x in per_sample_mse],
        "per_sample_l2": [float(x) for x in per_sample_l2],
        "grid": str(png),
    }
    (out_dir / f"cifar10_d{args.d_latent}_encode_decode_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
