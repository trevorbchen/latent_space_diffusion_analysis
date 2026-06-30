from __future__ import annotations

import json
from pathlib import Path


D_LATENTS = [5, 10, 15, 20, 25, 30, 40]
SEEDS = [42, 43, 44, 45, 46]


def main() -> None:
    config_root = Path("configs/mnist_n1k_fid100k_seedmajor")
    config_root.mkdir(parents=True, exist_ok=True)

    for seed in SEEDS:
        configs = []
        for d_latent in D_LATENTS:
            configs.append(
                {
                    "data": "mnist",
                    "model": "mlp",
                    "n_train": 1000,
                    "total_steps": 5_000_000,
                    "eval_interval": 10_000,
                    "mem_interval": 50_000,
                    "fid_interval": 100_000,
                    "hidden": 256,
                    "vae_checkpoint": f"vae_checkpoints/mnist_d{d_latent}/vae.pt",
                    "out": f"results/mnist_n1k_multiseed_fid100k/d{d_latent}_s{seed}",
                    "seed": seed,
                }
            )
        (config_root / f"seed{seed}.json").write_text(
            json.dumps(configs, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
