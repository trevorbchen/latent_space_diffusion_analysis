from __future__ import annotations

import json
from pathlib import Path


D_LATENTS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


def main() -> None:
    out = Path("configs/celeba_hq_diverse1k_resnet_modernloss_5m.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    configs = []
    for d in D_LATENTS:
        configs.append(
            {
                "d_latent": d,
                "vae_checkpoint": f"vae_checkpoints/celeba_resnet_modernloss_d{d}/vae.pt",
                "out": f"results/celeba_hq_diverse1k_resnet_modernloss_5m/d{d}",
                "subset_path": "diagnostics/celeba_diverse_1k_mean_plus_sd/subset_indices.json",
                "n_train": 1000,
                "total_steps": 5_000_000,
                "eval_interval": 10_000,
                "mem_interval": 100_000,
                "fid_interval": 100_000,
                "fid_n_real": 1000,
                "fid_n_gen": 1000,
                "n_gen_samples": 1000,
                "n_sde_steps": 500,
                "batch_size": 256,
                "hidden": 256,
                "lr": 1e-4,
                "seed": 42,
            }
        )
    out.write_text(json.dumps(configs, indent=2) + "\n")
    print(out)


if __name__ == "__main__":
    main()
