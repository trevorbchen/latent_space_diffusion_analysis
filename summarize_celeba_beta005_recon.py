from __future__ import annotations

import json
from pathlib import Path


print("d,final_epoch,final_train_recon,final_test_recon,best_epoch,best_test_loss,best_test_recon,best_test_kl")
for d_latent in [10, 15, 20, 25, 30, 35, 40, 45, 50]:
    path = Path(f"vae_checkpoints/celeba_standard_beta005_d{d_latent}/training.log")
    rows = []
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if "epoch" in row and "test_recon" in row:
            rows.append(row)
    final = rows[-1]
    best = min(rows, key=lambda row: row["test_loss"])
    print(
        f"{d_latent},"
        f"{final['epoch']},"
        f"{final['train_recon']:.3f},"
        f"{final['test_recon']:.3f},"
        f"{best['epoch']},"
        f"{best['test_loss']:.3f},"
        f"{best['test_recon']:.3f},"
        f"{best['test_kl']:.3f}"
    )
