from __future__ import annotations

import json
from pathlib import Path


D_LATENTS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
ROOT = Path("vae_checkpoints")
PREFIX = "celeba_resnet_modernloss_d"


def main() -> None:
    rows = []
    for d in D_LATENTS:
        out = ROOT / f"{PREFIX}{d}"
        log_path = out / "training.log"
        train_stdout = out / "train_stdout.log"
        epoch_rows = []
        if log_path.exists():
            for line in log_path.read_text().splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "epoch" in row and "test_loss" in row:
                    epoch_rows.append(row)
        final_row = epoch_rows[-1] if epoch_rows else {}
        saved_line = ""
        if train_stdout.exists():
            saved = [line for line in train_stdout.read_text().splitlines() if "Saved VAE" in line]
            saved_line = saved[-1] if saved else ""
        rows.append(
            {
                "d_latent": d,
                "checkpoint_exists": (out / "vae.pt").exists(),
                "final_epoch": final_row.get("epoch"),
                "best_epoch": final_row.get("best_epoch"),
                "final_test_loss": final_row.get("test_loss"),
                "final_test_recon": final_row.get("test_recon"),
                "final_test_mse": final_row.get("test_mse"),
                "final_test_l1": final_row.get("test_l1"),
                "final_test_kl": final_row.get("test_kl"),
                "best_test_loss": final_row.get("best_test_loss"),
                "saved_line": saved_line,
            }
        )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
