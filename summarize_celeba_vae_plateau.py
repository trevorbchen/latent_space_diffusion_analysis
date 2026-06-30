from __future__ import annotations

import json
from pathlib import Path


D_LATENTS = [10, 15, 20, 25, 30, 35, 40, 45, 50]


def main() -> None:
    print(
        "d,epochs_ran,best_epoch,best_test,final_test,"
        "final_minus_best,last10_min,last10_max,last10_slope,stop_reason"
    )
    for d_latent in D_LATENTS:
        log_path = Path(f"vae_checkpoints/celeba_d{d_latent}/training.log")
        rows = []
        stop_reason = ""
        for line in log_path.read_text().splitlines():
            row = json.loads(line)
            if row.get("event") in ("config",):
                continue
            if row.get("event") == "early_stop":
                stop_reason = row.get("reason", "")
                continue
            if "epoch" in row:
                rows.append(row)

        tests = [r["test_loss"] for r in rows]
        best_epoch = min(range(len(tests)), key=lambda i: tests[i])
        best_test = tests[best_epoch]
        final_test = tests[-1]
        last = tests[-10:] if len(tests) >= 10 else tests
        denom = max(len(last) - 1, 1)
        last_slope = (last[-1] - last[0]) / denom
        print(
            f"{d_latent},"
            f"{len(rows)},"
            f"{best_epoch},"
            f"{best_test:.3f},"
            f"{final_test:.3f},"
            f"{final_test - best_test:.3f},"
            f"{min(last):.3f},"
            f"{max(last):.3f},"
            f"{last_slope:.3f},"
            f"{stop_reason}"
        )

        print("  last10", " ".join(f"{x:.2f}" for x in last))


if __name__ == "__main__":
    main()
