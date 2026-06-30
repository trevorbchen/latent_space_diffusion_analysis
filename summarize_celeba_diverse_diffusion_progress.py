from __future__ import annotations

import json
from pathlib import Path


D_LATENTS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
ROOT = Path("results/celeba_hq_diverse1k_resnet_modernloss_5m")
TARGET = 5_000_000


def main() -> None:
    for d in D_LATENTS:
        metrics = ROOT / f"d{d}" / "metrics.jsonl"
        last = None
        n_rows = 0
        n_fid = 0
        if metrics.exists():
            for line in metrics.read_text().splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "step" in row:
                    last = row
                    n_rows += 1
                    if row.get("fid") is not None:
                        n_fid += 1
        step = int(last["step"]) if last else 0
        pct = 100.0 * step / TARGET
        fid = last.get("fid") if last else None
        mem = last.get("memorization_fraction_pixel") if last else None
        loss = last.get("train_loss") if last else None
        print(
            f"d={d:3d} step={step:7d}/{TARGET} ({pct:5.1f}%) "
            f"rows={n_rows:4d} fid_points={n_fid:3d} "
            f"loss={loss if loss is not None else 'NA'} "
            f"fid={fid if fid is not None else 'NA'} "
            f"mem={mem if mem is not None else 'NA'}"
        )


if __name__ == "__main__":
    main()
