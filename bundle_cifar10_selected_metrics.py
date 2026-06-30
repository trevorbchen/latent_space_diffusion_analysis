from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path("results/cifar10_beta005_diverse1k_bigmlp_sgd_lr001_m08_10k_5m")
OUT_ROOT = Path("diagnostics/cifar10_highd_selected_metrics") / "results" / ROOT.name
DIMS = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260]
SELECTED_RUNS = {
    20: [(42, "d20"), (43, "d20"), (44, "d20"), (45, "d20"), (46, "d20")],
    40: [(42, "d40"), (43, "d40"), (44, "d40"), (45, "d40"), (46, "d40")],
    60: [(42, "d60"), (43, "d60"), (44, "d60"), (45, "d60"), (46, "d60")],
    80: [(42, "d80"), (43, "d80"), (44, "d80"), (45, "d80"), (46, "d80")],
    100: [(42, "d100"), (43, "d100"), (44, "d100"), (45, "d100"), (46, "d100")],
    120: [(42, "d120"), (43, "d120"), (44, "d120"), (45, "d120"), (46, "d120")],
    140: [(42, "d140"), (43, "d140"), (44, "d140"), (45, "d140"), (46, "d140")],
    160: [(42, "d160"), (43, "d160"), (44, "d160"), (45, "d160"), (46, "d160")],
    180: [(42, "d180"), (43, "d180"), (44, "d180"), (45, "d180"), (46, "d180")],
    200: [(42, "d200"), (43, "d200"), (44, "d200"), (45, "d200"), (46, "d200")],
    220: [(42, "d220"), (43, "d220"), (44, "d220"), (45, "d220"), (47, "d220")],
    240: [(42, "d240"), (43, "d240"), (45, "d240"), (46, "d240"), (47, "d240")],
    260: [(43, "d260"), (47, "d260"), (48, "d260"), (49, "d260"), (50, "d260")],
}


def main() -> None:
    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
    for d in DIMS:
        for seed, folder in SELECTED_RUNS[d]:
            src = ROOT / f"seed{seed}" / folder
            dst = OUT_ROOT / f"seed{seed}" / folder
            dst.mkdir(parents=True, exist_ok=True)
            for name in ("metrics.jsonl", "config.json", "fid_retry.json"):
                path = src / name
                if path.exists():
                    shutil.copy2(path, dst / name)
            if not (dst / "metrics.jsonl").exists():
                raise FileNotFoundError(src / "metrics.jsonl")
    print(f"wrote {OUT_ROOT}")


if __name__ == "__main__":
    main()
