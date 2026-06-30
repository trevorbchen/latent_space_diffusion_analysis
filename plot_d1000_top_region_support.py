"""Make a clearer d=1000 top-region diagnostic plot."""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).parent
RUN = ROOT / "clean_four_bulk_candidates" / "pratio64_d1000_top_regions"
EIGS = np.sort(np.load(RUN / "top_eigenvalues_d1000.npy"))[::-1]


def region(name, vals, color, y):
    vals = np.log10(np.maximum(vals, 1e-30))
    return {
        "name": name,
        "vals": vals,
        "color": color,
        "y": y,
        "min": float(vals.min()),
        "med": float(np.median(vals)),
        "max": float(vals.max()),
        "count": len(vals),
    }


def main():
    regions = [
        region("signal", EIGS[:5], "#e03131", 3),
        region("noise-dim", EIGS[5:1000], "#2f9e44", 2),
        region("sample", EIGS[1000:1500], "#1971c2", 1),
    ]

    fig, ax = plt.subplots(figsize=(6.0, 2.8))
    for r in regions:
        ax.hlines(r["y"], r["min"], r["max"], color=r["color"], lw=10, alpha=0.25)
        ax.plot(r["med"], r["y"], marker="|", color=r["color"], ms=22, mew=2)
        ax.scatter(r["vals"], np.full_like(r["vals"], r["y"]), s=8,
                   color=r["color"], alpha=0.35)
        ax.text(r["max"] + 0.05, r["y"],
                f"{r['name']} (n={r['count']})",
                va="center", ha="left", color=r["color"], fontsize=10)

    ax.set_yticks([])
    ax.set_xlabel(r"$\log_{10}\lambda_i(U)$")
    ax.set_title(r"$p=64d_{lat}$, $d_{lat}=1000$: top regions only")
    ax.grid(True, axis="x", alpha=0.25)
    ax.set_ylim(0.4, 3.6)
    fig.tight_layout()
    out = RUN / "top_region_support.png"
    fig.savefig(out, dpi=180)
    fig.savefig(out.with_suffix(".pdf"))
    print(out)


if __name__ == "__main__":
    main()
