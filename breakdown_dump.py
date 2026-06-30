"""Dump 3-metric x 2-sigma breakdowns into the experiment-specific
folders at the project root.

Writes one `breakdown.md` per experiment:
  exp2_v2/breakdown.md     -- scaled MLP, vary d_latent, fixed d_int=5
  exp2_rfnn/breakdown.md   -- RFNN,        vary d_latent, fixed d_int=5
  exp3_v2/breakdown.md     -- scaled MLP, vary d_int,    fixed d_latent=20
  exp3_rfnn/breakdown.md   -- RFNN,        vary d_int,    fixed d_latent=20
"""
import json, os
from pathlib import Path

ROOT = Path(__file__).parent
GT = json.loads((ROOT / "ground_truth.json").read_text())

METRICS = [
    ("score_error_min", "min score error"),
    ("final_train_loss", "final train loss"),
    ("final_test_loss",  "final test loss"),
]


def fmt(v):
    if v is None:
        return "--"
    if abs(v) < 1e-3 or abs(v) > 1000:
        return f"{v:.2e}"
    return f"{v:.4f}"


def render_table(rows_by_sigma, sweep_key, sigmas, sweep_label):
    """rows_by_sigma: dict[sigma -> dict[sweep_value -> run]]."""
    keys = sorted(set().union(*[set(rows.keys()) for rows in rows_by_sigma.values()]))
    out = []
    for metric, label in METRICS:
        out.append(f"### {label}\n")
        header = f"| {sweep_label} | " + " | ".join(f"sigma={s}" for s in sigmas) + " |"
        sep = "|" + "|".join(["---"] * (len(sigmas) + 1)) + "|"
        out.append(header)
        out.append(sep)
        for k in keys:
            cells = []
            for s in sigmas:
                run = rows_by_sigma[s].get(k)
                cells.append(fmt(run.get(metric)) if run else "--")
            out.append(f"| {k} | " + " | ".join(cells) + " |")
        out.append("")
    return "\n".join(out)


def run(spec):
    out_dir = ROOT / spec["folder"]
    out_dir.mkdir(parents=True, exist_ok=True)

    rows_by_sigma = {}
    for sigma, root_name in spec["roots"].items():
        if root_name not in GT:
            print(f"WARN: results root {root_name} not in ground_truth.json")
            continue
        rows_by_sigma[sigma] = {r[spec["sweep_key"]]: r for r in GT[root_name]}

    body = [
        f"# {spec['title']}\n",
        f"_{spec['preamble']}_\n",
        render_table(rows_by_sigma, spec["sweep_key"],
                     sorted(spec["roots"].keys()), spec["sweep_label"]),
    ]
    out = out_dir / "breakdown.md"
    out.write_text("\n".join(body))
    print(f"wrote {out}")


SPECS = [
    {
        "folder": "exp2_v2",
        "title": "Exp 2 (scaled MLP) breakdown",
        "preamble": "Vary d_latent at fixed d_intrinsic=5, n=500, h=8*d_latent.",
        "sweep_key": "d_latent",
        "sweep_label": "d_latent",
        "roots": {
            0.01: "results_mlp_exp2_scaled_sn001",
            0.5:  "results_mlp_exp2_scaled_sn05",
        },
    },
    {
        "folder": "exp2_rfnn",
        "title": "Exp 2 (RFNN) breakdown",
        "preamble": "Vary d_latent at fixed d_intrinsic=5, n=500, p=64*d_latent, t_fixed=0.01.",
        "sweep_key": "d_latent",
        "sweep_label": "d_latent",
        "roots": {
            0.01: "results_rfnn_exp2_sn001",
            0.5:  "results_rfnn_exp2v3",
        },
    },
    {
        "folder": "exp3_v2",
        "title": "Exp 3 (scaled MLP) breakdown",
        "preamble": "Vary d_intrinsic at fixed d_latent=20, n=500, h=8*d_latent.",
        "sweep_key": "d_intrinsic",
        "sweep_label": "d_intrinsic",
        "roots": {
            0.01: "results_mlp_exp3_sn001",
            0.5:  "results_mlp_exp3_sn05",
        },
    },
    {
        "folder": "exp3_rfnn",
        "title": "Exp 3 (RFNN) breakdown",
        "preamble": "Vary d_intrinsic at fixed d_latent=20, n=500, p=64*d_latent, t_fixed=0.01.",
        "sweep_key": "d_intrinsic",
        "sweep_label": "d_intrinsic",
        "roots": {
            0.01: "results_rfnn_exp3_sn001",
            0.5:  "results_rfnn_exp3",
        },
    },
]

if __name__ == "__main__":
    for spec in SPECS:
        run(spec)
