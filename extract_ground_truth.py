"""Walk results_*/ dirs, extract tau_gen / tau_mem / score_error / eigenvalue summaries
for every run. Emits ground_truth.json next to this script.

tau_gen := first step where test_loss is within 5% of per-run minimum.
tau_mem := first step where memorization_fraction > 0.01 (None = "never within budget").
"""
import json, os, glob, math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent

def tau_gen(rows):
    test = [r["test_loss"] for r in rows if "test_loss" in r]
    if not test:
        return None
    tmin = min(test)
    threshold = tmin * 1.05
    for r in rows:
        if r.get("test_loss", float("inf")) <= threshold:
            return r["step"]
    return None

def tau_mem(rows, frac=0.01):
    for r in rows:
        if r.get("memorization_fraction", 0.0) > frac:
            return r["step"]
    return None

def score_error_min(rows):
    vals = [r.get("score_error") for r in rows if r.get("score_error") is not None]
    return min(vals) if vals else None

def gen_gap_max(rows):
    vals = [r.get("gen_gap") for r in rows if r.get("gen_gap") is not None]
    return max(vals) if vals else None

def final_metrics(rows):
    if not rows:
        return {}
    return rows[-1]

def eigenvalue_summary(run_dir):
    summary = {}
    for tag in ("pre", "post"):
        f = run_dir / f"eigenvalues_{tag}.npy"
        if not f.exists():
            continue
        eigs = np.load(f)
        eigs_sorted = np.sort(eigs)[::-1]
        summary[tag] = {
            "n": int(eigs_sorted.size),
            "max": float(eigs_sorted[0]),
            "min": float(eigs_sorted[-1]),
            "median": float(np.median(eigs_sorted)),
            # Boundaries at d_int and d_lat read from config later
            "log10_top16": np.log10(np.maximum(eigs_sorted[:16], 1e-30)).tolist(),
        }
    return summary

def cliff_indices(eigs_sorted, d_int, d_lat):
    """For each candidate boundary index k, compute the log10 ratio
    eigs[k-1] / eigs[k] (the size of the cliff just past index k).
    Return ratios at the two predicted indices (d_int, d_lat) and the index of
    the largest cliff in [1, min(eigs.size-1, 2*d_lat))."""
    e = np.asarray(eigs_sorted)
    if e.size < 2:
        return {}
    log_e = np.log10(np.maximum(e, 1e-30))
    drops = -np.diff(log_e)  # drops[i] = log10(e[i] / e[i+1]); cliff between index i and i+1
    out = {}
    if d_int < drops.size:
        # Cliff "at index d_int" means between sorted positions d_int-1 and d_int (1-indexed)
        # In 0-indexed drops, that's drops[d_int-1].
        if d_int - 1 < drops.size:
            out["drop_at_dint"] = float(drops[d_int - 1])
    if d_lat - 1 < drops.size:
        out["drop_at_dlat"] = float(drops[d_lat - 1])
    upper = min(drops.size, max(2 * d_lat, d_lat + 4))
    region = drops[:upper]
    if region.size:
        out["argmax_cliff_in_region"] = int(np.argmax(region) + 1)  # 1-indexed cliff position
        out["max_cliff_in_region"] = float(np.max(region))
    return out

def collect():
    data = {}
    for results_dir in sorted(ROOT.glob("results_*")):
        if not results_dir.is_dir():
            continue
        for run_dir in sorted(results_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            cfg_path = run_dir / "config.json"
            metrics_path = run_dir / "metrics.jsonl"
            if not cfg_path.exists() or not metrics_path.exists():
                continue
            cfg = json.loads(cfg_path.read_text())
            rows = [json.loads(l) for l in metrics_path.read_text().splitlines() if l.strip()]
            entry = {
                "results_root": results_dir.name,
                "run_dir": run_dir.name,
                "d_intrinsic": cfg.get("d_intrinsic"),
                "d_latent": cfg.get("d_latent"),
                "n": cfg.get("n"),
                "sigma_noise": cfg.get("sigma_noise"),
                "sigma_signal": cfg.get("sigma_signal"),
                "scale": cfg.get("scale"),
                "k": cfg.get("k"),
                "hidden": cfg.get("hidden"),
                "p": cfg.get("p"),
                "total_steps": cfg.get("total_steps"),
                "n_rows": len(rows),
                "first_step": rows[0]["step"] if rows else None,
                "last_step": rows[-1]["step"] if rows else None,
                "tau_gen": tau_gen(rows),
                "tau_mem": tau_mem(rows),
                "score_error_min": score_error_min(rows),
                "gen_gap_max": gen_gap_max(rows),
                "final_train_loss": rows[-1].get("train_loss") if rows else None,
                "final_test_loss": rows[-1].get("test_loss") if rows else None,
                "final_score_error": rows[-1].get("score_error") if rows else None,
                "final_mem_frac": rows[-1].get("memorization_fraction") if rows else None,
                "final_mean_nn_ratio": rows[-1].get("mean_nn_ratio") if rows else None,
                "wall_time_total_sec": rows[-1].get("wall_time") if rows else None,
                "total_flops_final": rows[-1].get("total_flops") if rows else None,
            }
            eig = eigenvalue_summary(run_dir)
            if eig:
                entry["eigenvalues"] = eig
                # Compute cliff diagnostics on the post-training eigenvalues if present
                pre_path = run_dir / "eigenvalues_pre.npy"
                if pre_path.exists():
                    eigs = np.sort(np.load(pre_path))[::-1]
                    entry["eigenvalues"]["pre"]["cliffs"] = cliff_indices(
                        eigs, cfg.get("d_intrinsic", 0), cfg.get("d_latent", 0)
                    )
            data.setdefault(results_dir.name, []).append(entry)
    return data

if __name__ == "__main__":
    data = collect()
    out = ROOT / "ground_truth.json"
    out.write_text(json.dumps(data, indent=2))
    n_runs = sum(len(v) for v in data.values())
    print(f"Wrote {out} ({n_runs} runs across {len(data)} result roots).")
    for root, runs in data.items():
        cells = sorted({(r["d_intrinsic"], r["d_latent"], r["n"], r["sigma_noise"]) for r in runs})
        print(f"  {root}: {len(runs)} runs, {len(cells)} cells")
