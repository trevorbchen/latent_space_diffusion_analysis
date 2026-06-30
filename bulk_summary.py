"""For each RFNN run with eigenvalues, summarize per-bulk max/min/mean and
the spectral gaps between bulks (signal->noise-dim, noise-dim->sample,
signal->sample). Uses the cumulative-count definition: top d_int
eigenvalues = signal bulk, next d_lat-d_int = noise-dim, next n = sample.
"""
import json, math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent

def summarize(eigs, d_int, d_lat, n):
    e = np.sort(np.asarray(eigs))[::-1]
    p = e.size
    bulks = {}
    bulks["signal"] = e[:d_int]
    bulks["noise_dim"] = e[d_int:d_lat] if d_lat > d_int else np.array([])
    sample_end = min(d_lat + n, p)
    bulks["sample"] = e[d_lat:sample_end]
    bulks["rank_null"] = e[sample_end:]
    out = {}
    for name, arr in bulks.items():
        if arr.size == 0:
            out[name] = {"size": 0}
            continue
        out[name] = {
            "size": int(arr.size),
            "max": float(arr[0]),
            "min": float(arr[-1]),
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
        }
    # Gaps (in decades, log10) between adjacent bulks
    def gap(a, b):
        if not a or not b or a.get("size", 0) == 0 or b.get("size", 0) == 0:
            return None
        return math.log10(max(a["min"], 1e-30)) - math.log10(max(b["max"], 1e-30))
    out["gap_signal_to_noise_dec"] = gap(out["signal"], out["noise_dim"])
    out["gap_noise_to_sample_dec"] = gap(out["noise_dim"], out["sample"])
    out["gap_sample_to_null_dec"] = gap(out["sample"], out["rank_null"])
    out["gap_signal_to_sample_dec"] = gap(out["signal"], out["sample"])
    return out

def main():
    summary = {}
    for results_root in sorted(ROOT.glob("results_rfnn_*")):
        if not results_root.is_dir():
            continue
        for run_dir in sorted(results_root.iterdir()):
            cfg_path = run_dir / "config.json"
            eigf = run_dir / "eigenvalues_pre.npy"
            if not cfg_path.exists() or not eigf.exists():
                continue
            cfg = json.loads(cfg_path.read_text())
            eigs = np.load(eigf)
            d_int = cfg["d_intrinsic"]
            d_lat = cfg["d_latent"]
            n = cfg["n"]
            sn = cfg["sigma_noise"]
            key = f"{results_root.name}/{run_dir.name}"
            summary[key] = {
                "d_int": d_int,
                "d_lat": d_lat,
                "n": n,
                "sigma_noise": sn,
                "p": int(eigs.size),
                "bulks": summarize(eigs, d_int, d_lat, n),
            }
    out = ROOT / "bulk_summary.json"
    out.write_text(json.dumps(summary, indent=2, default=str))
    print(f"Wrote {out}")

    # Also pretty-print headline numbers
    cols = [
        ("results_rfnn_exp2_sn001", 0.01),
        ("results_rfnn_exp2v3", 0.5),
        ("results_rfnn_exp2_wide", 0.5),
    ]
    for root_name, sn in cols:
        print(f"\n=== {root_name} (sn~{sn}) ===")
        print(f"  {'d_lat':>6} {'sig_max':>10} {'sig_min':>10} {'nd_max':>10} {'nd_min':>10} {'samp_max':>10} {'samp_min':>10}  {'g_sig_nd':>9} {'g_nd_sa':>9} {'g_sig_sa':>9}")
        runs = [v for k, v in summary.items() if k.startswith(root_name + "/")]
        for r in sorted(runs, key=lambda x: x["d_lat"]):
            b = r["bulks"]
            sig = b.get("signal", {})
            nd  = b.get("noise_dim", {})
            sa  = b.get("sample", {})
            def f(x): return f"{x:.3g}" if isinstance(x, (int, float)) else str(x)
            def fd(x): return f"{x:.2f}" if isinstance(x, (int, float)) else "-"
            print(f"  {r['d_lat']:>6} {f(sig.get('max', 0)):>10} {f(sig.get('min', 0)):>10} "
                  f"{f(nd.get('max', 0)):>10} {f(nd.get('min', 0)):>10} "
                  f"{f(sa.get('max', 0)):>10} {f(sa.get('min', 0)):>10} "
                  f"{fd(b.get('gap_signal_to_noise_dec')):>9} "
                  f"{fd(b.get('gap_noise_to_sample_dec')):>9} "
                  f"{fd(b.get('gap_signal_to_sample_dec')):>9}")

if __name__ == "__main__":
    main()
