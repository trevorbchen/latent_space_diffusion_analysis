"""Recover a CORRECT analytic score error from the saved synthetic-MLP sweeps.

The logged `score_error` column compared the model to a true score built with a
transposed covariance (Q^T D^-1 Q instead of Q D^-1 Q^T).  Training never used
the true score, and `test_loss` is a valid held-out DSM loss, so the correct
score error is recovered through the DSM identity

    E||s_theta - s_cond||^2 = E||s_theta - s_true||^2 + E||s_true - s_cond||^2,
    s_cond := grad_{x_t} log p(x_t|x_0) = -eps / sqrt(Delta_t),

(the cross term vanishes in expectation because E[s_cond | x_t] = s_true(x_t)).

Matching code/experiment_v2.py `evaluate()` (lines 316-341):
    t_eval = config.t_eval = 0.1 (single fixed t, NOT the training U[t_min,t_max])
    x_t   = e^{-t} x_test + sqrt(Delta_t) eps,  Delta_t = 1 - e^{-2t}
    test_loss = mean_i ||sqrt(Delta_t) s_theta + eps_i||^2 / d
              = (Delta_t / d) * mean_i ||s_theta - s_cond_i||^2
    x_test: generate_test_samples(..., n=2048, seed=9999)   (line 415-419)
    eps   : torch.randn_like(x_test), first torch draw after torch.manual_seed(seed) (395, 420)
So      corrected_error(total)   = test_loss * d / Delta_t - C_sample
        corrected_error_per_dim  = test_loss / Delta_t - C_sample / d
with C_sample = mean_i ||s_true(x_t,i) - s_cond,i||^2 on the SAME (x_test, eps).
"""
import argparse, glob, json, math, os, sys
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "code"))
from experiment_v2 import (generate_data, generate_test_samples, true_score,   # noqa: E402
                           precompute_score_params)
OUT = REPO / "_next_steps" / "e_scoreerr"

SWEEPS = {
    # name: (dir, x-variable)
    "paper_h256_5seed_sn05_5M": ("multiseed_runs/exp2_mlp_dlat_sn05_5m_bonnaire_hidden256", "d_latent"),
    "scaled_8dlat_5seed_sn05_5M": ("multiseed_runs/exp2_mlp_dlat_sn05_5m_combined", "d_latent"),
    "exp2_h256_sn05_300k": ("results_mlp_exp2_sn05", "d_latent"),
    "exp2_scaled_sn05_300k": ("results_mlp_exp2_scaled_sn05", "d_latent"),
    "exp2_h256_sn001_300k": ("results_mlp_exp2_sn001", "d_latent"),
    "exp2_scaled_sn001_300k": ("results_mlp_exp2_scaled_sn001", "d_latent"),
    "exp3_dint_h256_sn05_300k": ("results_mlp_exp3_sn05", "d_intrinsic"),
    "exp3_dint_h256_sn001_300k": ("results_mlp_exp3_sn001", "d_intrinsic"),
}


def buggy_sigma_t_inv(d_int, d_lat, Q, sn, t, ss):
    """The pre-fix precision: Q^T diag(1/sigma_t) Q (wrong frame)."""
    delta_t = 1 - math.exp(-2 * t); e2 = math.exp(-2 * t)
    diag = torch.ones(d_lat); diag[:d_int] = ss ** 2
    if d_lat > d_int: diag[d_int:] = sn ** 2
    std = delta_t + e2 * diag
    Qt = Q.float()
    return Qt.T @ torch.diag(1.0 / std) @ Qt, torch.log(std).sum()


def sq(a): return (a ** 2).sum(-1)


def run_constants(cfg, n_pop, chunk):
    d_int, d_lat, k = cfg["d_intrinsic"], cfg["d_latent"], cfg["k"]
    sn, ss, scale, n, seed, t = (cfg["sigma_noise"], cfg["sigma_signal"], cfg["scale"],
                                 cfg["n"], cfg["seed"], cfg["t_eval"])
    delta = 1 - math.exp(-2 * t); e_neg_t = math.exp(-t)
    # --- replicate run_experiment() call order exactly (lines 395-420) ---
    torch.manual_seed(seed); np.random.seed(seed)
    data, labels, means, Q = generate_data(n, d_int, d_lat, k, sn, scale, seed, sigma_signal=ss)
    _ = torch.linalg.svdvals(data)
    si_fix, ld_fix = precompute_score_params(d_int, d_lat, Q, sn, t, sigma_signal=ss)
    x_test = generate_test_samples(means.cpu(), k, d_int, d_lat, Q, sn, n=2048, seed=9999, sigma_signal=ss)
    eps = torch.randn_like(x_test)
    si_bug, ld_bug = buggy_sigma_t_inv(d_int, d_lat, Q, sn, t, ss)

    def consts(x0, e):
        xt = e_neg_t * x0 + math.sqrt(delta) * e
        s_cond = -e / math.sqrt(delta)
        s_fix = true_score(xt, t, means, sigma_t_inv=si_fix, log_det_sigma_t=ld_fix)
        s_bug = true_score(xt, t, means, sigma_t_inv=si_bug, log_det_sigma_t=ld_bug)
        return {"C_fix": sq(s_fix - s_cond), "C_bug": sq(s_bug - s_cond),
                "M": sq(s_fix - s_bug), "S2_fix": sq(s_fix), "S2_cond": sq(s_cond),
                "cross_fix": ((s_fix - s_cond) * s_fix).sum(-1)}

    out = {"d_intrinsic": d_int, "d_latent": d_lat, "seed": seed, "hidden": cfg["hidden"],
           "sigma_noise": sn, "t_eval": t, "delta_t": delta, "n_test": 2048,
           "eps_sq_per_dim_sample": sq(eps).mean().item() / d_lat}
    with torch.no_grad():
        smp = consts(x_test, eps)
        for kk, v in smp.items():
            out[f"{kk}_sample"] = v.mean().item()
            out[f"{kk}_sample_se"] = v.std(unbiased=True).item() / math.sqrt(len(v))
        # --- population constant: fresh draws from the same distribution ---
        gen = torch.Generator().manual_seed(10_000 + seed)
        acc = {kk: [] for kk in smp}
        done = 0; j = 0
        while done < n_pop:
            m = min(chunk, n_pop - done)
            x0 = generate_test_samples(means.cpu(), k, d_int, d_lat, Q, sn, n=m,
                                       seed=1_000_000 + 1000 * seed + j, sigma_signal=ss)
            e = torch.randn(x0.shape, generator=gen)
            for kk, v in consts(x0, e).items(): acc[kk].append(v)
            done += m; j += 1
        for kk, vs in acc.items():
            v = torch.cat(vs)
            out[f"{kk}_pop"] = v.mean().item()
            out[f"{kk}_pop_se"] = v.std(unbiased=True).item() / math.sqrt(len(v))
        out["n_pop"] = n_pop
    return out


def process_sweep(name, sdir, xvar, n_pop, chunk):
    rows, consts = [], []
    run_dirs = sorted(glob.glob(str(REPO / sdir / "*" / "config.json")))
    for cpath in run_dirs:
        rdir = Path(cpath).parent
        cfg = json.load(open(cpath))
        mets = [json.loads(l) for l in open(rdir / "metrics.jsonl") if l.strip()]
        if not mets or "test_loss" not in mets[-1]:
            print(f"  skip {rdir} (no test_loss)"); continue
        c = run_constants(cfg, n_pop, chunk)
        c["run_dir"] = str(rdir.relative_to(REPO)); c["n_eval_rows"] = len(mets)
        c["final_step"] = mets[-1]["step"]; c["test_loss_step1"] = mets[0]["test_loss"]
        consts.append(c)
        d, delta = cfg["d_latent"], c["delta_t"]
        for m in mets:
            corr = m["test_loss"] * d / delta - c["C_fix_sample"]
            corr_pop = m["test_loss"] * d / delta - c["C_fix_pop"]
            rows.append({
                "d_lat": d, "d_int": cfg["d_intrinsic"], "seed": cfg["seed"], "hidden": cfg["hidden"],
                "sigma_noise": cfg["sigma_noise"], "step": m["step"], "is_final": int(m["step"] == c["final_step"]),
                "test_loss": m["test_loss"], "train_loss": m["train_loss"],
                "C": c["C_fix_sample"], "C_per_dim": c["C_fix_sample"] / d,
                "corrected_error": corr, "corrected_error_per_dim": corr / d,
                "corrected_error_per_dim_popC": corr_pop / d,
                "logged_score_error_BUGGY": m["score_error"],
                "C_bug_per_dim": c["C_bug_sample"] / d, "mismatch_per_dim": c["M_sample"] / d,
                "memorization_fraction": m.get("memorization_fraction"),
            })
        print(f"  {name}: d_int={cfg['d_intrinsic']} d_lat={d} seed={cfg['seed']} "
              f"C/d={c['C_fix_sample']/d:.4f} (pop {c['C_fix_pop']/d:.4f} +- {c['C_fix_pop_se']/d:.4f}) "
              f"M/d={c['M_sample']/d:.4f} final corr/d={rows[-1]['corrected_error_per_dim']:.4f} "
              f"buggy={rows[-1]['logged_score_error_BUGGY']:.4f}", flush=True)
    import csv
    cols = list(rows[0].keys())
    with open(OUT / f"corrected_score_error_{name}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
    json.dump(consts, open(OUT / f"constants_{name}.json", "w"), indent=1)
    return rows, consts


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweeps", nargs="+", default=list(SWEEPS))
    ap.add_argument("--n_pop", type=int, default=200_000)
    ap.add_argument("--chunk", type=int, default=20_000)
    ap.add_argument("--only_first", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(max(1, os.cpu_count() // 2))
    for name in a.sweeps:
        sdir, xvar = SWEEPS[name]
        print(f"== {name} ({sdir})", flush=True)
        if a.only_first:
            cpath = sorted(glob.glob(str(REPO / sdir / "*" / "config.json")))[0]
            c = run_constants(json.load(open(cpath)), a.n_pop, a.chunk); print(json.dumps(c, indent=1)); break
        process_sweep(name, sdir, xvar, a.n_pop, a.chunk)
