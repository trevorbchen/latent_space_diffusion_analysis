"""Figures + sanity checks from the CSVs written by compute_corrected_score_error.py."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))
from compute_corrected_score_error import SWEEPS  # noqa: E402

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e6e3", "grid.linewidth": 0.6,
                     "axes.edgecolor": "#8a8a86", "legend.frameon": False})


def agg(df, xcol, ycol):
    g = df.groupby(xcol)[ycol]
    return g.mean(), g.std(ddof=0), g.count()


def fig_final(name, df, xcol):
    fin = df[df.is_final == 1]
    fig, ax = plt.subplots(figsize=(4.6, 3.2), dpi=150)
    for ycol, col, lab in [("corrected_error_per_dim", BLUE, "corrected  (test_loss/Δ − C/d, fixed score)"),
                           ("logged_score_error_BUGGY", ORANGE, "logged score_error (buggy Qᵀ D⁻¹ Q)"),
                           ("mismatch_per_dim", AQUA, "model-free mismatch E‖s_true−s_bug‖²/d")]:
        m, s, n = agg(fin, xcol, ycol)
        ax.plot(m.index, m.values, "-o", color=col, lw=2, ms=5, label=lab)
        if n.max() > 1:
            ax.fill_between(m.index, (m - s).values, (m + s).values, color=col, alpha=0.18, lw=0)
    ax.axhline(0, color="#8a8a86", lw=0.8)
    ax.set_xlabel("latent width d_lat" if xcol == "d_lat" else "intrinsic dim d_int")
    ax.set_ylabel("score error per dim at t = 0.1")
    nseed = fin.seed.nunique()
    ax.set_title(f"{name}\nfinal step {int(fin.step.max())}, {nseed} seed(s), band = ±1 sd over seeds", fontsize=8)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    for ext in ("pdf", "png"): fig.savefig(OUT / f"fig_final_corrected_vs_{xcol}_{name}.{ext}")
    plt.close(fig)


def fig_vs_step(name, df, xcol):
    xs = sorted(df[xcol].unique())
    pick = [xs[0]] + [xs[len(xs) // 3], xs[2 * len(xs) // 3]] + [xs[-1]]
    pick = sorted(set(pick))[:4]
    fig, ax = plt.subplots(figsize=(4.6, 3.2), dpi=150)
    for x, col in zip(pick, [BLUE, ORANGE, AQUA, YELLOW]):
        sub = df[df[xcol] == x]
        m, s, n = agg(sub, "step", "corrected_error_per_dim")
        ax.plot(m.index, m.values, "-o", color=col, lw=1.6, ms=2.5, label=f"{xcol}={x}")
        if n.max() > 1:
            ax.fill_between(m.index, (m - s).values, (m + s).values, color=col, alpha=0.18, lw=0)
    ax.axhline(0, color="#8a8a86", lw=0.8)
    ax.set_xscale("log"); ax.set_xlabel("training step")
    ax.set_ylabel("corrected score error per dim (t = 0.1)")
    ax.set_title(f"{name}: corrected error vs step (mean ± sd over seeds)\nmarkers = logged evals (step 1, then every eval_interval)", fontsize=8)
    ax.legend(fontsize=7)
    fig.tight_layout()
    for ext in ("pdf", "png"): fig.savefig(OUT / f"fig_corrected_vs_step_{name}.{ext}")
    plt.close(fig)


def checks(name, df, consts, xcol):
    rep = {"sweep": name}
    rep["n_rows"] = len(df); rep["n_runs"] = len(consts)
    rep["min_corrected_error"] = float(df.corrected_error.min())
    neg = df[df.corrected_error < 0]
    rep["n_negative_rows"] = int(len(neg))
    if len(neg):
        rep["negative_rows"] = neg[["d_lat", "d_int", "seed", "step", "corrected_error_per_dim"]].to_dict("records")[:10]
    # eps replication diagnostic: step-1 test_loss minus regenerated ||eps||^2/d
    r = np.array([c["test_loss_step1"] - c["eps_sq_per_dim_sample"] for c in consts])
    rep["eps_check_resid_mean"] = float(r.mean()); rep["eps_check_resid_std"] = float(r.std())
    rep["eps_check_resid_maxabs"] = float(np.abs(r).max())
    # C precision (population MC)
    rel = np.array([c["C_fix_pop_se"] / c["C_fix_pop"] for c in consts])
    rep["C_pop_rel_se_max_pct"] = float(100 * rel.max())
    rel_s = np.array([c["C_fix_sample_se"] / c["C_fix_sample"] for c in consts])
    rep["C_sample_rel_se_max_pct"] = float(100 * rel_s.max())
    # identity checks on the true score: E|s_cond|^2 = d/Delta ; cross term = 0
    rep["max_abs_cross_fix_pop_over_C"] = float(max(abs(c["cross_fix_pop"]) / c["C_fix_pop"] for c in consts))
    rep["max_rel_err_S2cond_vs_d_over_delta"] = float(max(
        abs(c["S2_cond_pop"] - c["d_latent"] / c["delta_t"]) / (c["d_latent"] / c["delta_t"]) for c in consts))
    # bug inert at d_lat == d_int
    inert = [c for c in consts if c["d_latent"] == c["d_intrinsic"]]
    rep["inert_cases"] = [{"d": c["d_latent"], "seed": c["seed"],
                           "C_fix_sample": c["C_fix_sample"], "C_bug_sample": c["C_bug_sample"],
                           "M_sample": c["M_sample"]} for c in inert]
    tab = (df[df.is_final == 1].groupby(xcol)
           .agg(C_per_dim=("C_per_dim", "mean"), C_bug_per_dim=("C_bug_per_dim", "mean"),
                mismatch_per_dim=("mismatch_per_dim", "mean"),
                corrected_mean=("corrected_error_per_dim", "mean"),
                corrected_sd=("corrected_error_per_dim", lambda v: v.std(ddof=0)),
                buggy_mean=("logged_score_error_BUGGY", "mean"),
                buggy_minus_corrected=("logged_score_error_BUGGY", "mean"),
                test_loss=("test_loss", "mean"), n_seeds=("seed", "nunique")))
    tab["buggy_minus_corrected"] = tab["buggy_mean"] - tab["corrected_mean"]
    # zero-score baseline E||s_true||^2/d (population MC) and best-over-training corrected error
    base = pd.DataFrame([{xcol: c["d_latent" if xcol == "d_lat" else "d_intrinsic"],
                          "zero_score_baseline": c["S2_fix_pop"] / c["d_latent"]} for c in consts]).groupby(xcol).mean()
    tab = tab.join(base)
    curve = df.groupby([xcol, "step"])["corrected_error_per_dim"].mean()
    tab["min_over_steps"] = curve.groupby(level=0).min()
    tab["argmin_step"] = curve.groupby(level=0).idxmin().map(lambda t: t[1])
    rep["final_table"] = tab.reset_index().round(4).to_dict("records")
    if len(tab) > 2:
        rep["corr_buggy_vs_mismatch_across_x"] = float(np.corrcoef(tab.buggy_mean, tab.mismatch_per_dim)[0, 1])
        rep["corr_buggy_minus_corr_vs_mismatch"] = float(np.corrcoef(tab.buggy_minus_corrected, tab.mismatch_per_dim)[0, 1])
    return rep, tab


if __name__ == "__main__":
    reports = {}
    for name, (sdir, xvar) in SWEEPS.items():
        p = OUT / f"corrected_score_error_{name}.csv"
        if not p.exists(): print("missing", p); continue
        df = pd.read_csv(p); consts = json.load(open(OUT / f"constants_{name}.json"))
        xcol = "d_lat" if xvar == "d_latent" else "d_int"
        fig_final(name, df, xcol); fig_vs_step(name, df, xcol)
        rep, tab = checks(name, df, consts, xcol)
        reports[name] = rep
        tab.round(4).to_csv(OUT / f"final_table_{name}.csv")
        print(f"\n### {name}  (x = {xcol})")
        print(tab.round(4).to_string())
        print({k: v for k, v in rep.items() if k not in ("final_table", "negative_rows")})
    json.dump(reports, open(OUT / "sanity_checks.json", "w"), indent=1)
