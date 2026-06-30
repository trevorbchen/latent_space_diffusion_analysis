from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DIMS = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260]
THRESHOLDS = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75]
T95_N5 = 2.7764451051977987
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


def t_label(t: float) -> str:
    return str(t).replace(".", "p")


def finite_or_nan(value) -> float:
    if value is None:
        return math.nan
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out if math.isfinite(out) else math.nan


def mean_ci(vals: list[float]) -> tuple[float, float, float, int]:
    arr = np.asarray([v for v in vals if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return math.nan, math.nan, math.nan, 0
    mean = float(arr.mean())
    sd = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    ci = T95_N5 * sd / math.sqrt(arr.size) if arr.size > 1 else 0.0
    return mean, sd, ci, int(arr.size)


def pava_non_decreasing(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    blocks: list[dict[str, object]] = []
    for i, (v, w) in enumerate(zip(values, weights)):
        blocks.append({"idx": [i], "value": float(v), "weight": float(max(w, 1e-12))})
        while len(blocks) >= 2 and blocks[-2]["value"] > blocks[-1]["value"]:
            b2 = blocks.pop()
            b1 = blocks.pop()
            weight = float(b1["weight"]) + float(b2["weight"])
            value = (float(b1["value"]) * float(b1["weight"]) + float(b2["value"]) * float(b2["weight"])) / weight
            blocks.append({"idx": list(b1["idx"]) + list(b2["idx"]), "value": value, "weight": weight})
    out = np.empty_like(values, dtype=float)
    for block in blocks:
        for idx in block["idx"]:
            out[idx] = float(block["value"])
    return out


def pressure_u(eig: np.ndarray, u_values, weights: np.ndarray | None = None) -> np.ndarray:
    eig = np.asarray(eig, dtype=float)
    u = np.asarray(u_values, dtype=float)
    if weights is None:
        w = np.ones_like(eig)
    else:
        w = np.asarray(weights, dtype=float).copy()
        if not np.isfinite(w).all() or float(w.sum()) <= 0:
            w = np.ones_like(eig)
    w = w / float(w.sum())
    x = np.outer(u, np.maximum(eig, 0.0))
    gains = 1.0 - np.exp(-np.clip(x, 0.0, 60.0))
    return (gains * gains) @ w


def pressure(eig: np.ndarray, steps, kappa: float, weights: np.ndarray | None = None) -> np.ndarray:
    return pressure_u(eig, np.asarray(steps, dtype=float) * kappa, weights=weights)


def predicted_mem_from_pressure(
    p_values: np.ndarray,
    theta_map: dict[float, float],
    calibrated_thresholds: set[float],
) -> np.ndarray:
    """Map spectral pressure to a continuous memorization surrogate.

    The calibrated model fundamentally predicts hitting times for fixed
    memorization thresholds. This interpolation is just a visualization helper:
    it turns those threshold anchors into a smooth-ish predicted mem curve.
    """
    anchors: dict[float, float] = {0.0: 0.0}
    for q, theta in theta_map.items():
        if float(q) not in calibrated_thresholds:
            continue
        theta = float(theta)
        if not np.isfinite(theta):
            continue
        anchors[theta] = max(float(q), anchors.get(theta, 0.0))
    items = sorted(anchors.items())
    theta = np.asarray([x for x, _ in items], dtype=float)
    mem = np.asarray([y for _, y in items], dtype=float)
    if theta.size == 1:
        return np.zeros_like(p_values, dtype=float)
    return np.interp(np.asarray(p_values, dtype=float), theta, mem, left=0.0, right=float(mem[-1]))


def read_empirical(metrics_root: Path) -> pd.DataFrame:
    rows = []
    for d in DIMS:
        for seed, folder in SELECTED_RUNS[d]:
            path = metrics_root / f"seed{seed}" / folder / "metrics.jsonl"
            if not path.exists():
                raise FileNotFoundError(path)
            count = 0
            with path.open() as f:
                for line in f:
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if "step" not in row:
                        continue
                    mem = finite_or_nan(row.get("memorization_fraction_pixel"))
                    rows.append({"seed": seed, "folder": folder, "d": d, "step": int(row["step"]), "mem": mem})
                    if np.isfinite(mem):
                        count += 1
            if count != 51:
                raise RuntimeError(f"Expected 51 memorization evaluations for seed={seed}, d={d}; got {count}")
    return pd.DataFrame(rows)


def empirical_onsets(emp: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    seed_rows = []
    summary_rows = []
    for d in DIMS:
        for q in THRESHOLDS:
            hits = []
            censored = 0
            for seed, folder in SELECTED_RUNS[d]:
                sub = emp[(emp["d"] == d) & (emp["seed"] == seed) & (emp["folder"] == folder)].dropna(subset=["mem"]).sort_values("step")
                hit = sub[sub["mem"] >= q]
                if hit.empty:
                    censored += 1
                    seed_rows.append({"d": d, "threshold": q, "seed": seed, "folder": folder, "tau_obs": math.nan, "censored": True})
                else:
                    tau = float(hit.iloc[0]["step"])
                    hits.append(tau)
                    seed_rows.append({"d": d, "threshold": q, "seed": seed, "folder": folder, "tau_obs": tau, "censored": False})
            mean, sd, ci, n = mean_ci(hits)
            summary_rows.append(
                {
                    "d": d,
                    "threshold": q,
                    "tau_obs_mean": mean,
                    "tau_obs_sd": sd,
                    "tau_obs_ci95": ci,
                    "obs_n": n,
                    "censored_n": censored,
                }
            )
    return pd.DataFrame(seed_rows), pd.DataFrame(summary_rows)


def load_spectra(spectral_dir: Path, primary_t: float) -> tuple[pd.DataFrame, dict[int, np.ndarray], dict[int, np.ndarray]]:
    features = pd.read_csv(spectral_dir / "spectral_features.csv")
    features = features[np.isclose(features["t"], primary_t)].copy()
    spectra_npz = np.load(spectral_dir / "spectra.npz")
    eigs: dict[int, np.ndarray] = {}
    weights: dict[int, np.ndarray] = {}
    key_t = t_label(primary_t)
    for d in DIMS:
        eigs[d] = np.asarray(spectra_npz[f"eig_d{d}_t{key_t}"], dtype=float)
        weights[d] = np.asarray(spectra_npz[f"weights_excess_d{d}_t{key_t}"], dtype=float)
        if eigs[d].shape[0] != d:
            raise RuntimeError(f"Spectrum length mismatch for d={d}")
        if not np.all(np.isfinite(eigs[d])) or np.min(eigs[d]) < -1e-8:
            raise RuntimeError(f"Bad spectrum for d={d}")
    return features, eigs, weights


def calibrate(onset_summary: pd.DataFrame, eigs: dict[int, np.ndarray], *, weights_by_d: dict[int, np.ndarray] | None) -> tuple[float, dict[float, float], pd.DataFrame, dict[str, float]]:
    fit_df = onset_summary.dropna(subset=["tau_obs_mean"]).copy()
    if fit_df.empty:
        raise RuntimeError("No uncensored empirical onset rows to fit")

    u_grid = np.concatenate(([0.0], np.logspace(-10.0, 3.0, 2600)))
    inverse_grids: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for d in DIMS:
        w = None if weights_by_d is None else weights_by_d[d]
        p_grid = pressure_u(eigs[d], u_grid, weights=w)
        p_grid = np.maximum.accumulate(p_grid)
        unique_p, idx = np.unique(p_grid, return_index=True)
        inverse_grids[d] = (unique_p, u_grid[idx])

    def invert_fast(d: int, theta: float, kappa: float) -> float:
        unique_p, unique_u = inverse_grids[d]
        if theta <= unique_p[0]:
            return 0.0
        if theta > unique_p[-1]:
            return math.nan
        u = float(np.interp(theta, unique_p, unique_u))
        return u / kappa

    best = None
    for log_kappa in np.linspace(-12.0, -4.0, 121):
        kappa = 10.0 ** log_kappa
        raw_thetas = []
        theta_weights = []
        for q in THRESHOLDS:
            q_vals = []
            q_weights = []
            for _, row in fit_df[fit_df["threshold"] == q].iterrows():
                d = int(row["d"])
                w = None if weights_by_d is None else weights_by_d[d]
                q_vals.append(float(pressure(eigs[d], [row["tau_obs_mean"]], kappa, weights=w)[0]))
                q_weights.append(float(row["obs_n"]))
            weight_sum = float(np.sum(q_weights))
            if q_vals and weight_sum > 0:
                raw_thetas.append(float(np.average(q_vals, weights=q_weights)))
                theta_weights.append(weight_sum)
            else:
                raw_thetas.append(math.nan)
                theta_weights.append(0.0)
        raw = np.asarray(raw_thetas, dtype=float)
        valid = np.isfinite(raw)
        if not valid.any():
            continue
        # Fully censored thresholds have no calibration target. Fill them with
        # neighboring calibrated values and give them near-zero PAVA weight.
        raw_filled = raw.copy()
        for i in range(len(raw_filled)):
            if np.isfinite(raw_filled[i]):
                continue
            left = next((raw_filled[j] for j in range(i - 1, -1, -1) if np.isfinite(raw_filled[j])), math.nan)
            right = next((raw_filled[j] for j in range(i + 1, len(raw_filled)) if np.isfinite(raw_filled[j])), math.nan)
            if np.isfinite(left):
                raw_filled[i] = left
            elif np.isfinite(right):
                raw_filled[i] = right
            else:
                raw_filled[i] = 0.5
        theta_weights_arr = np.asarray(theta_weights, dtype=float)
        theta_weights_arr[~valid] = 1e-9
        theta = pava_non_decreasing(raw_filled, theta_weights_arr)
        theta = np.maximum.accumulate(theta)
        theta = np.clip(theta, 1e-9, 1.0 - 1e-9)

        residuals = []
        for _, row in fit_df.iterrows():
            d = int(row["d"])
            q_idx = THRESHOLDS.index(float(row["threshold"]))
            pred = invert_fast(d, theta[q_idx], kappa)
            obs = float(row["tau_obs_mean"])
            if np.isfinite(pred) and pred > 0 and obs > 0:
                residuals.append((math.log(pred) - math.log(obs)) ** 2)
        if not residuals:
            continue
        objective = float(np.mean(residuals))
        if best is None or objective < best[0]:
            best = (objective, kappa, theta)

    if best is None:
        raise RuntimeError("Failed to fit kappa")

    objective, kappa, theta_arr = best
    theta_map = {q: float(theta_arr[i]) for i, q in enumerate(THRESHOLDS)}
    pred_rows = []
    for _, row in onset_summary.iterrows():
        d = int(row["d"])
        q = float(row["threshold"])
        pred = invert_fast(d, theta_map[q], kappa)
        obs = row["tau_obs_mean"]
        pred_rows.append(
            {
                **row.to_dict(),
                "tau_pred": pred,
                "residual_steps": pred - obs if np.isfinite(obs) and np.isfinite(pred) else math.nan,
                "residual_log": math.log(pred / obs) if np.isfinite(obs) and obs > 0 and np.isfinite(pred) and pred > 0 else math.nan,
                "theta": theta_map[q],
                "kappa": kappa,
            }
        )
    stats = {
        "objective_mse_log": float(objective),
        "kappa": float(kappa),
        **{f"theta_{int(q * 100)}pct": float(theta_map[q]) for q in THRESHOLDS},
    }
    return kappa, theta_map, pd.DataFrame(pred_rows), stats


def savefig(out_dir: Path, name: str) -> None:
    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])
    for ext in ("png", "pdf"):
        plt.savefig(out_dir / f"{name}.{ext}", dpi=220, bbox_inches="tight")
    plt.close()


def plot_mem_curves(
    emp: pd.DataFrame,
    tau_table: pd.DataFrame,
    eigs: dict[int, np.ndarray],
    kappa: float,
    theta_map: dict[float, float],
    calibrated_thresholds: set[float],
    out_dir: Path,
) -> None:
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(DIMS)))
    steps_dense = np.linspace(0, 5_000_000, 420)
    plt.figure(figsize=(9.8, 6.1))
    for d, color in zip(DIMS, colors):
        grouped = []
        for step, g in emp[emp["d"] == d].dropna(subset=["mem"]).groupby("step"):
            mean, _, ci, _ = mean_ci(list(g["mem"]))
            grouped.append((step, mean, ci))
        arr = np.asarray(grouped, dtype=float)
        x = arr[:, 0] / 1_000_000
        y = arr[:, 1]
        ci = arr[:, 2]
        plt.plot(x, y, color=color, linewidth=1.7, label=f"d={d}")
        plt.fill_between(x, np.maximum(y - ci, 0), np.minimum(y + ci, 1), color=color, alpha=0.08)
        p_curve = pressure(eigs[d], steps_dense, kappa)
        pred_curve = predicted_mem_from_pressure(p_curve, theta_map, calibrated_thresholds)
        plt.plot(steps_dense / 1_000_000, pred_curve, color=color, linestyle="--", linewidth=1.0, alpha=0.75)
        sub = tau_table[tau_table["d"] == d]
        for _, row in sub.iterrows():
            if float(row["threshold"]) not in calibrated_thresholds:
                continue
            if np.isfinite(row["tau_pred"]) and row["tau_pred"] <= 5_000_000:
                plt.scatter(row["tau_pred"] / 1_000_000, row["threshold"], marker="x", s=22, color=color, linewidths=1.2, zorder=4)
    for q in THRESHOLDS:
        plt.axhline(q, color="0.75", linewidth=0.6, alpha=0.35)
    plt.xlabel("training steps (millions)")
    plt.ylabel("memorization fraction")
    plt.xlim(0, 5.05)
    plt.ylim(-0.02, 0.92)
    plt.grid(True, alpha=0.18)
    plt.legend(ncol=5, fontsize=7.2, frameon=False)
    savefig(out_dir, "mem_curves_with_predicted_thresholds")


def plot_small_multiples(
    emp: pd.DataFrame,
    tau_table: pd.DataFrame,
    eigs: dict[int, np.ndarray],
    kappa: float,
    theta_map: dict[float, float],
    calibrated_thresholds: set[float],
    out_dir: Path,
) -> None:
    fig, axes = plt.subplots(5, 3, figsize=(10.4, 12.0), sharex=True, sharey=True)
    axes = axes.ravel()
    steps_dense = np.linspace(0, 5_000_000, 420)
    for ax, d in zip(axes, DIMS):
        grouped = []
        for step, g in emp[emp["d"] == d].dropna(subset=["mem"]).groupby("step"):
            mean, _, ci, _ = mean_ci(list(g["mem"]))
            grouped.append((step, mean, ci))
        arr = np.asarray(grouped, dtype=float)
        x = arr[:, 0] / 1_000_000
        y = arr[:, 1]
        ci = arr[:, 2]
        ax.plot(x, y, color="#2a6fbb", linewidth=1.8)
        ax.fill_between(x, np.maximum(y - ci, 0), np.minimum(y + ci, 1), color="#2a6fbb", alpha=0.14)
        p_curve = pressure(eigs[d], steps_dense, kappa)
        pred_curve = predicted_mem_from_pressure(p_curve, theta_map, calibrated_thresholds)
        ax.plot(steps_dense / 1_000_000, pred_curve, color="#c43c39", linestyle="--", linewidth=1.15, alpha=0.85)
        sub = tau_table[tau_table["d"] == d]
        for _, row in sub.iterrows():
            if float(row["threshold"]) not in calibrated_thresholds:
                continue
            if np.isfinite(row["tau_obs_mean"]):
                ax.scatter(row["tau_obs_mean"] / 1_000_000, row["threshold"], s=18, facecolor="white", edgecolor="black", linewidth=0.8)
            if np.isfinite(row["tau_pred"]) and row["tau_pred"] <= 5_000_000:
                ax.scatter(row["tau_pred"] / 1_000_000, row["threshold"], marker="x", s=22, color="#c43c39", linewidths=1.0)
        ax.set_title(f"d={d}", fontsize=10)
        ax.grid(True, alpha=0.18)
        ax.set_xlim(0, 5.05)
        ax.set_ylim(-0.02, 0.92)
    for ax in axes[len(DIMS):]:
        ax.axis("off")
    for ax in axes[-3:]:
        if ax.axison:
            ax.set_xlabel("steps (M)")
    axes[6].set_ylabel("mem fraction")
    fig.text(
        0.5,
        0.995,
        "dashed red: pressure-to-mem surrogate; black circles: observed threshold hits; red x: predicted threshold hits",
        ha="center",
        va="top",
        fontsize=10,
    )
    savefig(out_dir, "mem_curve_small_multiples")


def plot_pred_vs_obs(tau_table: pd.DataFrame, calibrated_thresholds: set[float], out_dir: Path) -> None:
    plt.figure(figsize=(6.2, 5.3))
    plot_thresholds = [q for q in THRESHOLDS if float(q) in calibrated_thresholds]
    colors = plt.cm.plasma(np.linspace(0.08, 0.92, max(len(plot_thresholds), 1)))
    maxv = 0.0
    for q, color in zip(plot_thresholds, colors):
        sub = tau_table[(tau_table["threshold"] == q)].dropna(subset=["tau_obs_mean", "tau_pred"])
        plt.scatter(sub["tau_obs_mean"] / 1_000_000, sub["tau_pred"] / 1_000_000, label=f"{int(q*100)}%", color=color, s=35)
        if not sub.empty:
            maxv = max(maxv, float(sub["tau_obs_mean"].max()), float(sub["tau_pred"].max()))
    lim = max(5_000_000.0, maxv) / 1_000_000
    plt.plot([0, lim], [0, lim], color="0.25", linewidth=1, linestyle="--")
    plt.xlabel("observed hitting time (M steps)")
    plt.ylabel("predicted hitting time (M steps)")
    plt.xlim(0, lim * 1.03)
    plt.ylim(0, lim * 1.03)
    plt.grid(True, alpha=0.22)
    plt.legend(title="mem level", frameon=False, fontsize=8)
    savefig(out_dir, "tau_pred_vs_obs_by_threshold")


def plot_tau_vs_d(tau_table: pd.DataFrame, calibrated_thresholds: set[float], out_dir: Path) -> None:
    plot_thresholds = [q for q in THRESHOLDS if float(q) in calibrated_thresholds]
    ncols = 2
    nrows = int(math.ceil(max(len(plot_thresholds), 1) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(9.2, 3.0 * nrows), sharex=True)
    axes = axes.ravel()
    for ax, q in zip(axes, plot_thresholds):
        sub = tau_table[tau_table["threshold"] == q].sort_values("d")
        ax.errorbar(sub["d"], sub["tau_obs_mean"] / 1_000_000, yerr=sub["tau_obs_ci95"] / 1_000_000, marker="o", linewidth=1.4, capsize=2.5, color="#2a6fbb", label="observed")
        ax.plot(sub["d"], sub["tau_pred"] / 1_000_000, marker="x", linewidth=1.4, color="#c43c39", label="predicted")
        cens = sub[sub["obs_n"] == 0]
        if not cens.empty:
            ax.scatter(cens["d"], np.zeros(len(cens)), marker="v", color="0.45", s=20, label="censored")
        ax.set_title(f"{int(q*100)}% memorization")
        ax.grid(True, alpha=0.2)
        ax.set_ylabel("tau (M steps)")
    for ax in axes[len(plot_thresholds):]:
        ax.axis("off")
    for ax in axes[-ncols:]:
        if ax.axison:
            ax.set_xlabel(r"$d_{\mathrm{latent}}$")
    axes[0].legend(frameon=False, fontsize=8)
    savefig(out_dir, "tau_vs_d_by_threshold")


def plot_pressure(eigs: dict[int, np.ndarray], weights: dict[int, np.ndarray] | None, kappa: float, out_dir: Path, name: str) -> None:
    steps = np.linspace(0, 5_000_000, 420)
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(DIMS)))
    plt.figure(figsize=(7.4, 4.8))
    for d, color in zip(DIMS, colors):
        w = None if weights is None else weights[d]
        p = pressure(eigs[d], steps, kappa, weights=w)
        plt.plot(steps / 1_000_000, p, color=color, linewidth=1.7, label=f"d={d}")
    plt.xlabel("training steps (millions)")
    plt.ylabel(r"spectral pressure $P_d(s)$")
    plt.ylim(-0.02, 1.02)
    plt.grid(True, alpha=0.22)
    plt.legend(ncol=5, fontsize=7.2, frameon=False)
    savefig(out_dir, name)


def plot_features(features: pd.DataFrame, out_dir: Path) -> None:
    f = features.sort_values("d")
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.6), sharex=True)
    axes = axes.ravel()
    axes[0].plot(f["d"], f["eta_star"], marker="o", color="#2a6fbb")
    axes[0].set_ylabel(r"$\eta_\star$")
    axes[1].plot(f["d"], f["beta_floor"], marker="o", color="#2a6fbb")
    axes[1].set_ylabel(r"$\beta_t^2$ floor")
    axes[2].plot(f["d"], f["effective_rank_excess"], marker="o", color="#2a6fbb", label="excess PR")
    axes[2].plot(f["d"], f["effective_rank_centered_excess"], marker="s", color="#569b64", label="centered")
    axes[2].set_ylabel("effective rank")
    axes[2].legend(frameon=False, fontsize=8)
    axes[3].plot(f["d"], f["buffer_proxy"], marker="o", color="#2a6fbb")
    axes[3].set_ylabel("buffer proxy")
    for ax in axes:
        ax.grid(True, alpha=0.22)
        ax.set_xlabel(r"$d_{\mathrm{latent}}$")
    savefig(out_dir, "spectral_features_vs_d")


def write_readme(out_dir: Path, primary_t: float, stats: dict[str, float], weighted_stats: dict[str, float]) -> None:
    selection = {str(d): [f"seed{seed}/{folder}" for seed, folder in SELECTED_RUNS[d]] for d in DIMS}
    readme = f"""# CIFAR-10 Spectral Memorization Predictor

Data-only predictor for CIFAR-10 memorization onset. The prediction uses
existing CIFAR-10 VAE encoder means on the same diverse 1k training subset and
does not inspect trained diffusion-model weights.

Primary statistic:

`M_t(d) = exp(-2t) Z^T Z / n + (1 - exp(-2t)) I`, with `t = {primary_t}`.

Primary spectral pressure:

`P_d(s) = mean_i (1 - exp(-kappa * lambda_i(d) * s))^2`.

Calibration:

- one global `kappa = {stats['kappa']:.6g}`
- one monotone threshold map over memorization levels
- no per-dimension fitting

The empirical curves use only unclipped selected runs. Clipped diagnostic runs
are excluded because clipping strongly suppressed measured memorization.

Generated files:

- `mem_curves_with_predicted_thresholds.*`
- `mem_curve_small_multiples.*`
- `tau_pred_vs_obs_by_threshold.*`
- `tau_vs_d_by_threshold.*`
- `spectral_pressure_curves.*`
- `spectral_pressure_curves_weighted.*`
- `spectral_features_vs_d.*`
- `cifar10_spectral_tau_table.csv`
- `cifar10_spectral_tau_table_weighted.csv`
- `calibration.json`
- `selected_runs.json`

Plot interpretation note: the red x markers are iso-memorization threshold-hit
predictions. Their y-values are fixed thresholds by construction. The dashed
red curve maps spectral pressure through the fitted threshold map and is the
more natural visual surrogate for predicted memorization over time. Thresholds
that are globally unobserved in the empirical runs are not shown as calibrated
theoretical overlays.
"""
    (out_dir / "README.md").write_text(readme)
    (out_dir / "calibration.json").write_text(json.dumps({"unweighted": stats, "weighted": weighted_stats}, indent=2) + "\n")
    (out_dir / "selected_runs.json").write_text(json.dumps(selection, indent=2) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--metrics_root", default="results/cifar10_beta005_diverse1k_bigmlp_sgd_lr001_m08_10k_5m")
    p.add_argument("--spectral_dir", default="diagnostics/cifar10_spectral_mem_predictor")
    p.add_argument("--out", default="diagnostics/cifar10_spectral_mem_predictor_figures")
    p.add_argument("--primary_t", type=float, default=0.1)
    args = p.parse_args()

    metrics_root = Path(args.metrics_root)
    spectral_dir = Path(args.spectral_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    emp = read_empirical(metrics_root)
    emp.to_csv(out_dir / "cifar10_empirical_mem_curves.csv", index=False)
    _, onset_summary = empirical_onsets(emp)
    onset_summary.to_csv(out_dir / "cifar10_empirical_tau_summary.csv", index=False)
    calibrated_thresholds = {
        float(q)
        for q, group in onset_summary.groupby("threshold")
        if float(group["obs_n"].sum()) > 0
    }

    features, eigs, weights = load_spectra(spectral_dir, args.primary_t)
    features.to_csv(out_dir / "cifar10_spectral_features_primary_t.csv", index=False)

    kappa, theta_map, tau_table, stats = calibrate(onset_summary, eigs, weights_by_d=None)
    tau_table.to_csv(out_dir / "cifar10_spectral_tau_table.csv", index=False)

    weighted_kappa, _, weighted_tau_table, weighted_stats = calibrate(onset_summary, eigs, weights_by_d=weights)
    weighted_tau_table.to_csv(out_dir / "cifar10_spectral_tau_table_weighted.csv", index=False)

    plot_mem_curves(emp, tau_table, eigs, kappa, theta_map, calibrated_thresholds, out_dir)
    plot_small_multiples(emp, tau_table, eigs, kappa, theta_map, calibrated_thresholds, out_dir)
    plot_pred_vs_obs(tau_table, calibrated_thresholds, out_dir)
    plot_tau_vs_d(tau_table, calibrated_thresholds, out_dir)
    plot_pressure(eigs, None, kappa, out_dir, "spectral_pressure_curves")
    plot_pressure(eigs, weights, weighted_kappa, out_dir, "spectral_pressure_curves_weighted")
    plot_features(features, out_dir)
    write_readme(out_dir, args.primary_t, stats, weighted_stats)
    print(f"Wrote CIFAR-10 spectral predictor figures to {out_dir}")


if __name__ == "__main__":
    main()
