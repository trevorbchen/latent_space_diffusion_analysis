"""Post-processing and plotting for v2 + v3 metrics.jsonl files.

The v2 and v3 schemas are identical (deliberate), so this module reads
both. v2 runs encode their config in the directory name
(`di{d_intrinsic}_d{d_latent}_n{n}_s{seed}`); v3 runs additionally write a
`config.json`. Either is fine.

Subcommands cover the placeholder figures in the ICML draft:
    fig1_gap         tau_mem - tau_gen (or paired bars) vs d_latent
    fig2_losses      train/test loss bars vs d_latent
    fig3_curves      score_error over step, one curve per d_latent
    fig4_minscore    min score_error vs d_latent/d_intrinsic
    fig15_low        low-d regime training panel (d_latent == d_intrinsic)
    fig16_high       high-d regime training panel (d_latent ~= 3*d_intrinsic)
    all              regenerate every figure into one out_dir

Usage:
    python plot.py all \
        --runs ../sigma_noise_0.5/exp2_mlp/raw_data \
        --out_dir ../sigma_noise_0.5/exp2_mlp/figures
"""
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Run loading
# ---------------------------------------------------------------------------

DIR_PATTERN = re.compile(
    r'^di(?P<d_intrinsic>\d+)_d(?P<d_latent>\d+)_n(?P<n>\d+)_s(?P<seed>\d+)$'
)


@dataclass
class Run:
    path: Path
    config: dict[str, Any] = field(default_factory=dict)
    rows: list[dict[str, Any]] = field(default_factory=list)

    # ---- accessors ----
    def __len__(self) -> int:
        return len(self.rows)

    def col(self, key: str) -> np.ndarray:
        """Return column as np.ndarray, NaN where missing."""
        out = np.full(len(self.rows), np.nan)
        for i, r in enumerate(self.rows):
            v = r.get(key)
            if v is not None and not isinstance(v, (dict, list)):
                out[i] = float(v)
        return out

    @property
    def step(self) -> np.ndarray:
        return self.col('step').astype(int)

    @property
    def d_latent(self) -> int:
        return int(self.config.get('d_latent'))

    @property
    def d_intrinsic(self) -> int:
        return int(self.config.get('d_intrinsic'))


def _read_metrics(path: Path) -> tuple[dict, list[dict]]:
    config: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    config_json = path / 'config.json'
    if config_json.exists():
        config = json.loads(config_json.read_text())
    metrics_jsonl = path / 'metrics.jsonl'
    if not metrics_jsonl.exists():
        return config, rows
    for line in metrics_jsonl.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if obj.get('event') == 'meta':
            # v3 meta line — fold into config
            for k, v in obj.items():
                if k != 'event' and k not in config:
                    config[k] = v
            continue
        rows.append(obj)
    return config, rows


def load_run(path: Path) -> Run:
    """Load a single run directory (v2 layout or v3 layout)."""
    config, rows = _read_metrics(path)
    # Backfill v2-style fields from the directory name if needed.
    m = DIR_PATTERN.match(path.name)
    if m:
        for k, v in m.groupdict().items():
            config.setdefault(k, int(v))
    return Run(path=path, config=config, rows=rows)


def load_sweep(root: Path) -> list[Run]:
    """Load every run under root that has a metrics.jsonl."""
    runs: list[Run] = []
    for child in sorted(root.iterdir() if root.is_dir() else []):
        if (child / 'metrics.jsonl').exists():
            runs.append(load_run(child))
    return runs


# ---------------------------------------------------------------------------
# Tau extraction
# ---------------------------------------------------------------------------

@dataclass
class Taus:
    tau_gen: int | None         # None when not crossed within the run
    tau_mem: int | None
    tau_gen_method: str
    tau_mem_method: str
    censored_steps: int          # max step in the run; tau_x > censored when None


def extract_taus(run: Run,
                 *,
                 mem_threshold: float = 0.01,
                 gen_gap_threshold: float = 0.02,
                 test_loss_tolerance: float = 0.05) -> Taus:
    """Paper's definitions (Appendix A.4) with two robustness fixes:

    - tau_gen: first step from which the test_loss STAYS within
      (1 + tolerance) * per-run minimum for the rest of the run. Avoids
      catching transient dips in the noisy early curve.
    - tau_mem: prefer memorization_fraction whenever the column is
      populated (any finite value) — even if it never crosses the
      threshold (that's genuine censoring). The gen_gap proxy is only
      consulted when the run has no memorization_fraction column at all
      (the RFNN case).
    """
    steps = run.step
    if steps.size == 0:
        return Taus(None, None, 'na', 'na', 0)

    # ---- tau_gen ----
    # Paper's definition (Appendix A.4): first step at which test_loss is
    # within (1+tolerance)*per-run-min. These runs have non-monotonic
    # test_loss (descend, then overfit), so the global min is hit early
    # in the descent and tau_gen lands on the descent boundary.
    test = run.col('test_loss')
    finite = np.isfinite(test)
    if finite.any():
        tmin = np.nanmin(test[finite])
        thresh = tmin * (1 + test_loss_tolerance)
        crossed = np.where(finite & (test <= thresh))[0]
        tau_gen = int(steps[crossed[0]]) if crossed.size else None
    else:
        tau_gen = None
    method_gen = f'first step where test_loss <= (1+{test_loss_tolerance})*min'

    # ---- tau_mem ----
    mem = run.col('memorization_fraction')
    has_mem = np.isfinite(mem).any()
    censored_step = int(steps.max())

    if has_mem:
        cross = np.where(np.isfinite(mem) & (mem > mem_threshold))[0]
        if cross.size:
            tau_mem = int(steps[cross[0]])
            method_mem = f'mem_fraction>{mem_threshold}'
        else:
            tau_mem = None
            method_mem = f'mem_fraction never crossed; censored>{censored_step}'
    else:
        gap = run.col('gen_gap')
        cross = np.where(np.isfinite(gap) & (gap > gen_gap_threshold))[0]
        if cross.size:
            tau_mem = int(steps[cross[0]])
            method_mem = f'gen_gap>{gen_gap_threshold}'
        else:
            tau_mem = None
            method_mem = f'censored>{censored_step}'

    return Taus(
        tau_gen=tau_gen,
        tau_mem=tau_mem,
        tau_gen_method=method_gen,
        tau_mem_method=method_mem,
        censored_steps=censored_step,
    )


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------

def _by_d_latent(runs: Iterable[Run]) -> list[Run]:
    return sorted(runs, key=lambda r: r.d_latent)


def _censor_value(tau: int | None, censor: int) -> int:
    """For plotting: replace None with the censoring upper bound."""
    return tau if tau is not None else censor


def _save(fig, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  wrote {out}")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig1_gap(runs: Sequence[Run], out: Path,
             *, paired_bars: bool = True) -> None:
    """Fig 1: tau_mem - tau_gen vs d_latent (or paired bars)."""
    runs = _by_d_latent(runs)
    d = [r.d_latent for r in runs]
    taus = [extract_taus(r) for r in runs]
    censor = max(t.censored_steps for t in taus)
    gen = np.array([_censor_value(t.tau_gen, censor) for t in taus])
    mem = np.array([_censor_value(t.tau_mem, censor) for t in taus])

    fig, ax = plt.subplots(figsize=(7, 4))
    if paired_bars:
        x = np.arange(len(d))
        w = 0.4
        ax.bar(x - w / 2, gen, w, label=r'$\tau_{\rm gen}$', color='C0')
        ax.bar(x + w / 2, mem, w, label=r'$\tau_{\rm mem}$', color='C3')
        ax.set_xticks(x); ax.set_xticklabels(d)
        # Mark censored memorization (no crossing) with hatching
        for i, t in enumerate(taus):
            if t.tau_mem is None:
                ax.bar(x[i] + w / 2, mem[i], w, color='none',
                       edgecolor='C3', hatch='//')
    else:
        gap = mem - gen
        ax.plot(d, gap, 'o-')
        ax.set_ylabel(r'$\tau_{\rm mem} - \tau_{\rm gen}$ (steps)')
    ax.set_xlabel(r'$d_{\rm latent}$')
    if paired_bars:
        ax.set_ylabel('training step')
    ax.set_yscale('log')
    ax.legend()
    ax.set_title('Timescale gap (synthetic, scaled MLP)')
    _save(fig, out)


def fig2_losses(runs: Sequence[Run], out: Path) -> None:
    """Fig 2: end-of-training train + test loss vs d_latent (paired bars)."""
    runs = _by_d_latent(runs)
    d = [r.d_latent for r in runs]
    train = []
    test = []
    for r in runs:
        # use min over the last 10% of steps as a stable end-of-run estimate
        steps = r.step
        if steps.size == 0:
            train.append(np.nan); test.append(np.nan); continue
        cutoff = int(0.9 * steps.max())
        mask = steps >= cutoff
        train.append(float(np.nanmean(r.col('train_loss')[mask])))
        test.append(float(np.nanmean(r.col('test_loss')[mask])))
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(d)); w = 0.4
    ax.bar(x - w / 2, train, w, label='train', color='C0')
    ax.bar(x + w / 2, test, w, label='test', color='C3')
    ax.set_xticks(x); ax.set_xticklabels(d)
    ax.set_xlabel(r'$d_{\rm latent}$')
    ax.set_ylabel('loss / d_latent')
    ax.set_title('End-of-training loss (synthetic, scaled MLP)')
    ax.legend()
    _save(fig, out)


def fig3_curves(runs: Sequence[Run], out: Path) -> None:
    """Fig 3: score_error vs training step, one curve per d_latent."""
    runs = _by_d_latent(runs)
    fig, ax = plt.subplots(figsize=(7, 4))
    cmap = plt.cm.viridis(np.linspace(0, 1, len(runs)))
    for r, c in zip(runs, cmap):
        steps = r.step
        se = r.col('score_error')
        mask = np.isfinite(se)
        ax.plot(steps[mask], se[mask], color=c,
                label=f'$d={r.d_latent}$')
    ax.set_xlabel('training step')
    ax.set_ylabel('score error')
    ax.set_yscale('log')
    ax.set_title('Score error during training (synthetic, scaled MLP)')
    ax.legend(ncols=2, fontsize=8)
    _save(fig, out)


def fig4_minscore(runs: Sequence[Run], out: Path) -> None:
    """Fig 4: min score_error vs d_latent / d_intrinsic."""
    runs = _by_d_latent(runs)
    ratios = []
    minse = []
    for r in runs:
        se = r.col('score_error')
        if not np.isfinite(se).any():
            continue
        ratios.append(r.d_latent / max(r.d_intrinsic, 1))
        minse.append(float(np.nanmin(se)))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ratios, minse, 'o-')
    ax.set_xlabel(r'$d_{\rm latent} / d_{\rm intrinsic}$')
    ax.set_ylabel('min score error')
    ax.set_yscale('log')
    ax.set_title('Minimum score error vs latent ratio (synthetic, scaled MLP)')
    _save(fig, out)


def _regime_panel(run: Run, out: Path, *, label: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    steps = run.step

    axes[0].plot(steps, run.col('score_error'))
    axes[0].set_yscale('log')
    axes[0].set_xlabel('step'); axes[0].set_ylabel('score error')

    axes[1].plot(steps, run.col('gen_gap'))
    axes[1].axhline(0, color='gray', lw=0.5)
    axes[1].set_xlabel('step'); axes[1].set_ylabel('gen gap (test - train)')

    mf = run.col('memorization_fraction')
    if np.isfinite(mf).any():
        axes[2].plot(steps, mf)
        axes[2].axhline(0.01, color='gray', ls='--', lw=0.5)
        axes[2].set_xlabel('step'); axes[2].set_ylabel('mem fraction')
    else:
        axes[2].text(0.5, 0.5, 'no mem data', ha='center', va='center',
                     transform=axes[2].transAxes)
        axes[2].set_xticks([]); axes[2].set_yticks([])

    fig.suptitle(f'{label} (d_latent={run.d_latent}, d_intrinsic={run.d_intrinsic})')
    fig.tight_layout()
    _save(fig, out)


def fig15_low(runs: Sequence[Run], out: Path) -> None:
    """Fig 15: low-d regime panel, d_latent == d_intrinsic."""
    target = next(
        (r for r in runs if r.d_latent == r.d_intrinsic), None
    )
    if target is None:
        print('  fig15: no run with d_latent == d_intrinsic; skipping')
        return
    _regime_panel(target, out, label='Low-score regime')


def fig16_high(runs: Sequence[Run], out: Path,
               ratio_target: float = 3.0) -> None:
    """Fig 16: high-d regime, d_latent ~ ratio_target * d_intrinsic.

    Picks the run whose ratio is closest to ratio_target.
    """
    if not runs:
        return
    best = min(runs, key=lambda r:
               abs(r.d_latent / max(r.d_intrinsic, 1) - ratio_target))
    _regime_panel(best, out, label='High-score regime')


# ---------------------------------------------------------------------------
# RFNN figures
# ---------------------------------------------------------------------------

def _load_eigvals(run: Run, name: str = 'eigenvalues_pre.npy') -> np.ndarray | None:
    p = run.path / name
    if not p.exists():
        return None
    return np.load(p)


def fig9_cliffs(runs: Sequence[Run], out: Path,
                *, n_cols: int = 4, zoom_factor: float = 4.0) -> None:
    """Fig 9: sorted log-spectrum of U at fixed d_intrinsic, varying d_latent.

    Vertical dashed lines mark the bulk boundaries at indices d_intrinsic
    (red) and d_latent (green). The cliffs at those indices are the paper's
    headline empirical observation. We crop the x-axis to ~zoom_factor *
    d_latent so the cliffs are visible (the rank-null tail otherwise
    dominates at high d_latent).
    """
    runs = [r for r in _by_d_latent(runs) if _load_eigvals(r) is not None]
    if not runs:
        print('  fig9: no eigenvalues_pre.npy files found; skipping')
        return
    n_rows = (len(runs) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 2.8 * n_rows),
                              sharey=True, squeeze=False)
    for i, r in enumerate(runs):
        eigs = _load_eigvals(r)
        if eigs is None:
            continue
        x_max = int(min(eigs.size, max(zoom_factor * r.d_latent, 80)))
        ax = axes[i // n_cols, i % n_cols]
        ax.semilogy(np.arange(x_max), eigs[:x_max])
        ax.axvline(r.d_intrinsic, color='red',   ls='--', lw=0.8,
                   label=f'$d_{{\\rm intrinsic}}$={r.d_intrinsic}')
        ax.axvline(r.d_latent,    color='green', ls='--', lw=0.8,
                   label=f'$d_{{\\rm latent}}$={r.d_latent}')
        ax.set_title(f'$d_{{\\rm latent}}$={r.d_latent}', fontsize=9)
        if i % n_cols == 0:
            ax.set_ylabel('eigenvalue (log)')
    for j in range(len(runs), n_rows * n_cols):
        axes[j // n_cols, j % n_cols].axis('off')
    fig.suptitle(f'Four-bulk cliffs ($d_{{\\rm intrinsic}}$ = {runs[0].d_intrinsic}, '
                 f'n = {runs[0].config.get("n", "?")})')
    fig.supxlabel('eigenvalue index')
    fig.tight_layout()
    _save(fig, out)


def fig10_density(runs: Sequence[Run], out: Path,
                  picks: Sequence[int] | None = None) -> None:
    """Fig 10: histogram of log10 eigenvalues for selected d_latent."""
    runs = [r for r in _by_d_latent(runs) if _load_eigvals(r) is not None]
    if not runs:
        return
    if picks is not None:
        runs = [r for r in runs if r.d_latent in picks]
    if not runs:
        return
    fig, axes = plt.subplots(1, len(runs), figsize=(3.5 * len(runs), 3),
                              sharey=True, squeeze=False)
    for i, r in enumerate(runs):
        eigs = _load_eigvals(r)
        log_eigs = np.log10(np.maximum(eigs, 1e-12))
        axes[0, i].hist(log_eigs, bins=40, log=True, color='C0')
        axes[0, i].set_title(f'$d_{{\\rm latent}}$={r.d_latent}', fontsize=9)
        axes[0, i].set_xlabel(r'$\log_{10} \lambda$')
    axes[0, 0].set_ylabel('count')
    fig.suptitle('Eigenvalue density (four-bulk peaks)')
    fig.tight_layout()
    _save(fig, out)


def fig11_vary_dintrinsic(runs: Sequence[Run], out: Path,
                          *, n_cols: int = 6) -> None:
    """Fig 11: cliffs view at fixed d_latent, varying d_intrinsic.

    As d_intrinsic -> d_latent the noise-dim bulk shrinks to zero width.
    """
    runs = [r for r in runs if _load_eigvals(r) is not None]
    runs = sorted(runs, key=lambda r: r.d_intrinsic)
    if not runs:
        print('  fig11: no eigenvalues_pre.npy files found; skipping')
        return
    n_rows = (len(runs) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 2.8 * n_rows),
                              sharey=True, squeeze=False)
    for i, r in enumerate(runs):
        eigs = _load_eigvals(r)
        ax = axes[i // n_cols, i % n_cols]
        ax.loglog(np.arange(1, eigs.size + 1), eigs)
        ax.axvline(r.d_intrinsic, color='red',   ls='--', lw=0.8)
        ax.axvline(r.d_latent,    color='green', ls='--', lw=0.8)
        ax.set_title(f'$d_{{\\rm intrinsic}}$={r.d_intrinsic}', fontsize=9)
        if i % n_cols == 0:
            ax.set_ylabel('eigenvalue')
    for j in range(len(runs), n_rows * n_cols):
        axes[j // n_cols, j % n_cols].axis('off')
    fig.suptitle(f'Four-bulk varying $d_{{\\rm intrinsic}}$ '
                 f'($d_{{\\rm latent}}$={runs[0].d_latent})')
    fig.supxlabel('eigenvalue index')
    fig.tight_layout()
    _save(fig, out)


def _bulk_absorption_series(run: Run) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Extract per-bulk absorption fractions from a v3 RFNN metrics.jsonl.

    Returns (steps, {bulk_name: fraction_array}). Steps are the eval steps
    where 'absorption' was logged; missing bulks default to NaN.
    """
    steps = []
    series: dict[str, list[float]] = {
        'signal': [], 'noise_dim': [], 'sample': [], 'rank_null': [],
    }
    for r in run.rows:
        absorption = r.get('absorption')
        if not isinstance(absorption, dict):
            continue
        steps.append(r['step'])
        for key in series:
            entry = absorption.get(key, {})
            if isinstance(entry, dict):
                series[key].append(entry.get('fraction', np.nan))
            else:
                # legacy bare-float format
                series[key].append(float(entry))
    return np.array(steps, dtype=int), {k: np.array(v) for k, v in series.items()}


def fig7_fid_curves(runs: Sequence[Run], out: Path) -> None:
    """Fig 7: FID vs training step, one curve per d_latent (real data)."""
    runs = _by_d_latent(runs)
    fig, ax = plt.subplots(figsize=(7, 4))
    cmap = plt.cm.viridis(np.linspace(0, 1, max(len(runs), 1)))
    plotted = False
    for r, c in zip(runs, cmap):
        steps = r.step
        fid = r.col('fid')
        mask = np.isfinite(fid)
        if not mask.any():
            continue
        ax.plot(steps[mask], fid[mask], color=c,
                label=f'$d={r.d_latent}$')
        plotted = True
    if not plotted:
        print('  fig7: no FID values logged in any run; skipping')
        plt.close(fig)
        return
    ax.set_xlabel('training step')
    ax.set_ylabel('FID')
    ax.set_yscale('log')
    ax.set_title('FID during training (real data)')
    ax.legend(ncols=2, fontsize=8)
    _save(fig, out)


def fig8_min_fid(runs: Sequence[Run], out: Path) -> None:
    """Fig 8: minimum FID (over training) vs d_latent."""
    runs = _by_d_latent(runs)
    d_vals = []
    min_fid = []
    for r in runs:
        fid = r.col('fid')
        if not np.isfinite(fid).any():
            continue
        d_vals.append(r.d_latent)
        min_fid.append(float(np.nanmin(fid)))
    if not d_vals:
        print('  fig8: no FID values logged in any run; skipping')
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(d_vals, min_fid, 'o-')
    ax.set_xlabel(r'$d_{\rm latent}$')
    ax.set_ylabel('min FID')
    ax.set_yscale('log')
    ax.set_title('Best FID across training (real data)')
    _save(fig, out)


def fig14_data_pca(runs: Sequence[Run], out: Path,
                   *, n_cols: int = 4, zoom_factor: float = 4.0) -> None:
    """Fig 14: eigenvalues of X^T X / n on the synthetic training data.

    Regenerates each run's training data deterministically from its config
    (none of v2 stores X). Shows the two-block structure of Sigma_data
    that the paper says the random-feature lift carries into the signal
    and noise-dim bulks of U.
    """
    from lib.data_synthetic import generate_data
    runs = _by_d_latent(runs)
    valid: list[Run] = []
    pcas: list[np.ndarray] = []
    for r in runs:
        cfg = r.config
        keys_required = ('n', 'd_intrinsic', 'd_latent', 'sigma_noise',
                          'sigma_signal', 'scale', 'k', 'seed')
        if not all(k in cfg for k in keys_required):
            continue
        X, *_ = generate_data(
            n=int(cfg['n']),
            d_intrinsic=int(cfg['d_intrinsic']),
            d_latent=int(cfg['d_latent']),
            k=int(cfg['k']),
            sigma_noise=float(cfg['sigma_noise']),
            sigma_signal=float(cfg['sigma_signal']),
            scale=float(cfg['scale']),
            seed=int(cfg['seed']),
        )
        Xn = X.numpy()
        # Eigenvalues of X^T X / n -- equivalent to squared singular values / n
        s = np.linalg.svd(Xn, compute_uv=False)
        pcas.append(np.sort((s ** 2) / Xn.shape[0])[::-1])
        valid.append(r)

    if not valid:
        print('  fig14: no runs with full config; skipping')
        return

    n_rows = (len(valid) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(3.2 * n_cols, 2.8 * n_rows),
                              sharey=True, squeeze=False)
    for i, (r, eigs) in enumerate(zip(valid, pcas)):
        ax = axes[i // n_cols, i % n_cols]
        x_max = int(min(eigs.size, max(zoom_factor * r.d_latent, 40)))
        ax.semilogy(np.arange(x_max), eigs[:x_max])
        ax.axvline(r.d_intrinsic, color='red',   ls='--', lw=0.8)
        ax.axvline(r.d_latent,    color='green', ls='--', lw=0.8)
        ax.set_title(f'$d_{{\\rm latent}}$={r.d_latent}', fontsize=9)
        if i % n_cols == 0:
            ax.set_ylabel('eig of $X^T X / n$ (log)')
    for j in range(len(valid), n_rows * n_cols):
        axes[j // n_cols, j % n_cols].axis('off')
    fig.suptitle(f'Data PCA ($d_{{\\rm intrinsic}}$={valid[0].d_intrinsic}, '
                 f'$\\sigma_\\perp$={valid[0].config.get("sigma_noise")})')
    fig.supxlabel('PCA index')
    fig.tight_layout()
    _save(fig, out)


def fig12_13_bulk_evolution(runs: Sequence[Run], out: Path,
                            ratio_target: float = 4.0) -> None:
    """Figs 12-13: per-bulk absorption fraction over training.

    Picks one RFNN run (closest to ratio_target = d_latent/d_intrinsic) and
    plots the four bulks' fraction-of-readout-mass over training steps.
    """
    cands = []
    for r in runs:
        steps, series = _bulk_absorption_series(r)
        if steps.size:
            cands.append((r, steps, series))
    if not cands:
        print('  fig12_13: no absorption data found in any run; skipping')
        return
    target = min(cands, key=lambda c:
                 abs(c[0].d_latent / max(c[0].d_intrinsic, 1) - ratio_target))
    r, steps, series = target

    fig, ax = plt.subplots(figsize=(7, 4))
    palette = {'signal': 'C0', 'noise_dim': 'C1',
               'sample': 'C3', 'rank_null': 'gray'}
    for name, vals in series.items():
        if not np.isfinite(vals).any():
            continue
        ax.plot(steps, vals, label=name, color=palette[name])
    ax.set_xlabel('training step')
    ax.set_ylabel('fraction of readout mass in bulk')
    ax.set_ylim(0, 1.05)
    ax.set_title(f'Per-bulk absorption (RFNN, $d_{{\\rm latent}}$={r.d_latent}, '
                 f'$d_{{\\rm intrinsic}}$={r.d_intrinsic})')
    ax.legend()
    _save(fig, out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _make_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    single = ('fig1_gap', 'fig2_losses', 'fig3_curves', 'fig4_minscore',
              'fig15_low', 'fig16_high', 'fig9_cliffs', 'fig10_density',
              'fig11_vary_dintrinsic', 'fig12_13_bulk_evolution',
              'fig14_data_pca', 'fig7_fid_curves', 'fig8_min_fid')
    for name in single:
        sp = sub.add_parser(name)
        sp.add_argument('--runs', type=Path, required=True)
        sp.add_argument('--out',  type=Path, required=True)
    sp_all = sub.add_parser('all')
    sp_all.add_argument('--runs',    type=Path, required=True)
    sp_all.add_argument('--out_dir', type=Path, required=True)
    sp_rfnn = sub.add_parser('all_rfnn',
        help='Generate all RFNN-only figures (9, 10, 11, 12-13, 14).')
    sp_rfnn.add_argument('--runs',    type=Path, required=True,
        help='RFNN run dir (e.g. exp2_rfnn/raw_data, exp3_rfnn/raw_data).')
    sp_rfnn.add_argument('--out_dir', type=Path, required=True)
    sp_real = sub.add_parser('all_real',
        help='Generate all real-data figures: timescale gap, train/test loss, '
             'FID over training, min FID vs d_latent (Figs 5, 6, 7, 8).')
    sp_real.add_argument('--runs',    type=Path, required=True,
        help='Real-data run dir (e.g. results/mnist_mlp_*).')
    sp_real.add_argument('--out_dir', type=Path, required=True)
    sp_taus = sub.add_parser('taus',
        help='Print extracted tau_gen, tau_mem per run as JSON.')
    sp_taus.add_argument('--runs', type=Path, required=True)
    return p


def main():
    args = _make_arg_parser().parse_args()
    runs = load_sweep(args.runs)
    print(f"Loaded {len(runs)} run(s) from {args.runs}")

    if args.cmd == 'taus':
        for r in _by_d_latent(runs):
            taus = extract_taus(r)
            print(json.dumps({
                'path': str(r.path), 'd_latent': r.d_latent,
                'tau_gen': taus.tau_gen, 'tau_mem': taus.tau_mem,
                'tau_mem_method': taus.tau_mem_method,
            }))
        return

    if args.cmd == 'all':
        out = args.out_dir
        fig1_gap(runs,        out / 'fig1_gap.png')
        fig2_losses(runs,     out / 'fig2_losses.png')
        fig3_curves(runs,     out / 'fig3_score_curves.png')
        fig4_minscore(runs,   out / 'fig4_minscore.png')
        fig15_low(runs,       out / 'fig15_low_regime.png')
        fig16_high(runs,      out / 'fig16_high_regime.png')
        return

    if args.cmd == 'all_rfnn':
        out = args.out_dir
        fig9_cliffs(runs,                out / 'fig9_cliffs.png')
        fig10_density(runs,              out / 'fig10_density.png')
        fig11_vary_dintrinsic(runs,      out / 'fig11_vary_dintrinsic.png')
        fig14_data_pca(runs,             out / 'fig14_data_pca.png')
        fig12_13_bulk_evolution(runs,    out / 'fig12_13_bulk_evolution.png')
        return

    if args.cmd == 'all_real':
        out = args.out_dir
        # Figs 5/6 reuse the synthetic helpers (the metric columns are the same).
        fig1_gap(runs,        out / 'fig5_gap.png')
        fig2_losses(runs,     out / 'fig6_losses.png')
        fig7_fid_curves(runs, out / 'fig7_fid_curves.png')
        fig8_min_fid(runs,    out / 'fig8_min_fid.png')
        return

    fn = {
        'fig1_gap':                fig1_gap,
        'fig2_losses':             fig2_losses,
        'fig3_curves':             fig3_curves,
        'fig4_minscore':           fig4_minscore,
        'fig15_low':               fig15_low,
        'fig16_high':              fig16_high,
        'fig9_cliffs':             fig9_cliffs,
        'fig10_density':           fig10_density,
        'fig11_vary_dintrinsic':   fig11_vary_dintrinsic,
        'fig12_13_bulk_evolution': fig12_13_bulk_evolution,
        'fig14_data_pca':          fig14_data_pca,
        'fig7_fid_curves':         fig7_fid_curves,
        'fig8_min_fid':            fig8_min_fid,
    }[args.cmd]
    fn(runs, args.out)


if __name__ == '__main__':
    main()
