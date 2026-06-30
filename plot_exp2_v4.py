"""Generate exp2_v4 figures mirroring exp2_v2:
  - exp2_v4_results.png      (6-panel grid: train/test, score, mem, nn, gap, timescales)
  - exp2_v4_by_xaxis.png     (test loss vs step / epochs / flops / wall_time + train, gap)
  - exp2_v4_timescales.png   (2x3: tau_mem and tau_gen in steps + FLOPs, +ref/divergence)

Source: exp2_v4/di5_d{N}_n500_s42/metrics.jsonl pulled from ml6/ml7
        (5M-step scaled MLP, hidden=8*d_latent, sigma_noise=0.5).

Some configs are not yet at 5M (ml6 still running) — plots reflect current state.
"""
import json, glob, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT_DIR = 'exp2_v4'
D_INTRINSIC = 5

# --- Load all runs ---------------------------------------------------------
runs = {}
for fp in sorted(glob.glob(f'{OUT_DIR}/di5_d*_n500_s42/metrics.jsonl')):
    with open(fp.replace('metrics.jsonl', 'config.json')) as f:
        config = json.load(f)
    metrics = [json.loads(l) for l in open(fp)]
    runs[config['d_latent']] = sorted(metrics, key=lambda x: x['step'])

ds = sorted(runs.keys())
print(f'Loaded d_latent={ds}')
for d in ds:
    print(f'  d={d:3d}: {len(runs[d])} evals, last step {runs[d][-1]["step"]}, '
          f'mem.frac={runs[d][-1]["memorization_fraction"]:.4f}')

colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(ds)))
cmap = {d: colors[i] for i, d in enumerate(ds)}


def first_above(metrics, key, threshold):
    """First step at which metrics[step][key] > threshold; None if never."""
    for m in metrics:
        if m.get(key, 0) > threshold:
            return m['step']
    return None


def first_within_5pct(metrics, key='test_loss'):
    """First step where metrics[step][key] is within 5% of run's min."""
    vmin = min(m[key] for m in metrics)
    for m in metrics:
        if m[key] < vmin * 1.05:
            return m['step']
    return None


# ============================================================================
# PLOT 1: exp2_v4_results.png  (6-panel grid)
# ============================================================================
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle('Exp 2 v4: Sweep $d_\\mathrm{latent}$ (unscaled MLP, hidden=256, $n$=500, '
             '$\\sigma_\\mathrm{noise}$=0.5, 5M steps)', fontsize=14)

# (a) Train vs Test Loss
ax = axes[0, 0]
for d in ds:
    m = runs[d]
    s = [x['step'] for x in m]
    ax.plot(s, [x['train_loss'] for x in m], color=cmap[d], lw=1.5, label=f'd={d} train')
    ax.plot(s, [x['test_loss']  for x in m], color=cmap[d], lw=1.5, ls='--', label=f'd={d} test')
ax.set_xlabel('Step'); ax.set_ylabel('Loss'); ax.set_title('(a) Train (solid) vs Test (dashed) Loss')
ax.legend(fontsize=5, ncol=2)

# (b) Score Error
ax = axes[0, 1]
for d in ds:
    m = runs[d]
    ax.plot([x['step'] for x in m], [x['score_error'] for x in m],
            color=cmap[d], lw=2, label=f'd={d}')
ax.set_xlabel('Step'); ax.set_ylabel('Score Error'); ax.set_title('(b) Score Error vs True Mixture Score')
ax.legend(fontsize=8)

# (c) Memorization Fraction
ax = axes[0, 2]
for d in ds:
    m = runs[d]
    ax.plot([x['step'] for x in m], [x['memorization_fraction'] for x in m],
            color=cmap[d], lw=2, label=f'd={d}')
ax.axhline(0.01, color='gray', ls=':', alpha=0.6, label='1% threshold')
ax.set_xlabel('Step'); ax.set_ylabel('Mem Fraction'); ax.set_title('(c) Memorization Fraction')
ax.legend(fontsize=8)

# (d) Mean NN Ratio
ax = axes[1, 0]
for d in ds:
    m = runs[d]
    ax.plot([x['step'] for x in m], [x['mean_nn_ratio'] for x in m],
            color=cmap[d], lw=2, label=f'd={d}')
ax.set_xlabel('Step'); ax.set_ylabel('NN Ratio'); ax.set_title('(d) Mean Nearest-Neighbor Ratio')
ax.legend(fontsize=8)

# (e) Gen Gap
ax = axes[1, 1]
for d in ds:
    m = runs[d]
    ax.plot([x['step'] for x in m], [x['gen_gap'] for x in m],
            color=cmap[d], lw=2, label=f'd={d}')
ax.set_xlabel('Step'); ax.set_ylabel('Gen Gap (test - train)')
ax.set_title('(e) Generalization Gap'); ax.legend(fontsize=8)

# (f) tau_mem and tau_gen vs d_latent
ax = axes[1, 2]
for d in ds:
    m = runs[d]
    last_step = m[-1]['step']
    tm = first_above(m, 'memorization_fraction', 0.01)
    tg = first_within_5pct(m, key='test_loss')
    ratio = d / D_INTRINSIC
    if tm is not None:
        ax.scatter(ratio, tm, s=150, c='red', zorder=5, edgecolors='black', linewidths=1.5)
        ax.annotate(f'd={d}', (ratio, tm), xytext=(8, 5), textcoords='offset points', fontsize=8)
    else:
        # never crossed 1%; mark at last logged step with triangle
        ax.scatter(ratio, last_step, s=150, marker='^', c='red', zorder=5,
                   edgecolors='black', linewidths=1.5, alpha=0.4)
        ax.annotate(f'd={d}', (ratio, last_step), xytext=(8, 5),
                    textcoords='offset points', fontsize=8, color='gray')
    if tg is not None:
        ax.scatter(ratio, tg, s=150, c='blue', zorder=5, edgecolors='black', linewidths=1.5)

ax.legend(handles=[
    Line2D([0],[0], marker='o', color='w', mfc='red',  ms=10, label=r'$\tau_\mathrm{mem}$ (>1%)'),
    Line2D([0],[0], marker='o', color='w', mfc='blue', ms=10, label=r'$\tau_\mathrm{gen}$ (5% of min)'),
    Line2D([0],[0], marker='^', color='w', mfc='gray', ms=10, label='never (last logged step)'),
], fontsize=8)
ax.set_xlabel('$d_\\mathrm{latent} / d_\\mathrm{intrinsic}$'); ax.set_ylabel('Step')
ax.set_title('(f) Timescales vs $d_\\mathrm{latent}$')

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/exp2_v4_results.png', dpi=150, bbox_inches='tight')
print('Saved exp2_v4_results.png')


# ============================================================================
# PLOT 2: exp2_v4_by_xaxis.png  (test loss vs step / epochs / flops / wall_time)
# ============================================================================
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle('Exp 2 v4: Test Loss by Different X-axes (unscaled MLP, hidden=256, $n$=500, $\\sigma_\\mathrm{noise}$=0.5)',
             fontsize=14)

xs = [('step', 'Step'), ('epochs', 'Epochs'),
      ('total_flops', 'Total FLOPs'), ('wall_time', 'Wall Time (s)')]
for idx, (xk, xl) in enumerate(xs):
    r, c = divmod(idx, 3)
    ax = axes[r, c]
    for d in ds:
        m = runs[d]
        ax.plot([x[xk] for x in m], [x['test_loss'] for x in m],
                color=cmap[d], lw=2, label=f'd={d}')
    ax.set_xlabel(xl); ax.set_ylabel('Test Loss')
    ax.set_title(f'Test Loss vs {xl}'); ax.legend(fontsize=8)

ax = axes[1, 1]
for d in ds:
    m = runs[d]
    ax.plot([x['epochs'] for x in m], [x['train_loss'] for x in m],
            color=cmap[d], lw=2, label=f'd={d}')
ax.set_xlabel('Epochs'); ax.set_ylabel('Train Loss')
ax.set_title('Train Loss vs Epochs'); ax.legend(fontsize=8)

ax = axes[1, 2]
for d in ds:
    m = runs[d]
    ax.plot([x['epochs'] for x in m], [x['gen_gap'] for x in m],
            color=cmap[d], lw=2, label=f'd={d}')
ax.set_xlabel('Epochs'); ax.set_ylabel('Gen Gap')
ax.set_title('Gen Gap vs Epochs'); ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/exp2_v4_by_xaxis.png', dpi=150, bbox_inches='tight')
print('Saved exp2_v4_by_xaxis.png')


# ============================================================================
# PLOT 3: exp2_v4_timescales.png  (2x3: tau_mem and tau_gen in steps + FLOPs)
# ============================================================================
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle('Exp 2 v4: Timescales (unscaled MLP, hidden=256, $n$=500, $\\sigma_\\mathrm{noise}$=0.5)', fontsize=14)

# Reference: use d=5's min test loss (within 5%)
ref_min = min(x['test_loss'] for x in runs[D_INTRINSIC])
ref_threshold = ref_min * 1.05

for col, (uk, ul) in enumerate([('step', 'Steps'), ('total_flops', 'FLOPs')]):
    # tau_mem
    ax = axes[0, col]
    for d in ds:
        m = runs[d]
        hit = next((x for x in m if x['memorization_fraction'] > 0.01), None)
        ratio = d / D_INTRINSIC
        if hit is not None:
            ax.scatter(ratio, hit[uk], s=150, c=cmap[d], zorder=5,
                       edgecolors='black', linewidths=1.5)
            ax.annotate(f'd={d}', (ratio, hit[uk]), xytext=(8, 5),
                        textcoords='offset points', fontsize=8)
        else:
            v = m[-1][uk]
            ax.scatter(ratio, v, s=150, marker='^', c=cmap[d], zorder=5,
                       edgecolors='black', linewidths=1.5, alpha=0.4)
            ax.annotate(f'd={d}', (ratio, v), xytext=(8, 5),
                        textcoords='offset points', fontsize=8, color='gray')
    ax.set_xlabel('$d_\\mathrm{latent} / d_\\mathrm{intrinsic}$')
    ax.set_ylabel(f'$\\tau_\\mathrm{{mem}}$ ({ul})')
    ax.set_title(f'$\\tau_\\mathrm{{mem}}$ vs $d_\\mathrm{{latent}}$ ({ul})\n(mem > 1%)')

    # tau_gen (within 5% of own min)
    ax = axes[1, col]
    for d in ds:
        m = runs[d]
        vmin = min(x['test_loss'] for x in m)
        hit = next((x for x in m if x['test_loss'] < vmin * 1.05), None)
        ratio = d / D_INTRINSIC
        if hit is not None:
            ax.scatter(ratio, hit[uk], s=150, c=cmap[d], zorder=5,
                       edgecolors='black', linewidths=1.5)
            ax.annotate(f'd={d}', (ratio, hit[uk]), xytext=(8, 5),
                        textcoords='offset points', fontsize=8)
    ax.set_xlabel('$d_\\mathrm{latent} / d_\\mathrm{intrinsic}$')
    ax.set_ylabel(f'$\\tau_\\mathrm{{gen}}$ ({ul})')
    ax.set_title(f'$\\tau_\\mathrm{{gen}}$ vs $d_\\mathrm{{latent}}$ ({ul})\n(within 5% of own min test loss)')

# tau_gen at d=5 reference quality (steps)
ax = axes[0, 2]
for d in ds:
    m = runs[d]
    hit = next((x for x in m if x['test_loss'] < ref_threshold), None)
    ratio = d / D_INTRINSIC
    if hit is not None:
        ax.scatter(ratio, hit['step'], s=150, c=cmap[d], zorder=5,
                   edgecolors='black', linewidths=1.5)
        ax.annotate(f'd={d}', (ratio, hit['step']), xytext=(8, 5),
                    textcoords='offset points', fontsize=8)
    else:
        v = m[-1]['step']
        ax.scatter(ratio, v, s=150, marker='^', c=cmap[d], zorder=5,
                   edgecolors='black', linewidths=1.5, alpha=0.4)
        ax.annotate(f'd={d}', (ratio, v), xytext=(8, 5),
                    textcoords='offset points', fontsize=8, color='gray')
ax.set_xlabel('$d_\\mathrm{latent} / d_\\mathrm{intrinsic}$'); ax.set_ylabel('Steps')
ax.set_title(f'$\\tau_\\mathrm{{gen}}$ at d=5 quality (test < {ref_threshold:.3f})')

# Divergence (test > 1.05 * train) by steps
ax = axes[1, 2]
for d in ds:
    m = runs[d]
    hit = next((x for x in m if x['step'] > 5000 and x['test_loss'] > x['train_loss'] * 1.05), None)
    ratio = d / D_INTRINSIC
    if hit is not None:
        ax.scatter(ratio, hit['step'], s=150, c=cmap[d], zorder=5,
                   edgecolors='black', linewidths=1.5)
        ax.annotate(f'd={d}', (ratio, hit['step']), xytext=(8, 5),
                    textcoords='offset points', fontsize=8)
    else:
        v = m[-1]['step']
        ax.scatter(ratio, v, s=150, marker='^', c=cmap[d], zorder=5,
                   edgecolors='black', linewidths=1.5, alpha=0.4)
        ax.annotate(f'd={d}', (ratio, v), xytext=(8, 5),
                    textcoords='offset points', fontsize=8, color='gray')
ax.set_xlabel('$d_\\mathrm{latent} / d_\\mathrm{intrinsic}$'); ax.set_ylabel('Step')
ax.set_title('Divergence time (steps)\n(test > 1.05 * train)')

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/exp2_v4_timescales.png', dpi=150, bbox_inches='tight')
print('Saved exp2_v4_timescales.png')


# ============================================================================
# Console summary
# ============================================================================
print('\nSummary:')
for d in ds:
    m = runs[d]
    vmin = min(x['test_loss'] for x in m)
    tg = first_within_5pct(m, 'test_loss')
    tm = first_above(m, 'memorization_fraction', 0.01)
    td = next((x['step'] for x in m if x['step'] > 5000 and x['test_loss'] > x['train_loss'] * 1.05), None)
    final = m[-1]
    print(f'd={d:3d} ({d/D_INTRINSIC:.0f}x) | tau_gen={tg} | tau_mem={tm or "never"} | '
          f'diverge={td or "never"} | min_test={vmin:.4f} | final_mem={final["memorization_fraction"]:.4f} | '
          f'last_step={final["step"]}')
