"""Plot min(score_error) vs d_latent for all 6 sweeps:
  MLP unscaled, MLP scaled, RFNN  ×  sigma_noise in {0.01, 0.5}.
"""
import json, os, re
import matplotlib.pyplot as plt

def collect(*bases):
    rows = {}
    for base in bases:
        if not os.path.isdir(base): continue
        for cfg in os.listdir(base):
            m = re.search(r'_d(\d+)_', cfg)
            if not m: continue
            fp = f'{base}/{cfg}/metrics.jsonl'
            if not os.path.exists(fp): continue
            M = [json.loads(l) for l in open(fp)]
            se_min = min(x.get('score_error', 1e9) for x in M)
            d = int(m.group(1))
            if d not in rows or se_min < rows[d]:
                rows[d] = se_min
    return sorted(rows.items())

groups = {
    'MLP unscaled, $\\sigma_{\\mathrm{noise}}=0.01$': collect('results_mlp_exp2_sn001'),
    'MLP unscaled, $\\sigma_{\\mathrm{noise}}=0.5$':  collect('results_mlp_exp2_sn05'),
    'MLP scaled, $\\sigma_{\\mathrm{noise}}=0.01$':   collect('results_mlp_exp2_scaled_sn001'),
    'MLP scaled, $\\sigma_{\\mathrm{noise}}=0.5$':    collect('results_mlp_exp2_scaled_sn05'),
    'RFNN, $\\sigma_{\\mathrm{noise}}=0.01$':         collect('results_rfnn_exp2_sn001'),
    'RFNN, $\\sigma_{\\mathrm{noise}}=0.5$':          collect('results_rfnn_exp2v3', 'results_rfnn_exp2_wide'),
}

# 2x3 grid: rows = sigma, cols = MLP_unscaled / MLP_scaled / RFNN
fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True)
col_titles = ['MLP unscaled (hidden=256)',
              r'MLP scaled (hidden=$8 d_{\mathrm{latent}}$)',
              r'RFNN ($p = 64 d_{\mathrm{latent}}$)']
sigmas = [('0.01', 'C0'), ('0.5', 'C3')]
keys_in_order = [
    ['MLP unscaled, $\\sigma_{\\mathrm{noise}}=0.01$', 'MLP scaled, $\\sigma_{\\mathrm{noise}}=0.01$', 'RFNN, $\\sigma_{\\mathrm{noise}}=0.01$'],
    ['MLP unscaled, $\\sigma_{\\mathrm{noise}}=0.5$',  'MLP scaled, $\\sigma_{\\mathrm{noise}}=0.5$',  'RFNN, $\\sigma_{\\mathrm{noise}}=0.5$'],
]

for r, ((sigma, color), keys) in enumerate(zip(sigmas, keys_in_order)):
    for c, (key, ct) in enumerate(zip(keys, col_titles)):
        ax = axes[r, c]
        rows = groups.get(key, [])
        if rows:
            ax.plot([d for d, _ in rows], [s for _, s in rows], 'o-',
                    color=color, linewidth=2, markersize=7)
        ax.axvline(5, color='gray', linestyle=':', alpha=0.5, label=r'$d_{\mathrm{int}}=5$')
        ax.set_xscale('log')
        ax.grid(alpha=0.3, which='both')
        if r == 0: ax.set_title(ct, fontsize=11)
        if c == 0: ax.set_ylabel(fr'min score error  ($\sigma_{{\mathrm{{noise}}}}={sigma}$)')
        if r == 1: ax.set_xlabel(r'$d_{\mathrm{latent}}$')

fig.suptitle(r'U-shape audit: min(score error) vs $d_{\mathrm{latent}}$ across 6 sweeps',
             fontsize=13, y=1.00)
fig.tight_layout()
out = 'figures/u_shape_all.png'
fig.savefig(out, dpi=140, bbox_inches='tight')
fig.savefig('ICML/figures/u_shape_all.png', dpi=140, bbox_inches='tight')
print(f'wrote {out}')

print()
print('Numbers:')
for k, rows in groups.items():
    print(f'  {k}: {[(d, round(s, 3)) for d, s in rows]}')
