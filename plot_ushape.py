"""Plot the U-shape: min(score_error) vs d_latent for the MLP at sigma_noise=0.01.
Both unscaled (hidden=256) and scaled (hidden=8*d_latent) variants.
"""
import json, os, re
import matplotlib.pyplot as plt

def collect(base):
    rows = []
    for cfg in os.listdir(base):
        m = re.search(r'_d(\d+)_', cfg)
        if not m: continue
        fp = f'{base}/{cfg}/metrics.jsonl'
        if not os.path.exists(fp): continue
        M = [json.loads(l) for l in open(fp)]
        se_min = min(x.get('score_error', 1e9) for x in M)
        rows.append((int(m.group(1)), se_min))
    return sorted(rows)

unscaled = collect('results_mlp_exp2_sn001')
scaled = collect('results_mlp_exp2_scaled_sn001')

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot([d for d, _ in unscaled], [s for _, s in unscaled], 'o-',
        color='C0', label='unscaled MLP (hidden=256)', linewidth=2, markersize=8)
ax.plot([d for d, _ in scaled], [s for _, s in scaled], 's-',
        color='C3', label=r'scaled MLP (hidden=$8 d_{\mathrm{latent}}$)',
        linewidth=2, markersize=8)
ax.axvline(5, color='gray', linestyle=':', alpha=0.6)
ax.text(5.3, ax.get_ylim()[1]*0.93, r'$d_{\mathrm{intrinsic}}=5$', fontsize=9, color='gray')
ax.set_xlabel(r'$d_{\mathrm{latent}}$')
ax.set_ylabel(r'min score error over training')
ax.set_title(r'MLP U-shape: score error vs $d_{\mathrm{latent}}$ at $\sigma_{\mathrm{noise}}=0.01$')
ax.set_xscale('log')
ax.legend(loc='best')
ax.grid(alpha=0.3, which='both')
fig.tight_layout()

out = 'figures/sigma_noise_0.01/exp2_mlp/u_shape.png'
fig.savefig(out, dpi=140)
fig.savefig('ICML/figures/u_shape.png', dpi=140)
print(f'wrote {out} and ICML/figures/u_shape.png')
print()
print('Data:')
print('  unscaled:', unscaled)
print('  scaled:  ', scaled)
