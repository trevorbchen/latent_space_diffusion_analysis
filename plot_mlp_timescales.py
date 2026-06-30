"""Plot tau_gen and tau_mem extracted from MLP exp 2 (d_latent sweep) and exp 3 (d_intrinsic sweep).

tau_gen = step at which score_error reaches its minimum
tau_mem = first step where memorization_fraction > MEM_THRESHOLD
"""
import json, os, re
import matplotlib.pyplot as plt

BASE = 'figures'
MEM_THRESHOLD = 0.005  # 0.5% — most MLP runs never reach 1%

def load_metrics(path):
    out = []
    with open(path) as f:
        for line in f:
            try: out.append(json.loads(line))
            except: pass
    return out

def tau_gen(metrics):
    se = [(m['step'], m['score_error']) for m in metrics if 'score_error' in m]
    return min(se, key=lambda x: x[1])[0] if se else None

def tau_mem(metrics, threshold):
    for m in metrics:
        if m.get('memorization_fraction', 0) > threshold:
            return m['step']
    return None

def collect(folder, key_regex):
    """Return list of (key, tau_gen, tau_mem) sorted by key."""
    rows = []
    if not os.path.isdir(folder): return rows
    for cfg in os.listdir(folder):
        m = re.search(key_regex, cfg)
        if not m: continue
        key = int(m.group(1))
        mp = f'{folder}/{cfg}/metrics.jsonl'
        if not os.path.exists(mp): continue
        metrics = load_metrics(mp)
        rows.append((key, tau_gen(metrics), tau_mem(metrics, MEM_THRESHOLD)))
    return sorted(rows)

def plot_pair(rows, xlabel, title, out_path, last_step=300000):
    keys = [r[0] for r in rows]
    tg = [r[1] for r in rows]
    tm = [r[2] if r[2] is not None else last_step for r in rows]
    tm_censored = [r[2] is None for r in rows]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(keys, tg, 'o-', color='C0', label=r'$\tau_{\mathrm{gen}}$ (argmin score error)')
    ax.plot(keys, tm, 's--', color='C3',
            label=fr'$\tau_{{\mathrm{{mem}}}}$ (mem.frac > {MEM_THRESHOLD:.0%})')
    # Mark censored tau_mem points (never crossed threshold)
    for k, t, c in zip(keys, tm, tm_censored):
        if c:
            ax.annotate('never', (k, t), textcoords='offset points', xytext=(0, 6),
                        ha='center', fontsize=8, color='C3')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('training step')
    ax.set_title(title)
    ax.set_yscale('log')
    ax.legend(loc='best')
    ax.grid(alpha=0.3, which='both')
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f'  wrote {out_path}')

for sn in ['0.5', '0.01']:
    # exp 2: d_intrinsic = 5 fixed, vary d_latent
    folder = f'{BASE}/sigma_noise_{sn}/exp2_mlp/raw_data'
    rows = collect(folder, r'd(\d+)_n')
    out = f'{BASE}/sigma_noise_{sn}/exp2_mlp/exp2_mlp_timescales.png'
    plot_pair(rows, r'$d_{\mathrm{latent}}$',
              fr'MLP Exp 2: timescales vs $d_{{\mathrm{{latent}}}}$  ($\sigma_{{\mathrm{{noise}}}}={sn}$, $d_{{\mathrm{{int}}}}=5$)',
              out)

    # exp 3: d_latent = 20 fixed, vary d_intrinsic
    folder = f'{BASE}/sigma_noise_{sn}/exp3_mlp/raw_data'
    rows = collect(folder, r'di(\d+)_d')
    out = f'{BASE}/sigma_noise_{sn}/exp3_mlp/exp3_mlp_timescales.png'
    plot_pair(rows, r'$d_{\mathrm{intrinsic}}$',
              fr'MLP Exp 3: timescales vs $d_{{\mathrm{{intrinsic}}}}$  ($\sigma_{{\mathrm{{noise}}}}={sn}$, $d_{{\mathrm{{latent}}}}=20$)',
              out)
