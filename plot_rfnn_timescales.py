"""Plot tau_gen and tau_mem extracted from RFNN exp 2 (d_latent sweep) and exp 3 (d_intrinsic sweep).

RFNN metrics do not include memorization_fraction (Bonnaire-style),
so we use the train/test gap as a memorization proxy:

    tau_gen = step at which score_error reaches its minimum
    tau_mem = first step where gen_gap > GAP_THRESHOLD (= 0.02)
"""
import json, os, re
import matplotlib.pyplot as plt

BASE = 'figures'
GAP_THRESHOLD = 0.02

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
        if m.get('gen_gap', 0) > threshold:
            return m['step']
    return None

def collect(folder, key_regex):
    rows = []
    if not os.path.isdir(folder): return rows
    for cfg in os.listdir(folder):
        m = re.search(key_regex, cfg)
        if not m: continue
        key = int(m.group(1))
        mp = f'{folder}/{cfg}/metrics.jsonl'
        if not os.path.exists(mp): continue
        metrics = load_metrics(mp)
        rows.append((key, tau_gen(metrics), tau_mem(metrics, GAP_THRESHOLD)))
    return sorted(rows)

def plot_pair(rows, xlabel, title, out_path, last_step=300000):
    keys = [r[0] for r in rows]
    tg = [r[1] for r in rows]
    tm = [r[2] if r[2] is not None else last_step for r in rows]
    tm_censored = [r[2] is None for r in rows]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(keys, tg, 'o-', color='C0', label=r'$\tau_{\mathrm{gen}}$ (argmin score error)')
    ax.plot(keys, tm, 's--', color='C3',
            label=fr'$\tau_{{\mathrm{{mem}}}}$ (gen gap > {GAP_THRESHOLD})')
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
    folder = f'{BASE}/sigma_noise_{sn}/exp2_rfnn/raw_data'
    rows = collect(folder, r'd(\d+)_n')
    out = f'{BASE}/sigma_noise_{sn}/exp2_rfnn/exp2_rfnn_timescales.png'
    plot_pair(rows, r'$d_{\mathrm{latent}}$',
              fr'RFNN Exp 2: timescales vs $d_{{\mathrm{{latent}}}}$  ($\sigma_{{\mathrm{{noise}}}}={sn}$, $d_{{\mathrm{{int}}}}=5$)',
              out)

    folder = f'{BASE}/sigma_noise_{sn}/exp3_rfnn/raw_data'
    rows = collect(folder, r'di(\d+)_d')
    out = f'{BASE}/sigma_noise_{sn}/exp3_rfnn/exp3_rfnn_timescales.png'
    plot_pair(rows, r'$d_{\mathrm{intrinsic}}$',
              fr'RFNN Exp 3: timescales vs $d_{{\mathrm{{intrinsic}}}}$  ($\sigma_{{\mathrm{{noise}}}}={sn}$, $d_{{\mathrm{{latent}}}}=20$)',
              out)
