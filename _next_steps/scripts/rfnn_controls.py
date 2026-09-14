"""RFNN spectrum under the paper's three width controls, 3 seeds, with fitted d_lat exponents.

Controls (matching ICLR_2026/sec-rfnn.tex):
  1. p = 64*d_lat                       (psi_p = 64 fixed)
  2. p = d_lat + n + 300                (psi_p -> 1)
  3. p = 1280 fixed, sigma_perp^2 = 1.25/(d_lat - d_int)   (fixed total null energy)

Key result (2026-08-31): controls 1 and 2 give IDENTICAL edges at every d_lat despite psi_p differing
5.8x at d_lat=80 -> the bulk edges are p-independent, which kills the old /psi_p table.
Fitted exponents at p=64d: signal ~ d_lat^-0.65, noise-dim ~ d_lat^-0.73, sample ~ d_lat^+0.24.
The gap from the naive -1 is the drifting effective Hermite coefficient a_1(q), q = tr(M_t)/d_lat.

Run:  python3 _next_steps/scripts/rfnn_controls.py     (~80 s)
"""
import numpy as np

T = 0.01; E2T = np.exp(-2 * T); DELTA = 1 - E2T
D_INT, K, S, SIG_SIG = 5, 10, 3.0, 1.0


def spec(d_lat, n, p, sig_perp, seed, n_mc=40):
    rng = np.random.default_rng(seed)
    C = rng.normal(size=(K, D_INT)); C /= np.linalg.norm(C, axis=1, keepdims=True); C *= S
    lab = rng.integers(0, K, size=n); x = np.zeros((n, d_lat))
    x[:, :D_INT] = C[lab] + SIG_SIG * rng.normal(size=(n, D_INT))
    if d_lat > D_INT:
        x[:, D_INT:] = sig_perp * rng.normal(size=(n, d_lat - D_INT))
    W = rng.normal(size=(p, d_lat)); U = np.zeros((p, p))
    for i0 in range(0, n, 64):
        xb = x[i0:i0 + 64]; b = xb.shape[0]
        eta = rng.normal(size=(n_mc, b, d_lat))
        xt = np.exp(-T) * xb[None] + np.sqrt(DELTA) * eta
        Phi = np.tanh(xt.reshape(-1, d_lat) @ W.T / np.sqrt(d_lat)) / np.sqrt(p)
        U += Phi.T @ Phi
    U /= (n * n_mc)
    ev = np.linalg.eigvalsh(U)[::-1]
    m = lambda a: float(np.median(a)) if len(a) else np.nan
    return m(ev[:D_INT]), m(ev[D_INT:d_lat]), m(ev[d_lat:d_lat + n]), float(ev.sum())


N = 500; SEEDS = [11, 22, 33]; DL = [10, 20, 40, 80]
CONTROLS = [
    ("CONTROL 1: p = 64*d_lat  (psi_p = 64 fixed)",                          lambda d: 64 * d,      lambda d: 0.5),
    ("CONTROL 2: p = d_lat+n+300  (psi_p -> 1)",                              lambda d: d + N + 300, lambda d: 0.5),
    ("CONTROL 3: p = 1280 fixed, fixed null energy sig_perp^2 = 1.25/(d-5)",  lambda d: 1280,        lambda d: np.sqrt(1.25 / (d - D_INT))),
]

if __name__ == "__main__":
    cache = {}
    for name, pfun, spfun in CONTROLS:
        print(f"\n=== {name} ===")
        print(f"{'d_lat':>6}{'p':>7}{'sig_perp':>10}{'signal':>12}{'noise-dim':>12}{'sample':>12}{'tr(U)':>9}")
        for d in DL:
            p = pfun(d); sp = spfun(d)
            r = np.array([spec(d, N, p, sp, s) for s in SEEDS]); mu = r.mean(0)
            cache[(name, d)] = mu
            print(f"{d:>6}{p:>7}{sp:>10.4f}{mu[0]:>12.3e}{mu[1]:>12.3e}{mu[2]:>12.3e}{mu[3]:>9.4f}")

    print("\n=== scaling exponents (fit log(edge) ~ a*log(d_lat)) ===")
    L = np.log(DL)
    fit = lambda y: np.polyfit(L, np.log(y), 1)[0]
    for name, _, _ in CONTROLS:
        S_ = [cache[(name, d)][0] for d in DL]; Nz = [cache[(name, d)][1] for d in DL]; Sm = [cache[(name, d)][2] for d in DL]
        print(f"  {name[:12]:>12}:  signal d_lat^{fit(S_):+.2f}   noise-dim d_lat^{fit(Nz):+.2f}   sample d_lat^{fit(Sm):+.2f}")
    print("\n  old theorem predicts exponent 0.00 for all three under control 1 (edges ~ 1/psi_p, psi_p fixed).")
