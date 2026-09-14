"""Independent RFNN feature-spectrum simulation (not the repo's code path).

Tests the d_lat- and n-scaling of the four bulk edges of
U = (1/n) sum_mu E_xi[phi(x_t^mu) phi(x_t^mu)^T],  phi(x) = tanh(W x / sqrt(d_lat)) / sqrt(p),
with the paper's Exp-1 data (d_int=5, k=10 centers of norm s=3, sigma_sig=1, sigma_perp=0.5, n=500, t=0.01).

NOTE: this uses the THEORY normalization (1/sqrt(p) on phi). The repo's compute_U omits it,
so saved eigenvalues in sigma_noise_*/ are p x the numbers here.

TEST A: d_lat sweep at p = 64*d_lat (psi_p fixed).  Old theorem (edges ~ 1/psi_p) predicts CONSTANT edges.
TEST B: n sweep at fixed d_lat, p.                  Old theorem predicts sample edge CONSTANT in n.
Both predictions fail; see rfnn_controls.py for the three width controls with seeds.

Run:  python3 _next_steps/scripts/rfnn_spectrum.py     (~15 s)
"""
import numpy as np

T_DIFF = 0.01
E2T = np.exp(-2 * T_DIFF); DELTA = 1.0 - E2T
D_INT, K, S, SIG_SIG, SIG_PERP = 5, 10, 3.0, 1.0, 0.5
MU1 = 0.6057  # E[z tanh z], z ~ N(0,1)  (unit-variance Hermite-1 coefficient; NOT constant across the sweep)


def build_U(d_lat, n, p, n_mc=40, seed=0):
    rng = np.random.default_rng(seed)
    C = rng.normal(size=(K, D_INT)); C /= np.linalg.norm(C, axis=1, keepdims=True); C *= S
    lab = rng.integers(0, K, size=n)
    x = np.zeros((n, d_lat))
    x[:, :D_INT] = C[lab] + SIG_SIG * rng.normal(size=(n, D_INT))
    if d_lat > D_INT:
        x[:, D_INT:] = SIG_PERP * rng.normal(size=(n, d_lat - D_INT))
    W = rng.normal(size=(p, d_lat))
    U = np.zeros((p, p))
    CH = 64
    for i0 in range(0, n, CH):
        xb = x[i0:i0 + CH]; b = xb.shape[0]
        eta = rng.normal(size=(n_mc, b, d_lat))
        xt = np.exp(-T_DIFF) * xb[None] + np.sqrt(DELTA) * eta
        z = xt.reshape(-1, d_lat) @ W.T / np.sqrt(d_lat)
        Phi = np.tanh(z) / np.sqrt(p)
        U += Phi.T @ Phi
    U /= (n * n_mc)
    return np.linalg.eigvalsh(U)[::-1]


def blocks(ev, d_lat, n):
    med = lambda a: float(np.median(a)) if len(a) else float('nan')
    return (med(ev[:D_INT]), med(ev[D_INT:d_lat]), med(ev[d_lat:d_lat + n]),
            med(ev[d_lat + n:]), float(ev.sum()))


if __name__ == "__main__":
    alpha2 = E2T * (S**2 / D_INT + SIG_SIG**2) + DELTA
    beta2 = E2T * SIG_PERP**2 + DELTA
    print(f"alpha_t^2={alpha2:.4f}  beta_t^2={beta2:.4f}  mu1^2={MU1**2:.4f}  Delta_t={DELTA:.5f}\n")

    print("=== TEST A: d_lat sweep, p = 64*d_lat, n = 500 ===")
    print("(old theorem: edges ~ 1/psi_p -> CONSTANT.  corrected: edges fall with d_lat)")
    print(f"{'d_lat':>6}{'p':>7}{'tr(U)':>9}{'sig_med':>11}{'noise_med':>11}{'samp_med':>11}{'rank_med':>11}"
          f"{'pred_sig/psi':>13}{'pred_sig/dlat':>14}")
    resA = []
    for d_lat in [10, 20, 40, 80]:
        p = 64 * d_lat; n = 500
        ev = build_U(d_lat, n, p, seed=1)
        s_, nz_, sm_, rn_, tr_ = blocks(ev, d_lat, n)
        psi_p = p / d_lat
        print(f"{d_lat:>6}{p:>7}{tr_:>9.4f}{s_:>11.3e}{nz_:>11.3e}{sm_:>11.3e}{rn_:>11.3e}"
              f"{MU1**2 * alpha2 / psi_p:>13.3e}{MU1**2 * alpha2 / d_lat:>14.3e}")
        resA.append((d_lat, s_, nz_, sm_, tr_))
    print("\n  ratios vs d_lat=10 (constant = old theorem, 1/x = naive corrected):")
    d0, s0, n0, m0, t0 = resA[0]
    for d_lat, s_, nz_, sm_, tr_ in resA:
        print(f"   d_lat={d_lat:>3}  x{d_lat / d0:>4.0f}:  sig {s_ / s0:>6.3f}   noise {nz_ / n0:>6.3f}"
              f"   sample {sm_ / m0:>6.3f}   trace {tr_ / t0:>6.3f}   (1/x = {d0 / d_lat:.3f})")

    print("\n=== TEST B: n sweep, d_lat = 20, p = 1280 fixed ===")
    print("(old theorem: sample edge CONSTANT in n.  corrected: falls with n)")
    print(f"{'n':>6}{'tr(U)':>9}{'sig_med':>11}{'noise_med':>11}{'samp_med':>11}")
    resB = []
    for n in [250, 500, 1000]:
        ev = build_U(20, n, 1280, seed=2)
        s_, nz_, sm_, rn_, tr_ = blocks(ev, 20, n)
        print(f"{n:>6}{tr_:>9.4f}{s_:>11.3e}{nz_:>11.3e}{sm_:>11.3e}")
        resB.append((n, sm_))
    n0, m0 = resB[0]
    print("\n  sample-bulk ratios vs n=250:")
    for n, sm_ in resB:
        print(f"   n={n:>5}  x{n / n0:>4.0f}:  sample {sm_ / m0:>6.3f}   (1/x = {n0 / n:.3f})")
