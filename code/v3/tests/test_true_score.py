"""Regression test for the analytic GMM score.

Guards the frame convention in `precompute_sigma_t`. The generator applies
`data = data @ Q.T`, i.e. x_rot = Q x_orig as a column vector, so the data-frame
covariance is Q D Q^T and its inverse is Q D^{-1} Q^T. A previous version
computed Q^T D^{-1} Q, which has the right eigenvalues and the wrong
eigenvectors: a valid precision matrix for a *different* Gaussian. The error is
exactly inert at d_latent == d_intrinsic (D is isotropic there), which is why it
survived so long.

The check is Stein's identity, E[s(x) x^T] = -I for any smooth density, which
holds independently of how Sigma_t is parameterised.

Run:  python3 code/v3/tests/test_true_score.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.data_synthetic import generate_data          # noqa: E402
from lib.true_score import precompute_sigma_t, true_score  # noqa: E402

N = 40_000
TOL_DIAG = 0.02   # |mean diag + 1|
TOL_OFF = 0.02    # mean |off-diagonal|


def stein_residual(d_intrinsic: int, d_latent: int, sigma_noise: float,
                   t: float, seed: int = 42):
    """Return (mean diagonal, mean |off-diagonal|) of E[s(x_t) x_t^T]."""
    data, _, means, Q = generate_data(
        N, d_intrinsic, d_latent, k=10, sigma_noise=sigma_noise, seed=seed
    )
    gen = torch.Generator().manual_seed(seed)
    x_t = (math.exp(-t) * data
           + math.sqrt(1 - math.exp(-2 * t)) * torch.randn(data.shape, generator=gen))

    sigma_t_inv, log_det = precompute_sigma_t(
        d_intrinsic, d_latent, Q, sigma_noise, t
    )
    s = true_score(x_t, t, means, sigma_t_inv=sigma_t_inv, log_det_sigma_t=log_det)

    c = (s.T @ x_t) / N
    diag = c.diagonal().mean().item()
    off = (c - torch.diag(c.diagonal())).abs().mean().item()
    return diag, off


def main() -> int:
    cases = [
        # (d_intrinsic, d_latent, sigma_noise, t)
        (5, 5, 0.5, 0.1),     # isotropic block: the old bug was inert here
        (5, 20, 0.5, 0.1),
        (5, 40, 0.5, 0.1),
        (5, 20, 0.01, 0.1),   # strongest anisotropy, where the bug was worst
        (5, 20, 0.5, 0.01),
    ]
    failures = 0
    print(f"{'d_int':>6}{'d_lat':>7}{'sig_perp':>10}{'t':>7}"
          f"{'mean diag':>12}{'mean |off|':>12}  result")
    for d_int, d_lat, sp, t in cases:
        diag, off = stein_residual(d_int, d_lat, sp, t)
        ok = abs(diag + 1.0) < TOL_DIAG and off < TOL_OFF
        failures += not ok
        print(f"{d_int:>6}{d_lat:>7}{sp:>10}{t:>7}"
              f"{diag:>12.4f}{off:>12.4f}  {'ok' if ok else 'FAIL'}")

    if failures:
        print(f"\n{failures} case(s) failed Stein's identity: "
              f"E[s(x) x^T] != -I. Check the Q orientation in "
              f"precompute_sigma_t (should be Q D^-1 Q^T).")
        return 1
    print("\nall cases satisfy Stein's identity")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
