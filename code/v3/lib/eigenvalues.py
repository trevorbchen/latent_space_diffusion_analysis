"""Feature-correlation matrix U(t) and bulk-evolution tracking for the RFNN.

U_{ab} = (1/n) sum_mu E_eps[ phi_a(x_t^mu(eps)) phi_b(x_t^mu(eps)) ],
phi(x) = tanh(W x), shape (n, p) -> (p, p).

We expose:
- compute_U: standard pre-/post-training U(t) computation (matches v2)
- bulk_indices: partition the sorted spectrum into the four bulks predicted
  by the paper (signal, noise-dim, sample, rank-null) using d_intrinsic,
  d_latent, n.
- absorption_fraction: how much of each bulk's eigenmode subspace has been
  absorbed by the readout A — used to populate Figs 12-13.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch


def compute_U(W: torch.Tensor,
              data: torch.Tensor,
              t: float,
              n_noise_samples: int = 50) -> torch.Tensor:
    """E_eps[ phi(x_t) phi(x_t)^T ] over the training set, MC over eps."""
    p, d = W.shape
    n = data.shape[0]
    delta_t = 1 - math.exp(-2 * t)
    e_neg_t = math.exp(-t)

    W_cpu = W.detach().cpu()
    data_cpu = data.detach().cpu()
    U = torch.zeros(p, p)
    for _ in range(n_noise_samples):
        noise = torch.randn(n, d)
        x_t = e_neg_t * data_cpu + math.sqrt(delta_t) * noise
        phi = torch.tanh(x_t @ W_cpu.T)
        U += phi.T @ phi / n
    U /= n_noise_samples
    return U


def eigendecompose_U(U: torch.Tensor):
    """Returns (eigvals desc, eigvecs cols matched to eigvals) — float64 path."""
    U64 = U.detach().cpu().double()
    eigvals, eigvecs = torch.linalg.eigh(U64)        # ascending
    eigvals = eigvals.flip(0)
    eigvecs = eigvecs.flip(1)
    return eigvals.numpy(), eigvecs.numpy()


# ---------------------------------------------------------------------------
# Bulk partitioning
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BulkLayout:
    signal: slice
    noise_dim: slice
    sample: slice
    rank_null: slice

    def as_dict(self) -> dict[str, slice]:
        return {
            'signal':    self.signal,
            'noise_dim': self.noise_dim,
            'sample':    self.sample,
            'rank_null': self.rank_null,
        }


def bulk_indices(d_intrinsic: int, d_latent: int, n: int, p: int) -> BulkLayout:
    """Paper's predicted boundaries land at indices d_intrinsic and d_latent.

    Layout (over a length-p sorted-descending eigenvalue array):
        [0, d_intrinsic)               -- signal bulk
        [d_intrinsic, d_latent)        -- noise-dim bulk
        [d_latent,    d_latent + n)    -- sample bulk
        [d_latent + n, p)              -- rank-null tail
    """
    s_end = d_intrinsic
    nd_end = d_latent
    samp_end = min(d_latent + n, p)
    return BulkLayout(
        signal=slice(0, s_end),
        noise_dim=slice(s_end, nd_end),
        sample=slice(nd_end, samp_end),
        rank_null=slice(samp_end, p),
    )


# ---------------------------------------------------------------------------
# Per-bulk absorption (Figs 12-13)
# ---------------------------------------------------------------------------

def absorption_per_bulk(A: torch.Tensor,
                        eigvecs_U: np.ndarray,
                        layout: BulkLayout,
                        eigvals_U: np.ndarray,
                        epsilon: float = 1e-12) -> dict[str, dict[str, float]]:
    """Per-bulk readout mass after rotating A into U's eigenbasis.

    A: (d_latent, p). eigvecs_U: (p, p) columns = U eigenvectors sorted
    descending in eigenvalue.

    For each bulk B with index range I_B we report:
      mass     = ||(A V)_{:, I_B}||_F^2          (raw squared mass, unbounded)
      fraction = mass / ||A||_F^2                (in [0, 1], sums to 1)

    The fraction is the cleanest mechanism plot (Fig 13): signal grows to
    ~1 first, then noise-dim claims its share, then sample, then rank-null.
    The raw mass is kept for diagnostics (e.g. comparing across timesteps
    on the same axes as Fig 12).
    """
    A_np = A.detach().cpu().double().numpy()
    AV = A_np @ eigvecs_U                        # in U-eigenbasis
    total_mass = float((A_np ** 2).sum())
    denom = max(total_mass, epsilon)

    out: dict[str, dict[str, float]] = {}
    for name, sl in layout.as_dict().items():
        if sl.start >= sl.stop:
            out[name] = {'mass': 0.0, 'fraction': 0.0}
            continue
        mass = float((AV[:, sl] ** 2).sum())
        out[name] = {'mass': mass, 'fraction': mass / denom}
    return out


# ---------------------------------------------------------------------------
# Spectrum snapshot summary
# ---------------------------------------------------------------------------

def bulk_summary(eigvals: np.ndarray, layout: BulkLayout) -> dict[str, dict]:
    """Min / max / median / count per bulk, for compact JSONL logging."""
    out: dict[str, dict] = {}
    for name, sl in layout.as_dict().items():
        chunk = eigvals[sl]
        if chunk.size == 0:
            out[name] = {'count': 0}
            continue
        out[name] = {
            'count':  int(chunk.size),
            'min':    float(chunk.min()),
            'max':    float(chunk.max()),
            'median': float(np.median(chunk)),
        }
    return out
