# p_fixed

RFNN Exp-3 over-intrinsic sweep.

Fixed observed `d_latent=20`. Data are regenerated for each
`d_intrinsic`. For `d_intrinsic > d_latent`, the source GMM is sampled in
`d_intrinsic` dimensions and compressed into the fixed latent space with a
random energy-preserving map.

Width mode: `fixed`.
Feature width: `1800`.
MC samples: `500`.

Index regions use `active_dim = min(d_intrinsic, d_latent)`:

- signal/source: `eigs[:active_dim]`
- noise-dim: `eigs[active_dim:d_latent]`
- sample: `eigs[d_latent:d_latent+n]`
- rank-null: `eigs[d_latent+n:]`
