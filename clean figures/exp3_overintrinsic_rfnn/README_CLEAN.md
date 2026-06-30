# Exp 3 over-intrinsic RFNN

Clean RFNN figures for the fixed-latent Exp 3 over-intrinsic sweep.

Fixed observed `d_latent=20`; data are regenerated for each source
`d_intrinsic`:

`d_intrinsic = 2, 5, 8, 12, 16, 20, 25, 30, 35, 40`

For `d_intrinsic > 20`, the source GMM is sampled in `d_intrinsic`
dimensions and compressed into the fixed 20D latent space with a random
energy-preserving map.

Width folders:

- `p_n_plus_d_plus_r`: `p=d_latent+n+300`
- `p64`: `p=64*d_latent`
- `p_fixed`: `p=1800`

Each folder contains:

- `hist_by_index_region.png/pdf`
- `spectrum.png/pdf`
- per-cell `eigenvalues.npy`

Index convention uses `active_dim = min(d_intrinsic, d_latent)`.
