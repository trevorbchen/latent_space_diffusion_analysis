# Clean four-bulk candidates

Candidate RFNN spectrum runs for finding cleaner four-bulk figures.

These are **not** marked as paper-ready. The earlier `clean graphs` package
was removed so these can be inspected first.

## Common setup

All runs here are spectral-only RFNN runs:

```text
activation        = tanh
MC samples for U  = 500
d_intrinsic       = 5 for Exp-2-style d_latent sweeps
n                 = 500
t_fixed           = 0.01
width rule        = p = d_latent + n + 300 unless noted otherwise
```

The plots color eigenvalues by theoretical index region:

```text
signal:     eigs[:d_intrinsic]
noise-dim:  eigs[d_intrinsic:d_latent]
sample:     eigs[d_latent:d_latent+n]
rank-null:  eigs[d_latent+n:]
```

Histogram bins use full min/max range per panel, not percentile clipping.

## Candidate groups

### Data-kind comparison

Folders:

```text
gaussian_exp2
gmm_exp2
gaussian_exp3
gmm_exp3
```

Purpose: compare pure anisotropic Gaussian data against anisotropic GMM data.

### Sigma-noise sweep

Folders:

```text
sigma_0.1_gaussian_exp2
sigma_0.2_gaussian_exp2
sigma_0.3_gaussian_exp2
sigma_0.5_gaussian_exp2
sigma_0.1_gmm_exp2
sigma_0.2_gmm_exp2
sigma_0.3_gmm_exp2
sigma_0.5_gmm_exp2
```

Purpose: find a null-noise level that separates signal, noise-dim, sample,
and rank-null cleanly without creating extra visual shoulders.

### Data-scale sweep

Folders:

```text
sigscale_0.5_gaussian_exp2
sigscale_0.75_gaussian_exp2
sigscale_1_gaussian_exp2
sigscale_1.5_gaussian_exp2
centerscale_0.75_gmm_exp2
centerscale_1_gmm_exp2
centerscale_1.5_gmm_exp2
centerscale_2_gmm_exp2
```

Purpose: keep `tanh` away from saturation by lowering signal scale for pure
Gaussian data or lowering center scale for GMM data.

## Commands used

```powershell
python clean_four_bulk_sweep.py --preset data-kind
python clean_four_bulk_sweep.py --preset sigma
python clean_four_bulk_sweep.py --preset scale
```

