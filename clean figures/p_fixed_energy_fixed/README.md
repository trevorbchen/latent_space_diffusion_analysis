# Clean figures: p=1800, fixed total null variance

Curated candidate figures for the paper.

All figures use RFNN tanh features and `MC=500`.
Width: `p=1800, fixed total null variance`. Total null variance is held fixed by rescaling `sigma_noise^2 = E_null/(d_latent-d_intrinsic)`.

Folders:

- `exp2_main_gmm`: main GMM Exp-2-style `d_latent` sweep.
- `exp3_main_gmm`: main GMM Exp-3-style `d_intrinsic` sweep.
- `exp2_robust_gaussian_sigma`: Gaussian sigma-noise robustness for Exp 2.
- `exp2_robust_gaussian_sigscale`: Gaussian signal-scale robustness for Exp 2.
- `exp3_robust_gaussian_sigma`: Gaussian sigma-noise robustness for Exp 3.
- `exp3_robust_gaussian_sigscale`: Gaussian signal-scale robustness for Exp 3.

Color/index convention:

- signal: `eigs[:d_intrinsic]`
- noise-dim: `eigs[d_intrinsic:d_latent]`
- sample: `eigs[d_latent:d_latent+n]`
- rank-null: `eigs[d_latent+n:]`

Histogram bins span the full min/max range per panel.
