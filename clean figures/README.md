# Clean figures

Curated candidate figures for the paper.

There are three complete versions:

- `p_n_plus_d_plus_r`: controlled-width figures with `p=d_latent+n+300`.
- `p64`: fixed-ratio figures with `p=64*d_latent`.
- `p_fixed_energy_fixed`: fixed-capacity/energy ablation with `p=1800`
  and `sigma_noise^2 = E_null/(d_latent-d_intrinsic)`.

Each width folder contains the same six experiment families:

- `exp2_main_gmm`
- `exp3_main_gmm`
- `exp2_robust_gaussian_sigma`
- `exp2_robust_gaussian_sigscale`
- `exp3_robust_gaussian_sigma`
- `exp3_robust_gaussian_sigscale`

Each run stores the full dense eigenspectrum in `eigenvalues.npy`.

Additional real-data diffusion figures:

- `celeba_highd_multiseed_5m`: CelebA diverse-1k VAE-latent MLP sweep,
  seeds 42-46, `d_latent = 10, 20, ..., 100, 120, 140, 160, 180, 200`, 5M steps. Includes
  final memorization/FID vs latent dimension with and without 95% confidence
  intervals, training-step trajectories for memorization, FID, and loss, and
  per-dimension individual-seed trajectory panels.
- `celeba_spectral_mem_predictor`: data-only spectral memorization predictor
  for the full CelebA sweep, using existing VAE encoder means and empirical
  KNN memorization curves. Includes multi-threshold predicted-vs-observed
  onset plots and spectral feature diagnostics.
- `celeba_vae_diagnostics`: VAE-side diagnostics for the CelebA sweep. Includes
  encode/decode reconstruction grids, VAE training metrics, and pixel-space
  reconstruction memorization on the same diverse 1k subset.
- `cifar10_highd_multiseed_5m`: CIFAR-10 diverse-1k VAE-latent MLP sweep,
  unclipped selected seeds, `d_latent = 20, 40, ..., 260`, 5M steps. Includes
  final memorization/FID/loss vs latent dimension with and without confidence
  intervals, training-step trajectories, and per-dimension individual-seed
  trajectory panels.
- `cifar10_spectral_mem_predictor`: data-only spectral memorization predictor
  for the CIFAR-10 sweep, using existing CIFAR-10 VAE encoder means and the
  unclipped selected empirical memorization curves. Includes the same
  multi-threshold overlay and spectral diagnostic views as the CelebA version.
- `cifar10_vae_diagnostics`: VAE-side diagnostics for the CIFAR-10 sweep.
  Includes encode/decode reconstruction grids, VAE training metrics, and
  pixel-space reconstruction memorization on the same diverse 1k subset.
- `exp3_overintrinsic_rfnn_updated_params`: RFNN Exp-3-style source
  dimensionality sweep with the updated cleaner microscope parameters
  (`sigma_noise=0.3`, `center_scale=1.5`, `mc_samples=500`). It fixes
  observed `d_latent=20`, varies source `d_intrinsic` through over-intrinsic
  values, and includes the same three width controls.

## CIFAR gradient-clipping note

Some exploratory CIFAR high-dimensional runs used gradient clipping
(`grad_clip=1.0`) to prevent NaNs in unstable seeds. Those runs are diagnostic
only and should not be mixed into clean paper figures: clipping substantially
changes the training trajectory and led to essentially no measured
memorization in the affected runs.

In particular, the clipped CIFAR folders `d260_clip1` and `d220_clip1` should
be treated as evidence that clipping suppresses memorization, not as clean
replacements for the original sweep. The clean CIFAR figures should instead
use unclipped replacement seeds with the same optimizer, learning rate, and
momentum as the rest of the sweep.
