# CelebA Spectral Memorization Predictor

Data-only predictor for CelebA memorization onset. The prediction uses existing CelebA VAE encoder means on the same diverse 1k training subset and does not inspect trained diffusion-model weights.

Primary statistic:

`M_t(d) = exp(-2t) Z^T Z / n + (1 - exp(-2t)) I`, with `t = 0.1`.

Primary spectral pressure:

`P_d(s) = mean_i (1 - exp(-kappa * lambda_i(d) * s))^2`.

Calibration:

- one global `kappa = 1.35936e-06`
- one monotone threshold map over memorization levels
- no per-dimension fitting

Diagnostic weighted version uses de-noised eigenvalue excess weights `max(lambda_i - beta_floor, 0)`.

Important caveat: this is a data-spectrum surrogate inspired by the RFNN mode-timescale theory, not a literal RFNN theorem application to the trainable MLP.

Generated files:

- `mem_curves_with_predicted_thresholds.*`
- `mem_curve_small_multiples.*`
- `tau_pred_vs_obs_by_threshold.*`
- `tau_vs_d_by_threshold.*`
- `spectral_pressure_curves.*`
- `spectral_pressure_curves_weighted.*`
- `spectral_features_vs_d.*`
- `celeba_spectral_tau_table.csv`
- `celeba_spectral_tau_table_weighted.csv`
- `calibration.json`
