# CIFAR-10 Spectral Memorization Predictor

Data-only predictor for CIFAR-10 memorization onset. The prediction uses
existing CIFAR-10 VAE encoder means on the same diverse 1k training subset and
does not inspect trained diffusion-model weights.

Primary statistic:

`M_t(d) = exp(-2t) Z^T Z / n + (1 - exp(-2t)) I`, with `t = 0.1`.

Primary spectral pressure:

`P_d(s) = mean_i (1 - exp(-kappa * lambda_i(d) * s))^2`.

Calibration:

- one global `kappa = 1e-06`
- one monotone threshold map over memorization levels
- no per-dimension fitting

The empirical curves use only unclipped selected runs. Clipped diagnostic runs
are excluded because clipping strongly suppressed measured memorization.

Generated files:

- `mem_curves_with_predicted_thresholds.*`
- `mem_curve_small_multiples.*`
- `tau_pred_vs_obs_by_threshold.*`
- `tau_vs_d_by_threshold.*`
- `spectral_pressure_curves.*`
- `spectral_pressure_curves_weighted.*`
- `spectral_features_vs_d.*`
- `cifar10_spectral_tau_table.csv`
- `cifar10_spectral_tau_table_weighted.csv`
- `calibration.json`
- `selected_runs.json`

Plot interpretation note: the red x markers are iso-memorization threshold-hit
predictions. Their y-values are fixed thresholds by construction. The dashed
red curve maps spectral pressure through the fitted threshold map and is the
more natural visual surrogate for predicted memorization over time. Thresholds
that are globally unobserved in the empirical runs are not shown as calibrated
theoretical overlays.
