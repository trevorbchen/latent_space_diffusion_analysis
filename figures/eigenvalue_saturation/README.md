# Eigenvalue Saturation Tests

These plots demonstrate a critical technical detail: the RFNN's tanh activation saturates when data magnitudes are too large, destroying the eigenvalue structure.

## eigenvalue_hist_signal100.png

Eigenvalue histograms with scale=200, sigma_signal=100, sigma_noise=10 (top row) vs sigma_noise~0 (bottom row). Both rows look identical because tanh is fully saturated -- tanh(50) = tanh(5) = 1. Changing sigma_noise has no effect when the input is already clipped.

## eigenvalue_hist_signal1000_noise100.png

Even more extreme: sigma_signal=1000, sigma_noise=100. Everything is +/-1 in feature space, so eigenvalues collapse into one structureless blob. The four-bulk structure is completely invisible.

## Lesson

For RFNN eigenvalue analysis to reflect the data's anisotropic covariance structure (signal vs noise dimensions), the data must stay in the tanh linear regime. With W ~ N(0, 1/d), this requires data magnitudes on the order of sqrt(d_latent). We use scale=3, sigma_signal=1, sigma_noise=0.5, which keeps tanh inputs at ~0.4 on average -- well within the linear regime where the activation can distinguish different scales.
