# Experiment 2 RFNN: d_latent Sweep

Sweep d_latent = {5, 8, 10, 15, 20, 30, 40} with d_intrinsic = 5 fixed. This tests the core hypothesis: does increasing d_latent change the eigenvalue structure of U in a way that delays memorization?

## exp2_rfnn_eigenvalues.png

Log-y line plots of the eigenvalue spectrum for each d_latent value. Red dashed line marks d_intrinsic=5, green dashed line marks d_latent.

Shows that as d_latent increases, the gap between the signal modes (top 5 eigenvalues) and the sample modes widens, filled by exactly (d_latent - d_intrinsic) noise-dimension eigenvalues. The cliff at the green line is razor-sharp in all cases.

## exp2_rfnn_eigenvalue_hist.png

Log-log histograms of the eigenvalue distribution for each d_latent. Shows the bulk structure:

- d_latent=5: two visible peaks (signal at ~10^1, sample at ~10^-2 to 10^0), no noise-dim peak
- d_latent=10-15: three peaks emerge (signal, noise-dim, sample), plus a growing near-zero bulk
- d_latent=30-40: four peaks visible (signal, noise-dim, sample, rank-null). The rank-null peak at ~10^-4 to 10^-1 grows because p = 64*d_latent increases while n=500 stays fixed, so more feature directions are unused.

Parameters: scale=3, sigma_signal=1, sigma_noise=0.5, n=500, p=64*d_latent, t=0.01, 300k steps.

## Raw data
`raw_data/` in this folder contains all training metrics (`metrics.jsonl`), eigenvalue arrays (`eigenvalues_pre.npy`, `eigenvalues_post.npy`), and configs (`config.json`) used to generate the plots above.
