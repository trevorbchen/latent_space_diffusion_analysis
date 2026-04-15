# Experiment 3 RFNN: d_intrinsic Sweep

Sweep d_intrinsic = {2, 5, 8, 12, 16, 20} with d_latent = 20 fixed. This is the complement of Experiment 2: instead of adding null-space dimensions, we vary how many dimensions carry signal.

## exp3_rfnn_eigenvalues.png

Log-y line plots of the eigenvalue spectrum for each d_intrinsic value. Red line marks d_intrinsic, green line marks d_latent=20.

Key observations:
- d_intrinsic=2: 2 signal eigenvalues, 18 noise-dim eigenvalues between the lines, then sample drop at index 20
- d_intrinsic=5: 5 signal, 15 noise-dim
- d_intrinsic=12: 12 signal, 8 noise-dim, gap is shrinking
- d_intrinsic=20: red and green lines overlap, NO gap -- one continuous bulk from signal into sample. This is Bonnaire's original isotropic case.

As d_intrinsic approaches d_latent, the noise-dimension buffer shrinks to zero. The model goes directly from generalization to memorization with no delay.

## exp3_rfnn_eigenvalue_hist.png

Log-log histograms showing the bulk structure collapse:
- d_intrinsic=2: clear three-peak structure (signal isolated at far right, noise-dim bump in middle, sample bulk on left)
- d_intrinsic=20: single continuous bulk, no separation -- all dimensions carry signal equally

Parameters: scale=3, sigma_signal=1, sigma_noise=0.5, n=500, d_latent=20, p=1280, t=0.01, 300k steps.

## Raw data
`raw_data/` in this folder contains all training metrics (`metrics.jsonl`), eigenvalue arrays (`eigenvalues_pre.npy`, `eigenvalues_post.npy`), and configs (`config.json`) used to generate the plots above.
