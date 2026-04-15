# Four-Bulk Structure

## four_bulk_structure.png

Eigenvalue spectrum of U (feature correlation matrix) plotted as log-y line plots for d_latent = 5, 10, 20, 40 with d_intrinsic = 5 fixed.

The x-axis is eigenvalue index (sorted descending). Two vertical dashed lines mark:
- **Red (d_intrinsic=5)**: boundary between signal and noise-dim eigenvalues
- **Green (d_latent)**: boundary between noise-dim and sample eigenvalues

The key observation: there is a **sharp cliff** at both boundaries. Between the red and green lines sit exactly (d_latent - d_intrinsic) eigenvalues -- the noise-dimension bulk. As d_latent increases, this gap widens, inserting more eigenvalues that the model must learn before reaching sample-specific modes (memorization).

At d_latent=5 (no null-space), there is no green line and no gap -- just signal into sample. This is Bonnaire's original two-bulk case.

Parameters: scale=3, sigma_signal=1, sigma_noise=0.01, n=500, p=64*d_latent, t=0.01.

## Raw data
Uses eigenvalue arrays from `../exp2_rfnn/raw_data/` (configs d_latent = 5, 10, 20, 40 with d_intrinsic=5).
