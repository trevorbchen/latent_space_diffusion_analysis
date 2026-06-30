# Four-bulk participation-ratio diagnostics

This folder is not part of `clean figures`.

The script recomputes the RFNN feature covariance `U`, keeps eigenvectors,
and computes feature-coordinate participation ratio

`PR(v) = 1 / sum_j v_j^4`

for each eigenvector.  Rows are grouped using the same index convention as
the clean four-bulk histograms:

- signal: `eigs[:d_intrinsic]`
- noise-dim: `eigs[d_intrinsic:d_latent]`
- sample: `eigs[d_latent:d_latent+n]`
- rank-null: `eigs[d_latent+n:]`

Important caveat: these are PRs across RFNN feature coordinates, not sample
coordinates. They diagnose whether eigenmodes are feature-localized. A
separate sample-Gram/eigenvector diagnostic would be needed for literal
sample localization.
