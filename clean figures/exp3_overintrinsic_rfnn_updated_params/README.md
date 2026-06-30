# Exp 3 over-intrinsic RFNN, updated parameters

Clean-figure version of the Exp 3 source-dimensionality sweep using the
updated four-bulk microscope parameters.

Fixed choices:

- `d_latent = 20`
- `d_intrinsic = [2, 5, 8, 12, 16, 20, 25, 30, 35, 40]`
- `n = 500`
- `sigma_signal = 1.0`
- `sigma_noise = 0.3`
- `center_scale = 1.5`
- `t = 0.01`
- `mc_samples = 500`
- `rank_null = 300`

Width folders:

- `p_n_plus_d_plus_r`: `p = d_latent + n + 300 = 820`
- `p64`: `p = 64*d_latent = 1280`
- `p_fixed`: `p = 1800`

For `d_intrinsic > d_latent`, data are generated in source dimension
`d_intrinsic` and compressed into the fixed observed latent space by a random
energy-preserving map.
