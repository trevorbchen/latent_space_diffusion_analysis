# Exp 3 Gaussian signal-scale robustness

Fixed `d_latent=20`, sweep `d_intrinsic`.

Data: pure anisotropic Gaussian. Rows vary `sigma_signal` at
`sigma_noise=0.3`. RFNN uses tanh features and `MC=500`.
Width: `p=1800, fixed total null variance`. Total null variance is held fixed by rescaling `sigma_noise^2 = E_null/(d_latent-d_intrinsic)`.
