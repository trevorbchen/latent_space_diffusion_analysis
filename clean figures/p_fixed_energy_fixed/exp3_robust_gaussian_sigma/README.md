# Exp 3 Gaussian sigma-noise robustness

Fixed `d_latent=20`, sweep `d_intrinsic`.

Data: pure anisotropic Gaussian. Rows vary `sigma_noise`.
RFNN uses tanh features and `MC=500`.
Width: `p=1800, fixed total null variance`. Total null variance is held fixed by rescaling `sigma_noise^2 = E_null/(d_latent-d_intrinsic)`.
