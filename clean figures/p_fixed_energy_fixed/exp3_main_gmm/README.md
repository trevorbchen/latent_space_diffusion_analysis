# Exp 3 main GMM

Fixed `d_latent=20`, sweep `d_intrinsic`.

Data: anisotropic GMM, `sigma_noise=0.5`, `sigma_signal=1.0`,
`center_scale=3.0`. RFNN uses tanh features and `MC=500`.
Width: `p=1800, fixed total null variance`. Total null variance is held fixed by rescaling `sigma_noise^2 = E_null/(d_latent-d_intrinsic)`.
