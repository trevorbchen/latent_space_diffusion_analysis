# Final Figure Data

This folder stores the paper-facing summaries from the corrected Bonnaire-ratio runs.

## Synthetic MLP, corrected 5-seed run

Source:

- `multiseed_runs/exp2_mlp_dlat_sn05_5m_bonnaire_hidden256`

Setup:

- `d_intrinsic=5`
- `d_latent in {5, 8, 10, 12, 15, 20, 25, 30, 35, 40}`
- `n=500`
- `sigma_noise=0.5`
- seeds `42, 43, 44, 45, 46`
- pure MLP, fixed hidden width `256`
- 5M training steps per run
- corrected Bonnaire/Somepalli memorization ratio: generated sample NN1 distance divided by generated sample NN2 distance, threshold `1/3`

Files:

- `synthetic_mlp_seed_summary_bonnaire.csv`: one row per `d_latent` and seed.
- `synthetic_mlp_d_summary_bonnaire.csv`: 5-seed mean/CI summaries by `d_latent`.
- `synthetic_mlp_tau_summary_bonnaire.csv`: tau_gen and tau_mem summaries by `d_latent`.

Clean figures regenerated from this data:

- `../mlp_multiseed_5m`
- `../multiseed_tau`

## CelebA big-MLP SGD diagnostic

Source:

- ML6 `results/celeba_diverse1k_bigmlp_sgd_10k_5m`

Setup:

- CelebA-HQ diverse 1k subset
- ResNet modern-loss VAE latents
- 5-layer MLP, hidden width `1024`
- SGD optimizer with momentum `0.95`, learning rate `0.01`
- pixel-space memorization with corrected Bonnaire NN1/NN2 ratio
- 10k generated samples for memorization and FID

File:

- `celeba_bigmlp_sgd_mem_fid.csv`

Notes:

- `d=10,20,30,40` completed and produced valid mem/FID.
- `d=50,60,70,80,90,100` failed early from SGD instability: training loss became NaN by 10k steps, so final mem/FID is invalid for those runs.
