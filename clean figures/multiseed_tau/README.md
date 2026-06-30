# Multi-seed tau figures

Timescale figures from the corrected 5-seed synthetic MLP sweep.

Source:

- `multiseed_runs/exp2_mlp_dlat_sn05_5m_bonnaire_hidden256`

Setup:

- `d_intrinsic=5`
- `d_latent in {5, 8, 10, 12, 15, 20, 25, 30, 35, 40}`
- `n=500`
- `sigma_noise=0.5`
- seeds `42, 43, 44, 45, 46`
- fixed hidden width `256`
- corrected Bonnaire/Somepalli generated-sample NN1/NN2 memorization ratio

Files:

- `tau_timescales_5seed.png/pdf`: tau_gen and tau_mem with 95% CI.
- `tau_summary.csv/json`: aggregated values used by the plot.
