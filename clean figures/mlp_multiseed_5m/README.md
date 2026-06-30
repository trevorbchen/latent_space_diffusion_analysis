# MLP multi-seed 5M figures

Clean MLP section figures from the corrected 5M-step sweep.

Runs:

- `d_intrinsic=5`
- `d_latent in {5, 8, 10, 12, 15, 20, 25, 30, 35, 40}`
- `n=500`
- `sigma_noise=0.5`
- seeds `42, 43, 44, 45, 46`
- pure MLP, not RFNN
- fixed hidden width `256`
- source folder: `multiseed_runs/exp2_mlp_dlat_sn05_5m_bonnaire_hidden256`
- memorization uses the corrected Bonnaire/Somepalli generated-sample
  NN1/NN2 ratio with threshold `1/3`

Figures:

- `score_error_over_steps_ci`: mean score error over training with 95% CI.
- `score_error_per_dim_over_steps_ci`: mean score error divided by
  `d_latent` over training with 95% CI.
- `mem_ratio_over_steps_ci`: mean memorization ratio over training with 95% CI.
- `max_mem_ratio_bar_ci`: bar plot of each seed's maximum memorization ratio,
  averaged over seeds with 95% CI.
- `final_mem_ratio_line_ci`: final memorization ratio at 5M steps with 95% CI.

Tables:

- `mlp_seed_summary.csv`: one row per seed/run.
- `mlp_d_summary.csv`: one row per `d_latent`, averaged over seeds.
