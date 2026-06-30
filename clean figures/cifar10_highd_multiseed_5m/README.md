# CIFAR-10 high-d multiseed figures

Source: `results/cifar10_beta005_diverse1k_bigmlp_sgd_lr001_m08_10k_5m` on ML6.

Sweep: CIFAR-10 diverse-1k VAE-latent MLP diffusion, 5M training steps,
`d_latent = 20, 40, ..., 260`.

The clean selected set intentionally excludes all gradient-clipped diagnostic
runs. Replacement seeds are used for unstable or contaminated runs:

- d220: seeds 42, 43, 44, 45, 47
- d240: seeds 42, 43, 45, 46, 47
- d260: seeds 43, 47, 48, 49, 50

All selected runs are unclipped and use the same optimizer settings:
SGD, lr=0.001, momentum=0.80.

If the final metrics row had a nonfinite FID but a `fid_retry.json` with
successful finite samples existed, the final FID summary uses the mean retry
FID. Training-trajectory plots still drop nonfinite evaluation rows.

Confidence intervals are 95% t-intervals over five seeds:
`mean +/- t_0.975,4 * SEM`.

Generated files:

- `final_mem_mean_ci.*`, `final_mem_mean_no_ci.*`
- `final_fid_mean_ci.*`, `final_fid_mean_no_ci.*`
- `final_loss_mean_ci.*`, `final_loss_mean_no_ci.*`
- `train_mem_over_steps_ci.*`
- `train_fid_over_steps_ci.*`
- `individual_mem_by_d_over_steps.*`
- `individual_fid_by_d_over_steps.*`
- `individual_by_d/mem/d*_mem_over_steps.*`
- `individual_by_d/fid/d*_fid_over_steps.*`
- `train_loss_over_steps_ci.*`
- `train_score_loss_per_dim_over_steps_ci.*`
- `cifar10_highd_final_summary.csv`
- `cifar10_selected_runs.csv`
- `cifar10_highd_all_metrics.csv`
