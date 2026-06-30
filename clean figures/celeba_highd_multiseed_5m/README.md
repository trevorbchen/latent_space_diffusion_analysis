# CelebA high-d multiseed figures

Source: `results/celeba_diverse1k_bigmlp_sgd_lr001_m08_10k_5m` on ML6.

Sweep: CIFAR-style big MLP diffusion on the diverse CelebA 1k subset, seeds 42-46, `d_latent = 10, 20, ..., 100, 120, 140, 160, 180, 200`, 5M training steps.

Confidence intervals are 95% t-intervals over five seeds: `mean +/- t_0.975,4 * SEM`.

Generated files:
- `final_mem_mean_ci.*`, `final_mem_mean_no_ci.*`
- `final_fid_mean_ci.*`, `final_fid_mean_no_ci.*`
- `train_mem_over_steps_ci.*`
- `train_fid_over_steps_ci.*`
- `individual_mem_by_d_over_steps.*`
- `individual_fid_by_d_over_steps.*`
- `individual_by_d/mem/d*_mem_over_steps.*`
- `individual_by_d/fid/d*_fid_over_steps.*`
- `train_loss_over_steps_ci.*`
- `celeba_highd_final_summary.csv`
