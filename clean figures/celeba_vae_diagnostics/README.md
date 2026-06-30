# CelebA VAE Diagnostics

Clean VAE-side diagnostics for the real-data latent diffusion experiments.

Included latent dimensions: `10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200`.

What is here:

- `vae_training_summary.csv`: final and best validation metrics parsed from each VAE training log.
- `reconstruction_mem_summary.csv`: encode/decode reconstruction memorization on the same diverse 1k subset used for diffusion training.
- `vae_metrics_vs_d.png/pdf`: best validation loss, MSE, L1, and KL against latent dimension.
- `vae_training_curves.png/pdf`: validation reconstruction MSE and KL over VAE training epochs.
- `reconstruction_mem_vs_d.png/pdf`: reconstruction memorization, NN1/NN2 ratio, and pixel MSE against latent dimension.
- `vae_quality_and_reconstruction_mem.png/pdf`: compact comparison of VAE quality and reconstruction-memorization behavior.
- `vae_reconstruction_contact_sheet.png/pdf`: representative reconstruction grids copied from the VAE checkpoints.
- `recon_grids/`: per-dimension reconstruction grids and any extra encode/decode comparison grids.

Reconstruction memorization uses the Bonnaire/Somepalli nearest-neighbor ratio
test in pixel space on the 1k subset:
`NN1(reconstruction, training subset) / NN2(training image, training subset) < 1/3`.
The VAE was trained on the full available training split; this diagnostic asks
whether encoding and decoding the 1k diffusion subset returns images close
enough to their original subset examples to trigger the memorization criterion.

CelebA uses the modern ResNet VAE objective (beta warmup, capacity warmup, free bits, EMA, and early stopping).

Parsed `15` VAE training logs and `15` reconstruction-mem rows.
