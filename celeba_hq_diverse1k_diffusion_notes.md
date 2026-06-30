# CelebA-HQ Diverse-1k Latent Diffusion Sweep

## Subset

- Source data: `/overflow/data/cerberus/celeba_hq_256_partitioned_ws/train/celeba_hq_256-train-0000.tar`
- Preprocessing: grayscale, 32x32, normalized to `[-1, 1]`.
- Subset: 1000 real CelebA-HQ training images selected by farthest-first sampling in pixel L2 space.
- Subset indices: `diagnostics/celeba_diverse_1k_mean_plus_sd/subset_indices.json`
- Selection diagnostics: `diagnostics/celeba_diverse_1k_mean_plus_sd/summary.json`

Key selection stats:

```text
full dataset pairwise mean:       21.3491
full dataset pairwise std:         5.2587
target mean + 1 sd:               26.6078
selected pairwise mean:           26.4513
selected nearest-neighbor mean:   17.1545
```

## VAE Family

- VAE checkpoints: `vae_checkpoints/celeba_resnet_modernloss_d{d}/vae.pt`
- Latent dimensions used for diffusion: `10,20,30,40,50,60,70,80,90,100`
- Architecture: ResNet-style convolutional VAE with GroupNorm and SiLU.
- VAE objective: `L1 + 0.5*MSE + 0.05*KL_freebits`, beta warmup, EMA.

## Diffusion Sweep

- Training set: latent encodings of the farthest-first 1k subset.
- Score model: MLP, hidden width 256.
- Diffusion objective: OU score-matching loss.
- Steps: 5,000,000 per latent dimension.
- Batch size: 256.
- Optimizer: Adam, lr `1e-4`.
- Cheap metrics every 10,000 steps:
  - `train_loss_step`
  - full subset score-matching eval loss, stored as `train_loss` and `subset_score_loss`
- Expensive metrics every 100,000 steps, plus step 1 baseline:
  - FID against the same farthest-first 1k image subset
  - pixel-space NN-ratio memorization against the same subset

Important caveat: this is train-subset FID, not population FID. It measures whether generated samples match the deliberately spread-out 1k training subset.
