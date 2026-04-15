# Experiment 3 MLP: d_intrinsic Sweep (sigma_noise=0.5)

Sweep d_intrinsic ∈ {2, 5, 8, 12, 16, 20} with d_latent=20, n=500, hidden=256.

## exp3_mlp_dynamics.png

Four panels: score error, test loss, memorization fraction, gen gap.

Key observations at sigma_noise=0.5:
- d_intrinsic=2 (most null-space buffer) has the lowest score error (~1.39) and zero memorization
- d_intrinsic=20 (= d_latent, no buffer) has the highest score error (~5.07) and highest memorization (mem=0.008)
- Memorization fraction monotonically increases with d_intrinsic

This is the complement of Exp 2: as d_intrinsic approaches d_latent, the noise-dim buffer in the U spectrum shrinks to zero, and the model can memorize freely.

## Raw data
`raw_data/` in this folder contains all training metrics (`metrics.jsonl`), eigenvalue arrays (`eigenvalues_pre.npy`, `eigenvalues_post.npy`), and configs (`config.json`) used to generate the plots above.
