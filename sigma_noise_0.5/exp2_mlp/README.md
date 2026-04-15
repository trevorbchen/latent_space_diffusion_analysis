# Experiment 2 MLP: d_latent Sweep (sigma_noise=0.5)

Sweep d_latent ∈ {5, 8, 10, 15, 20, 30, 40, 50, 100, 150, 200} with d_intrinsic=5, n=500. MLP hidden = 8 * d_latent (capacity scaled with input size to avoid the fixed-hidden confound).

## exp2_mlp_dynamics.png

Four panels: score error, test loss, memorization fraction, gen gap, all vs training step.

Key observations at sigma_noise=0.5:
- d_latent=5 has the lowest score error (~0.14) and the only meaningful memorization (mem ≈ 0.006)
- Score error grows roughly monotonically with d_latent up to d=200 (3.21)
- Memorization vanishes for 8 ≤ d_latent ≤ 100, then reappears slightly at d=150 (mem=0.002) and d=200 (mem=0.014)
- Gen gap also grows with d_latent

This pattern is consistent with the four-bulk eigenvalue mechanism: more null-space dimensions = more noise-dim modes the MLP must traverse before reaching memorization.

## Raw data
`raw_data/` in this folder contains all training metrics (`metrics.jsonl`), eigenvalue arrays (`eigenvalues_pre.npy`, `eigenvalues_post.npy`), and configs (`config.json`) used to generate the plots above.
