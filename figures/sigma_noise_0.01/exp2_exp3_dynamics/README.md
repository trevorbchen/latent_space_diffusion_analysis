# RFNN Training Dynamics: Exp 2 and Exp 3

## exp2_exp3_rfnn_dynamics.png

Four panels showing how training dynamics match the eigenvalue predictions.

### Top left: Exp 2 Score Error vs Step (d_latent sweep)
Score error = ||s_model - s_true||^2 on held-out test data (2048 points). Measures how well the model has learned the true score function.

- d_latent=5 achieves the lowest score error (~0.29) -- no noise-dim buffer, most efficient learning
- d_latent=10-15 has the highest score error (~3.0) -- stuck learning noise-dim modes
- d_latent=20-40 shows a U-shape, score error improves somewhat -- possibly because p=64*d_latent gives more model capacity, or an SAE-like effect where excess dimensions asymptotically wash out

### Top right: Exp 2 Generalization Gap vs Step
Gen gap = test_loss - train_loss. Noisy but roughly similar across d_latent values. No clear memorization onset within 300k steps at these settings.

### Bottom left: Exp 3 Score Error vs Step (d_intrinsic sweep)
- d_intrinsic=20 (= d_latent) converges fastest and lowest (~0.37) -- all modes are signal
- d_intrinsic=2 converges to ~1.14 -- only 2 signal modes, 18 noise-dim modes to learn through
- Intermediate values form a smooth gradient between these extremes

### Bottom right: Exp 3 Generalization Gap vs Step
Similar pattern -- noisy, no clear memorization divergence in 300k steps.

Score error ordering in both experiments matches the eigenvalue predictions: fewer noise-dim buffer modes = faster convergence to the true score.

## Raw data
- d_latent sweep (Exp 2): `../exp2_rfnn/raw_data/`.
- d_intrinsic sweep (Exp 3): `../exp3_rfnn/raw_data/`.
