# Experimental Methods

This document describes every implementation choice needed to reproduce the experiments in this directory. The two main scripts are in [code/experiment_v2.py](code/experiment_v2.py) (MLP score model) and [code/experiment_v2_rfnn.py](code/experiment_v2_rfnn.py) (RFNN score model).

## Data Generation

Anisotropic Gaussian mixture in `R^d_latent` with intrinsic dimension `d_intrinsic`.

1. **Cluster centers**: k=10 centers in `R^d_intrinsic`, constructed to be orthogonal and equal-magnitude. For k <= d_intrinsic, take the first k columns of QR of a random Gaussian matrix and rescale each to norm `scale`. For k > d_intrinsic, the first d_intrinsic centers are orthogonal at norm `scale`, the remaining k - d_intrinsic are random unit vectors rescaled to `scale`.
2. **Sample assignments**: each of n training points is assigned uniformly to one of the k clusters.
3. **Signal dimensions** (first d_intrinsic): cluster center + Gaussian noise with std `sigma_signal`.
4. **Null-space dimensions** (remaining d_latent - d_intrinsic): Gaussian noise with std `sigma_noise`.
5. **Random rotation**: a random orthogonal matrix Q (from QR of a random Gaussian d_latent x d_latent matrix) is applied to the data and to the cluster centers, removing axis alignment.

This data has Sigma_data = Q diag(sigma_signal^2, ..., sigma_noise^2, ...) Q^T.

The fixed test set is generated the same way with seed=9999, n=2048, using the same Q.

### Default data parameters

| Parameter | Value | Reason |
|---|---|---|
| k | 10 | clusters |
| sigma_signal | 1.0 | within-cluster std in signal dims |
| sigma_noise | 0.5 (or 0.01) | std in null-space dims |
| scale | 3.0 | cluster center magnitude |
| n | 500 | training samples |

**Critical: tanh saturation**. The RFNN computes features `tanh(Wx/sqrt(d))` with W ~ N(0, 1/d). For the eigenvalue analysis to reflect the data's anisotropic structure, the inputs to tanh must stay in the linear regime, roughly `|Wx/sqrt(d)| < 1`. With these defaults, mean tanh input is ~0.4. Larger data scales (e.g. scale=200, sigma_signal=100) cause tanh saturation, destroying the eigenvalue structure -- the eigenvalues become invariant to sigma_signal/sigma_noise.

## Diffusion Process (OU)

Forward process: `dx = -x dt + sqrt(2) dW`, giving `x_t = e^{-t} x_0 + sqrt(delta_t) eps` where `delta_t = 1 - e^{-2t}` and `eps ~ N(0, I)`.

## True Score (Analytic)

For Gaussian mixture P_0 = (1/k) sum_j N(mu_j, Sigma_data), the noised distribution is P_t(x) = (1/k) sum_j N(e^{-t} mu_j, Sigma_t) where Sigma_t = delta_t I + e^{-2t} Sigma_data.

The score is `nabla log P_t(x) = sum_j w_j(x,t) Sigma_t^{-1} (e^{-t} mu_j - x)` with softmax mixture weights w_j.

We **precompute** Sigma_t^{-1} and log det(Sigma_t) by diagonalizing in the un-rotated frame. This requires passing Q through the score function -- a bug in early experiments was assuming Sigma_t = sigma_t I (isotropic) which gave incorrect score gradients in the null-space dimensions.

## Track 1: MLP Score Model

3-layer GELU MLP with sinusoidal time embedding.

- Input: `[x, sin(t * freqs), cos(t * freqs)]` with n_freq=32 log-spaced frequencies in [1, 1000]
- Hidden width: `8 * d_latent` (scaled with input size to avoid capacity confound). Earlier runs with fixed hidden=256 are deprecated -- they showed misleading "high d_latent generalizes best" results because the model was relatively underparameterized at high d_latent and couldn't memorize.
- Output: d_latent

**Training**:
- Random `t ~ Uniform(t_min=0.01, t_max=3.0)` per minibatch (standard diffusion training)
- Loss: `||sqrt(delta_t) * s_theta(x_t, t) + eps||^2`, summed over dims, mean over batch
- Optimizer: Adam, lr=1e-4, batch_size=min(n, 256)
- Steps: 300,000

**Sample generation** (for memorization metric): Euler-Maruyama reverse SDE with 500 steps from t_max=3.0 to t_min=0.01.

## Track 2: RFNN Score Model

`s_A(x) = (1/sqrt(p)) A * tanh(W x / sqrt(d))`

- W: frozen, shape (p, d_latent), entries i.i.d. N(0, 1/d_latent)
- A: learnable, shape (d_latent, p), zero-initialized (matches Bonnaire convention)
- p = 64 * d_latent (Bonnaire's psi_p = 64)

**Training**:
- **Fixed t = 0.01** (required for theoretical tractability -- U(t) has solvable structure at fixed t)
- Loss: `||sqrt(delta_t) * s_A(x_t) + eps||^2 / (d_latent * n)` (Bonnaire normalization)
- Optimizer: SGD, no momentum
- Learning rate: `lr = 0.01 * d_latent / delta_t` (Bonnaire scaling, ~50 at t=0.01)
- Full-batch
- Steps: 300,000

This setup makes the loss quadratic in A, so gradient descent is a linear ODE whose dynamics are determined entirely by the eigenvalues of the feature correlation matrix U.

## Feature Correlation Matrix U

The key object for the eigenvalue analysis:

```
U = (1/n) sum_nu E_eps[ phi(x_t^nu(eps)) phi(x_t^nu(eps))^T ]
```

where `phi(x) = tanh(Wx)` and the expectation is over the diffusion noise eps. Computed before training:

1. For each of n_noise_samples=50 i.i.d. noise draws:
   - Form noised data x_t = e^{-t} x_train + sqrt(delta_t) * eps
   - Compute features Phi = tanh(x_t W^T), shape (n, p)
   - Accumulate Phi^T Phi / n
2. Average over the 50 noise draws

Then eigendecompose U with `numpy.linalg.eigvalsh`. Each eigenvalue lambda_i corresponds to a learning timescale tau_i = 1/lambda_i (large eigenvalues learned first).

## Evaluation Metrics

- **train_loss**: score matching loss on training data, divided by d_latent. Computed on x_t with fresh noise each eval step (could be made deterministic, but train loss noise is small relative to test signal).
- **test_loss**: same loss on the 2048 fixed test samples with fixed noise (precomputed once before training, NOT regenerated per eval).
- **gen_gap**: test_loss - train_loss (overfitting indicator).
- **score_error** (E_score): `||s_theta(x_t_test) - s_true(x_t_test)||^2 / d_latent` evaluated at t_eval = 0.1 (MLP) or t_fixed = 0.01 (RFNN). Primary indicator of generalization quality (tau_gen).
- **memorization_fraction** (MLP only): generate 5000 samples via reverse SDE, compute NN ratio = `d(gen, NN1_train) / d(NN1_train, NN2_train)`, count fraction with NN ratio < 1/3 (Bonnaire's threshold).
- **mean_nn_ratio** (MLP only): mean of the NN ratios.

For a consistent eval, both `test_data_fixed` and `test_noise_fixed` are precomputed once at the start of training -- this avoids noise in the eval curves that would otherwise come from resampling test noise each step.

## Experiment 2: d_latent Sweep

Vary d_latent ∈ {5, 8, 10, 15, 20, 30, 40, 50, 100, 150, 200} with d_intrinsic=5, n=500. Tests how the noise-dimension buffer in U affects training dynamics.

## Experiment 3: d_intrinsic Sweep

Vary d_intrinsic ∈ {2, 5, 8, 12, 16, 20} with d_latent=20, n=500. The complementary experiment: how does shrinking the noise-dim buffer affect things?

## Sigma_noise Variants

Each experiment is run at sigma_noise ∈ {0.5, 0.01}. The 0.5 setting gives the cleanest four-bulk separation in the U spectrum because the noise-dim eigenvalues sit clearly between signal and sample bulks. The 0.01 setting matches the original experiment_plan_v2.md but produces sharper score gradients in the null-space, leading to higher absolute score_error values (the true score blows up as sigma_noise -> 0).

## Compute Tracking

Each metrics row logs: `step`, `wall_time_sec`, `n_params`, `total_flops`, `samples_seen`. The flops estimate is `6 * n_params * batch_size * step` (rough: 2x forward + 4x backward). Used to verify that effects across d_latent are not confounded by compute differences.

## Reproducibility

- Single seed (42) for all reported experiments. Multi-seed runs would be needed for error bars on tau_gen / tau_mem.
- All randomness uses `numpy.random.default_rng(seed)` for data generation and `torch.manual_seed(seed)` for model initialization.
- Q matrix is generated with the same seed as the data and saved with the run.

## Hardware

- RFNN runs: CPU (numpy linalg dominates) or single-chip TPU v6e (3x speedup mostly from PyTorch ops, eigendecomp still on CPU).
- MLP runs: TPU v6e for sigma_noise=0.5 (~5 min per config at hidden=8*d_latent, d_latent <= 50), CPU fallback for sigma_noise=0.01.
- All experiments fit on a single v2-8 or v6e-4 TPU. Multi-host TPU was not used -- a known issue is that `xm.optimizer_step()` hangs when only one chip is being used on a multi-chip device. Use `optimizer.step()` + `xm.mark_step()` instead.

## Known Issues / Pitfalls

1. **tanh saturation in RFNN**: data magnitudes that look reasonable for the MLP cause tanh to clip to +/-1, destroying eigenvalue structure. Keep `scale / sqrt(d_latent) < 2`.
2. **Capacity confound for MLP across d_latent**: a fixed hidden width gives the false impression that high d_latent generalizes better. Always scale hidden with d_latent (we use 8x).
3. **Anisotropic Sigma_t**: the true score formula must use the actual data covariance. Assuming Sigma_data = I gives wildly wrong scores when sigma_noise != sigma_signal.
4. **Test noise resampling**: regenerating test noise every eval step adds spurious oscillation to the eval curves. Fix the test noise once.
5. **Append vs overwrite for metrics file**: opening metrics.jsonl with "a" causes duplicate rows if a config is rerun. Always use "w".
