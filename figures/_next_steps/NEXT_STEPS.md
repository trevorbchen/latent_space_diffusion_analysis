# Next Steps

Current state: We have empirical evidence of a four-bulk eigenvalue structure for the RFNN on synthetic anisotropic Gaussian mixtures, with a sharp cliff at index d_intrinsic and another at index d_latent. MLP empirics qualitatively match the prediction.

---

## 1. Clean up existing RFNN results

The RFNN sweeps are essentially done. We just need to:
- Plot tau_mem and tau_gen extracted from the existing curves (vs d_latent and d_intrinsic), with proper threshold-crossing definitions
- Make a single summary plot tying U-spectrum predictions to observed training timescales (tau_i = 1/lambda_i overlaid on actual learning curves)
- Verify the bulk boundaries match the noise-dim count exactly across all configs we ran

---

## 2. Theory (headline contribution)

The main finding of the paper is the four-bulk structure, so this is the most important section.

### 2a. Prove the four-bulk structure analytically
Bonnaire derives U's spectrum via the replica method assuming isotropic Sigma_data. Extend to block-diagonal Sigma_data = diag(sigma_signal² I_{d_intrinsic}, sigma_noise² I_{d_latent - d_intrinsic}). Predict the four-bulk structure analytically. Proof should show both:
- **Order**: signal eigenvalues > noise-dim eigenvalues > sample eigenvalues > rank-null eigenvalues (in magnitude)
- **Count**: exactly d_intrinsic signal modes, exactly (d_latent - d_intrinsic) noise-dim modes, ~n sample modes, ~(p - n) rank-null modes

I don't know how hard this math actually is — could be straightforward extension of Bonnaire's replica computation, could require new tools.

### 2b. tau_mem lower bound
From the spectrum, derive a bound showing tau_mem grows with the noise-dim count. Direct consequence of 2a.

### 2c. Derive bulk count formulas
Beyond just identifying that four bulks exist, derive analytic expressions for **how many eigenvalues are in each bulk** as a function of (d_intrinsic, d_latent, n, p) and possibly (sigma_signal, sigma_noise, t). The conjectured counts from our empirical data are d_intrinsic, d_latent - d_intrinsic, n, and p - n — but we should derive these from the theory in 2a rather than just observing them. Edge cases worth working out: what happens when n < d_latent? When p < n? When sigma_noise -> 0?

### 2d. Derive the U-shape theoretically (stretch goal)
Ideally we'd have a theory that predicts the MLP U-shape from first principles, not just empirical evidence. Possible angles:
- If the SAE hypothesis (3a-3c) is right, then for an MLP with ReLU/GELU, prove that gradient descent finds sparse features whose effective rank approaches d_intrinsic at high d_latent. This would give an analytic curve for score_error(d_latent) with a minimum at some d* > d_intrinsic.
- Use NTK / mean-field analysis of the MLP. The NTK eigenvalues inherit the four-bulk structure, but trainable representations can reweight them. Predict score_error from the NTK spectrum reweighted by the learned representation map.
- A simpler heuristic: model score_error as a sum of two competing terms — fitting cost (worsens with d_latent due to noise-dim modes) and approximation flexibility (improves with d_latent due to more features). The crossover point predicts the U-shape minimum.

I genuinely don't know if any of these lead to clean math. The double-descent literature has analytic U-shape curves (Belkin et al., Mei-Montanari) — worth checking whether their tools apply here.

---

## 3. Investigate the U-shape (interesting secondary finding)

The MLP score error shows a U-shape vs d_latent (best at small d, worst around d=15, recovery at large d). The RFNN doesn't show this. Most plausible explanation: **the MLP learns sparse / SAE-like representations that effectively project onto the d_intrinsic signal subspace at high d_latent**. The RFNN can't because its first layer is frozen.

Concrete ways to demonstrate the SAE hypothesis:

### 3a. Probe MLP representations for sparsity
- Compute activation density (fraction of non-near-zero activations) per neuron at each layer, across d_latent values. If sparsity emerges at high d_latent, that's evidence.
- Measure activation rank: SVD of the post-activation matrix on training data. Effective rank should approach d_intrinsic at high d_latent if SAE-like compression is happening.

### 3b. Project weights onto signal subspace
We have the ground-truth signal subspace (the first d_intrinsic columns of Q from data generation). Project the MLP's first-layer weight matrix onto this subspace and onto the null-space, separately. If the MLP is "ignoring" null-space, signal-subspace mass should grow with d_latent.

### 3c. Compare to explicit sparsity / regularization
Add L1 or top-k sparsity to the MLP and see if it reproduces or amplifies the U-shape recovery. If L1-regularized MLPs show even cleaner recovery at high d_latent, that supports the SAE hypothesis.

### 3d. Optional: rule out double descent
I don't have a clean way to prove it's double descent specifically. But we could vary parameter count at fixed d_latent and see if classic double descent appears at all in this setup. If it doesn't, the U-shape is unlikely to be double descent.

---

## 4. Real-data validation

### 4a. Get d_intrinsic estimates for real datasets
Need actual citations or measurements:
- **Pope et al. 2021 "The Intrinsic Dimensionality of Images and its Impact on Learning"** is the standard reference. Look up their reported values for MNIST, CIFAR, ImageNet — don't cite from memory.
- If their estimates aren't sufficient, run **Levina-Bickel MLE** or **TwoNN** ourselves.

### 4b. What to vary for the real-data sweep
We can't directly choose d_latent for real images — it's fixed by the dataset. Two options:

1. **Train a VAE at varying latent dim** and run diffusion on the VAE latent. This is the actual latent-diffusion (LDM) setting and matches practice. Sweep the bottleneck width over {d_intrinsic / 2, d_intrinsic, 2*d_intrinsic, 8*d_intrinsic, 32*d_intrinsic}, train diffusion on each, measure tau_mem. **This is probably the right choice** because d_latent here is a real architectural choice, not a synthetic projection.
2. **Random projection up**: take images in R^d_ambient and project to R^d_latent for varying d_latent (with small noise to avoid singular covariance). Cleaner control over d_latent but less practical relevance.

The VAE approach also gives a way to verify that our theory predicts real LDM behavior.

### 4c. Verify four-bulk on real data
Compute U (RFNN feature correlation) on real images or VAE latents at each d_latent. Check if the four-bulk structure appears. Critical sanity check that the synthetic story carries over.

### 4d. Train DDPM at varying d_latent
With 4a-4c in place, train a small DDPM at each d_latent setting and measure tau_gen (FID minimum) and tau_mem (memorization fraction first crosses 1%). Predict tau_mem grows with d_latent / d_intrinsic.

---

## 5. Lower-priority extras

- Activation function ablation (ReLU, GELU, sin)
- Different noise schedules (VP, VE, EDM)
- Different cluster counts k
- Curved manifolds (Swiss roll, MoG on sphere)

---

## Open questions

- Why does the MLP show a U-shape but RFNN doesn't? (3a-3c try to answer; SAE is the leading hypothesis)
- Does the four-bulk hold when the data manifold is non-linear?
- Is Bonnaire's replica method easy to extend to anisotropic Sigma, or does it require new tools?
- At very small n, does the noise-dim bulk dominate the sample bulk?
- What is the optimal d_latent for a given dataset, given the noise-dim buffer? Is there a practical recipe for LDMs?
