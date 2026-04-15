# Project Overview: Latent Dimensionality and the Generalization-Memorization Transition

## Research question

How does excess latent dimensionality (d_latent >> d_intrinsic) affect when diffusion models memorize their training data?

## Background

Bonnaire et al. (NeurIPS 2025 Best Paper) showed that diffusion model training has two timescales:
- **tau_gen**: when the model learns the population distribution (generalization)
- **tau_mem**: when the model starts memorizing individual training points

They proved this for a Random Feature Neural Network (RFNN) where the loss is quadratic in the learnable weights, so gradient descent follows a linear ODE whose dynamics are determined entirely by the eigenvalues of a feature correlation matrix U. Large eigenvalues are learned first, small ones later. Bonnaire's spectrum splits into **two bulks**:
- Population modes (large eigenvalues, learned first → generalization)
- Sample-specific modes (small eigenvalues, learned later → memorization)

Their analysis assumes isotropic data (all dimensions carry signal equally). We extend it to anisotropic data — a low-dimensional signal manifold embedded in a higher-dimensional ambient space, which is the realistic setting for most generative models.

---

## Finding 1: Four-bulk eigenvalue structure (main contribution)

When d_intrinsic < d_latent, the U spectrum splits into **four** bulks instead of two:

1. **Signal bulk** (~d_intrinsic eigenvalues, largest magnitude). Modes capturing the population/cluster structure. Bonnaire's rho_2.
2. **Noise-dimension bulk** (~(d_latent - d_intrinsic) eigenvalues, intermediate magnitude). Modes capturing variance in the null-space dimensions. **New — does not appear in Bonnaire's isotropic analysis.**
3. **Sample bulk** (~n eigenvalues, small magnitude). Modes encoding individual training points. Bonnaire's rho_1.
4. **Rank-null bulk** (~(p - n) eigenvalues, near-zero). Trivially empty feature directions from RFNN overparameterization.

The boundaries between bulks land at exactly eigenvalue indices d_intrinsic and d_latent — **razor-sharp cliffs**. Verified empirically across many configs (see [sigma_noise_0.5/four_bulk/](../sigma_noise_0.5/four_bulk/) and the RFNN sweeps).

### Mechanism: noise-dim modes act as a buffer

Since the RFNN learns modes in order of decreasing eigenvalue (tau_i = 1/lambda_i):

1. **Signal** is learned first → generalization (timing roughly independent of d_latent)
2. **Noise-dim** is learned second → no effect on sample quality, just consumes training time
3. **Sample** is learned third → memorization

Increasing d_latent adds more noise-dim modes between signal and sample, **mechanistically delaying memorization**. The buffer width is exactly (d_latent - d_intrinsic).

This explains why higher d_latent increases tau_mem without substantially affecting tau_gen, which Bonnaire's two-bulk picture cannot.

---

## Finding 2: U-shape in MLP score error (secondary, partly understood)

When we run the same sweep with an MLP score network (instead of the frozen-feature RFNN), the score error vs d_latent shows a **U-shape**, not the monotone behavior the RFNN predicts:
- Best at small d_latent (no noise-dim buffer)
- Worst around d_latent ~ 15
- Recovery at large d_latent (down to even lower error than small d in the unscaled MLP, monotone increase in the scaled MLP)

The RFNN never recovers — at large d_latent its score error keeps climbing because the buffer just keeps growing.

**Most plausible explanation**: the MLP learns sparse / SAE-like representations that effectively project onto the d_intrinsic signal subspace at high d_latent, "ignoring" the null-space dimensions. The frozen RFNN can't do this because its first layer is fixed. We have not proved this — it's the leading hypothesis but other explanations (double descent, mixed capacity effects) aren't ruled out.

---

## Why these findings matter

- **For theory**: extends Bonnaire's two-bulk picture to anisotropic data, which is what real datasets actually look like. The four-bulk structure is a clean prediction from data geometry alone.
- **For practice**: predicts that latent diffusion models with d_latent >> d_intrinsic get "free" memorization protection from the noise-dim buffer. This may explain why large LDMs memorize less than naive scaling laws would suggest, and gives a practical knob (d_latent) to tune memorization risk.
- **For mechanistic interpretability**: the order of feature learning is interpretable — first signal, then noise, then memorization — and the boundaries are exactly predictable from the data dimensions.

---

## Open questions

- Can we prove the four-bulk structure analytically (extending Bonnaire's replica computation to anisotropic Sigma)?
- Can we derive the U-shape theoretically? Is it really SAE-style sparse compression?
- Does the four-bulk hold on real image data (after measuring d_intrinsic)?
- What is the optimal d_latent for memorization protection in practical LDMs?

See [_next_steps/NEXT_STEPS.md](../_next_steps/NEXT_STEPS.md) for the concrete plan.

---

## Where to find things

- [METHODS.md](../METHODS.md) — full experimental methods and design choices for reproduction
- [code/](../code/) — experiment scripts
- [sigma_noise_0.5/](../sigma_noise_0.5/) and [sigma_noise_0.01/](../sigma_noise_0.01/) — figures + raw data, split by noise level
- [eigenvalue_saturation/](../eigenvalue_saturation/) — technical pitfall (tanh saturation in RFNN)
- [_next_steps/](../_next_steps/) — the to-do list
