# Latent Dimensionality and the Generalization–Memorization Transition in Diffusion Models

## Research Question

How does the ratio d_latent / d_intrinsic affect the generalization window [τ_gen, τ_mem] in diffusion models? Specifically, does excess latent dimensionality (d_latent >> d_intrinsic) make memorization easier or faster?

## Background: Bonnaire et al.

Bonnaire et al. ("Why Diffusion Models Don't Memorize") identify two timescales in diffusion model training:

- **τ_gen**: when the model begins generating high-quality samples (learns the population score)
- **τ_mem**: when the model begins memorizing training data (learns the empirical score)

They find τ_gen is independent of dataset size n, while τ_mem grows linearly with n. This creates a generalization window [τ_gen, τ_mem] that widens with more data.

**How they measure these (U-Net experiments on CelebA):**
- τ_gen: training step where FID (computed on 10K generated samples vs 10K test samples) reaches its minimum
- τ_mem: training step where the memorization fraction f_mem first departs from zero, where a sample is "memorized" if its NN ratio d(gen, NN1_train) / d(NN1_train, NN2_train) < 1/3

**The RFNN theoretical framework:**

They analyze a Random Feature Neural Network (RFNN) score model:

    s_A(x) = (1/√p) A σ(Wx/√d)

W (p × d) is a frozen random Gaussian matrix. A (d × p) is the only learnable parameter. σ is tanh applied elementwise. Because W is frozen, the output is linear in A, which means:

1. The score matching loss at a fixed diffusion time t is **quadratic in A**
2. Gradient descent on A is a **linear ODE**: Ȧ(τ) = -A(τ) · (Δ_t/ψ_p) U + constant
3. The solution decomposes exactly into **eigenmodes of U**

where U is the p × p feature correlation matrix:

    U = (1/n) Σ_ν E_ξ [ σ(Wx^ν_t/√d) σ(Wx^ν_t/√d)^T ]

U is a fixed matrix — it is computed from the frozen features and the training data, before any training happens. Each eigenvalue λ_i of U maps to a training timescale τ_i = ψ_p / (Δ_t λ_i). Large eigenvalue → fast learning. Small eigenvalue → slow learning.

Using random matrix theory (replica method), Bonnaire derives the spectrum of U analytically and finds it splits into **two bulks** (in the overparameterized regime p >> n >> 1):

- **ρ₂** (large eigenvalues, scale ~ ψ_p): "population" modes that depend only on the data covariance Σ, not on specific training samples. Learning these gives you the population score → **generalization**. Timescale: τ_gen ~ 1/Δ_t. Independent of n.

- **ρ₁** (small eigenvalues, scale ~ ψ_p/ψ_n): "sample-specific" modes that encode the difference between the empirical score and the population score. Learning these gives you the empirical score → **memorization**. Timescale: τ_mem ~ ψ_n/Δ_t = n/(d·Δ_t). Grows linearly with n.

The gap between the two bulks in eigenvalue space directly translates to the gap between τ_gen and τ_mem in time. This is the mechanistic explanation for the generalization window.

**Key detail: fixed t.** The RFNN analysis is done at a fixed diffusion time t. This is necessary because U(t) has a specific structure at each t that maps onto solvable forms in random matrix theory. Integrating over t would mix different U(t) matrices, and while the dynamics would still be linear in A, the analytic eigenvalue decomposition via the replica method wouldn't factor as cleanly. Fixed t is what makes the theory tractable.

**What Bonnaire does NOT study:** The case where data has lower intrinsic dimensionality than the ambient space. All their analysis assumes isotropic data (Σ = I or Σ with eigenvalues all on the same scale). Our contribution is adding d_latent vs d_intrinsic as a new axis.

---

## Our Extension

When d_intrinsic < d_latent, the data covariance Σ has two scales: ~1 in d_intrinsic signal dimensions, ~σ²_noise ≈ 0 in the remaining (d_latent - d_intrinsic) null dimensions. This means U(t) inherits an anisotropic structure from Σ. Bonnaire's theorems take Σ as input (it appears explicitly in the replica equations 17-18), so the spectrum of U will change.

**Hypothesis:** As d_latent/d_intrinsic grows (more null-space dimensions), the generalization window [τ_gen, τ_mem] shrinks. The mechanistic explanation would be that null-space dimensions introduce additional modes in U's spectrum that correspond to sample-specific (memorization) structure, because the model has degrees of freedom in directions where the population distribution carries essentially no signal.

**What the spectrum of U might look like** with d_intrinsic < d_latent is unknown — possibly the ρ₂ bulk splits (population modes in signal vs null subspace), or ρ₁ gets more weight, or the gap between bulks changes. This is what we'll find out.

---

## Experimental Design

### Two Tracks

**Track 1 — Small MLP Diffusion Model (empirical, do first):**

A realistic nonlinear score network. Trains with random t per batch (standard diffusion training). Can generate samples via reverse SDE. Use for:
- Measuring τ_gen (via score error against known true score) and τ_mem (via NN ratio on generated samples)
- Building the phase diagram in (d_latent/d_intrinsic, n) space
- Establishing that the effect exists empirically

**Track 2 — RFNN (mechanistic, do second):**

Bonnaire's model. Trains at fixed t (required for eigenvalue theory). Use for:
- Computing U(t) explicitly and examining its spectrum across parameter sweeps
- Confirming that eigenvalue predictions match observed training dynamics
- Providing a mechanistic explanation for whatever Track 1 finds

The tracks answer complementary questions. Track 1: "does excess latent dimensionality affect memorization?" Track 2: "if so, what's the spectral mechanism?"

**Important caveat on comparing tracks:** The RFNN has frozen features and can only combine them linearly. A real network can learn to ignore null-space dimensions by reshaping its internal representations. So the RFNN may show a *stronger* effect of d_latent than the MLP. If Track 1 shows a weaker effect than Track 2 predicts, that's interesting — it means representation learning partially compensates for excess latent dimensions.

---

## Shared: Data Generation

Gaussian mixture in d_intrinsic dimensions, embedded into d_latent dimensions with small noise in null-space directions, then randomly rotated to remove axis alignment.

```python
def generate_data(n, d_intrinsic, d_latent, k=10, sigma_noise=1e-2,
                  scale=3.0, seed=42):
    rng = np.random.default_rng(seed)

    # Cluster centers in d_intrinsic dims, zero-padded to d_latent
    means_intrinsic = rng.standard_normal((k, d_intrinsic)) * scale
    means_full = np.zeros((k, d_latent))
    means_full[:, :d_intrinsic] = means_intrinsic

    # Sample assignments
    labels = rng.integers(0, k, size=n)
    data = np.zeros((n, d_latent))

    # Signal dimensions: unit variance around cluster centers
    data[:, :d_intrinsic] = (
        means_intrinsic[labels] + rng.standard_normal((n, d_intrinsic))
    )

    # Null-space dimensions: small noise (keeps distribution well-posed)
    data[:, d_intrinsic:] = (
        rng.standard_normal((n, d_latent - d_intrinsic)) * sigma_noise
    )

    # Random orthogonal rotation
    Q, _ = np.linalg.qr(rng.standard_normal((d_latent, d_latent)))
    data = data @ Q.T
    means_full = means_full @ Q.T

    return (
        torch.tensor(data, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.long),
        torch.tensor(means_full, dtype=torch.float32),
        torch.tensor(Q, dtype=torch.float32),
    )


def generate_test_samples(means, k, d_intrinsic, d_latent, Q,
                          sigma_noise=1e-2, n=2048, seed=None):
    """Fresh samples from the true distribution (for test loss / score error)."""
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, k, size=n)
    # Work in pre-rotation coordinates
    means_orig = means @ Q  # un-rotate to get original coords
    data = np.zeros((n, d_latent))
    data[:, :d_intrinsic] = (
        means_orig.numpy()[labels, :d_intrinsic]
        + rng.standard_normal((n, d_intrinsic))
    )
    data[:, d_intrinsic:] = (
        rng.standard_normal((n, d_latent - d_intrinsic)) * sigma_noise
    )
    data = data @ Q.T.numpy()
    return torch.tensor(data, dtype=torch.float32)
```

**Sanity check before experiments:** PCA the generated data and verify you see d_intrinsic large eigenvalues (~1) and (d_latent - d_intrinsic) small eigenvalues (~σ²_noise).

---

## Shared: True Score (analytic)

For a Gaussian mixture P_0 = (1/k) Σ_j N(μ_j, Σ_data), the noised distribution at diffusion time t (OU forward process) is:

    P_t(x) = (1/k) Σ_j N(e^{-t} μ_j, Σ_t)

where Σ_t = Δ_t I + e^{-2t} Σ_data and Δ_t = 1 - e^{-2t}.

For isotropic within-cluster covariance (variance 1 in signal dims, σ²_noise in null dims), and σ_noise << 1, Σ_t ≈ I in signal dims. The true score is:

    ∇ log P_t(x) = Σ_j w_j(x,t) · Σ_t^{-1} (e^{-t} μ_j - x)

where w_j are softmax posterior mixture weights.

```python
def true_score(x, t, means):
    """
    Analytically compute ∇ log P_t(x).
    Assumes isotropic within-cluster covariance ≈ I.
    """
    delta_t = 1 - math.exp(-2 * t)
    e_neg_t = math.exp(-t)
    sigma_t = delta_t + math.exp(-2 * t)  # = 1 for unit-variance clusters

    shifted = x.unsqueeze(1) - e_neg_t * means.unsqueeze(0)  # (batch, k, d)
    log_w = -0.5 * (shifted ** 2).sum(-1) / sigma_t           # (batch, k)
    log_w -= log_w.max(dim=1, keepdim=True).values
    w = torch.softmax(log_w, dim=1)                            # (batch, k)

    score = (w.unsqueeze(-1) * (-shifted / sigma_t)).sum(1)    # (batch, d)
    return score
```

---

## Shared: Evaluation Metrics

```python
def evaluate(model, train_data, means, n_gen=10000, t_eval=0.1,
             d_intrinsic=None, d_latent=None, Q=None, sigma_noise=1e-2, k=10):
    d = train_data.shape[1]
    results = {}

    with torch.no_grad():
        e_neg_t = math.exp(-t_eval)
        delta_t = 1 - math.exp(-2 * t_eval)

        # --- Train loss ---
        noise = torch.randn_like(train_data)
        x_t = e_neg_t * train_data + math.sqrt(delta_t) * noise
        t_batch = torch.full((len(train_data),), t_eval)
        pred = model(x_t, t_batch)
        results['train_loss'] = (
            (math.sqrt(delta_t) * pred + noise) ** 2
        ).sum(-1).mean().item() / d

        # --- Test loss ---
        test_data = generate_test_samples(means, k, d_intrinsic, d_latent, Q,
                                          sigma_noise)
        noise_test = torch.randn_like(test_data)
        x_t_test = e_neg_t * test_data + math.sqrt(delta_t) * noise_test
        t_batch_test = torch.full((len(test_data),), t_eval)
        pred_test = model(x_t_test, t_batch_test)
        results['test_loss'] = (
            (math.sqrt(delta_t) * pred_test + noise_test) ** 2
        ).sum(-1).mean().item() / d

        results['gen_loss'] = results['test_loss'] - results['train_loss']

        # --- Score error (E_score) ---
        noise_eval = torch.randn_like(test_data)
        x_t_eval = e_neg_t * test_data + math.sqrt(delta_t) * noise_eval
        pred_score = model(x_t_eval, t_batch_test)
        true_s = true_score(x_t_eval, t_eval, means)
        results['score_error'] = (
            (pred_score - true_s) ** 2
        ).sum(-1).mean().item() / d

        # --- Generate samples → memorization metrics ---
        generated = generate_samples(model, n_gen, d)

        # NN ratio: d(gen, NN1_train) / d(NN1_train, NN2_train)
        dists_to_train = torch.cdist(generated, train_data)
        nn1_dists, nn1_idx = dists_to_train.min(dim=1)

        train_dists = torch.cdist(train_data, train_data)
        train_dists.fill_diagonal_(float('inf'))
        nn2_dists = train_dists[nn1_idx].min(dim=1).values

        nn_ratio = nn1_dists / (nn2_dists + 1e-10)
        results['mean_nn_ratio'] = nn_ratio.mean().item()
        results['memorization_fraction'] = (nn_ratio < 1/3).float().mean().item()

        # --- MMD between generated and true distribution ---
        true_samples = generate_test_samples(means, k, d_intrinsic, d_latent, Q,
                                             sigma_noise, n=n_gen)
        results['mmd'] = compute_mmd(generated, true_samples)

    return results
```

### τ_gen definition

**Primary:** Score error plateau. First step where the running-minimum of score_error hasn't improved by more than 5% over the last 10 eval windows (10K steps). This directly measures when the model has learned the population score, connects to Bonnaire's E_score metric for the RFNN, and is available on both tracks.

**Secondary (Track 1 only):** MMD between generated samples and true-distribution samples. Calibrate by computing MMD between two independent true-distribution sample sets (the "noise floor"). Useful as a sanity check that score_error plateau actually corresponds to good generation.

```python
def compute_mmd(x, y, bandwidth=1.0):
    """MMD² between sample sets x and y using Gaussian kernel."""
    def kernel(a, b):
        dists = torch.cdist(a, b) ** 2
        return torch.exp(-dists / (2 * bandwidth ** 2))
    xx = kernel(x, x).mean()
    yy = kernel(y, y).mean()
    xy = kernel(x, y).mean()
    return (xx + yy - 2 * xy).item()
```

### τ_mem definition

First step at which memorization_fraction exceeds 1% (>1% of generated samples have NN ratio < 1/3). Matches Bonnaire's approach. For Track 2 (RFNN, no generation), use train/test loss divergence as proxy: first step where gen_loss > 5% of test_loss.

---

## Shared: Compute Tracking

Log these every eval step so metrics can be plotted against any x-axis:

```python
def log_step(step, start_time, model, batch_size):
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    flops_per_step = 6 * n_params * batch_size  # rough: 2x forward + 4x backward
    return {
        'step': step,
        'wall_time_sec': time.time() - start_time,
        'n_params': n_params,
        'total_flops': step * flops_per_step,
    }
```

Primary x-axis is **step**. Use **wall_time** and **total_flops** as secondary axes when comparing across model sizes (Experiment 2) to check that effects aren't compute artifacts.

---

## Track 1: Small MLP Diffusion Model (Empirical)

### Model

```python
class MLPScore(nn.Module):
    """
    Small MLP score network. Trains with random t (standard diffusion).
    Can generate samples via reverse SDE.
    """
    def __init__(self, d_latent, hidden=256, n_freq=32):
        super().__init__()
        self.n_freq = n_freq
        self.register_buffer(
            'freqs', torch.exp(torch.linspace(0, math.log(1000), n_freq))
        )
        d_input = d_latent + 2 * n_freq
        self.net = nn.Sequential(
            nn.Linear(d_input, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_latent),
        )

    def forward(self, x_t, t):
        if isinstance(t, (int, float)):
            t = torch.full((x_t.shape[0],), t, device=x_t.device)
        if t.dim() == 0:
            t = t.expand(x_t.shape[0])
        t_emb = torch.cat([
            torch.sin(t.unsqueeze(-1) * self.freqs),
            torch.cos(t.unsqueeze(-1) * self.freqs),
        ], dim=-1)
        inp = torch.cat([x_t, t_emb], dim=-1)
        return self.net(inp)
```

### Training (random t, standard diffusion)

```python
def train_step(model, data, optimizer, t_min=0.01, t_max=3.0):
    batch_size = data.shape[0]
    t = torch.rand(batch_size) * (t_max - t_min) + t_min
    delta_t = 1 - torch.exp(-2 * t)
    e_neg_t = torch.exp(-t)

    noise = torch.randn_like(data)
    x_t = e_neg_t.unsqueeze(-1) * data + delta_t.sqrt().unsqueeze(-1) * noise

    pred_score = model(x_t, t)
    residual = delta_t.sqrt().unsqueeze(-1) * pred_score + noise
    loss = (residual ** 2).sum(dim=-1).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()
```

### Sample Generation (Euler-Maruyama reverse SDE)

```python
@torch.no_grad()
def generate_samples(model, n_gen, d, n_steps=500, t_max=3.0, t_min=0.01):
    dt = (t_max - t_min) / n_steps
    x = torch.randn(n_gen, d)

    for i in range(n_steps):
        t_curr = t_max - i * dt
        t_batch = torch.full((n_gen,), t_curr)
        score = model(x, t_batch)
        noise = torch.randn_like(x) * math.sqrt(2 * dt)
        x = x + (x + 2 * score) * dt + noise

    return x
```

### Track 1 Config

```python
track1_config = {
    'model': 'mlp',
    'hidden': 256,
    'n_freq': 32,
    'optimizer': 'adam',
    'lr': 1e-4,
    'batch_size': 256,     # or min(n, 256)
    'total_steps': 300000,
    'eval_interval': 1000,
    'n_gen_samples': 10000,
}
```

### What to look for (Track 1)

- τ_gen and τ_mem at each (d_latent, d_intrinsic, n) config
- Does τ_mem decrease as d_latent/d_intrinsic grows?
- Does τ_gen change?
- Does the generalization window [τ_gen, τ_mem] shrink?
- Does the MLP learn to ignore null-space dimensions? (Check by looking at whether generated samples have appropriate variance in signal vs null directions)

---

## Track 2: RFNN (Mechanistic)

### Model

```python
class RFNNScore(nn.Module):
    """
    t-conditioned RFNN. W frozen, A learned.
    Trains at fixed t for theoretical tractability.
    """
    def __init__(self, d_latent, p, t_fixed):
        super().__init__()
        self.d = d_latent
        self.p = p
        self.t_fixed = t_fixed
        # Frozen random first layer
        self.register_buffer('W', torch.randn(p, d_latent) / math.sqrt(d_latent))
        # Learned second layer, zero-initialized (matches Bonnaire)
        self.A = nn.Parameter(torch.zeros(d_latent, p))

    def forward(self, x_t):
        """No t input — trained at fixed t."""
        features = torch.tanh(x_t @ self.W.T)  # (batch, p)
        return features @ self.A.T / math.sqrt(self.p)
```

### Training (fixed t, full-batch)

```python
def train_step_rfnn(model, data, optimizer):
    t = model.t_fixed
    delta_t = 1 - math.exp(-2 * t)
    e_neg_t = math.exp(-t)

    noise = torch.randn_like(data)
    x_t = e_neg_t * data + math.sqrt(delta_t) * noise

    pred_score = model(x_t)
    residual = math.sqrt(delta_t) * pred_score + noise
    loss = (residual ** 2).sum(dim=-1).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()
```

### Computing U(t) directly

This is the key Track 2 analysis. U is a p × p matrix you can compute before training:

```python
def compute_U(W, data, t, n_noise_samples=50):
    """
    U = (1/n) Σ_ν E_ξ [ φ(x^ν_t(ξ)) φ(x^ν_t(ξ))^T ]
    where φ(x) = tanh(Wx/√d)

    Approximate the expectation over ξ with n_noise_samples draws.
    """
    p, d = W.shape
    n = data.shape[0]
    delta_t = 1 - math.exp(-2 * t)
    e_neg_t = math.exp(-t)

    U = torch.zeros(p, p)
    for _ in range(n_noise_samples):
        noise = torch.randn_like(data)
        x_t = e_neg_t * data + math.sqrt(delta_t) * noise
        phi = torch.tanh(x_t @ W.T)  # (n, p)
        U += phi.T @ phi / n          # (p, p)
    U /= n_noise_samples

    return U
```

### Eigenvalue analysis of U

```python
def analyze_U_spectrum(U):
    """Eigendecompose U and characterize the bulk structure."""
    eigenvalues = torch.linalg.eigvalsh(U)  # sorted ascending
    return eigenvalues.cpu().numpy()
```

For each (d_latent, d_intrinsic, n) config:
1. Generate data
2. Compute U(t)
3. Eigendecompose U
4. Plot the spectrum
5. Look for bulk structure — does the two-bulk picture change with d_latent/d_intrinsic?

### Score Jacobian eigenvalues during training (cheap for RFNN)

The Jacobian ∂s_A/∂x has a closed form:

    J(x) = (1/√p) A · diag(σ'(Wx/√d)) · W / √d

```python
def rfnn_jacobian_eigenvalues(model, x_points, n_points=200):
    """Analytic Jacobian eigenvalues. Cheap for RFNN."""
    d = model.d
    results = []

    with torch.no_grad():
        for x in x_points[:n_points]:
            x = x.unsqueeze(0)
            pre_act = x @ model.W.T / math.sqrt(d)       # (1, p)
            sigma_prime = 1 - torch.tanh(pre_act) ** 2    # tanh', (1, p)

            # J = (1/√p) A diag(σ') W / √d
            J = (model.A
                 @ torch.diag(sigma_prime.squeeze())
                 @ model.W) / (math.sqrt(model.p) * math.sqrt(d))  # (d, d)

            eigs = torch.linalg.eigvalsh(J)
            results.append(eigs)

    return torch.stack(results)  # (n_points, d)
```

What to look for in the Jacobian spectrum during training:
- Early training: eigenvalues grow in signal-subspace directions only
- Late training (memorization): eigenvalues grow in null-subspace directions too
- Project eigenvalues onto known signal vs null subspaces (using Q from data generation) to make this concrete

### Track 2 Config

```python
track2_config = {
    'model': 'rfnn',
    'p_ratio': 16,          # p = p_ratio * d_latent
    't_fixed': 0.01,        # matches Bonnaire
    'optimizer': 'sgd',
    'lr': 1e-3,
    'momentum': 0.0,
    'batch_size': 'full',   # full-batch (data is small, RFNN is cheap)
    'total_steps': 300000,
    'eval_interval': 1000,
    'eigenvalue_tracking': True,
    'eigenvalue_n_points': 200,
}
```

### What to look for (Track 2)

- **Spectrum of U**: How does it change across (d_latent, d_intrinsic, n)? Does the two-bulk picture become three-bulk? Does the gap between bulks shrink with d_latent?
- **Eigenvalue predictions vs observed dynamics**: Do training timescales (τ_gen from score_error plateau, τ_mem from train/test loss divergence) match 1/λ predictions from U's spectrum?
- **Jacobian spectrum evolution**: Does the model first learn signal-subspace structure (generalization) then null-subspace structure (memorization)?
- **Comparison with Track 1**: Is the d_latent effect stronger in RFNN than in MLP? If so, this suggests the MLP learns to ignore null-space dimensions (representation learning compensates for excess dimensions).

Note: RFNN at fixed t cannot generate samples (no score at other t values). Use score_error and train/test loss divergence as proxies for τ_gen and τ_mem instead of FID and NN ratio. Alternatively, train a separate t-conditioned RFNN for generation quality checks, but this is secondary.

---

## Experiment Sweeps

All experiments run on **Track 1 (MLP) first**. Track 2 (RFNN) runs on the same data configs afterward.

### Experiment 0: Sanity checks

Before any sweeps:
1. PCA the generated data — verify d_intrinsic large eigenvalues, rest at σ²_noise
2. Train MLP on a single config, verify it can generate reasonable samples
3. Train RFNN on a single config, verify score_error decreases
4. Verify true_score function by checking it matches finite-difference gradient of log P_t
5. Check that MLP is in the overparameterized regime — if hidden=256 can't memorize at small n, increase hidden
6. Compute MMD noise floor (MMD between two independent true-distribution sample sets)

### Experiment 1: Reproduce τ_mem ∝ n (validation)

Confirm the Bonnaire scaling before testing dimensionality.

| Parameter   | Values                      |
|-------------|-----------------------------|
| d_intrinsic | 5                           |
| d_latent    | 20                          |
| k           | 10                          |
| n           | 100, 200, 500, 1000, 2000   |
| σ_noise     | 1e-2                        |
| steps       | 500,000                     |

Track 1 (MLP): hidden=256, Adam lr=1e-4, batch_size=min(n, 256)
Track 2 (RFNN): p=320, SGD lr=1e-3, full-batch, t_fixed=0.01

Eval every 1000 steps. 5 seeds per config for error bars.

**Expected:** τ_mem ∝ n. τ_gen constant.

**Graphs:**
- (a) τ_mem vs n with linear fit (both tracks)
- (b) τ_gen vs n — should be flat (both tracks)
- (c) Training curves overlaid for all n: score_error vs step (both tracks)
- (d) Training curves overlaid for all n: memorization_fraction vs step (Track 1)
- (e) Training curves overlaid for all n: gen_loss vs step (both tracks)
- (f) Spectrum of U for each n, overlaid (Track 2)

### Experiment 2: Sweep d_latent (main hypothesis)

| Parameter   | Values                              |
|-------------|-------------------------------------|
| d_intrinsic | 5                                   |
| d_latent    | 5, 8, 10, 15, 20, 30, 40           |
| k           | 10                                  |
| n           | 500                                 |
| σ_noise     | 1e-2                                |
| steps       | 300,000                             |

Two sub-experiments for controlling capacity confound:

**(2a) Constant ψ_p** (p/d fixed):
- Track 1: hidden = 16 × d_latent
- Track 2: p = 16 × d_latent

**(2b) Constant absolute capacity:**
- Track 1: hidden = 256
- Track 2: p = 640

3 seeds per config.

**Expected:** As d_latent/d_intrinsic grows, τ_mem decreases, generalization window shrinks.

**Graphs:**
- (a) τ_gen and τ_mem vs d_latent/d_intrinsic (both tracks, both sub-experiments)
- (b) Generalization window width (τ_mem - τ_gen) vs d_latent/d_intrinsic
- (c) Same as (a) but x-axis is total_flops instead of step (for 2a where model size varies)
- (d) Score_error vs step for all d_latent values overlaid (both tracks)
- (e) Memorization_fraction vs step for all d_latent values overlaid (Track 1)
- (f) MMD vs step for all d_latent values overlaid (Track 1)
- (g) Spectrum of U for each d_latent, overlaid or stacked (Track 2) — **key mechanistic plot**
- (h) Comparison of 2a vs 2b to show effect is not a capacity artifact

### Experiment 3: Sweep d_intrinsic (complementary)

| Parameter   | Values                    |
|-------------|---------------------------|
| d_intrinsic | 2, 5, 8, 12, 16, 20      |
| d_latent    | 20                        |
| k           | 10                        |
| n           | 500                       |
| steps       | 300,000                   |

Track 1: hidden=256. Track 2: p=320. 3 seeds per config.

**Expected:** As d_intrinsic → d_latent, τ_mem increases, generalization window widens.

**Graphs:**
- (a) τ_gen and τ_mem vs d_intrinsic (or vs d_latent/d_intrinsic for direct comparison with Exp 2)
- (b) Generalization window width vs d_intrinsic
- (c) Spectrum of U for each d_intrinsic, overlaid (Track 2)
- (d) Score_error and memorization_fraction vs step for all d_intrinsic overlaid

### Experiment 4: Phase diagram (d_latent × n)

| Parameter   | Values                          |
|-------------|---------------------------------|
| d_intrinsic | 5                               |
| d_latent    | 5, 10, 20, 40                   |
| n           | 100, 200, 500, 1000, 2000       |
| k           | 10                              |
| steps       | 500,000                         |

Track 1: hidden=256. Track 2: p=16 × d_latent. 3 seeds per config. 20 configs.

**Graphs:**
- (a) Heatmap of generalization window width (τ_mem - τ_gen) in (d_latent/d_intrinsic, n) space (Track 1)
- (b) Same heatmap but for Track 2
- (c) Heatmap of memorization_fraction at a fixed step (e.g., step 200K) in same space (Track 1)
- (d) Contour lines showing τ_mem = const in (d_latent/d_intrinsic, n) space — does τ_mem depend on both, or primarily on one?
- (e) Bonnaire-style plot: for each d_latent, plot τ_mem vs n and check if the linear scaling τ_mem ∝ n holds with a slope that depends on d_latent/d_intrinsic

---

## Metrics Summary

| Metric                 | Definition                                         | Identifies  | Track |
|------------------------|-----------------------------------------------------|------------|-------|
| train_loss             | Score matching loss on training data                | fitting     | 1 & 2 |
| test_loss              | Score matching loss on held-out true-dist samples   | overfitting | 1 & 2 |
| gen_loss               | test_loss - train_loss                              | overfitting | 1 & 2 |
| score_error (E_score)  | ‖s_θ - s_true‖² on test points                      | **τ_gen** (primary) | 1 & 2 |
| MMD                    | MMD² between generated and true-dist samples        | τ_gen (secondary) | **1 only** |
| mean_nn_ratio          | Mean d(gen, NN1) / d(NN1, NN2_train)                | memorization| **1 only** |
| memorization_fraction  | Frac of gen samples with NN ratio < 1/3             | **τ_mem**   | **1 only** |
| U eigenvalues          | Spectrum of feature correlation matrix               | mechanistic | **2 only** |
| Jacobian eigenvalues   | Spectrum of ∂s_θ/∂x at test points                  | mechanistic | **2 only** |
| step                   | Training step number                                | x-axis      | 1 & 2 |
| wall_time_sec          | Seconds since training start                        | x-axis      | 1 & 2 |
| total_flops            | Cumulative estimated FLOPs                          | x-axis      | 1 & 2 |

For Track 2 (RFNN, fixed t, no generation), use **train/test loss divergence** as proxy for τ_mem instead of NN ratio.

---

## Run Budget

| Experiment | Configs | Seeds | Runs per track | Total (both tracks) |
|------------|---------|-------|----------------|---------------------|
| 0 (sanity) | 1       | 1     | 1              | 2                   |
| 1          | 5       | 5     | 25             | 50                  |
| 2a         | 7       | 3     | 21             | 42                  |
| 2b         | 7       | 3     | 21             | 42                  |
| 3          | 6       | 3     | 18             | 36                  |
| 4          | 20      | 3     | 60             | 120                 |
| **Total**  |         |       |                | **292**             |

---

## Potential Issues and Mitigations

**MLP can't memorize:** If hidden=256 MLP doesn't memorize even at n=100, it's in the architectural regularization regime. Fix: increase hidden until memorization is observed at small n, then use that size everywhere.

**RFNN generation for NN ratio:** RFNN at fixed t can't generate samples. Options: (a) use train/test loss divergence as τ_mem proxy, (b) train a separate t-conditioned RFNN for generation checks (secondary analysis). The U spectrum analysis doesn't need generation at all.

**σ_noise sensitivity:** If too small → numerical issues in score near manifold. If too large → intrinsic dimension blurred. Run sanity check (Exp 0). Try σ_noise ∈ {1e-3, 1e-2, 5e-2} and pick whichever gives cleanest eigenvalue separation in PCA.

**p/hidden scaling confound:** Experiment 2b (fixed capacity) directly addresses this. If 2a and 2b disagree, report both and discuss.

**Batch size vs n:** For Track 1 with batch_size=256, small n (100) means each sample seen ~2.5x per step vs large n (2000) at ~0.13x per step. This doesn't affect the FLOPs view but does affect the step view. Note this when interpreting Experiment 1.

**Score error at single t_eval:** We compute score_error at t_eval=0.1, but the model trains across all t (Track 1). Could miss dynamics at other t. Mitigation: also log score_error at t=0.01 and t=1.0 as secondary metrics.

---

## Logging

```python
import json, os, time

def setup_run(config, run_dir):
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(f'{run_dir}/checkpoints', exist_ok=True)
    with open(f'{run_dir}/config.json', 'w') as f:
        json.dump(config, f, indent=2)
    return open(f'{run_dir}/metrics.jsonl', 'w')

def log_metrics(metrics_file, metrics):
    """Write one JSON line per eval step."""
    metrics_file.write(json.dumps(metrics) + '\n')
    metrics_file.flush()

# Save model checkpoints at detected τ_gen and τ_mem
# Save eigenvalue arrays (Track 2) as separate .npz file
```

Use **wandb** if available for live monitoring, otherwise JSON is fine.
