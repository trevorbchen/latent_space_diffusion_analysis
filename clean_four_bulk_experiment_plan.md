# Clean four-bulk spectrum experiment plan

Goal: find a cleaner RFNN spectral figure while keeping the Bonnaire-style
`tanh` random features. These are spectral-only runs: build `U`, eigendecompose
it, and plot the full-rank sorted spectrum. No score-network training.

## Fixed choices

- Activation: `tanh`
- Monte Carlo samples for `U`: `500`
- Diffusion time: `t = 0.01`
- Sample count: start with `n = 500`
- Width: `p = d_latent + n + rank_null`, with `rank_null = 300`
- Full rank retained in plots: signal, noise-dim buffer, sample, and rank-null

Using `p = d_latent + n + rank_null` keeps the rank-null tail visible without
letting it dominate the figure the way `p_ratio = 64` sometimes does.

## Experiment axes

### Axis A: data distribution

1. `gaussian`
   - Pure anisotropic Gaussian.
   - Best microscope for the variance-hierarchy mechanism.
   - Removes cluster-center spikes and mixture imbalance.

2. `gmm`
   - Anisotropic Gaussian mixture.
   - Closer to the main synthetic diffusion task.
   - More likely to show extra shoulders from center structure.

### Axis B: null noise

Try:

- `sigma_noise = 0.1`
- `sigma_noise = 0.2`
- `sigma_noise = 0.3`
- `sigma_noise = 0.5`

Expectation:

- Too small: huge anisotropy, but null/sample edges can get weird.
- Too large: weaker separation.
- Likely sweet spot: `0.2` or `0.3`.

### Axis C: signal/data scale

For pure Gaussian:

- `sigma_signal = 0.5`
- `sigma_signal = 0.75`
- `sigma_signal = 1.0`
- `sigma_signal = 1.5`

For GMM:

- `center_scale = 0.75`
- `center_scale = 1.0`
- `center_scale = 1.5`
- `center_scale = 2.0`

Expectation:

- Lower scale keeps `tanh` away from saturation.
- Too low may collapse signal/noise separation.
- Likely sweet spot: Gaussian `sigma_signal = 1.0`; GMM `center_scale = 1.0`
  or `1.5`.

## Run order

### 1. Quick sanity

```powershell
python clean_four_bulk_sweep.py --preset quick
```

Runs both data kinds on both sweeps:

- Exp 2: vary `d_latent`, fixed `d_intrinsic = 5`
- Exp 3: vary `d_intrinsic`, fixed `d_latent = 20`

Default candidate:

- `sigma_noise = 0.3`
- `sigma_signal = 1.0`
- `center_scale = 1.5`
- `mc_samples = 500`
- `rank_null = 300`

### 2. Null-noise sweep

```powershell
python clean_four_bulk_sweep.py --preset sigma
```

Purpose: find the `sigma_noise` value that separates the signal, noise-dim,
sample, and rank-null regions with the fewest shoulders.

### 3. Data-scale sweep

```powershell
python clean_four_bulk_sweep.py --preset scale
```

Purpose: reduce `tanh` saturation artifacts without making the signal bulk
too weak.

### 4. Data-kind comparison

```powershell
python clean_four_bulk_sweep.py --preset data-kind
```

Purpose: decide whether the paper-body spectral figure should use the pure
Gaussian microscope, the GMM, or both.

## What counts as cleaner

Prefer configurations where:

- The signal/noise-dim cut at `d_intrinsic` is visible.
- The noise-dim/sample cut at `d_latent` is visible.
- The sample/rank-null cut at `d_latent + n` is visible.
- The sample region has at most a soft shoulder, not a clear fifth peak.
- The rank-null tail remains visible but does not visually dominate.

Recommended paper framing if the pure Gaussian is cleanest:

> We first show a spectral microscope on anisotropic Gaussian data, where the
> four index-defined populations are cleanly separated. We then verify the
> same index boundaries and buffer-width scaling on the GMM data used for the
> trainable diffusion experiments.
