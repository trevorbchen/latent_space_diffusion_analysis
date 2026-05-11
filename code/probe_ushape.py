"""
probe_ushape.py  –  minimal numpy MLP for the U-shape recovery probes.

Replicates the sigma_perp=0.01 dlatent sweep (h = 8*d_latent, 300k / 50k steps,
OU diffusion, Gaussian-mixture data) in pure numpy/scipy.

Probes:
  1. Signal-subspace mass of W1:  sig_mass = ||W1 @ Q_sig||^2_F / ||W1||^2_F
  2. Effective rank of hidden activation matrix (SVD on test set)
  3. Activation density (fraction of neurons with |pre-act| > 0.1)
  4. Per-dim null-score MSE vs Var(s*_null) at t_eval

All probe data is written to probe_results.json.
"""
import sys, json, math, time
import numpy as np
from scipy.special import erf

RNG_SEED = 42

# --------------------------------------------------------------------------
# GELU and its derivative
# --------------------------------------------------------------------------
def gelu(x):
    return 0.5 * x * (1 + erf(x / math.sqrt(2)))

def gelu_prime(x):
    cdf = 0.5 * (1 + erf(x / math.sqrt(2)))
    pdf = np.exp(-0.5 * x**2) / math.sqrt(2 * math.pi)
    return cdf + x * pdf

# --------------------------------------------------------------------------
# Data generation  (matches paper: k=10, s=3, sigma_sig=1)
# --------------------------------------------------------------------------
def generate_data(n, d_int, d_lat, sigma_perp=0.01, sigma_sig=1.0,
                  scale=3.0, k=10, seed=42):
    rng = np.random.default_rng(seed)
    if k <= d_int:
        raw = rng.standard_normal((k, d_int))
        Qc, _ = np.linalg.qr(raw.T)
        means_int = Qc.T * scale
    else:
        raw = rng.standard_normal((d_int, d_int))
        Qc, _ = np.linalg.qr(raw)
        means_int = np.zeros((k, d_int))
        means_int[:d_int] = Qc * scale
        for i in range(d_int, k):
            v = rng.standard_normal(d_int)
            means_int[i] = v / np.linalg.norm(v) * scale

    means_full = np.zeros((k, d_lat))
    means_full[:, :d_int] = means_int

    labels = rng.integers(0, k, size=n)
    data = np.zeros((n, d_lat))
    data[:, :d_int] = (means_int[labels]
                       + rng.standard_normal((n, d_int)) * sigma_sig)
    if d_lat > d_int:
        data[:, d_int:] = rng.standard_normal((n, d_lat - d_int)) * sigma_perp

    # random rotation Q
    Q, _ = np.linalg.qr(rng.standard_normal((d_lat, d_lat)))
    data = data @ Q.T
    means_full = means_full @ Q.T
    return data, labels, means_full, Q, means_int

def analytic_score(x_t, t, means_full, sigma_sig=1.0, sigma_perp=0.01,
                   d_int=5):
    """Analytic Gaussian-mixture score ∇ log p_t(x_t)."""
    e2t = math.exp(-2 * t)
    dt = 1 - math.exp(-2 * t)
    k, d = means_full.shape
    # per-cluster variance: signal dims get e2t*sigma_sig^2+dt, null gets e2t*sigma_perp^2+dt
    var_sig  = e2t * sigma_sig**2  + dt
    var_null = e2t * sigma_perp**2 + dt
    # Build full covariance mask (assumes Q already applied to means_full)
    # We need to identify signal vs null dims from Q – but Q is orthogonal rotation.
    # For the analytic score we work in the rotated frame where block structure is known,
    # then rotate back.  Here we use the raw formula via cluster weights.
    n_pts = x_t.shape[0]
    scores = np.zeros_like(x_t)
    # In the original (pre-Q) frame, signal dims 0:d_int, null dims d_int:d_lat.
    # We don't have x in that frame here, so we compute in rotated frame.
    # For a Gaussian mixture: score = sum_c w_c * score_c / sum_c w_c
    # score_c(x_t) = -(x_t - e^{-t} mu_c) / Sigma_c
    # where Sigma_c is the diagonal covariance in the ORIGINAL frame, then rotated.
    # Easier: compute in original frame, then rotate.
    # We don't have Q here – pass it as argument.
    raise NotImplementedError("Use analytic_score_full below")

def analytic_score_full(x_t, t, means_full, Q,
                        sigma_sig=1.0, sigma_perp=0.01, d_int=5):
    """
    x_t: (n, d_lat) in rotated frame
    means_full: (k, d_lat) in rotated frame
    Q: (d_lat, d_lat) the rotation matrix  (x_orig @ Q.T = x_rotated)
    Returns score (n, d_lat) in rotated frame.
    """
    n_pts, d_lat = x_t.shape
    e_t  = math.exp(-t)
    e2t  = math.exp(-2 * t)
    dt   = 1 - e2t
    k    = means_full.shape[0]

    var_sig  = e2t * sigma_sig**2  + dt
    var_null = e2t * sigma_perp**2 + dt

    # Work in original frame: x_orig = x_rotated @ Q
    x_orig  = x_t   @ Q          # (n, d_lat)
    mu_orig = means_full @ Q      # (k, d_lat)

    # Per-dim noise variance in original frame
    var_vec = np.ones(d_lat) * var_null
    var_vec[:d_int] = var_sig

    # Compute per-cluster weights and contributions
    # log w_c(x_t) = -0.5 * ||x_t - e^{-t} mu_c||^2_Sigma^{-1} + const
    scaled_x   = x_orig   / var_vec                  # (n, d_lat)
    scaled_mu  = mu_orig * e_t / var_vec              # (k, d_lat)
    # quadratic terms
    xx = 0.5 * np.sum(x_orig * scaled_x, axis=-1)    # (n,)
    xmu = x_orig @ scaled_mu.T                       # (n, k)
    mumu = 0.5 * e2t * np.sum(mu_orig * (mu_orig / var_vec), axis=-1)  # (k,)

    log_w = - xx[:, None] + xmu - mumu[None, :]      # (n, k)
    log_w -= log_w.max(axis=-1, keepdims=True)
    w = np.exp(log_w)
    w /= w.sum(axis=-1, keepdims=True)                # (n, k)  softmax weights

    # Per-cluster score in original frame: -(x_t - e_t*mu) / var_vec
    # score = sum_c w_c * [-(x_t - e_t*mu_c) / var_vec]
    # = -(x_t / var_vec) + sum_c w_c * e_t * mu_c / var_vec
    mean_mu = w @ mu_orig                             # (n, d_lat)
    score_orig = (-x_orig + e_t * mean_mu) / var_vec # (n, d_lat)

    # Rotate back
    return score_orig @ Q.T

# --------------------------------------------------------------------------
# MLP (3-layer GELU with sinusoidal time embedding)
# --------------------------------------------------------------------------
class MLP:
    def __init__(self, d_lat, hidden, n_freq=32, rng=None):
        if rng is None:
            rng = np.random.default_rng(42)
        self.d_lat   = d_lat
        self.hidden  = hidden
        self.n_freq  = n_freq
        freqs = np.exp(np.linspace(0, math.log(1000), n_freq))
        self.freqs = freqs.astype(np.float32)
        d_in = d_lat + 2 * n_freq
        h    = hidden
        # He init
        s = lambda fan_in: math.sqrt(2.0 / fan_in)
        self.W1 = rng.standard_normal((d_in, h)).astype(np.float32) * s(d_in)
        self.b1 = np.zeros(h, dtype=np.float32)
        self.W2 = rng.standard_normal((h,  h)).astype(np.float32) * s(h)
        self.b2 = np.zeros(h, dtype=np.float32)
        self.W3 = rng.standard_normal((h,  h)).astype(np.float32) * s(h)
        self.b3 = np.zeros(h, dtype=np.float32)
        self.W4 = rng.standard_normal((h,  d_lat)).astype(np.float32) * s(h)
        self.b4 = np.zeros(d_lat, dtype=np.float32)
        # Adam state
        self._init_adam()

    def _init_adam(self):
        self._m = {k: np.zeros_like(v) for k, v in self._params()}
        self._v = {k: np.zeros_like(v) for k, v in self._params()}
        self._t = 0

    def _params(self):
        return [('W1',self.W1),('b1',self.b1),
                ('W2',self.W2),('b2',self.b2),
                ('W3',self.W3),('b3',self.b3),
                ('W4',self.W4),('b4',self.b4)]

    def _time_embed(self, t_batch):
        # t_batch: (B,)
        angles = t_batch[:, None] * self.freqs[None, :]   # (B, n_freq)
        return np.concatenate([np.sin(angles), np.cos(angles)], axis=-1)  # (B, 2*n_freq)

    def forward(self, x, t_batch, store_cache=False):
        te  = self._time_embed(t_batch)
        inp = np.concatenate([x, te], axis=-1)           # (B, d_in)
        z1  = inp  @ self.W1 + self.b1;  a1 = gelu(z1)  # (B, h)
        z2  = a1   @ self.W2 + self.b2;  a2 = gelu(z2)
        z3  = a2   @ self.W3 + self.b3;  a3 = gelu(z3)
        out = a3   @ self.W4 + self.b4                   # (B, d_lat)
        if store_cache:
            self._cache = dict(inp=inp, z1=z1, a1=a1, z2=z2, a2=a2,
                               z3=z3, a3=a3, out=out)
        return out

    def backward(self, x, t_batch, target_score):
        """Score matching loss: ||sqrt(dt)*s_theta + eps||^2 averaged over batch.
        Equivalent to dt * ||s_theta - s*||^2 up to variance.
        We use the simpler form: L = mean ||s_theta(x_t,t) - s*(x_t,t)||^2
        (direct score MSE – fine for learning, slight difference from training loss)."""
        B  = x.shape[0]
        out = self.forward(x, t_batch, store_cache=True)
        c   = self._cache
        err = (out - target_score) / B               # (B, d_lat), scaled by 1/B
        loss = 0.5 * np.sum((out - target_score)**2) / B

        # layer 4 backward
        dW4 = c['a3'].T @ err                        # (h, d_lat)
        db4 = err.sum(axis=0)
        da3 = err @ self.W4.T                        # (B, h)

        # layer 3 backward
        dz3 = da3 * gelu_prime(c['z3'])
        dW3 = c['a2'].T @ dz3
        db3 = dz3.sum(axis=0)
        da2 = dz3 @ self.W3.T

        # layer 2 backward
        dz2 = da2 * gelu_prime(c['z2'])
        dW2 = c['a1'].T @ dz2
        db2 = dz2.sum(axis=0)
        da1 = dz2 @ self.W2.T

        # layer 1 backward
        dz1 = da1 * gelu_prime(c['z1'])
        dW1 = c['inp'].T @ dz1
        db1 = dz1.sum(axis=0)

        grads = dict(W1=dW1,b1=db1,W2=dW2,b2=db2,W3=dW3,b3=db3,W4=dW4,b4=db4)
        return loss, grads

    def adam_step(self, grads, lr=1e-4, beta1=0.9, beta2=0.999, eps=1e-8):
        self._t += 1
        t_ = self._t
        for name, p in self._params():
            g = grads[name]
            self._m[name] = beta1 * self._m[name] + (1-beta1) * g
            self._v[name] = beta2 * self._v[name] + (1-beta2) * g**2
            mhat = self._m[name] / (1 - beta1**t_)
            vhat = self._v[name] / (1 - beta2**t_)
            p -= lr * mhat / (np.sqrt(vhat) + eps)

    # ------- probes -------
    def sig_mass(self, Q, d_int):
        """Fraction of W1[:d_lat, :] (input rows) mass on signal subspace.
        W1 shape: (d_in, h) — first d_lat rows correspond to x_t input,
        remaining 2*n_freq rows are time embedding.
        """
        W1_x = self.W1[:self.d_lat, :]    # (d_lat, h)
        Q_sig = Q[:, :d_int]              # (d_lat, d_int) -- signal columns of Q
        # projection of each W1_x row onto signal subspace
        proj  = Q_sig.T @ W1_x            # (d_int, h)
        sig_sq  = np.sum(proj**2)
        total_sq = np.sum(W1_x**2) + 1e-12
        return float(sig_sq / total_sq)

    def activation_density(self, x, t_batch, threshold=0.1):
        """Fraction of first-layer pre-activations with |z1| > threshold."""
        te  = self._time_embed(t_batch)
        inp = np.concatenate([x, te], axis=-1)
        z1  = inp @ self.W1 + self.b1
        return float(np.mean(np.abs(z1) > threshold))

    def hidden_rank(self, x, t_batch, layer=3):
        """Effective rank of hidden activation matrix (layer 1, 2, or 3)."""
        te  = self._time_embed(t_batch)
        inp = np.concatenate([x, te], axis=-1)
        z1  = inp  @ self.W1 + self.b1;  a1 = gelu(z1)
        z2  = a1   @ self.W2 + self.b2;  a2 = gelu(z2)
        z3  = a2   @ self.W3 + self.b3;  a3 = gelu(z3)
        acts = {1: a1, 2: a2, 3: a3}[layer]
        sv = np.linalg.svd(acts, compute_uv=False)
        sv = sv / sv.sum()
        eff_rank = float(np.exp(-np.sum(sv * np.log(sv + 1e-12))))
        return eff_rank

    def score_error_per_dim(self, x, t_eval, true_score):
        """E[||s_theta - s*||^2] / d_lat on test set x."""
        t_batch = np.full(x.shape[0], t_eval, dtype=np.float32)
        pred = self.forward(x, t_batch)
        return float(np.mean((pred - true_score)**2))

# --------------------------------------------------------------------------
# OU diffusion helpers
# --------------------------------------------------------------------------
def ou_noisy(x0, t, rng):
    e_t   = math.exp(-t)
    dt    = 1 - math.exp(-2*t)
    noise = rng.standard_normal(x0.shape).astype(np.float32)
    x_t   = e_t * x0 + math.sqrt(dt) * noise
    return x_t, noise

# --------------------------------------------------------------------------
# Training run
# --------------------------------------------------------------------------
def run_experiment(d_lat, d_int=5, sigma_perp=0.01, sigma_sig=1.0,
                   n_train=500, k=10, scale=3.0,
                   n_steps=50000, eval_every=2000,
                   t_min=0.01, t_max=3.0, t_eval=0.1,
                   lr=1e-4, batch_size=256, seed=42, verbose=True):
    h = 8 * d_lat
    if verbose:
        print(f"\n=== d_lat={d_lat}, h={h}, n_steps={n_steps} ===")

    rng_data  = np.random.default_rng(seed)
    rng_train = np.random.default_rng(seed + 1)

    # data
    train_x, _, means, Q, _ = generate_data(
        n_train, d_int, d_lat, sigma_perp=sigma_perp, sigma_sig=sigma_sig,
        scale=scale, k=k, seed=seed)
    test_x,  _, _,    _,  _ = generate_data(
        2048, d_int, d_lat, sigma_perp=sigma_perp, sigma_sig=sigma_sig,
        scale=scale, k=k, seed=9999)
    train_x  = train_x.astype(np.float32)
    test_x   = test_x.astype(np.float32)
    means    = means.astype(np.float32)

    # precompute true score on test set at t_eval
    t_eval_arr = np.full(test_x.shape[0], t_eval, dtype=np.float32)
    e_t   = math.exp(-t_eval)
    dt    = 1 - math.exp(-2*t_eval)
    test_noise = rng_data.standard_normal(test_x.shape).astype(np.float32)
    test_xt  = (e_t * test_x + math.sqrt(dt) * test_noise).astype(np.float32)
    true_s   = analytic_score_full(test_xt, t_eval, means, Q,
                                   sigma_sig=sigma_sig, sigma_perp=sigma_perp,
                                   d_int=d_int).astype(np.float32)

    # Var(s*_null) per dim at t_eval (theoretical)
    e2t  = math.exp(-2*t_eval)
    var_s_null = 1.0 / (e2t*sigma_perp**2 + dt)
    var_s_sig  = 1.0 / (e2t*(sigma_sig**2 + scale**2/d_int) + dt)

    model = MLP(d_lat, h, rng=np.random.default_rng(seed+2))

    history = []
    t0 = time.time()

    for step in range(1, n_steps + 1):
        # sample batch
        idx = rng_train.integers(0, n_train, size=batch_size)
        xb  = train_x[idx]

        # sample t
        t_batch = (rng_train.random(batch_size) * (t_max - t_min) + t_min).astype(np.float32)

        # forward-noise
        e_t_b  = np.exp(-t_batch).astype(np.float32)[:, None]
        dt_b   = (1 - np.exp(-2*t_batch)).astype(np.float32)[:, None]
        noise  = rng_train.standard_normal(xb.shape).astype(np.float32)
        xt_b   = e_t_b * xb + np.sqrt(dt_b) * noise

        # analytic score target (too slow to compute per step; use denoising proxy)
        # Denoising target: s* ≈ -noise / sqrt(dt)  (per-sample approximation)
        # This is an unbiased estimator of the score for any distribution.
        target = (-noise / np.sqrt(dt_b + 1e-8))

        loss, grads = model.backward(xt_b, t_batch, target)
        model.adam_step(grads, lr=lr)

        if step % eval_every == 0 or step == 1:
            # score error on test set
            se = model.score_error_per_dim(test_xt, t_eval, true_s)
            sm = model.sig_mass(Q, d_int)
            ar = model.hidden_rank(test_xt, t_eval_arr)
            ad = model.activation_density(test_xt, t_eval_arr)

            # per-dim null score error
            null_dims   = list(range(d_int, d_lat)) if d_lat > d_int else []
            t_batch_te  = t_eval_arr
            pred_s = model.forward(test_xt, t_batch_te)
            if null_dims:
                null_err_per_dim = float(np.mean(
                    (pred_s[:, null_dims] - true_s[:, null_dims])**2))
                null_ratio = null_err_per_dim / var_s_null
            else:
                null_err_per_dim = 0.0
                null_ratio = 0.0

            row = dict(step=step, loss=float(loss), score_error=se,
                       sig_mass=sm, hidden_rank=ar, act_density=ad,
                       null_err_per_dim=null_err_per_dim,
                       null_ratio=null_ratio,
                       var_s_null=var_s_null, var_s_sig=var_s_sig)
            history.append(row)
            if verbose:
                elapsed = time.time() - t0
                print(f"  step={step:6d} loss={loss:.4f} se={se:.4f} "
                      f"sig_mass={sm:.3f} rank={ar:.1f} "
                      f"null_ratio={null_ratio:.2f}  [{elapsed:.0f}s]")

    return {
        'd_lat': d_lat, 'h': h, 'd_int': d_int,
        'sigma_perp': sigma_perp, 'n_steps': n_steps,
        'var_s_null': var_s_null, 'var_s_sig': var_s_sig,
        'history': history,
        'final': history[-1] if history else {},
    }

# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--d_lats', nargs='+', type=int, default=[5, 8, 15, 40])
    ap.add_argument('--n_steps', type=int, default=50000)
    ap.add_argument('--seeds',   nargs='+', type=int, default=[42])
    ap.add_argument('--out', default='probe_results.json')
    args = ap.parse_args()

    all_results = []
    for seed in args.seeds:
        for d_lat in args.d_lats:
            res = run_experiment(d_lat, n_steps=args.n_steps, seed=seed)
            all_results.append(res)

    out_path = args.out
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults written to {out_path}")
