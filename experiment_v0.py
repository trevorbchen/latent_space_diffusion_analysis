"""
Experiment v0: Does latent dimension affect memorization timing?

Core question: For a d_intrinsic-dimensional manifold, how does d_latent
affect when generalization emerges (tau_gen) vs when memorization begins (tau_mem)?

Setup:
  - Synthetic data: points on S^4 (5D hypersphere) embedded in R^20
  - Linear autoencoder with varying bottleneck d_latent
  - Simple MLP denoiser (DDPM-style)
  - Sweep d_latent across [2, 4, 8, 12, 20] to span below/at/above d_intrinsic=4
"""

import math
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    # Data — Gaussian mixture
    n_clusters: int = 10            # number of Gaussian clusters
    cluster_dim: int = 5            # cluster centers live in this many dims (≈ d_intrinsic)
    ambient_dim: int = 20           # full ambient dimension
    cluster_std: float = 0.1        # within-cluster spread (small = tight clusters, large = merged)
    n_samples: int = 500

    # Encoder
    encoder_type: str = "vae"       # "linear" or "vae"
    encoder_hidden: int = 512
    encoder_lr: float = 1e-3
    encoder_steps: int = 10000
    kl_weight: float = 1e-6         # tiny KL to spread latent usage

    # Diffusion
    T: int = 100                    # number of diffusion timesteps (toy data, don't need 1000)
    beta_start: float = 1e-4        # Bonnaire schedule
    beta_end: float = 0.02

    # Denoiser MLP
    hidden_dim: int = 1024          # big model to enable memorization
    n_layers: int = 5

    # Training
    lr: float = 3e-4
    batch_size: int = 500           # full batch (= n_samples)
    n_steps: int = 200000
    eval_every: int = 20000
    n_eval_samples: int = 100

    # Memorization detection
    mem_tau: float = 0.333          # NN ratio threshold (Bonnaire: 1/3)

    # Sweep — d_intrinsic=4, so ratios 0.25x through 5x
    d_latents: tuple = (1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 18, 20, 25, 30, 35, 40)
    seed: int = 42

    # Output
    results_dir: str = "results_v3"


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def get_device():
    try:
        import torch_xla.core.xla_model as xm
        return xm.xla_device()
    except ImportError:
        pass
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def generate_gaussian_mixture(n, n_clusters, cluster_dim, ambient_dim, cluster_std=0.1, seed=42):
    """Gaussian mixture: k clusters whose centers span a cluster_dim-dimensional subspace of R^ambient_dim.

    d_intrinsic ≈ cluster_dim (the subspace the cluster centers live in).
    Each point also has isotropic noise in all ambient_dim dimensions.
    """
    rng = torch.Generator().manual_seed(seed)

    # Cluster centers: random points in a cluster_dim-dimensional subspace of R^ambient_dim
    # Generate centers in R^cluster_dim, then embed via random rotation
    raw_centers = torch.randn(n_clusters, cluster_dim, generator=rng)
    random_mat = torch.randn(ambient_dim, ambient_dim, generator=rng)
    Q, _ = torch.linalg.qr(random_mat)
    centers_padded = F.pad(raw_centers, (0, ambient_dim - cluster_dim))
    centers = centers_padded @ Q.T  # (n_clusters, ambient_dim)

    # Assign points to clusters uniformly
    assignments = torch.randint(0, n_clusters, (n,), generator=rng)

    # Sample points: center + isotropic noise in all ambient dims
    data = centers[assignments] + cluster_std * torch.randn(n, ambient_dim, generator=rng)

    return data


# ---------------------------------------------------------------------------
# Encoders
# ---------------------------------------------------------------------------

class LinearAE(nn.Module):
    def __init__(self, input_dim, d_latent):
        super().__init__()
        self.encoder = nn.Linear(input_dim, d_latent, bias=False)
        self.decoder = nn.Linear(d_latent, input_dim, bias=False)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        return self.decode(self.encode(x))


class VAE(nn.Module):
    def __init__(self, input_dim, d_latent, hidden_dim=512):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.enc_mu = nn.Linear(hidden_dim, d_latent)
        self.enc_logvar = nn.Linear(hidden_dim, d_latent)
        self.dec = nn.Sequential(
            nn.Linear(d_latent, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x):
        h = self.enc(x)
        return self.enc_mu(h)

    def encode_stochastic(self, x):
        h = self.enc(x)
        mu = self.enc_mu(h)
        logvar = self.enc_logvar(h)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        return z, mu, logvar

    def decode(self, z):
        return self.dec(z)

    def forward(self, x):
        z, mu, logvar = self.encode_stochastic(x)
        return self.decode(z), mu, logvar


def _opt_step(optimizer, device):
    optimizer.step()
    if 'xla' in str(device):
        import torch_xla.core.xla_model as xm
        xm.mark_step()


def train_encoder(data, d_latent, config, device):
    """Train encoder (linear or VAE), return frozen model."""
    data_dev = data.to(device)

    if config.encoder_type == "linear":
        model = LinearAE(config.ambient_dim, d_latent).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.encoder_lr)
        for step in range(config.encoder_steps):
            loss = F.mse_loss(model(data_dev), data_dev)
            optimizer.zero_grad()
            loss.backward()
            _opt_step(optimizer, device)
    else:
        model = VAE(config.ambient_dim, d_latent, config.encoder_hidden).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.encoder_lr)
        for step in range(config.encoder_steps):
            recon, mu, logvar = model(data_dev)
            recon_loss = F.mse_loss(recon, data_dev)
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + config.kl_weight * kl_loss
            optimizer.zero_grad()
            loss.backward()
            _opt_step(optimizer, device)

    # Freeze
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    with torch.no_grad():
        final_mse = F.mse_loss(model.decode(model.encode(data_dev)), data_dev).item()
    return model, final_mse


# ---------------------------------------------------------------------------
# Diffusion (DDPM)
# ---------------------------------------------------------------------------

def linear_beta_schedule(T, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, T)


def precompute_diffusion_params(betas):
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    sqrt_recip_alphas = 1.0 / torch.sqrt(alphas)
    posterior_variance = betas * (1.0 - torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])) / (1.0 - alphas_cumprod)
    return {
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
        "sqrt_alphas_cumprod": sqrt_alphas_cumprod,
        "sqrt_one_minus_alphas_cumprod": sqrt_one_minus_alphas_cumprod,
        "sqrt_recip_alphas": sqrt_recip_alphas,
        "posterior_variance": posterior_variance,
    }


def q_sample(x0, t, diff_params, noise=None):
    """Forward diffusion: add noise to x0 at timestep t."""
    if noise is None:
        noise = torch.randn_like(x0)
    sqrt_alpha = diff_params["sqrt_alphas_cumprod"][t].unsqueeze(-1)
    sqrt_one_minus = diff_params["sqrt_one_minus_alphas_cumprod"][t].unsqueeze(-1)
    return sqrt_alpha * x0 + sqrt_one_minus * noise, noise


@torch.no_grad()
def p_sample_loop(model, shape, diff_params, device):
    """DDPM reverse sampling."""
    x = torch.randn(shape, device=device)
    T = len(diff_params["betas"])

    for t_idx in reversed(range(T)):
        t_batch = torch.full((shape[0],), t_idx, device=device, dtype=torch.long)
        predicted_noise = model(x, t_batch)

        beta = diff_params["betas"][t_idx]
        sqrt_recip_alpha = diff_params["sqrt_recip_alphas"][t_idx]
        sqrt_one_minus = diff_params["sqrt_one_minus_alphas_cumprod"][t_idx]

        # Mean of p(x_{t-1} | x_t)
        mean = sqrt_recip_alpha * (x - beta / sqrt_one_minus * predicted_noise)

        if t_idx > 0:
            noise = torch.randn_like(x)
            sigma = torch.sqrt(diff_params["posterior_variance"][t_idx])
            x = mean + sigma * noise
        else:
            x = mean

    return x


# ---------------------------------------------------------------------------
# MLP Denoiser
# ---------------------------------------------------------------------------

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device, dtype=torch.float32) * -emb)
        emb = t.float().unsqueeze(-1) * emb.unsqueeze(0)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class MLPDenoiser(nn.Module):
    def __init__(self, d_latent, hidden_dim=256, n_layers=3, time_emb_dim=64):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, hidden_dim),
            nn.ReLU(),
        )

        # First layer takes latent + time embedding
        layers = [nn.Linear(d_latent + hidden_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_layers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, d_latent))
        self.net = nn.Sequential(*layers)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        inp = torch.cat([x, t_emb], dim=-1)
        return self.net(inp)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_nn_ratios(generated, training_data):
    """Compute nearest-neighbor ratio r = d(x,NN1)/d(x,NN2) for each generated sample."""
    # Pairwise distances: generated (m) vs training (n)
    dists = torch.cdist(generated, training_data)  # (m, n)
    # Top-2 nearest
    top2 = torch.topk(dists, k=2, dim=1, largest=False)
    d1 = top2.values[:, 0]
    d2 = top2.values[:, 1]
    ratios = d1 / (d2 + 1e-10)
    return ratios


def memorization_ratio(generated, training_data, tau=0.5):
    """Fraction of generated samples with NN ratio < tau."""
    ratios = compute_nn_ratios(generated, training_data)
    return (ratios < tau).float().mean().item(), ratios


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_diffusion(latents, d_latent, config, diff_params, device,
                    encoder=None, original_data=None):
    """Train MLP denoiser on encoded latents. Return metrics history.

    encoder and original_data are used to decode generated samples back to
    data space for metrics computation.
    """
    # Precompute training data stats for generation quality check
    # Use NN distance within training set as reference for "plausible" generated points
    train_dists = torch.cdist(original_data, original_data)
    train_dists.fill_diagonal_(float('inf'))
    train_nn_dist = train_dists.min(dim=1).values.mean().item()  # avg NN dist in training data
    model = MLPDenoiser(d_latent, config.hidden_dim, config.n_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    # Normalize latents to zero mean, unit variance per dimension
    latent_mean = latents.mean(dim=0)
    latent_std = latents.std(dim=0).clamp(min=1e-6)
    latents_norm = (latents - latent_mean) / latent_std

    # Move diffusion params to device
    dp = {k: v.to(device) for k, v in diff_params.items()}
    latents_dev = latents_norm.to(device)
    latent_mean_dev = latent_mean.to(device)
    latent_std_dev = latent_std.to(device)
    n = len(latents_dev)

    metrics_history = []

    for step in range(1, config.n_steps + 1):
        model.train()

        # Sample batch
        idx = torch.randint(0, n, (min(config.batch_size, n),))
        x0 = latents_dev[idx]

        # Sample random timesteps
        t = torch.randint(0, config.T, (len(x0),), device=device)

        # Forward diffusion
        x_noisy, noise = q_sample(x0, t, dp)

        # Predict noise
        pred_noise = model(x_noisy, t)
        loss = F.mse_loss(pred_noise, noise)

        optimizer.zero_grad()
        loss.backward()
        _opt_step(optimizer, device)

        # Evaluate — copy to CPU to avoid XLA graph corruption
        if step % config.eval_every == 0 or step == 1:
            model.eval()
            with torch.no_grad():
                cpu_model = MLPDenoiser(d_latent, config.hidden_dim, config.n_layers)
                cpu_model.load_state_dict({k: v.cpu() for k, v in model.state_dict().items()})
                cpu_model.eval()
                cpu_dp = {k: v.cpu() for k, v in diff_params.items()}

                gen_latents = p_sample_loop(
                    cpu_model, (config.n_eval_samples, d_latent), cpu_dp, torch.device('cpu')
                )
                # Denormalize back to original latent scale
                gen_latents = gen_latents * latent_std.cpu() + latent_mean.cpu()
                # Decode on CPU
                if config.encoder_type == "vae":
                    cpu_enc = VAE(config.ambient_dim, d_latent, config.encoder_hidden)
                else:
                    cpu_enc = LinearAE(config.ambient_dim, d_latent)
                cpu_enc.load_state_dict({k: v.cpu() for k, v in encoder.state_dict().items()})
                cpu_enc.eval()
                gen_decoded = cpu_enc.decode(gen_latents)

                # 1) Generation quality: is each generated point near the training distribution?
                #    "on manifold" = within 3x avg training NN distance of some training point
                gen_to_train = torch.cdist(gen_decoded, original_data)
                gen_nn_dist = gen_to_train.min(dim=1).values
                frac_on_manifold = (gen_nn_dist < 3 * train_nn_dist).float().mean().item()
                norm_mean = gen_decoded.norm(dim=1).mean().item()
                norm_std = gen_decoded.norm(dim=1).std().item()

                # 2) Memorization: NN ratio in data space
                mem_ratio, nn_ratios = memorization_ratio(
                    gen_decoded, original_data, config.mem_tau
                )

                metrics = {
                    "step": step,
                    "loss": loss.item(),
                    "mem_ratio": mem_ratio,
                    "nn_ratio_mean": nn_ratios.mean().item(),
                    "nn_ratio_median": nn_ratios.median().item(),
                    "norm_mean": norm_mean,
                    "norm_std": norm_std,
                    "frac_on_manifold": frac_on_manifold,
                }
                metrics_history.append(metrics)

                print(f"  step {step:6d} | loss {loss.item():.4f} | "
                      f"on_manifold {frac_on_manifold:.2f} | "
                      f"mem {mem_ratio:.3f} | "
                      f"norm {norm_mean:.3f}±{norm_std:.3f}")

    return metrics_history, model


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_all(all_results, config, save_dir):
    """Generate all plots from sweep results."""
    d_latents = list(all_results.keys())
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(d_latents)))
    d_intrinsic = config.cluster_dim

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f"Memorization vs Latent Dimension (d_intrinsic={d_intrinsic}, n={config.n_samples})",
        fontsize=14,
    )

    # --- Plot 1: Encoder reconstruction error vs d_latent ---
    ax = axes[0, 0]
    recon_errors = [all_results[d]["recon_mse"] for d in d_latents]
    ratios = [d / d_intrinsic for d in d_latents]
    ax.bar(range(len(d_latents)), recon_errors, color=colors, tick_label=[
        f"{d}\n({d/d_intrinsic:.1f}x)" for d in d_latents
    ])
    ax.set_xlabel("d_latent (ratio to d_intrinsic)")
    ax.set_ylabel("Reconstruction MSE")
    ax.set_title("Encoder Quality")
    ax.axvline(x=d_latents.index(d_intrinsic) if d_intrinsic in d_latents else -1,
               color="red", linestyle="--", alpha=0.5, label=f"d_intrinsic={d_intrinsic}")
    ax.legend()

    # --- Plot 2: Training loss vs step ---
    ax = axes[0, 1]
    for i, d in enumerate(d_latents):
        history = all_results[d]["metrics"]
        steps = [m["step"] for m in history]
        losses = [m["loss"] for m in history]
        ax.plot(steps, losses, color=colors[i], label=f"d={d} ({d/d_intrinsic:.1f}x)")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Denoising Loss")
    ax.set_title("Training Loss")
    ax.legend(fontsize=8)
    ax.set_yscale("log")

    # --- Plot 3: Memorization ratio vs step (THE KEY PLOT) ---
    ax = axes[0, 2]
    for i, d in enumerate(d_latents):
        history = all_results[d]["metrics"]
        steps = [m["step"] for m in history]
        mem = [m["mem_ratio"] for m in history]
        ax.plot(steps, mem, color=colors[i], label=f"d={d} ({d/d_intrinsic:.1f}x)", linewidth=2)
    ax.set_xlabel("Training Step")
    ax.set_ylabel(f"Memorization Ratio (τ={config.mem_tau})")
    ax.set_title("Memorization Over Training")
    ax.legend(fontsize=8)
    ax.axhline(y=0.1, color="gray", linestyle=":", alpha=0.5, label="τ_mem threshold (10%)")

    # --- Plot 4: Fraction on manifold (norm ≈ 1) vs step ---
    ax = axes[1, 0]
    for i, d in enumerate(d_latents):
        history = all_results[d]["metrics"]
        steps = [m["step"] for m in history]
        on_man = [m["frac_on_manifold"] for m in history]
        ax.plot(steps, on_man, color=colors[i], label=f"d={d} ({d/d_intrinsic:.1f}x)")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Fraction with |norm - 1| < 0.1")
    ax.set_title("Generation Quality (on manifold)")
    ax.legend(fontsize=8)
    ax.set_ylim(-0.05, 1.05)

    # --- Plot 5: tau_mem and tau_gen vs d_latent/d_intrinsic ---
    # Scatter only — no interpolation between points
    ax = axes[1, 1]
    tau_mems = []
    tau_gens = []
    tau_mem_found = []  # track which ones actually crossed threshold
    tau_gen_found = []

    mem_threshold = 0.1
    gen_threshold = 0.9   # tau_gen = first step where >90% of samples are on manifold
    for d in d_latents:
        history = all_results[d]["metrics"]

        # tau_mem: first step where mem_ratio > threshold
        tau_mem = None
        for m in history:
            if m["mem_ratio"] > mem_threshold:
                tau_mem = m["step"]
                break
        tau_mems.append(tau_mem)
        tau_mem_found.append(tau_mem is not None)

        # tau_gen: first step where frac_on_manifold > gen_threshold
        tau_gen = None
        for m in history:
            if m["frac_on_manifold"] > gen_threshold:
                tau_gen = m["step"]
                break
        tau_gens.append(tau_gen)
        tau_gen_found.append(tau_gen is not None)

    ratios = [d / d_intrinsic for d in d_latents]

    # Plot only points that actually crossed the threshold (no lines!)
    for i, d in enumerate(d_latents):
        r = ratios[i]
        if tau_mem_found[i]:
            ax.scatter(r, tau_mems[i], color="red", s=100, zorder=5,
                       marker="o", label="τ_mem" if i == 0 else None)
        else:
            # Arrow pointing up = never memorized in training
            ax.scatter(r, config.n_steps, color="red", s=100, zorder=5,
                       marker="^", alpha=0.4)
        if tau_gen_found[i]:
            ax.scatter(r, tau_gens[i], color="blue", s=100, zorder=5,
                       marker="s", label="τ_gen" if i == 0 else None)
        else:
            ax.scatter(r, config.n_steps, color="blue", s=100, zorder=5,
                       marker="^", alpha=0.4)

    ax.set_xlabel("d_latent / d_intrinsic")
    ax.set_ylabel("Training Step")
    ax.set_title("Timescales vs Latent Dimension")
    ax.axvline(x=1.0, color="green", linestyle="--", alpha=0.5)
    # Custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="red", markersize=10, label=f"τ_mem (mem > {mem_threshold:.0%} at τ=1/3)"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="blue", markersize=10, label="τ_gen (>90% on manifold)"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="gray", markersize=10, label="never crossed threshold"),
    ]
    ax.legend(handles=legend_elements, fontsize=7)

    # --- Plot 6: Generalization window (tau_mem - tau_gen) vs d_latent/d_intrinsic ---
    ax = axes[1, 2]
    windows = []
    window_labels = []
    window_colors_list = []
    for i, d in enumerate(d_latents):
        if tau_mem_found[i] and tau_gen_found[i]:
            windows.append(tau_mems[i] - tau_gens[i])
        elif tau_gen_found[i] and not tau_mem_found[i]:
            # Quality emerged but never memorized — best case, show as large bar
            windows.append(config.n_steps - tau_gens[i])
        else:
            windows.append(0)
        window_labels.append(f"{d}\n({d/d_intrinsic:.1f}x)")
        window_colors_list.append(colors[i])

    bars = ax.bar(range(len(d_latents)), windows, color=window_colors_list,
                  tick_label=window_labels)
    # Mark bars where tau_mem was never reached
    for i in range(len(d_latents)):
        if not tau_mem_found[i] and tau_gen_found[i]:
            bars[i].set_edgecolor("green")
            bars[i].set_linewidth(2)
            bars[i].set_hatch("//")
    ax.set_xlabel("d_latent (ratio to d_intrinsic)")
    ax.set_ylabel("τ_mem - τ_gen (steps)")
    ax.set_title("Generalization Window")
    ax.axhline(y=0, color="black", linestyle="-", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / "all_plots.png", dpi=150, bbox_inches="tight")
    plt.savefig(save_dir / "all_plots.pdf", bbox_inches="tight")
    print(f"\nPlots saved to {save_dir / 'all_plots.png'}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=None, help="Override n_samples")
    args = parser.parse_args()

    config = Config()
    if args.n is not None:
        config.n_samples = args.n
        config.batch_size = min(args.n, 500)
        config.results_dir = f"results_n{args.n}"
    device = get_device()
    print(f"Device: {device}")
    d_intrinsic = config.cluster_dim
    print(f"d_intrinsic ≈ {d_intrinsic} (cluster_dim), ambient_dim = {config.ambient_dim}")
    print(f"n_clusters = {config.n_clusters}, cluster_std = {config.cluster_std}")
    print(f"n_samples = {config.n_samples}")
    print(f"d_latent sweep: {config.d_latents}")
    print()

    # Setup results dir
    save_dir = Path(config.results_dir)
    save_dir.mkdir(exist_ok=True)

    # Fix seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Generate data
    print("Generating Gaussian mixture data...")
    data = generate_gaussian_mixture(
        config.n_samples, config.n_clusters, config.cluster_dim,
        config.ambient_dim, config.cluster_std, seed=config.seed,
    )
    print(f"  Data shape: {data.shape}")
    print(f"  Data SVD: {torch.linalg.svdvals(data).numpy().round(2)}")
    print()

    # Precompute diffusion schedule
    betas = linear_beta_schedule(config.T, config.beta_start, config.beta_end)
    diff_params = precompute_diffusion_params(betas)

    # Sweep over d_latent
    all_results = {}

    for d_latent in config.d_latents:
        ratio = d_latent / config.cluster_dim
        print(f"{'='*60}")
        print(f"d_latent = {d_latent}  (ratio to d_intrinsic: {ratio:.2f}x)")
        print(f"{'='*60}")

        # Train encoder
        print("Training encoder...")
        torch.manual_seed(config.seed)
        encoder, recon_mse = train_encoder(data, d_latent, config, device)
        print(f"  Reconstruction MSE: {recon_mse:.6f}")

        # Encode data
        with torch.no_grad():
            latents = encoder.encode(data.to(device)).cpu()
        print(f"  Latent shape: {latents.shape}")
        print(f"  Latent std: {latents.std():.4f}")

        # Train diffusion — pass encoder + original data so we decode back for metrics
        print("Training diffusion model...")
        torch.manual_seed(config.seed)
        metrics_history, model = train_diffusion(
            latents, d_latent, config, diff_params, device,
            encoder=encoder, original_data=data,
        )

        all_results[d_latent] = {
            "recon_mse": recon_mse,
            "metrics": metrics_history,
        }

        # Save per-d_latent results
        with open(save_dir / f"metrics_d{d_latent}.json", "w") as f:
            json.dump({"d_latent": d_latent, "recon_mse": recon_mse,
                       "metrics": metrics_history}, f, indent=2)

        print()

    # Generate plots
    print("Generating plots...")
    plot_all(all_results, config, save_dir)

    # Save config
    with open(save_dir / "config.json", "w") as f:
        json.dump(asdict(config), f, indent=2)

    print("\nDone!")


if __name__ == "__main__":
    main()
