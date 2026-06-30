"""Generate the clean four-bulk figure suite.

This is the curated set intended for paper consideration:

- Exp 2 main GMM d_latent sweep.
- Exp 3 main GMM d_intrinsic sweep.
- Exp 2 Gaussian robustness over sigma_noise.
- Exp 2 Gaussian robustness over sigma_signal.
- Exp 3 Gaussian robustness over sigma_noise.
- Exp 3 Gaussian robustness over sigma_signal.

All runs use tanh RFNN features and MC=500 for U. The output is split
into two width choices:

- p_n_plus_d_plus_r: p = d_latent + n + rank_null with rank_null=300.
- p64: p = 64 * d_latent.
"""

import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - optional GPU acceleration
    torch = None


OUT_ROOT = Path("clean figures")


@dataclass(frozen=True)
class Config:
    name: str
    data_kind: str
    sweep: str
    row_label: str = ""
    row_value: float | None = None
    n: int = 500
    k: int = 10
    d_intrinsic: int = 5
    d_latent: int = 20
    sigma_signal: float = 1.0
    sigma_noise: float = 0.5
    center_scale: float = 3.0
    t: float = 0.01
    mc_samples: int = 500
    rank_null: int = 300
    p_mode: str = "n_plus_d_plus_r"
    p_ratio: int = 64
    p_fixed: int = 1800
    null_energy: float | None = None
    seed: int = 42


def random_rotation(rng, d):
    q, r = np.linalg.qr(rng.standard_normal((d, d)))
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1
    return q * signs


def effective_sigma_noise(cfg, dint, dlat):
    null_dims = dlat - dint
    if cfg.null_energy is None or null_dims <= 0:
        return cfg.sigma_noise
    return math.sqrt(cfg.null_energy / null_dims)


def sample_gaussian(rng, cfg, dint, dlat):
    x = np.zeros((cfg.n, dlat), dtype=np.float64)
    x[:, :dint] = rng.standard_normal((cfg.n, dint)) * cfg.sigma_signal
    if dlat > dint:
        sigma_noise = effective_sigma_noise(cfg, dint, dlat)
        x[:, dint:] = rng.standard_normal((cfg.n, dlat - dint)) * sigma_noise
    return x


def sample_gmm(rng, cfg, dint, dlat):
    raw = rng.standard_normal((cfg.k, dint))
    raw = raw / np.linalg.norm(raw, axis=1, keepdims=True).clip(1e-12) * cfg.center_scale
    labels = rng.integers(0, cfg.k, size=cfg.n)
    x = np.zeros((cfg.n, dlat), dtype=np.float64)
    x[:, :dint] = raw[labels] + rng.standard_normal((cfg.n, dint)) * cfg.sigma_signal
    if dlat > dint:
        sigma_noise = effective_sigma_noise(cfg, dint, dlat)
        x[:, dint:] = rng.standard_normal((cfg.n, dlat - dint)) * sigma_noise
    return x


def sample_data(cfg, dint, dlat, seed):
    rng = np.random.default_rng(seed)
    if cfg.data_kind == "gaussian":
        x = sample_gaussian(rng, cfg, dint, dlat)
    elif cfg.data_kind == "gmm":
        x = sample_gmm(rng, cfg, dint, dlat)
    else:
        raise ValueError(cfg.data_kind)
    q = random_rotation(rng, dlat)
    return x @ q.T


def compute_eigs(cfg, dint, dlat, seed):
    if torch is not None and torch.cuda.is_available():
        return compute_eigs_torch(cfg, dint, dlat, seed)
    return compute_eigs_numpy(cfg, dint, dlat, seed)


def feature_width(cfg, dlat):
    if cfg.p_mode == "n_plus_d_plus_r":
        return dlat + cfg.n + cfg.rank_null
    if cfg.p_mode == "p64":
        return cfg.p_ratio * dlat
    if cfg.p_mode == "fixed":
        return cfg.p_fixed
    raise ValueError(cfg.p_mode)


def compute_eigs_numpy(cfg, dint, dlat, seed):
    rng = np.random.default_rng(seed)
    x = sample_data(cfg, dint, dlat, seed + 1000)
    p = feature_width(cfg, dlat)
    w = rng.standard_normal((p, dlat)) / math.sqrt(dlat)
    delta_t = 1 - math.exp(-2 * cfg.t)
    e_neg_t = math.exp(-cfg.t)
    U = np.zeros((p, p), dtype=np.float64)
    for _ in range(cfg.mc_samples):
        noise = rng.standard_normal((cfg.n, dlat))
        x_t = e_neg_t * x + math.sqrt(delta_t) * noise
        phi = np.tanh(x_t @ w.T)
        U += phi.T @ phi / cfg.n
    U /= cfg.mc_samples
    return np.sort(np.linalg.eigvalsh(U))[::-1]


def compute_eigs_torch(cfg, dint, dlat, seed):
    rng = np.random.default_rng(seed)
    x_np = sample_data(cfg, dint, dlat, seed + 1000)
    p = feature_width(cfg, dlat)
    w_np = rng.standard_normal((p, dlat)) / math.sqrt(dlat)

    device = torch.device("cuda:0")
    dtype = torch.float32 if cfg.p_mode == "p64" else torch.float64
    gen = torch.Generator(device=device)
    gen.manual_seed(seed + 2000)

    x = torch.as_tensor(x_np, dtype=dtype, device=device)
    w = torch.as_tensor(w_np, dtype=dtype, device=device)
    delta_t = 1 - math.exp(-2 * cfg.t)
    e_neg_t = math.exp(-cfg.t)
    sqrt_delta_t = math.sqrt(delta_t)

    U = torch.zeros((p, p), dtype=dtype, device=device)
    for _ in range(cfg.mc_samples):
        noise = torch.randn((cfg.n, dlat), dtype=dtype, device=device,
                            generator=gen)
        x_t = e_neg_t * x + sqrt_delta_t * noise
        phi = torch.tanh(x_t @ w.T)
        U.add_(phi.T @ phi, alpha=1.0 / cfg.n)
    U.div_(cfg.mc_samples)
    if cfg.p_mode == "p64" and p >= 30000:
        import scipy.linalg
        print(f"copy U.T to CPU for scipy eigvalsh p={p}", flush=True)
        U_cpu_base = U.T.contiguous().detach().cpu().numpy()
        del U
        if device.type == "cuda":
            torch.cuda.empty_cache()
        eigs = scipy.linalg.eigvalsh(
            U_cpu_base.T,
            overwrite_a=True,
            check_finite=False,
            driver="evr",
        )[::-1]
        del U_cpu_base
        return eigs
    eigs = torch.linalg.eigvalsh(U).flip(0)
    return eigs.detach().cpu().numpy()


def exp2_cells():
    return [(5, dlat) for dlat in [5, 10, 20, 40, 100, 200, 500, 1000]]


def exp2_small_cells():
    return [(5, dlat) for dlat in [5, 10, 20, 40]]


def exp3_cells():
    return [(dint, 20) for dint in [2, 5, 8, 12, 16, 20]]


def exp3_small_cells():
    return [(dint, 20) for dint in [2, 5, 12, 20]]


def region_logs(eigs, dint, dlat, n):
    p = len(eigs)
    regions = [
        ("rank-null", eigs[min(dlat + n, p):], "#5f3dc4"),
        ("sample", eigs[dlat:min(dlat + n, p)], "#1971c2"),
        ("noise-dim", eigs[dint:dlat], "#2f9e44"),
        ("signal", eigs[:dint], "#e03131"),
    ]
    out = []
    for label, vals, color in regions:
        if len(vals) == 0:
            continue
        out.append((label, np.log10(np.maximum(vals, 1e-30)), color))
    return out


def plot_grid(out_dir, title, row_configs, cells, cell_title_fn):
    n_rows, n_cols = len(row_configs), len(cells)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.2 * n_cols, max(3.8 * n_rows, 3.8)),
        sharex=False,
        sharey=False,
        squeeze=False,
    )

    for r, cfg in enumerate(row_configs):
        for c, (dint, dlat) in enumerate(cells):
            ax = axes[r, c]
            eig_path = out_dir / f"{cfg.name}_di{dint}_d{dlat}" / "eigenvalues.npy"
            eigs = np.sort(np.load(eig_path))[::-1]
            regions = region_logs(eigs, dint, dlat, cfg.n)
            panel_vals = np.concatenate([vals for _label, vals, _color in regions])
            lo, hi = float(panel_vals.min()), float(panel_vals.max())
            pad = 0.04 * max(hi - lo, 1e-6)
            bins = np.linspace(lo - pad, hi + pad, 70)
            for label, vals, color in regions:
                alpha = 0.70 if label == "signal" else 0.34
                linewidth = 1.8 if label == "signal" else 0.25
                zorder = 5 if label == "signal" else 2
                ax.hist(
                    vals, bins=bins, histtype="stepfilled", alpha=alpha,
                    color=color, edgecolor=color, linewidth=linewidth,
                    zorder=zorder,
                )
                if label == "signal":
                    ax.hist(vals, bins=bins, histtype="step", color=color,
                            linewidth=2.3, zorder=zorder + 1)
                ax.axvline(float(np.median(vals)), color=color, lw=0.8,
                           ls="--", alpha=0.85)
            ax.set_yscale("log")
            ax.grid(True, which="both", alpha=0.16, lw=0.4)
            if r == 0:
                ax.set_title(cell_title_fn(dint, dlat), pad=3)
            if c == 0:
                row_text = cfg.row_label
                if cfg.row_value is not None:
                    row_text += f"={cfg.row_value:g}"
                ax.set_ylabel(f"{row_text}\ncount")
            if r == n_rows - 1:
                ax.set_xlabel(r"$\log_{10}\lambda_i(U)$")
            ax.text(0.03, 0.93, rf"$p={len(eigs)}$", transform=ax.transAxes,
                    fontsize=7, ha="left", va="top")

    from matplotlib.patches import Patch
    legend_items = [
        Patch(facecolor="#e03131", alpha=0.45, label="signal"),
        Patch(facecolor="#2f9e44", alpha=0.34, label="noise-dim"),
        Patch(facecolor="#1971c2", alpha=0.34, label="sample"),
        Patch(facecolor="#5f3dc4", alpha=0.34, label="rank-null"),
    ]
    fig.suptitle(title, y=1.01)
    fig.legend(handles=legend_items, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.012))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out_dir / "hist_by_index_region.png", dpi=180)
    fig.savefig(out_dir / "hist_by_index_region.pdf")
    plt.close(fig)


def run_suite(root, name, row_configs, cells, title, cell_title_fn, description):
    out_dir = root / name
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, cfg in enumerate(row_configs):
        for j, (dint, dlat) in enumerate(cells):
            run_dir = out_dir / f"{cfg.name}_di{dint}_d{dlat}"
            run_dir.mkdir(exist_ok=True)
            eig_path = run_dir / "eigenvalues.npy"
            if not eig_path.exists():
                eigs = compute_eigs(cfg, dint, dlat, cfg.seed + 1009 * i + 97 * j)
                np.save(eig_path, eigs)
            (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
            print(f"{name}: {run_dir.name}", flush=True)
    plot_grid(out_dir, title, row_configs, cells, cell_title_fn)
    (out_dir / "README.md").write_text(description)


def run_all(root, p_mode, width_text, title_width_text,
            energy_fixed=False, p_fixed=1800):
    root.mkdir(parents=True, exist_ok=True)
    main_null_energy = (20 - 5) * 0.5 ** 2 if energy_fixed else None
    gauss_null_energy = (20 - 5) * 0.3 ** 2 if energy_fixed else None
    energy_text = (
        " Total null variance is held fixed by rescaling "
        "`sigma_noise^2 = E_null/(d_latent-d_intrinsic)`."
        if energy_fixed else ""
    )
    base_gmm = Config(name="gmm_main", data_kind="gmm", sweep="exp2",
                      row_label="GMM", sigma_noise=0.5, center_scale=3.0,
                      p_mode=p_mode, p_fixed=p_fixed,
                      null_energy=main_null_energy)
    base_gauss = Config(name="gaussian", data_kind="gaussian", sweep="exp2",
                        row_label="Gaussian", sigma_noise=0.3, sigma_signal=1.0,
                        center_scale=0.0, p_mode=p_mode, p_fixed=p_fixed,
                        null_energy=gauss_null_energy)

    run_suite(
        root,
        "exp2_main_gmm",
        [base_gmm],
        exp2_cells(),
        rf"Exp 2 main: GMM $d_{{lat}}$ sweep, {title_width_text}",
        lambda _dint, dlat: rf"$d_{{lat}}={dlat}$",
        f"""# Exp 2 main GMM

Fixed `d_intrinsic=5`, sweep `d_latent`.

Data: anisotropic GMM, `sigma_noise=0.5`, `sigma_signal=1.0`,
`center_scale=3.0`. RFNN uses tanh features and `MC=500`.
Width: `{width_text}`.{energy_text}
""",
    )

    run_suite(
        root,
        "exp3_main_gmm",
        [replace(base_gmm, sweep="exp3")],
        exp3_cells(),
        rf"Exp 3 main: GMM $d_{{int}}$ sweep, {title_width_text}",
        lambda dint, _dlat: rf"$d_{{int}}={dint}$",
        f"""# Exp 3 main GMM

Fixed `d_latent=20`, sweep `d_intrinsic`.

Data: anisotropic GMM, `sigma_noise=0.5`, `sigma_signal=1.0`,
`center_scale=3.0`. RFNN uses tanh features and `MC=500`.
Width: `{width_text}`.{energy_text}
""",
    )

    sigma_rows = [
        replace(base_gauss, name=f"gaussian_sn{sn:g}", sigma_noise=sn,
                null_energy=((20 - 5) * sn ** 2 if energy_fixed else None),
                row_label=(r"$E_{null}^{1/2}@20$" if energy_fixed else r"$\sigma_\perp$"),
                row_value=sn)
        for sn in [0.1, 0.3, 0.5]
    ]
    run_suite(
        root,
        "exp2_robust_gaussian_sigma",
        sigma_rows,
        exp2_small_cells(),
        rf"Exp 2 robustness: Gaussian $\sigma_\perp$ sweep, {title_width_text}",
        lambda _dint, dlat: rf"$d_{{lat}}={dlat}$",
        f"""# Exp 2 Gaussian sigma-noise robustness

Fixed `d_intrinsic=5`, sweep `d_latent`.

Data: pure anisotropic Gaussian. Rows vary `sigma_noise`.
RFNN uses tanh features and `MC=500`.
Width: `{width_text}`.{energy_text}
""",
    )

    sig_rows = [
        replace(base_gauss, name=f"gaussian_sig{ss:g}", sigma_signal=ss,
                sigma_noise=0.3, row_label=r"$\sigma_{signal}$", row_value=ss)
        for ss in [0.5, 1.0, 1.5]
    ]
    run_suite(
        root,
        "exp2_robust_gaussian_sigscale",
        sig_rows,
        exp2_small_cells(),
        rf"Exp 2 robustness: Gaussian signal-scale sweep, {title_width_text}",
        lambda _dint, dlat: rf"$d_{{lat}}={dlat}$",
        f"""# Exp 2 Gaussian signal-scale robustness

Fixed `d_intrinsic=5`, sweep `d_latent`.

Data: pure anisotropic Gaussian. Rows vary `sigma_signal` at
`sigma_noise=0.3`. RFNN uses tanh features and `MC=500`.
Width: `{width_text}`.{energy_text}
""",
    )

    run_suite(
        root,
        "exp3_robust_gaussian_sigma",
        [replace(cfg, sweep="exp3") for cfg in sigma_rows],
        exp3_small_cells(),
        rf"Exp 3 robustness: Gaussian $\sigma_\perp$ sweep, {title_width_text}",
        lambda dint, _dlat: rf"$d_{{int}}={dint}$",
        f"""# Exp 3 Gaussian sigma-noise robustness

Fixed `d_latent=20`, sweep `d_intrinsic`.

Data: pure anisotropic Gaussian. Rows vary `sigma_noise`.
RFNN uses tanh features and `MC=500`.
Width: `{width_text}`.{energy_text}
""",
    )

    run_suite(
        root,
        "exp3_robust_gaussian_sigscale",
        [replace(cfg, sweep="exp3") for cfg in sig_rows],
        exp3_small_cells(),
        rf"Exp 3 robustness: Gaussian signal-scale sweep, {title_width_text}",
        lambda dint, _dlat: rf"$d_{{int}}={dint}$",
        f"""# Exp 3 Gaussian signal-scale robustness

Fixed `d_latent=20`, sweep `d_intrinsic`.

Data: pure anisotropic Gaussian. Rows vary `sigma_signal` at
`sigma_noise=0.3`. RFNN uses tanh features and `MC=500`.
Width: `{width_text}`.{energy_text}
""",
    )

    (root / "README.md").write_text(f"""# Clean figures: {width_text}

Curated candidate figures for the paper.

All figures use RFNN tanh features and `MC=500`.
Width: `{width_text}`.{energy_text}

Folders:

- `exp2_main_gmm`: main GMM Exp-2-style `d_latent` sweep.
- `exp3_main_gmm`: main GMM Exp-3-style `d_intrinsic` sweep.
- `exp2_robust_gaussian_sigma`: Gaussian sigma-noise robustness for Exp 2.
- `exp2_robust_gaussian_sigscale`: Gaussian signal-scale robustness for Exp 2.
- `exp3_robust_gaussian_sigma`: Gaussian sigma-noise robustness for Exp 3.
- `exp3_robust_gaussian_sigscale`: Gaussian signal-scale robustness for Exp 3.

Color/index convention:

- signal: `eigs[:d_intrinsic]`
- noise-dim: `eigs[d_intrinsic:d_latent]`
- sample: `eigs[d_latent:d_latent+n]`
- rank-null: `eigs[d_latent+n:]`

Histogram bins span the full min/max range per panel.
""")


def main():
    OUT_ROOT.mkdir(exist_ok=True)
    run_all(
        OUT_ROOT / "p_n_plus_d_plus_r",
        "n_plus_d_plus_r",
        "p=d_latent+n+300",
        r"$p=d_{lat}+n+300$",
    )
    run_all(
        OUT_ROOT / "p64",
        "p64",
        "p=64*d_latent",
        r"$p=64d_{lat}$",
    )
    run_all(
        OUT_ROOT / "p_fixed_energy_fixed",
        "fixed",
        "p=1800, fixed total null variance",
        r"$p=1800$, fixed $E_{null}$",
        energy_fixed=True,
        p_fixed=1800,
    )
    (OUT_ROOT / "README.md").write_text("""# Clean figures

Curated candidate figures for the paper.

There are three complete versions:

- `p_n_plus_d_plus_r`: controlled-width figures with `p=d_latent+n+300`.
- `p64`: fixed-ratio figures with `p=64*d_latent`.
- `p_fixed_energy_fixed`: fixed-capacity/energy ablation with `p=1800`
  and `sigma_noise^2 = E_null/(d_latent-d_intrinsic)`.

Each width folder contains the same six experiment families:

- `exp2_main_gmm`
- `exp3_main_gmm`
- `exp2_robust_gaussian_sigma`
- `exp2_robust_gaussian_sigscale`
- `exp3_robust_gaussian_sigma`
- `exp3_robust_gaussian_sigscale`

Each run stores the full dense eigenspectrum in `eigenvalues.npy`.
""")


if __name__ == "__main__":
    main()
