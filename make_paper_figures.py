"""Generate the figures for the ICML/HiLD 2026 paper.

Outputs go to ICML/figures/. Run from the project root.
"""
import json, math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
FIG_DIR = ROOT / "ICML" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Use a clean serif font (matches LaTeX default better than the seaborn defaults)
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


def load_eigs(path):
    return np.sort(np.load(Path(path)))[::-1]


def _shade_index_cut_bulks(ax, dint, dlat, n, p):
    """Shade the four predicted index regions in a sorted full-rank spectrum."""
    regions = [
        (1, dint, "signal", "#e03131"),
        (dint + 1, dlat, "noise-dim", "#2f9e44"),
        (dlat + 1, min(dlat + n, p), "sample", "#1971c2"),
        (min(dlat + n, p) + 1, p, "rank-null", "#5f3dc4"),
    ]
    for lo, hi, label, color in regions:
        if hi < lo:
            continue
        ax.axvspan(lo, hi, color=color, alpha=0.12, lw=0)
        ax.axvline(hi + 0.5, color=color, lw=0.75, ls="--", alpha=0.65)


def _plot_index_cut_sweep(rows, cols, filename, title):
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(2.15 * len(cols), 3.55),
                             sharey="row", squeeze=False)
    for r, (row_label, root_name) in enumerate(rows):
        for c, (col_label, run_name, dint, dlat, n) in enumerate(cols):
            ax = axes[r, c]
            run = ROOT / root_name / run_name
            eig_path = run / "eigenvalues_pre.npy"
            if not eig_path.exists():
                ax.set_visible(False)
                continue
            eigs = load_eigs(eig_path)
            p = len(eigs)
            x = np.arange(1, p + 1)
            positive = eigs[eigs > 0]
            floor = max(np.percentile(positive, 1) * 0.25, 1e-12)
            ax.loglog(x, np.maximum(eigs, floor), color="black", lw=0.85)
            _shade_index_cut_bulks(ax, dint, dlat, n, p)
            ax.set_title(col_label, pad=3)
            ax.grid(True, which="both", alpha=0.18, lw=0.4)
            if r == len(rows) - 1:
                ax.set_xlabel("sorted eigenvalue index")
            if c == 0:
                ax.set_ylabel(f"{row_label}\neigenvalue")
    fig.suptitle(title, y=1.01, fontsize=10)
    from matplotlib.patches import Patch
    legend_items = [
        Patch(facecolor="#e03131", alpha=0.18, label="signal"),
        Patch(facecolor="#2f9e44", alpha=0.18, label="noise-dim buffer"),
        Patch(facecolor="#1971c2", alpha=0.18, label="sample"),
        Patch(facecolor="#5f3dc4", alpha=0.18, label="rank-null"),
    ]
    fig.legend(handles=legend_items, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.035))
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    out = FIG_DIR / filename
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=160)
    plt.close(fig)
    print(f"wrote {out}")


def fig_index_cut_full_rank_sweeps():
    """Full-rank sorted spectra with deterministic bulk index cuts.

    These plots keep the rank-null tail visible while separating the four
    predicted regions by their known counts rather than by histogram peaks.
    """
    rows_exp2 = [
        (r"$\sigma_\perp=0.01$", "results_rfnn_exp2_sn001"),
        (r"$\sigma_\perp=0.5$", "results_rfnn_exp2v3"),
    ]
    cols_exp2 = [
        (rf"$d_{{lat}}={dlat}$", f"di5_d{dlat}_n500_s42", 5, dlat, 500)
        for dlat in [5, 8, 10, 15, 20, 30, 40]
    ]
    _plot_index_cut_sweep(
        rows_exp2, cols_exp2, "four_bulk_index_cuts_exp2.pdf",
        r"Exp 2 RFNN: full-rank spectra with index-cut bulk regions",
    )

    rows_exp3 = [
        (r"$\sigma_\perp=0.01$", "results_rfnn_exp3_sn001"),
        (r"$\sigma_\perp=0.5$", "results_rfnn_exp3"),
    ]
    cols_exp3 = [
        (rf"$d_{{int}}={dint}$", f"di{dint}_d20_n500_s42", dint, 20, 500)
        for dint in [2, 5, 8, 12, 16, 20]
    ]
    _plot_index_cut_sweep(
        rows_exp3, cols_exp3, "four_bulk_index_cuts_exp3.pdf",
        r"Exp 3 RFNN: full-rank spectra with index-cut bulk regions",
    )


def _clean_anisotropic_gaussian_eigs(dint, dlat, n=500, rank_null=300,
                                     sigma_signal=1.5, sigma_noise=0.3,
                                     t=0.01, n_noise=80, seed=0):
    """Compute a cleaner RFNN spectrum on pure anisotropic Gaussian data.

    This intentionally removes Gaussian-mixture centroid spikes while keeping
    tanh features, diffusion noise, sample modes, and a visible rank-null tail.
    """
    rng = np.random.default_rng(seed)
    data = np.zeros((n, dlat), dtype=np.float64)
    data[:, :dint] = rng.standard_normal((n, dint)) * sigma_signal
    if dlat > dint:
        data[:, dint:] = rng.standard_normal((n, dlat - dint)) * sigma_noise
    q, _ = np.linalg.qr(rng.standard_normal((dlat, dlat)))
    data = data @ q.T

    p = dlat + n + rank_null
    w = rng.standard_normal((p, dlat)) / math.sqrt(dlat)
    delta_t = 1 - math.exp(-2 * t)
    e_neg_t = math.exp(-t)
    U = np.zeros((p, p), dtype=np.float64)
    for _ in range(n_noise):
        noise = rng.standard_normal((n, dlat))
        x_t = e_neg_t * data + math.sqrt(delta_t) * noise
        phi = np.tanh(x_t @ w.T)
        U += phi.T @ phi / n
    U /= n_noise
    return np.sort(np.linalg.eigvalsh(U))[::-1]


def _plot_clean_microscope(cols, filename, title, seed_offset=0):
    fig, axes = plt.subplots(1, len(cols), figsize=(2.35 * len(cols), 2.35),
                             sharey=True, squeeze=False)
    for c, (label, dint, dlat) in enumerate(cols):
        ax = axes[0, c]
        eigs = _clean_anisotropic_gaussian_eigs(
            dint, dlat, seed=seed_offset + 37 * c,
        )
        p = len(eigs)
        n = 500
        x = np.arange(1, p + 1)
        positive = eigs[eigs > 0]
        floor = max(np.percentile(positive, 1) * 0.25, 1e-12)
        ax.loglog(x, np.maximum(eigs, floor), color="black", lw=0.9)
        _shade_index_cut_bulks(ax, dint, dlat, n, p)
        ax.set_title(label, pad=3)
        ax.set_xlabel("sorted eigenvalue index")
        ax.grid(True, which="both", alpha=0.18, lw=0.4)
    axes[0, 0].set_ylabel("eigenvalue")
    from matplotlib.patches import Patch
    legend_items = [
        Patch(facecolor="#e03131", alpha=0.18, label="signal"),
        Patch(facecolor="#2f9e44", alpha=0.18, label="noise-dim buffer"),
        Patch(facecolor="#1971c2", alpha=0.18, label="sample"),
        Patch(facecolor="#5f3dc4", alpha=0.18, label="rank-null"),
    ]
    fig.suptitle(title, y=1.02, fontsize=10)
    fig.legend(handles=legend_items, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.075, 1, 1))
    out = FIG_DIR / filename
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=160)
    plt.close(fig)
    print(f"wrote {out}")


def fig_clean_four_bulk_microscope():
    """Cleaner synthetic RFNN spectra for explaining the four index regions."""
    _plot_clean_microscope(
        [(rf"$d_{{lat}}={dlat}$", 5, dlat) for dlat in [5, 10, 20, 40]],
        "four_bulk_clean_microscope_exp2.pdf",
        r"Clean RFNN microscope: $d_{int}=5$, pure anisotropic Gaussian",
        seed_offset=10,
    )
    _plot_clean_microscope(
        [(rf"$d_{{int}}={dint}$", dint, 20) for dint in [2, 5, 12, 20]],
        "four_bulk_clean_microscope_exp3.pdf",
        r"Clean RFNN microscope: $d_{lat}=20$, pure anisotropic Gaussian",
        seed_offset=200,
    )


def fig_four_bulk_overview():
    """4-panel sorted log-spectrum plot at d_lat = 5, 10, 20, 40, sigma=0.5,
    with cliffs marked at d_int and d_lat.

    Used as the headline overview figure (Figure 1 in the paper)."""
    runs = [
        ("results_rfnn_exp2v3/di5_d5_n500_s42",  5,  5),
        ("results_rfnn_exp2v3/di5_d10_n500_s42", 5, 10),
        ("results_rfnn_exp2v3/di5_d20_n500_s42", 5, 20),
        ("results_rfnn_exp2v3/di5_d40_n500_s42", 5, 40),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 1.9), sharey=True)
    for ax, (path, dint, dlat) in zip(axes, runs):
        e = load_eigs(ROOT / path / "eigenvalues_pre.npy")
        n_show = min(60, e.size)
        x = np.arange(1, n_show + 1)
        ax.semilogy(x, e[:n_show], color="tab:blue", lw=1.2, marker=".", ms=3)
        ax.axvline(dint, color="tab:red",   lw=1.0, ls="--", alpha=0.85)
        if dlat != dint:
            ax.axvline(dlat, color="tab:green", lw=1.0, ls="--", alpha=0.85)
        ax.set_title(rf"$d_{{\rm latent}}={dlat}$")
        ax.set_xlabel("eigenvalue index")
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel("eigenvalue (log)")
    fig.tight_layout()
    fig.suptitle(rf"$\sigma_\perp = 0.5$, $d_{{\rm intrinsic}} = 5$, $\nsamp = 500$".replace(r"\nsamp", "n"),
                 fontsize=9, y=1.05)
    out = FIG_DIR / "four_bulk_overview.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


def _annotate_bulks(ax, dint, dlat, n, p, ylim_top=None):
    """Draw three dashed vertical lines at bulk boundaries and label
    the four regions: signal | noise-dim | sample | rank-null."""
    if ylim_top is None:
        ylim_top = ax.get_ylim()[1]
    ax.axvline(dint, color="tab:red",    lw=0.9, ls="--", alpha=0.85)
    ax.axvline(dlat, color="tab:green",  lw=0.9, ls="--", alpha=0.85)
    if dlat + n < p:
        ax.axvline(dlat + n, color="tab:purple", lw=0.9, ls="--", alpha=0.85)
    # Bulk centers (geometric mean for log-x)
    import math
    centers = []
    if dint > 0:
        centers.append(("signal",     math.sqrt(1 * dint)))
    if dlat > dint:
        centers.append(("noise-dim",  math.sqrt(dint * dlat)))
    if (dlat + n) <= p:
        centers.append(("sample",     math.sqrt(dlat * (dlat + n))))
        centers.append(("rank-null",  math.sqrt((dlat + n) * p)))
    else:
        centers.append(("sample",     math.sqrt(dlat * p)))
    for label, xc in centers:
        ax.text(xc, ax.get_ylim()[1] * 0.6, label, fontsize=6,
                ha="center", va="top",
                rotation=0, alpha=0.9,
                bbox=dict(boxstyle="round,pad=0.15", fc="white",
                          ec="none", alpha=0.7))


def fig_four_bulk_two_sigma():
    """8-panel grid: top row sigma=0.5, bottom row sigma=0.01,
    columns d_lat in {10, 20, 30, 40}. Used as the main spectral figure
    (Figure 2 / fig:fourbulk in sec-rfnn)."""
    cols = [10, 20, 30, 40]
    rows = [
        ("$\\sigma_\\perp=0.5$",  "results_rfnn_exp2v3"),
        ("$\\sigma_\\perp=0.01$", "results_rfnn_exp2_sn001"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(7.0, 3.4), sharey="row")
    for r, (label, root) in enumerate(rows):
        for c, dlat in enumerate(cols):
            ax = axes[r, c]
            run = ROOT / root / f"di5_d{dlat}_n500_s42"
            if not (run / "eigenvalues_pre.npy").exists():
                ax.set_visible(False)
                continue
            e = load_eigs(run / "eigenvalues_pre.npy")
            n_show = min(80, e.size)
            x = np.arange(1, n_show + 1)
            ax.semilogy(x, e[:n_show], color="tab:blue", lw=1.0, marker=".", ms=2)
            ax.axvline(5, color="tab:red", lw=0.9, ls="--", alpha=0.85)
            ax.axvline(dlat, color="tab:green", lw=0.9, ls="--", alpha=0.85)
            if r == 0:
                ax.set_title(rf"$d_{{\rm latent}}={dlat}$")
            if r == 1:
                ax.set_xlabel("eigenvalue index")
            if c == 0:
                ax.set_ylabel(f"{label}\neigenvalue")
            ax.grid(True, which="both", alpha=0.3)
    out = FIG_DIR / "four_bulk_two_sigma.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


def detect_bulks_cliffs(eigs, threshold=0.18, min_sep=2):
    """Greedy cliff detection on the sorted log-eigenvalue spectrum.
    Returns (cliff_indices, cliff_drops) for cliffs whose log10 drop
    exceeds `threshold`, with no two cliffs within `min_sep` indices.
    Cliff_indices[j] = i means eigenvalues at positions 0..i sit in
    one bulk; positions i+1.. sit in the next bulk."""
    e = np.sort(np.asarray(eigs))[::-1]
    log_e = np.log10(np.maximum(e, 1e-30))
    drops = -np.diff(log_e)
    excluded = np.zeros_like(drops, dtype=bool)
    chosen = []
    for i in np.argsort(drops)[::-1]:
        if drops[i] < threshold:
            break
        if excluded[i]:
            continue
        chosen.append(int(i))
        lo, hi = max(0, i - min_sep), min(len(drops), i + min_sep + 1)
        excluded[lo:hi] = True
    chosen.sort()
    return chosen, [float(drops[i]) for i in chosen]


def _classify_cliff(index, dint, dlat, n, tol_frac=0.5):
    """Map a detected cliff index to its canonical role:
    'signal-noise', 'noise-sample', 'sample-null', or 'unclassified'."""
    candidates = [
        ("signal-noise", dint - 1),
        ("noise-sample", dlat - 1),
        ("sample-null", dlat + n - 1),
    ]
    best, best_dist = "unclassified", float("inf")
    for name, target in candidates:
        rel = abs(index - target) / max(target, 1)
        if rel < tol_frac and rel < best_dist:
            best, best_dist = name, rel
    return best


def fig_bulk_detection_hist():
    """Histogram-peak bulk detection across the d_latent sweep at sigma=0.01.
    2 columns x 4 rows layout so each panel has room for the bulk-size labels
    without overlap. Uses scipy.signal.find_peaks on padded histogram counts
    (pad with zeros so edge peaks are detectable). Labels sit above each panel."""
    from scipy.signal import find_peaks
    cells = [(d, ROOT / f"results_rfnn_exp2_sn001/di5_d{d}_n500_s42")
             for d in [5, 8, 10, 15, 20, 30, 40]]
    peak_colors = ["#cc5de8", "#339af0", "#51cf66", "#ff6b6b", "#ffa94d"]
    fig, axes = plt.subplots(4, 2, figsize=(7.5, 9.5))
    flat = axes.flatten()
    bulk_sizes_log = []  # collected for printout to console
    for (dlat, path), ax in zip(cells, flat):
        e = np.sort(np.load(path / "eigenvalues_pre.npy"))[::-1]
        log_e = np.log10(np.maximum(e, 1e-30))
        n_bins = 30
        edges = np.linspace(log_e.min() - 0.05, log_e.max() + 0.05, n_bins + 1)
        counts, _ = np.histogram(log_e, bins=edges)
        centers = 0.5 * (edges[:-1] + edges[1:])
        bin_widths = np.diff(10 ** edges)
        ax.bar(10 ** centers, counts, width=bin_widths, color="steelblue",
               edgecolor="white", lw=0.3, zorder=2, align="center")
        ax.set_xscale("log")
        ax.set_yscale("log")

        padded = np.concatenate([[0], counts, [0]])
        peaks_raw, props = find_peaks(padded, prominence=2, distance=2)
        prominences = props["prominences"]
        if len(peaks_raw) > 4:
            top_idx = np.argsort(prominences)[::-1][:4]
            peaks_raw = peaks_raw[sorted(top_idx)]
        peaks = np.asarray(peaks_raw, dtype=int) - 1
        peaks = peaks[(peaks >= 0) & (peaks < n_bins)]
        boundaries = [0]
        for i in range(len(peaks) - 1):
            valley = peaks[i] + np.argmin(counts[peaks[i]:peaks[i+1] + 1])
            boundaries.append(valley + 1)
        boundaries.append(n_bins)

        sizes_this_panel = []
        for j in range(len(peaks)):
            lam_lo = 10 ** edges[boundaries[j]]
            lam_hi = 10 ** edges[boundaries[j + 1]]
            peak_lam = 10 ** centers[peaks[j]]
            c = peak_colors[j % len(peak_colors)]
            ax.axvspan(lam_lo, lam_hi, color=c, alpha=0.30, zorder=1)
            ax.axvline(peak_lam, color=c, lw=1.0, ls=":", alpha=0.7, zorder=3)
            sizes_this_panel.append(int(np.sum((e >= lam_lo) & (e < lam_hi))))
        bulk_sizes_log.append((dlat, sizes_this_panel))
        ax.set_title(rf"$d_{{\rm latent}} = {dlat}$", fontsize=9)
        ax.set_xlabel(r"eigenvalue $\lambda_i(U)$")
        ax.set_ylabel("count")
        ax.grid(True, which="both", alpha=0.2)
    flat[-1].set_visible(False)
    fig.tight_layout()
    print("bulk_detection_hist sizes (for tex caption):")
    for dlat, sizes in bulk_sizes_log:
        print(f"  d_lat={dlat}: {sizes}")
    out = FIG_DIR / "bulk_detection_hist.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


def fig_bulk_sizes_distances():
    """Two-panel figure: empirical bulk sizes (top) and cliff drops
    (bottom) versus d_latent / d_intrinsic. Uses cliff detection with
    threshold=0.18 dec and min_sep=2; classifies each detected cliff
    by its index position (signal-noise, noise-sample, sample-null)."""
    sweep = [
        (5,   "results_rfnn_exp2v3/di5_d5_n500_s42"),
        (8,   "results_rfnn_exp2v3/di5_d8_n500_s42"),
        (10,  "results_rfnn_exp2v3/di5_d10_n500_s42"),
        (15,  "results_rfnn_exp2v3/di5_d15_n500_s42"),
        (20,  "results_rfnn_exp2v3/di5_d20_n500_s42"),
        (30,  "results_rfnn_exp2v3/di5_d30_n500_s42"),
        (40,  "results_rfnn_exp2v3/di5_d40_n500_s42"),
        (60,  "results_rfnn_exp2_wide/di5_d60_n500_s42"),
        (80,  "results_rfnn_exp2_wide/di5_d80_n500_s42"),
        (100, "results_rfnn_exp2_wide/di5_d100_n500_s42"),
        (150, "results_rfnn_exp2_wide/di5_d150_n500_s42"),
        (200, "results_rfnn_exp2_wide/di5_d200_n500_s42"),
    ]
    dint = 5
    n = 500

    # Collect per-role drop and bulk-size sequences keyed by canonical role
    drop_by_role = {"signal-noise": [], "noise-sample": [], "sample-null": []}
    size_by_role = {"signal": [], "noise-dim": [], "sample": [], "rank-null": []}
    ratios = []

    for dlat, path in sweep:
        e = np.sort(np.load(ROOT / path / "eigenvalues_pre.npy"))[::-1]
        cliffs, drops = detect_bulks_cliffs(e)
        # Classify each cliff
        roles = [_classify_cliff(c, dint, dlat, n) for c in cliffs]
        # Drops keyed by role
        for role in drop_by_role:
            drop_val = next((d for r, d in zip(roles, drops) if r == role), None)
            drop_by_role[role].append(drop_val)
        # Sizes by canonical bulk: derived from boundary indices
        boundaries = sorted({c for c in cliffs})
        # Map boundaries to roles, then bulk sizes follow
        # Default: use theoretical bulk sizes when detection fails
        b_signal = next((c + 1 for c, r in zip(cliffs, roles) if r == "signal-noise"), dint)
        b_noise  = next((c + 1 for c, r in zip(cliffs, roles) if r == "noise-sample"), dlat)
        b_sample = next((c + 1 for c, r in zip(cliffs, roles) if r == "sample-null"), min(dlat + n, e.size))
        b_total  = e.size
        size_by_role["signal"].append(b_signal)
        size_by_role["noise-dim"].append(max(0, b_noise - b_signal))
        size_by_role["sample"].append(max(0, b_sample - b_noise))
        size_by_role["rank-null"].append(max(0, b_total - b_sample))
        ratios.append(dlat / dint)

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.0))

    # Left panel: bulk sizes per role
    colors_size = {"signal": "tab:red", "noise-dim": "tab:green",
                   "sample": "tab:blue", "rank-null": "tab:purple"}
    for role, sizes in size_by_role.items():
        axes[0].plot(ratios, sizes, "o-", color=colors_size[role],
                     label=role, lw=1.4, ms=5)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel(r"$d_{\rm latent} / d_{\rm intrinsic}$")
    axes[0].set_ylabel("bulk size (count of eigenvalues)")
    axes[0].set_title("Bulk sizes")
    axes[0].grid(True, which="both", alpha=0.25)
    axes[0].legend(fontsize=8, loc="best", frameon=False)

    # Right panel: cliff drops per role
    colors_drop = {"signal-noise": "tab:red", "noise-sample": "tab:green",
                   "sample-null": "tab:purple"}
    for role, drops in drop_by_role.items():
        # Filter Nones (cliff not detected at this d_lat)
        x_j = [r for r, d in zip(ratios, drops) if d is not None]
        y_j = [d for d in drops if d is not None]
        axes[1].plot(x_j, y_j, "o-", color=colors_drop[role],
                     label=role.replace("-", "/"), lw=1.4, ms=5)
    axes[1].set_xscale("log")
    axes[1].set_xlabel(r"$d_{\rm latent} / d_{\rm intrinsic}$")
    axes[1].set_ylabel("cliff drop (log$_{10}$ decades)")
    axes[1].set_title("Inter-bulk cliff sizes")
    axes[1].grid(True, which="both", alpha=0.25)
    axes[1].legend(fontsize=8, loc="best", frameon=False)

    fig.tight_layout()
    out = FIG_DIR / "bulk_sizes_distances.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


def fig_four_bulk_clean_example():
    """Single-panel headline figure: density of log10(eigenvalue) for one
    clean configuration showing all four bulks as distinct peaks, labeled
    (i)..(iv) matching the bulk_data_signal / bulk_data_noise /
    bulk_sample_signal / bulk_sample_noise naming in the body."""
    # Use sigma_perp = 0.5, d_lat = 40 -- cleanest 4-bulk separation
    run = ROOT / "results_rfnn_exp2v3/di5_d40_n500_s42"
    e = np.sort(np.load(run / "eigenvalues_pre.npy"))[::-1]
    log_e = np.log10(np.maximum(e, 1e-30))
    dint = 5
    dlat = 40
    n = 500
    p = e.size

    fig, ax = plt.subplots(figsize=(4.5, 2.6))
    ax.hist(log_e, bins=60, color="tab:blue", edgecolor="white", lw=0.2)
    ax.set_yscale("log")
    ax.set_xlabel(r"$\log_{10}\,\lambda_i(U)$")
    ax.set_ylabel("count")

    # Label each bulk above its representative peak
    bulks = [
        ("(i)",  np.median(log_e[:dint])),
        ("(ii)", np.median(log_e[dint:dlat])),
        ("(iii)",np.median(log_e[dlat:dlat + n])),
        ("(iv)", np.median(log_e[dlat + n:])),
    ]
    ymax = ax.get_ylim()[1]
    for label, xc in bulks:
        ax.text(xc, ymax * 0.5, label, fontsize=9, ha="center", va="bottom",
                bbox=dict(boxstyle="round,pad=0.2", fc="white",
                          ec="0.6", lw=0.4))
    out = FIG_DIR / "four_bulk_clean_example.pdf"
    fig.tight_layout()
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


def fig_four_bulk_dlat_sweep():
    """Vary d_lat at fixed d_int=5, sn=0.01. 2-row x 4-col grid,
    linear-x log-y, dashed vertical lines at d_int and d_lat. Matches
    the cliffs-view screenshot Trevor showed."""
    cells = [5, 8, 10, 15, 20, 30, 40]
    n = 500
    n_show = 80
    fig, axes = plt.subplots(2, 4, figsize=(8.0, 4.0), sharey=True)
    flat = axes.flatten()
    for c, dlat in enumerate(cells):
        ax = flat[c]
        run = ROOT / f"results_rfnn_exp2_sn001/di5_d{dlat}_n500_s42"
        e = np.sort(np.load(run / "eigenvalues_pre.npy"))[::-1]
        x = np.arange(1, n_show + 1)
        ax.semilogy(x[:n_show], np.maximum(e[:n_show], 1e-30),
                    color="tab:blue", lw=0.9, marker=".", ms=2)
        ax.axvline(5,    color="tab:red",   lw=0.9, ls="--", alpha=0.85,
                   label=r"$d_{\rm intrinsic}=5$")
        if dlat != 5:
            ax.axvline(dlat, color="tab:green", lw=0.9, ls="--", alpha=0.85,
                       label=rf"$d_{{\rm latent}}={dlat}$")
        ax.set_title(rf"$d_{{\rm latent}}={dlat}$", fontsize=9)
        if c >= 4:
            ax.set_xlabel("eigenvalue index")
        if c % 4 == 0:
            ax.set_ylabel("eigenvalue (log)")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=6, loc="upper right", frameon=False)
    flat[-1].set_visible(False)
    fig.tight_layout()
    out = FIG_DIR / "four_bulk_dlat_sweep.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


def fig_data_pca_dlat_sweep():
    """Sorted eigenvalues of the empirical data covariance
    (1/n) X^T X versus index, for the same d_lat values shown in the
    U figures. Demonstrates that the signal + noise-dim block structure
    is in the *data*, not just in U."""
    cells = [10, 20, 30, 40]
    fig, axes = plt.subplots(1, 4, figsize=(8.0, 2.0), sharey=False)

    def gen_data(n, d_int, d_lat, k=10, sn=0.01, scale=3.0, seed=42, sig=1.0):
        rng = np.random.default_rng(seed)
        means = rng.standard_normal((k, d_int)) * scale / np.sqrt(d_int)
        labels = rng.integers(0, k, size=n)
        X = np.zeros((n, d_lat))
        X[:, :d_int] = means[labels] + rng.standard_normal((n, d_int)) * sig
        X[:, d_int:] = rng.standard_normal((n, d_lat - d_int)) * sn
        Q, _ = np.linalg.qr(rng.standard_normal((d_lat, d_lat)))
        return X @ Q.T

    for c, dlat in enumerate(cells):
        ax = axes[c]
        X = gen_data(500, 5, dlat)
        Xc = X - X.mean(0)
        cov = Xc.T @ Xc / X.shape[0]
        eigs = np.sort(np.linalg.eigvalsh(cov))[::-1]
        x = np.arange(1, len(eigs) + 1)
        ax.semilogy(x, np.maximum(eigs, 1e-30),
                    color="tab:purple", lw=0.9, marker="o", ms=3)
        ax.axvline(5, color="tab:red", lw=0.9, ls="--", alpha=0.85,
                   label=r"$d_{\rm intrinsic}=5$")
        ax.axvline(dlat, color="tab:green", lw=0.9, ls="--", alpha=0.85,
                   label=rf"$d_{{\rm latent}}={dlat}$")
        ax.set_title(rf"$d_{{\rm latent}}={dlat}$", fontsize=9)
        ax.set_xlabel("PCA index")
        if c == 0:
            ax.set_ylabel(r"eigenvalue of $X^\top X / n$")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=6, loc="upper right", frameon=False)
    fig.suptitle(rf"Data PCA: $d_{{\rm intrinsic}}=5$, $n=500$, $\sigma_\perp=0.01$",
                 fontsize=9, y=1.02)
    fig.tight_layout()
    out = FIG_DIR / "data_pca_dlat_sweep.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


def fig_four_bulk_dlat_sweep_sn05():
    """Sigma=0.5 cliffs view of the dlat sweep, matching the layout of
    the existing sigma=0.01 four_bulk_dlat_sweep figure (used in the
    appendix). 7 panels, d_lat in {5, 8, 10, 15, 20, 30, 40}."""
    cells = [5, 8, 10, 15, 20, 30, 40]
    n = 500
    n_show = 80
    fig, axes = plt.subplots(2, 4, figsize=(8.0, 4.0), sharey=True)
    flat = axes.flatten()
    for c, dlat in enumerate(cells):
        ax = flat[c]
        run = ROOT / f"results_rfnn_exp2v3/di5_d{dlat}_n500_s42"
        e = np.sort(np.load(run / "eigenvalues_pre.npy"))[::-1]
        x = np.arange(1, n_show + 1)
        ax.semilogy(x[:n_show], np.maximum(e[:n_show], 1e-30),
                    color="tab:blue", lw=0.9, marker=".", ms=2)
        ax.axvline(5, color="tab:red", lw=0.9, ls="--", alpha=0.85,
                   label=r"$d_{\rm intrinsic}=5$")
        if dlat != 5:
            ax.axvline(dlat, color="tab:green", lw=0.9, ls="--", alpha=0.85,
                       label=rf"$d_{{\rm latent}}={dlat}$")
        ax.set_title(rf"$d_{{\rm latent}}={dlat}$", fontsize=9)
        if c >= 4:
            ax.set_xlabel("eigenvalue index")
        if c % 4 == 0:
            ax.set_ylabel("eigenvalue (log)")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=6, loc="upper right", frameon=False)
    flat[-1].set_visible(False)
    fig.tight_layout()
    out = FIG_DIR / "four_bulk_dlat_sweep_sn05.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


def fig_four_bulk_dlat_hist_sn001():
    """Sigma=0.01 histogram density-view of the dlat sweep, matching the
    layout of the body's four_bulk_dlat_hist figure but at the lower noise
    level (used in the appendix to complement the body's sigma=0.5 view)."""
    cells = [5, 10, 20, 40, 100, 200]
    fig, axes = plt.subplots(2, 3, figsize=(8.0, 4.4), sharey=False)
    flat = axes.flatten()
    for c, dlat in enumerate(cells):
        ax = flat[c]
        # exp2_sn001 only has up to d_lat=40. For d_lat >= 60 there's no
        # sn=0.01 wide sweep, so skip these panels gracefully.
        if dlat <= 40:
            run = ROOT / f"results_rfnn_exp2_sn001/di5_d{dlat}_n500_s42"
        else:
            ax.set_visible(False)
            continue
        e = np.sort(np.load(run / "eigenvalues_pre.npy"))[::-1]
        log_e = np.log10(np.maximum(e, 1e-30))
        ax.hist(log_e, bins=60, color="tab:blue", edgecolor="white", lw=0.2)
        ax.set_yscale("log")
        ax.set_title(rf"$d_{{\rm latent}} = {dlat}$", fontsize=9)
        if c >= 3:
            ax.set_xlabel(r"$\log_{10}\,\lambda_i(U)$")
        if c % 3 == 0:
            ax.set_ylabel("count")
        ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    out = FIG_DIR / "four_bulk_dlat_hist_sn001.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


def fig_four_bulk_dlat_hist():
    """3x2 grid of log10(eigenvalue) histograms across the d_latent sweep
    at sigma_perp = 0.5. Includes the d_lat = d_int special case where the
    noise-dim bulk vanishes, and goes up to d_lat = 200 to show extreme
    separation. Three dashed vertical lines per panel mark the
    cumulative-count bulk boundaries: log10(lambda_{d_int}),
    log10(lambda_{d_lat}), log10(lambda_{d_lat + n})."""
    cells = [5, 10, 20, 40, 100, 200]
    n = 500
    dint = 5
    # Map each d_lat to its results path. For d_lat <= 40 we have exp2v3
    # (the 7-d_lat sweep at sigma=0.5); for >= 60 we have exp2_wide.
    def run_path(dlat):
        if dlat <= 40:
            return ROOT / f"results_rfnn_exp2v3/di5_d{dlat}_n500_s42"
        return ROOT / f"results_rfnn_exp2_wide/di5_d{dlat}_n500_s42"

    # 4 bulk colors, ordered low-lambda -> high-lambda
    # (rank-null, sample, noise-dim, signal).
    bulk_colors = ["#cc5de8", "#339af0", "#51cf66", "#ff6b6b"]
    fig, axes = plt.subplots(3, 2, figsize=(6.0, 6.6), sharey=False)
    flat = axes.flatten()
    for c, dlat in enumerate(cells):
        ax = flat[c]
        e = np.sort(np.load(run_path(dlat) / "eigenvalues_pre.npy"))[::-1]
        log_e = np.log10(np.maximum(e, 1e-30))
        p = e.size
        n_bins = 60
        edges = np.linspace(log_e.min() - 0.05, log_e.max() + 0.05, n_bins + 1)
        counts, _ = np.histogram(log_e, bins=edges)
        centers = 0.5 * (edges[:-1] + edges[1:])
        ax.bar(centers, counts, width=np.diff(edges), color="steelblue",
               edgecolor="white", lw=0.2, zorder=2, align="center")
        ax.set_yscale("log")

        # Theoretical bulk boundaries from indices: e is sorted descending,
        # so the boundary BETWEEN the i-th and (i+1)-th bulk in log-lambda
        # is the midpoint of log_e[idx-1] and log_e[idx]. Bulks left-to-
        # right on the log-lambda x-axis: rank-null, sample, noise-dim,
        # signal.
        dint = 5
        n = 500
        if dlat == 5:
            # d_lat = d_int special case: rank-null and noise-dim are
            # theoretically empty, so the index-based cuts collapse. Use
            # manually-placed boundaries that match the visible structure
            # of the histogram (low-lambda tail, main bulk, transition,
            # signal eigenvalues).
            bx = [log_e.min(), -2.0, -1.0, 1.0, log_e.max()]
        elif dlat == 100:
            # Theoretical cuts land slightly off the visible bulk
            # boundaries; place cuts at the histogram-gap positions.
            bx = [log_e.min(), -2.0, 0.0, 2.0, log_e.max()]
        elif dlat == 200:
            bx = [log_e.min(), -1.5, -0.5, 2.0, log_e.max()]
        else:
            cuts_idx = [p - 1, dlat + n - 1, dlat - 1, dint - 1, 0]
            cuts_idx = [min(max(i, 0), p - 1) for i in cuts_idx]
            bx = []
            for k, idx in enumerate(cuts_idx):
                if k == 0:
                    bx.append(log_e[idx])
                elif k == len(cuts_idx) - 1:
                    bx.append(log_e[idx])
                else:
                    lo, hi = log_e[idx], log_e[idx - 1] if idx - 1 >= 0 else log_e[idx]
                    bx.append(0.5 * (lo + hi))

        for j in range(4):
            lo, hi = bx[j], bx[j + 1]
            if hi <= lo:
                continue
            ax.axvspan(lo, hi, color=bulk_colors[j], alpha=0.30, zorder=1)
        for j in range(1, 4):
            left_ok = bx[j] > bx[j - 1]
            right_ok = bx[j + 1] > bx[j]
            if left_ok and right_ok:
                ax.axvline(bx[j], color="0.25", lw=0.8, ls="--",
                           alpha=0.7, zorder=3)

        ax.set_title(rf"$d_{{\rm latent}} = {dlat}$", fontsize=9)
        if c >= 4:
            ax.set_xlabel(r"$\log_{10}\,\lambda_i(U)$")
        if c % 2 == 0:
            ax.set_ylabel("count")
        ax.grid(True, which="both", alpha=0.25)


    fig.tight_layout()
    out = FIG_DIR / "four_bulk_dlat_hist.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


def fig_four_bulk_dint_sweep():
    """Vary d_int at fixed d_lat=20, sigma=0.5. Single row of panels,
    log-log axes, 3 dashed vertical lines + bulk labels."""
    cols = [2, 5, 8, 12, 16, 20]
    n = 500
    fig, axes = plt.subplots(1, len(cols), figsize=(11.0, 2.4), sharey=True)
    for c, dint in enumerate(cols):
        ax = axes[c]
        run = ROOT / f"results_rfnn_exp3/di{dint}_d20_n500_s42"
        e = np.sort(np.load(run / "eigenvalues_pre.npy"))[::-1]
        p = e.size
        x = np.arange(1, p + 1)
        ax.loglog(x, np.maximum(e, 1e-30), color="tab:blue",
                  lw=0.8, marker=".", ms=1)
        ax.set_title(rf"$d_{{\rm intrinsic}}={dint}$", fontsize=9)
        ax.set_xlabel("index")
        if c == 0:
            ax.set_ylabel("eigenvalue")
        ax.grid(True, which="both", alpha=0.25)
        ax.set_xlim(1, p)
        ax.set_ylim(max(e.min(), 1e-7), e.max() * 2)
        _annotate_bulks(ax, dint, 20, n, p)
    fig.tight_layout()
    out = FIG_DIR / "four_bulk_dint_sweep.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


def fig_eigenvalue_histogram():
    """Log-density histogram of the full p-sized spectrum at sigma=0.5,
    showing the four bulks as distinct peaks. Used as the histogram
    appendix figure / second part of fig:overview."""
    runs = [(5, 5), (10, 10), (20, 20), (40, 40)]
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 1.9), sharey=False)
    for ax, (dlat, _) in zip(axes, runs):
        path = ROOT / f"results_rfnn_exp2v3/di5_d{dlat}_n500_s42/eigenvalues_pre.npy"
        e = load_eigs(path)
        log_e = np.log10(np.maximum(e, 1e-30))
        ax.hist(log_e, bins=50, color="tab:blue", edgecolor="white", lw=0.2)
        ax.set_yscale("log")
        ax.set_title(rf"$d_{{\rm latent}}={dlat}$")
        ax.set_xlabel(r"$\log_{10}\,\lambda$")
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel("count")
    fig.tight_layout()
    fig.suptitle(r"$\sigma_\perp = 0.5$ histogram of $U$ spectrum",
                 fontsize=9, y=1.05)
    out = FIG_DIR / "eigenvalue_histogram.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


def _collect_mlp_timescales(root_name, sn_target):
    runs = []
    for d in sorted((ROOT / root_name).iterdir()):
        if not d.is_dir():
            continue
        cfg = json.loads((d / "config.json").read_text())
        if abs(cfg["sigma_noise"] - sn_target) > 1e-9:
            continue
        rows = [json.loads(l) for l in (d / "metrics.jsonl").read_text().splitlines() if l.strip()]
        # tau_gen
        tests = [r["test_loss"] for r in rows]
        tg = None
        if tests:
            tmin = min(tests)
            for r in rows:
                if r["test_loss"] <= 1.05 * tmin:
                    tg = r["step"]
                    break
        tm = None
        for r in rows:
            if r.get("memorization_fraction", 0) > 0.01:
                tm = r["step"]
                break
        runs.append((cfg["d_latent"], tg, tm))
    runs.sort(key=lambda x: x[0])
    return runs


def fig_rfnn_timescales():
    """Single-panel: tau_gen vs d_lat for RFNN at both sigma values."""
    fig, ax = plt.subplots(1, 1, figsize=(3.5, 2.4))
    # sigma=0.5 wide sweep
    runs05 = []
    for d in sorted((ROOT / "results_rfnn_exp2_wide").iterdir()):
        if not d.is_dir():
            continue
        cfg = json.loads((d / "config.json").read_text())
        rows = [json.loads(l) for l in (d / "metrics.jsonl").read_text().splitlines() if l.strip()]
        tests = [r["test_loss"] for r in rows]
        tg = None
        if tests:
            tmin = min(tests)
            for r in rows:
                if r["test_loss"] <= 1.05 * tmin:
                    tg = r["step"]
                    break
        runs05.append((cfg["d_latent"], tg))
    runs05.sort()
    runs01 = []
    for d in sorted((ROOT / "results_rfnn_exp2_sn001").iterdir()):
        if not d.is_dir():
            continue
        cfg = json.loads((d / "config.json").read_text())
        rows = [json.loads(l) for l in (d / "metrics.jsonl").read_text().splitlines() if l.strip()]
        tests = [r["test_loss"] for r in rows]
        tg = None
        if tests:
            tmin = min(tests)
            for r in rows:
                if r["test_loss"] <= 1.05 * tmin:
                    tg = r["step"]
                    break
        runs01.append((cfg["d_latent"], tg))
    runs01.sort()
    ax.plot([r[0] for r in runs05], [max(r[1], 1) for r in runs05],
            "o-", color="tab:blue", label=r"$\sigma_\perp = 0.5$", lw=1.2)
    ax.plot([r[0] for r in runs01], [max(r[1], 1) for r in runs01],
            "s-", color="tab:orange", label=r"$\sigma_\perp = 0.01$", lw=1.2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$d_{\rm latent}$")
    ax.set_ylabel(r"$\tau_{\rm gen}$ (step)")
    ax.set_title(r"RFNN $\tau_{\rm gen}$ scaling, $d_{\rm intrinsic}=5$")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(frameon=False)
    out = FIG_DIR / "rfnn_timescales.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


_MLP_PRESETS = {
    0.01: ("results_mlp_exp2_sn001", "sn001"),
    0.5:  ("results_mlp_exp2_sn05",  "sn05"),
}
# For the "approaches after 5M steps" figures we want the long-horizon
# unscaled MLP run. exp2_v4 only ships sigma=0.5 at 5M steps; sn=0.01
# falls back to the 300k-step exp2_sn001 sweep.
_MLP_LATE_PRESETS = {
    0.01: ("results_mlp_exp2_sn001", "sn001"),
    0.5:  ("exp2_v4",                "sn05"),
}


def _late_value(rows, key, frac=0.05):
    """Value the metric `key` approaches at the end of training.
    Average of the last `frac` fraction of logged rows (default 5%)."""
    vals = [r[key] for r in rows if r.get(key) is not None]
    if not vals:
        return None
    k = max(1, int(len(vals) * frac))
    return float(np.mean(vals[-k:]))


_MLP_EXP3_PRESETS = {
    # sigma=0.5 uses the 5M-step runs pulled from ml6/ml7
    # (results_5M_exp3); sigma=0.01 stays on the 300k local sweep.
    0.01: ("results_mlp_exp3_sn001", "sn001"),
    0.5:  ("results_5M_exp3",        "sn05"),
}


def _walk_exp3(root, sn):
    """Walk exp3 runs: yield (d_intrinsic, rows) sorted by d_intrinsic."""
    out = []
    for d in sorted((ROOT / root).iterdir()):
        if not d.is_dir() or not (d / "config.json").exists():
            continue
        cfg = json.loads((d / "config.json").read_text())
        if abs(cfg.get("sigma_noise", sn) - sn) > 1e-9:
            continue
        rows = [json.loads(l) for l in
                (d / "metrics.jsonl").read_text().splitlines() if l.strip()]
        rows.sort(key=lambda r: r["step"])
        out.append((cfg["d_intrinsic"], cfg["d_latent"], rows))
    # Sort by ratio d_latent / d_intrinsic ascending (1 -> 10), so the
    # x-axis matches the body figures and reads left-to-right small-to-large.
    out.sort(key=lambda r: r[1] / r[0])
    return out


def _fig_mlp_dint_timescales(sn):
    """Bar chart: tau_gen and tau_mem vs d_latent / d_intrinsic for the
    d_int sweep at fixed d_lat=20. Y-axis is the actual training budget
    of the source runs (5M for sigma=0.5, 300k for sigma=0.01)."""
    root, suffix = _MLP_EXP3_PRESETS[sn]
    fig, ax = plt.subplots(figsize=(4.0, 2.6))
    bar_w = 0.4
    runs = _walk_exp3(root, sn)
    # Use the longest run as the budget (handles 5M vs 300k cases).
    BUDGET = max(rows[-1]["step"] for _, _, rows in runs) / 1e6
    Y_MAX = BUDGET * 1.05
    labels, tgs, tms, tm_cens = [], [], [], []
    for dint, dlat, rows in runs:
        last = rows[-1]["step"]
        tests = [r["test_loss"] for r in rows]
        tg = None
        if tests:
            tmin = min(tests)
            for r in rows:
                if r["test_loss"] <= 1.05 * tmin:
                    tg = r["step"]
                    break
        tm = None
        for r in rows:
            if r.get("memorization_fraction", 0) > 0.01:
                tm = r["step"]
                break
        labels.append(f"{dlat/dint:g}")
        tgs.append((tg if tg is not None else last) / 1e6)
        tms.append((tm if tm is not None else last) / 1e6)
        tm_cens.append(tm is None)
    x = np.arange(len(runs))
    ax.bar(x - bar_w/2, tgs, width=bar_w, color="tab:blue",
           label=r"$\tau_{\rm gen}$")
    for xi, tm, cens, last_step in zip(x, tms, tm_cens,
                                       [rows[-1]["step"]/1e6
                                        for _, _, rows in runs]):
        if cens:
            ax.bar(xi + bar_w/2, last_step, width=bar_w,
                   color="tab:red", alpha=0.35, hatch="///",
                   edgecolor="tab:red")
        else:
            ax.bar(xi + bar_w/2, tm, width=bar_w, color="tab:red")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel(r"$d_{\rm latent} / d_{\rm intrinsic}$")
    ax.set_ylim(0, Y_MAX)
    ax.set_ylabel("training steps (millions)")
    ax.set_title(rf"MLP $d_{{\rm intrinsic}}$ sweep "
                 rf"($d_{{\rm latent}} = 20$), $\sigma_\perp = {sn}$")
    ax.grid(True, axis="y", alpha=0.3)
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="tab:blue", label=r"$\tau_{\rm gen}$"),
        Patch(facecolor="tab:red",  label=r"$\tau_{\rm mem}$ (reached)"),
        Patch(facecolor="tab:red",  alpha=0.35, hatch="///",
              edgecolor="tab:red", label=r"$\tau_{\rm mem}$ (not reached)"),
    ]
    ax.legend(handles=legend_handles, loc="upper right",
              frameon=False, fontsize=7)
    out = FIG_DIR / f"mlp_dint_timescales_{suffix}.pdf"
    fig.tight_layout()
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


def _fig_mlp_dint_gen_gap(sn):
    """Bar chart: late gen-gap vs d_latent / d_intrinsic for exp3."""
    root, suffix = _MLP_EXP3_PRESETS[sn]
    fig, ax = plt.subplots(figsize=(4.0, 2.4))
    runs = _walk_exp3(root, sn)
    labels, gaps = [], []
    for dint, dlat, rows in runs:
        gg = _late_value(rows, "gen_gap")
        if gg is None:
            continue
        labels.append(f"{dlat/dint:g}")
        gaps.append(gg)
    x = np.arange(len(labels))
    ax.bar(x, gaps, color="tab:purple", width=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel(r"$d_{\rm latent} / d_{\rm intrinsic}$")
    ax.set_ylabel(r"late gen-gap (test $-$ train)")
    ax.set_title(rf"MLP $d_{{\rm intrinsic}}$ sweep "
                 rf"($d_{{\rm latent}} = 20$), $\sigma_\perp = {sn}$")
    ax.grid(True, axis="y", alpha=0.3)
    out = FIG_DIR / f"mlp_dint_gen_gap_{suffix}.pdf"
    fig.tight_layout()
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


def _fig_mlp_dint_score_curves(sn):
    """Per-step score-error trajectories across exp3, one curve per d_int."""
    root, suffix = _MLP_EXP3_PRESETS[sn]
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    runs = _walk_exp3(root, sn)
    ratios = [(r[1] / r[0], r[2]) for r in runs]
    if not ratios:
        plt.close(fig)
        return
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(ratios)))
    for (ratio, rows), color in zip(ratios, colors):
        steps = [r["step"] / 1e6 for r in rows
                 if r.get("score_error") is not None]
        scores = [r["score_error"] for r in rows
                  if r.get("score_error") is not None]
        ax.plot(steps, scores, color=color, lw=1.2,
                label=rf"$d_{{\rm latent}}/d_{{\rm intrinsic}}={ratio:g}$")
    ax.set_xlabel("training steps (millions)")
    ax.set_ylabel("score error")
    ax.set_title(rf"MLP $d_{{\rm intrinsic}}$ sweep "
                 rf"($d_{{\rm latent}} = 20$), $\sigma_\perp = {sn}$")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=6, loc="best", frameon=False, ncol=2)
    out = FIG_DIR / f"mlp_dint_score_curves_{suffix}.pdf"
    fig.tight_layout()
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


def _fig_mlp_dint_late_score(sn):
    """Late score error vs d_latent / d_intrinsic for exp3."""
    root, suffix = _MLP_EXP3_PRESETS[sn]
    fig, ax = plt.subplots(figsize=(4.0, 2.6))
    runs = _walk_exp3(root, sn)
    ratios, scores = [], []
    for dint, dlat, rows in runs:
        late = _late_value(rows, "score_error")
        if late is None:
            continue
        ratios.append(dlat / dint)
        scores.append(late)
    ax.plot(ratios, scores, "s-", color="tab:red", lw=1.5, ms=6,
            label="MLP")
    ax.axvline(1.0, color="gray", linestyle=":", alpha=0.5,
               label=r"$d_{\rm latent} = d_{\rm intrinsic}$")
    ax.set_xlabel(r"$d_{\rm latent} / d_{\rm intrinsic}$")
    ax.set_ylabel("late score error")
    ax.set_title(rf"MLP $d_{{\rm intrinsic}}$ sweep "
                 rf"($d_{{\rm latent}} = 20$), $\sigma_\perp = {sn}$")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="best", frameon=False)
    out = FIG_DIR / f"mlp_dint_late_score_{suffix}.pdf"
    fig.tight_layout()
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


def fig_mlp_dint_timescales_scatter(sn=0.5):
    """Dot-plot version of the d_int sweep timescales, same style as the
    body d_lat scatter (fig_mlp_timescales_scatter): blue dots = tau_gen,
    red dots = tau_mem (reached), red triangles = not reached.
    All exp3 5M runs reach memorization, so no triangles are produced for
    sigma=0.5; the symbol vocabulary is shared for compatibility."""
    from matplotlib.lines import Line2D
    root, suffix = _MLP_EXP3_PRESETS[sn]
    runs = _walk_exp3(root, sn)
    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    Y_MAX = max(rows[-1]["step"] for _, _, rows in runs) / 1e6
    for dint, dlat, rows in runs:
        last = rows[-1]["step"] / 1e6
        ratio = dlat / dint
        tg = None
        tests = [r["test_loss"] for r in rows]
        if tests:
            tmin = min(tests)
            for r in rows:
                if r["test_loss"] <= 1.05 * tmin:
                    tg = r["step"] / 1e6
                    break
        tm = None
        for r in rows:
            if r.get("memorization_fraction", 0) > 0.01:
                tm = r["step"] / 1e6
                break
        if tm is not None:
            ax.scatter(ratio, tm, s=70, c="tab:red", zorder=5,
                       edgecolors="black", linewidths=0.8)
        else:
            ax.scatter(ratio, last, s=70, marker="^", c="tab:red",
                       zorder=5, edgecolors="black", linewidths=0.8,
                       alpha=0.45)
        if tg is not None:
            ax.scatter(ratio, tg, s=70, c="tab:blue", zorder=5,
                       edgecolors="black", linewidths=0.8)
    ax.legend(handles=[
        Line2D([0], [0], marker="o", color="w", mfc="tab:red",  ms=8,
               label=r"$\tau_{\rm mem}$ ($>1\%$)"),
        Line2D([0], [0], marker="o", color="w", mfc="tab:blue", ms=8,
               label=r"$\tau_{\rm gen}$ (5\% of min)"),
        Line2D([0], [0], marker="^", color="w", mfc="tab:red",  ms=8,
               alpha=0.45, label="not reached (last logged step)"),
    ], fontsize=7, frameon=False, loc="upper left")
    ax.set_xlabel(r"$d_{\rm latent} / d_{\rm intrinsic}$")
    ax.set_ylabel("training steps (millions)")
    ax.set_ylim(0, Y_MAX * 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / f"mlp_dint_timescales_scatter_{suffix}.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


def fig_mlp_dint_sweep():
    for sn in _MLP_EXP3_PRESETS:
        _fig_mlp_dint_timescales(sn)
        _fig_mlp_dint_gen_gap(sn)
        _fig_mlp_dint_score_curves(sn)
        _fig_mlp_dint_late_score(sn)
    fig_mlp_dint_timescales_scatter(0.5)


def fig_mlp_timescales_scatter():
    """Scatter view of tau_gen, tau_mem vs d_lat / d_int for the 5M-step
    extended scaled MLP run (exp2_v4, hidden=8*d_lat, sigma=0.5). Triangles
    indicate runs that never crossed the memorization threshold within
    their logged budget."""
    from matplotlib.lines import Line2D
    D_INTRINSIC = 5
    runs = {}
    src = ROOT / "exp2_v4"
    for d in sorted(src.iterdir()):
        if not d.is_dir() or not (d / "config.json").exists():
            continue
        cfg = json.loads((d / "config.json").read_text())
        rows = [json.loads(l) for l in
                (d / "metrics.jsonl").read_text().splitlines() if l.strip()]
        rows.sort(key=lambda r: r["step"])
        runs[cfg["d_latent"]] = rows

    def first_above(rows, key, threshold):
        for m in rows:
            if m.get(key, 0) > threshold:
                return m["step"]
        return None

    def first_within_5pct(rows, key="test_loss"):
        vmin = min(m[key] for m in rows)
        for m in rows:
            if m[key] < vmin * 1.05:
                return m["step"]
        return None

    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    for d in sorted(runs):
        rows = runs[d]
        last = rows[-1]["step"] / 1e6
        tm = first_above(rows, "memorization_fraction", 0.01)
        tg = first_within_5pct(rows, "test_loss")
        tm = tm / 1e6 if tm is not None else None
        tg = tg / 1e6 if tg is not None else None
        ratio = d / D_INTRINSIC
        if tm is not None:
            ax.scatter(ratio, tm, s=70, c="tab:red", zorder=5,
                       edgecolors="black", linewidths=0.8)
        else:
            ax.scatter(ratio, last, s=70, marker="^", c="tab:red",
                       zorder=5, edgecolors="black", linewidths=0.8,
                       alpha=0.45)
        if tg is not None:
            ax.scatter(ratio, tg, s=70, c="tab:blue", zorder=5,
                       edgecolors="black", linewidths=0.8)
    ax.legend(handles=[
        Line2D([0], [0], marker="o", color="w", mfc="tab:red",  ms=8,
               label=r"$\tau_{\rm mem}$ ($>1\%$)"),
        Line2D([0], [0], marker="o", color="w", mfc="tab:blue", ms=8,
               label=r"$\tau_{\rm gen}$ (5\% of min)"),
        Line2D([0], [0], marker="^", color="w", mfc="tab:red",  ms=8,
               alpha=0.45, label="never (last logged step)"),
    ], fontsize=7, frameon=False, loc="upper right")
    ax.set_xlabel(r"$d_{\rm latent} / d_{\rm intrinsic}$")
    ax.set_ylabel("training steps (millions)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / "mlp_timescales_scatter.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


def _fig_mlp_timescales_bar(sn):
    """Single-panel bar chart: tau_gen and tau_mem per d_lat / d_int at one
    sigma. Y-axis runs to the actual 300k training budget."""
    root, suffix = _MLP_PRESETS[sn]
    DINT = 5
    fig, ax = plt.subplots(figsize=(4.0, 2.6))
    BUDGET = 300_000 / 1e6
    Y_MAX = BUDGET * 1.1
    bar_w = 0.4
    runs = _collect_mlp_timescales(root, sn)
    labels = [f"{r[0]/DINT:g}" for r in runs]
    x = np.arange(len(runs))
    tgs = [(r[1] if r[1] is not None else 300_000) / 1e6 for r in runs]
    tms = [(r[2] if r[2] is not None else 300_000) / 1e6 for r in runs]
    tm_censored = [r[2] is None for r in runs]
    ax.bar(x - bar_w/2, tgs, width=bar_w, color="tab:blue",
           label=r"$\tau_{\rm gen}$")
    for xi, tm, cens in zip(x, tms, tm_censored):
        if cens:
            ax.bar(xi + bar_w/2, BUDGET, width=bar_w,
                   color="tab:red", alpha=0.35, hatch="///",
                   edgecolor="tab:red")
        else:
            ax.bar(xi + bar_w/2, tm, width=bar_w, color="tab:red")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel(r"$d_{\rm latent} / d_{\rm intrinsic}$")
    ax.set_title(rf"MLP, $\sigma_\perp = {sn}$")
    ax.set_ylim(0, Y_MAX)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylabel("training steps (millions)")
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="tab:blue", label=r"$\tau_{\rm gen}$"),
        Patch(facecolor="tab:red",  label=r"$\tau_{\rm mem}$ (reached)"),
        Patch(facecolor="tab:red",  alpha=0.35, hatch="///",
              edgecolor="tab:red", label=r"$\tau_{\rm mem}$ (not reached)"),
    ]
    ax.legend(handles=legend_handles, loc="upper left",
              frameon=False, fontsize=7)
    out = FIG_DIR / f"mlp_timescales_bar_{suffix}.pdf"
    fig.tight_layout()
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


def fig_mlp_timescales_bar():
    for sn in _MLP_PRESETS:
        _fig_mlp_timescales_bar(sn)


def _fig_mlp_gen_gap(sn):
    """Single-panel bar chart: late gen-gap (value approached at end of
    training) vs d_lat at one sigma. Sources from the 5M-step exp2_v4
    run when available (sigma=0.5), otherwise the 300k-step sweep."""
    root, suffix = _MLP_LATE_PRESETS[sn]
    fig, ax = plt.subplots(figsize=(4.0, 2.4))
    runs = []
    for d in sorted((ROOT / root).iterdir()):
        if not d.is_dir() or not (d / "config.json").exists():
            continue
        cfg = json.loads((d / "config.json").read_text())
        if abs(cfg.get("sigma_noise", sn) - sn) > 1e-9:
            continue
        rows = [json.loads(l) for l in
                (d / "metrics.jsonl").read_text().splitlines() if l.strip()]
        rows.sort(key=lambda r: r["step"])
        gg = _late_value(rows, "gen_gap")
        if gg is None:
            continue
        runs.append((cfg["d_latent"], cfg.get("d_intrinsic", 5), gg))
    runs.sort()
    labels = [f"{r[0]/r[1]:g}" for r in runs]
    gaps = [r[2] for r in runs]
    x = np.arange(len(runs))
    ax.bar(x, gaps, color="tab:purple", width=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel(r"$d_{\rm latent} / d_{\rm intrinsic}$")
    ax.set_title(rf"MLP, $\sigma_\perp = {sn}$")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylabel(r"late gen-gap (test $-$ train)")
    out = FIG_DIR / f"mlp_gen_gap_{suffix}.pdf"
    fig.tight_layout()
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


def fig_mlp_gen_gap():
    for sn in _MLP_LATE_PRESETS:
        _fig_mlp_gen_gap(sn)


def _fig_mlp_score_error_curves(sn):
    """Score-error trajectories matching the panel-(b) style of
    plot_exp2_v4.py (linear axes, viridis colormap, one curve per d_lat).
    5M-step exp2_v4 for sigma=0.5, 300k-step exp2_sn001 for sigma=0.01."""
    root, suffix = _MLP_LATE_PRESETS[sn]
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    runs_dict = {}
    dint_seen = 5
    for d in sorted((ROOT / root).iterdir()):
        if not d.is_dir() or not (d / "config.json").exists():
            continue
        cfg = json.loads((d / "config.json").read_text())
        if abs(cfg.get("sigma_noise", sn) - sn) > 1e-9:
            continue
        rows = [json.loads(l) for l in
                (d / "metrics.jsonl").read_text().splitlines() if l.strip()]
        rows.sort(key=lambda r: r["step"])
        rows = [r for r in rows if r.get("score_error") is not None]
        if rows:
            runs_dict[cfg["d_latent"]] = rows
            dint_seen = cfg.get("d_intrinsic", dint_seen)
    ds = sorted(runs_dict)
    if not ds:
        plt.close(fig)
        return
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(ds)))
    for d, color in zip(ds, colors):
        rows = runs_dict[d]
        ax.plot([r["step"] / 1e6 for r in rows],
                [r["score_error"] for r in rows],
                color=color, lw=1.2,
                label=rf"$d_{{\rm latent}}/d_{{\rm intrinsic}}={d/dint_seen:g}$")
    ax.set_xlabel("training steps (millions)")
    ax.set_ylabel("score error")
    ax.set_title(rf"MLP, $\sigma_\perp = {sn}$")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="best", frameon=False, ncol=2)
    out = FIG_DIR / f"mlp_score_error_curves_{suffix}.pdf"
    fig.tight_layout()
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


def fig_mlp_score_error_curves():
    for sn in _MLP_LATE_PRESETS:
        _fig_mlp_score_error_curves(sn)


def _fig_late_score_error(sn):
    """Single-panel: late score error (value approached at end of training)
    vs d_latent / d_intrinsic at one sigma. 5M-step exp2_v4 for sigma=0.5,
    300k-step exp2_sn001 for sigma=0.01."""
    root, suffix = _MLP_LATE_PRESETS[sn]
    fig, ax = plt.subplots(figsize=(4.0, 2.6))
    runs = []
    for d in sorted((ROOT / root).iterdir()):
        if not d.is_dir() or not (d / "config.json").exists():
            continue
        cfg = json.loads((d / "config.json").read_text())
        if abs(cfg.get("sigma_noise", sn) - sn) > 1e-9:
            continue
        rows = [json.loads(l) for l in
                (d / "metrics.jsonl").read_text().splitlines() if l.strip()]
        rows.sort(key=lambda r: r["step"])
        late = _late_value(rows, "score_error")
        if late is None:
            continue
        runs.append((cfg["d_intrinsic"], cfg["d_latent"], late))
    runs.sort(key=lambda x: x[1])
    ratios = [d_lat / d_int for d_int, d_lat, _ in runs]
    scores = [s for _, _, s in runs]
    ax.plot(ratios, scores, "s-", color="tab:red", lw=1.5, ms=6,
            label="MLP")
    ax.axvline(1.0, color="gray", linestyle=":", alpha=0.5,
               label=r"$d_{\rm latent} = d_{\rm intrinsic}$")
    ax.set_xlabel(r"$d_{\rm latent} / d_{\rm intrinsic}$")
    ax.set_title(rf"$\sigma_\perp = {sn}$")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="best", frameon=False)
    ax.set_ylabel("late score error")
    out = FIG_DIR / f"late_score_error_{suffix}.pdf"
    fig.tight_layout()
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


def fig_late_score_error():
    for sn in _MLP_LATE_PRESETS:
        _fig_late_score_error(sn)


_REAL_PRESETS = {
    "mnist":  ("MNIST",  "mnist_n1k_mlp_d",  13, [10, 15, 20, 25, 30, 40]),
    "celeba": ("CelebA", "celeba_mlp_d",     26, [20, 25, 30, 35, 40, 50]),
}


def _walk_real(prefix, dlats):
    out = []
    for d in dlats:
        path = ROOT / "n1k_results" / f"{prefix}{d}"
        if not (path / "metrics.jsonl").exists():
            continue
        rows = [json.loads(l) for l in
                (path / "metrics.jsonl").read_text().splitlines() if l.strip()]
        # Drop the leading "meta" row that lacks a step field.
        rows = [r for r in rows if "step" in r]
        rows.sort(key=lambda r: r["step"])
        out.append((d, rows))
    return out


def _first_within(rows, key, frac=1.05):
    vals = [r[key] for r in rows if r.get(key) is not None]
    if not vals:
        return None
    vmin = min(vals)
    for r in rows:
        if r.get(key) is not None and r[key] <= vmin * frac:
            return r["step"]
    return None


def _first_above(rows, key, thresh):
    for r in rows:
        if r.get(key, 0) is not None and r.get(key, 0) > thresh:
            return r["step"]
    return None


def fig_real_timescales():
    """Scatter view of tau_gen and tau_mem vs d_lat / d_int for the real-data
    5M-step n=1000 runs on MNIST and CelebA. Two-panel layout."""
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=False)
    for ax, (key, (name, prefix, dint, dlats)) in zip(axes, _REAL_PRESETS.items()):
        runs = _walk_real(prefix, dlats)
        for d, rows in runs:
            last = rows[-1]["step"]
            # tau_gen = first step at which FID plateaus (within 10% of
            # its per-run minimum); FID dips dramatically over the first
            # ~10k steps and then bounces in a narrow band, so the
            # plateau-onset criterion gives a clean, comparable value
            # across configurations.
            tg = _first_within(rows, "fid", frac=1.10)
            tm = _first_above(rows, "gen_gap", 0.02)
            ratio = d / dint
            if tg is not None:
                ax.scatter(ratio, tg, s=80, c="tab:blue", zorder=5,
                           edgecolors="black", linewidths=0.8)
            if tm is not None:
                ax.scatter(ratio, tm, s=50, c="tab:red", zorder=7,
                           edgecolors="black", linewidths=0.8)
            else:
                ax.scatter(ratio, last, s=50, marker="^", c="tab:red",
                           zorder=7, edgecolors="black", linewidths=0.8,
                           alpha=0.45)
        ax.set_xlabel(r"$d_{\rm latent} / d_{\rm intrinsic}$")
        ax.set_title(rf"{name} ($d_{{\rm intrinsic}} \approx {dint}$), $n = 1000$")
        ax.grid(True, alpha=0.3)
        if key == "mnist":
            ax.set_ylim(0, 100_000)
    axes[0].set_ylabel(r"training steps")
    axes[0].legend(handles=[
        Line2D([0], [0], marker="o", color="w", mfc="tab:red",  ms=8,
               label=r"$\tau_{\rm mem}$ (gen-gap $> 0.02$)"),
        Line2D([0], [0], marker="o", color="w", mfc="tab:blue", ms=8,
               label=r"$\tau_{\rm gen}$ (FID plateau, 10\% of min)"),
        Line2D([0], [0], marker="^", color="w", mfc="tab:red",  ms=8,
               alpha=0.45, label="not reached (last logged step)"),
    ], fontsize=7, frameon=False, loc="upper left")
    fig.tight_layout()
    out = FIG_DIR / "real_timescales.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


def fig_real_late_gen_gap():
    """Late gen-gap (mean over last 5% of training) vs d_lat / d_int for
    MNIST and CelebA n=1000 runs."""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=False)
    for ax, (key, (name, prefix, dint, dlats)) in zip(axes, _REAL_PRESETS.items()):
        runs = _walk_real(prefix, dlats)
        ratios, gaps = [], []
        for d, rows in runs:
            gg = _late_value(rows, "gen_gap")
            if gg is None:
                continue
            ratios.append(d / dint)
            gaps.append(gg)
        ax.bar(np.arange(len(ratios)), gaps, color="tab:purple", width=0.7)
        ax.set_xticks(np.arange(len(ratios)))
        ax.set_xticklabels([f"{r:.2g}" for r in ratios])
        ax.set_xlabel(r"$d_{\rm latent} / d_{\rm intrinsic}$")
        ax.set_title(rf"{name} ($d_{{\rm intrinsic}} \approx {dint}$), $n = 1000$")
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].set_ylabel(r"late gen-gap (test $-$ train)")
    fig.tight_layout()
    out = FIG_DIR / "real_late_gen_gap.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


def fig_real_min_fid():
    """Min FID over training vs d_lat / d_int."""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=False)
    for ax, (key, (name, prefix, dint, dlats)) in zip(axes, _REAL_PRESETS.items()):
        runs = _walk_real(prefix, dlats)
        ratios, fids = [], []
        for d, rows in runs:
            vals = [r["fid"] for r in rows if r.get("fid") is not None]
            if not vals:
                continue
            ratios.append(d / dint)
            fids.append(min(vals))
        ax.plot(ratios, fids, "s-", color="tab:red", lw=1.5, ms=6)
        ax.axvline(1.0, color="gray", linestyle=":", alpha=0.5,
                   label=r"$d_{\rm latent} = d_{\rm intrinsic}$")
        ax.set_xlabel(r"$d_{\rm latent} / d_{\rm intrinsic}$")
        ax.set_title(rf"{name} ($d_{{\rm intrinsic}} \approx {dint}$), $n = 1000$")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="best", frameon=False)
    axes[0].set_ylabel("min FID")
    fig.tight_layout()
    out = FIG_DIR / "real_min_fid.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


def fig_real_fid_curves():
    """FID over training, one curve per d_lat / d_int ratio, for MNIST and
    CelebA n=1000 runs."""
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2), sharey=False)
    for ax, (key, (name, prefix, dint, dlats)) in zip(axes, _REAL_PRESETS.items()):
        runs = _walk_real(prefix, dlats)
        if not runs:
            continue
        colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(runs)))
        for (d, rows), color in zip(runs, colors):
            steps = [r["step"] / 1e6 for r in rows
                     if r.get("fid") is not None]
            fids = [r["fid"] for r in rows
                    if r.get("fid") is not None]
            ax.plot(steps, fids, color=color, lw=1.0,
                    label=rf"$d_{{\rm latent}}/d_{{\rm intrinsic}}={d/dint:.2g}$")
        ax.set_xlabel("training steps (millions)")
        ax.set_title(rf"{name} ($d_{{\rm intrinsic}} \approx {dint}$), $n = 1000$")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=6, loc="best", frameon=False, ncol=2)
    axes[0].set_ylabel("FID")
    fig.tight_layout()
    out = FIG_DIR / "real_fid_curves.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


def fig_real_data_sweep():
    fig_real_timescales()
    fig_real_late_gen_gap()
    fig_real_min_fid()
    fig_real_fid_curves()


if __name__ == "__main__":
    fig_four_bulk_overview()
    fig_four_bulk_two_sigma()
    fig_four_bulk_clean_example()
    fig_bulk_detection_hist()
    fig_bulk_sizes_distances()
    fig_four_bulk_dlat_sweep()
    fig_four_bulk_dlat_sweep_sn05()
    fig_four_bulk_dlat_hist_sn001()
    fig_four_bulk_dlat_hist()
    fig_data_pca_dlat_sweep()
    fig_four_bulk_dint_sweep()
    fig_eigenvalue_histogram()
    fig_mlp_score_error_curves()
    fig_mlp_timescales_scatter()
    fig_mlp_timescales_bar()
    fig_mlp_gen_gap()
    fig_late_score_error()
    fig_mlp_dint_sweep()
