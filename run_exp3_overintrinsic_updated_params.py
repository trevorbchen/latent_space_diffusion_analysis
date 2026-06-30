"""Run the Exp 3 over-intrinsic RFNN sweep with updated clean parameters."""

from pathlib import Path

from exp3_overintrinsic_rfnn import Config, feature_width, run_one


OUT_ROOT = Path("clean figures") / "exp3_overintrinsic_rfnn_updated_params"
D_INTRINSICS = [2, 5, 8, 12, 16, 20, 25, 30, 35, 40]


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    shared = dict(
        sigma_noise=0.3,
        center_scale=1.5,
        sigma_signal=1.0,
        mc_samples=500,
        rank_null=300,
        d_latent=20,
        n=500,
        t=0.01,
        seed=42,
    )
    configs = [
        Config(name="p_n_plus_d_plus_r", p_mode="n_plus_d_plus_r", **shared),
        Config(name="p64", p_mode="p64", **shared),
        Config(name="p_fixed", p_mode="fixed", p_fixed=1800, **shared),
    ]
    for cfg in configs:
        run_one(OUT_ROOT, cfg, D_INTRINSICS)

    (OUT_ROOT / "README.md").write_text(
        f"""# Exp 3 over-intrinsic RFNN, updated parameters

Clean-figure version of the Exp 3 source-dimensionality sweep using the
updated four-bulk microscope parameters.

Fixed choices:

- `d_latent = 20`
- `d_intrinsic = {D_INTRINSICS}`
- `n = 500`
- `sigma_signal = 1.0`
- `sigma_noise = 0.3`
- `center_scale = 1.5`
- `t = 0.01`
- `mc_samples = 500`
- `rank_null = 300`

Width folders:

- `p_n_plus_d_plus_r`: `p = d_latent + n + 300 = {feature_width(configs[0])}`
- `p64`: `p = 64*d_latent = {feature_width(configs[1])}`
- `p_fixed`: `p = {feature_width(configs[2])}`

For `d_intrinsic > d_latent`, data are generated in source dimension
`d_intrinsic` and compressed into the fixed observed latent space by a random
energy-preserving map.
""",
        encoding="utf-8",
    )
    print(f"wrote {OUT_ROOT}")


if __name__ == "__main__":
    main()
