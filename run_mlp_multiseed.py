"""Run the 5-seed MLP headline sweep.

This is the multi-seed version of the d_latent/tau headline experiment.
It keeps each seed in its own directory so the paper figures can report
mean +/- 95% CI across seeds.
"""

import argparse

from experiment_v2 import Config, run_experiment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", default="multiseed_runs/exp2_mlp_dlat_sn05")
    parser.add_argument("--d-intrinsic", type=int, default=5)
    parser.add_argument("--d-latents", type=int, nargs="+",
                        default=[5, 10, 20, 40])
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--sigma-noise", type=float, default=0.5)
    parser.add_argument("--sigma-signal", type=float, default=1.0)
    parser.add_argument("--scale", type=float, default=3.0)
    parser.add_argument("--steps", type=int, default=300000)
    parser.add_argument("--eval-interval", type=int, default=5000)
    parser.add_argument("--n-gen", type=int, default=5000)
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[42, 43, 44, 45, 46])
    parser.add_argument("--hidden-mult", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=None,
                        help="If set, use this fixed hidden width for every d_latent.")
    args = parser.parse_args()

    for dlat in args.d_latents:
        if dlat < args.d_intrinsic:
            print(f"skip d_latent={dlat} < d_intrinsic={args.d_intrinsic}",
                  flush=True)
            continue
        for seed in args.seeds:
            hidden = args.hidden if args.hidden is not None else args.hidden_mult * dlat
            out = (
                f"{args.base_dir}/di{args.d_intrinsic}_d{dlat}"
                f"_n{args.n}_s{seed}"
            )
            cfg = Config(
                d_intrinsic=args.d_intrinsic,
                d_latent=dlat,
                n=args.n,
                hidden=hidden,
                sigma_noise=args.sigma_noise,
                sigma_signal=args.sigma_signal,
                scale=args.scale,
                total_steps=args.steps,
                eval_interval=args.eval_interval,
                n_gen_samples=args.n_gen,
                seed=seed,
                results_dir=out,
            )
            print(
                f"run d_intrinsic={args.d_intrinsic} d_latent={dlat} "
                f"hidden={hidden} seed={seed} -> {out}",
                flush=True,
            )
            run_experiment(cfg)


if __name__ == "__main__":
    main()
