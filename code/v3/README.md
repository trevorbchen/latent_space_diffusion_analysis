# v3: Unified Experimental Pipeline

Single codebase for synthetic and real-data diffusion runs (MLP + RFNN), targeting
the placeholder figures in the ICML draft.

## Quick start

```bash
./setup.sh          # create .venv and install deps
source .venv/bin/activate

# synthetic MLP at d_latent=20
python run_experiment.py --data synthetic --model mlp --d_latent 20

# real (MNIST), VAE bottleneck d_latent=10, MLP score
python train_vae.py --dataset mnist --d_latent 10
python run_experiment.py --data mnist --model mlp --d_latent 10

# multi-config dispatch across all visible CUDA GPUs
python runner.py configs/exp_mnist_sweep.yaml
```

## Layout

```
v3/
├── run_experiment.py     # main CLI (synthetic | mnist | celeba) x (mlp | rfnn)
├── train_vae.py          # train + cache VAE encoders for real data
├── runner.py             # multi-GPU config dispatcher
├── plot.py               # post-run plotting helpers (\tau extraction, paper figs)
└── lib/
    ├── data_synthetic.py # anisotropic Gaussian-mixture data
    ├── data_real.py      # MNIST/CelebA -> VAE-encoded latents (cached)
    ├── vae.py            # small MLP/conv VAE with configurable bottleneck
    ├── models.py         # MLPScore, RFNNScore
    ├── true_score.py     # analytic GMM score (synthetic only)
    ├── diffusion.py      # OU forward, Euler-Maruyama reverse SDE
    ├── metrics.py        # score error, mem fraction, MMD, FID, U eigvals
    ├── training.py       # unified training loop + checkpointing
    └── eigenvalues.py    # U(t), per-bulk absorption tracking (RFNN)
```

## Logging

One `metrics.jsonl` per run, one JSON object per eval step. Cadences:

| Metric                     | Cadence       | Cost   |
|---------------------------|---------------|--------|
| train_loss, test_loss     | 1k steps      | cheap  |
| score_error / proxy       | 2k steps      | cheap  |
| memorization_fraction     | 10k steps     | SDE-gen|
| FID (real data)           | 25k steps     | gen+net|
| U eigenvalues (RFNN)      | 25k steps     | eigh   |

For synthetic data we have an analytic true score, so `score_error` is exact.
For real data there is no analytic score; we substitute test-loss-plateau
(which is what the paper uses for tau_gen).

## Reproducing v2

`run_experiment.py --data synthetic --model mlp --d_latent 20 --sigma_noise 0.5`
should reproduce `sigma_noise_0.5/exp2_mlp/raw_data/di5_d20_n500_s42` to within
seed-level noise. Sanity test in `tests/test_port_v2.py`.
