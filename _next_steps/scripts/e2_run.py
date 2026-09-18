"""E2 runs: re-train the paper's fixed-width synthetic MLP config, saving what the original sweep did not:
generated samples, model checkpoints, the training set, the GMM means and the rotation Q.

The paper's training code (experiment_v2.py) is imported UNMODIFIED; this wrapper only hooks
generate_data (to save the data once) and generate_samples (to save samples + weights at selected evals).
Config is copied field-for-field from multiseed_runs/exp2_mlp_dlat_sn05_5m_bonnaire_hidden256/*/config.json.
Same seed => same training set and Q as the paper's run of that (d_lat, seed).

  python3 _next_steps/scripts/e2_run.py --d_latent 40 --seed 42
"""
import argparse, sys, os, numpy as np, torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
torch.set_num_threads(1)
import experiment_v2 as E

ap = argparse.ArgumentParser(); ap.add_argument("--d_latent", type=int, required=True); ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--total_steps", type=int, default=5_000_000); ap.add_argument("--out_root", default="_next_steps/e2_runs")
a = ap.parse_args()
out = os.path.join(a.out_root, f"di5_d{a.d_latent}_n500_s{a.seed}"); os.makedirs(os.path.join(out, "samples"), exist_ok=True)
EVAL = 50_000
cfg = E.Config(d_intrinsic=5, d_latent=a.d_latent, k=10, sigma_noise=0.5, sigma_signal=1.0, scale=3.0, n=500, hidden=256, n_freq=32,
               lr=1e-4, batch_size=256, total_steps=a.total_steps, t_min=0.01, t_max=3.0, eval_interval=EVAL,
               n_gen_samples=5000, n_sde_steps=500, t_eval=0.1, seed=a.seed, results_dir=out)

_gd = E.generate_data
def generate_data(*args, **kw):
    res = _gd(*args, **kw); data, labels, means, Q = res[:4]
    to = lambda x: x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
    np.savez(os.path.join(out, "data.npz"), train=to(data), labels=to(labels), means=to(means), Q=to(Q)); return res
E.generate_data = generate_data

_gs = E.generate_samples; _k = {"i": 0}
def generate_samples(model, *args, **kw):
    x = _gs(model, *args, **kw); i = _k["i"]; _k["i"] += 1; step = 1 if i == 0 else i * EVAL
    if step > 1 and (step <= 1_000_000 or step % 250_000 == 0):          # dense early, sparse late
        np.save(os.path.join(out, "samples", f"gen_step{step:08d}.npy"), x.detach().cpu().numpy().astype(np.float32))
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, os.path.join(out, "samples", f"model_step{step:08d}.pt"))
    return x
E.generate_samples = generate_samples
E.run_experiment(cfg)
