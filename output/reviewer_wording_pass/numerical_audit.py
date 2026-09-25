"""Reproduce the manuscript consistency audit; run from any working directory.

Reads existing CSVs, samples and checkpoints. No training or source-data writes.
Outputs numerical_audit.json next to this script.
"""
from pathlib import Path
import csv
import hashlib
import json
import math
import os
import statistics as stats
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/iclr-audit-mpl")
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import experiment_v2 as E

torch.set_num_threads(4)
sources = {}


def source(relative):
    path = ROOT / relative
    sources[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path


def rows(relative):
    with source(relative).open() as f:
        return list(csv.DictReader(f))


mem = rows("_next_steps/e2_results.csv")
blocks = rows("_next_steps/e2_score_blocks.csv")
scores = rows("_next_steps/e_scoreerr/corrected_score_error_paper_h256_5seed_sn05_5M.csv")
result = {"memorization": {}, "alignment": {}, "score_error": {}, "preactivation": []}
for d in (5, 10, 20, 40):
    for n in (2, 5):
        selected = lambda r: int(r["d_lat"]) == d and int(r["step"]) == 5000000 and int(r["seed"]) in range(42, 42+n)
        mr = [r for r in mem if selected(r)]
        br = [r for r in blocks if selected(r)]
        assert len(mr) == len(br) == n
        result["memorization"][f"d{d}_{n}seeds"] = {
            k: 100*stats.mean(float(r[k]) for r in mr) for k in ("mem_full", "mem_sig")}
        result["alignment"][f"d{d}_{n}seeds"] = stats.mean(
            (float(r["true_sig"])+float(r["ref_sig"])-float(r["emp_sig"]))/ (2*float(r["ref_sig"])) for r in br)
    early = [r for r in blocks if int(r["d_lat"]) == d and int(r["step"]) == 250000 and int(r["seed"]) in range(42,47)]
    assert len(early) == 5
    result["alignment"][f"d{d}_5seeds_250k"] = stats.mean(
        (float(r["true_sig"])+float(r["ref_sig"])-float(r["emp_sig"]))/(2*float(r["ref_sig"])) for r in early)
for d in sorted({int(r["d_lat"]) for r in scores}):
    rs = [r for r in scores if int(r["d_lat"]) == d and int(r["step"]) == 5000000]
    assert len(rs) == 5
    values = [float(r["corrected_error_per_dim"]) for r in rs]
    for r in rs:
        assert math.isclose(float(r["test_loss"])/(1-math.exp(-.2))-float(r["C_per_dim"]),float(r["corrected_error_per_dim"]),abs_tol=1e-10)
    result["score_error"][str(d)] = {"mean": stats.mean(values), "population_sd": stats.pstdev(values)}

# Independent raw-sample check of the disputed full-space d=20 result.
raw_mem = []
for seed in range(42,47):
    run = Path(f"_next_steps/e2_runs/di5_d20_n500_s{seed}")
    train = torch.tensor(np.load(source(run/"data.npz"))["train"])
    gen = torch.tensor(np.load(source(run/"samples/gen_step05000000.npy")))
    distances = torch.cdist(gen,train).topk(2,largest=False).values
    fraction = (distances[:,0]/distances[:,1] < 1/3).float().mean().item()
    saved = next(float(r["mem_full"]) for r in mem if int(r["d_lat"]) == 20 and int(r["seed"]) == seed and int(r["step"]) == 5000000)
    assert abs(fraction-saved) < 1e-7
    raw_mem.append(fraction)
result["raw_mem_d20_percent"] = 100*stats.mean(raw_mem)

# Use the same held-out/noising protocol as e2_score_blocks.py.
# h includes the learned bias and the deterministic Fourier time embedding.
# q2 is an uncentered second moment, not a centered variance.
for d in (5,40):
    for seed in range(42,47):
        run = Path(f"_next_steps/e2_runs/di5_d{d}_n500_s{seed}")
        z = np.load(source(run/"data.npz"))
        Q, means = torch.tensor(z["Q"]), torch.tensor(z["means"])
        test = E.generate_test_samples(means,10,5,d,Q,sigma_noise=.5,n=4096,seed=9999,sigma_signal=1.)
        x = math.exp(-.1)*test+math.sqrt(1-math.exp(-.2))*torch.randn(test.shape,generator=torch.Generator().manual_seed(123))
        model = E.MLPScore(d,hidden=256,n_freq=32)
        for step in (50000,250000,1000000,5000000):
            model.load_state_dict(torch.load(source(run/f"samples/model_step{step:08d}.pt"),weights_only=True))
            model.eval()
            with torch.no_grad():
                emb = torch.cat([torch.sin(.1*model.freqs),torch.cos(.1*model.freqs)]).expand(len(x),-1)
                h = model.net[0](torch.cat([x,emb],dim=1))
            result["preactivation"].append({"d_lat":d,"seed":seed,"step":step,
                "q2":h.square().mean().item(),"global_variance":h.var(unbiased=False).item(),
                "mean_neuron_variance":h.var(0,unbiased=False).mean().item()})
result["preactivation_summary"] = {}
for step in (50000,250000,1000000,5000000):
    vals = {d:stats.mean(r["q2"] for r in result["preactivation"] if r["d_lat"] == d and r["step"] == step) for d in (5,40)}
    result["preactivation_summary"][str(step)] = {"d5":vals[5],"d40":vals[40],"ratio_of_means":vals[40]/vals[5]}
result["mem_d5_50k_percent"] = 100*stats.mean(float(r["mem_sig"]) for r in mem if int(r["d_lat"]) == 5 and int(r["step"]) == 50000)
for p in ("experiment_v2.py", "_next_steps/scripts/e2_score_blocks.py", "_next_steps/scripts/e2_run.py", "_next_steps/e_scoreerr/compute_corrected_score_error.py"):
    source(p)
result["source_sha256"] = sources
(HERE/"numerical_audit.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps({k:v for k,v in result.items() if k not in ("source_sha256","preactivation")},indent=2))
