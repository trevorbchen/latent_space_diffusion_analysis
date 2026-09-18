"""E2: dimension-controlled memorization. Recompute the paper's NN-ratio test (Bonnaire's form, d(gen,NN1)/d(gen,NN2) < 1/3)
(a) in the full latent space, (b) after projecting generated and training points onto the true d_int-dim signal subspace,
(c) on the null subspace only -- for the model's samples and for three model-free controls:
   fresh      : new draws from the true population              -> chance level (no copying)
   full_copy  : training point + N(0, v_T I) in ALL coordinates  -> a sampler that reproduces the whole latent vector
   sig_copy   : training point's SIGNAL coords + N(0, v_T), NULL coords redrawn from N(0, sigma_perp^2)
                                                                 -> "copies the content, regenerates the null noise"
v_T = 0.0269 is the Euler-Maruyama terminal per-coordinate variance of the paper's sampler (appendix, bridge subsection).
Usage: python3 _next_steps/scripts/e2_analyze.py [--root _next_steps/e2_runs]
"""
import argparse, glob, json, os, re, sys, numpy as np, torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import experiment_v2 as E
ap=argparse.ArgumentParser(); ap.add_argument("--root",default="_next_steps/e2_runs"); ap.add_argument("--out",default="_next_steps/e2_results.csv"); a=ap.parse_args()
DINT=5; SIGP=0.5; SIGS=1.0; VT=0.0269
def frac(gen, train):
    d=torch.cdist(torch.as_tensor(gen,dtype=torch.float32), torch.as_tensor(train,dtype=torch.float32)); nn=torch.topk(d,2,dim=1,largest=False).values
    r=nn[:,0]/(nn[:,1]+1e-10); return float((r<1/3).float().mean()), float(r.mean())
def three(gen_o, tr_o):   # inputs already un-rotated: signal = first DINT coords
    out={"full":frac(gen_o,tr_o), "sig":frac(gen_o[:,:DINT],tr_o[:,:DINT])}
    out["null"]=frac(gen_o[:,DINT:],tr_o[:,DINT:]) if gen_o.shape[1]>DINT else (float('nan'),float('nan')); return out
rows=[]
for run in sorted(glob.glob(os.path.join(a.root,"di5_d*_n500_s*"))):
    m=re.search(r"_d(\d+)_n500_s(\d+)",run); d,seed=int(m.group(1)),int(m.group(2))
    if not os.path.exists(run+"/data.npz"): continue
    z=np.load(run+"/data.npz"); tr=z["train"]; Q=z["Q"]; means=z["means"]; tr_o=tr@Q          # data = data_orig @ Q.T  =>  data @ Q = data_orig
    if d>DINT: assert abs(tr_o[:,DINT:].var()-SIGP**2)<0.03, ("un-rotation check failed", tr_o[:,DINT:].var())
    rng=np.random.default_rng(1000+seed); N=5000
    fresh=E.generate_test_samples(torch.tensor(means),10,DINT,d,torch.tensor(Q),sigma_noise=SIGP,n=N,seed=777+seed,sigma_signal=SIGS).numpy()@Q
    mu=rng.integers(0,len(tr_o),N); full_copy=tr_o[mu]+np.sqrt(VT)*rng.standard_normal((N,d))
    sig_copy=full_copy.copy()
    if d>DINT: sig_copy[:,DINT:]=SIGP*rng.standard_normal((N,d-DINT))
    ctrl={k:three(v,tr_o) for k,v in (("fresh",fresh),("full_copy",full_copy),("sig_copy",sig_copy))}
    logged={}
    if os.path.exists(run+"/metrics.jsonl"):
        for l in open(run+"/metrics.jsonl"):
            j=json.loads(l); logged[int(j["step"])]=j.get("memorization_fraction")
    for f in sorted(glob.glob(run+"/samples/gen_step*.npy")):
        step=int(re.search(r"step(\d+)",f).group(1)); g=np.load(f)@Q; t=three(g,tr_o)
        rows.append(dict(d_lat=d,seed=seed,step=step,mem_full=t["full"][0],mem_sig=t["sig"][0],mem_null=t["null"][0],nn_full=t["full"][1],nn_sig=t["sig"][1],
                         logged=logged.get(step),**{f"{c}_{s}":ctrl[c][s][0] for c in ctrl for s in ("full","sig","null")}))
import csv
if rows:
    with open(a.out,"w",newline="") as fh: w=csv.DictWriter(fh,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
ds=sorted({r["d_lat"] for r in rows}); 
print("MODEL-FREE CONTROLS (fraction flagged as memorized by the 1/3 test; mean over seeds)")
print(f"{'d_lat':>6} | {'fresh: full':>11} {'sig':>6} | {'full_copy: full':>15} {'sig':>6} | {'sig_copy: full':>14} {'sig':>6} {'null':>6}")
for d in ds:
    R=[r for r in rows if r["d_lat"]==d]; mean=lambda k: np.nanmean([r[k] for r in R])
    print(f"{d:>6} | {mean('fresh_full'):>11.3f} {mean('fresh_sig'):>6.3f} | {mean('full_copy_full'):>15.3f} {mean('full_copy_sig'):>6.3f} | {mean('sig_copy_full'):>14.3f} {mean('sig_copy_sig'):>6.3f} {mean('sig_copy_null'):>6.3f}")
print("\nMODEL SAMPLES (mean over seeds)"); steps=sorted({r["step"] for r in rows}); show=[s for s in steps if s in (100000,250000,500000,1000000,2000000,3000000,5000000)] or steps[-3:]
print(f"{'d_lat':>6} {'step':>9} | {'mem_full':>9} {'(logged)':>9} | {'mem_sig':>8} {'fresh_sig':>10} | {'mem_null':>9}")
for d in ds:
    for s in show:
        R=[r for r in rows if r["d_lat"]==d and r["step"]==s]
        if not R: continue
        mean=lambda k: np.nanmean([r[k] for r in R if r[k] is not None])
        print(f"{d:>6} {s:>9} | {mean('mem_full'):>9.3f} {mean('logged'):>9.3f} | {mean('mem_sig'):>8.3f} {mean('fresh_sig'):>10.3f} | {mean('mem_null'):>9.3f}")
print(f"\nrows: {len(rows)}  -> {a.out}")
