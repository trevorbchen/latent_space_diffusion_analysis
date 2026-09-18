"""E2b: where does the learned score's error live? Per-coordinate squared distance of s_theta (t = 0.1) to
(i) the true population score and (ii) the exact empirical score of the training set, split into the d_int signal
coordinates and the d_lat - d_int null coordinates (after un-rotating by Q). Evaluated on noised held-out points, as
the paper's score-error metric is. Uses the checkpoints saved by e2_run.py and the FIXED true score."""
import glob, os, re, sys, math, numpy as np, torch, csv
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import experiment_v2 as E
torch.set_num_threads(4); DINT=5; T=0.1; ENT=math.exp(-T); DEL=1-math.exp(-2*T); STEPS=[50000,250000,500000,1000000,2000000,3000000,5000000]
def s_emp(x, train):   # exact score of (1/n) sum_mu N(e^-t x_mu, Delta I)
    m=ENT*train; d2=torch.cdist(x,m)**2; w=torch.softmax(-d2/(2*DEL),dim=1); return (w@m - x)/DEL
rows=[]
for run in sorted(glob.glob("_next_steps/e2_runs/di5_d*_n500_s*")):
    mm=re.search(r"_d(\d+)_n500_s(\d+)",run); d,seed=int(mm.group(1)),int(mm.group(2)); z=np.load(run+"/data.npz")
    train=torch.tensor(z["train"]); Q=torch.tensor(z["Q"]); means=torch.tensor(z["means"])
    test=E.generate_test_samples(means,10,DINT,d,Q,sigma_noise=0.5,n=4096,seed=9999,sigma_signal=1.0)
    g=torch.Generator().manual_seed(123); x_t=ENT*test+math.sqrt(DEL)*torch.randn(test.shape,generator=g)
    si,ld=E.precompute_score_params(DINT,d,Q,0.5,T,sigma_signal=1.0); st=E.true_score(x_t,T,means,sigma_t_inv=si,log_det_sigma_t=ld); se=s_emp(x_t,train)
    blk=lambda v: ((v@Q)[:,:DINT].pow(2).mean().item(), (v@Q)[:,DINT:].pow(2).mean().item() if d>DINT else float('nan'))
    ref=blk(se-st)   # how far the empirical score itself is from the population score, per coordinate
    model=E.MLPScore(d,hidden=256,n_freq=32)
    for step in STEPS:
        f=f"{run}/samples/model_step{step:08d}.pt"
        if not os.path.exists(f): continue
        model.load_state_dict(torch.load(f)); model.eval()
        with torch.no_grad(): sp=model(x_t, torch.full((len(x_t),),T))
        a=blk(sp-st); b=blk(sp-se)
        rows.append(dict(d_lat=d,seed=seed,step=step,true_sig=a[0],true_null=a[1],emp_sig=b[0],emp_null=b[1],ref_sig=ref[0],ref_null=ref[1],total_true=(sp-st).pow(2).sum(1).mean().item()/d))
with open("_next_steps/e2_score_blocks.csv","w",newline="") as fh: w=csv.DictWriter(fh,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
M=lambda d,s,k: np.nanmean([r[k] for r in rows if r["d_lat"]==d and r["step"]==s])
print("per-coordinate squared error at t = 0.1 (mean of 2 seeds).  'ref' = what a model that EXACTLY fit the training set would score.")
print(f"{'d_lat':>6}{'step':>9} | {'to TRUE: signal':>16}{'null':>8} | {'to EMPIRICAL: signal':>21}{'null':>8} | {'total/d':>8}")
for d in sorted({r['d_lat'] for r in rows}):
    for s in STEPS: print(f"{d:>6}{s:>9} | {M(d,s,'true_sig'):>16.3f}{M(d,s,'true_null'):>8.3f} | {M(d,s,'emp_sig'):>21.3f}{M(d,s,'emp_null'):>8.3f} | {M(d,s,'total_true'):>8.3f}")
    print(f"{d:>6}{'ref':>9} | {M(d,STEPS[0],'ref_sig'):>16.3f}{M(d,STEPS[0],'ref_null'):>8.3f} |"); print()
