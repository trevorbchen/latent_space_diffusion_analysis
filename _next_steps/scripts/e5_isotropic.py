"""E5: isotropic control for the buffer ratio.

Two data models, same RFNN feature map as rfnn_controls.py, t = 0.01, n = 500, p = 4096 fixed
(edges are p-independent; see rfnn_controls.py), 3 seeds:
  (a) ISO:  x ~ N(0, I_{d_lat})               -- no clusters, no signal/noise split at all
  (b) SP1:  cluster model with sigma_perp = 1   -- clusters exist, but sigma_perp = sigma_sig (isotropic *noise*)
For each: top-d_lat block ("data" modes), next n-d_lat ("sample" modes).
Reported: R_on = min(data)/max(sample) (onset ratio), R_med = median(data)/median(sample), data edges.
The question: does the ratio still grow with d_lat when the data carry no low-dimensional structure?
Run: python3 _next_steps/scripts/e5_isotropic.py   (~5 min)
"""
import numpy as np, time
T=0.01; E2T=np.exp(-2*T); DELTA=1-E2T; D_INT,K,S,SIG_SIG=5,10,3.0,1.0
def spec(d, n, p, seed, model, sig_perp=1.0, n_mc=40):
    rng=np.random.default_rng(seed)
    if model=="ISO": x=rng.normal(size=(n,d))
    else:
        C=rng.normal(size=(K,D_INT)); C/=np.linalg.norm(C,axis=1,keepdims=True); C*=S
        lab=rng.integers(0,K,size=n); x=np.zeros((n,d)); x[:,:D_INT]=C[lab]+SIG_SIG*rng.normal(size=(n,D_INT))
        if d>D_INT: x[:,D_INT:]=sig_perp*rng.normal(size=(n,d-D_INT))
    W=rng.normal(size=(p,d)); U=np.zeros((p,p))
    for i0 in range(0,n,64):
        xb=x[i0:i0+64]; b=xb.shape[0]; eta=rng.normal(size=(n_mc,b,d))
        xt=np.exp(-T)*xb[None]+np.sqrt(DELTA)*eta
        Phi=np.tanh(xt.reshape(-1,d)@W.T/np.sqrt(d))/np.sqrt(p); U+=Phi.T@Phi
    U/=(n*n_mc); ev=np.linalg.eigvalsh(U)[::-1]
    data=ev[:d]; samp=ev[d:d+n-d]
    return dict(dmin=data.min(), dmed=np.median(data), dmax=data.max(), smax=samp.max(), smed=np.median(samp),
                sig_min=ev[:D_INT].min() if model!="ISO" else np.nan, nd_max=ev[D_INT:d].max() if (model!="ISO" and d>D_INT) else np.nan,
                tr=ev.sum())
N=500; P=4096; SEEDS=[11,22,33]; DL=[5,10,20,40,80,160]
out=[]
for model in ["ISO","SP1"]:
    out.append(f"\n=== {model}: {'x ~ N(0, I_d)' if model=='ISO' else 'cluster model, sigma_perp = 1'}  (n={N}, p={P}, t={T}, 3 seeds) ===")
    out.append(f"{'d_lat':>6}{'data_min':>11}{'data_med':>11}{'data_max':>11}{'samp_max':>11}{'samp_med':>11}{'R_on':>9}{'R_med':>9}{'sig_min/samp_max':>18}{'tr':>8}")
    R0=None
    for d in DL:
        t0=time.time(); rs=[spec(d,N,P,s,model) for s in SEEDS]
        m={k:float(np.mean([r[k] for r in rs])) for k in rs[0]}
        R_on=m['dmin']/m['smax']; R_med=m['dmed']/m['smed']; Rs=m['sig_min']/m['smax'] if not np.isnan(m['sig_min']) else float('nan')
        if R0 is None: R0=(R_on,R_med)
        out.append(f"{d:>6}{m['dmin']:>11.3e}{m['dmed']:>11.3e}{m['dmax']:>11.3e}{m['smax']:>11.3e}{m['smed']:>11.3e}{R_on:>9.2f}{R_med:>9.2f}{Rs:>18.2f}{m['tr']:>8.3f}   [{time.time()-t0:.0f}s]")
        print(out[-1], flush=True)
    out.append(f"   R_on growth d=5->160: {R_on/R0[0]:.2f}x   R_med growth: {R_med/R0[1]:.2f}x")
    print(out[-1], flush=True)
open("_next_steps/e5_isotropic_results.txt","w").write("\n".join(out)+"\n"); print("\nsaved _next_steps/e5_isotropic_results.txt")
