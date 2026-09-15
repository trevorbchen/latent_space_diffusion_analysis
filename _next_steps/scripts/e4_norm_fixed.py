"""E4: norm-fixed control. Does the buffer ratio still grow with d_lat when the per-coordinate input
variance q_in := E||x||^2 / d_lat is held FIXED (so tanh does not de-saturate)?

Baseline (paper): sigma_perp = 0.5, so q_in = (E_sig + 0.25 (d-5)) / d falls from 2.8 (d=5) to 0.31 (d=200).
Fixed-q control:  sigma_perp^2(d) = (c d - E_sig) / (d - d_int) with E_sig = s^2 + d_int sig_sig^2 = 14,
                  c = 0.57 (= the baseline value at d = 40, so the two sweeps coincide there).
Prediction if the driver is de-saturation (a_1(q), a_*(q) fixed): R = lam_sig_min / lam_samp_max should STOP
growing and decline roughly like n / d_lat. Prediction if the driver is the mode count d_lat - d_int: R keeps growing.
Same RFNN as rfnn_controls.py: t = 0.01, n = 500, p = 4096, 3 seeds.  Run: python3 _next_steps/scripts/e4_norm_fixed.py (~10 min)
"""
import numpy as np, time
T=0.01; E2T=np.exp(-2*T); DELTA=1-E2T; D_INT,K,S,SIG_SIG=5,10,3.0,1.0; E_SIG=S**2+D_INT*SIG_SIG**2
def spec(d,n,p,seed,sig_perp,n_mc=40):
    rng=np.random.default_rng(seed)
    C=rng.normal(size=(K,D_INT)); C/=np.linalg.norm(C,axis=1,keepdims=True); C*=S
    lab=rng.integers(0,K,size=n); x=np.zeros((n,d)); x[:,:D_INT]=C[lab]+SIG_SIG*rng.normal(size=(n,D_INT))
    if d>D_INT: x[:,D_INT:]=sig_perp*rng.normal(size=(n,d-D_INT))
    W=rng.normal(size=(p,d)); U=np.zeros((p,p))
    for i0 in range(0,n,64):
        xb=x[i0:i0+64]; b=xb.shape[0]; eta=rng.normal(size=(n_mc,b,d)); xt=np.exp(-T)*xb[None]+np.sqrt(DELTA)*eta
        Phi=np.tanh(xt.reshape(-1,d)@W.T/np.sqrt(d))/np.sqrt(p); U+=Phi.T@Phi
    U/=(n*n_mc); ev=np.linalg.eigvalsh(U)[::-1]
    return dict(sig_min=ev[D_INT-1], nd_max=ev[D_INT], nd_min=ev[d-1], samp_max=ev[d], samp_med=np.median(ev[d:n]), tr=ev.sum(), qin=float((x**2).sum(1).mean()/d))
N=500; P=4096; SEEDS=[11,22,33]; DL=[40,60,80,120,160,240]; C_FIX=0.57
out=[]
for name,spfun in (("BASELINE sigma_perp=0.5", lambda d:0.5), (f"FIXED q_in={C_FIX}", lambda d: np.sqrt((C_FIX*d-E_SIG)/(d-D_INT)))):
    out.append(f"\n=== {name}  (n={N}, p={P}, t={T}, 3 seeds) ===")
    out.append(f"{'d_lat':>6}{'sig_perp':>10}{'q_in':>8}{'tr(U)':>8}{'sig_min':>11}{'nd_max':>11}{'nd_min':>11}{'samp_max':>11}{'R=sig/samp':>12}{'R*d/n':>8}")
    R0=None
    for d in DL:
        sp=spfun(d); t0=time.time(); rs=[spec(d,N,P,s,sp) for s in SEEDS]; m={k:float(np.mean([r[k] for r in rs])) for k in rs[0]}
        R=m['sig_min']/m['samp_max']; R0=R0 or R
        out.append(f"{d:>6}{sp:>10.3f}{m['qin']:>8.3f}{m['tr']:>8.3f}{m['sig_min']:>11.3e}{m['nd_max']:>11.3e}{m['nd_min']:>11.3e}{m['samp_max']:>11.3e}{R:>12.1f}{R*d/N:>8.2f}   [{time.time()-t0:.0f}s]")
        print(out[-1],flush=True)
    out.append(f"   R growth d=40->240: {R/R0:.2f}x   (mode-count story: grows ~{(240-5)/(40-5):.1f}x; fixed-q de-saturation story: falls toward ~{40/240:.2f}x)")
    print(out[-1],flush=True)
open("_next_steps/e4_norm_fixed_results.txt","w").write("\n".join(out)+"\n"); print("\nsaved _next_steps/e4_norm_fixed_results.txt")
