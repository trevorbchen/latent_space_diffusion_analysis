"""Small deterministic checks of repaired algebra, not a proof of spectral universality."""
from pathlib import Path
import json
import numpy as np
rng=np.random.default_rng(20260925)
n,d,p,K=8,3,20,80
t=.12; delta=1-np.exp(-2*t)
x=rng.normal(size=(n,d)); W=rng.normal(size=(p,d))
xt=np.exp(-t)*x[None]+np.sqrt(delta)*rng.normal(size=(K,n,d))
phi=np.tanh(xt@W.T/np.sqrt(d))/np.sqrt(p)
U1=phi[0].T@phi[0]/n
UK=np.einsum('knp,knq->pq',phi,phi)/(K*n)
m=phi.mean(0); B=m.T@m/n
center=phi-m[None]; C=np.einsum('knp,knq->pq',center,center)/(K*n)
rank1=int(np.linalg.matrix_rank(U1,tol=1e-11)); rankK=int(np.linalg.matrix_rank(UK,tol=1e-11))
err=float(np.max(np.abs(UK-B-C)))
assert rank1==n and rankK==p and err<1e-12
# Exact plug-in denoiser and nonzero condensation remainder.
mu=0; xtest=np.exp(-t)*x[mu]+.2*rng.normal(size=d)
a=-np.sum((xtest-np.exp(-t)*x)**2,axis=1)/(2*delta);weights=np.exp(a-a.max());weights/=weights.sum()
semp=np.sum(weights[:,None]*(np.exp(-t)*x-xtest),axis=0)/delta
r=semp-(np.exp(-t)*x[mu]-xtest)/delta
zeta=rng.normal(size=d); shat=semp+zeta
xhat=np.exp(t)*(xtest+delta*shat)
rem_err=float(np.max(np.abs(xhat-x[mu]-np.exp(t)*delta*(zeta+r))))
actual=abs(np.sum((xhat-x[mu])**2)-np.exp(2*t)*delta**2*np.sum(zeta**2))
bound=np.exp(2*t)*delta**2*(2*np.linalg.norm(zeta)*np.linalg.norm(r)+np.sum(r**2))
assert rem_err<1e-12 and actual<=bound+1e-12
# Rank-deficient Gram matrix must admit a zero lower endpoint.
H=rng.normal(size=(2,5));old=(1-np.sqrt(5/2)-1/np.sqrt(2))**2
new=max(1-np.sqrt(5/2)-1/np.sqrt(2),0)**2
assert old>0 and new==0 and np.linalg.matrix_rank(H.T@H)==2
# Projected-Wishart is approached by strong spikes; not an exact finite-spike restriction.
Z=rng.normal(size=(8,20));Z1=Z[:2];Z2=Z[2:]
P=np.eye(20)-Z1.T@np.linalg.solve(Z1@Z1.T,Z1)
projected=np.linalg.eigvalsh(Z2@P@Z2.T/20)
errors={}
for strength in [1.,1e3,1e6]:
 scale=np.sqrt(np.r_[np.full(2,1+strength),np.ones(6)])
 G=(scale[:,None]*Z)@(scale[:,None]*Z).T/20
 errors[str(strength)]=float(np.max(np.abs(np.linalg.eigvalsh(G)[:6]-projected)))
assert errors['1.0']>1e-3 and errors['1000000.0']<1e-5
result={'seed':20260925,'single_copy_rank':rank1,'averaged_rank':rankK,'conditional_covariance_max_error':err,'tweedie_algebra_max_error':rem_err,'tweedie_squared_error_correction':float(actual),'tweedie_correction_bound':float(bound),'rank_deficient_lower_endpoint':new,'finite_spike_vs_projected_eigenvalue_errors':errors,'scope':'Algebra and counterexample checks only; no validation of mixture universality, asymptotic edge accuracy, or sampler reduction.'}
Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
