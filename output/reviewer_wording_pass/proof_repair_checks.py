"""Regression checks for the proof repairs; NumPy only, no training required."""
import contextlib
import hashlib
import io
import json
import runpy
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
# Preserve the historical audit and its source hashes when rerunning its checks.
historical = ROOT/'proof_audit_checks.json'
original = historical.read_bytes()
try:
    with contextlib.redirect_stdout(io.StringIO()):
        runpy.run_path(str(ROOT/'proof_audit_checks.py'), run_name='__main__')
    checks = json.loads(historical.read_text())
finally:
    historical.write_bytes(original)

# The repaired absolute-alignment bound holds without a sign assumption.
rng = np.random.default_rng(9281)
max_slack = 0.
for _ in range(1000):
    lam = rng.uniform(.01,3,20); pi = rng.uniform(0,5,20)
    gamma = rng.normal(size=20); T = rng.uniform(0,10)
    mass = lam*pi*(-np.expm1(-lam*T))
    actual = abs(np.sum(gamma*mass)); bound = np.sum(abs(gamma)*mass)
    assert actual <= bound+1e-12
    max_slack = max(max_slack,float(bound-actual))

# Correct strict order-statistic inversion, including tied distances.
D=np.array([1.,1.,2.,3.,3.])
for q in [.01,.2,.21,.4,.6,.8,1.]:
    cutoff=np.sort(D)[::-1][int(np.ceil(q*len(D)))-1]
    for u in [0.,1.,1.5,2.,2.5,3.,4.]:
        assert (np.mean(D>u)>=q)==(u<cutoff)

tex=(ROOT/'sec-appendix-theory.tex').read_text()
main=(ROOT/'sec9-theory.tex').read_text()
for stale in [r'158\to43', r'3200/\lambda', r'T=0.02\dlat',
              'diverges as $c', 'reported fraction cannot exceed',
              'gain rescaled leaves the curve unchanged',
              r'\mathcal D_{t_c}=\{d_{\rm NN}^2\ge2\Delta_{t_c}']:
    assert stale not in tex+main, stale
for required in [r'\tfrac{266}{5}',r'\sum_{\mathcal M_{\rm blk}}|\gamma_i|',
                 r'2e^{2t_c}\Delta_{t_c}', 'diagonal eigenvalue gaps',
                 'this bound does not establish an asymptotic limit']:
    assert required in tex, required
abstract_hash=hashlib.sha256((ROOT/'abstract-rescoped.tex').read_bytes()).hexdigest()
assert abstract_hash=='2ba35c0374704516f7b8cdf9f9bfb60d10efc446dc3f52f5575c3169297c8f38'
checks['repair_regressions']={'absolute_alignment_trials':1000,
                             'empirical_order_statistics_with_ties':'passed',
                             'stale_claim_scan':'passed',
                             'abstract_unchanged':True}
(ROOT/'proof_repair_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
print('Audit algebra and repair regression checks passed; abstract unchanged.')
