"""Deterministic checks of proof-audit findings; run with Python and NumPy.

Counterexamples test stated algebraic implications, not simulated DSM datasets.
This is not a formal verification of the manuscript or its experiments.
"""
import hashlib
import json
import math
from fractions import Fraction as F
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
out = {}

# Same objective and parameter normalization as experiment_v2_rfnn.py.
rng = np.random.default_rng(20260925)
d, n, p = 5, 11, 17
delta = -math.expm1(-0.02)
phi = rng.normal(size=(n, p)) / math.sqrt(p)
A = rng.normal(size=(d, p))
noise = rng.normal(size=(n, d))
y = -noise / math.sqrt(delta)
error = phi @ A.T - y
theory_loss = np.sum(error**2) / (2*n)
logged_loss = np.sum((math.sqrt(delta)*(phi @ A.T)+noise)**2)/(d*n)
theory_gradient = error.T @ phi / n
logged_gradient = 2*math.sqrt(delta)*(math.sqrt(delta)*(phi @ A.T)+noise).T @ phi/(d*n)
assert np.allclose(logged_gradient, (2*delta/d)*theory_gradient)
assert np.isclose(logged_loss, (2*delta/d)*theory_loss)
out['clock'] = {'logged_loss_over_theory_loss': logged_loss/theory_loss,
                'expected_ratio': 2*delta/d, 'flow_time_per_default_step': 0.02,
                'assumptions': 'default lr=0.01*d/Delta; zero momentum; normalized features'}
out['clock']['signal_steps_from_rounded_paper_eigenvalues'] = [
    {'d': dim, 'lambda_code': lam, 'paper_conversion': 3200/lam,
     'correct_flow_efold_steps': 3200*dim/lam,
     'exact_mean_Euler_efold_steps': -1/math.log1p(-0.02*lam/(64*dim))}
    for dim, lam in [(5,20.30),(200,73.88)]]

# Negative gamma satisfies gamma <= gamma0 but not the needed absolute bound.
u = 1-math.exp(-1)
out['alignment_bound'] = {'gamma': -1, 'gamma0': 0.5,
                          'absolute_error': u, 'claimed_upper_bound': 0.5}
assert u > 0.5
out['positive_alignment_not_monotone'] = {
    'gamma': 0.6, 'T': 2,
    'population_loss_derivative': math.exp(-2)*(1-math.exp(-2)-0.6)}
assert out['positive_alignment_not_monotone']['population_loss_derivative'] > 0

out['diffusion_floor'] = {'t': 0.01, 'Delta': delta,
                          'clock_limit_as_aspect_ratio_goes_to_one': 1/(4*delta)}
matrix = np.array([[1., 0.9], [0.9, 1.]])
out['interlacing'] = {'matrix': matrix.tolist(),
                     'full_min': float(np.linalg.eigvalsh(matrix)[0]),
                     'principal_block_min': 1.0}
assert np.linalg.eigvalsh(matrix)[0] < 1

# Formal tanh series through x^9, using exact Gaussian moments. With s=sqrt(q),
# E[tanh(s Z)^2] - E[Z tanh(s Z)]^2 is the residual Hermite energy.
tanh = {1:F(1), 3:F(-1,3), 5:F(2,15), 7:F(-17,315), 9:F(62,2835)}
def moment(k):
    return F(0) if k%2 else F(math.prod(range(1,k,2)))
c1 = {k: v*moment(k+1) for k,v in tanh.items()}
residual = {}
for k,v in tanh.items():
    for j,w in tanh.items():
        if k+j <= 10:
            residual[(k+j)//2] = residual.get((k+j)//2,F(0)) + v*w*moment(k+j)-c1[k]*c1[j]
assert residual[3] == F(2,3) and residual[4] == F(-16,3)
assert residual[5] == F(532,15)
out['Hermite_residual'] = {'q_coefficients': {str(k):str(v) for k,v in residual.items()},
                           'correct_factored_q2_coefficient': str(residual[5]/residual[3]),
                           'paper_factored_q2_coefficient': '50'}

# Trace alone does not make a cloud covariance diffuse in operator norm.
C = np.diag([1.]+[0.]*99)
eps = 1e-6
gram = np.array([[1.,eps],[eps,1.]])
_, vec = np.linalg.eigh(gram)
out['localization'] = {'cloud_trace': float(np.trace(C)), 'cloud_operator_norm': 1.,
                       'trace_over_p': 0.01, 'gram_off_diagonal': eps,
                       'eigenvector_squared_coordinates': (vec**2).tolist()}
assert np.allclose(vec**2,0.5)

# The variance mismatch follows directly from the two W conventions.
out['fan_in'] = {'d':20, 'sigma_w_squared':0.05, 'x_norm_squared':20,
                 'variance_with_extra_sqrt_d':0.05, 'claimed_q':1.0}

# At fixed q, changing gain changes rho and hence coherent residual covariance.
nodes, weights = np.polynomial.hermite.hermgauss(100)
z, weights = math.sqrt(2)*nodes, weights/math.sqrt(math.pi)
delta2 = -math.expm1(-0.2)
q = 1+delta2
gain_new = q/(2+delta2)
rho_old, rho_new = 1-delta2/q, 1-gain_new*delta2/q
c1_num = np.sum(weights*z*np.tanh(math.sqrt(q)*z))
def coherent(rho):
    z2 = rho*z[:,None]+math.sqrt(1-rho*rho)*z[None,:]
    covariance = np.sum(weights[:,None]*weights[None,:]*
                        np.tanh(math.sqrt(q)*z[:,None])*np.tanh(math.sqrt(q)*z2))
    return float(covariance-rho*c1_num*c1_num)
out['fixed_q_gain_change'] = {'q':q, 'gain_old':1, 'gain_new':gain_new,
                              'rho_old':rho_old, 'rho_new':rho_new,
                              'g_old':coherent(rho_old), 'g_new':coherent(rho_new)}
assert coherent(rho_new) > coherent(rho_old)

# Positive check: closed-form GF and loss deficit for arbitrary SPD U.
B = rng.normal(size=(7,7)); U = B.T@B+np.eye(7)
V = rng.normal(size=(3,7))
lam, Q = np.linalg.eigh(U)
Astar = np.linalg.solve(U,V.T).T
T = 0.3
E = (Q*np.exp(-lam*T))@Q.T
At = Astar@(np.eye(7)-E)
Adot = Astar@((Q*(lam*np.exp(-lam*T)))@Q.T)
def loss(a):
    return 0.5*np.sum((a@U)*a)-np.sum(a*V)
pi = np.sum((Astar@Q)**2,axis=0)
direct = loss(At)-loss(Astar)
spectral = 0.5*np.sum(lam*pi*np.exp(-2*lam*T))
assert np.allclose(Adot,V-At@U)
assert np.isclose(direct,spectral)
out['valid_GF_identity'] = {'derivative_max_error':float(np.max(np.abs(Adot-(V-At@U)))),
                            'loss_deficit_error':float(abs(direct-spectral))}
out['source_sha256'] = {str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest()
                         for f in sorted(ROOT.glob('*.tex'))}
target = ROOT/'proof_audit_checks.json'
target.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k!='source_sha256'},indent=2))
