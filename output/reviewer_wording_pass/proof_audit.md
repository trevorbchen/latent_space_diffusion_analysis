# Mathematical audit of the revised manuscript

Audited 2026-09-25. Target: `main_iclr2027_9page.tex` and its included theory, checked against the repository's RFNN implementation. This follows the citation repair; that repair did not establish correctness of the remaining proofs. **The current manuscript still contains substantive mathematical errors.**

This is a source-level proof audit with deterministic algebraic and numerical checks, not formal verification, a fresh experimental replication, or a prediction of acceptance. No manuscript, abstract, bibliography, or PDF was changed during this audit. The nine-page layout therefore remains unchanged, but was not rebuilt or revalidated here. Source hashes are recorded in `proof_audit_checks.json`.

## Priority findings

### 1. Critical: the repository-step conversion omits division by dimension

Locations: `sec9-theory.tex:45`; `sec-appendix-theory.tex`, `rem:clock-p1` (line 124), and downstream code-clock predictions. Implementation: `experiment_v2_rfnn.py:80,98,236` (the copies in `code/` and `figures/code/` use the same normalization).

The implementation predicts `A phi`, where `phi = tanh(Wx)/sqrt(p)`, and minimizes

`L_logged = sum ||sqrt(Delta) A phi + noise||² / (d n)`.

For the theory objective `L = sum ||A phi - y||²/(2n)`, this is **`L_logged = (2 Delta/d) L`**. With the default learning rate `eta = 0.01 d/Delta`, the flow increment is **`Delta T = 0.02`**, not `0.02 d`. The logged feature covariance is unnormalized, so `lambda_theory = lambda_code/p`.

Consequently, with `p=64d`, the flow e-fold clock in steps is `3200 d/lambda_code`:

| Dimension | Rounded signal eigenvalue in manuscript | Current quoted conversion | Correct default conversion |
|---|---:|---:|---:|
| 5 | 20.30 | 157.64 | 788.18 |
| 200 | 73.88 | 43.31 | 8662.70 |

The exact mean-Euler e-fold counts are 787.68 and 8662.20. Thus the main-text claim that the signal clock falls in repository steps is reversed: it grows approximately elevenfold. These are spectral e-fold estimates, not observed stopping times. They assume default learning rate and zero momentum; run-specific overrides require a separate conversion.

The appendix itself already distinguishes coordinate-summed versus coordinate-averaged implementations in `rem:lin-gd` (line 707). A clock from the former cannot be applied to the latter. All absolute-step tables, tail-activation calculations, and cross-dimension comparisons using this conversion need reconciliation with run provenance. Within-run ratios of spectral clocks are unchanged. This finding does not change saved empirical observations.

**Repair:** derive one conversion from the actual saved-run loss/optimizer, correct the main sentence, and regenerate every dependent absolute-step calculation. Do not alter the reported eigenvalue ratios solely because of this error.

### 2. High: a one-sided alignment cutoff does not justify an absolute error bound

Location: `prop:gap-p1`, line 172; `def:taumem-p1`, line 191.

The block is defined by `gamma_i <= gamma0`, but its approximation error is bounded as if `|gamma_i| <= gamma0`. Negative alignment is not excluded by the stated definition. With `lambda=pi=1`, `gamma=-1`, `gamma0=1/2`, and `T=1`, the absolute omitted term is `1-exp(-1)=0.63212`, exceeding the asserted bound `0.5`.

This is a counterexample to the algebraic implication used in the proof, not a claim that a particular GMM run realizes those coefficients. A model-specific exclusion of negative alignment would need proof. The exact spectral gap identity survives.

The accompanying claim that absorbing every positively aligned sample mode lowers test loss is also too strong: for `ubar=lambda=pi=1`, `gamma=0.6`, its population-loss derivative becomes positive once `1-exp(-T)>0.6`.

**Repair:** retain the general bound `sum |gamma_i| lambda_i pi_i (1-exp(-lambda_i T))`, or explicitly restrict to an absolute-alignment block and treat the excluded modes separately. State the condition and time interval for population-loss improvement.

### 3. High: the diffusion floor prevents the claimed divergence

Location: `prop:lin-emp(c)`, line 739, especially `eq:lin-mp-edges` and the sentence following it.

The proposed slow edge is

`lambda_minus = exp(-2t) sigma_perp² (1-sqrt(c))² + Delta_t`.

At fixed positive diffusion time, `Delta_t>0`, so as `c -> 1` the clock tends to **`1/(4 Delta_t)`**, not infinity. At `t=0.01`, the limit is **12.6254** in this proposition's flow convention. For `c>1`, the shifted atom at `Delta_t` must also be included; the lower continuous-band edge is not the global minimum.

The same paragraph overuses interlacing: a principal-block lower edge is not a lower bound on the smallest full-matrix eigenvalue. For `[[1,0.9],[0.9,1]]`, the one-coordinate principal block has eigenvalue 1 while the full minimum is 0.1. Interlacing bounds particular shifted indices; it does not put all bottom eigenvalues in the principal block's bracket. The separate global diffusion floor remains available.

**Repair:** state the finite clock limit, specify the aspect-ratio regime/atom, and replace the full-matrix bracket with actual indexed interlacing inequalities. Any U-shape or monotonicity claim justified only by numerical evaluation must be labeled accordingly.

### 4. High: per-sample eigenvector localization is not established

Location: `prop:sample-bulk-persample-pred`, line 1012; downstream pointwise clocks and ordering claims.

The covariance decomposition `U_* = U_loc + U_cloud` is valid. The subsequent spectral conclusions need assumptions that are absent:

- Knowing the cloud trace and maximum possible rank does not bound each eigenvalue by `trace/p`. A PSD rank-one cloud with trace 1 has operator norm 1, not `1/p`.
- Small Gram off-diagonals relative to diagonal magnitudes do not ensure an eigenvector belongs to an individual point. `[[1,epsilon],[epsilon,1]]` has equally mixed eigenvectors for every positive epsilon, however small. Individual localization needs separation between diagonal values as well as a perturbation bound.
- Random signs alone do not justify the asserted square-root-size operator norm or a semicircle/MP law. Dependence between entries must be controlled.

These are gaps in the proposed proof, not demonstrations that the numerical approximation fails in every experiment. Matching traces or medians does not resolve them.

**Repair:** retain the exact covariance identity and empirical diagnostics; make the samplewise spectral attachment conditional on an explicit cloud operator bound, Gram perturbation bound, and individual eigenvalue gaps. Otherwise present the predictor as a heuristic rather than a derived pointwise clock.

### 5. Medium: the residual Hermite expansion has a wrong coefficient

Location: `lem:coeffs-buf(iii)`, line 459.

The stated `a_*² = (2/3)q³(1-8q+50q²+O(q³))` omits the leading fifth-Hermite contribution at order `q^5`. Exact Gaussian-moment expansion gives

`a_*² = (2/3)q³ - (16/3)q⁴ + (532/15)q⁵ + O(q⁶)`

`      = (2/3)q³[1-8q+(266/5)q²+O(q³)]`.

The coefficient is **53.2, not 50**. The leading cubic scaling remains correct. This does not by itself invalidate numbers obtained by direct quadrature. Separately, a numerical grid check of coefficient monotonicity is evidence on that grid, not a proof over a continuous interval (`lem:ge-coeff-pred`, line 973).

### 6. Medium: the predictor subsection mixes two fan-in conventions

Location: `prop:data-side-rates-pred`, line 999, and the following residual-feature definitions.

It sets `W_ij ~ N(0,sigma_w²)` and `gamma=d sigma_w²`, but writes the activation as `tanh(w^T x/sqrt(d))` while assigning variance `q_x=gamma ||x||²/d`. With the displayed extra divisor, the variance is actually `gamma ||x||²/d²`. For `d=20`, `sigma_w²=1/d`, and `||x||²=d`, the actual variance is 0.05 rather than 1.

**Repair:** consistently use either `W_ij ~ N(0,gamma/d)` with activation `tanh(Wx)`, as in the implementation, or `W_ij ~ N(0,gamma)` with activation `tanh(Wx/sqrt(d))`. Re-derive the spectral prefactors after choosing. A parameter redefinition alone also does not preserve optimizer learning rates.

### 7. Medium: keeping q fixed while changing gain does not preserve the predictor

Location: `cor:shape-from-qmu-pred(iii)`, line 1043, compared with `def:two-source-predictor-pred`, line 1037.

The predictor uses `Lambda_mu=g(q_mu,rho_mu)/n` and `rho_mu=1-gamma Delta/q_mu`. Changing gain to preserve q after dropping coordinates changes rho. The curve therefore does not generally stay unchanged.

Example: at `t=0.1`, dropping from 20 to 10 dimensions with noised clean squared norm 20 gives fixed `q=1.18127` after changing gain from 1 to 0.54155. Rho changes from 0.84655 to 0.91690; Gaussian quadrature gives coherent residual covariance **0.01880 -> 0.02474**. This counterexample concerns gain adjustment, not the main synthetic fixed-q intervention with unchanged gain and diffusion time.

**Repair:** require both q and rho to be preserved, or explicitly account for the changed correlation in the predicted clocks.

### 8. Medium: the collapsed-coordinate limit exits the assumed regime

Location: `prop:deadblock-real`, line 1221.

The argument invokes the Gaussian-equivalent regime `p>n>d`, then sends the number of dead coordinates to infinity with fixed sample count to conclude `R -> 0` from a factor `n/d`. That limit leaves `d<n`. If n scales with d to preserve the regime, `n/d` need not vanish.

Also distinguish a lower bound proportional to d from a claim about relative growth: because `a1(q(d))²` increases as q decreases, `d/a1(q(d))²` can grow by less than the ratio of dimensions. The explicit `1/d` spectral factor must not be described as dependence through q alone.

**Repair:** restrict to finite dimensions within the assumptions, or specify and prove a different joint limit. State linear lower bounds and asymptotic equivalences precisely.

### 9. Medium: the endpoint approximation is not a proved memorization ceiling

Locations: `lem:endpoint`, line 876; `cor:taumem-fraction`, line 918.

The ideal marginal mixture and single-component ODE calculation can be retained. Replacing a random Gaussian endpoint radius by its typical value gives an approximation, not a universal upper bound `F_max=1-G_NN(3 sqrt(d Delta_term))`. Gaussian radii can be smaller than that typical value, and a learned score need not have the ideal endpoint variance. A true ceiling needs a probability bound and a specified score/sampler class.

The empirical-CDF inversion also needs consistent strict inequalities at ties: an event written `D>u` cannot automatically be inverted with a non-strict quantile inequality. The claim that observed time ratios alone refute the distance-only width argument is incomplete when the factor `lambda T_1` remains unspecified.

**Repair:** label the endpoint calculation a sampler baseline/approximation, give an explicit concentration error if needed, and use order statistics with a stated tie convention.

### 10. Medium: the predictor's condensation filter drops a time factor

Location: `def:two-source-predictor-pred`, line 1037, versus `lem:condensation`, line 861.

The filter uses `d_NN² >= 2 Delta_tc log n`, while the earlier center-separation condition contains `2 exp(2tc) Delta_tc log n`. With distances in the clean latent coordinates, these are different thresholds; the factor is about 7.39 at `tc=1`.

**Repair:** retain the factor, redefine the distances as noised-center distances, or explicitly call the new filter a separate heuristic. The earlier condition is itself a pointwise probabilistic statement, not a guarantee of an entire reverse trajectory.

## Additional qualifications to resolve

- In the initial setup, an empirical second moment written as center second moment plus within-cluster covariance needs the center/noise cross terms, unless the latter symbol is explicitly defined to include them. Population cancellation does not imply exact finite-sample cancellation.
- Full rank does not imply every mode is active in gradient flow: a mode with `V v_i=0` has zero target amplitude.
- In `cor:dnn-scaling`, different minimizers cannot justify a downward discrepancy relative to the sum of the separate minima: `min_i(S_i+N_i) >= min_i S_i + min_i N_i`. The displayed nearest-neighbor estimate should remain heuristic.
- At nonzero `sigma_perp=0.01`, finite-sample corrections may be small but are not identically absent; “applies verbatim” overstates the zero-variance limit.
- Frozen-feature clocks do not establish memorization dynamics for a trainable MLP. Existing conjecture labels should be retained, including in downstream summaries.
- The frozen abstract still needs an author-level consistency check against the weaker, conditional theory in the body. This audit did not change the submitted abstract or assess conference rules governing amendments.

## What survives, and what remains conditional

| Claim family inspected | Assessment |
|---|---|
| Fixed-feature mean update and closed-form gradient flow (`prop:sgdmean-p1`, `lem:gf-p1`) | Core algebra valid for its stated normalization; repository conversion needs finding 1; full rank does not ensure activation. |
| Spectral training-loss/parameter deficits (`prop:rse-p1`) | Algebraic identities and elementary exponential bounds survive. |
| Expected operating point and leading small-q behavior (`lem:q-p1`) | Expectation/leading expansion survive; realized-data and monotonicity statements need their parameter assumptions. |
| Signal hitting-time bounds (`prop:taugen-p1`) | Valid conditional on active modes and the specified spectral bounds; code-step predictions need correction. |
| Exact gap decomposition (`prop:gap-p1`) | Exact identity survives; cutoff reduction fails as stated (finding 2). |
| Gaussian kernel/Hermite identities and single-copy rank counts | Exact components survive; higher-order coefficient needs finding 5. |
| Four-block edges and buffer ratio | Conditional spectral approximation, not a proved finite-GMM law. Ratio algebra survives conditional on the chosen edges; does not establish monotonicity by itself. |
| Exact linear score, population linear flow, empirical linear flow (a) | Algebra survives. |
| Empirical linear null-block formula (b) | Valid after the explicitly stated removal of cross blocks. |
| Empirical linear band/clock (c) | Finding 3 invalidates divergence and the blanket bracket. |
| Late-floor expansion and phase/control predictions | Leading expansion can be retained; global shape and extrapolations require their additional assumptions. |
| Condensation and sampler bridge | Pointwise condition and conditional bridge; do not remove assumptions about reverse paths and endpoints. |
| Per-sample predictor and collapsed-coordinate effects | Findings 4, 6–10 prevent treating these as established general theorems. |
| Conditional real-spectrum pushforward bounds | Repaired algebraic rank/Weyl/Ky Fan bounds can be retained; the Gaussian-equivalent premise remains assumed. |
| Scope/rebound/feature-learning predictions | Sketches or conjectures; no new experimental validation in this audit. |

This table summarizes the inspected argument families; it is not a certification of every constant, simulation claim, or appendix sentence.

## Reproducibility and repair order

Run:

```sh
python3 output/reviewer_wording_pass/proof_audit_checks.py
```

The script checks objective/gradient normalization, both clock conversions, the alignment counterexample, finite diffusion-floor limit, interlacing counterexample, exact rational Hermite coefficients, trace/localization counterexamples, the gain-change example, and a positive check of the GF/deficit identities. Results are in `proof_audit_checks.json`. Generic matrix/scalar counterexamples establish failure of the stated deductions; they are not claimed to be samples from the manuscript's data distribution.

Repair priority: (1) reconcile clocks and regenerate dependent quantitative statements; (2) correct false algebraic claims and normalization; (3) explicitly condition or downgrade spectral/localization/sampler arguments; (4) recheck all main-text dependencies and rebuild to verify the nine-page limit. Changing a theorem to an assumption must be propagated to conclusions, not just its heading. No fresh experiments were run, and the substantive findings remain unfixed in the manuscript delivered before this audit.
