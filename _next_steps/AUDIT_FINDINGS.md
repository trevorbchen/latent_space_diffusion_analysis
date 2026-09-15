# Theory audit — all findings (exported from the audit run, 2026-08-31)

157 findings from 11 independent lenses, each adversarially refereed. Verdict key: CONFIRMED = referee independently verified; 
PLAUSIBLE = likely but unverified; REFUTED = referee overturned it (kept here so you don't re-derive it); — = referee agent died, unrefereed.

Published summary: https://claude.ai/code/artifact/a8e70925-877b-45ff-ae73-fce67e37de8d


Counts: {'blocking': 58, 'major': 78, 'minor': 21}; verdicts: {'CONFIRMED': 109, 'REFUTED': 11, '—': 35, 'PLAUSIBLE': 2}


---

## [BLOCKING] The clock is built entirely from data-side modes; the sample bulk that IS memorization never enters

`clock-omits-sample-bulk` · verdict **CONFIRMED** · kind missing-theory · effort days  

**Where:** ICLR_2026/sec-spectral-predictor.tex:17 (definition of P_d); ICLR_2026/sec-appendix-integrated.tex:769-787; sec-appendix-fourbulk.tex Lemma 1 (lem:Ulin) and Lemma 2 (lem:Udiag); compute_celeba_spectral_predictor_data.py:eta_star_mc + `eta_star` column

**Claim:** P_d(s) uses only the eigenvalues of M_t(d), which by the paper's own Lemma 1 map onto the d_lat data-side modes of U (spec(U^lin) = mu_1^2 spec(M_t)/psi_p). The sample bulk, whose edge is etabar_star/psi_p (Lemma 2), is a function of E[(||x_t||^2/d)^k] and is NOT in spec(M_t). Memorization is by the paper's own argument the absorption of the sample bulk. The predictor therefore contains zero terms from the mechanism it claims to instantiate, and its agreement with data cannot be evidence for that mechanism.

**Evidence:** Lemma 1 covers only rank(U^lin)=d_lat: 'rank(U^lin) = d_lat almost surely, and its nonzero spectrum is the spectrum of M_t rescaled by mu_1^2/psi_p'. Lemma 2's Remark is explicit that this is insufficient: 'U^lin has rank exactly d_lat, so any spectral content beyond the two data-side blocks must come from a different contribution.' The sample-bulk edge etabar_star IS computed by the pipeline (compute_celeba_spectral_predictor_data.py, eta_star_mc, stored as `eta_star` in spectral_features.csv, plotted in fig:app-spectral-features) but never appears in `pressure_u`. Numerically the omitted term is the better predictor on CelebA: regressing log tau_obs on -log(etabar_star) with a per-threshold intercept and one shared slope gives fitted slope 0.997 (the theory predicts tau_mem = psi_p/etabar_star, i.e. slope exactly 1) and within-threshold R^2 = 0.677, versus R^2 = 0.447 for the 7-parameter clock on the same 73 points. (CIFAR-10 is the other way: 0.288 vs 0.391.)

**Fix:** Add an explicit sample-bulk term to the clock: a group of n modes at lambda_sample = etabar_star(d)/psi_p (etabar_star is already computed), and define memorization pressure as absorption of THAT group only, with the M_t modes entering only as the competing data-side modes. Then the predictor becomes a two-timescale object whose ratio is mu_1^2 beta_t^2/etabar_star -- the very ratio the four-bulk theorem controls -- and its d-dependence is a testable consequence rather than a fit.


**Referee:** Verified. sec-spectral-predictor.tex:17 and sec-appendix-integrated.tex:770-790 define P_d entirely from eig(M_t(d)) = eig(e^{-2t}Z^T Z/n + Delta_t I), and pressure_u() in make_celeba_spectral_mem_predictor_figures.py takes only `eig` (plus optional excess weights) — eta_star appears nowhere in the fit path, only in spectral_features.csv and the features figure. sec-appendix-fourbulk.tex lem:Ulin does state rank(U^lin)=d_lat with spectrum = spec(M_t)*mu_1^2/psi_p, and its Remark says explicitly that anything beyond the two data-side blocks must come from U^diag. So the mechanism the paper calls memorization (sample-bulk absorption, edge etabar_star/psi_p) has no term in the clock. I reproduced the regression: per-threshold intercept + one shared slope on -log(eta_star) gives slope 0.997, within-threshold R^2 = 0.677 on CelebA (73 points) vs 0.447 for the fitted clock; CIFAR 0.288 vs 0.391. This is a real and material gap.


**Referee correction:** Two overstatements. (a) The paper is not silent about it: the caption of fig:app-spectral-by-n states outright that a real n-dependence 'would require recomputing M_{t,n}(d) ... or adding an explicit sample-count/sample-bulk term to the predictor'. The defect is that this concession sits in a figure caption while the abstract and sec-intro.tex:29 still sell the clock as an instantiation of the mechanism. (b) 'contains zero terms from the mechanism' is too strong. eta_star_mc() computes E[(||x_t||^2/d)^3] (note: without the mu_k^2 factors, so the column is a constant multiple of etabar_star, not etabar_star itself), and E[||x_t||^2/d] = tr(M_t)/d = lambda_mean exactly — I checked at d=200: 0.8187*1.3020 + 0.1813 = 1.2473 = lambda_mean. So etabar_star is the third moment of the same per-sample norm whose first moment is the mean of the very eigenvalues P_d averages over; it is a functional of the same object, just not an eigenvalue of it. The correct statement is that the clock's functional form (unweighted mean of per-mode absorption over the d data-side modes) cannot express absorption of a separately gapped n-fold sample edge, and the paper never separates the two timescales.


---

## [BLOCKING] P_d's dynamic range is bounded by the spectral condition number, uniformly in d, contradicting the linear-in-(d_lat - d_int) buffer bound

`clock-cannot-express-buffer-bound` · verdict **CONFIRMED** · kind inconsistency · effort days  

**Where:** ICLR_2026/sec-spectral-predictor.tex:17 vs sec-rfnn-bounds.tex Eq. (eq:buffer-bound)

**Claim:** Because every eigenvalue satisfies lambda_i(M_t) in [lambda_min, lambda_max] and, by construction, lambda_min >= Delta_t = 1 - e^{-2t}, the inverse clock obeys L/lambda_max <= kappa*tau_q(d) <= L/lambda_min with L = -log(1 - sqrt(theta_q)). Hence for any two latent widths, tau_q(d)/tau_q(d') <= lambda_max/Delta_t, a constant independent of d. For CelebA at t=0.1 this ceiling is 6.644/0.18127 = 36.6x. Eq. (buffer-bound) instead asserts tau_mem - tau_gen >= (d_lat - d_int) psi_p/(mu_1^2 beta_t^2), i.e. unbounded linear growth in d_lat. The predictor and the buffer corollary are therefore two different, mutually incompatible mechanisms: the clock delays by mass dilution of an unweighted average (adding near-floor modes lowers the mean), the corollary delays by serial traversal of a bulk. Gradient flow decays all modes in parallel, so the clock's mechanism is the physically consistent one and Eq. (buffer-bound)'s 'width times slowest timescale' product is the one without a derivation.

**Evidence:** M_t(d) = e^{-2t} Z^T Z/n + (1-e^{-2t}) I implies lambda_min >= Delta_t exactly; spectral_features.csv confirms lambda_min = 0.18128 = Delta_t at d = 160, 180, 200 (collapsed VAE coordinates), with lambda_max = 6.644 at d=200. Eq. (buffer-bound) in sec-rfnn-bounds.tex: 'tau_mem - tau_gen >= (d_lat - d_int) * 1/lambda_min^{noise-dim} = (d_lat - d_int) * psi_p/(mu_1^2 beta_t^2)'. Fitted predictions bear the ceiling out: at q=0.75 the predicted tau ranges only 1.41e6 (d=10) to 1.32e7 (d=200), a 9.3x span against the 36.6x hard ceiling.

**Fix:** Either (a) drop Eq. (buffer-bound) and re-derive the delay as parallel-mode dilution (tau_mem is set by 1/lambda_sample, and the d-dependence must then come from etabar_star(d), not from mode counting), or (b) keep the serial-traversal claim and justify it -- but then the clock must be replaced by something that grows without bound in d_lat. As written the paper asserts both.


**Referee:** The math checks out. Every term of P_d is monotone in lambda_i, so P_d(s) is bracketed by (1-e^{-kappa lambda_min s})^2 and (1-e^{-kappa lambda_max s})^2, giving L/lambda_max <= kappa*tau_q(d) <= L/lambda_min with L = -log(1-sqrt(theta_q)); hence tau_q(d)/tau_q(d') <= max_d lambda_max / min_d lambda_min. M_t = e^{-2t}Z^T Z/n + Delta_t I forces lambda_min >= Delta_t for every d, so the ceiling is uniform in d. Confirmed numerically: celeba_spectral_features_primary_t.csv gives lambda_min = 0.181454/0.181329/0.181295/0.181283 at d=140/160/180/200 against Delta_{0.1} = 0.1812692, and lambda_max = 6.6439, ceiling 36.7x. Fitted predictions bear it out: at q=0.75, tau_pred spans 1.41e6 (d=10) to 1.32e7 (d=200), 9.3x. Against this, Eq. (buffer-bound) asserts unbounded growth linear in (d_lat - d_int). The count multiplication is also genuinely underived: under eq:rfnn-mode-decay all d_lat - d_int noise-dim modes share the same leading-order eigenvalue mu_1^2 beta_t^2/psi_p and therefore decay in parallel on one timescale psi_p/(mu_1^2 beta_t^2), independent of how many there are. _next_steps/theory_plan.md confirms this step was never done — it lists it as the plan's to-do, 'via a sum-of-exponentials lower bound', which does not appear in sec-rfnn-bounds.tex.


**Referee correction:** 'The paper asserts both' needs adjusting for the current draft. main.tex \inputs sec-appendix-integrated.tex, not sec-appendix.tex, and sec-appendix-integrated.tex contains zero theorem/lemma environments; Eq. (buffer-bound) currently lives only in the orphaned sec-rfnn-bounds.tex and in the uncompiled ICLR_2026/sec-appendix.tex:580, while sec-theory.tex is still a \todo stub. The inconsistency nevertheless survives verbatim in the compiled text: abstract.tex ('each excess latent direction adds a mode that must be absorbed before sample-specific memorization begins') and sec-intro.tex:26 ('increasing d_lat adds d_lat - d_int extra eigenmodes that the model must absorb before reaching the sample bulk') both assert the serial mode-counting mechanism, alongside the parallel-decay clock in sec-spectral-predictor.tex:17. Also, the referee's conclusion that the clock's mechanism is therefore 'the physically consistent one' is only half right: the clock's d-dependence comes from dilution by near-floor modes, which under the paper's own eq:rfnn-mode-decay likewise has no bearing on when the sample bulk is absorbed. Neither object as written derives a memorization delay; the honest fix is (a).


---

## [BLOCKING] The theory predicts one sample-bulk edge, hence simultaneous memorization of all n points; it cannot generate a memorization FRACTION curve at all

`no-per-sample-resolution-in-theory` · verdict **CONFIRMED** · kind missing-theory · effort weeks  

**Where:** sec-appendix-fourbulk.tex Lemma 2 (lem:Udiag) and Theorem thm:fourbulk (Sample row, edge etabar_star/psi_p); consumed by ICLR_2026/sec-spectral-predictor.tex:17

**Claim:** Lemma 2 gives all n sample modes a single concentrated eigenvalue etabar_star/psi_p ('its nonzero eigenvalues are concentrated around etabar_star/psi_p'). Under Eq. (rfnn-mode-decay) they are then absorbed on one common timescale, so the theory predicts a step function: all n training points become memorized at essentially the same time. The measured quantity is a memorization fraction that climbs from 1% to 75% over roughly a decade of training steps. The predictor manufactures that spread from the spread of the DATA-side eigenvalues of M_t, which by Lemma 1 have nothing to do with per-sample differences. So the clock's continuous rise and the theory's step disagree, and the object that maps them (the sigmoid / theta map) is doing all the work.

**Evidence:** Theorem thm:fourbulk lists the Sample bulk with a single 'Leading-order edge etabar_star/psi_p' and count n; Lemma 2's proof concentrates the eigenvalues 'around their mean as a self-averaging quadratic form'. Empirically the fitted levels span the whole clock range for one dataset -- CelebA theta = 0.256 (q=0.01) to 0.975 (q=0.75) -- i.e. the 1%-to-75% spread consumes almost the entire data-side pressure range. Observed CelebA hitting times at d=70 run 200k (1%) to 2.46M (75%), a 12x spread with no counterpart in the theory.

**Fix:** Either derive the WITHIN-sample-bulk eigenvalue spread (etabar_star(x_t^mu) varies with ||x_t^mu||^2, so the sample-bulk eigenvalues are Lambda_mu = eta_star(x_t^mu, x_t^mu)/psi_p and their empirical distribution over mu is computable from the encoded latents) and predict the memorization fraction as the fraction of samples whose own mode has been absorbed -- this is a real, checkable prediction and it would give per-sample resolution the current clock lacks -- or state plainly that the theory has no fraction-level content and the sigmoid is a fitted link function.


**Referee:** Verified. sec-appendix-fourbulk.tex lem:Udiag states the nonzero eigenvalues of U^diag are 'concentrated around etabar_star/psi_p', and thm:fourbulk lists the Sample bulk as count n with a single leading-order edge etabar_star/psi_p. Under eq:rfnn-mode-decay a bulk of n identical eigenvalues is absorbed on one timescale, so the theory as stated predicts an essentially simultaneous memorization event, with no within-bulk ordering. The measured object is a fraction climbing over a decade: at CelebA d=70 the observed hitting times run 200k (q=0.01) to 2.46M (q=0.75), a 12.3x spread, and the fitted CelebA theta levels span 0.256 to 0.975 — i.e. the 1%-to-75% range consumes nearly the entire clock. Nothing in the theory supplies that spread; the six-level theta map supplies it.


**Referee correction:** Two strengthenings the finding understates. (1) The concentration claim in lem:Udiag is itself doubtful at the operating scales, which makes the proposed fix more than a suggestion: eta_star(x_mu, x_mu) = sum_k mu_k^2 (||x_mu||^2/d)^k varies directly with the per-sample norm, and z_normsq_over_d_std/z_normsq_over_d_mean in celeba_spectral_features_primary_t.csv runs from 0.22 (d=200) to 0.48 (d=10) — cubed, that is a 60-100% relative spread in the per-sample sample-bulk eigenvalue. So Lambda_mu genuinely has an O(1) distribution over mu, which is exactly the missing fraction-level content. (2) lem:Udiag's proof does not in fact deliver the edge the lemma states: it computes the eigenvalues as eta_star(x,x)^2/(p n), whereas the lemma and theorem quote etabar_star/psi_p = etabar_star d/p. Those are not the same object, so even the single-edge claim is not established by the proof as written. Severity: major rather than blocking, since the four-bulk theorem never claims fraction-level content — the overclaim is in abstract.tex, which sells a clock that 'estimates memorization percentage as a function of training step'.


---

## [BLOCKING] Lemma 1's rescaling constant is wrong by a factor p/d_lat^2; the correct prefactor is mu_1^2/d_lat, not mu_1^2/psi_p

`ulin-prefactor-wrong` · verdict **CONFIRMED** · kind error · effort hours  

**Where:** sec-appendix-fourbulk.tex, Lemma \ref{lem:Ulin} (lines 112-131) and the Theorem \ref{thm:fourbulk} edge table (lines 214-227); propagated to sec-rfnn-bounds.tex Thm \ref{thm:fourbulk-body} and Eq (buffer-bound)

**Claim:** U^lin = (mu_1^2/(p d_lat)) W M_t W^T with W in R^{p x d_lat}, entries N(0,1). Its nonzero eigenvalues are those of (mu_1^2/(p d_lat)) M_t^{1/2} W^T W M_t^{1/2}. Since E[W^T W] = p I_{d_lat}, this is (mu_1^2/(p d_lat)) * p * spec(M_t) = mu_1^2 spec(M_t)/d_lat. The lemma instead asserts 'the spectrum of M_t rescaled by mu_1^2/psi_p' = mu_1^2 d_lat spec(M_t)/p. The two differ by psi_p/d_lat = p/d_lat^2. The proof performs the step eig(W M_t W^T/d_lat) = psi_p * eig(M_t) correctly and then simply never divides by p.

**Evidence:** Proof text: 'E[W^T W/d_lat] = psi_p I_{dlat} ... The nonzero eigenvalues of W M_t W^T equal those of M_t^{1/2} W^T W M_t^{1/2}, which are the eigenvalues of M_t rescaled by psi_p'. Multiplying that by the stated prefactor mu_1^2/(p d_lat) from Eq (U-lin) gives mu_1^2 psi_p lambda / p = mu_1^2 lambda / d_lat, not mu_1^2 lambda / psi_p. Numerically decisive: the stored eigenvalues are of U_code = (1/n) sum tanh(Wx)tanh(Wx)^T = p * U_paper (code/experiment_v2_rfnn.py, compute_U). In code units the lemma predicts mu_1^2 alpha_t^2 d_lat while the corrected algebra predicts mu_1_eff^2 psi_p alpha_t^2. Observed signal-bulk means (sigma_perp=0.5, psi_p=64, n=500, t=0.01): d_lat=5 -> 32.1, 10 -> 49.6, 20 -> 67.5, 40 -> 83.9, 100 -> 102.4, 200 -> 112.8. Lemma's formula predicts 5.1, 10.1, 20.3, 40.6, 101.4, 202.8 (linear in d_lat, off by 6.3x at d_lat=5 and 1.8x at d_lat=200). Corrected formula with mu_1_eff^2(tau_t) predicts 31.8, 49.7, 69.5, 87.3, 103.6, 110.6 — agreement within 3%. Same for the noise-dim bulk (observed 5.18, 7.10, 8.63, 9.95, 10.63; corrected 4.76, 6.66, 8.37, 9.93, 10.60; lemma 0.97, 1.94, 3.89, 9.72, 19.43). Per-mode check at d_lat=200: observed signal eigenvalues [167.8, 135.4, 99.8, 87.1, 73.9] vs corrected mu_1_eff^2 psi_p lambda_i(M_t^emp) = [172.1, 142.6, 90.8, 83.0, 77.9].

**Fix:** Replace 'mu_1^2/psi_p' by 'mu_1^2/d_lat' in Lemma 1 and in both edge rows of the Theorem table (equivalently mu_1^2 psi_p/p). Note that the same slip is present in Lemma 2: the sample-bulk eigenvalue of U^diag = (1/n) sum eta_star v v^T is eta_star/n, not eta_star/psi_p (the Lemma-2 proof's own line 'Lambda_mu * ||phi^perp||^2 -> eta_star^2/(p n)' is separately dimensionally wrong — it squares eta_star). Sanity-check any rewrite against tr(U) = E[tanh^2(||x_t|| z/sqrt(d_lat))], which the corrected edges reproduce and the current ones do not.


**Referee:** Independently re-derived and numerically decisive. Eq (U-lin) at /Users/ryan/Desktop/latent_space_diffusion_analysis/sec-appendix-fourbulk.tex:75 is U^lin = (mu_1^2/(p d_lat)) W M_t W^T with W entries N(0,1) (confirmed: ICLR_2026/sec-rfnn.tex:13 says W ~ N(0,1), and code/experiment_v2_rfnn.py folds the 1/sqrt(d) into the buffer). Nonzero spec = (mu_1^2/(p d)) spec(M_t^{1/2} W^T W M_t^{1/2}) ~ (mu_1^2/(p d)) * p * spec(M_t) = mu_1^2 spec(M_t)/d. Lemma 1 (line 115-116) instead states mu_1^2/psi_p = mu_1^2 d/p; ratio p/d^2 exactly as claimed. The proof (line 129) does the psi_p step correctly and never divides by p. Verified numerically: code/experiment_v2_rfnn.py compute_U returns U_code = (1/n) sum tanh tanh^T with no 1/p, so U_code = p U_paper. Signal-bulk means from sigma_noise_0.5/exp2_rfnn/raw_data/*/eigenvalues_pre.npy: d_lat=5,10,20,40,100,200 -> 32.08, 49.56, 67.46, 83.91, 102.41, 112.79. Lemma predicts d_lat*mu_1^2*alpha_t^2 = 5.07, 10.15, 20.29, 40.58, 101.45, 202.90 (6.3x low at d=5, 1.8x high at d=200, wrong d-dependence). Corrected formula with mu_1_eff^2 gives 31.80, 49.66, 69.44, 87.47, 105.08, 114.33 (<4% per-mode error, including the noise-dim band: at d=200 observed nd [max,med,min] = [27.17, 9.22, 2.12] vs predicted [26.27, 9.23, 2.17]). The trace check is decisive on its own: tr(U_paper) = sum(ev)/p must equal E[tanh^2(tau_t z)]; observed 0.5453/0.5751 (d=5) ... 0.2092/0.2100 (d=200), and the corrected linear trace + eta_star reproduces this to <1%, while the Lemma's linear trace is 0.0789 at d=5 (7x too small) and 0.3771 at d=200 — larger than the entire trace 0.2092, which is impossible for a PSD summand. The Lemma-2 slip is also real: with phi^perp carrying the 1/sqrt(p) normalization (as Eq U-diag's own line 86 E[||phi^perp||^2] = eta_star requires), U^diag = (1/n) sum phi^perp phi^perp^T has eigenvalues eta_star/n; Eq (U-diag) line 78 carries a spurious extra 1/p and eta_star, the proof line 154 then produces eta_star^2/(p n), and the Lemma statement line 139 asserts a third, different value eta_star/psi_p. Three mutually inconsistent expressions.


**Referee correction:** All correct as stated, with one caveat on the numerical support for the Lemma-2 half: the sample bulk in this data is broad rather than concentrated (at d_lat=200, sigma_perp=0.5 the block ev[d:d+n] runs 0.0029 to 0.1733 with median 0.0708 in code units), so eta_star/n -> p*eta_star/n = 0.1205 matches the sample-bulk scale within ~1.7x while the theorem's eta_star/psi_p -> d*eta_star = 0.941 is ~13x off. The eta_star/n correction is right, but 'concentrated around' should be dropped from Lemma 2 regardless.


---

## [BLOCKING] With the corrected prefactor tau_gen grows linearly in d_lat; the paper's central 'free delay' claim rests on the algebra error and is contradicted by its own RFNN data

`taugen-not-dlat-independent` · verdict **CONFIRMED** · kind error · effort days  

**Where:** sec-rfnn-bounds.tex lines 64-68 ('is d_lat-independent at fixed sigsig, s, dint'); ICLR_2026/sec-rfnn.tex line 65 ('defining taugen, dlat-independent ... while taugen stays fixed'); abstract/intro claim chain item (3)-(4)

**Claim:** tau_gen = 1/lambda_min^signal. Under the lemma's (wrong) prefactor this is psi_p/(mu_1^2 alpha_t^2), which is d_lat-free because psi_p = 64 is held fixed — that is exactly why the paper can claim the buffer is free. Under the correct prefactor tau_gen = d_lat/(mu_1_eff^2 lambda_min(signal block of M_t)), which grows essentially linearly in d_lat. The empirical RFNN spectra confirm the growth.

**Evidence:** From the stored spectra, converted to the paper's normalisation (lambda_paper = lambda_code/p), 1/lambda_min^signal at sigma_perp=0.5: d_lat=10 -> 20.1, 20 -> 30.5, 40 -> 47.7, 60 -> 63.3, 80 -> 81.4, 100 -> 94.8, 150 -> 137.0, 200 -> 173.3. That is an 8.6x increase over a 20x increase in d_lat; the corrected theory predicts d_lat/(mu_1_eff^2 alpha_t^2) growing 9.0x over the same range (12.9 -> 115.8, a constant 1.5x offset because lambda_min(signal) < alpha_t^2, see finding hatC-k-dependence). At sigma_perp=0.01: 17.1 (d=8) -> 39.5 (d=40). The training logs agree: argmin score_error moves from step 5e3 (d_lat=10) to step 9e4 (d_lat=200) in sigma_noise_0.5/exp2_rfnn/raw_data/*/metrics.jsonl. tau_gen is nowhere near constant.

**Fix:** Retract or heavily qualify the 'taugen stays fixed' claim. The defensible statement is a ratio statement, not an absolute one: both tau_gen and the noise-dim traversal time carry the same 1/d_lat eigenvalue suppression, so what is actually d_lat-independent to leading order is tau_nd/tau_gen = alpha_t^2/beta_t^2. If the paper wants an absolute delay claim it must be made about tau_mem (which the data show grows 688 -> 73857 over d_lat = 10 -> 200) and the mechanism for that growth must be re-derived, because under the corrected prefactor the sample-bulk edge scales as eta_star/n, which carries no explicit d_lat at all — its d_lat dependence enters only through eta_star(tau_t), i.e. through the same norm effect flagged in finding mu1-norm-dependence.


**Referee:** Verified directly, and it is the load-bearing half of the 'free delay' framing.

The claim appears verbatim at ICLR_2026/sec-rfnn.tex:65 ('signal bulk first (defining taugen, dlat-independent at fixed sigsig, k, cluster scale) ... while taugen stays fixed') and in the buffer corollary in sec-rfnn-bounds.tex ('is dlat-independent at fixed sigsig, s, dint'). It follows only from the erroneous 1/psi_p prefactor, which is d_lat-free because psi_p = 64 is held fixed by construction.

Measured signal-bulk median rises 3.4x (29.66 -> 99.80 over d_lat = 5..200). In gradient-flow time, tau_gen = 1/lambda_signal_theory = p/lambda_signal_code, i.e. 50p/lambda in the run's own optimizer steps: 539, 788, 1131, 1757, 2469, 2945, 3651, 4965, 6413 — an 11.9x rise over a 40x rise in d_lat (log-log slope ~0.68, steepening toward 1 as c1(q)^2 saturates). This is exactly what the corrected edge psi_p*c1(q)^2*alpha_t^2 predicts, and flatly contradicts 'd_lat-independent'.


**Referee correction:** Add the comparison the finding calls for but does not make, since it changes the verdict's weight: over the same sweep tau_mem (measured at the sample-bulk ONSET, 50p/lambda_sample_max) rises 221x, from 1.67e4 to 3.69e6. So tau_gen is not fixed, but it grows far more slowly than tau_mem and the separation still opens 18.6x. The correct rewrite is not 'the free-delay framing collapses' but 'the delay is not free: tau_gen ~ d_lat/(c1(q_t)^2 alpha_t^2) grows too, roughly one order more slowly than tau_mem, and the paper should state and plot both growth rates instead of asserting one is constant.'


---

## [BLOCKING] Eq (buffer-bound) multiplies a mode count by a per-mode timescale; under decoupled exponential dynamics this is a category error, and the resulting bound is violated by the paper's own data by up to 68x

`buffer-bound-count-times-timescale` · verdict **CONFIRMED** · kind error · effort days  

**Where:** sec-rfnn-bounds.tex, Eq \eqref{eq:buffer-bound} lines 56-63, and the sentence 'the time to traverse the noise-dim bulk is bounded below by its width times the slowest noise-dim timescale'

**Claim:** Eq (rfnn-mode-decay), a_i(T) - a_i* = (a_i(0)-a_i*) e^{-lambda_i T}, is a set of *decoupled* scalar ODEs. Mode i is absorbed at time 1/lambda_i regardless of how many other modes exist. There is no sequential 'traversal': the last noise-dim mode is absorbed at 1/lambda_min^{noise-dim}, full stop. Multiplying by the bulk width (d_lat - d_int) has no dynamical justification and inflates the claimed gap by a factor ~d_lat.

**Evidence:** Sec-rfnn.tex itself states the dynamics are diagonal ('Diagonalizing Eq. (rfnn-gradflow) in the eigenbasis of U gives an exponential per-mode evolution ... eigenmode i of U is absorbed by the readout with characteristic timescale tau_i = 1/lambda_i'). Empirical falsification of the bound as written, using tau_mem := 1/lambda_max^{sample} and tau_gen := 1/lambda_min^{signal} from the stored spectra (paper normalisation): sigma_perp=0.01, d_lat=40 — observed gap 4533, bound as written (with the lemma's prefactor) 306821, bound with the corrected prefactor 116876; violation by 68x / 26x. sigma_perp=0.5, d_lat=40 — observed 6796 vs 23053 / 10709. The bound fails at every sigma_perp=0.01 configuration and at sigma_perp=0.5 for all d_lat <= 100. Dropping the (d_lat - d_int) factor repairs it: 1/lambda_min^{nd} - 1/lambda_min^{sig} = 3977 <= 4533 (sigma=0.01, d=40) and 5855 <= 73684 (sigma=0.5, d=200) hold in every configuration.

**Fix:** Delete the (d_lat - d_int) multiplier. State instead tau_mem - tau_gen >= 1/lambda_min^{noise-dim} - 1/lambda_min^{signal}, and be explicit that the mode *count* enters nothing: the bulk width matters only insofar as adding latent coordinates changes where the sample-bulk edge sits. Then either (a) prove that the sample-bulk edge is pushed down as d_lat grows (the data support this: lambda_max^{sample} falls 1.45e-3 -> 1.35e-5 over d_lat = 10 -> 200 at sigma_perp=0.5), or (b) restate the whole mechanism in the ratio form tau_mem/tau_gen. As written the corollary is unsalvageable.


**Referee:** The category error is real and self-inflicted: ICLR_2026/sec-rfnn.tex:29-35 derives Eq (rfnn-mode-decay) as a set of decoupled scalar exponentials, a_i(T)-a_i* = (a_i(0)-a_i*) e^{-lambda_i T}, and states 'eigenmode i of U is absorbed by the readout with characteristic timescale tau_i = 1/lambda_i'. Under decoupled dynamics there is no sequential traversal and the count of modes in a bulk enters nothing; sec-rfnn-bounds.tex:54-63's 'the time to traverse the noise-dim bulk is bounded below by its width times the slowest noise-dim timescale' multiplies a dimensionless count by a time with no dynamical justification. Empirically falsified in every configuration I checked, using tau_mem = p/ev[d_lat] and tau_gen = p/ev[d_int-1]: sigma_perp=0.5 observed gap vs bound-as-written (d_lat -> gap / bound / ratio): 10 -> 668/3292/0.20, 20 -> 1888/9877/0.19, 40 -> 6796/23045/0.29, 60 -> 13322/36214/0.37, 80 -> 22653/49383/0.46, 100 -> 32154/62551/0.51, 150 -> 56789/95473/0.59, 200 -> 73684/128395/0.57. sigma_perp=0.01: 8 -> 654/26290/0.025, 10 -> 953/43817/0.022, 15 -> 1676/87635/0.019, 20 -> 2195/131452/0.017, 30 -> 3096/219086/0.014, 40 -> 4533/306721/0.015 (a 68x violation, matching the reviewer's 306821 to rounding). Dropping the (d_lat - d_int) multiplier repairs it: 1/lambda_min^nd - 1/lambda_min^sig holds in all 14 configurations (e.g. 3977 <= 4533 at sigma=0.01/d=40; 5854 <= 73684 at sigma=0.5/d=200).


**Referee correction:** The reviewer understates the scope of the empirical violation. The bound as written fails at EVERY configuration in both sweeps, not only 'at sigma_perp=0.5 for all d_lat <= 100' — the ratios at d_lat=150 and 200 are 0.595 and 0.574, i.e. still violated by ~1.7x. Everything else stands, including the proposed replacement.


---

## [BLOCKING] The central buffer claim does not follow from the mode-decay equation, and the four-bulk theorem's own edge formulas make tau_mem d_lat-independent

`buffer-bound-is-a-non-sequitur` · verdict **CONFIRMED** · kind error · effort weeks  

**Where:** /Users/ryan/Desktop/latent_space_diffusion_analysis/sec-rfnn-bounds.tex Eq. (buffer-bound), lines 55-70; asserted as fact in /Users/ryan/Desktop/latent_space_diffusion_analysis/ICLR_2026/abstract.tex and ICLR_2026/sec-rfnn.tex:63-65

**Claim:** Eq. (buffer-bound), tau_mem - tau_gen >= (d_lat - d_int) * psi_p/(mu_1^2 beta_t^2), multiplies a per-mode timescale by a mode COUNT. Eq. (rfnn-mode-decay) is a decoupled linear ODE: a_i(T) - a_i* = (a_i(0)-a_i*) e^{-lambda_i T}. Every mode relaxes in parallel; mode i's absorption time 1/lambda_i is completely unaffected by how many other modes exist. Adding d_lat - d_int noise-dim modes therefore cannot delay the sample modes by even one unit of time. Worse, plugging the theorem's own edges in: tau_sample = psi_p/etabar_star and tau_signal = psi_p/(mu_1^2 alpha_t^2), and with p = 64 d_lat (psi_p = 64 fixed), alpha_t^2 = e^{-2t}(s^2/d_int + sig_sig^2)+Delta_t and ||x||^2/d_lat -> e^{-2t}sig_perp^2 + Delta_t = beta_t^2 are both d_lat-independent, so etabar_star is d_lat-independent. Theorem thm:fourbulk therefore predicts tau_gen, tau_mem AND their difference are all d_lat-independent -- the exact negation of the paper's thesis.

**Evidence:** sec-rfnn-bounds.tex: "modes are absorbed in order of decreasing $\lambda_i$, so the time to traverse the noise-dim bulk is bounded below by its width times the slowest noise-dim timescale" -- no derivation is given for "width times". abstract.tex: "Because gradient flow learns modes in decreasing-eigenvalue order, this bulk acts as a buffer: each excess latent direction adds a mode that must be absorbed before sample-specific memorization begins." Nothing in Eq. (rfnn-mode-decay) makes absorption serial. Independently corroborated by the project's own figure caption in final_report.tex:746-752: "Signal and noise-dim peaks shift right with $\dlat$; sample mode stays fixed" -- a fixed sample edge is exactly a d_lat-independent tau_mem.

**Fix:** Either (a) define tau_mem as the time at which a fixed FRACTION of total readout mass has been absorbed (a sum over modes), in which case adding modes genuinely shifts the crossing and the delay can be derived, and redo the corollary with that definition; or (b) drop the count factor and state honestly that within the frozen-feature model the delay comes from the sample-bulk edge moving, then show that it does move with d_lat. As written, the paper's headline mechanism is asserted, not derived, and its own theorem contradicts it.


**Referee:** Verified in full. Eq. (rfnn-mode-decay) in ICLR_2026/sec-rfnn.tex:29-35 is a diagonalized linear ODE, a_i(T)-a_i* = (a_i(0)-a_i*)e^{-lambda_i T}; modes relax in parallel and 1/lambda_i is independent of how many other modes exist. sec-rfnn-bounds.tex gives no derivation for 'width times the slowest noise-dim timescale' -- the words 'modes are absorbed in order of decreasing lambda_i, so the time to traverse the noise-dim bulk is bounded below by its width times ...' are the whole argument. The internal contradiction is also real: Theorem thm:fourbulk's edges are mu_1^2 alpha_t^2/psi_p, mu_1^2 beta_t^2/psi_p, etabar_star/psi_p, and in the stated proportional limit (d_int fixed, d_lat->infinity) ||x||^2/d_lat -> e^{-2t}sigma_perp^2+Delta_t = beta_t^2, so etabar_star is d_lat-independent; at psi_p=64 fixed all three edges and hence tau_gen, tau_mem and their difference are d_lat-independent. Two 'leading-order' statements in the same appendix contradict each other.


**Referee correction:** Three refinements. (1) At finite d_lat there IS a subleading dependence in the right direction: ||x_t||^2/d_lat = [s^2 + d_int*sigsig^2 + (d_lat-d_int)*sigperp^2]/d_lat*e^{-2t} + Delta_t decreases toward beta_t^2 as d_lat grows, so etabar_star shrinks and tau_sample grows -- but this is a finite-size correction that vanishes in the limit the theorem is stated in, and it is not the mechanism the paper claims. (2) The theorem's edge normalization is itself inconsistent with its own Eq. (U-lin): U^lin = (mu_1^2/(p*d_lat)) W M_t W^T with the nonzero eigenvalues of W M_t W^T equal to ~p*spec(M_t) gives edges mu_1^2 alpha_t^2/d_lat, not /psi_p. So the only quantitative content of the theorem is off by a factor p/d_lat^2. (3) The conclusion may nevertheless be true empirically: in the paper's own main-text MLP protocol (multiseed_runs/exp2_mlp_dlat_sn05_5m_bonnaire_hidden256, hidden=256, 5M steps, 5 seeds) the first step with memorization fraction >1% moves 150k -> 250k -> 300k -> 400k -> 600k -> 1.1M -> 2.55M -> 4.85M -> never as d_lat goes 5->40. The defect is that Eq. (buffer-bound) is asserted, not derived, and is inconsistent with the theorem it is 'combined' with -- not that the phenomenon is absent.


---

## [BLOCKING] The submitted document contains zero theorem environments; Section 2 is a magenta TODO that is present in the compiled PDF

`theory-section-empty` · verdict **CONFIRMED** · kind gap · effort days  

**Where:** /Users/ryan/Desktop/latent_space_diffusion_analysis/ICLR_2026/sec-theory.tex (entire file, 8 lines); ICLR_2026/main.tex line ~135 \input{sec-theory}; ICLR_2026/main.pdf p.~(text line 216)

**Claim:** The paper is not currently submittable as a theory contribution. main.tex declares theorem/lemma/corollary environments and then never uses one: `grep -c "begin{theorem}" sec-appendix-integrated.tex` returns 0, and the same is true of every file main.tex \inputs. sec-theory.tex is a \todo stub, and pdftotext main.pdf shows the literal string "TODO: formalize the eigenmode learning story here" inside the compiled 50-page PDF.

**Evidence:** sec-theory.tex in full: "\todo{TODO: formalize the eigenmode learning story here. This section should state the RFNN gradient-flow timescale $\tau_i=1/\lambda_i$, define where $\taugen$ and $\taumem$ sit in the ordered spectrum, and derive the qualitative prediction that the $\dlat-\dint$ null-dimensional bulk delays the first sample-specific modes.}" README.md "Remaining manuscript notes" item 1 confirms this is known: "Section 2 is intentionally left as the theory section to formalize".

**Fix:** Either wire sec-appendix-fourbulk.tex and sec-rfnn-bounds.tex into main.tex (after fixing buffer-bound-is-a-non-sequitur and bulk-count-table-refutes-theorem), or rewrite the abstract and Contributions to make no theoretical claim at all and present the paper as a purely empirical spectral study.


**Referee:** Independently verified. ICLR_2026/sec-theory.tex is 8 lines, entirely a \todo{} (main.tex:58 defines \todo as magenta text, so it prints), and main.tex:121 \inputs it. pdftotext on ICLR_2026/main.pdf line 216 returns 'TODO: formalize the eigenmode learning story here.' The PDF timestamp is Aug 30 02:13, i.e. the current build. main.tex:47-55 declares theorem/proposition/lemma/corollary/definition/assumption/remark environments; grep for begin{theorem|lemma|corollary|proposition} across all ten inputted files (abstract, sec-intro, sec-theory, sec-design, sec-rfnn, sec-mlp, sec-real-data, sec-spectral-predictor, sec-discussion, sec-appendix-integrated) returns 0 in every file. ICLR_2026/README.md 'Remaining manuscript notes' item 1 confirms the authors know.


---

## [BLOCKING] Contribution 1 says "We extend Bonnaire's two-bulk theorem" when the extension is a self-declared sketch that is not in the document

`extend-theorem-overclaim` · verdict **CONFIRMED** · kind unsupported-claim · effort hours  

**Where:** /Users/ryan/Desktop/latent_space_diffusion_analysis/ICLR_2026/sec-intro.tex:25; source it points to: /Users/ryan/Desktop/latent_space_diffusion_analysis/sec-appendix-fourbulk.tex

**Claim:** "Extend the theorem" is not honest here, on three counts. (1) The extension is not in the submitted artifact at all. (2) The file that contains it titles itself \section{RFNN: sketch of an analytical derivation} and closes with "A full Stieltjes-transform / replica derivation ... is left to follow-up work" -- i.e. the authors' own status label is "sketch", not "theorem". (3) Bonnaire et al.'s result is a replica computation, which is a non-rigorous physics method; sec-rfnn.tex:46 nevertheless writes "\citeauthor{bonnaire2025} prove (replica limit, ...)". So the sentence extends a non-theorem with a non-proof, and calls the composite a theorem extension.

**Evidence:** sec-intro.tex:25: "We extend Bonnaire's two-bulk theorem to anisotropic data and identify a four-bulk eigenvalue structure...". sec-appendix-fourbulk.tex header: "\section{RFNN: sketch of an analytical derivation}"; final paragraph: "A full Stieltjes-transform / replica derivation that produces the bulk edges as exact roots of a polynomial fixed-point equation ... is left to follow-up work." sec-rfnn.tex:46: "\citeauthor{bonnaire2025} prove (replica limit, isotropic $\Sigmadata = \mathbb{I}$)".

**Fix:** Rewrite as: "We give a leading-order Hermite-expansion argument that generalizes the replica prediction of Bonnaire et al. to block-diagonal data covariance, and verify the resulting four-bulk structure numerically." Change "prove" to "predict (replica method)" at sec-rfnn.tex:46.


**Referee:** Sub-claims (1) and (2) verified. sec-intro.tex:25 says 'We extend Bonnaire's two-bulk theorem to anisotropic data and identify a four-bulk eigenvalue structure ... with bulk-count boundaries that land exactly at indices dint and dlat.' The only artifact containing that extension, sec-appendix-fourbulk.tex, is not \input by main.tex (main.tex inputs only sec-appendix-integrated), self-titles '\section{RFNN: sketch of an analytical derivation}', and ends 'A full Stieltjes-transform / replica derivation ... is left to follow-up work.' _next_steps/theory_plan.md:28 is even blunter: 'the appendix's rank-count is right but the derivation as written doesn't justify it.' So Contribution 1 promises a theorem extension that (a) is absent from the submitted document and (b) the authors label a sketch.


**Referee correction:** Sub-claim (3) is a weak nit and should be dropped from the framing. sec-rfnn.tex:46 writes '\citeauthor{bonnaire2025} prove (replica limit, isotropic Sigmadata = I)' -- the parenthetical already discloses the method, and 'prove' for a replica computation is standard usage in this literature, not a misrepresentation. The substantiated overclaim is narrower: 'extend Bonnaire's two-bulk theorem' should be 'give a leading-order Hermite-expansion argument generalizing the replica prediction of Bonnaire et al. to block-diagonal data covariance, verified numerically' -- and even that is only honest once the material is actually in the document.


---

## [BLOCKING] The only independent bulk-size measurement in the project contradicts the theorem's counts, yet is cited as confirming them

`bulk-count-table-refutes-theorem` · verdict **CONFIRMED** · kind error · effort days  

**Where:** Table tab:bulk-sizes at /Users/ryan/Desktop/latent_space_diffusion_analysis/ICLR_2026/sec-appendix.tex:172-192 and /Users/ryan/Desktop/latent_space_diffusion_analysis/final_report.tex:755-778; claimed to match at /Users/ryan/Desktop/latent_space_diffusion_analysis/sec-rfnn-bounds.tex:47-49 and final_report.tex:738-740

**Claim:** Theorem thm:fourbulk predicts counts (d_int, d_lat-d_int, n, p-d_lat-n) = (5, d_lat-5, 500, 64*d_lat-d_lat-500). The measured table gives B3 (noise-dim) = 27,30,35,40,48,60,70 for d_lat = 5,8,10,15,20,30,40, against predicted 0,3,5,10,15,25,35 -- an offset of +27 to +35 that never closes, and 27 modes assigned to a bulk that is predicted to be EMPTY at d_lat = d_int = 5. B2 (sample) = 138,133,305,216,311,458,578 against a predicted constant 500. Only B4 = 5 matches.

**Evidence:** final_report.tex:738-740 states "Empirical bulk sizes match the predicted $\dint, \dlat - \dint, \nsamp, \pwidth - \dlat - \nsamp$ counts across every configuration (\Cref{tab:bulk-sizes})." sec-rfnn-bounds.tex:47-49 states "Predicted bulk counts match the empirical \textsc{B1}\ldots\textsc{B4} sizes in Table~\ref{tab:bulk-sizes} across the full $\dlat$ sweep at the predicted indices." I verified B1+B2+B3+B4 = 64*d_lat exactly for every row, so the table is a genuine partition of the spectrum and the disagreement is not a transcription slip. Note also that Table~\ref{tab:bulk-sizes} does not exist anywhere in the ICLR_2026 compile set -- sec-rfnn-bounds.tex cites a table that would be an undefined reference if it were wired in.

**Fix:** Either report the peak-detected counts honestly with the offsets, and explain why the sample bulk detector returns 133-578 instead of n=500 (likely the sample bulk is not separately resolved and find_peaks is splitting the rank-null tail), or drop the "counts match" sentence entirely. Do not wire sec-rfnn-bounds.tex in with that sentence intact.


**Referee:** Verified arithmetic and provenance. Per the table's own caption ordering (ICLR_2026/sec-appendix.tex:176-179 and final_report.tex:759-762: 'B4 collapses to dint; B3 grows linearly with dlat-dint; B1 absorbs the rank-null tail'), B3 is the noise-dim bulk and B2 the sample bulk. B3 = 27,30,35,40,48,60,70 vs predicted d_lat-d_int = 0,3,5,10,15,25,35 -- offsets +27,+27,+30,+30,+33,+35,+35, including 27 modes in a bulk predicted to be empty at d_lat=d_int=5. B2 = 138,133,305,216,311,458,578 vs a predicted constant n=500. I reconfirmed the reviewer's partition check: B1+B2+B3+B4 = 320,512,640,960,1280,1920,2560 = 64*d_lat exactly for all seven rows, so this is a genuine partition of the spectrum, not a transcription error. The overclaiming sentences exist verbatim: final_report.tex:738-740 'Empirical bulk sizes match the predicted dint, dlat-dint, nsamp, pwidth-dlat-nsamp counts across every configuration' and sec-rfnn-bounds.tex:47-49 'Predicted bulk counts match the empirical B1...B4 sizes ... at the predicted indices.' I also confirmed tab:bulk-sizes and fig:bulk-sizes-distances appear nowhere in the ICLR_2026 compile set, so sec-rfnn-bounds.tex would produce an undefined reference if wired in as-is.


**Referee correction:** Two qualifications. (a) The archived appendix's own caption makes only the weaker claim ('B3 grows linearly with dlat-dint'), which the data roughly supports in slope (1.23 measured vs 1.0 predicted) though not in offset; the falsifiable 'counts match' sentences live only in final_report.tex and sec-rfnn-bounds.tex, neither of which is in the submitted document. So this blocks wiring the theory appendix in; it does not currently mis-state anything in main.pdf. (b) The likely cause of B2 = 133-578 rather than 500 is diagnosable from the raw spectra: the predicted sample/rank-null boundary at index d_lat+n has an eigenvalue gap ratio of 1.00-1.01 in every run, so there is no spectral feature there for find_peaks to lock onto and the detector is necessarily splitting the sample+rank-null mass arbitrarily.


---

## [BLOCKING] Contribution 2 is about tau_mem and tau_gen; neither quantity is measured anywhere in the submitted paper

`tau-gen-tau-mem-never-measured` · verdict **REFUTED** · kind unsupported-claim · effort days  

**Where:** ICLR_2026/sec-intro.tex:26; searched all files \input by main.tex

**Claim:** Grepping \taugen/\taumem across abstract.tex, sec-intro.tex, sec-theory.tex, sec-design.tex, sec-rfnn.tex, sec-mlp.tex, sec-real-data.tex, sec-spectral-predictor.tex, sec-discussion.tex and sec-appendix-integrated.tex returns hits only in prose (intro, design, rfnn narrative, and the TODO). No figure, table, or number in the submitted document reports tau_gen or tau_mem as a function of d_lat. The claim "delaying $\taumem$ while leaving $\taugen$ approximately fixed" therefore has zero empirical backing inside the paper, and the RFNN section is spectra-only (sec-appendix-integrated.tex:126: "The RFNN experiments are spectral diagnostics, not sampled diffusion runs. They therefore do not have an image-level memorization percentage.").

**Evidence:** sec-intro.tex:26: "increasing $\dlat$ adds $\dlat - \dint$ extra eigenmodes that the model must absorb before reaching the sample bulk, delaying $\taumem$ while leaving $\taugen$ approximately fixed." sec-design.tex:6 quietly concedes the substitution: "In the trainable and real-data experiments, we report the same transition more directly as the fraction of generated samples counted as memorized over training" -- a final level, not a time.

**Fix:** Either add a tau_gen/tau_mem-vs-d_lat figure with a stated crossing criterion and censoring rule, or restate contribution 2 in terms of what is measured (memorization fraction at a fixed step budget) and drop the assertion about tau_gen being fixed, which nothing in the paper tests.


**Referee:** The claim 'No figure, table, or number in the submitted document reports tau_gen or tau_mem as a function of d_lat' is false. ICLR_2026/sec-appendix-integrated.tex (which main.tex:137 does \input) contains, at lines 905-909, 'CelebA hitting times by dlat' (figures/celeba_spectral_tau_vs_d.pdf) and 'CIFAR-10 hitting times by dlat' -- exactly a tau_mem-vs-d_lat plot -- plus predicted-vs-observed hitting-time panels. It also states the crossing criterion and the censoring rule the finding asks for: line ~800, 'For a memorization level q, the predicted hitting time is tau_q(d)=inf{s: m_d(s)>=q}. Empirical hitting times are censored when a seed never reaches level q within 5M steps.' The grep that produced the finding searched for the macros \taugen/\taumem and missed the section that measures the quantity under the name 'hitting time'. The finding's own proposed fix is therefore already implemented in the appendix.


**Referee correction:** The residual valid point, which should be the finding: tau_gen is never measured anywhere in the submitted document, so the second half of Contribution 2 ('while leaving taugen approximately fixed') is untested. Separately, no synthetic tau_mem-vs-d_lat figure exists even though the data does and is striking -- in multiseed_runs/exp2_mlp_dlat_sn05_5m_bonnaire_hidden256 the median first step with memorization fraction >1% is 150k, 250k, 300k, 400k, 600k, 1.1M, 2.55M, 4.85M, never, never for d_lat = 5,8,10,12,15,20,25,30,35,40. Plotting that would be the single strongest empirical panel in the paper and it is currently omitted; sec-appendix-integrated.tex:382 explains the omission as a deliberate preference for full curves over hitting times.


---

## [BLOCKING] The paper's central novelty premise -- "Bonnaire assumes isotropic covariance" -- is factually false; their Theorem 3.1 is stated for arbitrary rho_Sigma

`bonnaire-already-proves-general-sigma` · verdict **—** · kind unsupported-claim · effort days  

**Where:** /Users/ryan/Desktop/latent_space_diffusion_analysis/ICLR_2026/sec-intro.tex:17 and :25 (contribution 1); /Users/ryan/Desktop/latent_space_diffusion_analysis/ICLR_2026/sec-rfnn.tex:46; vs. Bonnaire et al. arXiv:2505.17638v2, Assumptions (i)-(ii) p.7, Theorem 3.1 Eqs (17)-(19), SM Lemma C.1

**Claim:** sec-intro.tex:17 asserts "Their analysis assumes isotropic data covariance, which collapses the distinction between the d_int-dimensional signal subspace and the d_lat - d_int null directions," and sec-rfnn.tex:46 says they prove the two-bulk split in the "replica limit, isotropic Sigma_data = I". Both are wrong. Bonnaire et al. explicitly assume only that "the data distribution Px has sub-Gaussian tails and a covariance Sigma = E[xx^T] with bounded spectrum" whose ESD converges to "a deterministic density rho_Sigma(lambda)", and their Theorem 3.1 fixed-point system integrates against drho_Sigma(lambda). Their SM says outright: "we outline the derivation of the Gaussian Equivalence Principle (GEP) for the matrices U, Utilde, V and Vtilde under arbitrary input covariance. This generalizes the approach developed in [15], which considered only the case of data drawn from N(0, I_d)." The paper's "four-bulk theorem" is therefore obtained by substituting the two-atom measure rho_Sigma = (d_int/d_lat) delta(lambda - sig_sig^2) + (1 - d_int/d_lat) delta(lambda - sig_perp^2) into an already-published theorem -- it is a corollary, not a new theorem.

**Evidence:** Bonnaire p.7: "(ii) the data distribution Px has sub-Gaussian tails and a covariance Sigma = E_Px[xx^T] with bounded spectrum. We assume that the empirical distribution of eigenvalues of Sigma converges weakly in the high dimensional limit to a deterministic density rho_Sigma(lambda)". Theorem 3.1 Eq (17): s = INT drho_Sigma(lambda) 1/(shat(q) + lambda rhat(r,q)); Eq (18): r = INT drho_Sigma(lambda) lambda/(shat(q) + lambda rhat(r,q)). SM Sec. C.3: "the derivation of the GEP ... under arbitrary input covariance. This generalizes the approach developed in [15], which considered only ... N(0, I_d)."

**Fix:** Delete the isotropy claim from sec-intro.tex:17 and sec-rfnn.tex:46. Restate contribution 1 as an explicit *evaluation* of Bonnaire's Theorem 3.1 at a two-atom rho_Sigma, and either (a) actually solve Eqs (17)-(19) for that rho_Sigma numerically and overlay the resulting rho(lambda) on Fig. 5 (this is a half-day of root-finding and would be the strongest possible version of the section), or (b) drop the theorem framing entirely and present the four bulks as an empirical spectral characterization of the latent-diffusion regime. Do not claim to "extend" a theorem that already covers the case.


---

## [BLOCKING] Bonnaire's Figure 4 (right) already computes the anisotropic two-atom case, and its stated conclusion is a direct competing prediction for tau_mem

`bonnaire-fig4-right-already-anisotropic` · verdict **—** · kind inconsistency · effort days  

**Where:** /Users/ryan/Desktop/latent_space_diffusion_analysis/ICLR_2026/sec-rfnn.tex:59 vs. sec-intro.tex:17 and abstract.tex; Bonnaire Fig. 4 caption, p.8

**Claim:** sec-rfnn.tex:59 concedes that Bonnaire "already note (Figure 4, right panel) that rho_2 develops internal structure when Sigma_data has multiple eigenvalues," which flatly contradicts the intro and abstract. Worse, the Fig. 4 caption states a result the paper never engages with: the *sample* bulk rho_1 -- the one that sets tau_mem, the paper's headline quantity -- is invariant to the block structure of Sigma and depends on Sigma only through sigma_x^2 = Tr(Sigma)/d. Under the cited baseline theory, therefore, splitting Sigma_data into a signal block and a null block cannot move tau_mem at all except through the scalar Tr(Sigma_data)/d_lat; the entire "buffer between the population and sample bulks" story has no purchase on the left edge of rho_1.

**Evidence:** Bonnaire Fig. 4 caption: "(Right) Same as (Middle), but with rho_Sigma(lambda) = (1/2)delta(lambda-0.5) + (1/2)delta(lambda-1.5). The first bulk in blue remains unchanged, as it depends only on sigma_x^2 = Tr(Sigma)/d = 1 in both cases, while the second bulk varies with Sigma." Theorem 3.2: rho_1 support = [s_t^2 + v_t^2(1 - sqrt(psi_p/psi_n))^2, s_t^2 + v_t^2(1 + sqrt(psi_p/psi_n))^2], and Eqs (12)-(14) define s_t^2, v_t^2 purely through sigma_x^2 = Tr(Sigma)/d.

**Fix:** Compute Tr(Sigma_data)/d_lat = [s^2 + d_int*sig_sig^2 + (d_lat-d_int)*sig_perp^2]/d_lat across the d_lat sweep and plot Bonnaire's predicted rho_1 left edge and tau_mem = psi_p/(Delta_t lambda_min) against the measured tau_mem. Either the baseline already explains the d_lat trend (in which case the buffer mechanism must be withdrawn) or it does not (in which case this overlay is the paper's single most convincing figure). This must be in the paper.


---

## [BLOCKING] The proposed sample-bulk edge eta_star_bar/psi_p has no psi_n dependence, which contradicts Bonnaire's main result tau_mem ~ n

`sample-bulk-edge-loses-n-dependence` · verdict **—** · kind error · effort days  

**Where:** /Users/ryan/Desktop/latent_space_diffusion_analysis/sec-appendix-fourbulk.tex:139-141 (Lemma 2) and :223 (Theorem table, "Sample | n | eta_star_bar/psi_p")

**Claim:** Lemma 2 asserts the sample-bulk eigenvalues "are concentrated around eta_star_bar/psi_p" with eta_star_bar = sum_{k>=3} mu_k^2 E[(||x||^2/d_lat)^k] -- a quantity with no dependence whatsoever on n or psi_n. Bonnaire's rho_1 support is s_t^2 + v_t^2 (1 -/+ sqrt(psi_p/psi_n))^2, and it is precisely the psi_p/psi_n factor in that edge that produces their headline result ("tau_mem becomes large and of order psi_n/Delta_t, thus implying a scaling of tau_mem with n"). The proposed theorem, taken literally, predicts a memorization time independent of the training-set size, contradicting both the paper it builds on and that paper's U-Net experiments.

**Evidence:** sec-appendix-fourbulk.tex:139: "its nonzero eigenvalues are concentrated around $\bar\eta_\star/\psi_p$ where $\bar\eta_\star = \E_{x\sim P_t}[\eta_\star(x,x)]$". Bonnaire Theorem 3.2: rho_1 support [s_t^2 + v_t^2(1-sqrt(psi_p/psi_n))^2, s_t^2 + v_t^2(1+sqrt(psi_p/psi_n))^2]; text p.9: "In the overparameterized regime p >> n, tau_mem becomes large and of order psi_n/Delta_t, thus implying a scaling of tau_mem with n."

**Fix:** Replace Lemma 2's edge with Bonnaire's, cited: lambda_min^{sample} = s_t^2 + v_t^2 (1 - sqrt(psi_p/psi_n))^2 with s_t^2, v_t^2 from their Eqs (12)-(14) evaluated at sigma_x^2 = Tr(Sigma_data)/d_lat. Then check numerically against sigma_noise_0.5/four_bulk/ that the measured sample-bulk edge moves with n as sqrt(psi_p/psi_n) predicts; the existing sweeps hold n = 500 fixed, so an n-sweep at fixed d_lat is needed and is cheap.


---

## [BLOCKING] The rank-null bulk edge is not o(1): it is the strictly positive constant s_t^2, it lies at the left edge of the sample bulk, and its eigenvectors are provably irrelevant to both losses

`rank-null-edge-is-s_t-squared` · verdict **—** · kind error · effort hours  

**Where:** /Users/ryan/Desktop/latent_space_diffusion_analysis/sec-appendix-fourbulk.tex:224 (Theorem table row "Rank-null | p - d_lat - n | o(1)") and Theorem proof :230-238; vs. Bonnaire SM Lemma C.1 Eq (63), Lemma C.3 Eq (86), Theorem 3.2

**Claim:** The Gaussian-equivalent form of U proved by Bonnaire is U = GG^T/n + b_t^2 WW^T/d + s_t^2 I_p, so every eigenvalue carries an additive floor s_t^2, and the p - d - n "leftover" directions form a Dirac mass exactly at lambda = s_t^2 with weight 1 - (1+psi_n)/psi_p -- not a bulk at o(1). Since rho_1's support starts at s_t^2 + v_t^2(1-sqrt(psi_p/psi_n))^2, the delta sits at or below the sample bulk's left edge but is bounded away from zero. Bonnaire additionally prove these modes are dynamically inert: "The eigenvectors associated with delta(lambda - s_t^2) leave both training and test losses unchanged and are therefore irrelevant." The paper's four-bulk picture treats them as a genuine slowest bulk, which is both quantitatively and conceptually wrong, and it means the empirical "purple rank-null block" in Fig. 5 should be a spike, not a decaying tail.

**Evidence:** Bonnaire SM Lemma C.1 Eq (63): "U = GG^T/sqrt(n)sqrt(n) + b_t^2 WW^T/d + s_t^2 I_p"; Lemma C.3 Eq (86): "1 - 1/psi_p delta(lambda - (||sigma||^2 - mu_1^2(t))) + (1/psi_p) rho_bulk(lambda)"; Theorem 3.2 Regime I: "rho(lambda) = 1 - (1+psi_n)/psi_p delta(lambda - s_t^2) + ..."; p.9: "The eigenvectors associated with delta(lambda - s_t^2) leave both training and test losses unchanged and are therefore irrelevant."

**Fix:** Correct the table row to a Dirac at s_t^2 with the stated weight, cite Bonnaire Theorem 3.2, and check the empirical spectra in sigma_noise_0.5/four_bulk/ for the predicted spike (if the measured purple block is a spread tail rather than a delta at s_t^2, that is itself a finding worth reporting and needs explaining). Remove any argument that relies on the rank-null modes being the last to be absorbed.


---

## [BLOCKING] George, Veiga & Macris already prove the D-dim-subspace RFNN score result with a linear pencil whose block dimensions are exactly (p, d-D, D, n) -- the paper's four bulk counts -- and it is uncited

`george-veiga-macris-uncited` · verdict **—** · kind gap · effort days  

**Where:** /Users/ryan/Desktop/latent_space_diffusion_analysis/ICLR_2026/references.bib (absent); claim staked at ICLR_2026/sec-intro.tex:25 and sec-appendix-fourbulk.tex Theorem thm:fourbulk

**Claim:** "Denoising Score Matching with Random Features: Insights on Diffusion Models from Precise Learning Curves" (George, Veiga & Macris, arXiv:2502.00336, AISTATS 2026) -- Bonnaire's reference [15], and the paper Bonnaire say they generalize -- proves asymptotically exact test/train errors for an RFNN score when P_0 = N(0, Pi_||) is supported on a D-dimensional subspace of R^d with psi_D = D/d fixed. Their Theorem B.2 is derived via a linear pencil whose blocks have dimensions N1 = p, N2 = d - D, N3 = D, N4 = n, N5 = D: the exact rank classes the paper's "four-bulk theorem" claims as new (signal D, noise-dim d-D, sample n, feature space p). They also decompose the test error into parallel and orthogonal subspace components (E_test,|| and E_test,perp), which is the natural rigorous version of the paper's signal-vs-null-mode accounting. The paper cites neither this work nor Bonnaire's own pointer to it.

**Evidence:** arXiv:2502.00336 Theorem B.2: "Let M_|| be a D-dimensional subspace in R^D, Pi_|| a projection matrix onto it, and P_0 = N(0, Pi_||)"; proof: "The block dimensions are N1 = p, N2 = d - D, N3 = D, N4 = n, N5 = D"; and "U(q) := GG^T/n + h mu_1^2 WW^T/d + q W Sigma W^T/d + s^2 I_p = GG^T/n + (h mu_1^2 + q) W_|| W_||^T/d + (h mu_1^2 + h q) W_perp W_perp^T/d + s^2 I_p". Bonnaire p.8: "George et al. (2025) derive a coupled system of equations characterizing the Stieltjes transform of the eigenvalue density rho(lambda) of U for isotropic Gaussian data that lie in a D-dimensional subspace with D <= d and D = O(d)." grep over ICLR_2026/references.bib for "george" returns no matching entry.

**Fix:** Cite George, Veiga & Macris (2502.00336) in the intro and in the theory section. Then decide honestly whether Theorem 1 is anything more than their result at sig_perp > 0 rather than sig_perp = 0. The defensible remaining novelty is (a) the strictly-positive-sig_perp case (they take the null block to exactly zero variance) and (b) the *training-time* reading of the spectrum, which they do not do (they analyse the ridge minimizer, not gradient-flow mode absorption). Say exactly that.


---

## [BLOCKING] No cited work supports multiplying a bulk's *cardinality* by a per-mode timescale; every published definition of tau_mem in this literature is an inverse spectral edge

`buffer-bound-has-no-support-in-literature` · verdict **—** · kind unsupported-claim · effort days  

**Where:** /Users/ryan/Desktop/latent_space_diffusion_analysis/sec-rfnn-bounds.tex:56-63 (Eq (buffer-bound)) and :64-70; vs. Bonnaire p.9

**Claim:** Eq (buffer-bound) writes tau_mem - tau_gen >= (d_lat - d_int) * psi_p/(mu_1^2 beta_t^2), i.e. buffer *width* times per-mode relaxation time. Bonnaire's Eq (10) gradient flow -- the same linear ODE this paper uses -- decouples completely in the eigenbasis, so modes sharing an eigenvalue relax simultaneously, not serially; there is no "time to traverse a bulk" proportional to its cardinality. Correspondingly, every timescale in the cited literature is an inverse edge: Bonnaire define "tau^{-1} = psi_p/(Delta_t lambda_min)" and read tau_gen off rho_2's edge and tau_mem off rho_1's left edge. The paper's own sec-rfnn.tex text ("the RFNN absorbs modes in order of decreasing lambda_i, so the trajectory traverses the four bulks sequentially") reads the eigenvalue *ordering* as a serial schedule, which the ODE does not license. As written, contribution 2 has no derivation and no precedent.

**Evidence:** sec-rfnn-bounds.tex Eq (buffer-bound): "\taumem - \taugen \ge (\dlat - \dint) \cdot 1/\lambda_{\min}^{noise-dim} = (\dlat-\dint)\cdot \psi_p/(\mu_1^2\beta_t^2)". Bonnaire p.9: "We recall that training timescales are directly related to eigenvalues lambda via the relation tau^{-1} = psi_p/(Delta_t lambda_min)"; "[tau_mem] is related to the bulk rho_1, and scales as psi_p/(Delta_t lambda_min), where lambda_min is the left edge of rho_1."

**Fix:** Either derive the (d_lat - d_int) factor from an explicit tolerance criterion on a *summed* residual (e.g. tau_mem defined as the first time sum_{i in sample bulk} (a_i - a_i*)^2 exceeds epsilon, where the count enters through the sum and one can genuinely get a log(count) or count-dependent term) -- and then it will be logarithmic, not linear -- or drop Eq (buffer-bound) and state the mechanism as an ordering claim: tau_mem = 1/lambda_min^{sample}, tau_gen = 1/lambda_min^{signal}, and the gap changes with d_lat because sig_x^2 and psi_n change. The second option is short, correct, and directly citable to Bonnaire.


---

## [BLOCKING] Every bulk edge in the four-bulk table is mis-normalized; the table violates conservation of tr(U) and diverges with d_lat

`trace-violation-psi-p` · verdict **CONFIRMED** · kind error · effort days  

**Where:** sec-appendix-fourbulk.tex, Theorem \ref{thm:fourbulk} edge table (lines 214-227); Lemma \ref{lem:Ulin} statement (lines 116-121) and proof (lines 123-131); Lemma \ref{lem:Udiag} statement (lines 137-142)

**Claim:** The table's edges mu_1^2 alpha_t^2/psi_p, mu_1^2 beta_t^2/psi_p, etabar/psi_p do not sum to the exact trace of U, which is O(1) and d_lat-independent. Summing the table gives approximately mu_1^2 beta_t^2 d_lat/psi_p = mu_1^2 beta_t^2 d_lat^2/p, which diverges linearly in d_lat at fixed psi_p. The correct denominators are d_lat for the two linear bulks and n for the sample bulk. This is not cosmetic: it decides whether lambda_i shrinks as 1/d_lat, which is the crux of the entire buffer argument.

**Evidence:** Exact trace: phi = tanh(Wx/sqrt(d))/sqrt(p) so tr(U) = (1/n) sum_mu E||phi(x_t^mu)||^2 = E_x E_{z~N(0,tau_x)}[tanh^2 z] <= 1 with tau_x = ||x||^2/d_lat. Algebra: W has iid N(0,1) entries (the appendix itself asserts E[W^T W/d_lat] = psi_p I), so W^T W = p I + fluct, hence the nonzero eigenvalues of W M_t W^T are p*lambda(M_t) = psi_p d_lat lambda(M_t), NOT psi_p lambda(M_t) as the Lemma 1 proof says. Multiplying by the prefactor mu_1^2/(p d_lat) in Eq (U-lin) gives lambda = mu_1^2 lambda(M_t)/d_lat. The lemma's mu_1^2 lambda(M_t)/psi_p is larger by the factor d_lat/psi_p = d_lat^2/p. The corrected table conserves trace exactly: d_int*a*alpha^2/d + (d-d_int)*a*beta^2/d + n*etabar/n = a*tr(M_t)/d + etabar = a*tau + etabar = E[tanh^2(sqrt(tau) g)], which is tr(U) identically. Numerics (code's U is p*U_paper since compute_U omits the 1/p): the noise-dim bulk in code units is d_lat-INDEPENDENT (mean over indices [5:d] = 5.22, 7.03, 8.36, 9.01, 9.11, 9.32, 9.35, 9.22 for d_lat = 10..200 at sigma_perp=0.5), i.e. lambda_paper = lambda_code/p ~ 1/d_lat. The paper's formula predicts lambda_code = d_lat*mu_1^2*beta^2 = 1.0, 1.9, 3.9, 5.8, 7.8, 9.7, 14.6, 19.4 -- wrong trend and up to 5x off. Measured tr(U_code)/p = 0.44, 0.46, 0.29, ..., 0.217, 0.209 (saturating, O(1)) confirming tr(U) = O(1).

**Fix:** Replace 1/psi_p by 1/d_lat in the signal and noise-dim edges, and by 1/n in the sample edge, then re-derive every downstream consequence. Critically, tau_gen = 1/lambda_signal = d_lat/(a alpha_t^2) is then LINEAR in d_lat, contradicting the paper's core claim (3) that tau_gen is d_lat-independent. Under the corrected normalization tau_mem/tau_gen = alpha^2/beta^2 is d_lat-independent, i.e. widening d_lat rescales the whole clock rather than differentially delaying memorization. The paper must either restate the buffer claim in units where the clock rescaling is divided out, or abandon it.


**Referee:** Verified the algebra and the numerics independently. Lemma 1's proof (sec-appendix-fourbulk.tex:123-131, mirrored live at ICLR_2026/sec-appendix.tex:701-708) writes E[W^T W/d_lat] = psi_p I, i.e. E[W^T W] = p I, then concludes the nonzero eigenvalues of W M_t W^T are 'the eigenvalues of M_t rescaled by psi_p'. They are rescaled by p = psi_p*d_lat. With the prefactor mu_1^2/(p*d_lat) in Eq (U-lin) the edge is mu_1^2 lambda(M_t)/d_lat, not /psi_p. The table is therefore off by d_lat/psi_p = d_lat^2/p and its sum diverges linearly in d_lat at fixed psi_p, while tr(U) is provably O(1): I confirmed tr(U_code)/p equals E[tanh^2(sqrt(tau)g)] to 4 digits at large d (0.2092 measured vs 0.2093 predicted at d=200; 0.2171/0.2181 at d=150). Direct spectral test at sigma_perp=0.5, code units (lambda_code = p*lambda_paper): measured noise-dim bulk mean = 5.18, 7.10, 8.63, 9.25, 9.72, 9.95, 10.38, 10.63 for d_lat=10..200; the paper's formula predicts d_lat*mu_1^2*beta^2 = 0.97, 1.94, 3.89, 5.83, 7.77, 9.72, 14.58, 19.43 (wrong trend, 5x off at d=10, 1.8x the other way at d=200). The corrected psi_p*a(tau)*beta^2 gives 4.76, 6.66, 8.37, 9.16, 9.62, 9.93, 10.36, 10.60 - a few percent across the whole sweep. Same for the signal bulk: measured 49.56...112.79 vs corrected psi_p*a(tau)*alpha^2 = 49.69...110.61 (~2%) vs paper's d*mu_1^2*alpha^2 = 10.1...202.8. The error is real, material, decides the 1/d_lat scaling, and is in the live draft, not only the orphaned file.


**Referee correction:** The normalization error and the fix (1/d_lat for the two linear bulks, 1/n for the sample bulk) are correct and blocking. Two parts of the finding need amending. (a) 'The noise-dim bulk in code units is d_lat-INDEPENDENT' is an overstatement: it grows 5.18 -> 10.63 over the sweep, tracking a(tau); the correct statement is that it saturates rather than growing linearly. (b) The damaging conclusion 'tau_mem/tau_gen = alpha^2/beta^2 is d_lat-independent' misidentifies tau_mem with the NOISE-DIM bulk. Under the paper's own framing memorization is the SAMPLE bulk, whose corrected edge is etabar_star/n, so the invariant is tau_mem/tau_gen = n*a(tau)*alpha_t^2/(d_lat*etabar_star), which I measure rising from 1368 to 1816 over d_lat=10..200 (weakly increasing, not constant). Further, the paper's own RFNN protocol already implements the reviewer's proposed remedy: code/v3/run_experiment.py:543 sets lr = 0.01*d_latent/delta_t, so the gradient-flow clock is rescaled by d_lat by construction and the corrected 1/d_lat in lambda cancels in step units. The correct conclusion is therefore that the buffer claim must be RE-DERIVED in step units with the sample bulk as tau_mem, not that it must be abandoned.


---

## [BLOCKING] Lemma 2's statement and its own proof give mutually inconsistent eigenvalues (etabar/psi_p vs etabar^2/(p n)); both are wrong, and Eq (U-diag) double-counts eta_star

`lem2-eigenvalue-double-count` · verdict **CONFIRMED** · kind inconsistency · effort hours  

**Where:** sec-appendix-fourbulk.tex, Eq (U-diag) (lines 77-81), Lemma \ref{lem:Udiag} statement (lines 137-142), proof (lines 152-156)

**Claim:** The lemma claims nonzero eigenvalues concentrate at etabar_star/psi_p; the proof computes Lambda_mu ||phi^perp||^2 -> eta_star^2/(p n). These differ by a factor etabar*psi_p/(p n) = etabar/n^... i.e. by etabar/p * (n/psi_p)^-1; numerically they differ by 5 orders of magnitude. The root cause is that Eq (U-diag) carries an explicit eta_star(x,x) factor AND an outer product phi^perp phi^perp^T whose squared norm the text itself declares to equal eta_star(x,x) (line 86), AND a spurious 1/p. The correct object is U^diag = (1/n) sum_mu phi^perp(x_t^mu) phi^perp(x_t^mu)^T with eigenvalues eta_star(x^mu,x^mu)/n, i.e. sample edge etabar_star/n.

**Evidence:** Derivation: with psi(x) = tanh(Wx/sqrt d) and phi = psi/sqrt(p), U = (1/(p n)) sum_mu psi psi^T. Splitting psi = mu_1 Wx/sqrt(d) + psi^perp with E_W||psi^perp(x)||^2 = p eta_star(x,x) gives U^diag = (1/(p n)) sum psi^perp psi^perp^T, whose nonzero eigenvalues are ||psi^perp||^2/(p n) = eta_star/n. In the paper's own phi^perp = psi^perp/sqrt(p) normalization (line 86: E||phi^perp||^2 = eta_star), U^diag = (1/n) sum phi^perp phi^perp^T -- the 1/p and the eta_star prefactor in Eq (U-diag) are both spurious. Numerical adjudication at sigma_perp=0.5 (code units, lambda_code = p*lambda_paper): measured mean of the sample-bulk indices [d:d+n] = 0.0456 (d=40), 0.0541 (d=100), 0.0620 (d=200); p*etabar/n = 0.0625, 0.0796, 0.1146 (right order, within 1.4-1.8x); lemma's p*etabar/psi_p = d*etabar = 0.488, 0.622, 0.895 (8-14x too large); proof's p*etabar^2/(p n) = 3.0e-7, 7.7e-8, 4.0e-8 (1e5 times too small). etabar computed as E[tanh^2(sqrt(tau)g)] - tau*(E[tanh'(sqrt(tau)g)])^2.

**Fix:** Rewrite Eq (U-diag) as U^diag = (1/n) sum_mu phi^perp(x_t^mu) phi^perp(x_t^mu)^T (no eta_star prefactor, no 1/p), and state the sample-bulk edge as etabar_star/n in both Lemma 2 and the theorem table. Note this correction factor (n/psi_p) differs from the linear bulks' correction factor (d_lat/psi_p), so the errors do NOT cancel out of the bulk-gap ratio (see finding bulk-gap-missing-psi-n).


**Referee:** Read directly. Eq (U-diag) (sec-appendix-fourbulk.tex:77-81) carries a 1/(p*n) prefactor AND an explicit eta_star(x,x) factor AND an outer product phi^perp phi^perp^T whose squared norm the text itself sets to eta_star(x,x) at line 86. So the per-term trace in the paper's own Eq is eta_star^2/(p n), which is what the proof (lines 152-156) computes, while the Lemma statement (lines 137-142) asserts etabar_star/psi_p. Statement and proof are mutually inconsistent by a factor p*n/(psi_p*etabar) = d_lat*n/etabar. The correct object is U^diag = (1/n) sum phi^perp phi^perp^T with trace etabar_star. Numerical adjudication at sigma_perp=0.5 (code units): measured sample-bulk mean = 0.0362 (d=10), 0.0456 (d=40), 0.0621 (d=200); p*etabar/n = 0.0572, 0.0640, 0.1194 (within 1.4-1.9x); the Lemma's d_lat*etabar = 0.447, 0.500, 0.933 (8-15x too large); the proof's etabar^2/n = 3.1e-6, 3.1e-7, 4.4e-8 (5-7 orders too small). Independent trace check: measured residual mass sum(ev[d:])/p = 0.0283, 0.00965, 0.00317 vs etabar = 0.0447, 0.0125, 0.00467 (right order), vs the paper's implied etabar^2/p = 3.1e-6, 6.1e-8, 1.7e-9. The spurious eta_star and spurious 1/p are both confirmed by direct reading.


**Referee correction:** Substance fully confirmed; the proposed fix (sample edge = etabar_star/n) is the one the data supports. Two prose errors in the finding itself: the stated discrepancy factor ('etabar*psi_p/(p n) = etabar/n^... i.e. by etabar/p * (n/psi_p)^-1') is garbled - the actual ratio of statement to proof is p*n/(psi_p*etabar) = d_lat*n/etabar, about 1.6e6 at d_lat=40; and that is six orders of magnitude, not the five claimed.


---

## [BLOCKING] The asymptotic-orthonormality claim v_mu^T v_nu = delta + o_d(1) is false for clustered data, and the sample bulk is not concentrated at a single edge

`lem2-orthonormality-false` · verdict **CONFIRMED** · kind unsupported-claim · effort research-project  

**Where:** sec-appendix-fourbulk.tex, Lemma \ref{lem:Udiag} proof (lines 149-152), and the supporting assertion at lines 85-88 ('E_W[phi^perp(x)phi^perp(y)^T] = 0 for x != y')

**Claim:** The kernel decomposition's diagonal-only term is not an indicator 1[x=y]; it is the smooth kernel eta_star(x,y) = sum_{k>=3} mu_k^2 rho^k, which is negligible only if the input correlation rho = x^T y/(||x|| ||y||) is o(1) for ALL pairs. With k=10 clusters and n=500 (50 points per cluster) the same-cluster rho is O(1), so residual features of same-cluster points have normalized overlap approximately rho^3, and the Gram of the 50 residuals in a cluster develops a spike of size 1 + 49 rho^3. The lemma's conclusion ('nonzero eigenvalues concentrated around a single value') therefore fails exactly in the regime the paper cares about, since the sample bulk is supposed to be the memorization channel and near-duplicate/same-cluster points are the memorization-relevant case. The claim is asserted, never proven, and it is what makes the whole rank-n picture work.

**Evidence:** Direct measurement (projecting each tanh feature vector onto col(W)^perp, exactly the paper's phi^perp, then normalizing the residual Gram to unit diagonal): at d_lat=40, sigma_perp=0.5, same-cluster normalized overlap = +0.0765 vs predicted cos^3 = 0.0547; the normalized residual Gram's top eigenvalues are 14.96, 12.38, 10.36, 9.77, 7.44, 5.57, ... instead of all-ones. At sigma_perp=0.01: same-cluster overlap +0.310 vs cos^3 = 0.244; top eigenvalues 51.4, 42.4, 37.5, 32.4, 24.3, 15.9, ... Ablation with the paper's exact pipeline (generate_data + compute_U, d_lat=40, n=500, k=10, 50 noise draws), setting scale=3.0 vs scale=0.0 (clusters off): sample-bulk top eigenvalue 0.343 -> 0.099 at sigma_perp=0.5 and 0.520 -> 0.082 at sigma_perp=0.01, while the sample-bulk median barely moves (0.0296 -> 0.0171 and 0.0030 -> 0.0010). So the top of the 'sample bulk' is inflated 3.5-6.3x purely by cluster-induced residual-feature correlation. The saved real spectra agree: sigma_perp=0.5, d=40 gives sample max 0.374 / median 0.0294 / min 0.00355 -- a spread of 105x, not a concentrated bulk. At sigma_perp=0.01, d=40 the ten leading sample-bulk eigenvalues (0.520, 0.509, 0.480, 0.432, 0.417, 0.366, 0.349, 0.310, 0.298, 0.289 -- k=10 of them) sit within a factor 1.2 of the noise-dim minimum (0.637), i.e. the two bulks touch and the strict absorption ordering that the buffer argument needs is gone.

**Fix:** Replace the 1[x=y] step with an honest treatment of the nonlinear kernel eta_star(x,y) on anisotropic/clustered data: U^diag = (1/n) Phi^perp^T G Phi^perp with G_{mu nu} = eta_star(x^mu, x^nu), and state the sample bulk as the spectrum of the n x n matrix G/n. Then (i) the bulk is an MP-type band, not a point, and (ii) it carries k cluster spikes at approximately (n/k) rho^3 etabar/n plus d_int signal-subspace spikes from the cubic term (these appear even with clusters off: 5 outliers at 4.89, 4.59, 4.09, 4.02, 3.81 in the normalized Gram). Both the 'concentrated around' phrasing and any argument that treats the sample bulk as a single tau_mem must be revised.


**Referee:** Reproduced independently with the project's own generate_data and a fresh W. Projecting each tanh feature vector onto col(W)^perp (exactly the paper's phi^perp) and normalizing the residual Gram to unit diagonal: at d_lat=40, n=500, k=10, sigma_perp=0.5 the mean same-cluster off-diagonal overlap is +0.0730 against the cos^3 prediction +0.0801 (different-cluster: +0.0017); at sigma_perp=0.01 it is +0.2972 against +0.3086. So the off-diagonal is the smooth kernel eta_star(x,y) ~ rho^3, not the indicator 1[x=y] asserted at Eq (kernel-decomp) line 53 and at line 85. The normalized residual Gram's top eigenvalues are 16.41, 12.90, 9.97, 8.10, 7.92, 5.19, 4.66, 4.45 (sigma_perp=0.5) and 55.4, 47.2, 35.0, 28.3, 27.3, 16.1, 14.3, 13.5 (sigma_perp=0.01), not all-ones - v_mu^T v_nu = delta + o_d(1) is false. Ablation through the paper's exact pipeline (generate_data + compute_U, d_lat=40, n=500, k=10, 50 noise draws), scale=3.0 vs scale=0.0: sample-bulk top eigenvalue 0.400 -> 0.106 at sigma_perp=0.5 and 0.604 -> 0.087 at sigma_perp=0.01, while the sample-bulk median barely moves (0.0294 -> 0.0171 and 0.0030 -> 0.0010). With clusters off there are still 5 outliers (5.22, 4.53, 4.44, 4.37, 3.88) in the normalized residual Gram, i.e. d_int signal-subspace spikes from the cubic term, exactly as the finding predicts. The saved spectrum at sigma_perp=0.5, d=40 has sample max 0.374 / median 0.0294 / min 0.00355, a 105x spread - not 'concentrated around a single value'. And at sigma_perp=0.01, d=40 the ten leading sample eigenvalues (0.560, 0.553, 0.488, 0.428, 0.417, 0.367, 0.361, 0.310, 0.305, 0.300) sit within 1.15x of the noise-dim minimum 0.637: the bulks touch.


**Referee correction:** Confirmed in substance and effect size; my seeds differ from the reviewer's so the individual numbers differ slightly (same-cluster overlap 0.0730 not 0.0765; ablation 0.400->0.106 not 0.343->0.099), but every sign, ordering and magnitude is reproduced. The proposed fix (state the sample bulk as spec(G/n) with G_{mu nu} = eta_star(x^mu,x^nu)) is what the data supports.


---

## [BLOCKING] `precompute_score_params` rotates Sigma_t^{-1} the wrong way; every synthetic `score_error` number in the paper is measured against a fictitious vector field

`score-error-rotation-transpose-bug` · verdict **CONFIRMED** · kind error · effort days  

**Where:** /Users/ryan/Desktop/latent_space_diffusion_analysis/code/experiment_v2.py:218 and /Users/ryan/Desktop/latent_space_diffusion_analysis/code/v3/lib/true_score.py:63 (`sigma_t_inv = Q_t.T @ torch.diag(1.0 / sigma_t_diag) @ Q_t`)

**Claim:** The data frame is x_rot = Q x_orig (`data = data @ Q.T`, experiment_v2.py:131, v3/lib/data_synthetic.py:58), so Sigma_t in the data frame is Q D Q^T and its inverse is Q D^{-1} Q^T. Both the v2 and the v3 libraries compute Q^T D^{-1} Q instead. `true_score(...)` therefore returns a vector field that is not the score of any distribution whenever d_lat > d_int (at d_lat = d_int the block is isotropic and the bug is inert). Every `score_error` in sigma_noise_0.01/, sigma_noise_0.5/, multiseed_runs/ and code/v3/results/ is affected, i.e. Fig. 1(b) of sec-mlp.tex, Figs. late_score_error_sn001/sn05, mlp_score_error_curves_*, synthetic_mlp_score_error*, and every score-error number in n_shape_heuristic_derivation.md.

**Evidence:** Empirical covariance check on 2e5 samples, d_lat=40: ||C - Q D Q^T||_F = 2.17 (residual is the anisotropic cluster-mean block) vs ||C - Q^T D Q||_F = 8.76 with ||C||_F = 6.62. Recomputing E||s*||^2/d at t=0.1 in the correct frame gives per-signal-dim 0.73 and per-null-dim 5.51 = 1/lambda_null exactly (1/(e^{-2t}sigma_perp^2+Delta_t) = 5.514), for every d_lat; the buggy field gives per-signal-dim 21.6 and per-null-dim 5.02 at d=40, and 1856/67 at t=0.01. The buggy field reproduces the recorded step-1 score errors to 4 significant figures: d_lat = 8/15/40/200 predicted 4.914/8.535/7.096/5.878 vs recorded 4.916/8.537/7.096/5.876 (sigma_noise_0.01/exp2_mlp/raw_data/*/metrics.jsonl, first line); RFNN d=8/20/40 predicted 286.046/383.640/290.875 vs recorded 286.039/383.632/290.871. The correct target norms are 2.52/3.93/4.99/5.42 — nothing like the recorded values.

**Fix:** Change both files to `sigma_t_inv = Q @ torch.diag(1.0/sigma_t_diag) @ Q.T`, add a unit test asserting E[s_i x_i] = -1 per coordinate (Stein) and E[s_null^2] = 1/lambda_null, then re-evaluate every stored checkpoint. If checkpoints were not saved, all synthetic score-error figures must be regenerated from re-runs. Note probe_ushape.py:`analytic_score_full` already implements the rotation correctly, so the repo contains two mutually inconsistent score implementations and the wrong one produced all reported numbers.


**Referee:** Verified independently and decisively. Data frame: `data = data @ Q.T` (experiment_v2.py:131, v3/lib/data_synthetic.py:58) means x_rot = Q x_orig, so Sigma_t = Q D Q^T and Sigma_t^{-1} = Q D^{-1} Q^T. Both experiment_v2.py:215 and v3/lib/true_score.py:63 compute `Q_t.T @ diag(1/sigma_t_diag) @ Q_t` = Q^T D^{-1} Q. The code comment two lines above ("Sigma_t_inv = Q diag(1/sigma_t_diag) Q^T") contradicts the code itself. Numerical checks (2e5 samples, d_lat=40, sigma_perp=0.01, t=0.1): with the repo's matrix, Stein's identity E[s_i x_i] fails badly (-4.59,-4.93,-4.92,-4.73,-4.87 in the signal dims); with Q D^{-1} Q^T it is -1.000 in every one of the 40 coordinates and max |off-diagonal| of E[s x^T]+I is 0.022. E||s||^2/d: buggy per-signal-dim 21.78 / per-null-dim 4.93 vs correct 0.718 / 5.514 = 1/lambda_null exactly. I reproduced the recorded step-1 score errors from the buggy field to 4 significant figures using the exact generator, seeds and test set of the runs: MLP sigma_perp=0.01 d=8/15/40/200 predicted 4.9142/8.5349/7.0959/5.8780 vs recorded 4.9165/8.5370/7.0957/5.8764 (correct-field norms 2.5404/3.9278/4.9846/5.4175 - nothing like the recorded values). RFNN d=8/20/40 predicted 286.046/383.640/290.875 vs recorded 286.039/383.632/290.871. At d_lat=d_int=5 the mismatch is exactly 0.0000, confirming the bug is inert there. probe_ushape.py:analytic_score_full does implement the rotation correctly (`x_orig = x_t @ Q`, `return score_orig @ Q.T`), so the repo does contain two mutually inconsistent score implementations. This affects every `score_error` in sigma_noise_0.01/, sigma_noise_0.5/, multiseed_runs/ - including the main-text figure synthetic_mlp_score_error_per_dim.pdf.


**Referee correction:** Two refinements to the evidence, neither weakening the finding. (1) My covariance check gives ||C - QDQ^T||_F = 3.87 vs ||C - Q^T D Q||_F = 6.18 (||C||_F = 5.95), not 2.17 vs 8.76 - same direction, different magnitudes; the Stein test is the decisive check, not the covariance norm. (2) The severity is not uniform across the paper. At sigma_perp=0.01 the mismatch ||s_ok-s_bug||^2/d accounts for essentially the whole reported error (e.g. d=40: mismatch 2.412 vs reported final 2.452), and likewise for the RFNN (d=40: mismatch 249.1 vs reported final 252.6, correct target norm only 44.7). At sigma_perp=0.5 - the main-text 5M sweep - the mismatch is only 0.36-0.79 out of a reported ~2.5, i.e. roughly 15-30% of the number, so those figures are wrong but not wholly artifactual.


---

## [BLOCKING] The MLP n-shape (peak at d_lat=15, 1/d_lat recovery, A=112) is 100% an artifact of the score-error bug; the true score error is flat and monotone

`n-shape-is-entirely-the-metric-bug` · verdict **CONFIRMED** · kind error · effort days  

**Where:** /Users/ryan/Desktop/latent_space_diffusion_analysis/_next_steps/n_shape_heuristic_derivation.md (entire document); ICLR_2026/sec-appendix.tex:271-274, 303-305

**Claim:** Because the network is trained on the correct DSM objective but scored against the buggy field, the asymptotic value of the reported `score_error` is ||s*_true - s*_bug||^2/d, a pure function of Q and d_lat with no learning in it. That quantity, computed independently, matches the observed final score errors at every d_lat. The 'anti-learning zone', the 'peak at d_lat = 15', the 'A/d_lat recovery tail with A = 112', the 'gradient competition crossover d_lat* = 14.1', and the sigma_perp = 0.5 discrepancies are all descriptions of a random-rotation mismatch, not of learning dynamics.

**Evidence:** sigma_perp=0.01, t=0.1, seed 42, ||s_true - s_bug||^2/d vs observed final MLP score_error: d=8 3.358/3.366; d=10 4.889/4.951; d=15 5.359/5.325; d=20 3.417/3.423; d=30 2.891/2.907; d=40 2.412/2.452; d=50 1.849/1.890; d=100 1.008/1.053; d=150 0.691/0.745; d=200 0.520/0.593. Residual (true score error) = +0.008, +0.062, -0.034, +0.005, +0.016, +0.040, +0.040, +0.045, +0.054, +0.073 — flat at ~0.05/dim and weakly monotone increasing, with no peak and no recovery. The 'independently derived' quantities in the doc are just re-parameterisations of this curve: A = d_lat x eps_null/dim is constant at 112 precisely because ||s_true-s_bug||^2 (total, not per dim) saturates at large d_lat.

**Fix:** Retract n_shape_heuristic_derivation.md as a physical explanation; the phenomenon it explains does not exist. Delete the n-shape paragraphs at sec-appendix.tex:271-274 and 303-305 and the corresponding figures until the metric is fixed and the sweeps re-evaluated. If a genuine non-monotonicity survives the fix, re-derive from scratch.


**Referee:** Reproduced exactly. Computing ||s_ok - s_bug||^2/d_lat with the runs' own generator/seed/test set and comparing to the recorded final score_error (sigma_noise_0.01/exp2_mlp, T=300k, seed 42): d=5 0.0000/0.1473; d=8 3.3582/3.3662; d=10 4.8885/4.9507; d=15 5.3594/5.3250; d=20 3.4174/3.4228; d=30 2.8909/2.9071; d=40 2.4121/2.4522; d=50 1.8491/1.8895; d=100 1.0080/1.0531; d=150 0.6909/0.7451; d=200 0.5199/0.5927. Residuals +0.008,+0.062,-0.034,+0.005,+0.016,+0.040,+0.040,+0.045,+0.054,+0.073 - matching the reviewer's table to three decimals. The peak at d=15, the 'anti-learning zone' (ratio > 1 at d=8-15), the recovery tail and the A=112 constant are all properties of a fixed random-rotation mismatch with no learning in it. I also verified why A is constant: the TOTAL mismatch saturates (26.9 at d=8, 80.4 at d=15, 96.5 at d=40, 104.0 at d=200), and A = d_lat x eps_null/dim reproduces as total mismatch + total residual (d=200: 104.0 + 14.6 = 118.6 vs the doc's 120.8). Everything in n_shape_heuristic_derivation.md is a re-parameterisation of that curve.


**Referee correction:** One scoping mitigation the reviewer should state: the n-shape prose is at sec-appendix.tex:270-275 and 302-305, but main.tex \input{}s sec-appendix-integrated.tex, not sec-appendix.tex - so the n-shape claim is not currently compiled into the draft (sec-appendix.tex also contains dangling refs such as fig:mlp-train-test that nothing else defines). The retraction is still required for the planning doc and for _stash.tex:85-93, and the file must not be re-included as-is. Also, the residual is not strictly the true score error (cross terms can be either sign, as the -0.034 at d=15 shows); the correct statement is that the reported metric is dominated by the rotation mismatch and the true residual is small and roughly flat at ~0.05/dim.


---

## [BLOCKING] The main-text and abstract claim that feature learning finds 'sparser, signal-aligned directions' rests entirely on the artifact and is tested nowhere in the draft

`sparse-signal-aligned-claim-unsupported` · verdict **CONFIRMED** · kind unsupported-claim · effort days  

**Where:** ICLR_2026/sec-mlp.tex:41; ICLR_2026/camera_ready.tex:99 ('trainable score networks partially recover ... consistent with the first layer learning sparse, signal-aligned features'); ICLR_2026/_stash.tex:85-93

**Claim:** The only observable the claim is derived from is the 'partial recovery' of score error at large d_lat, which finding `n-shape-is-entirely-the-metric-bug` shows is the metric artifact. No sparsity, effective-rank, or weight-projection measurement appears anywhere in ICLR_2026/*.tex, and no probe output is checked into the repository. The word 'sparse' occurs exactly once in the draft (camera_ready.tex:99) and 'signal-aligned' once (sec-mlp.tex:41); both are conclusions with no supporting measurement.

**Evidence:** grep over ICLR_2026/*.tex for `sparse|signal-aligned|sig_mass|effective rank|activation density` returns only those two prose sentences. NEXT_STEPS.md sections 3a-3c list the three probes that would test it (activation density, activation effective rank, W1 projection onto Q_sig, L1/top-k ablation); none has been run — `find . -iname 'probe_results*'` returns nothing, and code/probe_ushape.py has never had its output committed.

**Fix:** Either (a) delete the sentence from sec-mlp.tex and the clause from the abstract, or (b) run the minimal three-probe set with proper nulls: (i) enrichment ratio ||Q_sig^T W1||_F^2/||W1||_F^2 divided by its random-init expectation d_int/d_lat (the raw ratio falls with d_lat at init, so the unnormalised probe is uninformative); (ii) participation ratio of the singular spectrum of W1 restricted to the x-input rows, versus a frozen-W1 control at the same d_lat; (iii) L1 / top-k ablation showing the effect strengthens. All must be run against the fixed score metric.


**Referee:** Verified. `grep -rn 'sparse|signal-aligned|sig_mass|effective rank|activation density' ICLR_2026/*.tex` returns exactly four hits: sec-mlp.tex:41 ('feature learning appears to find sparser, signal-aligned directions'), camera_ready.tex:99 ('consistent with the first layer learning sparse, signal-aligned features'), _stash.tex:92 ('Our leading hypothesis is sparse, SAE-like first-layer features'), and sec-appendix-integrated.tex:965 which merely lists 'effective rank, residual-mass proxies' as diagnostics. _stash.tex:85-93 makes the derivation chain explicit: the sparse/SAE hypothesis is introduced immediately after and solely to explain the 'MLP U-shape at low ambient noise ... partially recovers at large dlat' - the artifact established above. No sparsity, effective-rank, or weight-projection measurement appears in the draft; `find . -iname '*probe*'` returns only code/probe_ushape.py and its .pyc, with no committed output. Worse for the claim: in the main-text 5M sigma_perp=0.5 sweep there is no recovery at all to explain - final score error is flat (2.60, 2.85, 2.98, 2.92, 2.83, 2.71, 2.64, 2.59, 2.53, 2.50 across d=5..40), so sec-mlp.tex:41 asserts an interpretation of a trend its own figure does not show.


**Referee correction:** No correction needed; if anything the finding understates the problem. The main-text figure the sentence sits next to (sigma_perp=0.5, 5M steps, h=256) shows a flat, not recovering, score-error curve, so the 'partial recovery' the sparse-features hypothesis was invented to explain is absent from the compiled draft's own data as well as being an artifact in the appendix sweep it was originally read from.


---

## [BLOCKING] In the main-text 5M-step MLP sweep the trained score network is worse than predicting zero at every d_lat; the 'quality is well-behaved' control is inverted

`mlp-never-beats-the-trivial-predictor` · verdict **CONFIRMED** · kind error · effort days  

**Where:** ICLR_2026/sec-mlp.tex:24-36 (figure caption: 'rule out the simplest failure mode in which high-$\dlat$ models memorize less only because they collapse or fail to optimize'); data in multiseed_runs/exp2_mlp_dlat_sn05_5m_bonnaire_hidden256/*/metrics.jsonl

**Claim:** The dimension-normalised score error is nearly flat in d_lat (2.50-2.98) not because quality is preserved but because the model converges to the trivial predictor. Dividing by the correct target norm E||s*||^2/d (the score error of s_theta == 0) shows the ratio is greater than 1 at every d_lat and falls monotonically toward 1 as d_lat grows. The panel therefore does not rule out the failure mode it is cited to rule out; it exhibits it.

**Evidence:** Five-seed means, final/init score_error (init is exactly the zero-predictor baseline since s_theta(0) contributes negligibly): d=5 3.665, d=8 1.645, d=10 1.412, d=12 1.284, d=15 1.161, d=20 1.073, d=25 1.019, d=30 0.995, d=35 0.973, d=40 0.954. Using the bug-free target norm (5*0.730 + (d-5)*2.591)/d instead: obs/trivial = 3.56, 1.99, 1.79, 1.61, 1.44, 1.27, 1.19, 1.13, 1.09, 1.06 — never below 1. Simultaneously train_loss/test_loss at d=40 seed 42 is 0.195/0.996, so the network is fitting the 500 training points and predicting essentially nothing useful off-sample.

**Fix:** Report the relative score error eps/E||s*||^2 (fraction of score energy explained) rather than eps/d_lat; that is the quantity the mode-absorption theory actually predicts. Then either find a training configuration where the MLP demonstrably learns the population score (longer training, larger width, SGD, lower-variance objective) or restate the synthetic-MLP section as a memorization-only result and drop the quality-control claim.


**Referee:** Five-seed means reproduce the reviewer's table exactly: final/init score_error = 3.665, 1.645, 1.412, 1.284, 1.161, 1.073, 1.019, 0.995, 0.973, 0.954 at d=5..40; train/test loss at d=40 is 0.2015/0.9987 (seed-42 0.195/0.996). The claim survives independently of the rotation bug at d_lat=5, where the bug is provably inert (mismatch exactly 0.0000): there the trained model's true score error is 2.5223 against a zero-predictor baseline of 0.7241 - 3.5x worse than predicting nothing. The trajectory shows this is a training effect, not an init effect: at d=5 score_error runs 0.723 (step 1) -> 0.223 (50k) -> 1.049 (150k) -> 2.522 (5M); at d=40, 2.63 -> 0.525 (50k) -> 2.513 (5M). So the figure cited as the synthetic analogue of FID plots a quantity that degrades ~10x over training and ends at or below the trivial baseline at every d_lat. The caption's 'rule out the simplest failure mode ... collapse or fail to optimize' is not supported by this panel.


**Referee correction:** Three corrections. (1) The model does not 'converge to the trivial predictor' - at d_lat=5 it converges to something 3.5x WORSE than zero, and the flatness in d_lat is coincidence, not collapse to a baseline. (2) The specific obs/trivial ratios (3.56, 1.99, ... 1.06) divide a buggy-field numerator by a correct-field denominator, so they are not apples-to-apples; the clean anchor is d_lat=5. For d_lat>5 the defensible statement is a bound: at d=40 the true relative error lies in [0.41, 2.0] of the trivial baseline, and since a well-fit model would have reported ~0.36 (the mismatch) rather than 2.51, the true error is of order the trivial baseline. (3) Optimization is not failing - train_loss falls monotonically with d_lat (0.526 -> 0.20) and the model DOES beat trivial early (0.223/0.724 = 0.31 at 50k). The correct diagnosis is overfitting at the paper's 5M budget, which is arguably consistent with the memorization narrative; the defect is that the panel is presented as a quality control that it does not perform.


---

## [BLOCKING] The generalization gap is flat in d_lat while the nearest-neighbor memorization fraction falls 250x — the buffer effect is visible in only one of the two memorization proxies

`memorization-proxies-disagree` · verdict **CONFIRMED** · kind inconsistency · effort days  

**Where:** multiseed_runs/exp2_mlp_dlat_sn05_5m_bonnaire_hidden256/*/metrics.jsonl (final `gen_gap` vs `memorization_fraction`); claim in ICLR_2026/sec-mlp.tex:27-30 and sec-discussion.tex

**Claim:** If the excess null bulk delays memorization, the train/test score-matching gap should shrink with d_lat. It does not: it is constant at 0.80-0.84 across the entire main-text sweep, while the Somepalli nearest-neighbor memorization fraction collapses from 0.300 to 0.0012. The declining metric is the one whose known failure mode — nearest-neighbor distance concentration in higher ambient dimension — is exactly co-varying with the independent variable.

**Evidence:** Five-seed final means: d=5 gen_gap 0.818 / mem 0.3001; d=10 0.834/0.1709; d=20 0.823/0.0365; d=30 0.820/0.0076; d=40 0.797/0.0012. Also mean_nn_ratio (seed 42) rises from 0.524 at d=5 to 0.868 at d=40, i.e. the ratio statistic drifts toward its no-memorization value simply because distances concentrate.

**Fix:** Add a dimension-controlled memorization measure: e.g. the NN-ratio computed after projecting generated and training samples onto the ground-truth d_int signal subspace (or onto the top-d_int PCA of the training set), and/or report the train/test score-matching gap as the primary memorization curve. If the effect survives only in the raw NN ratio, the buffer claim must be scoped to that metric explicitly.


**Referee:** The data reproduce exactly. Five-seed finals: gen_gap 0.818 / 0.832 / 0.834 / 0.839 / 0.826 / 0.823 / 0.822 / 0.820 / 0.809 / 0.797 at d=5..40 (flat, and in absolute non-per-dim terms it GROWS 8x since it is reported per dimension), while memorization_fraction collapses 0.3001 -> 0.2172 -> 0.1709 -> 0.1265 -> 0.0800 -> 0.0365 -> 0.0155 -> 0.0076 -> 0.0030 -> 0.0012, and mean_nn_ratio drifts monotonically 0.520 -> 0.868 toward its no-memorization value. The train-side loss actually falls with d_lat (0.526 -> 0.20), i.e. the DSM train/test gap says the wide models fit the 500 training points at least as hard. The nearest-neighbor ratio's known distance-concentration failure mode co-varies exactly with the independent variable, and the draft contains no dimension-controlled memorization measure (grep for 'concentrat' in ICLR_2026/*.tex returns nothing). This is a material referee objection to the paper's central empirical claim.


**Referee correction:** The WHERE is wrong and should be fixed before the objection is raised: sec-mlp.tex:27-30 is inside the figure caption and says nothing about the generalization gap, and sec-discussion.tex never mentions it. The gen-gap figures (mlp_gen_gap_sn05.pdf etc.) are referenced only from sec-appendix.tex, which main.tex does not \input. So the paper does not currently claim the gap shrinks - the correct framing is that the paper reports only the proxy that moves, and the proxy that does not move (and is confound-free) is absent from the compiled draft.


---

## [BLOCKING] Main-text Section 2 does not exist: specify its contents, order, and length for a 5-page workshop paper

`sec2-skeleton` · verdict **CONFIRMED** · kind missing-theory · effort days  

**Where:** /Users/ryan/Desktop/latent_space_diffusion_analysis/ICLR_2026/sec-theory.tex (whole file, 9 lines); README.md "Remaining manuscript notes" item 1; main.tex line 124 \input{sec-theory}

**Claim:** The paper is titled around a mechanism and the abstract asserts it ("gradient flow learns modes in decreasing-eigenvalue order, this bulk acts as a buffer"), but Section 2 is a magenta \todo and the ICLR_2026 appendix (sec-appendix-integrated.tex, 987 lines) contains zero theorem/lemma/definition environments -- the theorem environments are declared in main.tex lines 47-55 and never used. The theory the paper needs already exists in draft form in two orphaned files that are not \input anywhere.

**Evidence:** sec-theory.tex: "\todo{TODO: formalize the eigenmode learning story here...}". `grep -n 'begin{theorem}' ICLR_2026/*.tex` returns nothing outside main.tex's \newtheorem declarations. sec-appendix-fourbulk.tex and sec-rfnn-bounds.tex contain Lemma 1, Lemma 2, Theorem (four-bulk), Eq (buffer-bound), but main.tex's \input list is: abstract, sec-intro, sec-theory, sec-design, sec-rfnn, sec-mlp, sec-real-data, sec-spectral-predictor, sec-discussion, sec-appendix-integrated -- neither orphan is included.

**Fix:** Write sec-theory.tex at ~1 column-page (60-75 lines of ICML two-column) in this order: (1) Setup, 5 lines: RFNN, gradient flow Eq (rfnn-gradflow), U, proportional limit, the three assumptions (p>d_lat+n, s/sqrt(d_lat)<~2 from the saturation appendix, fixed t). (2) Definition 1 (tau_gen) and Definition 2 (tau_mem) in spectral terms -- see finding `tau-definitions`. (3) Proposition 1: four-bulk spectrum with counts and Gaussian-equivalent edges (see `gaussian-equivalent-coefficients` and `mp-fixed-point`), stated as a display table, proof deferred. (4) Corollary 1: the buffer law, stated as a *ratio of edges plus a cumulative-absorption threshold*, not a mode count (see `buffer-bound-not-derived`). (5) A 4-line "scope" paragraph naming the three things the theory does not cover (trainable features, curved manifolds, continuous spectra) so the referee does not have to find them. Move the mechanism prose currently in sec-rfnn.tex ("Mechanism: a noise-dimension buffer", "What the gen-gap measures", "Decomposing the four-bulk structure", ~35 lines) *into* Section 2 and cut it from sec-rfnn.tex, which then becomes pure experiment reporting -- this pays for most of the new page. Port sec-appendix-fourbulk.tex into sec-appendix-integrated.tex as a new Appendix section (it is written and labelled app:proof-sketch already). For a full ICLR submission, expand Section 2 to 2.5-3 pages by promoting the Stieltjes fixed point, the BBP separation condition, the mode->duplicate bridge lemma, and the continuous-spectrum corollary from appendix to main text.


**Referee:** Verified directly. /Users/ryan/Desktop/latent_space_diffusion_analysis/ICLR_2026/sec-theory.tex is 9 lines, of which lines 4-8 are a single magenta \todo. main.tex line 121 \input{sec-theory}; the \input list (lines 117-127, 137) is exactly abstract, sec-intro, sec-theory, sec-design, sec-rfnn, sec-mlp, sec-real-data, sec-spectral-predictor, sec-discussion, sec-appendix-integrated. Neither /sec-appendix-fourbulk.tex nor /sec-rfnn-bounds.tex is input anywhere. Theorem environments are declared in main.tex lines 46-55 and the compiled document uses none. README.md 'Remaining manuscript notes' item 1 confirms this is deliberate-but-unfinished. So the paper's title mechanism has no formal statement anywhere in the compiled PDF.


**Referee correction:** One piece of the evidence is wrong and it makes the fix cheaper than the finding claims. `grep -n 'begin{theorem}' ICLR_2026/*.tex` does NOT return nothing: ICLR_2026/sec-appendix.tex contains \begin{lemma} at 690 (lem:Ulin), \begin{lemma} at 712 (lem:Udiag), \begin{theorem} at 786 (thm:fourbulk), plus the buffer bound Eq (eq:buffer-bound) at ~578 and prose subsections 'Main-text theoretical statement' and 'Derivation sketch'. That file is simply not \input by main.tex, which pulls sec-appendix-integrated.tex instead. So the theory appendix already exists *inside the current draft folder* in fully integrated form; the porting job is to reconcile sec-appendix.tex with sec-appendix-integrated.tex, not to import the two root-level orphans. Everything else in the finding stands.


---

## [BLOCKING] No formal definitions of tau_gen and tau_mem; the ones in use are three mutually inconsistent empirical proxies

`tau-definitions` · verdict **CONFIRMED** · kind missing-theory · effort days  

**Where:** ICLR_2026/sec-theory.tex (absent); ICLR_2026/sec-appendix.tex lines 417-423; "ICML (1)"/sec-appendix.tex lines 25-35, 79-108, 410-424; ICLR_2026/sec-mlp.tex (Somepalli 1/3 threshold)

**Claim:** Three different tau_mem definitions are in play across the project -- Somepalli NN-ratio<1/3 fraction crossing 1% (synthetic MLP), gen-gap (test-train) crossing 0.02 (RFNN and the July real-data runs), and 'iso-memorization hitting time of a fitted sigmoid' (spectral predictor) -- and none of them is defined as a functional of the spectrum. The theory therefore cannot be falsified by any of the measurements, because there is no stated map from {lambda_i} to any of the three.

**Evidence:** ICLR_2026/sec-appendix.tex:417 "\textbf{$\taumem$}: first step at which the gen-gap proxy~($\text{test loss} - \text{train loss} > 0.02$) is crossed"; "ICML (1)"/sec-appendix.tex:30 "For the RFNN at fixed $t$, generation is ill-defined, so we use train-test divergence (gen-gap $>0.02$) as the $\taumem$ proxy"; sec-spectral-predictor.tex "$\hat\tau_q(d)=\inf\{s:\widehat m_d(s)\ge q\}$" with $(\kappa,a,b)$ fit. The only theoretical statement, Eq (buffer-bound) in sec-rfnn-bounds.tex, uses tau_gen and tau_mem as undefined symbols.

**Fix:** Write two definitions that are functions of the spectrum and of the target's mode decomposition, and prove each matches one measured proxy. Let pi_i = ||P_i A*||_F^2 be the target mass in mode i and E(T) = sum_i pi_i e^{-2 lambda_i T} the residual score error (this is exact for the RFNN, from Eq (rfnn-mode-decay)). Definition 1: tau_gen(eps) = inf{T : E_pop(T) <= (1+eps) E_pop(inf)}, where E_pop restricts to modes whose eigenvector has O(1) overlap with the population score; show this equals (1/(2 lambda_min^signal)) log(1/eps) + O(1) and that it is what 'test loss within 5% of its per-run minimum' measures. Definition 2: tau_mem(q) = inf{T : G(T) >= g_q} with G(T) = E_test(T) - E_train(T) the spectral gen-gap, G(T) = sum_{i in sample} pi_i (1 - e^{-lambda_i T})^2 * (1 - overlap_i) -- this is a closed form, it makes the 0.02 threshold a statement about sample-mode mass, and it is the same functional as the spectral predictor's P_d(s), which uses exactly (1-e^{-kappa lambda_i s})^2. Then prove a bridge lemma tying g_q to the Somepalli 1/3-at-1% threshold (see `mode-to-duplicate-bridge`). Deliverable: two \begin{definition} blocks in Section 2 plus one appendix proposition giving G(T) in closed form for the four-bulk spectrum.


**Referee:** Quotes verified: ICLR_2026/sec-appendix.tex:26-32 ('$\taugen$ is the first step at which test loss reaches within 5% of its per-run minimum. $\taumem$ is the first step at which the somepalli2023diffusion memorization fraction crosses 1%... For the RFNN at fixed $t$, generation is ill-defined, so we use train-test divergence (gen-gap > 0.02) as the $\taumem$ proxy'); sec-appendix.tex:417-423 (gen-gap>0.02 for real data, tau_gen from FID plateau at 5%); sec-mlp.tex:8 (NN-ratio < 1/3); sec-spectral-predictor.tex:17 ($\hat\tau_q(d)=\inf\{s:\widehat m_d(s)\ge q\}$ with fitted $(\kappa,a,b)$). Confirmed that none is a functional of $\{\lambda_i\}$, and that the only theoretical statement (eq:buffer-bound in sec-rfnn-bounds.tex line 58 and sec-appendix.tex ~578) uses $\taugen,\taumem$ as undefined symbols. So no measurement can falsify the bound.


**Referee correction:** Two adjustments. (a) The situation in the *compiled* draft is worse, not milder: the definitions the finding quotes all live in ICLR_2026/sec-appendix.tex, which main.tex does not \input. In the document that actually compiles, $\taugen$ and $\taumem$ appear only as undefined prose symbols in sec-rfnn.tex lines 65 and 69, and sec-appendix-integrated.tex mentions $\tau_{\rm mem}$ exactly once (line 382) without defining it. (b) 'Three mutually inconsistent proxies' slightly overstates: sec-appendix.tex:103-108 explicitly justifies the substitution and asserts the gen-gap proxy 'monotonically tracks the canonical $\taumem$ on the MLP and calibrates the two-track comparison'. That calibration claim is asserted rather than shown anywhere in the corpus, so the finding's conclusion survives, but the paper is not simply silent about the mismatch.


---

## [BLOCKING] The buffer bound does not follow from the paper's own dynamics: the mode ODE is decoupled, so extra modes cannot delay other modes

`buffer-bound-not-derived` · verdict **CONFIRMED** · kind missing-theory · effort days  

**Where:** /Users/ryan/Desktop/latent_space_diffusion_analysis/sec-rfnn-bounds.tex, Eq (buffer-bound); ICLR_2026/sec-rfnn.tex Eq (rfnn-mode-decay) and paragraph "Mechanism: a noise-dimension buffer"

**Claim:** Eq (rfnn-mode-decay) is a *parallel*, fully decoupled linear ODE: a_i(T) - a_i* = (a_i(0)-a_i*) e^{-lambda_i T}. Mode i's trajectory does not depend on how many other modes exist. So there is no 'traversal' of the noise-dim bulk and the inequality tau_mem - tau_gen >= (d_lat - d_int) / lambda_min^noise-dim cannot be derived from it: under the stated per-mode picture, tau_mem = 1/lambda_max^sample, which is completely independent of the count d_lat - d_int. The paper's headline mechanism is currently an assertion with a sequential-learning metaphor standing in for a proof.

**Evidence:** sec-rfnn.tex: "Diagonalizing Eq.~\eqref{eq:rfnn-gradflow} in the eigenbasis of $\Umat$ gives an exponential per-mode evolution ... so eigenmode $i$ of $\Umat$ is absorbed by the readout with characteristic timescale $\tau_i = 1/\lambda_i$." sec-rfnn-bounds.tex then asserts "the time to traverse the noise-dim bulk is bounded below by its width times the slowest noise-dim timescale" -- 'width times timescale' has no support in a decoupled ODE; it would be correct only for a sequential/greedy algorithm. Empirically the delay is real but comes from elsewhere: in sigma_noise_0.5/exp2_rfnn/raw_data, the top sample-bulk eigenvalue falls 0.930 -> 0.173 (d_lat 10 -> 200), a 5.4x drop that alone predicts a 5.4x delay of the first sample mode with no counting argument.

**Fix:** Replace the counting bound with a cumulative-absorption criterion, which is both derivable and already the form the paper's own spectral predictor uses. Define the memorization share R(T) = [sum_{i in sample} pi_i (1-e^{-lambda_i T})^2] / [sum_{all i} pi_i (1-e^{-lambda_i T})^2] and tau_mem = inf{T : R(T) >= r}. Then the count *does* enter, through the denominator: adding d_lat - d_int noise-dim modes with lambda_nd >> lambda_sample adds population mass that is absorbed first, so R(T) crosses r later. Prove the resulting law, which will have the shape tau_mem ~ (1/lambda_sample) log(1 + c (d_lat-d_int) lambda_nd/(r * something)) -- i.e. logarithmic in the buffer width from the counting channel, plus the (larger) direct channel through lambda_max^sample(d_lat). State both channels separately and say which dominates. This is one page of algebra and it is the single most important missing derivation in the paper.


**Referee:** This is the most serious finding and it survives everything I threw at it. sec-rfnn.tex Eq (eq:rfnn-mode-decay) lines 30-34 is $a_i(T)-a_i^\star=(a_i(0)-a_i^\star)e^{-\lambda_i T}$, fully decoupled: mode $i$'s trajectory contains no dependence on the number of other modes. sec-rfnn-bounds.tex lines 54-63 then asserts 'the time to traverse the noise-dim bulk is bounded below by its width times the slowest noise-dim timescale' and multiplies by $(\dlat-\dint)$. Nothing in a parallel linear ODE licenses that product; it is the arithmetic of a sequential/greedy scheduler. The project's own theory_plan.md confirms the bound was never derived: the plan's step 4 reads 'the gap is $\tau_{mem}-\tau_{gen}$ = time to traverse noise-dim bulk = $\Omega((d_{lat}-d_{int})\cdot\tau_{noise-dim})$ via a sum-of-exponentials lower bound' -- a to-do, with the 'sum-of-exponentials lower bound' never written.


**Referee correction:** The finding can be made decisively quantitative from the repo's own data, which I recommend adding. In sigma_noise_0.5/exp2_rfnn/raw_data the top sample-bulk eigenvalue falls 0.9300 (d_lat=10) -> 0.1733 (d_lat=200), ratio 5.37x (finding's numbers reproduce exactly). The measured gen-gap>0.02 crossing time over the same range goes 15k -> 80k steps, ratio 5.33x. The counting bound predicts a delay ratio of $(200-5)/(10-5)=39$x. So the counting channel is not merely underived, it is empirically absent: the direct channel through $\lambda_{\max}^{\rm sample}(\dlat)$ accounts for the observed delay to within 1%. The proposed cumulative-absorption replacement is the right shape, but the paper should state up front that the eigenvalue channel dominates and the count channel is (at these parameters) unmeasurable.


---

## [BLOCKING] No bridge from 'mode absorbed' to 'generated sample is a near-duplicate'; the cleanest route is the noised-empirical-measure collapse condition

`mode-to-duplicate-bridge` · verdict **CONFIRMED** · kind missing-theory · effort weeks  

**Where:** ICLR_2026/sec-theory.tex (absent); sec-intro.tex paragraph citing scarvelis2023closedform and gu2023memorization; sec-mlp.tex (Somepalli NN-ratio < 1/3)

**Claim:** Every empirical memorization number in the paper is a pixel- or latent-space nearest-neighbour ratio on *generated samples*, but the theory only ever talks about eigenmodes of U being absorbed by a linear readout at one fixed t=0.01. Nothing connects the two. The intro already cites the two papers that give the closed form needed (Gu et al., Scarvelis et al.) but never uses them.

**Evidence:** sec-intro.tex: "the exact score of the noised empirical distribution has a closed form: it is built from kernels centered on the training examples, and following this score back to low noise can return the training samples themselves~\citep{scarvelis2023closedform}". sec-mlp.tex: "We count a generated sample as memorized when its nearest-neighbor ratio is below $1/3$". sec-rfnn.tex trains "at a single fixed diffusion time $t = 0.01$" and admits elsewhere ("ICML (1)"/sec-appendix.tex:29) "For the RFNN at fixed $t$, generation is ill-defined".

**Fix:** Write a bridge lemma using the empirical-score collapse condition. The exact score of the noised empirical measure is s_emp(x,t) = (sum_mu w_mu(x)(e^{-t}x^mu - x))/Delta_t with w_mu = softmax(-||x - e^{-t}x^mu||^2 / 2 Delta_t). Near-duplication at the end of the reverse ODE is exactly the condition that w concentrates on one mu, which holds when Delta_t < d_NN^2 / (2 log n), d_NN the training-set nearest-neighbour distance (this is the Biroli et al. condensation threshold, already cited as biroli2024dynamical). A partially trained score realizes an *effective* noise level Delta_eff(T) = Delta_t + E(T), where E(T) = sum_i pi_i e^{-2 lambda_i T} is the residual spectral error from `tau-definitions`. That gives the bridge in one line: tau_mem = inf{T : E(T) <= d_NN^2/(2 log n) - Delta_t}. Two payoffs: (a) the Somepalli 1/3 threshold becomes meaningful -- ratio < 1/3 is a statement about the generated point sitting within 1/3 of the NN spacing, so the relevant d_NN is the 1st-percentile of the training NN-distance distribution when the measured level is 'fraction crossing 1%'; (b) it predicts tau_mem depends on n and d_int through d_NN ~ n^{-1/d_int}, which is testable. Alternative (weaker) route if this stalls: a kernel-density argument showing the spectrally filtered score is the exact score of the empirical measure smoothed by a bandwidth h(T)^2 = E(T), and memorization iff h(T) <= d_NN/3.


**Referee:** Quotes verified verbatim: sec-intro.tex:11 cites scarvelis2023closedform and gu2023memorization for the closed-form empirical score and never uses either afterwards (grep shows no further occurrence outside the bib); sec-mlp.tex:8 and sec-appendix-integrated.tex:53-54, 803 define memorization purely as a pixel-space NN-ratio < 1/3 on generated/decoded samples; sec-rfnn.tex:13 trains 'at a single fixed diffusion time $t=0.01$' and ICLR_2026/sec-appendix.tex:30 concedes 'For the RFNN at fixed $t$, generation is ill-defined'. So the theory's object (eigenmode absorption in a frozen readout at one $t$) and every reported memorization number are connected by nothing but the word 'memorization'. That is a real hole in claim chain link (3)->(4).


**Referee correction:** Downgrade blocking -> major, and flag that the proposed fix is itself a heuristic. (a) No paper in this literature closes this bridge, including the paper's own baseline (bonnaire2025), so a workshop referee will not treat it as blocking; it is a scope statement the paper owes, not a proof it owes. (b) The proposed one-line bridge $\tau_{\rm mem}=\inf\{T: E(T)\le d_{NN}^2/(2\log n)-\Delta_t\}$ silently assumes the residual spectral error $E(T)=\sum_i\pi_i e^{-2\lambda_i T}$ acts as an *isotropic* added noise level $\Delta_{\rm eff}$. It does not: $E(T)$ is concentrated in the slowest (sample and null) modes, which is exactly the anisotropy the rest of the paper is about, so equating it with a scalar bandwidth begs the question. The weaker kernel-smoothing alternative the finding offers as a fallback has the same defect. Recommend stating the bridge as an assumption with the anisotropy caveat named, rather than presenting it as a derivation.


---

## [BLOCKING] Bulk edges must use Gaussian-equivalent coefficients a1(q), a*(q) at preactivation variance q=tr(M_t)/d_lat; with the fixed Hermite mu_1 the theorem misses a 3.4x trend and wrongly implies tau_gen is d_lat-independent

`gaussian-equivalent-coefficients` · verdict **CONFIRMED** · kind missing-theory · effort days  

**Where:** sec-appendix-fourbulk.tex, Theorem thm:fourbulk edge column and Lemma lem:Ulin; sec-rfnn.tex "Mechanism" paragraph ("$\taugen$, $\dlat$-independent at fixed $\sigsig$, $k$, cluster scale"); data: sigma_noise_0.5/exp2_rfnn/raw_data/*/eigenvalues_pre.npy

**Claim:** The four-bulk theorem uses the fixed Hermite coefficient mu_1^2 = 0.367 of tanh against N(0,1). The correct coefficient is the Gaussian-equivalent slope a1(q) = E[tanh'(sqrt(q) z)] at the *actual* preactivation variance q = tr(M_t)/d_lat, which depends on d_lat. With mu_1 fixed, every bulk edge is d_lat-independent and the paper concludes tau_gen is d_lat-independent. The measured signal-bulk median grows 3.4x across the sweep, so tau_gen is not d_lat-independent, and the a1(q) correction predicts the whole trend to ~15%.

**Evidence:** Measured (sigma_perp=0.5, p=64 d_lat, t=0.01) median signal eigenvalue vs d_lat: 29.7 (d=5), 40.6 (10), 56.6 (20), 72.8 (40), 77.8 (60), 86.9 (80), 87.6 (100), 96.7 (150), 99.8 (200). Fixed-mu_1 theory predicts a constant 64.9 for all d_lat (mu_1^2 * alpha_t^2 * psi_p = 0.367*2.764*64). With a1(q)^2, q(d_lat) = [e^{-2t}(s^2 + d_int sig_sig^2 + (d_lat-d_int) sig_perp^2) + Delta_t d_lat]/d_lat, the prediction is 31.9, 49.7, 69.6, 87.4, 95.7, 100.5, 103.6, 108.2, 110.6 -- within 15-25% and reproducing the monotone rise (q falls 2.76 -> 0.33, so tanh de-saturates and a1^2 rises 0.180 -> 0.625). Noise-dim bulk is even better: measured medians 5.22, 7.03, 8.36, 9.01, 9.11, 9.32, 9.35, 9.22 vs a1(q)^2 beta_t^2 psi_p = 4.76, 6.67, 8.37, 9.17, 9.63, 9.93, 10.37, 10.60; the fixed-mu_1 theory predicts a flat 6.22.

**Fix:** Restate Lemma lem:Ulin and Theorem thm:fourbulk with the Gaussian-equivalent linear-plus-noise model (cite Hu & Lu universality of the conjugate kernel; Goldt et al.; Mei-Montanari), phi(x) ~= a1(q) Wx/sqrt(d_lat) + a_*(q) theta, a1(q) = E[f'(sqrt(q) z)], a_*(q)^2 = E[f(sqrt(q) z)^2] - a1(q)^2 q, q = tr(M_t)/d_lat. Every edge in the theorem table then carries an explicit d_lat dependence through q. Add one sentence to Section 2 retracting 'tau_gen is d_lat-independent' and replacing it with 'tau_gen decreases mildly with d_lat because the feature map de-saturates', and add the measured-vs-predicted table above to the appendix as the quantitative validation the theorem currently claims but never shows (sec-appendix-fourbulk.tex asserts the edges 'match (within Marchenko-Pastur fluctuations) the empirical bulk locations in Table~\ref{tab:bulk-sizes}' -- that comparison is not in any file).


**Referee:** I reproduced every number independently from sigma_noise_0.5/exp2_rfnn/raw_data/*/eigenvalues_pre.npy. Measured signal-bulk medians (median of top d_int=5): 29.66, 40.59, 56.58, 72.84, 77.76, 86.94, 87.64, 96.67, 99.80 at d_lat = 5,10,20,40,60,80,100,150,200 -- identical to the finding, a 3.4x monotone rise. Fixed-$\mu_1$ theory gives a flat 64.91 ($\mu_1^2=0.36688$, $\alpha_t^2=2.7644$, $\psi_p=64$). With $q(\dlat)=\mathrm{tr}(M_t)/\dlat$ (2.764 -> 0.327 across the sweep) and $a_1(q)^2=E[\tanh'(\sqrt q z)]^2$ (0.180 -> 0.625), the GE prediction is 31.8, 49.7, 69.5, 87.3, 95.6, 100.5, 103.6, 108.2, 110.6 -- the finding's numbers exactly, reproducing the monotone rise within 7-25%. Noise-dim is better still: measured 5.22, 7.03, 8.36, 9.01, 9.11, 9.32, 9.35, 9.22 vs GE 4.76, 6.66, 8.37, 9.16, 9.63, 9.93, 10.36, 10.60 (within 4% out to d_lat=60, 15% at 200), against a flat fixed-$\mu_1$ prediction of 6.22. Lemma lem:Ulin and Theorem thm:fourbulk do use the fixed $\mu_1$ (sec-appendix-fourbulk.tex lines 39, 219-222), and sec-rfnn-bounds.tex line 66 draws the '$\dlat$-independent' conclusion from it. The theorem's edge column is therefore quantitatively wrong in a way the repo's own data refute.


**Referee correction:** One inference is stronger than the evidence supports. 'The measured signal-bulk median grows 3.4x across the sweep, so $\taugen$ is not $\dlat$-independent' is a statement about the theory's own $\tau_i=1/\lambda_i$ map, not about a measurement. The paper's actual $\taugen$ proxy (first step test loss is within 5% of its per-run minimum) computed from the same metrics.jsonl gives 1, 1, 5k, 5k, 5k, 5k, 10k, 10k, 10k steps -- flat-to-mildly-increasing, i.e. it does not fall 3.4x. So the correct retraction sentence is: 'the signal-mode timescale $1/\lambda^{\rm signal}$ falls by ~3.4x across the sweep because the feature map de-saturates; the measured $\taugen$ proxy does not track this, which is itself a calibration problem for the proxy.' Restate the theorem with $a_1(q), a_*(q)$ as proposed, and add the measured-vs-predicted table, which genuinely does not exist anywhere (see mp-fixed-point note on tab:bulk-sizes).


---

## [BLOCKING] State the anisotropic spectrum as a generalized Marchenko-Pastur problem with an explicit BBP separation condition; it also fixes the bulk counts, which the data say are (d_int, d_lat-d_int, n-d_lat, p-n), not (..., n, p-d_lat-n)

`mp-fixed-point` · verdict **REFUTED** · kind missing-theory · effort weeks  

**Where:** sec-appendix-fourbulk.tex Theorem thm:fourbulk (Count column and the paragraph "Toward a full Stieltjes derivation"); _next_steps/theory_plan.md route B; ICLR_2026/sec-rfnn.tex (index-block colouring at d_lat+1 : d_lat+n)

**Claim:** The requested Stieltjes/self-consistent equation for block-diagonal Sigma_data does not need a new replica computation. Under Gaussian equivalence, U is a *sample covariance matrix of n vectors with population covariance* Sigma_phi = a1^2 W M_t W^T/d_lat + a_*^2 I_p, so the limiting spectrum is the classical Silverstein fixed point with population measure H = spec(Sigma_phi), and the separation condition is BBP. This immediately gives (a) closed-form bulk edges, (b) the separation condition the lens asks for, and (c) the correct counts. The counts matter: the current theorem says the sample bulk has n modes and the rank-null tail p - d_lat - n, but a sample covariance of n vectors has rank n, so the counts should be d_int, d_lat - d_int, n - d_lat, p - n. The data agree with the latter.

**Evidence:** Largest log-eigenvalue gaps in sigma_noise_0.5/exp2_rfnn/raw_data (sorted spectrum, gap in decades): d_lat=100, p=6400, n=500 -> gaps after index 5 (0.534 dec), index 100 (1.255 dec), index 500 (0.226 dec); d_lat=200, p=12800 -> after index 5 (0.434), index 200 (1.088), index 500 (0.237). The third feature sits at index n=500, not at d_lat+n = 600 / 700. Separately, the MP top-edge formula lambda_max^sample = a_*(q)^2 (1+sqrt(p/n))^2 predicts 0.126, 0.122, 0.126, 0.140, 0.158 at d_lat = 40, 80, 100, 150, 200 against measured top sample eigenvalues 0.374, 0.225, 0.199, 0.169, 0.173 -- converging to a ratio of 1.10 at d_lat=200 (it is 4.7x off at d_lat=10 where the index cut at d_lat is crude). The current theorem gives lambda^sample = etabar_star/psi_p with no n and no p dependence at all, so it cannot produce this decreasing trend, which is the quantity that actually sets tau_mem.

**Fix:** Write the fixed point explicitly: for the companion Stieltjes transform m(z) of the limiting spectral measure of U with aspect ratio c = p/n, z = -1/m(z) + c * integral t dH(t)/(1 + t m(z)), where H is the spectral measure of Sigma_phi -- itself three atoms MP-broadened by W: a1^2 psi_p alpha_t^2 + a_*^2 (multiplicity d_int), a1^2 psi_p beta_t^2 + a_*^2 (multiplicity d_lat - d_int), a_*^2 (multiplicity p - d_lat). Solve for the four edges. State the separation condition as BBP: a population atom theta detaches from the a_*^2-MP bulk iff theta > a_*^2 (1 + sqrt(p/n)), i.e. the noise-dim bulk is resolvable iff a1(q)^2 psi_p beta_t^2 > a_*(q)^2 sqrt(p/n) -- this is the theory of the observed 'cliff at d_lat is sharp at sigma_perp=0.5 and degenerates to a smooth transition at sigma_perp=0.01' (sec-rfnn.tex) which currently has no explanation. Correct the count column to (d_int, d_lat - d_int, n - d_lat, p - n) and re-check the index-block colouring in the figures, which currently assumes the sample block runs d_lat+1 : d_lat+n. Also state lambda_max^sample = a_*^2 (1+sqrt(p/n))^2 as the quantity that sets tau_mem, since this is what the buffer argument actually needs.


**Referee:** The finding's central factual claim -- that the data say the counts are $(\dint, \dlat-\dint, n-\dlat, p-n)$ -- is wrong, and the paper's count column is what the data actually support. Log-log local-slope analysis of the sorted spectra shows the transition into the flat rank-null tail sits at index $\dlat+n$, not $n$: at d_lat=40 the slope breaks -4.34 -> -2.13 at index 540; at d_lat=100, -6.29 -> -1.43 at index 600; at d_lat=200, -8.74 -> -0.78 at index 700. Raw values at d_lat=200: idx 501 = 0.0323, idx 601 = 0.00849, idx 701 = 0.00291, then a slow smooth tail to 5.2e-5 at idx 12800. So the structured support is $\dlat+n$ and the rank-null tail is $p-\dlat-n$, exactly as thm:fourbulk states. The finding's premise ('$U$ is a sample covariance matrix of $n$ vectors, so rank $n$') is false: $U=\frac1n\sum_\mu E_\eta[\phi(x_t^\mu)\phi(x_t^\mu)^\top]$ is an average of $n$ *full-rank* matrices, and the paper's own $U^{\rm lin}$ (rank $\dlat$) $\oplus$ $U^{\rm diag}$ (rank $n$) decomposition is precisely what produces support $\dlat+n$. Separately, the proposed replacement edge $\lambda_{\max}^{\rm sample}=a_*^2(1+\sqrt{p/n})^2$ fails where it matters: I get 0.133, 0.130, 0.135, 0.152, 0.171 at d_lat=40,80,100,150,200 against measured 0.374, 0.225, 0.199, 0.169, 0.173 -- non-monotone where the data are monotone, 2.8x off at d_lat=40, agreeing only at d_lat=200. And '$\bar\eta_\star/\psi_p$ has no $n$ and no $p$ dependence at all' misreads the definition: $\bar\eta_\star=\sum_{k\ge3}\mu_k^2 E[(\|x\|^2/\dlat)^k]$ is a function of $q=\mathrm{tr}(M_t)/\dlat$ and does fall with $\dlat$.


**Referee correction:** Keep three salvageable pieces and drop the rest. (1) There IS an unexplained internal step at index exactly $n=500$ -- a factor ~1.7 drop with a slope break, present at every $\dlat$ (log-gap 0.121 at d=40, 0.318 at d=80, 0.226 at d=100, 0.279 at d=150, 0.237 at d=200), and it sits *inside* the sample bulk rather than at its edge. The theorem predicts no feature there and should either explain it or note it. (2) The Silverstein/BBP reformulation is still a good idea as a route to a sharper sample edge and to the separation condition for the '$\signoise=0.01$ smooth transition' remark in sec-rfnn.tex:57 -- but as an improvement, not a correction of the counts. (3) The finding is right that the theorem's claim to 'match (within Marchenko-Pastur fluctuations) the empirical bulk locations in Table~\ref{tab:bulk-sizes}' (sec-appendix-fourbulk.tex:262-264) is unsupported: tab:bulk-sizes (ICLR_2026/sec-appendix.tex:171-192) tabulates bulk *sizes*, not edge locations, and its own counts miss the prediction (at $\dlat=40$, $\signoise=0.01$: B3=70 vs predicted $\dlat-\dint=35$, B2=578 vs $n=500$, B1=1907 vs $p-\dlat-n=2020$). No edge comparison exists in any file. Do not change the index-block colouring in eigenvalues.py:bulk_indices; it is correct.


---

## [BLOCKING] Real-data theory: replace the four-bulk count with a spectral-density / effective-dimension statement, and derive the spectral predictor's weights instead of setting w_i=1

`continuous-spectrum-effective-dimension` · verdict **CONFIRMED** · kind missing-theory · effort weeks  

**Where:** ICLR_2026/sec-spectral-predictor.tex (definition of $M_t(d)$, $P_d(s)$, $w_i=1$); ICLR_2026/sec-appendix-integrated.tex lines 758-830 (app:spectral-predictor-clean, including the n-dependence figure caption)

**Claim:** The headline predictive contribution is built on a continuous VAE latent spectrum, but the theory is a four-atom counting statement about a synthetic two-block covariance. The predictor already silently makes the generalization -- it uses only M_t(d) = e^{-2t} Z^T Z/n + (1-e^{-2t}) I_d, which is the *linear* part of U with no sample bulk and no rank-null tail at all -- and then patches the missing physics with a fitted three-parameter sigmoid. Writing the continuous-spectrum theory is what would make the predictor a prediction rather than a curve fit, and it is the highest-leverage generalization available.

**Evidence:** sec-spectral-predictor.tex defines P_d(s) with "$w_i=1$ for the primary plots" and then "The predicted memorization fraction is an anchored sigmoid of this clock ... using one set of $(\kappa,a,b)$". The appendix concedes the missing sample bulk explicitly: "the curves are flat: this is not evidence that true memorization dynamics are independent of $n$ ... A real $n$-dependence for the 1\% time would require recomputing $M_{t,n}(d)$ on different $n$-sized subsets, or adding an explicit sample-count/sample-bulk term to the predictor." So the object the paper's real-data claim rests on has no theory behind two of its four bulks.

**Fix:** Three deliverables. (a) Restate the buffer result for a general spectral measure rho of M_t: define the *diffusion-resolved effective dimension* N_eff(t) = #{i : lambda_i(M_t) > (1+delta) Delta_t}, or the smooth version via the participation ratio (sum lambda_i)^2/sum lambda_i^2, and prove the buffer corollary with N_eff - d_int replacing d_lat - d_int. For a power-law spectrum lambda_i = lambda_1 i^{-a} (which is what VAE latents give) this yields N_eff(t) = (lambda_1/Delta_t)^{1/a} in closed form, so the theory predicts how the buffer scales with the VAE spectrum's exponent -- a quantity already computable from the saved spectra. (b) Derive the weights: for the exact linear-Gaussian score s*(x_t) = -M_t^{-1} x_t, the target energy in mode i is E[(s*_i)^2] = 1/lambda_i, so the principled weight is w_i = 1/lambda_i, not 1. Note the consequence, which the paper needs anyway: the score target's mass sits overwhelmingly in the *slowest* modes, which is a first-principles reason why widening d_lat degrades score error even while it delays memorization. (c) Add the missing sample-bulk term to P_d(s) using lambda_max^sample = a_*^2(1+sqrt(p/n))^2 from `mp-fixed-point`, giving the predictor an n-dependence. Test: refit with derived w_i and the sample term; if the fitted (kappa,a,b) become dataset-independent (one set shared by CelebA and CIFAR-10 rather than one per dataset), that is strong evidence the theory is right, and it costs no retraining because the spectra are already saved.


**Referee:** Quotes verified verbatim. sec-spectral-predictor.tex:17 defines $M_t(d)=e^{-2t}Z_d^\top Z_d/n+(1-e^{-2t})I_d$ and $P_d(s)=\sum_i w_i(1-e^{-\kappa\lambda_i s})^2/\sum_i w_i$ 'with $w_i=1$ for the primary plots', followed by a three-parameter anchored sigmoid with '(\kappa,a,b)' fit per dataset. That object is the linear part of $U$ only: no sample bulk, no rank-null tail, i.e. two of the four bulks the paper's own theorem is about are absent from the paper's headline predictive artifact. sec-appendix-integrated.tex:803-830 concedes it in the figure caption: 'this is not evidence that true memorization dynamics are independent of $n$... A real $n$-dependence for the 1% time would require recomputing $M_{t,n}(d)$ on different $n$-sized subsets, or adding an explicit sample-count/sample-bulk term to the predictor.' The derived weight is also correct: for $s^\star(x_t)=-M_t^{-1}x_t$, $E[(s^\star_i)^2]=\lambda_i/\lambda_i^2=1/\lambda_i$, so $w_i=1/\lambda_i$ is the principled choice and $w_i=1$ is not.


**Referee correction:** Two corrections. (a) 'Blocking' overstates: this is a generalization the paper would benefit from, not a defect in a stated claim -- sec-discussion.tex limitation (ii) already flags the predictor as 'an empirical frozen-VAE latent-spectrum rule, not yet a theorem'. Major, not blocking. (b) The finding says the predictor 'sets $w_i=1$', which is true of the primary plots but incomplete: sec-appendix-integrated.tex:788-791 documents a second weighting, the 'de-noised excess weights $w_i(d)=\max\{\lambda_i(d)-\beta_d,0\}$'. That is worth engaging because it weights in the *opposite* direction from the derived $1/\lambda_i$ -- it upweights the fast modes. So the deliverable is sharper than stated: the paper has already tried two ad hoc weightings, one of which is anti-correlated with the principled one, and deriving $w_i$ adjudicates between them. The proposed test (do fitted $(\kappa,a,b)$ become dataset-independent) is well posed and costs no retraining.


---

## [BLOCKING] "Splits almost surely into four contiguous bulks" — contiguity and spectral gaps are asserted, never proved; no separation criterion appears anywhere

`contiguity-and-gaps-never-proved` · verdict **—** · kind missing-theory · effort weeks  

**Where:** /Users/ryan/Desktop/latent_space_diffusion_analysis/sec-appendix-fourbulk.tex, Theorem thm:fourbulk lines 208-239 (identical text at ICLR_2026/sec-appendix.tex:785-808 and "ICML (1)/sec-appendix.tex":789)

**Claim:** The theorem's conclusion is a table of counts and one scalar per bulk. "Contiguous" (each bulk is an interval), "four" (exactly four, not three or five), and any gap between consecutive bulks are nowhere established. The proof only says the two ranges are asymptotically orthogonal and dimensions add — that gives a rank decomposition, not a spectral partition.

**Evidence:** Proof body in full: "Combine Lemmas 1 and 2 with the decomposition (U-split). The ranges ... are asymptotically W-orthogonal ... Together they span a subspace of dimension d_lat + n < p, leaving a rank-null tail" (lines 229-239). Nothing about intervals, supports, or gaps. Remark rem:MP (lines 190-205) concedes "the actual bulk in finite d_lat is a Marchenko-Pastur-deformed band around that edge" but supplies no support endpoints and no non-overlap condition.

**Fix:** The linear part is an anisotropic (generalized) Marchenko-Pastur matrix M_t^{1/2} W^T W M_t^{1/2}/p with population spectral distribution H = (d_int/d_lat) delta_{alpha_t^2} + (1 - d_int/d_lat) delta_{beta_t^2} and ratio 1/psi_p. The support of its limiting ESD, and the exact condition for the support to split into two intervals, are given by Silverstein-Choi (1995): with x(m) = -1/m + psi_p^{-1} \int tau/(1+tau m) dH(tau), the support edges are the images of the m at which x'(m) = 0, and two bulks separate iff x' has two sign changes on the relevant interval. State that condition as an explicit hypothesis (H1) and prove the two data bulks are disjoint intervals under it. Then bound the cross-term between U^lin and U^diag in operator norm with an explicit rate (not o_d(1)) and invoke Weyl to show the union of the three supports stays disjoint. Only then is "four contiguous, separated bulks" a conclusion.


---

## [BLOCKING] The bulk ordering the buffer argument requires is not in the theorem; it is a footnote-grade side condition, and that condition is itself quantitatively wrong

`ordering-is-a-side-condition-not-a-conclusion` · verdict **—** · kind gap · effort days  

**Where:** /Users/ryan/Desktop/latent_space_diffusion_analysis/sec-appendix-fourbulk.tex, "Bulk-gap scaling" paragraph lines 241-259; buffer corollary at sec-rfnn-bounds.tex:51-69

**Claim:** The theorem never asserts signal > noise-dim > sample > rank-null. The ordering appears only in prose as the condition beta_t^2 >~ etabar/mu_1^2, outside any theorem/lemma environment. Everything downstream (the sequential-traversal mechanism, the buffer bound, the whole paper's story) needs the ordering as a *conclusion*.

**Evidence:** Line 254-256: "so for beta_t^2 >~ etabar/mu_1^2 the noise-dim edge sits above the sample edge and the four bulks are spectrally resolvable." This is the only place ordering is discussed and it is stated as a regime, not proved. The theorem statement (lines 208-228) contains no inequality between the four edge entries. Separately, the condition is derived from the wrong sample edge (see finding sample-edge-inconsistent-three-ways): with the trace-correct sample edge etabar/n the condition becomes beta_t^2 >~ etabar/(mu_1^2 psi_n), i.e. it depends on psi_n = n/d_lat, which the stated condition omits entirely — a factor of 25 at the default d_lat=20, n=500.

**Fix:** Promote ordering to a conclusion by adding explicit hypotheses: (H2) alpha_t^2 / beta_t^2 exceeds the Silverstein-Choi separation threshold for the two-atom H (not the ad-hoc 1.5 in Remark rem:MP, whose exact value at psi_p=64 is (sqrt(psi_p)+1)^2/(sqrt(psi_p)-1)^2 = 1.653, not 1.50 as printed on line 197); (H3) mu_1^2 beta_t^2 / d_lat > C * etabar / n with an explicit constant covering the MP band width of the noise-dim bulk; (H4) etabar/n dominates the operator norm of the neglected remainder R_t. Then state the theorem as: under H1-H4, sup(bulk_{k+1}) < inf(bulk_k) for k = 1,2,3.


---

## [BLOCKING] The stated bulk edges mu_1^2 alpha_t^2/psi_p and mu_1^2 beta_t^2/psi_p have the wrong power of d_lat; the correct edge is mu_1^2 alpha_t^2/d_lat

`edge-normalization-wrong-power-of-dlat` · verdict **—** · kind error · effort days  

**Where:** /Users/ryan/Desktop/latent_space_diffusion_analysis/sec-appendix-fourbulk.tex, Lemma lem:Ulin lines 112-131 and Theorem table lines 219-222 (same in ICLR_2026/sec-appendix.tex:797-802, sec-rfnn-bounds.tex:37-38)

**Claim:** With the paper's own normalization phi(x) = tanh(Wx/sqrt(d_lat))/sqrt(p), U^lin = mu_1^2 W M_t W^T/(p d_lat) with W having N(0,1) entries. Its nonzero eigenvalues are those of mu_1^2 M_t^{1/2} W^T W M_t^{1/2}/(p d_lat) ~= mu_1^2 spec(M_t)/d_lat, since E[W^T W] = p I. The paper writes spec(M_t) * mu_1^2/psi_p = mu_1^2 spec(M_t) d_lat/p, which is off by a factor d_lat^2/p = d_lat/psi_p — a factor that *diverges* in the proportional limit the theorem is stated in.

**Evidence:** Trace test: sum of the d_lat linear eigenvalues must equal tr(U^lin) = mu_1^2 tr(M_t)/d_lat = mu_1^2 * v, where v = E||x_t||^2/d_lat. At the default d_lat=20, sigma_perp=0.5: v=0.890, mu_1^2 v = 0.327. Correct formula gives sum = 0.327; the paper's gives mu_1^2 tr(M_t)/psi_p = 0.102. Empirical test on the raw spectra (U_emp = p*U_theory, so the correct prediction for the noise-dim bulk is psi_p*mu_1^2*beta_t^2 = 6.22, d_lat-independent, versus the paper's mu_1^2*beta_t^2*d_lat which grows linearly): observed median of eigenvalues[d_int:d_lat] at sigma_perp=0.5 is 5.22, 7.03, 8.36, 9.01, 9.11, 9.32, 9.35, 9.22 for d_lat = 10,20,40,60,80,100,150,200 — flat, as the correct formula requires. The paper's formula predicts 0.97, 1.94, 3.89, 5.83, 7.77, 9.72, 14.58, 19.43 — off by 5.4x at d_lat=10 and 2.1x the other way at d_lat=200.

**Fix:** Recompute all four edge entries with the correct 1/d_lat scaling (equivalently state edges for the un-normalized kernel tanh tanh^T/n, where they are psi_p mu_1^2 alpha_t^2 etc.). Every downstream formula that carries psi_p — including the buffer bound's psi_p/(mu_1^2 beta_t^2) in Eq (buffer-bound) — must be rewritten.


---

## [BLOCKING] Eq (U-diag), the Lemma 2 proof, and the Lemma 2 statement give three mutually incompatible sample-bulk eigenvalues, and none matches the trace

`sample-edge-inconsistent-three-ways` · verdict **—** · kind error · effort days  

**Where:** /Users/ryan/Desktop/latent_space_diffusion_analysis/sec-appendix-fourbulk.tex, Eq (U-diag) lines 77-82, Lemma lem:Udiag lines 134-157

**Claim:** Eq (U-diag) writes U^diag = (1/(p n)) sum_mu eta_star(x_mu,x_mu) phi^perp phi^perp^T while also asserting E||phi^perp(x)||^2 = eta_star, so its eigenvalues are eta_star^2/(p n) — which is what the proof line 153-154 states. The Lemma's own statement says they concentrate at etabar/psi_p. The trace identity gives etabar/n. All three disagree.

**Evidence:** tr(U) = E[tanh^2] = mu_1^2 v + etabar exactly, and tr(U^lin) = mu_1^2 v, so tr(U^diag) = etabar spread over n modes, i.e. etabar/n per mode. At d_lat=20, sigma_perp=0.5, etabar=0.0183: trace-correct edge = 3.66e-5; Lemma statement (etabar/psi_p) = 2.86e-4 (8x too big; summed over n=500 it gives 0.143, 8x the entire available trace 0.0183); the proof's eta_star^2/(pn) = 5.2e-10 (5 orders of magnitude too small). The 1/p is double-counted (phi already carries 1/sqrt(p)) and eta_star is double-counted (it is already ||phi^perp||^2).

**Fix:** Rewrite Eq (U-diag) as U^diag = (1/n) sum_mu phi^perp(x_t^mu) phi^perp(x_t^mu)^T with ||phi^perp(x)||^2 -> eta_star(x,x), giving the sample edge etabar/n (equivalently p*etabar/n in un-normalized units). Then re-derive the noise-dim-to-sample gap, which becomes mu_1^2 beta_t^2 psi_n / etabar, a factor psi_n larger than the paper's claim, and re-derive the merge threshold in the Bulk-gap-scaling paragraph.


---

## [BLOCKING] The Hermite decomposition uses standard-normal coefficients mu_k for pre-activations whose variance is v = E||x_t||^2/d_lat != 1; the resulting eta_star series diverges at the paper's own small-d_lat settings

`hermite-expansion-wrong-variance` · verdict **—** · kind error · effort weeks  

**Where:** /Users/ryan/Desktop/latent_space_diffusion_analysis/sec-appendix-fourbulk.tex, Eq (kernel-decomp) lines 50-58 and the definition etabar = sum_{k>=3} mu_k^2 E[(||x||^2/d_lat)^k]

**Claim:** The identity E[phi(u)phi(w)] = sum_k mu_k^2 rho^k holds only when u,w are *standard* normal with correlation rho. Here (Wx)_a ~ N(0, ||x||^2/d_lat) with ||x||^2/d_lat = v != 1, so the correct expansion is sum_k c_k(v)^2 rho^k with c_k(v) the Hermite coefficients of tanh(sqrt(v) . ), not mu_k. The effective linear coefficient is c_1(v)^2/v, not mu_1^2, and the diagonal residual is sum_{k>=3} c_k(v)^2, not sum_{k>=3} mu_k^2 v^k.

**Evidence:** Along the paper's own d_lat sweep at sigma_perp=0.5, v runs 2.76 -> 0.33 and c_1(v)^2/v runs 0.28 -> 0.63, i.e. the quantity the paper treats as the fixed constant mu_1^2 = 0.367 varies by 2.2x. Substituting c_1(v)^2/v reproduces the observed noise-dim medians far better: predicted 4.76/6.66/8.37/9.16/9.62/9.93 vs observed 5.22/7.03/8.36/9.01/9.11/9.32 for d_lat=10/20/40/60/80/100 (5-15%), versus a constant-mu_1^2 prediction of 6.22 everywhere. Worse, the paper's series sum_{k>=3} mu_k^2 v^k diverges: at d_lat=5 (v=2.76) it evaluates to 1.13e3 and at d_lat=8 (v=1.74) to 0.65, while the true diagonal residual E[tanh^2(sqrt(v) z)] - c_1(v)^2 is bounded by 1 and equals 0.0447/0.0235 at v=1.515/0.890. The naive quantity E[tanh^2(sqrt(v)z)] - mu_1^2 v even goes *negative* for v > ~1.2, so eta_star as defined is not a nonnegative diagonal mass in the regime the experiments occupy.

**Fix:** Redo the expansion with variance-v Hermite coefficients c_k(v) (or normalize inputs so v = 1 by construction), replace mu_1^2 by c_1(v)^2/v and etabar by sum_{k>=3} c_k(v)^2 throughout, and add the hypothesis that v stays inside a fixed compact subinterval of (0, v_c) where the expansion is uniformly convergent. Note c_1(v)^2/v depends on d_lat through v, so the signal and noise-dim edges acquire a d_lat dependence the theorem currently denies.


---

## [BLOCKING] The buffer bound multiplies by the bulk width (d_lat - d_int), but Eq (rfnn-mode-decay) decouples the modes — they are absorbed in parallel, so no time accumulates per mode

`buffer-corollary-does-not-follow-modes-are-parallel` · verdict **—** · kind error · effort weeks  

**Where:** /Users/ryan/Desktop/latent_space_diffusion_analysis/sec-rfnn-bounds.tex, Eq (buffer-bound) lines 51-69; same equation at ICLR_2026/sec-appendix.tex:576-590

**Claim:** tau_mem - tau_gen >= (d_lat - d_int) * psi_p/(mu_1^2 beta_t^2) is asserted to "follow by combining the theorem with the per-mode decay". It does not. Eq (rfnn-mode-decay), a_i(T) - a_i* = (a_i(0) - a_i*) e^{-lambda_i T}, is a diagonal, decoupled linear ODE: all modes decay simultaneously. The time to absorb the whole noise-dim block is max_i 1/lambda_i = 1/lambda_min^{noise-dim}, independent of how many modes the block contains. There is no "traversal" that costs one timescale per mode.

**Evidence:** sec-rfnn.tex:66-69 states the mechanism as sequential: "the trajectory traverses the four bulks sequentially". But sec-rfnn.tex:41-45 defines the dynamics as fully diagonalized with independent exponentials. Under those dynamics, tau_gen is set by the slowest signal mode and tau_mem by the sample-bulk modes, so tau_mem - tau_gen = 1/lambda^{sample} - 1/lambda^{signal}; the count d_lat - d_int never enters. Correspondingly, all of the theory's actual d_lat dependence sits in etabar(v(d_lat)) through the sample edge, not in the buffer width.

**Fix:** Either (a) drop the (d_lat - d_int) factor and state the honest bound tau_mem - tau_gen >= 1/lambda_max^{sample} - 1/lambda_min^{signal}, then show that lambda^{sample} = etabar(v)/n decreases with d_lat because v = E||x_t||^2/d_lat -> beta_t^2 as d_lat grows (which predicts the delay saturates once sigma_perp^2 dominates (s^2 + d_int sigma_sig^2)/d_lat — a testable prediction the paper does not make); or (b) replace the decoupled-ODE model by a definition of tau_mem that is genuinely cumulative over modes (e.g. tau_mem = first time a fixed fraction of *summed* sample-mode mass is absorbed) and prove the width factor from that definition. As written the corollary is not implied by its stated premises.


---

## [BLOCKING] The predicted bulk counts d_int, d_lat-d_int, n, p-d_lat-n are contradicted by the paper's own Table tab:bulk-sizes and by the raw eigenvalue files; the real tail boundary sits at index n, not d_lat+n

`predicted-counts-contradicted-by-own-table-and-raw-spectra` · verdict **—** · kind unsupported-claim · effort weeks  

**Where:** /Users/ryan/Desktop/latent_space_diffusion_analysis/sec-rfnn-bounds.tex:46-49 ("Predicted bulk counts match the empirical B1...B4 sizes in Table tab:bulk-sizes across the full d_lat sweep at the predicted indices"); table at ICLR_2026/sec-appendix.tex:164-190

**Claim:** The cited table falsifies the claim it is cited for, and the raw spectra show the theorem's third/fourth boundary does not exist while an unpredicted boundary at index n does.

**Evidence:** Table tab:bulk-sizes at d_lat=20, sigma_perp=0.01 (p=1280, n=500, d_int=5) reports B1..B4 = 916, 311, 48, 5. Theorem predicts 760, 500, 15, 5. Only B4 matches. Same at d_lat=40: observed 1907, 578, 70, 5 vs predicted 2020, 500, 35, 5. Raw spectra (sigma_noise_0.5/exp2_rfnn/raw_data/*/eigenvalues_pre.npy): ranking all consecutive log-gaps, the gap at the predicted boundary d_lat+n ranks 390/1279 (d_lat=20), 1574/2559 (d_lat=40), 1078/5119 (d_lat=80), 557/12799 (d_lat=200) — i.e. no gap at all. Instead a clear gap sits at index exactly n=500 for every d_lat >= 40 (log-gap 0.28-0.73, ratio up to 2.1x: at d_lat=80, ev[499]=0.01956 -> ev[500]=0.00941, while ev[579]=0.003349 -> ev[580]=0.003345). This is consistent with total non-null rank ~ n, i.e. counts d_int, d_lat-d_int, n-d_lat, p-n — the noise-dim block eats into the sample block rather than stacking on top of it, which would make the 'buffer' a reallocation, not an addition. Note _next_steps/NEXT_STEPS.md sec 2c independently records the empirical conjecture as 'd_intrinsic, d_latent - d_intrinsic, n, and p - n', which sums to p + d_lat and so is itself inconsistent with the theorem's p - d_lat - n.

**Fix:** Re-derive the rank bookkeeping. Either prove range(U^diag) is genuinely transverse to range(U^lin) with a quantitative angle bound (the current proof only claims orthogonality 'in expectation'), in which case explain the index-n boundary; or accept counts d_int, d_lat-d_int, n-d_lat, p-n and restate the buffer mechanism accordingly. Delete or correct the sec-rfnn-bounds.tex sentence claiming the table matches.


---

## [BLOCKING] eta_star(x,x) = sum_{k>=3} mu_k^2 (||x||^2/d)^k is a divergent series and violates a hard upper bound; the 1/k! is missing

`eta-star-divergent-series` · verdict **CONFIRMED** · kind error · effort hours  

**Where:** sec-appendix-fourbulk.tex line 57-58 (and ICML (1)/sec-appendix.tex line 633-634); propagated into Lemma 2 (line 140-142) and the Theorem table row 'Sample' (line 223)

**Claim:** With the paper's own definition mu_k = E_{z~N(0,1)}[h_k(z) tanh(z)] (probabilists' Hermite, unnormalized), the correct diagonal residual is sum_{k>=2} mu_k^2/k! (tau^2)^k, NOT sum_{k>=3} mu_k^2 (tau^2)^k. Dropping 1/k! makes the series divergent for every tau>0 (radius of convergence zero) and produces values that exceed the hard bound eta_star(x,x) <= K(x,x) = E[tanh^2] < 1.

**Evidence:** Numerically (Gauss-Hermite, 200 nodes): mu_1=+0.6057, mu_3=-0.3636, mu_5=+0.6852, mu_7=-2.2324, mu_9=+10.494, mu_11=-65.02, mu_13=+502.9, mu_15=-4680.3, i.e. mu_k^2 = 0.367, 0.132, 0.469, 4.98, 1.10e2, 4.23e3, 2.53e5, 2.19e7 -> sum_k mu_k^2 tau^{2k} diverges for all tau>0. With 1/k!: mu_k^2/k! = 0.3669, 0.0220, 0.00391, 0.00099, 3.0e-4, 1.1e-4, ... convergent. At tau^2=||x||^2/d=1 the paper's formula (truncated at k=7) already gives 5.585 whereas the true residual is E[tanh^2] - mu_1^2 = 0.394294 - 0.366879 = 0.027415 — a factor 204 too large and above the impossible ceiling 1. At the paper's actual d_lat=20 operating point (tau^2=0.706) the paper's formula gives 0.5646 versus the true 0.01699 (33x).

**Fix:** Replace by the standard identity eta_star(x,x) = E_u[tanh^2(tau u)] - tau^2 (E_u[tanh'(tau u)])^2 = sum_{k>=2} mu_k(tau)^2/k!, with tau^2 = ||x||^2/d_lat and mu_k(tau) = E_u[h_k(u) tanh(tau u)]. Recompute bar-eta_star and every downstream number (Lemma 2, Theorem table, bulk-gap paragraph) from this.


**Referee:** Verified against the text and independently by quadrature. sec-appendix-fourbulk.tex:36 defines mu_k = E_{z~N(0,1)}[h_k(z) tanh(z)] with probabilists' h_k (unnormalized), and line 57 writes eta_star(x,x) = sum_{k>=3} mu_k^2 (||x||^2/d)^k with no 1/k!. With that definition the Hermite expansion is tanh = sum_k (mu_k/k!) h_k and Parseval gives E[tanh^2] = sum_k mu_k^2/k!, so the 1/k! is genuinely missing. My Gauss-Hermite computation (200 nodes) reproduces the reviewer's coefficients exactly: mu_1=0.605706, mu_3=-0.363595, mu_5=0.685174, mu_7=-2.232403, mu_9=10.4937, mu_11=-65.02, mu_13=502.9, mu_15=-4680.3. So mu_k^2 = 0.3669, 0.1322, 0.4695, 4.984, 110.1, 4228, 2.53e5, 2.19e7 and sum_k mu_k^2 tau^{2k} has radius of convergence zero — the quoted eta_star is a divergent series at every tau>0, and any value it produces is only an artifact of where one truncates. With 1/k! restored, sum_{k>=2} mu_k^2/k! = 0.0274087, which agrees to 5 digits with the independent identity E[tanh^2]-mu_1^2 = 0.394294-0.366879 = 0.0274153. The hard-ceiling argument is also right: eta_star(x,x) <= K(x,x) = E[tanh^2] < 1, yet truncating the paper's formula at k=7 gives 5.585 at tau^2=1 and 0.5646 at the d_lat=20 operating point tau^2=0.706, against exact residuals 0.02742 and 0.01699 (204x and 33x). I confirmed the 0.01699 figure by exact 2-D Gauss-Hermite evaluation of E_W[tanh(z_mu)tanh(z_nu)] on 200 of the paper's own training points: mean diagonal residual = 0.016953. The error is load-bearing: it propagates into Lemma 2 (line 140-142), the Theorem 'Sample' row (line 223), and the bulk-gap threshold (line 254). The same text appears verbatim at 'ICML (1)'/sec-appendix.tex:633-634.


**Referee correction:** The proposed fix is correct. Note the exact identity should be written with tau-dependent coefficients: eta_star(x,x) = E_u[tanh^2(tau u)] - tau^2 (E_u[tanh'(tau u)])^2 = sum_{k>=2} c_k(tau)^2/k! with tau^2 = ||x||^2/d_lat and c_k(tau) = E_u[h_k(u) tanh(tau u)] — i.e. the fix cannot be applied without simultaneously fixing wrong-gaussian-measure-tau, since the k=1 term subtracted off is tau^2 a_1(tau)^2, not mu_1^2.


---

## [BLOCKING] mu_k are defined against N(0,1) but applied to a N(0, ||x||^2/d_lat) pre-activation; tau^2 ranges 0.36-2.76 across the paper's own d_lat sweep

`wrong-gaussian-measure-tau` · verdict **CONFIRMED** · kind error · effort days  

**Where:** sec-appendix-fourbulk.tex lines 31-58 ('Hermite expansion of tanh' paragraph), especially lines 36-39 and Eq. (kernel-decomp) lines 50-56

**Claim:** The paragraph correctly states that ((Wx)_a,(Wy)_a) has covariance diag(||x||^2,||y||^2)/d_lat, and then silently uses the unit-variance Hermite coefficients mu_k of tanh. The expansion coefficients are tau-dependent: the coefficient multiplying x^T y/d_lat is a_1(tau_x) a_1(tau_y) with a_1(tau) = E[tanh'(tau u)] (Stein), which equals mu_1 only at tau=1. tau=1 never holds in the paper's experiments.

**Evidence:** tau^2 = e^{-2t}(s^2 + d_int sig_sig^2 + (d_lat-d_int) sig_perp^2)/d_lat + Delta_t. With s=3, sig_sig=1, d_int=5, sig_perp=0.01, t=0.01: tau^2 = 2.764 (d_lat=5), 1.735 (8), 1.392 (10), 0.935 (15), 0.706 (20), 0.477 (30), 0.363 (40). Computed a_1(tau)^2: 0.178, 0.243, 0.283, 0.373, 0.446, 0.532, 0.729 versus the fixed mu_1^2 = 0.3669 used throughout. The ratio a_1(tau)^2/mu_1^2 runs 0.485 -> 1.99, a 4.1x systematic swing that is monotone in d_lat — i.e. exactly along the axis of the paper's headline claim. Direct simulation confirms the tau-corrected coefficient tracks the data while mu_1^2 does not: at d_lat=5 empirical signal-bulk median = 9.74e-2 vs a_1(tau)^2 alpha_t^2/d_lat = 9.95e-2 (mu_1^2 version = 2.03e-1).

**Fix:** Either (a) rescale inputs so tau=1 by construction and say so, or (b) carry tau-dependent coefficients a_k(tau) everywhere: the linear coefficient becomes E[tanh'(tau u)]^2, the buffer bound's mu_1^2 becomes a_1(tau_{d_lat})^2, and the d_lat-dependence of tau must be tracked when claiming the buffer grows with d_lat.


**Referee:** The text does exactly what the finding says. sec-appendix-fourbulk.tex:40-45 correctly states that ((Wx)_a,(Wy)_a) has covariance diag(||x||^2,||y||^2)/d_lat with off-diagonal x^T y/d_lat, and then Eq. (kernel-decomp) at line 52 uses the N(0,1) coefficient mu_1^2 as the multiplier of x^T y/d_lat. That is only valid at tau=1. The correct coefficient follows from writing z_x = tau_x u, z_y = tau_y(rho u + sqrt(1-rho^2) v) with rho = (x^T y/d)/(tau_x tau_y): E[tanh(z_x)tanh(z_y)] = sum_k c_k(tau_x)c_k(tau_y) rho^k / k! with c_k(tau)=E[h_k(u)tanh(tau u)], and c_1(tau) = tau E[tanh'(tau u)], so the coefficient of x^T y/d_lat is a_1(tau_x)a_1(tau_y) with a_1(tau)=E[tanh'(tau u)], equal to mu_1 only at tau=1. I confirmed the sweep's tau^2 values (s=3, sig_sig=1, d_int=5, sig_perp=0.01, t=0.01) exactly as the finding reports: 2.7644 (d_lat=5), 1.7352 (8), 1.3921 (10), 0.9347 (15), 0.7060 (20), 0.4773 (30), 0.3630 (40) — tau=1 is crossed only once, between d_lat=15 and 20. Materiality is confirmed on the paper's own saved spectra: dividing eigenvalues_pre.npy by p, the signal-bulk median is 9.27e-2 (d=5), 5.01e-2 (20), 3.45e-2 (40), whereas the tau-corrected prediction a_1(tau)^2 alpha_t^2/d_lat gives 9.95e-2, 6.17e-2, 4.16e-2 and the mu_1^2 version gives 2.03e-1, 5.07e-2, 2.54e-2. The tau-correction is the difference between agreeing at the small-d_lat end and being 2x off there, and the correction is monotone in d_lat, i.e. aligned with the paper's headline axis.


**Referee correction:** Two of the finding's numbers are slightly off, without changing the conclusion. Computing a_1(tau)^2 = (E[tanh'(tau u)])^2 at sigma_perp=0.01 I get 0.180, 0.255, 0.297, 0.382, 0.446, 0.539, 0.602 for d_lat = 5, 8, 10, 15, 20, 30, 40 — the finding's last entry (0.729 at d_lat=40) is a typo and breaks its own monotone trend. So the swing in a_1(tau)^2/mu_1^2 across the sweep is 0.49 -> 1.64, a factor 3.3 (not 4.1). Also the empirical signal-bulk median at d_lat=5 is 9.27e-2 by my reading of eigenvalues_pre.npy/p, not 9.74e-2.


---

## [BLOCKING] Bulk-edge normalization is wrong: /psi_p should be /d_lat for the linear bulks and /n for the sample bulk; predicts a d_lat-independence the paper's own data contradicts

`edge-normalization-psi-p` · verdict **CONFIRMED** · kind error · effort days  

**Where:** sec-appendix-fourbulk.tex Lemma 1 (lines 112-131, esp. line 129 'rescaled by psi_p'), Lemma 2 (line 139), Theorem table (lines 219-224), and the buffer bound Eq. (buffer-bound) in ICML (1)/sec-appendix.tex line 584 / sec-rfnn-bounds.tex

**Claim:** U^lin = mu_1^2/(p d) W M_t W^T has nonzero eigenvalues = mu_1^2/(p d) * eig(M_t^{1/2} W^T W M_t^{1/2}) ~= mu_1^2/(p d) * p * eig(M_t) = mu_1^2 eig(M_t)/d_lat. The proof at line 124-129 correctly notes E[W^T W/d] = psi_p I but then forgets it already divided by d, giving /psi_p instead of /d_lat. Likewise the sample bulk sits at eta_star/n (n equal eigenvalues of (1/n) * eta_star I_n), not eta_star/psi_p. Since psi_p = 64 is held fixed while d_lat sweeps 5->40, the paper's formulas predict d_lat-independent bulk edges; the measured edges fall with d_lat.

**Evidence:** From the paper's own saved spectra (eigenvalues_pre.npy, divided by p to undo the code's missing 1/p): signal-bulk median = 9.27e-2 (d_lat=5), 5.01e-2 (20), 3.45e-2 (40) — a 2.7x decline. Paper's formula mu_1^2 alpha_t^2/psi_p = 1.585e-2, constant (off by 5.8x at d_lat=5, 2.2x at 40, wrong trend). Correct mu_1^2 alpha_t^2/d_lat = 2.03e-1, 5.07e-2, 2.54e-2, and with the tau-corrected coefficient 9.95e-2, 6.17e-2, 4.16e-2 — right magnitude and right trend. Noise-dim: measured 1.75e-3 (d_lat=8) -> 3.17e-4 (40); paper's mu_1^2 beta_t^2/psi_p = 1.14e-4, constant (15x low at d_lat=8). Sample bulk at d_lat=20: measured median 1.85e-6 vs paper's eta_star/psi_p (using the corrected eta_star, being generous) = 2.65e-4, off by 143x; using the paper's own eta_star formula the sample bulk alone would carry n*eta_star/psi_p = 4.41 of trace against tr(U) = 0.317.

**Fix:** Redo the lift bookkeeping: signal edge a_1(tau)^2 alpha_t^2/d_lat, noise-dim edge a_1(tau)^2 beta_t^2/d_lat, sample edge eta_star/n. Then re-derive the buffer bound: 1/lambda_noise-dim = d_lat/(a_1^2 beta_t^2), so tau_mem - tau_gen >~ (d_lat - d_int) d_lat/(a_1(tau)^2 beta_t^2) — superlinear in d_lat, not linear. Check the trace identity sum_i lambda_i = E[tanh^2(tau u)] as a sanity constraint on any corrected table.


**Referee:** The algebra error is real and I re-derived it. With W entries N(0,1) (ICLR_2026/sec-rfnn.tex:13 and eq:rfnn make the 1/sqrt(d_lat) explicit in the activation, so W itself is O(1)), Eq. (U-lin) U^lin = mu_1^2/(p d) W M_t W^T has nonzero eigenvalues (mu_1^2/(p d)) eig(M_t^{1/2} W^T W M_t^{1/2}). Since E[W^T W] = p I_d, this is (mu_1^2/(p d)) * p * eig(M_t) = mu_1^2 eig(M_t)/d_lat. The proof at lines 124-129 writes E[W^T W/d_lat] = psi_p I (true) and then asserts 'the eigenvalues of M_t rescaled by psi_p' — it applied the already-divided-by-d_lat matrix while the prefactor 1/(p d_lat) is still standing, losing a factor d_lat^2/p. The paper's edge is therefore off by exactly d_lat^2/p = d_lat/psi_p from the correct one. Empirically: alpha_t^2 = 2.7644, so mu_1^2 alpha_t^2/psi_p = 1.585e-2 for every d_lat, while the measured signal-bulk median (eigenvalues_pre.npy/p) falls 9.27e-2 -> 5.01e-2 -> 3.45e-2 across d_lat = 5, 20, 40 — wrong magnitude and wrong trend; mu_1^2 alpha_t^2/d_lat gives 2.03e-1, 5.07e-2, 2.54e-2, right trend and exact at d_lat=20. Noise-dim, sigma_perp=0.5 (beta_t^2 = 0.2649): paper's mu_1^2 beta_t^2/psi_p = 1.52e-3 constant, measured 9.07e-3 (d=8) -> 5.50e-3 (20) -> 3.27e-3 (40) -> 7.20e-4 (200), a 12x span; mu_1^2 beta_t^2/d_lat gives 9.12e-3, 4.86e-3, 2.43e-3, 4.86e-4 — right trend and within ~1.5x throughout. At sigma_perp=0.01, measured noise-dim 1.75e-3 (d=8) -> 3.17e-4 (40) vs the paper's constant 1.14e-4. Since psi_p = 64 is held fixed while d_lat sweeps 5->200, the paper's formulas predict d_lat-independent edges and the paper's own data contradicts that at every point. The trace sanity check also holds: sum(eigenvalues_pre)/p = 0.545, 0.466, 0.438, 0.369, 0.317, 0.253, 0.213 for d_lat = 5..40, matching E[tanh^2(tau u)] = 0.576, 0.495, 0.455, 0.382, 0.332, 0.266, 0.224 at the corresponding tau — so the correct normalization is pinned by the trace identity.


**Referee correction:** The re-derived buffer bound is right in form: 1/lambda_noise-dim = d_lat/(a_1(tau)^2 beta_t^2) gives tau_mem - tau_gen >~ (d_lat - d_int) d_lat / (a_1(tau)^2 beta_t^2), superlinear rather than linear in d_lat. But note this only holds at fixed sigma_perp; the paper's fixed-p, fixed-null-energy control rescales sigma_perp^2 ∝ 1/(d_lat - d_int), which changes the exponent again, so the superlinear claim must be stated per width/energy control, not globally.


---

## [BLOCKING] Rank arithmetic double-counts: rank(U) = min(p,n) = n, not d_lat + n; the paper's own spectra show the cliff at index n, not d_lat+n

`rank-overcount-dlat-plus-n` · verdict **CONFIRMED** · kind error · effort days  

**Where:** sec-appendix-fourbulk.tex Theorem proof line 236 ('they span a subspace of dimension dlat + nsamp < pwidth'), Theorem table rows 'Sample' (count n) and 'Rank-null' (count p - d_lat - n), lines 223-224; also code/v3/lib/eigenvalues.py bulk_indices() which hard-codes the boundary at d_latent + n

**Claim:** U = (1/n) sum_mu phi(x_t^mu)phi(x_t^mu)^T is a sum of n rank-one terms, so rank(U) <= n regardless of the Hermite decomposition. U^lin and U^diag are both PSD with generically-independent ranges of size d_lat and n, so rank(U^lin + U^diag) = d_lat + n > n = rank(U); the discrepancy cannot be absorbed by a remainder with ||R_t||_op = o(1) — R_t must annihilate d_lat directions whose eigenvalues are exactly of noise-dim-bulk size. The correct counts are sample = n - d_lat and rank-null = p - n.

**Evidence:** Direct simulation at d_lat=20, n=500, p=1280 with a single diffusion draw (an exact n-point Gram): lambda[499] = 2.60e-8, lambda[500] = 1.25e-17 (machine zero), lambda[520] = 3.41e-18 — the hard rank cliff is at index n=500, not at d_lat+n=520. Same with no diffusion noise: lambda[499]=2.54e-10, lambda[500]=1.61e-17. The paper's own saved spectra agree: scanning for the largest log-drop in indices [400,800] of eigenvalues_pre.npy at sigma_perp=0.5 gives index 499 for d_lat=40 (ratio 1.32), 499 for d_lat=100 (1.68), 499 for d_lat=200 (1.73) — never d_lat+n (540/600/700). At d_lat=200 the paper's boundary is 200 indices (40% of the sample bulk) past the real one.

**Fix:** State rank(U) = min(p,n) and re-count: signal d_int, noise-dim d_lat-d_int, sample n-d_lat, rank-null p-n. Fix bulk_indices() in code/v3/lib/eigenvalues.py accordingly and re-generate Table 3 / the bulk-detection overlays, since the a-priori coloring currently mislabels d_lat eigenvalues.


**Referee:** Verified by direct simulation and by the paper's own spectra. Simulating at the paper's defaults (d_lat=20, d_int=5, n=500, p=1280, sigma_perp=0.01, t=0.01, phi = tanh(Wx/sqrt(d))/sqrt(p)) with a single diffusion draw: lambda[499] = 2.38e-8, lambda[500] = 1.13e-17, lambda[520] = 3.44e-18 — an exact machine-precision rank cliff at index n=500, not at d_lat+n=520. Without diffusion noise, lambda[499]=2.80e-10, lambda[500]=1.49e-17. The saved spectra agree wherever the sample bulk is resolved at all: at sigma_perp=0.5, lambda[499]/lambda[500] = 1.32 (d_lat=40), 1.70 (60), 2.08 (80), 1.68 (100), 1.90 (150), 1.73 (200), while lambda[d_lat+n-1]/lambda[d_lat+n] = 1.001-1.009 at all of those — the drop is at index n, and at d_lat=200 the paper's boundary (700) is 200 indices past the real one. The theorem proof's asserted orthogonality (line 232-236) is the source: phi^perp(x_t^mu) is the residual of the same n feature vectors that generate U, so U^lin's range is not a fifth independent direction set — for a single draw both live in span{phi(x_t^mu)}, of dimension at most n. code/v3/lib/eigenvalues.py bulk_indices() hard-codes sample = slice(d_latent, d_latent+n), and ICLR_2026/sec-rfnn.tex:53 states colors are 'assigned by sorted eigenvalue-index blocks', so the mislabeling propagates into the published figures.


**Referee correction:** Two refinements. (i) 'rank(U) = min(p,n) = n' is exactly true only for the single-noise-draw Gram. The object actually defined in ICLR_2026 Eq. (eq:U) and computed by code/v3/lib/eigenvalues.py averages over diffusion noise (50-500 MC draws), so U is a sum of n full-rank covariances and is generically rank p; my MC-50 reproduction gives lambda[499]/lambda[500] = 1.016, a soft cliff rather than a hard one. The correct statement is that the effective rank boundary sits at n, and that U^lin + U^diag as written overcounts by d_lat. (ii) The argument 'the discrepancy cannot be absorbed by a remainder with ||R_t||_op = o(1)' is not sound as stated: every one of the four claimed edges is itself O(1/d_lat) = o_d(1), so the o(1) remainder bound in Eq. (U-split) is vacuous and cannot be violated. That vacuity is itself a defect worth stating, but the load-bearing evidence for this finding is the empirical cliff at index n plus the exact single-draw rank, not the remainder bound.


---

## [BLOCKING] The 1[x=y] term is vacuous as a kernel-function identity, and the neglected off-diagonal Hermite-k>=3 mass is not o(1) for this data

`indicator-not-a-kernel-statement` · verdict **CONFIRMED** · kind error · effort weeks  

**Where:** sec-appendix-fourbulk.tex Eq. (kernel-decomp), lines 50-58; supporting citation to ELKaroui2010spectrum at lines 59-60

**Claim:** As written, Eq. (kernel-decomp) is an identity between functions on R^d x R^d; 1[x=y] is supported on a Lebesgue-null set, so the equation asserts E_W[tanh tanh] = mu_1^2 x^T y/d + o_d(1) for all x != y, with the eta_star term contributing nothing. The correct statement (Pennington-Worah; Louart-Liao-Couillet; Benigni-Peche; El Karoui) is a Gram-matrix statement: K_{mu nu} = zeta (x_mu^T x_nu/d) + (eta - zeta) delta_{mu nu} + E, with zeta = (E[z sigma(z)])^2, eta = E[sigma(z)^2], eta - zeta = sum_{k>=2} mu_k^2/k!, and with ||E||_op (not entrywise) controlled. That control requires El Karoui's hypothesis that off-diagonal x_mu^T x_nu/d concentrate near a common value, which the paper's k=10 well-separated Gaussian-mixture data violates.

**Evidence:** Exact 2D Gauss-Hermite evaluation of K_{mu nu} = E_W[tanh(z_mu)tanh(z_nu)] on n=200 of the paper's own training points (d_lat=20, sigma_perp=0.01, t=0.01), minus the exact tau-corrected linear part: the residual D = K - K^lin has mean diagonal 0.01695 (= eta_star) but ||offdiag(D)||_op = 0.4229, i.e. 25x eta_star, and max|offdiag(D)| = 0.0487, nearly 3x the diagonal value. Eigenvalues of D: 0.448, 0.319, 0.282, 0.254, 0.221, ... — a structured rank-~k(=10) block, not eta_star*I_n. Cause: same-cluster pairs have correlation rho up to 0.979 (90th pct 0.688), so x_mu^T x_nu/d_lat = O(1), not o(1), and the higher-Hermite terms sum_k (mu_k^2/k!) rho^k do not vanish off-diagonal.

**Fix:** Restate as a Gram-matrix theorem with delta_{mu nu} and an explicit ||.||_op error bound; then either (a) restrict to a data model satisfying El Karoui's off-diagonal concentration (which the k-cluster mixture with s=3 does not), or (b) keep the k>=3 off-diagonal terms and treat the resulting rank-k cluster block explicitly — it is currently discarded even though its operator norm dominates the term it is being discarded in favour of.


**Referee:** Both halves check out. Eq. (kernel-decomp) at lines 50-56 is written as an equality of functions of (x,y) in R^d x R^d, and 1[x=y] is supported on a Lebesgue-null set, so as a function identity it says E_W[tanh tanh] = mu_1^2 x^T y/d + o_d(1) for all x != y and the eta_star term is inert — the intended content is a Gram-matrix statement with delta_{mu nu} and an operator-norm error bound, which is what Pennington-Worah / El Karoui / Louart-Liao-Couillet actually prove. The second half is the material one and I reproduced it. Exact 2-D Gauss-Hermite evaluation of K_{mu nu} = E_W[tanh(z_mu)tanh(z_nu)] on n=200 of the paper's own training points (d_lat=20, sigma_perp=0.01, t=0.01, generated by code/v3/lib/data_synthetic.generate_data with seed 42), minus the exact tau-corrected linear part a_1(tau_mu)a_1(tau_nu) x_mu^T x_nu/d: mean diagonal 0.016953 (= eta_star), but ||offdiag(D)||_op = 0.4219 (25x eta_star) and max|offdiag| = 0.0480 (2.8x the diagonal). Top eigenvalues of D: 0.4472, 0.3193, 0.2728, 0.2613, 0.2243, 0.1377, 0.1301, 0.1185, 0.0990, 0.0916 — a structured rank-~k block (k=10 clusters), not eta_star*I_n. Cause confirmed: off-diagonal correlations x_mu^T x_nu/(||x_mu|| ||x_nu||) reach 0.984 with 90th percentile 0.690 and mean 0.086, so El Karoui's off-diagonal concentration hypothesis fails on this data. Translated to U eigenvalues (divide by n), the discarded cluster block sits at 0.447/500 = 8.9e-4, versus eta_star/n = 3.4e-5 and versus the measured noise-dim bulk at sigma_perp=0.01, d_lat=20 of 5.3e-4 — the neglected term is the size of a bulk the theorem claims to explain, not a remainder.


**Referee correction:** Worth adding that the cluster block is also the most likely true explanation of the sigma_perp=0.01 'merge' (lambda[19]/lambda[20] = 1.013), which the paper attributes to the bulk-gap criterion — so this is not merely a rigor complaint, it displaces the paper's stated mechanism for one of its own observations.


---

## [BLOCKING] Lemma 2's asymptotic orthonormality of the residual features is false at the paper's parameters

`lemma2-orthonormality-false` · verdict **CONFIRMED** · kind error · effort weeks  

**Where:** sec-appendix-fourbulk.tex Lemma 2 proof, lines 149-152 ('The kernel decomposition gives v_mu^T v_nu = delta_{mu nu} + o_d(1), so the {v_mu} are asymptotically orthonormal and the rank is nsamp')

**Claim:** The residual directions v_mu = phi^perp(x_t^mu)/||phi^perp(x_t^mu)|| are far from orthonormal for the paper's data, so neither the rank-n conclusion nor the eigenvalue-concentration conclusion follows.

**Evidence:** Using the exact residual Gram computed above (d_lat=20, sigma_perp=0.01, n=200 of the paper's training points), the normalized overlaps C_{mu nu} = D_{mu nu}/sqrt(D_mu mu D_nu nu) have max|off-diagonal| = 0.928, 99th percentile 0.732, mean|off| = 0.117, and ||C||_op = 24.2 (would be 1.0 if orthonormal). The same-cluster structure of the data is what breaks it.

**Fix:** Drop the orthonormality shortcut. Either prove an operator-norm bound on the residual Gram under an explicit assumption ruling out near-collinear samples, or acknowledge that with k clusters the residual Gram has a rank-k spike and give the spectrum of eta_star I + (cluster block) instead.


**Referee:** The proof text at lines 149-152 does assert v_mu^T v_nu = delta_{mu nu} + o_d(1) and derive both the rank-n conclusion and the eigenvalue concentration from it. Using the exact residual Gram D computed above on the paper's own d_lat=20, sigma_perp=0.01, n=200 training points, the normalized overlaps C_{mu nu} = D_{mu nu}/sqrt(D_mu mu D_nu nu) have max|off-diagonal| = 0.948, 99th percentile 0.735, mean|off| = 0.117, and ||C||_op = 24.25 (would be 1.0 if orthonormal). These reproduce the reviewer's numbers (0.928 / 0.732 / 0.117 / 24.2) to within the seed. The failure is driven by the same-cluster structure — with k=10 centers of norm s=3 in d_int=5 and only sigma_sig=1 within-cluster spread, same-cluster pairs have correlation up to 0.98. The claim is false at the paper's actual operating point, not merely unproved.


---

## [BLOCKING] Remark 'Fixed-W vs W-averaged kernel' contains an algebra error; the claimed error term actually diverges, and empirically the finite-width fluctuation dwarfs eta_star

`fixedW-remark-algebra` · verdict **CONFIRMED** · kind error · effort weeks  

**Where:** sec-appendix-fourbulk.tex Remark rem:fixedW, lines 184-187: 'operator-norm corrections of order nsamp/sqrt(pwidth) = sqrt(psi_n/(psi_p pwidth)) -> 0'

**Claim:** n/sqrt(p) does NOT equal sqrt(psi_n/(psi_p p)). Writing n = psi_n d, p = psi_p d gives n/sqrt(p) = psi_n sqrt(d/psi_p), which DIVERGES like sqrt(d_lat) in the proportional limit, whereas sqrt(psi_n/(psi_p p)) = sqrt(psi_n/(psi_p^2 d_lat)) -> 0. The remark's conclusion is the opposite of what its own quantity does. Moreover the finite-W fluctuation is not a small correction to the sample bulk — it is larger than the entire mechanism the sample bulk is attributed to.

**Evidence:** At the paper's d_lat=20 defaults: n/sqrt(p) = 500/sqrt(1280) = 13.975 while sqrt(psi_n/(psi_p p)) = 0.0175 — the two sides differ by 800x. Empirically, over 20 independent draws of W at p=1280, n=500: the s.d. of an off-diagonal kernel entry phi_mu^T phi_nu around its W-mean is 0.0075, i.e. 44% of eta_star = 0.01699; and ||K_W - E_W K||_op = 3.32, which is 195x eta_star. So the 'sample bulk' of the matrix the paper actually diagonalizes is dominated by finite-width randomness, not by the Hermite-k>=3 mass.

**Fix:** Redo the concentration estimate honestly (the right scale is a Wishart-type fluctuation with aspect ratio n/p = 0.39, giving O(1) relative spread on the small eigenvalues), and state that at p = 64 d_lat the Hermite tail is not the dominant source of the sample bulk. Either push p far higher, or attribute the sample bulk to the finite-width kernel fluctuation and derive its edge from that.


**Referee:** The algebra error is plain on the page. sec-appendix-fourbulk.tex:185-186 writes 'operator-norm corrections of order nsamp/sqrt(pwidth) = sqrt(psi_n/(psi_p pwidth)) -> 0'. With n = psi_n d and p = psi_p d, n/sqrt(p) = psi_n sqrt(d/psi_p), which diverges like sqrt(d_lat) in the very proportional limit the appendix works in, while sqrt(psi_n/(psi_p p)) = sqrt(psi_n/(psi_p^2 d_lat)) -> 0. At the paper's defaults the two sides are 500/sqrt(1280) = 13.975 and sqrt(25/81920) = 0.01747 — a factor 800. So the remark's stated quantity does the opposite of what the remark concludes, and the concentration upgrade it is supposed to license is not established. I also reproduced the empirical half: over 20 independent draws of W at p=1280, n=500 on the paper's own training points, the s.d. of an off-diagonal kernel entry around its W-mean is 0.00767, i.e. 45% of eta_star = 0.01699, and ||K_W - E_W K||_op = 2.71, i.e. 160x eta_star. So the finite-width fluctuation is not a correction to the sample-bulk mechanism, it is larger than the mechanism.


**Referee correction:** Numeric detail: I get ||K_W - E_W K||_op = 2.71 (160x eta_star), not 3.32 (195x) — the difference is seed/point-set, and the conclusion is unaffected. One caveat to the empirical half: an operator-norm bound is a statement about the top of the spectrum and does not by itself prove the small eigenvalues are dominated by the fluctuation. The correct framing is that the perturbation is not small relative to the term whose spectrum is being claimed, so the concentration statement as written is unsupported — establishing that the sample bulk *is* the Wishart fluctuation requires the separate calculation the finding proposes.


---

## [BLOCKING] The (d_lat - d_int) factor in Eq. (buffer-bound) contradicts the paper's own mode-decay equation: modes relax in parallel, not in a queue

`buffer-count-is-a-non-sequitur` · verdict **CONFIRMED** · kind error · effort days  

**Where:** ICLR_2026/sec-appendix.tex Eq. (buffer-bound), lines 578-591; sec-rfnn-bounds.tex Eq. (buffer-bound), lines 51-69; ICLR_2026/sec-rfnn.tex line 65 ("Mechanism: a noise-dimension buffer"); ICLR_2026/abstract.tex ("each excess latent direction adds a mode that must be absorbed before ... memorization begins")

**Claim:** Eq. (rfnn-mode-decay) is a diagonal linear ODE: a_i(T) - a_i* = (a_i(0) - a_i*) e^{-lambda_i T}. Every mode's trajectory depends on lambda_i, a_i(0), a_i* and NOTHING else. The time at which mode i reaches tolerance eps is T_i(eps) = log(1/eps)/lambda_i, independent of how many other modes exist, of their eigenvalues, and of whether they have converged. Therefore the time to "reach" the sample bulk is 1/lambda_sample, full stop; multiplying a per-mode timescale by the COUNT (d_lat - d_int) of noise-dim modes has no basis in the dynamics. Deleting every noise-dim mode while holding lambda_sample fixed would change tau_mem by exactly zero. The queue/customer-in-line metaphor requires a shared serial resource (a budget, a norm constraint, a capacity bottleneck); Eq. (rfnn-gradflow) contains none. Verdict: the corollary is a non-sequitur, not a loose bound.

**Evidence:** sec-rfnn-bounds.tex line 54-56: "modes are absorbed in order of decreasing lambda_i, so the time to traverse the noise-dim bulk is bounded below by its width times the slowest noise-dim timescale." "Order of decreasing lambda_i" is a statement about the ORDER in which modes finish, which is a consequence of parallel exponentials with different rates; it is silently converted into a claim that they finish SEQUENTIALLY, i.e. that mode j must wait for mode i. Direct empirical falsification of the serial picture on the paper's own U (n=500, dint=5, p=64 d_lat, t=0.01, MC over 80 diffusion-noise draws, paper normalization phi = tanh(Wx/sqrt(d))/sqrt(p)): the literal serial traversal time sum_{i in noise-dim} 1/lambda_i vastly exceeds the actual 1/lambda_sample. sigma_perp=0.5: d_lat=100 gives sum 1/lambda_nd = 73,926 vs tau_mem = 1/lambda_max^sample = 32,866. sigma_perp=0.01: d_lat=100 gives sum 1/lambda_nd = 621,474 vs tau_mem = 26,740 (23x). If the queue existed, memorization could not begin until ~6e5; it begins at ~2.7e4.

**Fix:** Delete the (d_lat - d_int) factor. Under Eq. (rfnn-mode-decay) the only defensible statement is tau_mem - tau_gen = log(1/eps)(1/lambda_sample - 1/lambda_min^signal), with no mode count anywhere. If the paper wants a count-dependent delay it must introduce an explicit mechanism that couples modes -- a fixed parameter/step budget, a norm or capacity constraint on A, a shared learning-rate normalization, or a global observable whose threshold aggregates over modes -- and derive the count from that mechanism. The abstract, intro contribution bullet, and the sec-rfnn "Mechanism" paragraph must all be rewritten, because all three assert the serial-absorption story.


**Referee:** Verified the text. ICLR_2026/sec-rfnn.tex lines 29-35 give Eq. (rfnn-mode-decay) as a_i(T)-a_i* = (a_i(0)-a_i*)e^{-lambda_i T}, a fully decoupled diagonal linear ODE; nothing in Eq. (rfnn-gradflow) couples modes. The per-mode tolerance-crossing time log(1/eps)/lambda_i therefore cannot depend on how many other modes exist, so the factor (d_lat-d_int) in Eq. (buffer-bound) (ICLR_2026/sec-appendix.tex, label at line 586; sec-rfnn-bounds.tex lines 51-69) has no derivation behind it. The slide is exactly where the finding says: sec-rfnn-bounds.tex 'modes are absorbed in order of decreasing lambda_i, so the time to traverse the noise-dim bulk is bounded below by its width times the slowest noise-dim timescale' converts an ordering statement into a serial-queue statement. sec-rfnn.tex line 65 ('traverses the four bulks sequentially') and abstract.tex ('each excess latent direction adds a mode that must be absorbed before sample-specific memorization begins') both assert the serial picture. I reproduced the empirical falsification on U built per compute_U with the paper's phi normalization (U_code/p), n=500, dint=5, p=64*d_lat, t=0.01, S=100 MC draws, seed 42: sigma_perp=0.5, d_lat=100, sum_{i in noise-dim} 1/lambda_i = 73,746 vs 1/lambda_max^sample = 33,651; sigma_perp=0.01, d_lat=100, 620,829 vs 26,065 (24x). A literal serial traversal would put memorization onset far later than the sample-bulk timescale actually is.


**Referee correction:** Two nits on the write-up, not on the verdict: (i) the finding says deleting noise-dim modes 'would change tau_mem by exactly zero' — true at fixed lambda_sample, but in this construction d_lat cannot be changed at fixed lambda_sample (both the signal edge ~1/d_lat and the sample edge ~eta_bar_star(d_lat)/n move), so the thought experiment is counterfactual rather than something the sweep does; (ii) Eq. (buffer-bound) appears in two places in the current draft (ICLR_2026/sec-appendix.tex around line 578-591 and the orphaned sec-rfnn-bounds.tex), and the main-text theory section that would host it (ICLR_2026/sec-theory.tex) is still a \todo stub, so the fix list must include the appendix copy, the abstract, the intro bullet at sec-intro.tex line 26, and sec-rfnn.tex line 65.


---

## [BLOCKING] Eq. (buffer-bound) is violated numerically on the paper's own configurations, by up to 31x, and the violation grows with d_lat

`buffer-bound-numerically-false` · verdict **CONFIRMED** · kind error · effort hours  

**Where:** ICLR_2026/sec-appendix.tex Eq. (buffer-bound) line 578-587; sec-rfnn-bounds.tex Eq. (buffer-bound) line 57-63

**Claim:** Instantiating the claimed inequality tau_mem - tau_gen >= (d_lat - d_int)/lambda_min^{noise-dim} with the *measured* eigenvalues of the paper's own U (so the four-bulk structure is granted, and only the bound is tested) shows it is false for every d_lat >= 20 at sigma_perp=0.5 and for every d_lat tested at sigma_perp=0.01. The failure is not marginal and it worsens monotonically with d_lat, i.e. precisely in the regime the paper's headline claim lives.

**Evidence:** Measured (n=500, dint=5, p=64 d_lat, t=0.01, S=100 MC noise draws, seed 42; tau_gen := 1/lambda[dint-1], tau_mem := 1/lambda[d_lat] = top of the sample bulk, lambda_min^{nd} := lambda[d_lat-1]):
sigma_perp=0.5:  d=10 LHS 741.9 vs RHS 714.4 (holds); d=20 LHS 2049.6 vs RHS 3789.1 (FAILS 1.8x); d=40 LHS 6801 vs RHS 18433 (2.7x); d=60 LHS 14053 vs RHS 45939 (3.3x); d=100 LHS 32769 vs RHS 171329 (5.2x).
sigma_perp=0.01: d=10 LHS 887 vs RHS 4204 (4.7x); d=20 LHS 2146 vs RHS 31845 (14.8x); d=40 LHS 4512 vs RHS 139049 (30.8x); d=60 LHS 9481 vs RHS 303351 (32x); d=100 LHS 26627 vs RHS 816475 (30.7x).
The single case that holds (sigma_perp=0.5, d=10) holds by 4%.

**Fix:** Retract Eq. (buffer-bound) as stated. If any quantitative bound is to be kept, it must be re-derived and then checked against this exact numerical test (build U per code/experiment_v2_rfnn.py compute_U, divide by p for the paper's phi normalization, eigendecompose, read off lambda[dint-1], lambda[d_lat-1], lambda[d_lat]) before it goes back in the paper.


**Referee:** I rebuilt U independently (same recipe: compute_U from code/experiment_v2_rfnn.py, divided by p, seed 42, S=100) and instantiated the inequality with measured eigenvalues (tau_gen := 1/lambda[dint-1], tau_mem := 1/lambda[d_lat], lambda_min^nd := lambda[d_lat-1]). sigma_perp=0.5: d=10 LHS 745 vs RHS 695 (holds by 7%); d=20 2112 vs 3808 (fails 1.8x); d=40 7352 vs 19,294 (2.6x); d=100 33,556 vs 169,197 (5.0x). sigma_perp=0.01: d=10 890 vs 4370 (4.9x); d=20 2112 vs 32,058 (15x); d=40 4889 vs 88,081... (using lambda[d-1]: 35/0.000251296 = 139,281, 28x); d=100 25,990 vs 795,612 (31x). These match the reviewer's numbers to within Monte-Carlo noise. The violation is large, one-directional, and grows monotonically with d_lat, i.e. worst exactly where the headline claim lives, so it is not a finite-size or MP-fluctuation artifact. Note this test is generous to the paper: it grants the four-bulk structure and uses measured edges rather than the (separately wrong) analytic form psi_p/(mu_1^2 beta_t^2).


---

## [BLOCKING] Lemma 1 / Theorem 1 bulk edges are wrong by a factor p/d_lat^2: they should be divided by d_lat, not by psi_p

`lemma1-psi-p-algebra-error` · verdict **CONFIRMED** · kind error · effort hours  

**Where:** ICLR_2026/sec-appendix.tex Lemma "Rank and block structure of U^lin" lines 690-710 and Theorem "Four-bulk structure" edge table lines 786-805; identical text in sec-appendix-fourbulk.tex lines 112-131 and 208-228

**Claim:** U^lin = mu_1^2 W M_t W^T/(p d_lat) with W having i.i.d. N(0,1) entries, so W^T W ~ p I_{d_lat} and the nonzero eigenvalues of W M_t W^T are p*eig(M_t). Hence eig(U^lin) = mu_1^2 eig(M_t)/d_lat. The paper states mu_1^2 eig(M_t)/psi_p. These differ by psi_p/d_lat = p/d_lat^2. The error is exactly what makes the signal edge (and therefore tau_gen) look d_lat-independent at fixed psi_p, which is the second load-bearing half of the buffer story. A trace sanity check kills the stated version independently: tr(U) = E[tanh^2] = O(1) and d_lat-independent-ish, but summing the stated edges gives mu_1^2 d_lat tr(M_t)/(d_lat psi_p) + n eta_bar/psi_p, which grows linearly in d_lat at fixed psi_p.

**Evidence:** The lemma's own proof contains the slip: line 703-707, "E[W^T W/d_lat] = psi_p I_{d_lat} ... the nonzero eigenvalues of W M_t W^T equal those of M_t^{1/2} W^T W M_t^{1/2}, which are the eigenvalues of M_t rescaled by psi_p" -- W^T W (not W^T W/d_lat) rescales by p = psi_p d_lat, and the prefactor 1/(p d_lat) then leaves 1/d_lat. Decisive numerical discriminator, varying p at fixed d_lat=20, sigma_perp=0.5 (the /psi_p form predicts the edges fall 3.1x as p triples; the /d_lat form predicts they are constant):
 p=820  (psi_p=41):  lambda_min^signal 0.030616, lambda_min^nd 0.0039806, lambda_max^sample 0.00050823
 p=1280 (psi_p=64):  0.032792, 0.0039652, 0.00048038
 p=2560 (psi_p=128): 0.033049, 0.0038612, 0.00046497
The edges are p-independent to within 8% over a 3.1x range of p. Second discriminator, varying d_lat at fixed psi_p=64, sigma_perp=0.5: median signal eigenvalue 0.04866 (d=20) -> 0.01397 (d=100), a 3.5x fall, while the theorem predicts a constant 0.01585 and the corrected mu_1^2 alpha_t^2/d_lat predicts 0.05071 -> 0.01014.

**Fix:** Replace psi_p by d_lat in Lemma 1 and in the Theorem's signal and noise-dim rows: signal edge mu_1^2 alpha_t^2/d_lat, noise-dim edge mu_1^2 beta_t^2/d_lat. Then state explicitly that the data-side edges are independent of p (for p > d_lat + n), which is a stronger and correct claim, and re-derive every downstream consequence -- in particular tau_gen and the width-control argument.


**Referee:** The algebra is wrong as the finding states. With W = randn(p,d)/sqrt(d) and features tanh(x W^T) (code/experiment_v2_rfnn.py, RFNNScore.__init__/forward), the paper's W has i.i.d. N(0,1) entries and U^lin = mu_1^2 W M_t W^T/(p d). Nonzero eig(W M_t W^T) = eig(M_t^{1/2} W^T W M_t^{1/2}) ~ p*eig(M_t), so eig(U^lin) ~ mu_1^2 eig(M_t)/d_lat, not /psi_p. The slip is visible verbatim in the proof (ICLR_2026/sec-appendix.tex lines ~703-707): E[W^T W/d]=psi_p I is correct, but the rescaling applied is to W^T W, which is p = psi_p*d_lat. Two independent numerical discriminators, both run by me: (1) p sweep at d=20, sigma_perp=0.5 over a 6.2x range of p (p=820/1280/2560/5120, psi_p=41/64/128/256): lambda_min^signal = 0.03555, 0.03576, 0.03475, 0.03435; lambda_min^nd = 0.003834, 0.003938, 0.004021, 0.003986; lambda_max^sample = 5.22e-4, 4.69e-4, 4.85e-4, 4.72e-4 — all p-independent to a few percent, whereas the /psi_p form predicts a 6.2x fall. (2) Trace partition: sum of the top d_lat eigenvalues equals the exact Gaussian-equivalent linear share q*E[tanh'(sqrt(q)z)]^2 essentially exactly (d=100, sigma_perp=0.5: measured 0.2283 vs predicted 0.2285; d=20: 0.3440 vs 0.3495), which is only consistent with per-mode edges scaling as 1/d_lat. The stated form happens to agree with measurement near d_lat ~ psi_p = 64 and diverges from it in both directions, which is presumably why the error survived.


**Referee correction:** One refinement to the proposed fix: the correct prefactor is not exactly mu_1^2 but the Gaussian-equivalent linear coefficient c_1(q)^2/q = E[tanh'(sqrt(q) z)]^2 with q = E||x_t||^2/d_lat, which equals mu_1^2 only at q = 1. At the paper's operating points q ranges from ~1.5 (d=10) to ~0.16 (d=100), so E[tanh']^2 ranges 0.28 to 0.77 versus mu_1^2 = 0.367 — a factor-2 correction on top of the psi_p -> d_lat fix. The qualitative conclusion (edges are p-independent for p > d_lat + n, and the signal edge falls as 1/d_lat so tau_gen grows with d_lat) is unaffected.


---

## [BLOCKING] "tau_gen is d_lat-independent" is false both under the corrected theorem and in the paper's own raw data, where it grows up to 51x across the sweep

`tau-gen-is-not-dlat-independent` · verdict **CONFIRMED** · kind unsupported-claim · effort hours  

**Where:** ICLR_2026/sec-rfnn.tex line 65 ("signal bulk first (defining tau_gen, d_lat-independent at fixed sigsig, k, cluster scale) ... while tau_gen stays fixed"); ICLR_2026/sec-intro.tex line 26; ICLR_2026/sec-appendix.tex lines 573-574 and 588-591; sec-rfnn-bounds.tex lines 64-68

**Claim:** With the corrected signal edge mu_1^2 alpha_t^2/d_lat, tau_gen = d_lat/(mu_1^2 alpha_t^2) grows linearly in d_lat. The paper's own logged runs confirm this: tau_gen, computed with the paper's own definition (first eval step at which test loss is within 5% of its per-run minimum), grows monotonically and dramatically with d_lat. This destroys the "free lunch" framing: whatever delays memorization also delays generalization.

**Evidence:** Recomputed from the raw metrics.jsonl with the paper's definition (test_loss within 5% of run minimum, in optimizer steps):
 sigma_noise_0.01/exp2_rfnn: d_lat = 8,10,15,20,30,40 -> tau_gen = 90k, 120k, 170k, 210k, 255k, 275k steps (3.1x over a 5x d_lat range).
 sigma_noise_0.01/exp2_mlp: d_lat = 5,8,10,15,20,30,40,50,100,150,200 -> tau_gen = 5k, 15k, 10k, 10k, 25k, 35k, 55k, 110k, 160k, 255k steps (51x).
 sigma_noise_0.5/exp2_rfnn: tau_gen = 1, 1, 5k, 5k, 5k, 5k, 10k, 10k, 10k (monotone increase, resolution-limited by the 5k eval interval).
Direct spectral confirmation on the measured U (sigma_perp=0.5): 1/lambda_min^signal = 20.3, 30.5, 45.6, 96.2 for d_lat = 10, 20, 40, 100 -- a 4.7x increase, not a constant. Independent cross-check that this is the right clock: for lr = 0.01 d_lat/Delta_t (code/experiment_v2_rfnn.py line 236) the gradient-flow time per step is exactly 0.02 U-units, d_lat-independent, so predicted step-to-generalization = 50*log(1/eps)/lambda_min^signal; with a single fitted log(1/eps) ~ 10 this gives 10.2k, 15.3k, 22.8k, 48.1k steps for d_lat = 10, 20, 40, 100, against observed 5k, 15k, 25k, 55k.

**Fix:** Remove every claim that tau_gen is d_lat-independent from the abstract, intro, sec-rfnn, and the appendix statement. Report the measured tau_gen(d_lat) curves. If a delay claim survives at all it must be stated as a claim about the RATIO tau_mem/tau_gen or about tau_mem at matched sample quality, not about an unchanged tau_gen.


**Referee:** Both halves check out. Theory: with the corrected signal edge mu_1^2 alpha_t^2/d_lat (alpha_t^2 has no d_lat), tau_gen grows linearly in d_lat. Spectrally I measure 1/lambda_min^signal = 21.0, 28.0, 45.6, 95.4 for d_lat = 10, 20, 40, 100 at sigma_perp=0.5 — a 4.6x rise, not a constant. Empirically, recomputing the paper's own definition (first eval step with test_loss <= 1.05 * run minimum) from the raw metrics.jsonl I get exactly the reviewer's numbers: sigma_noise_0.01/exp2_rfnn d_lat=8..40 -> 90k, 120k, 170k, 210k, 255k, 275k; sigma_noise_0.01/exp2_mlp d_lat=5..200 -> 5k, 15k, 10k, 10k, 10k, 25k, 35k, 55k, 110k, 160k, 255k (51x); sigma_noise_0.5/exp2_rfnn -> 1, 1, 5k, 5k, 5k, 5k, 10k, 10k, 10k. I also independently confirmed the step<->flow-time conversion the cross-check relies on: with lr = 0.01*d_lat/Delta_t and loss = ||sqrt(Delta) A phi + eta||^2/(d n), dA/dstep = -lr*(2 Delta/d) A U = -0.02 A U, so flow time advances 0.02 U-units per step independent of d_lat. So the claims at sec-rfnn.tex line 65, sec-intro.tex line 26, sec-appendix.tex 573-574/588-591 and sec-rfnn-bounds.tex 64-68 are unsupported.


**Referee correction:** Two qualifications the fix list should absorb. (1) The reviewer's cross-check series 'observed 5k, 15k, 25k, 55k' does not correspond to any single logged sweep I could locate (sigma_noise_0.5/exp2_rfnn gives 1, 5k, 5k, 10k at d=10/20/40/100); the spectral evidence and the sigma_perp=0.01 sweeps carry the claim on their own, so drop that particular comparison. (2) There is one sweep where tau_gen genuinely looks flat: multiseed_runs/exp2_mlp_dlat_sn05_5m_bonnaire_hidden256 (5 seeds, d_lat=5..40, 5M steps) has tau_gen = 50,000 for every run — but that is the first eval after step 1 (eval_interval=50,000) and the test loss rises monotonically afterwards, so tau_gen there is the argmin at the coarsest possible resolution, not evidence of d_lat-independence. Say so explicitly rather than letting that sweep be read as support.


---

## [BLOCKING] tau_gen and tau_mem are never defined in the theory; the empirical proxies are step-count threshold crossings of generated-sample statistics with no derivation connecting them to any eigenvalue

`no-bridge-lemma-tau-definitions` · verdict **CONFIRMED** · kind missing-theory · effort weeks  

**Where:** ICLR_2026/sec-theory.tex (entire file is a \todo stub); ICLR_2026/sec-appendix.tex lines 26-32 and 88-107 (empirical definitions); Eq. (buffer-bound) line 578

**Claim:** Eq. (buffer-bound) is an inequality between two symbols that have no definition in the theory. Empirically tau_gen is "first step at which test loss is within 5% of its per-run minimum" and tau_mem is "first step at which the Somepalli memorization fraction crosses 1%" (or, in every synthetic RFNN run and every real-data run, the gen-gap > 0.02 proxy). Nothing in the paper connects "eigenmode i has been absorbed to tolerance eps" to "x% of generated samples are memorized", or even to "the gen-gap has reached 0.02". Without that bridge, the corollary is not a theorem about anything measured.

**Evidence:** sec-theory.tex: "\todo{TODO: formalize the eigenmode learning story here. This section should state the RFNN gradient-flow timescale tau_i = 1/lambda_i, define where tau_gen and tau_mem sit in the ordered spectrum...}" -- i.e. the paper itself records that the definitions are missing. _next_steps/theory_plan.md, the plan's task, step 3: "some care: memorization is about *fitting* sample modes, generalization is about fitting signal modes. make sure the definitions line up with what we measure empirically" -- flagged and never done. Note also a units mismatch that the missing bridge hides: 1/lambda_i is a gradient-flow time in U-units, whereas the measured tau's are optimizer step counts; for the RFNN these are related by the undocumented factor 0.02 per step, and for the MLP/real-data runs (Adam, batch 256, t ~ Unif[0.01,3]) by nothing at all.

**Fix:** Write the bridge lemma explicitly. Minimally it must (i) express the gen-gap at flow time T as a mode sum, G(T) = sum_i g_i (1 - e^{-lambda_i T})^2 with g_i the target's projection on mode i, decomposed into population and sample modes; (ii) show that the sample-mode contribution dominates G above a threshold and give the T at which G crosses theta as a function of lambda_sample, the sample-mode target mass, and theta; (iii) separately, for the generated-sample memorization fraction, bound the Somepalli nearest-neighbour statistic by the score error, which is a sum over ALL modes -- and note that this is the only route by which noise-dim modes could legitimately gate a memorization observable, and that even then the gate is max_i 1/lambda_i over the bulk, i.e. 1/lambda_min^{noise-dim}, with no count factor. Finally, state the flow-time-to-step conversion for each experimental track.


**Referee:** ICLR_2026/sec-theory.tex is 8 lines and entirely a \todo: 'TODO: formalize the eigenmode learning story here. This section should state the RFNN gradient-flow timescale tau_i = 1/lambda_i, define where taugen and taumem sit in the ordered spectrum...' — the paper itself records that the definitions are missing. The only definitions anywhere are operational (sec-appendix.tex lines 26-32 and 88-107): tau_gen = first step within 5% of the per-run test-loss minimum; tau_mem = first step the Somepalli fraction crosses 1%, with the gen-gap > 0.02 proxy substituted for the RFNN and for real data (lines 411-424). Nothing connects 'mode i absorbed to tolerance eps' to either observable, so Eq. (buffer-bound) is an inequality between undefined symbols. _next_steps/theory_plan.md line 79 flags exactly this ('some care: memorization is about *fitting* sample modes, generalization is about fitting signal modes. make sure the definitions line up with what we measure empirically') and it was never done. The units mismatch is real and I verified the RFNN conversion factor by deriving it from the code: 1/lambda_i is flow time in U-units, the logged taus are optimizer steps, and for the RFNN the map is T = 0.02 * step; for the MLP and real-data tracks (Adam, batch 256, t ~ Unif[0.01,3]) there is no stated map at all.


**Referee correction:** One point in the proposed fix is stronger than the finding realizes and worth stating: the corpus contains a direct test of the proxy's validity that the paper does not report. In multiseed_runs/exp2_mlp_dlat_sn05_5m_bonnaire_hidden256, the canonical Somepalli tau_mem runs 150k -> 250k -> 300k -> 400k -> 600k -> 1050k -> 2550k for d_lat = 5..25, while the gen-gap > 0.02 proxy fires at the first eval (50k) for every single d_lat. So on the one sweep where both can be compared, the proxy is flat where the canonical metric varies 17x — which falsifies the appendix's own justification at lines 103-107 ('monotonically tracks the canonical taumem on the MLP').


---

## [BLOCKING] Theorem's bulk-edge prefactor 1/psi_p is algebraically wrong; it should be 1/d_lat

`edge-prefactor-psip-vs-dlat` · verdict **CONFIRMED** · kind error · effort hours  

**Where:** ICLR_2026/sec-appendix.tex:694-700 (Lemma \ref{lem:Ulin} + proof), theorem table lines 797-802; identical text in /Users/ryan/Desktop/latent_space_diffusion_analysis/sec-appendix-fourbulk.tex (Lemma 1) and sec-rfnn-bounds.tex (Thm restatement)

**Claim:** Lemma 1 states the nonzero spectrum of U^lin = (mu_1^2/(p*d_lat)) W M_t W^T is 'the spectrum of M_t rescaled by mu_1^2/psi_p'. With W having i.i.d. N(0,1) entries (as the lemma's own line E[W^T W/d_lat] = psi_p I asserts), W^T W ~ p*I, so eig(W M_t W^T) = p*lambda(M_t) and U^lin's edges are mu_1^2*lambda(M_t)/d_lat, not mu_1^2*lambda(M_t)/psi_p. The proof divides by d_lat twice. The stated edge is therefore too large by a factor d_lat^2/p = d_lat/psi_p, i.e. it is wrong by a factor that grows linearly with the very variable the paper sweeps.

**Evidence:** Exact algebra: U^lin = (mu_1^2/(p d_lat)) W M_t W^T; nonzero eig = (mu_1^2/(p d_lat)) * eig(M_t^{1/2} W^T W M_t^{1/2}) = (mu_1^2/(p d_lat)) * p * lambda(M_t) = mu_1^2 lambda(M_t)/d_lat. Numerically, from the code's un-normalized U (code/experiment_v2_rfnn.py:111-128 computes U=(1/n)sum phi phi^T with phi=tanh(x_t W^T), W~N(0,1/d), NO 1/p, so U_code = p*U_theory), signal-bulk median vs theorem prediction d_lat*mu_1^2*alpha_t^2 (sigma_perp=0.5, sigma_noise_0.5/exp2_rfnn/raw_data/*/eigenvalues_pre.npy): d_lat=5: 29.66 vs 4.34 (ratio 6.83); 10: 40.59 vs 8.69 (4.67); 20: 56.58 vs 17.46 (3.24); 40: 72.84 vs 35.0 (2.08); 60: 77.76 vs 52.8 (1.47); 100: 87.64 vs 89.1 (0.98); 150: 96.67 vs 135.3 (0.71); 200: 99.80 vs 182.3 (0.55). The ratio spans 12x and crosses 1 exactly at d_lat ~ psi_p = 64, which is the signature of the missing d_lat/psi_p factor. The alternative prefactor psi_p*mu_1^2*alpha_t^2 (= the 1/d_lat form in code units) gives a nearly constant 55.6-58.3 with ratio range 0.53-1.71, i.e. 3.2x instead of 12x.

**Fix:** Replace 1/psi_p by 1/d_lat in Lemma 1, in the theorem table (signal and noise-dim rows), in the buffer bound Eq. (buffer-bound), and everywhere the edge magnitudes are quoted. Fix the Lemma-1 proof sentence 'eigenvalues of M_t rescaled by psi_p' -> 'rescaled by p'. Note the consequence: the signal edge becomes explicitly 1/d_lat, so tau_gen ~ d_lat and is NOT d_lat-independent.


**Referee:** I reproduced the algebra and the numbers independently.

Text: /Users/ryan/Desktop/latent_space_diffusion_analysis/ICLR_2026/sec-appendix.tex:691-707 (identical at sec-appendix-fourbulk.tex Lemma 1 and sec-rfnn-bounds.tex Thm restatement). Eq. (U-lin) is U^lin = (mu_1^2/(p*d_lat)) W M_t W^T. Definitions are consistent between paper and code: sec-rfnn.tex Eq. (rfnn) has phi = (1/sqrt p) tanh(Wx/sqrt d) with W~N(0,I); code/experiment_v2_rfnn.py:73-78 has W~N(0,1/d), phi=tanh(x W^T), output/sqrt(p); compute_U (lines 111-124) omits the 1/p, so U_code = p*U_theory exactly.

The error is a single wrong step in the proof. E[W^T W/d_lat] = psi_p*I is correct, but the proof then applies it to the UN-normalized W M_t W^T, concluding 'eigenvalues of M_t rescaled by psi_p'. W^T W concentrates at p*I, not psi_p*I, so nonzero eig(U^lin) = (mu_1^2/(p d)) * p * lambda(M_t) = mu_1^2 lambda(M_t)/d_lat. The stated edge is too large by d_lat^2/p = d_lat/psi_p.

Numerics (my own, from sigma_noise_0.5/exp2_rfnn/raw_data/*/eigenvalues_pre.npy; alpha_t^2 = 2.7644 at t=0.01, s=3, d_int=5, sigsig=1; mu_1 = E[z tanh z] = 0.60542, mu_1^2 = 0.36653): measured signal-bulk median = 29.66, 40.59, 56.58, 72.84, 77.76, 86.94, 87.64, 96.67, 99.80 for d_lat = 5..200 (exactly the reviewer's values). Theorem prediction in code units (d_lat*mu_1^2*alpha^2) gives ratio measured/predicted = 5.85, 4.00, 2.79, 1.80, 1.28, 1.07, 0.86, 0.64, 0.49 — an 11.9x span crossing 1 near d_lat ~ psi_p. The 1/d_lat form (psi_p*mu_1^2*alpha^2 = 64.9, constant) gives 0.46-1.54, a 3.4x span with no trend reversal. Same signature in the noise-dim bulk: published form spans 5.4x-too-big to 0.47x-too-small (11.4x), the 1/d_lat form does not.

The consequence the reviewer draws is also right: with 1/d_lat, tau_gen ~ d_lat, which removes the 'tau_gen is d_lat-independent' claim that the buffer corollary rests on.


**Referee correction:** Two scope corrections, neither of which rescues the formula. (1) 'The proof divides by d_lat twice' is loose; the single defective step is applying the E[W^T W/d_lat] = psi_p normalization to the un-normalized W M_t W^T. (2) These formulas are currently in sec-appendix.tex, sec-appendix-fourbulk.tex and sec-rfnn-bounds.tex, none of which is compiled — ICLR_2026/main.tex:137 inputs sec-appendix-integrated.tex, which contains zero theorem/lemma environments and none of the psi_p edges. So the error is not yet in the built PDF; it is blocking for whatever gets written into the sec-theory.tex stub.


---

## [BLOCKING] Sample-bulk edge eta_star/psi_p has no n-dependence and is wrong by 12-215x in the repo's own data

`sample-bulk-edge-missing-n` · verdict **CONFIRMED** · kind error · effort days  

**Where:** ICLR_2026/sec-appendix.tex:713-722 (Lemma \ref{lem:Udiag}) and line 801 (theorem table 'Sample & n & \bar\eta_\star/\psi_p'); same in sec-appendix-fourbulk.tex Lemma 2

**Claim:** The sample-bulk edge is derived as bar_eta_star/psi_p, independent of n. The exact Gram-matrix argument gives bar_eta_star/n. Since U's nonzero spectrum equals that of K/n where K_{mu,nu} = phi_mu . phi_nu = p[c1^2 x_mu.x_nu/d_lat + delta_{mu,nu} eta_star], the diagonal part contributes (p/n)*eta_star in code units, i.e. eta_star/n in theory units. The theorem's value is too large by n/psi_p = 500/64 = 7.8 and, worse, carries the wrong d_lat trend.

**Evidence:** Predicted vs measured sample-bulk median (code units), sigma_noise_0.5/exp2_rfnn: theorem d_lat*eta_bar vs measured -- d_lat=5: 0.387 vs 0.00180 (215x too big); 10: 0.437 vs 0.00370 (118x); 40: 0.485 vs 0.0294 (16.5x); 200: 0.887 vs 0.0708 (12.5x). Corrected form p*eta_bar/n: 0.0495 vs 0.00180, 0.0560 vs 0.00370, 0.0620 vs 0.0294, 0.1136 vs 0.0708. Using the per-sample refinement lambda_mu = (p/n)*eta_star(q_mu) with q_mu = (e^{-2t}||x_mu||^2 + Delta_t d)/d and comparing bulk MEANS, measured/predicted = 0.98 (d=5), 0.69 (10), 0.74 (20), 0.76 (40), 0.76 (60), 0.72 (100), 0.59 (200) -- i.e. correct to within a factor 1.0-1.7, versus 12-215x for the published formula.

**Fix:** Re-derive Lemma 2 from the n x n Gram matrix K/n rather than the p x p outer-product sum. State the sample edge as bar_eta_star/n (theory units). Then redo the 'Bulk-gap scaling' paragraph: the noise-dim-to-sample ratio becomes c1^2 beta_t^2 * psi_n / bar_eta_star, i.e. it depends on psi_n = n/d_lat, which the current text omits entirely.


**Referee:** Verified, and the defect is worse than stated: Lemma 2's own proof contradicts its own statement.

sec-appendix.tex:713-736 (= sec-appendix-fourbulk.tex Lemma 2). The lemma claims the nonzero eigenvalues of U^diag concentrate at bar_eta_star/psi_p. Its proof sets Lambda_mu = eta_star/(p*n) and computes the eigenvalues as Lambda_mu*||phi^perp||^2 -> eta_star^2/(p*n) — which is neither the lemma's own bar_eta/psi_p nor the correct answer. Eq. (U-diag) itself carries a spurious extra eta_star factor relative to U = (1/n) sum phi phi^T with phi already carrying 1/sqrt p.

Correct derivation (mine): U's nonzero spectrum equals that of K/(n p) with K_{mu,nu} = tanh_mu . tanh_nu. Writing tanh_mu = c1(W x_mu/sqrt d) + r_mu with ||r_mu||^2 ~ p*eta_star gives residual eigenvalues eta_star/n in theory units, i.e. p*eta_star/n in code units. No 1/psi_p, and an explicit 1/n.

Numerics (mine, sigma=0.5, eta_star = E[tanh^2(sqrt q z)] - c1(q)^2 q): theorem value d_lat*bar_eta = 0.389, 0.442, 0.462, 0.488, 0.527, 0.573, 0.623, 0.758, 0.898 versus measured sample-bulk medians 0.00180, 0.00370, 0.01446, 0.02941, 0.03834, 0.04487, 0.05041, 0.06218, 0.07077 — too large by 216x down to 12.7x. Corrected p*eta/n gives 0.0498...0.1149, i.e. within a factor 1.6-2.7 for d_lat>=40. The reviewer's per-sample mean refinement is not something I re-ran, but the two-orders-of-magnitude failure of the published form and the correctness of the 1/n form are both solid.


**Referee correction:** Add: the small-d_lat entries (d_lat = 5, 10) are not a fair test of any sample-edge formula because p = 320, 640 is not much larger than n = 500, so the 'sample bulk' index window d_lat..d_lat+n is truncated or overlaps the tail; the clean comparisons are d_lat >= 40, where the published form is still 12-17x too large. Also note the deeper issue: with the corrected edge eta_star/n = eta_star/(psi_n d_lat), the sample edge is o(1) in the stated proportional limit, so it is no longer separated in ORDER from the 'o(1)' rank-null bulk — the theorem's four-way edge hierarchy does not survive its own asymptotics.


---

## [BLOCKING] mu_1^2 = 0.367 is not a constant here: the effective Hermite-1 coefficient varies 3.5x across the d_lat sweep

`mu1-is-q-dependent` · verdict **CONFIRMED** · kind missing-theory · effort hours  

**Where:** ICLR_2026/sec-appendix.tex:640-660 (Hermite expansion paragraph, Eq. kernel-decomp) and eigenvalue_saturation/README.md

**Claim:** Eq. (kernel-decomp) uses the Hermite coefficients of tanh against a STANDARD normal, so mu_1^2 = 0.367 is treated as a constant. But the preactivation is w.x_t with w ~ N(0, I/d_lat), so its variance is q = ||x_t||^2/d_lat, which is not 1 and varies systematically over the sweep. The correct linear coefficient is c1(q)^2 = (E[tanh'(sqrt(q) z)])^2. Since the whole paper is a d_lat sweep at fixed n and fixed data scale, q drifts monotonically and c1(q)^2 drifts with it, so the 'constant mu_1^2' assumption fails exactly along the swept axis.

**Evidence:** q = tr(M_t)/d_lat measured from the actual generated data (code/experiment_v2.py generate_data, seed 42, sigma_perp=0.5): 2.752 (d_lat=5), 1.507 (10), 0.889 (20), 0.578 (40), 0.390 (100), 0.329 (200). Corresponding c1(q)^2 = 0.181, 0.282, 0.393, 0.493, 0.586, 0.624, i.e. c1(q)^2/mu_1^2 = 0.49, 0.77, 1.07, 1.34, 1.60, 1.70 -- a 3.5x drift, equal to 0.367 only near d_lat ~ 25. Substituting c1(q)^2 for mu_1^2 AND 1/d_lat for 1/psi_p, the predicted signal edge psi_p*c1(q)^2*alpha_t^2 matches the measured signal-bulk median to within 8% at every point: 27.38 vs 29.66, 42.75 vs 40.59, 59.88 vs 56.58, 75.3 vs 72.84, 82.98 vs 77.76, 87.45 vs 86.94, 90.94 vs 87.64, 96.12 vs 96.67, 99.23 vs 99.80 (d_lat = 5,10,20,40,60,80,100,150,200). Same for the noise-dim bulk at sigma_perp=0.5: 4.76 vs 5.215, 6.56 vs 7.034, 8.13 vs 8.36, 8.87 vs 9.011, 9.22 vs 9.113, 9.17 vs 9.316, 9.39 vs 9.348, 9.23 vs 9.216 (ratios 0.99-1.10).

**Fix:** Replace mu_1^2 by c1(q_t)^2 = (E_z[tanh'(sqrt(q_t) z)])^2 with q_t = tr(M_t)/d_lat throughout, and state q_t explicitly as a function of d_lat, d_int, s, sigma_sig, sigma_perp, t. This is a two-line change that turns a 12x-error formula into a 6%-accurate one and should be stated as the theorem's edge, with mu_1^2 recovered only in the q -> 1 limit.


**Referee:** Verified independently, including the derivation.

sec-appendix.tex:640-660 defines mu_k = E_{z~N(0,1)}[h_k(z) tanh(z)] and then writes Eq. (kernel-decomp) with linear coefficient mu_1^2 * x.y/d_lat. But the preactivation is (Wx)_a/sqrt(d_lat), whose variance is q = ||x||^2/d_lat. Doing the expansion correctly: with u = sqrt(q) z1, v = sqrt(q) z2 at correlation r = rho/q, the k=1 coefficient is a_1(q) = E[z tanh(sqrt q z)] = sqrt(q) E[tanh'(sqrt q z)], so the k=1 term is a_1(q)^2 r = c1(q)^2 * x.y/d_lat with c1(q) = E[tanh'(sqrt q z)]. mu_1^2 is the q=1 special case. The appendix is internally inconsistent about this: it lets the k>=3 terms carry q explicitly (eta_star = sum_{k>=3} mu_k^2 q^k) while freezing k=1 at mu_1^2.

q measured from the actual construction (code/experiment_v2.py:90-140: centers have norm exactly scale=3, so E||x||^2 = s^2 + d_int*sigsig^2 + (d_lat-d_int)*sigperp^2): q = 2.764, 1.515, 0.890, 0.577, 0.473, 0.421, 0.390, 0.348, 0.327 for d_lat = 5..200 — an 8.4x drift along the swept axis, passing through 1 near d_lat ~ 25. c1(q)^2 = 0.180, 0.281, 0.393, 0.494, 0.541, 0.568, 0.586, 0.612, 0.625, i.e. c1^2/mu_1^2 = 0.49 to 1.71, a 3.5x drift (reviewer's numbers reproduce).

The corrected edge psi_p*c1(q)^2*alpha_t^2 tracks the measured noise-dim bulk to within 1-13% (predicted 4.76, 6.66, 8.37, 9.17, 9.63, 9.93, 10.37, 10.60 vs measured 5.22, 7.03, 8.36, 9.01, 9.11, 9.32, 9.35, 9.22). The independent check tr(U_code)/p = E[tanh^2(sqrt q z)] holds to 1-5% at every point, confirming q is the controlling parameter.


**Referee correction:** One number to soften: I get the corrected signal-bulk prediction matching to within 7-19% (ratio 0.81-0.93 across the sweep), not 'within 8% at every point'. The reviewer's tighter agreement comes from using the empirical top-block eigenvalues of M_t in place of the population alpha_t^2. The material claim — that the residual is a nearly CONSTANT offset instead of a 12x trend — holds either way.


---

## [BLOCKING] The measured mode clock says the buffer CLOSES as d_lat grows: tau_mem/tau_gen falls 7.8x and tau_mem is U-shaped

`buffer-shrinks-not-grows` · verdict **REFUTED** · kind error · effort weeks  

**Where:** ICLR_2026/sec-appendix.tex:576-591 (Eq. buffer-bound); ICLR_2026/sec-rfnn.tex:65; abstract.tex:2

**Claim:** The paper claims tau_mem - tau_gen >= (d_lat-d_int)*psi_p/(mu_1^2 beta_t^2) with tau_gen fixed, so that widening d_lat monotonically delays memorization. Applying the paper's own clock tau_i = 1/lambda_i to the repo's own measured eigenvalues gives the opposite: the dimensionless separation tau_mem/tau_gen = lambda_signal/lambda_sample SHRINKS monotonically with d_lat, and the absolute tau_mem in optimizer steps is U-shaped with a minimum near d_lat = 40.

**Evidence:** From sigma_noise_0.5/exp2_rfnn/raw_data/*/eigenvalues_pre.npy, lambda_signal/lambda_sample (bulk medians): 10975 (d_lat=10), 3913 (20), 2477 (40), 2028 (60), 1938 (80), 1739 (100), 1555 (150), 1410 (200) -- a 7.8x SHRINK. lambda_noise-dim/lambda_sample: 1410, 486, 284, 235, 203, 185, 150, 130 -- a 10.8x shrink. Same at sigma_perp=0.01: 57750 (d_lat=8) -> 26011 (30) -> 29468 (40). In steps (rate = 2*lr*Delta_t*lambda_code/(d*p) with lr = 0.01*d/Delta_t from code/experiment_v2_rfnn.py:236, so tau_i = 50p/lambda_i^code): tau_gen = 788, 1131, 1757, 2469, 2945, 3651, 4965, 6413 (8.1x GROWTH, contradicting 'tau_gen is d_lat-independent'), tau_mem = 8.65e6, 4.43e6, 4.35e6, 5.01e6, 5.71e6, 6.35e6, 7.72e6, 9.04e6 (U-shaped; going from d_lat=10 to d_lat=40 HALVES tau_mem). The published bound itself is violated at d_lat=200: tau_mem - tau_gen = 9.04e6 < (d_lat-d_int)/lambda_nd = 1.354e7.

**Fix:** Withdraw Eq. (buffer-bound) as stated. Two separate problems must be fixed: (a) the count factor (d_lat-d_int) presumes SERIAL mode absorption, but gradient flow absorbs all modes in parallel, so the traversal time is 1/lambda_sample - 1/lambda_signal, which contains no count; (b) with the corrected edges, tau_gen ~ d_lat/(c1(q)^2 alpha_t^2) grows linearly in d_lat, at the same order as the claimed buffer growth, so the buffer does not open relative to tau_gen. Re-derive what the spectrum actually predicts (a U-shaped tau_mem) and reconcile with the MLP evidence.


**Referee:** The headline is an artifact of the reviewer's choice of statistic, and it reverses under the statistic the paper's own language calls for.

The paper defines tau_mem as the ONSET of sample-specific fitting ('sample bulk last (defining taumem)', sec-rfnn.tex:65; 'the moment when null structure BEGINS to be learned', sec-rfnn.tex:69; abstract: 'before sample-specific memorization BEGINS'). Onset corresponds to the LARGEST sample-bulk eigenvalue, not its median. The reviewer used the median, i.e. the moment half the sample bulk has already been absorbed.

With tau_mem = 50p/lambda_sample_max (same clock, same runs), sigma_perp = 0.5: tau_gen = 539, 788, 1131, 1757, 2469, 2945, 3651, 4965, 6413 and tau_mem = 1.67e4, 3.44e4, 9.59e4, 3.42e5, 6.69e5, 1.14e6, 1.61e6, 2.85e6, 3.69e6 for d_lat = 5..200. tau_mem grows 221x and tau_mem/tau_gen grows MONOTONICALLY from 31 to 576 — an 18.6x OPENING, not a 7.8x shrink. Same at sigma_perp = 0.01: ratio 31, 50, 66, 109, 111, 127, 158 monotone. The 90th-percentile statistic is also monotone. Only the median produces the reviewer's U-shape.

So 'the buffer CLOSES as d_lat grows' and 'tau_mem is U-shaped' are not properties of the spectrum; they are properties of measuring memorization at the middle of the sample bulk.


**Referee correction:** Two sub-claims inside this finding do survive and should be re-filed separately.

(a) The count factor in Eq. (buffer-bound) (sec-appendix.tex:576-591; sec-rfnn-bounds.tex) is a real derivation error. Eq. (rfnn-mode-decay) makes every mode decay independently as exp(-lambda_i T); there is no queue, so 'width of the bulk times the slowest timescale in it' has no dynamical basis. The traversal time is 1/lambda_sample - 1/lambda_signal, which contains no count.

(b) The bound as written is quantitatively false, and more broadly than the reviewer found. Comparing (d_lat-d_int)/lambda_nd against tau_mem - tau_gen using the ONSET statistic (all in the same units), the bound is violated at every d_lat >= 20: d_lat=20, 1.37e5 vs 9.5e4; d_lat=40, 5.4e5 vs 3.4e5; d_lat=100, 3.26e6 vs 1.61e6; d_lat=200, 1.35e7 vs 3.69e6 (1.4x to 3.7x). The reviewer's single d_lat=200 violation under the median statistic is the weakest version of this.

Corrected finding: Eq. (buffer-bound) must be withdrawn as a quantitative bound (bogus count factor, violated at every d_lat >= 20), but the QUALITATIVE conclusion — the separation between generalization and memorization opens monotonically with d_lat — is supported by the repo's own spectra.


---

## [BLOCKING] There is no spectral boundary at index d_lat + n; the third boundary is at n, and the 'rank-null' bulk count p - d_lat - n is wrong

`fourth-bulk-boundary-does-not-exist` · verdict **CONFIRMED** · kind unsupported-claim · effort days  

**Where:** ICLR_2026/sec-appendix.tex:565-570 (counts) and line 802 ('Rank-null & p-d_lat-n & o(1)'); ICLR_2026/sec-rfnn-bounds.tex theorem restatement

**Claim:** The theorem asserts four CONTIGUOUS bulks with the fourth beginning at index d_lat + n and having edge o(1). Blind boundary detection (largest single-step drops in log-eigenvalue, assuming nothing) never finds a boundary at d_lat + n. The contrast ratio at that index is 1.00. The only feature in that region sits at index exactly n = 500 (independent of d_lat), which is the fixed-noise rank of the n x p feature matrix, so the sample bulk has count n - d_lat and the tail has count p - n.

**Evidence:** Contrast r(k) = lambda_{k-1}/lambda_k at the claimed boundary d_lat+n, sigma_noise_0.5/exp2_rfnn: 1.003 (d_lat=10), 1.008 (20), 1.001 (40), 1.009 (60), 1.001 (80), 1.006 (100), 1.005 (150), 1.003 (200); sigma_noise_0.01: 1.178, 1.007, 1.012, 1.008, 1.005, 1.002. Meanwhile r(n=500) reaches 1.32 (d_lat=40), 1.70 (60), 2.08 (80), 1.68 (100), 1.90 (150), 1.73 (200) -- the kink is at n, not d_lat+n. Raw values at sigma=0.5, d_lat=40 (p=2560): lambda[499]=0.006807, lambda[500]=0.005149 (24% drop) but lambda[539]=0.003546, lambda[540]=0.003541, lambda[541]=0.003534 -- perfectly smooth through d_lat+n=540. The repo's own bulk_summary.json confirms it: gap_sample_to_null_dec = 0.0005-0.008 decades (factor 1.001-1.02) in 32 of 34 runs. The tail is also not o(1): at d_lat=40, sigma=0.5, lambda[1000] = 0.00139, only 21x below the sample-bulk median, decaying as a smooth power law with no zero plateau.

**Fix:** Recount: rank(U) at fixed diffusion noise is at most n, so the three non-null bulks have counts d_int, d_lat-d_int, n-d_lat, and the tail has count p-n. Explain the smooth tail as an artifact of averaging U over n_noise_samples=50 diffusion draws (code/experiment_v2_rfnn.py:117-127), which lifts the strict rank-n structure into a power-law tail. Drop the 'o(1) rank-null edge' claim or replace it with the measured decay exponent.


**Referee:** I reproduced every number and the conclusion holds.

Contrast r(k) = lambda_{k-1}/lambda_k at the theorem's claimed third boundary d_lat+n (sec-appendix.tex:565-570, 802): sigma=0.5 gives 1.003, 1.008, 1.001, 1.009, 1.001, 1.006, 1.005, 1.003 for d_lat = 10..200; sigma=0.01 gives 1.179, 1.007, 1.012, 1.008, 1.005, 1.002. Every value is ~1.00 — no boundary. Raw spectrum at d_lat=40, sigma=0.5: lambda[537..543] = 0.003621, 0.003589, 0.003546, 0.003541, 0.003534, 0.003521, 0.003503 — perfectly smooth through d_lat+n=540, while lambda[499]=0.006807 -> lambda[500]=0.005149 is a real 24% step. A blind largest-log-gap detector returns {1,2,4,5,d_lat,500} at sigma=0.5 for d_lat>=40 and never returns d_lat+n at any setting.

The 'o(1) rank-null' edge is also unsupported: at d_lat=40, sigma=0.5, lambda[1000]=0.00139, only 21x below the sample-bulk median, and the region beyond d_lat+n is a smooth power law (log-log slope -1.9 at d_lat=40, -1.2 at d_lat=100/200) with no plateau.

The reviewer's diagnosis of the mechanism is also right and is a real hole in the theorem's proof: compute_U (code/experiment_v2_rfnn.py:111-124) averages over n_noise_samples=50 diffusion draws, and more importantly the THEORY's U is itself defined as E_xi[phi phi^T] (sec-rfnn.tex Eq. U). Under that expectation E_xi[phi^perp phi^perp^T] is a full-rank p x p matrix per sample, so the rank-n claim in Lemma 2 and the rank-(d_lat+n) claim in the theorem's proof (sec-appendix.tex:808-816) do not follow from the stated definition of U at all.

I also confirm bulk_summary.py's docstring hard-codes the boundaries at d_int, d_lat, d_lat+n ('Uses the cumulative-count definition'), so bulk_summary.json cannot be cited as independent confirmation of the counts.


**Referee correction:** Trim one over-general claim: 'the only feature in that region sits at index exactly n=500' is true only at sigma_perp=0.5 with d_lat >= 40 (r(500) = 1.32, 1.70, 2.08, 1.68, 1.90, 1.73). At d_lat=10, 20 and at every sigma_perp=0.01 run, r(500) is also ~1.00-1.02, so there is no boundary at n either. The safe statement is: there is no third spectral boundary at d_lat+n anywhere, and where a third boundary is detectable at all it sits at n, not at d_lat+n. Also note that the main text does not claim a cliff at d_lat+n (sec-rfnn.tex:59 claims cliffs only at d_int and d_lat) — the defect is confined to the theorem's counts and its rank/orthogonality proof.


---

## [BLOCKING] The paper's own empirical bulk-size table contradicts the theorem's counts, while the text claims they match

`bulk-counts-contradicted-by-own-table` · verdict **CONFIRMED** · kind inconsistency · effort days  

**Where:** ICLR_2026/sec-appendix.tex:171-193 (Table tab:bulk-sizes) vs the counts at lines 565-570; claim of agreement at /Users/ryan/Desktop/latent_space_diffusion_analysis/sec-rfnn-bounds.tex ('Predicted bulk counts match the empirical B1...B4 sizes in Table~\ref{tab:bulk-sizes} across the full d_lat sweep at the predicted indices')

**Claim:** Table tab:bulk-sizes reports counts from an independent density-peak detector. Its B3 (noise-dim) and B2 (sample) sizes do not match the theorem's d_lat - d_int and n at any d_lat, and the mismatch is large and systematic. The claim of agreement in sec-rfnn-bounds.tex is false against the paper's own table.

**Evidence:** Table (sigma_perp=0.01, d_int=5, n=500, p=64 d_lat). B3 measured vs predicted d_lat-5: 27 vs 0 (d_lat=5), 30 vs 3 (8), 35 vs 5 (10), 40 vs 10 (15), 48 vs 15 (20), 60 vs 25 (30), 70 vs 35 (40). The residual B3-(d_lat-5) is 27,27,30,30,33,35,35 -- an offset of ~30 that never goes away, including at d_lat=d_int where the theorem predicts a bulk of size zero. A linear fit gives B3 = 1.25*d_lat + 21.4, versus the predicted B3 = 1.00*d_lat - 5.0. B2 measured vs predicted 500: 138, 133, 305, 216, 311, 458, 578 -- non-monotone, spanning 0.27x to 1.16x of n. B1 measured vs predicted p-d_lat-n: 150 vs -185, 344 vs 4, 295 vs 130, 699 vs 445, 916 vs 760, 1397 vs 1390, 1907 vs 2020.

**Fix:** Either delete the agreement claim or replace the theorem's counts with the ones the detector actually finds. Also note that /Users/ryan/Desktop/latent_space_diffusion_analysis/bulk_summary.py hard-codes the bulk boundaries at d_int, d_lat, d_lat+n ('Uses the cumulative-count definition'), so every summary derived from it assumes the conclusion and cannot be cited as confirmation.


**Referee:** All arithmetic verified against Table tab:bulk-sizes (ICLR_2026/sec-appendix.tex:171-193) and the counts at lines 565-570.

B3 - (d_lat - d_int) = 27, 27, 30, 30, 33, 35, 35 for d_lat = 5, 8, 10, 15, 20, 30, 40 — a persistent ~30 offset that does not shrink, and is 27 at d_lat = d_int where the theorem predicts exactly zero. My least-squares fit gives B3 = 1.250*d_lat + 21.44 versus the predicted 1.000*d_lat - 5.0.

B2 vs the predicted n = 500: 138, 133, 305, 216, 311, 458, 578 — non-monotone, spanning 0.27x to 1.16x.

B1 vs predicted p - d_lat - n: 150 vs -185, 344 vs 4, 295 vs 130, 699 vs 445, 916 vs 760, 1397 vs 1390, 1907 vs 2020. Only d_lat = 30 matches.

Independent corroboration that the detector's B3 boundary is not at d_lat: my blind log-gap detector on the same sigma=0.01 spectra places its third boundary at index 45 (d_lat=15), 53-54 (20), 64-65 (30), 75 (40) — exactly d_int + B3 from the table, i.e. ~d_lat + 30, never d_lat.

The agreement claim in /Users/ryan/Desktop/latent_space_diffusion_analysis/sec-rfnn-bounds.tex ('Predicted bulk counts match the empirical B1...B4 sizes in Table~\ref{tab:bulk-sizes} across the full dlat sweep at the predicted indices') is false for B1, B2 and B3; only B4 = d_int = 5 matches. sec-appendix.tex:840-841 makes a parallel false agreement claim about edge magnitudes ('match within Marchenko-Pastur fluctuations'), which findings 1 and 2 show fail by 12x-216x.


**Referee correction:** The table caption itself (sec-appendix.tex:174-177, 'B3 grows linearly with dlat - dint') is defensible as written — B3 does grow linearly in d_lat (slope 1.25). The false claim is the equality of counts asserted in sec-rfnn-bounds.tex and the edge-magnitude agreement asserted in sec-appendix.tex:840-841. Also worth stating: the ~30 offset is a stable, reproducible feature that a corrected theory should explain rather than merely be re-fit to.


---

## [BLOCKING] tau_gen is not d_lat-independent: the measured signal-bulk eigenvalue moves 3.4x over the sweep

`taugen-not-dlat-independent` · verdict **CONFIRMED** · kind error · effort hours  

**Where:** ICLR_2026/sec-rfnn.tex:65 ('signal bulk first (defining taugen, d_lat-independent at fixed sigsig, k, cluster scale)'); sec-rfnn-bounds.tex buffer corollary; ICLR_2026/sec-appendix.tex:571-575

**Claim:** The claim that tau_gen is d_lat-independent rests entirely on the erroneous 1/psi_p prefactor (psi_p is held fixed at 64 in the sweep). With the correct 1/d_lat prefactor, and with the measured q-drift, the signal edge is strongly d_lat-dependent, so tau_gen grows roughly linearly in d_lat.

**Evidence:** Measured signal-bulk median (sigma_noise_0.5/exp2_rfnn): 29.66, 40.59, 56.58, 72.84, 77.76, 86.94, 87.64, 96.67, 99.80 for d_lat = 5,10,20,40,60,80,100,150,200 -- a 3.4x rise, not a constant. In optimizer steps, tau_gen = 50p/lambda_signal = 788, 1131, 1757, 2469, 2945, 3651, 4965, 6413 for d_lat = 10..200: an 8.1x rise over a 20x rise in d_lat (log-log slope 0.70, rising to 0.81 over the last octave, i.e. approaching linear). Corrected theory reproduces this: tau_gen ~ d_lat/(c1(q)^2 alpha_t^2) with c1(q)^2 saturating toward 1, hence tau_gen ~ d_lat asymptotically.

**Fix:** Remove the 'd_lat-independent tau_gen' claim from sec-rfnn.tex, the buffer corollary, and the abstract's framing. State instead tau_gen ~ d_lat/(c1(q_t)^2 alpha_t^2) and compare that growth rate directly against the growth of tau_mem; the paper's whole 'free delay' framing depends on this comparison, which currently is not made.


**Referee:** Verified directly, and it is the load-bearing half of the 'free delay' framing.

The claim appears verbatim at ICLR_2026/sec-rfnn.tex:65 ('signal bulk first (defining taugen, dlat-independent at fixed sigsig, k, cluster scale) ... while taugen stays fixed') and in the buffer corollary in sec-rfnn-bounds.tex ('is dlat-independent at fixed sigsig, s, dint'). It follows only from the erroneous 1/psi_p prefactor, which is d_lat-free because psi_p = 64 is held fixed by construction.

Measured signal-bulk median rises 3.4x (29.66 -> 99.80 over d_lat = 5..200). In gradient-flow time, tau_gen = 1/lambda_signal_theory = p/lambda_signal_code, i.e. 50p/lambda in the run's own optimizer steps: 539, 788, 1131, 1757, 2469, 2945, 3651, 4965, 6413 — an 11.9x rise over a 40x rise in d_lat (log-log slope ~0.68, steepening toward 1 as c1(q)^2 saturates). This is exactly what the corrected edge psi_p*c1(q)^2*alpha_t^2 predicts, and flatly contradicts 'd_lat-independent'.


**Referee correction:** Add the comparison the finding calls for but does not make, since it changes the verdict's weight: over the same sweep tau_mem (measured at the sample-bulk ONSET, 50p/lambda_sample_max) rises 221x, from 1.67e4 to 3.69e6. So tau_gen is not fixed, but it grows far more slowly than tau_mem and the separation still opens 18.6x. The correct rewrite is not 'the free-delay framing collapses' but 'the delay is not free: tau_gen ~ d_lat/(c1(q_t)^2 alpha_t^2) grows too, roughly one order more slowly than tau_mem, and the paper should state and plot both growth rates instead of asserting one is constant.'


---

## [MAJOR] The square in (1 - exp(-kappa lambda_i s))^2 is not the absorbed-error fraction and is not identified by the fit

`square-not-derived` · verdict **CONFIRMED** · kind unsupported-claim · effort hours  

**Where:** ICLR_2026/sec-spectral-predictor.tex:17; ICLR_2026/sec-appendix-integrated.tex:786; derived against ICLR_2026/sec-rfnn.tex Eq. (eq:rfnn-mode-decay)

**Claim:** Under Eq. (rfnn-mode-decay) with A initialized to zero (stated in sec-rfnn.tex), a_i(T) = a_i*(1 - e^{-lambda_i T}); the residual coefficient is e^{-lambda_i T} and the absorbed fraction of SQUARED error is 1 - e^{-2 lambda_i T}, not (1 - e^{-lambda_i T})^2. The squared form does have one principled reading -- it is the per-mode learned-energy fraction a_i(T)^2/(a_i*)^2 -- but that reading requires w_i proportional to |a_i*|^2 so that P_d = ||A(s)||_F^2/||A*||_F^2. The paper sets w_i = 1, which is neither the absorbed-error clock nor the learned-energy clock. No derivation of the square is given and no ablation over the functional form is reported.

**Evidence:** sec-rfnn.tex Eq. (eq:rfnn-mode-decay): 'a_i(T) - a_i* = (a_i(0) - a_i*) e^{-lambda_i T}' with 'a trained readout A ... initialized to zero'. Appendix text asserts only that the predictor is 'inspired by the RFNN mode-timescale theory' and never derives the exponent. The two forms are also empirically unidentifiable once kappa and the threshold levels are free: in a two-population toy (lambda_hi=2.0, lambda_lo=Delta_t=0.1813, fraction f at the floor) the delay factor relative to f=0 is, for the squared form, 1.18/1.48/1.91 at f=0.2/0.4/0.55 (theta=0.25) versus 1.26/1.70/2.30 for 1-e^{-2 lambda u}; at theta=0.5, 1.26/1.87/3.62 versus 1.34/2.02/3.18. Same order of magnitude, opposite sign of the discrepancy at the two levels -- so the fit cannot distinguish them, and the appeal to RFNN theory buys nothing.

**Fix:** State the exact quantity being modelled. If it is absorbed squared score error, use 1 - e^{-2 kappa lambda_i s}. If it is learned readout energy, keep the square and set w_i = |a_i*|^2 (computable in the RFNN, approximable on real data by the mode's target score amplitude). Report an ablation over {1-e^{-x}, (1-e^{-x})^2, 1-e^{-2x}} showing whether the choice matters; on present evidence it does not, which should be said.


**Referee:** Verified as stated. sec-rfnn.tex explicitly says A is 'initialized to zero', so eq:rfnn-mode-decay gives a_i(T) = a_i*(1 - e^{-lambda_i T}); the absorbed fraction of squared error is 1 - e^{-2 lambda_i T}, while (1-e^{-lambda_i T})^2 = a_i(T)^2/(a_i*)^2 is the learned-energy fraction, which as a normalized clock requires w_i proportional to |a_i*|^2. The paper sets w_i = 1 (sec-spectral-predictor.tex:17, appendix line 789, and `w = np.ones_like(eig)` in pressure_u). No derivation of the exponent appears anywhere in ICLR_2026/*.tex, and no ablation over the functional form is reported or implemented.


**Referee correction:** Severity should be downgraded from 'unsupported-claim/major' to a derivation/presentation defect. The appendix calls P_d 'the average absorbed-mode mass', which is a fair informal gloss on the learned-energy reading, and both sec-appendix-integrated.tex:758-762 and the pipeline README label the whole object a surrogate 'inspired by' the RFNN theory rather than a consequence of it. The referee's own toy calculation shows the two functional forms are not identifiable once kappa and the six theta levels are free, so the choice cannot be changing any reported conclusion. What is genuinely wrong is the paper stating a specific form without saying which quantity it models; the remedy is one sentence plus the three-way ablation, not a result-level correction.


---

## [MAJOR] The stated 3-parameter anchored sigmoid is not what the released code fits (7 parameters), and the main-text figure is produced by no script in the repo

`dof-paper-vs-code` · verdict **CONFIRMED** · kind error · effort hours  

**Where:** ICLR_2026/sec-spectral-predictor.tex:17 and sec-appendix-integrated.tex:790-800 vs make_celeba_spectral_mem_predictor_figures.py:`calibrate` and make_cifar10_spectral_mem_predictor_figures.py:`predicted_mem_from_pressure`

**Claim:** The paper describes 'one anchored sigmoid response' with 'one set of (kappa, a, b)' -- 3 free parameters -- 'fit once per dataset using the dimensions where the VAE already preserves pixel-space identity: d>=70 for CelebA and d>=140 for CIFAR-10'. The only implementation present fits kappa PLUS a free monotone level theta_q for each of the 6 thresholds (7 parameters, constrained only by PAVA monotonicity), and it fits on ALL dimensions with no identity-threshold filter. The 'predicted memorization curve' shown in red is, in the code, piecewise-linear np.interp through those 6 fitted anchors, with a docstring saying so. The main-text figure file (figures/real_spectral_predictor_compact_2x2.pdf) is generated by no script anywhere in the repository, so the 3-parameter claim is unverifiable.

**Evidence:** make_celeba_spectral_mem_predictor_figures.py: `fit_df = onset_summary.dropna(subset=["tau_obs_mean"])` with `DIMS = [10, 20, 30, ..., 200]` and no d-filter; `theta = pava_non_decreasing(raw, ...)` produces one level per threshold; calibration.json records kappa plus theta_1pct..theta_75pct (6 values: 0.2558, 0.4366, 0.5266, 0.7423, 0.8862, 0.9753). make_cifar10_spectral_mem_predictor_figures.py, `predicted_mem_from_pressure` docstring: 'This interpolation is just a visualization helper: it turns those threshold anchors into a smooth-ish predicted mem curve.' `grep -rn 'compact_2x2'` matches only the two .tex files. Forcing the 6 CelebA thetas onto an anchored sigmoid gives a=8.24, b=0.868 with predicted q of 0.006/0.027/0.056/0.262/0.538/0.708 against true 0.01/0.05/0.10/0.25/0.50/0.75 -- a ~2x relative error at the 10% level -- so the two parameterizations are not interchangeable. sec-discussion.tex:61 contradicts the main text by referring to 'the shared time scale and threshold map'.

**Fix:** Publish the script that produced the main-text figure, and describe in the paper exactly the parameterization it uses. If the released 7-parameter version is what was run, say '1 shared time constant plus 6 monotone level parameters' and revise the DOF discussion accordingly. Fix the appendix's false statement that the fit is restricted to post-identity-threshold dimensions.


**Referee:** All four sub-claims verified. (1) `grep -rl sigmoid|anchored` over the whole repo matches only ICLR_2026/sec-spectral-predictor.tex and sec-appendix-integrated.tex — no Python file anywhere implements the anchored sigmoid; the released `calibrate` grid-searches one kappa over 121 log-spaced values and computes six theta levels as obs_n-weighted averages of P_d(tau_obs) then PAVA-monotonized, and calibration.json stores exactly kappa + theta_1pct..theta_75pct. (2) `DIMS = [10, 20, ..., 200]` with `fit_df = onset_summary.dropna(subset=['tau_obs_mean'])` and no d-filter, directly contradicting the appendix's 'fit once per dataset using the dimensions where the VAE already preserves pixel-space identity: d>=70 for CelebA and d>=140 for CIFAR-10' — that sentence is factually false about the released code. (3) The cifar script's `predicted_mem_from_pressure` docstring does say 'This interpolation is just a visualization helper'. (4) `grep -rn compact_2x2` and `grep -rn real_spectral_predictor` match only sec-spectral-predictor.tex:10 and the PDF itself; no script in the repo generates the main-text figure. I also reproduced the sigmoid refit on the six CelebA thetas exactly: a = 8.2374, b = 0.8677, predicted q = 0.0056/0.0271/0.0561/0.2619/0.5377/0.7079 against true 0.01/0.05/0.10/0.25/0.50/0.75 — the parameterizations are not interchangeable. sec-discussion.tex ('the shared time scale and threshold map') describes the code's version, not the main text's.


**Referee correction:** '7 parameters' overstates the effective degrees of freedom slightly. The six thetas are not free-optimized against the log-tau objective; each is the obs_n-weighted mean of P_d(tau_obs(d)) at its threshold given kappa, then projected to monotone by PAVA — a plug-in estimator, and on CIFAR the PAVA projection pooled two of them, so only five distinct levels survive. The accurate description is '1 shared time constant plus a monotone six-level threshold map estimated from the same hitting times', which is still six more fitted quantities than the paper's '(kappa, a, b)' claims and still absorbs all per-threshold level information.


---

## [MAJOR] No held-out evaluation of any kind; against the model's own per-threshold free levels the d-dependence R^2 is 0.45 / 0.39

`no-holdout-and-honest-r2` · verdict **CONFIRMED** · kind gap · effort days  

**Where:** ICLR_2026/sec-appendix-integrated.tex:758-830 (whole predictor appendix); ICLR_2026/sec-spectral-predictor.tex:19

**Claim:** The predictor is fit and evaluated on the same hitting times, with no leave-one-d-out, no held-out threshold, and no cross-dataset transfer of (kappa, theta). Because theta_q is fit as the across-d average of P_d(tau_obs(d)) at each threshold, the residual is centered within every threshold by construction: the ONLY testable content is the d-ordering, and the reported 'predicted vs observed' scatter (fig:app-spectral-tau) is flattered by 6 fitted per-level offsets. Scored honestly against the baseline the model already contains (a free constant per threshold), the clock explains 44.7% of the log-variance on CelebA and 39.1% on CIFAR-10; against an unconditional mean it looks like 0.87/0.81.

**Evidence:** grep for 'held.out|leave-one|cross-valid' over ICLR_2026/*.tex returns only two synthetic-RFNN test-set mentions in sec-appendix.tex, none in the predictor section. Recomputed from clean figures/{celeba,cifar10}_spectral_mem_predictor/*_spectral_tau_table.csv: 73 (CelebA) and 49 (CIFAR) uncensored (d,q) pairs; RMS log error 0.326 / 0.305 (typical 1.39x / 1.36x multiplicative error) against an observed within-threshold tau range of only 2.0x-4.5x. Within-threshold R^2 = 0.447 / 0.391; the reported objective_mse_log in calibration.json (0.1066 / 0.0928) is the raw fit residual, never converted to a variance-explained figure. Baselines with ONE free shared slope beat or match it: -log(etabar_star) gives 0.677 on CelebA, -log(lambda_mean) gives 0.725 / 0.326.

**Fix:** Report leave-one-d-out: refit (kappa, theta) excluding each latent width and predict its hitting times. Report cross-dataset transfer (CelebA-fitted parameters applied to CIFAR). Report within-threshold R^2, not just the raw log-MSE, and compare against the one-parameter baselines tau ~ 1/etabar_star(d) and tau ~ 1/lambda_mean(d), which currently do as well or better.


**Referee:** Every number reproduces exactly from the committed CSVs. CelebA: 73 uncensored (d,q) rows, RMS log error 0.32643, raw log-MSE 0.106554 (identical to objective_mse_log in calibration.json), within-threshold R^2 = 0.4471, unconditional R^2 = 0.8683. CIFAR-10: 49 rows, RMS 0.30471, log-MSE 0.092847, R^2 = 0.3914 within / 0.8099 unconditional. One-parameter-slope baselines with the same per-threshold intercepts: -log(eta_star) 0.677/0.288, -log(lambda_mean) 0.725/0.326 — so on CelebA both single-feature baselines beat the clock and on CIFAR lambda_mean roughly matches it. `grep -rni 'held.out|leave-one|cross-valid'` over ICLR_2026/*.tex returns only two synthetic-RFNN test-set mentions in the uncompiled sec-appendix.tex; there is no leave-one-d-out, no held-out threshold, no cross-dataset transfer, and calibration.json's log-MSE is never converted to a variance-explained number in the paper.


**Referee correction:** 'Centered within every threshold by construction' is approximately, not exactly, true. theta_q is a weighted mean of P_d(tau_obs) in pressure space, then PAVA-projected, while the objective is in log-tau; the resulting per-threshold mean log residuals are small but nonzero (CelebA: +0.033, +0.003, -0.018, -0.016, -0.013, -0.065 at q = 0.01...0.75). The conclusion is unaffected — within-threshold R^2 is the right scoring baseline against a model that already contains a free level per threshold — but the mechanism is 'fit absorbs the per-threshold offsets', not an exact algebraic identity.


---

## [MAJOR] Censored rows are silently dropped and are concentrated exactly in the two regimes the buffer claim is about

`censoring-removes-the-test-regime` · verdict **CONFIRMED** · kind gap · effort days  

**Where:** ICLR_2026/sec-appendix-integrated.tex:800-802 ('Empirical hitting times are censored when a seed never reaches level q within 5M steps'); make_*_spectral_mem_predictor_figures.py:`empirical_onsets`

**Claim:** 17 of 90 (CelebA) and 29 of 78 (CIFAR-10) dimension-threshold cells are fully censored and are removed from both the fit and every reported agreement plot. They are not missing at random: they sit at the small-d end (representation-limited) and, more damagingly, at the large-d end which is precisely the claimed excess-buffer regime. On CIFAR-10 all of q in {0.25, 0.50, 0.75} are censored at d = 220, 240, 260, so the three widest latents -- all above the stated identity threshold of 140 -- are tested only at q = 0.01 and 0.05. Where seeds are partially censored, tau_obs_mean averages only the seeds that did hit, biasing observed onset downward at large d.

**Evidence:** Recomputed from the tau tables: CelebA fully censored at d=10 (all six q), d=20 (q>=0.10), d=30 (q>=0.50), d=40/160/180 (q=0.75), d=200 (q>=0.50); CIFAR at d=220/240/260 (q>=0.25). Partially censored rows with a survivor-biased mean: CelebA d=120,q=0.75 (obs_n=1, censored_n=4, tau_obs_mean=5.0e6) and d=140,q=0.75 (obs_n=2). The code's `empirical_onsets` computes `mean_ci(hits)` over uncensored seeds only. Related degeneracy: CIFAR calibration.json has theta_50pct = theta_75pct = 0.9658122228431161 exactly (PAVA pooled them), so the fitted clock predicts identical hitting times for 50% and 75% memorization on CIFAR-10 -- it has no resolution at high memorization levels.

**Fix:** Fit and score with a censored-data likelihood (Tobit / Cox-style) so that 'did not reach q within 5M steps' is a right-censored observation carrying information rather than a deleted row, and report the censoring pattern in the paper. Note the theta_50 = theta_75 degeneracy explicitly.


**Referee:** Reproduced exactly from the tau tables. CelebA: 17 of 90 cells fully censored — d=10 (all six q), d=20 (q>=0.10), d=30 (q>=0.50), d=40 (q=0.75), d=160 and 180 (q=0.75), d=200 (q>=0.50). CIFAR-10: 29 of 78, including all of q in {0.25, 0.50, 0.75} at d = 220, 240, 260, so the three widest latents — all above the stated identity threshold of 140 — contribute only q = 0.01 and 0.05. `empirical_onsets` calls `mean_ci(hits)` over uncensored seeds only, and censored rows carry NaN into `fit_df.dropna(...)`, so they are deleted from both fit and every reported scatter. Partial censoring confirmed: CelebA d=120/q=0.75 (obs_n=1, censored_n=4) and d=140/q=0.75 (obs_n=2, censored_n=3); CIFAR also d=140/q=0.50 (obs_n=4), d=180/q=0.50 (obs_n=2), d=260/q=0.10 (obs_n=4). The theta degeneracy is exact: CIFAR theta_50pct = theta_75pct = 0.9658122228431161, so the calibrated clock predicts identical hitting times for 50% and 75% memorization on CIFAR-10. The censoring is concentrated at both ends of d, i.e. in the excess-buffer regime the claim is about — that is the core of the finding and it holds.


**Referee correction:** The survivor-bias claim is directionally right but bounded in magnitude at the two CelebA cells named: the surviving means are 5.00e6 and 4.85e6 against a 5M-step ceiling, so the downward bias there cannot exceed a few percent. The stronger and cleaner version of the objection is the deletion itself — 17/90 and 29/78 cells dropped, non-randomly, from the regime under test — not the bias of the few partially-censored means.


---

## [MAJOR] The 'excess-weighted' diagnostic assigns zero weight to exactly the buffer modes, and the fit barely degrades

`weighted-variant-deletes-the-buffer` · verdict **CONFIRMED** · kind inconsistency · effort hours  

**Where:** ICLR_2026/sec-appendix-integrated.tex:788-790 ('weights w_i(d) = max{lambda_i(d) - beta_d, 0}'); compute_celeba_spectral_predictor_data.py `eig_excess`; fig:app-spectral-pressure

**Claim:** In the weighted variant the near-floor modes -- the excess latent directions that ARE the buffer -- receive weight exactly zero, because beta_d converges to the diffusion floor at the large latent widths. Removing every buffer mode from the clock costs almost nothing in fit quality, which is direct evidence that the predictor's agreement with the memorization curves is not driven by the buffer modes but by the top of the spectrum.

**Evidence:** spectral_features.csv (CelebA, t=0.1): beta_floor = 0.1894 (d=160), 0.1815 (d=180), 0.1814 (d=200), against Delta_t = 0.18127 and lambda_min = 0.18128 -- so all ~90-112 collapsed coordinates get max(lambda_i - beta_d, 0) = 0. Recomputed within-threshold R^2: CelebA 0.447 unweighted vs 0.366 weighted; CIFAR 0.391 vs 0.322. The appendix nevertheless states the buffer expectation for these curves: 'If the latent-buffer picture is correct, dimensions with broader null buffers should require more spectral time before reaching the same pressure level.'

**Fix:** Report this as a negative result, or use it as an ablation that isolates the buffer contribution: fit the clock (i) on the full spectrum, (ii) on the top-r modes only, (iii) on the near-floor modes only, and show which subset carries the d-ordering. Currently the evidence points at (ii).


**Referee:** Verified structurally and numerically. w_i(d) = max{lambda_i(d) - beta_d, 0} with beta_d = median of the smallest ceil(0.30 d) eigenvalues; celeba_spectral_features_primary_t.csv gives beta_floor = 0.18945 / 0.18151 / 0.18136 at d = 160/180/200 against lambda_min = 0.18133 / 0.18129 / 0.18128 and Delta_{0.1} = 0.181269. So at the widest latents the floor estimate sits essentially on top of the diffusion floor and every collapsed near-floor coordinate — the modes the buffer narrative is about — receives weight exactly zero. Recomputed within-threshold R^2: CelebA 0.447 unweighted vs 0.366 weighted; CIFAR 0.391 vs 0.322. The internal inconsistency is real: the caption of fig:app-spectral-pressure states the buffer expectation ('dimensions with broader null buffers should require more spectral time before reaching the same pressure level') for a pair of panels one of which cannot express it, since with excess weights the buffer modes contribute to neither numerator nor denominator.


**Referee correction:** Two caveats. (a) The exact count of zero-weight modes could not be verified: spectra.npz (and the tmp_* directories) are not in the repo, so 'all ~90-112 collapsed coordinates get zero' is an inference. What is certain by construction is that at least the bottom 15% of modes get zero weight at every d, and that wherever beta_d ~ lambda_min ~ Delta_t the entire collapsed block does. (b) 'costs almost nothing' overstates: R^2 drops by 0.081 and 0.069, an 18% relative loss on both datasets, and the weighted version is presented as a diagnostic rather than the primary. The defensible claim is that most of the clock's predictive content survives after deleting every buffer mode, which still points at the top of the spectrum, not the buffer, as the source of the d-ordering.


---

## [MAJOR] The identity threshold is defined by the same KNN statistic as the outcome, is unablated in rho, and excludes precisely where the predictor fails hardest

`identity-gate-shares-the-outcome-statistic` · verdict **CONFIRMED** · kind unsupported-claim · effort days  

**Where:** ICLR_2026/sec-real-data.tex:36 (definition of dhat_ID^VAE(rho)); ICLR_2026/sec-spectral-predictor.tex:11,19; ICLR_2026/sec-appendix-integrated.tex:797-799

**Claim:** dhat_ID^VAE(rho) is 'the smallest latent dimension whose immediate VAE encode/decode reconstructions are counted as memorized for at least a fraction rho of the same diverse 1k subset, using the pixel-space nearest-neighbor ratio test' -- the same statistic, same threshold (1/3), same subset as the outcome variable. It is a ceiling argument, which is legitimate in kind, but (a) rho = 0.95 is asserted with no sensitivity analysis, (b) no numeric table of reconstruction-memorization vs d appears anywhere, only a figure, and (c) the gate removes exactly the dimensions where the predictor is most badly wrong, so the reported agreement is conditioned on a selection made after seeing the residuals. The appendix compounds this by claiming the fit uses only post-threshold dimensions when the code fits on all of them.

**Evidence:** At CelebA d=10 the clock predicts the FASTEST 1% memorization of all 15 widths (tau_pred = 201,428 steps, vs 286,790 at d=50 and 433,596 at d=200), while all five seeds are censored -- they never reach 1% in 5M steps. That single point is a qualitative sign error, and it lies below the gate. `grep -i 'rho'` over the predictor and VAE appendix sections finds no sensitivity sweep over rho; app:vae-clean contains only figures. make_celeba_spectral_mem_predictor_figures.py fits with `DIMS = [10, 20, 30, ...]` and no filter, contradicting sec-appendix-integrated.tex:797 ('fit once per dataset using the dimensions where the VAE already preserves pixel-space identity: d>=70 for CelebA and d>=140 for CIFAR-10').

**Fix:** Report dhat_ID^VAE(rho) for rho in {0.5, 0.8, 0.9, 0.95, 0.99} with the resulting fit and R^2 for each, so the reader can see how much the conclusion depends on the cut. Publish the reconstruction-memorization fractions as a table. Use an identity proxy that is not the outcome statistic (e.g. LPIPS or reconstruction PSNR relative to nearest-neighbour distance) as a robustness check. And make the code's fit set match the text.


**Referee:** Verified point by point. sec-real-data.tex:36 defines dhat_ID^VAE(rho) using 'the pixel-space nearest-neighbor ratio test' on 'the same diverse 1k subset' — the same statistic, subset, and 1/3 threshold as the outcome variable, with rho = 0.95 asserted and no sweep anywhere in the .tex files. app:vae-clean (sec-appendix-integrated.tex:392ff) contains only figure environments; no numeric reconstruction-memorization table exists. The code/text mismatch is real: DIMS starts at 10 with no filter, contradicting sec-appendix-integrated.tex:797. And the d=10 sign error checks out — at q = 0.01 the clock predicts tau = 201,428 steps, the fastest of all 15 widths (next is 286,790 at d=50, and 433,596 at d=200), while all five seeds are censored at 5M.


**Referee correction:** The finding contains an internal tension worth flagging when it is written up. Because the released code fits on all dimensions with no gate, the claim that 'the gate removes exactly the dimensions where the predictor is most badly wrong, so the reported agreement is conditioned on a selection made after seeing the residuals' cannot be true of the reported numbers — the fit really did include d=10 through d=60, and the R^2 of 0.447 is computed over all of them. The selection operates on the narrative (the paper tells the reader to disregard the sub-threshold region where the predictor inverts sign) rather than on the arithmetic. Both defects are real, but they are alternatives: either the paper's description is wrong, or the analysis is selected — not both at once.


---

## [MAJOR] No link at all from per-mode absorption to pixel-space nearest-neighbour memorization; the response curve is a black box standing in for the whole chain

`missing-link-score-error-to-knn` · verdict **CONFIRMED** · kind missing-theory · effort days  

**Where:** ICLR_2026/sec-spectral-predictor.tex:6,17; ICLR_2026/sec-discussion.tex:59-63; clean figures/celeba_spectral_mem_predictor/README.md

**Claim:** The stated chain is (VAE latent spectrum -> U spectrum -> per-mode absorption -> score error -> KNN memorization fraction). Link 1 exists trivially. Link 2 exists only for the data-side modes (Lemma 1) and is missing for the sample bulk. Link 3 exists only for a FROZEN-feature linear readout under full-batch gradient flow at a single diffusion time; the real experiments train a nonlinear MLP with SGD+momentum over the full range of t. Link 4 requires the target coefficients a_i* and is skipped (w_i = 1). Link 5 -- from score error to a decoded sample landing within nearest-neighbour ratio 1/3 of a training image -- does not exist anywhere in the paper. The sigmoid/theta map absorbs Links 3-5 into two-to-six fitted numbers, and kappa alone conflates learning rate, momentum, batch size, feature learning, and the t-marginal of the diffusion loss with the mu_1^2/psi_p rescaling that Lemma 1 actually supplies.

**Evidence:** sec-rfnn.tex Eq. (eq:rfnn-gradflow) is derived from 'Freezing W makes the score-matching loss quadratic in A'; the real-data models are trainable MLPs. The pipeline README states the gap outright: 'this is a data-spectrum surrogate inspired by the RFNN mode-timescale theory, not a literal RFNN theorem application to the trainable MLP.' sec-discussion.tex:59 concedes 'the spectral predictor is an empirical frozen-VAE latent-spectrum rule, not yet a theorem for the trainable MLP', yet the abstract sells it as 'a mode-absorption clock that estimates memorization percentage as a function of training step' and sec-intro.tex:29 as a general estimation method.

**Fix:** State the chain explicitly in the paper with each link marked derived / assumed / fitted, as the discussion half-concedes. Minimally, close Link 4 by weighting modes with |a_i*|^2 (available in closed form for the Gaussian-mixture score) and validate Links 2-3 in the synthetic RFNN, where U, its eigenvalues, and the true memorization time are all directly measurable -- that is a cheap, decisive test of whether the clock built from M_t predicts RFNN memorization onset, and it is currently not run.


**Referee:** The chain audit is accurate. Link 2 is established for the data-side modes only (lem:Ulin) and, per the same file's own Remark, cannot reach the sample bulk. Link 3 holds for a frozen-W linear readout under full-batch gradient flow at t = 0.01 (sec-rfnn.tex: 'Freezing W makes the score-matching loss quadratic in A'; 'score matching at a single fixed diffusion time t = 0.01'), while the real-data models are depth-3 trainable MLPs under SGD with momentum 0.80 trained across all t (sec-real-data.tex). Link 4 is skipped: w_i = 1, no a_i* anywhere. Link 5 — score error to a decoded sample landing within KNN ratio 1/3 — appears nowhere in the corpus. kappa is a single free scalar absorbing all of it. I also checked the referee's proposed decisive test: no script in the repo runs the clock against synthetic RFNN memorization onset (only make_celeba_* and make_cifar10_* use `pressure`), so that validation is indeed not run.


**Referee correction:** Severity should come down from major, because the paper discloses the gap in two places, not one: sec-discussion.tex says 'the spectral predictor is an empirical frozen-VAE latent-spectrum rule, not yet a theorem for the trainable MLP', and the pipeline README says 'a data-spectrum surrogate inspired by the RFNN mode-timescale theory, not a literal RFNN theorem application to the trainable MLP.' The residual defect is a calibration mismatch between sections — abstract.tex and sec-intro.tex:29 present the clock as a general estimation method with no such hedge — rather than an undisclosed hole. The concrete actionable items are the per-link derived/assumed/fitted table and the synthetic-RFNN validation, which is genuinely cheap and genuinely missing.


---

## [MAJOR] mu_1^2 = 0.367 is only valid at ||x_t||^2/d_lat = 1; across the sweep the effective linear coefficient drifts by 3.5x and is itself d_lat-dependent, which the derivation treats as a constant

`mu1-norm-dependence` · verdict **CONFIRMED** · kind missing-theory · effort days  

**Where:** sec-appendix-fourbulk.tex Eq \eqref{eq:kernel-decomp} (lines 50-58), the 'numerically mu_1 approx 0.606, mu_3 approx -0.099' line, and the 'Bulk-gap scaling' paragraph (lines 241-259)

**Claim:** Eq (kernel-decomp) writes the off-diagonal kernel term as mu_1^2 x^T y/d_lat with mu_1 the unit-variance Hermite coefficient, but writes the diagonal term as sum_{k>=3} mu_k^2 (||x||^2/d_lat)^k — i.e. it carries the norm scaling on the higher-Hermite terms and drops it on the linear term. That is internally inconsistent. For pre-activations of variance tau^2 = ||x||^2/d_lat the correct linear coefficient is (c_1(tau)/tau)^2 with c_1(tau) = E_z[z tanh(tau z)] = tau E[sech^2(tau z)], which equals mu_1^2 = 0.367 only at tau = 1 and tends to 1 as tau -> 0. Since d_int, s, sigma_sig are held fixed while d_lat grows, tau_t^2 = Delta_t + e^{-2t}(s^2 + d_int sigma_sig^2 + (d_lat-d_int) sigma_perp^2)/d_lat varies strongly over the sweep, so the 'constant' mu_1^2 is a d_lat-dependent quantity.

**Evidence:** tau_t^2 over the sigma_perp=0.5 sweep: 2.764 (d=5), 1.515 (d=10), 0.890 (d=20), 0.577 (d=40), 0.390 (d=100), 0.327 (d=200). Corresponding (c_1/tau)^2: 0.180, 0.281, 0.393, 0.494, 0.586, 0.625 — a 3.5x drift, spanning both sides of the quoted 0.367. Two independent estimates from the data agree with this and not with 0.367: sig_obs/(psi_p alpha_t^2) = 0.181, 0.280, 0.381, 0.474, 0.579, 0.638 and nd_obs/(psi_p beta_t^2) = 0.306, 0.419, 0.509, 0.587, 0.627. The corrected formula lambda_i(U_code) = (c_1(tau_t)/tau_t)^2 psi_p lambda_i(M_t^emp) reproduces the full d_lat-dimensional linear block to within 3-8% per mode, including the band edges (d_lat=200, sigma=0.5: observed nd [max, med, min] = [27.17, 9.22, 2.12] vs predicted [26.28, 9.30, 2.19]).

**Fix:** Restate Eq (kernel-decomp) with norm-dependent coefficients: E_W[tanh((Wx)_a/sqrt(d))tanh((Wy)_a/sqrt(d))] = c_1(tau_x)c_1(tau_y) rho_{xy} + ... with rho_{xy} = x^T y/(||x|| ||y||), so the linear term is [c_1(tau_x)c_1(tau_y)/(tau_x tau_y)] x^T y/d_lat. Carry the resulting mu_1_eff^2(tau_t) through Lemma 1, the Theorem table, and the 'Bulk-gap scaling' paragraph (whose quoted mu_1^2 = 0.367 and mu_3^2 = 0.0097 are both off-regime). Note this also explains the apparent saturation of the signal bulk with d_lat that the eigenvalue_saturation appendix treats as a separate phenomenon.


**Referee:** The internal inconsistency in Eq (kernel-decomp) (sec-appendix-fourbulk.tex:50-58) is exactly as described: the off-diagonal term carries a norm-free mu_1^2 x^T y / d_lat while the diagonal term carries the norm scaling (||x||^2/d_lat)^k. By Price's theorem, for jointly Gaussian pre-activations with per-coordinate std tau the k=1 term is E[tanh'(tau z)]^2 * x^T y / d_lat = (E[sech^2(tau z)])^2 x^T y / d_lat, which equals mu_1^2 = 0.3669 only at tau = 1 (E[sech^2(z)] = 0.6057 = mu_1, consistent with the appendix's own numbers at line 39). I recomputed tau_t^2 = e^{-2t} E||x||^2/d_lat + Delta_t from the actual generator (code/experiment_v2.py generate_data, seed 42): 2.752, 1.507, 0.889, 0.578, 0.474, 0.422, 0.390, 0.349, 0.329 for d_lat = 5..200, giving mu_1_eff^2 = 0.181, 0.282, 0.393, 0.493, 0.540, 0.567, 0.585, 0.611, 0.624 — a 3.5x drift spanning both sides of 0.367, and monotone in d_lat because d_int, s, sigma_sig are held fixed. Matches the reviewer's numbers to 3 decimals. The corrected law lambda_i(U_code) = mu_1_eff^2(tau_t) psi_p lambda_i(M_t^emp) reproduces the full linear block per-mode to 3-8% at every d_lat (numbers in my ulin-prefactor-wrong reasoning), where the mu_1^2 = 0.367 version does not. This is not addressed elsewhere: the saturation appendix ('ICML (1)/sec-appendix.tex':854-863) only covers the opposite, large-tau failure mode (tanh saturating when ||Wx||/sqrt(d_lat) >~ 2), and sec-appendix-fourbulk.tex:32-35 only asserts 'order-one variance per coordinate', which is not the same as tau = 1. The 'Bulk-gap scaling' paragraph's quoted mu_1^2 = 0.367 and mu_3^2 = 0.0097 are therefore both off-regime.


**Referee correction:** The reviewer's aside that this 'explains the apparent saturation of the signal bulk with d_lat that the eigenvalue_saturation appendix treats as a separate phenomenon' is loosely worded — the saturation appendix is about a different (large-s) regime. The accurate statement is that the observed flattening of the signal bulk (32 -> 113 over d_lat = 5 -> 200, not the 20x growth the corrected constant-mu_1 formula would give) is the mu_1_eff^2 -> 1 ceiling as tau_t -> 0, and is a distinct effect from the tanh saturation appendix.


---

## [MAJOR] Lemma 1 substitutes the population block spectrum of M_t for the empirical one; the noise-dim 'bulk' is in fact a Marchenko-Pastur band of hat S that spans 12x at d_lat=200 and whose lower edge sets the buffer bound

`Mt-empirical-not-population` · verdict **CONFIRMED** · kind gap · effort days  

**Where:** sec-appendix-fourbulk.tex, 'Block decomposition of M_t' paragraph (lines 90-109), Eq \eqref{eq:Mt}, and Lemma 1's use of spec(M_t) = {alpha_t^2}^{dint} u {beta_t^2}^{dlat-dint}

**Claim:** Eq (Mt) defines M_t with the *empirical* hat C and hat S over n samples, but the very next sentence replaces them by their population values ('In population, E hat C has rank dint ... and E hat S = Sigmadata'), and Lemma 1 then quotes a two-point spectrum. hat S = (1/n) sum xi xi^T with n = 500 and d_lat up to 240 is an MP-distributed estimate, not sigma_perp^2 I. The noise-dim block of M_t is therefore a band [e^{-2t}sigma_perp^2(1-sqrt(d_lat/n))^2 + Delta_t, e^{-2t}sigma_perp^2(1+sqrt(d_lat/n))^2 + Delta_t], not beta_t^2 I. This matters directly because Eq (buffer-bound) uses lambda_min^{noise-dim} = mu_1^2 beta_t^2/psi_p, i.e. the band *centre* where the *lower edge* is required.

**Evidence:** Computed from the actual generated data (code/experiment_v2.py generate_data, seed 42, n=500, sigma_signal=1, scale=3): at d_lat=200, sigma_perp=0.5 the noise-dim block of M_t spans [0.0549, 0.6578], ratio 12.0, versus the claimed single value beta_t^2 = 0.2649 — the MP prediction sigma_perp^2(1 +/- sqrt(d_lat/n))^2 + Delta_t = [0.034, 0.666] matches almost exactly. At d_lat=40: [0.1575, 0.4155], ratio 2.6. So lambda_min^{noise-dim} is 4.8x smaller than the value plugged into Eq (buffer-bound) at d_lat=200. The empirical U spectrum inherits this exactly (observed nd min 2.124 in code units, predicted 2.194). At sigma_perp=0.01 the block is Delta_t-dominated and the approximation is harmless ([0.0198, 0.0201]).

**Fix:** State M_t's spectrum as the empirical one and give the noise-dim bulk as an MP band with an explicit lower edge, then use that lower edge in the buffer corollary. Add the hypothesis d_lat < n explicitly — the stated proportional limit (psi_n = n/d_lat fixed, psi_p > 1 + psi_n) permits psi_n < 1, i.e. d_lat > n, where rank(hat S) <= n and the noise-dim block degenerates to Delta_t I plus a rank-n perturbation; in that regime the entire 'noise-dim bulk at beta_t^2' statement is false and the bulk merges into the diffusion floor. If the paper wants to extrapolate to the CelebA/CIFAR latent dimensions it must say which regime those live in.


**Referee:** Eq (Mt) (sec-appendix-fourbulk.tex:92-96) is written with empirical hat C and hat S, and line 100-109 immediately substitutes population values, yielding the two-point spectrum {alpha_t^2}^{d_int} u {beta_t^2}^{d_lat-d_int} that Lemma 1 then quotes. I computed the actual M_t from the generator (seed 42, n=500): the noise-dim block spans [0.2211, 0.3002] at d_lat=10 (ratio 1.4), [0.1562, 0.4103] at 40 (2.6), [0.0981, 0.5159] at 100 (5.3), and [0.0544, 0.6575] at 200 (ratio 12.1) — versus the claimed single value beta_t^2 = 0.2649. The MP prediction e^{-2t} sigma_perp^2 (1 +/- sqrt(d_lat/n))^2 + Delta_t gives [0.0529, 0.6728] at d_lat=200 and [0.1458, 0.4231] at 40, matching almost exactly. This is material precisely where the reviewer says: Eq (buffer-bound) plugs in lambda_min^noise-dim = mu_1^2 beta_t^2/psi_p, i.e. the band centre, where the lower edge is 4.9x smaller at d_lat=200. The empirical U spectrum inherits the band exactly (observed nd min 2.124 code units vs 2.17 predicted from the empirical M_t lower edge). At sigma_perp=0.01 the block is Delta_t-dominated and the approximation is indeed harmless. The missing d_lat < n hypothesis is also real: the stated setup (line 27-29) fixes only psi_p, psi_n and psi_p > 1 + psi_n, which admits psi_n < 1.


**Referee correction:** None material. Worth adding that the reviewer's point interacts with finding psip-not-fixed-in-shown-experiments: n = 500 is held fixed across the whole d_lat sweep, so psi_n runs 100 -> 2.5. The MP widening of the noise-dim block is therefore an artifact of the experiments violating the theorem's own psi_n-fixed hypothesis, not something that would occur in the stated proportional limit (where the band width would be constant). Both the hypothesis and the experiment need reconciling.


---

## [MAJOR] alpha_t^2 = e^{-2t}(s^2/d_int + sigma_sig^2) + Delta_t hides an O(1) k-dependence: with k=10 centres and d_int=5 the signal block spans 3.9x and its slowest mode — the one that sets tau_gen — sits 46% below alpha_t^2

`hatC-k-dependence` · verdict **CONFIRMED** · kind unsupported-claim · effort hours  

**Where:** sec-appendix-fourbulk.tex lines 97-109 ('In population, E hat C has rank dint with eigenvalues Theta(s^2/dint)'), and the 'Signal' row of Theorem \ref{thm:fourbulk}

**Claim:** hat C = (1/n) sum_mu m_{c(mu)} m_{c(mu)}^T = sum_c (n_c/n) m_c m_c^T is a k-term sum, i.e. a k-sample estimate of a d_int-dimensional covariance. Its relative fluctuation is O(sqrt(d_int/k)), which does NOT vanish in the stated limit because both d_int = 5 and k = 10 are held fixed while d_lat -> infinity. Quoting a single alpha_t^2 for all d_int signal modes is therefore not an asymptotic statement but an O(1) approximation, and the error falls on the mode that actually determines tau_gen. Two further silent assumptions: (i) rank(hat C) = min(k, d_int), so the theorem's count of d_int signal modes holds only for k >= d_int — for k < d_int the signal bulk has k (or k-1 if centred) modes, and the theorem asserts d_int unconditionally; (ii) the generator does not produce isotropic centres, so E hat C is not (s^2/d_int) I even in expectation over the design.

**Evidence:** For the actual construction (code/experiment_v2.py: first d_int centres orthogonal with norm scale=3, remaining k-d_int random on the sphere of radius 3), seed 42: spec(hat C) = [3.399, 2.452, 1.271, 1.001, 0.877], mean 1.8 = s^2/d_int as claimed but max/min = 3.9. Signal block of M_t = [4.288, 3.545, 2.217, 2.024, 1.888] versus alpha_t^2 = 2.764; the slowest signal mode is 1.888, so alpha_t^2 over-predicts lambda_min^signal by 46% and under-predicts tau_gen by the same factor — precisely the 1.5x constant offset seen between observed 1/lambda_min^signal and the alpha_t^2-based prediction in finding taugen-not-dlat-independent. The construction is not double counting cluster centres against within-cluster covariance (x = m_c + xi with xi ~ N(0, Sigma_data) is correct), but the claim that k does not enter is wrong.

**Fix:** Either (a) add k -> infinity with d_int/k -> 0 to the asymptotic hypotheses and say so, or (b) state the signal row as the empirical spectrum e^{-2t}(lambda_i(hat C) + sigma_sig^2) + Delta_t with alpha_t^2 as its mean, and use lambda_min not the mean wherever a timescale is derived. Add the explicit hypothesis k >= d_int for the count d_int, and state the centre distribution (the first d_int orthogonal, the rest uniform on the sphere) since E hat C is not proportional to the identity under it.


**Referee:** Reproduced exactly from the generator (code/experiment_v2.py:90-139, k=10 > d_int=5 branch: first 5 centres orthogonal at norm scale=3, remaining 5 uniform on the radius-3 sphere; seed 42): spec(hat C) = [3.399, 2.452, 1.271, 1.001, 0.877], mean 1.800 = s^2/d_int = 9/5 exactly as the appendix claims, but max/min = 3.88. The resulting signal block of M_t is [4.218, 3.533, 2.483, 2.198, 1.878] at d_lat=200 (essentially d_lat-invariant: [4.105, 3.431, 2.368, 2.110, 1.754] at d_lat=10), against alpha_t^2 = 2.764. So alpha_t^2 over-states lambda_min^signal by 47%, and lambda_min^signal is exactly the quantity that sets tau_gen — this is the source of the constant ~1.5x offset seen in the taugen finding. The structural argument is also right: hat C = sum_c (n_c/n) m_c m_c^T is a k-term sum with k=10 and d_int=5 both held fixed as d_lat -> infinity, so its O(sqrt(d_int/k)) relative fluctuation does not vanish in the stated limit (line 27-29 lists only psi_p and psi_n as fixed), and the appendix's 'In population, E hat C has rank d_int with eigenvalues Theta(s^2/d_int)' (line 100-101) is an O(1) approximation, not an asymptotic statement. The rank(hat C) = min(k, d_int) point is correct too: the k <= d_int branch of generate_data builds k orthogonal centres, so the theorem's unconditional signal count d_int fails for k < d_int. And E hat C is not proportional to the identity under this centre distribution.


**Referee correction:** None. The reviewer's own hedge ('the construction is not double counting cluster centres against within-cluster covariance') is correct and worth keeping — this is purely about the k-dependence and about using the block mean where a block minimum is required.


---

## [MAJOR] With d_int fixed the 'signal bulk' has vanishing relative size — it is a finite-rank spike, so no a.s. ESD statement or MP machinery applies to it and the BBP separation threshold is never checked

`signal-bulk-is-a-spike` · verdict **CONFIRMED** · kind missing-theory · effort weeks  

**Where:** sec-appendix-fourbulk.tex lines 27-29 (proportional limit) vs Theorem \ref{thm:fourbulk} 'splits almost surely into four contiguous bulks with the following counts', signal row count d_int

**Claim:** The stated limit is d_lat, n, p -> infinity with psi_p, psi_n fixed; d_int = 5 is held fixed throughout the experiments and never appears in the list of ratios held fixed. Hence d_int/d_lat -> 0 and the signal block carries zero mass in the limiting spectral distribution. Almost-sure convergence of the empirical spectral distribution says nothing whatsoever about d_int outlier eigenvalues, so the Theorem's a.s. claim about the signal bulk's count and edge is not delivered by the cited MP/replica machinery. The correct framework is a finite-rank (BBP) spiked model, and it is nested twice here: hat C spikes over the hat S MP bulk inside M_t (aspect gamma = d_lat/n), and then M_t's outliers over the W-side MP (gamma = 1/psi_p). Each level carries a separation threshold below which the outlier sticks to the bulk edge and a BBP displacement formula above it; neither appears anywhere in the appendix.

**Evidence:** Setup line: 'We work in the proportional limit dlat, nsamp, pwidth -> infinity with psi_p = pwidth/dlat and psi_n = nsamp/dlat fixed' — d_int is absent. Theorem: 'the spectrum of U ... splits almost surely into four contiguous bulks with the following counts and leading-order edges', with count d_int for the signal row. The inner threshold is non-trivial at the paper's own settings: at d_lat=200, sigma_perp=0.5, n=500 the hat S bulk edge is sigma_perp^2(1+sqrt(d_lat/n))^2 = 0.666, and the signal-block eigenvalues of M_t reach down to 1.888 — a margin of only 2.8x, which shrinks monotonically with d_lat (ratio 14.2 at d_lat=10, 4.6 at d_lat=40, 2.8 at d_lat=200). Extrapolating the same construction, the two collide near d_lat ~ 1.5e3.

**Fix:** Add d_int fixed (or d_int/d_lat -> phi) to the hypotheses and restate the signal block as a rank-d_int spiked perturbation: give the BBP threshold for each of the two nested levels and the outlier locations with their displacement corrections, rather than asserting the spikes sit at mu_1^2 alpha_t^2/d_lat. Downgrade 'almost surely' to a statement about the ESD (which legitimately covers only the noise-dim, sample and rank-null blocks) plus a separate finite-rank outlier statement for the signal modes. Note also that the signal block perturbs both the mean (hat C) and the covariance (sigma_sig^2 vs sigma_perp^2), so it is not a pure additive spike and standard BBP needs adapting.


**Referee:** Verified textually and numerically. The proportional limit at sec-appendix-fourbulk.tex:27-29 fixes only psi_p and psi_n; d_int never appears, and d_int = 5 is fixed throughout, so d_int/d_lat -> 0 and the signal block has zero mass in the limiting ESD. Theorem 1 (line 208-228) nonetheless asserts the spectrum 'splits almost surely into four contiguous bulks with the following counts and leading-order edges' with count d_int for the signal row. Almost-sure ESD convergence delivers nothing about d_int outliers; that requires a finite-rank / BBP argument, and no separation threshold or displacement formula appears anywhere in the appendix or in sec-rfnn-bounds.tex. The nesting the reviewer describes (hat C spiking over the hat S bulk inside M_t, then M_t's outliers over the W-side MP) is real. The threshold is genuinely non-trivial at the paper's own settings: lambda_min(signal block of M_t) vs the noise-dim band top of M_t gives margins 5.84 (d_lat=10), 4.88 (20), 4.31 (40), 3.51 (100), 2.86 (200) — monotone shrinkage, and solving 1.9 = e^{-2t} sigma_perp^2 (1+sqrt(d_lat/500))^2 + Delta_t puts the collision at d_lat ~ 1.5e3, matching the reviewer's extrapolation.


**Referee correction:** The reviewer's quoted margin at d_lat=10 is wrong: I get 5.8, not 14.2 (their d_lat=40 value 4.6 vs my 4.31 and d_lat=200 value 2.8 vs my 2.86 are fine). The trend, the conclusion, and the ~1.5e3 collision estimate are unaffected. Also worth noting that this shrinkage is again driven by n=500 being held fixed while d_lat sweeps — in the theorem's own stated limit (psi_n fixed) the margin would be constant, so the BBP threshold is a live concern for the experiments, not for the asymptotic statement.


---

## [MAJOR] The theorem assumes psi_p fixed, but the main-text figure and the 'strictest control' use p = d_lat + n + 300, where psi_p -> 1 and the claimed O(1/sqrt(psi_p)) MP correction reaches 585%

`psip-not-fixed-in-shown-experiments` · verdict **CONFIRMED** · kind inconsistency · effort hours  

**Where:** sec-appendix-fourbulk.tex Remark \ref{rem:MP} and sec-rfnn-bounds.tex lines 82-85 ('sub-dominant at our psi_p = 64') vs ICLR_2026/sec-rfnn.tex width controls and Fig \ref{fig:fourbulk-clean} (figures/rfnn_exp2_pndr_main_selected.pdf)

**Claim:** Every quantitative MP statement in the appendix is evaluated at psi_p = 64, but the main-text four-bulk figure and two of the three width controls do not hold psi_p fixed. For p = d_lat + n + 300 with n = 500, psi_p = 1 + 800/d_lat, which tends to 1 as d_lat grows — precisely the MP hard edge where the deformation diverges. The fixed-p = 1800 ablation is worse still at large d_lat. The theorem as stated therefore does not cover the setting the paper displays as its primary evidence.

**Evidence:** Computed psi_p and Delta_MP = (sqrt(psi_p)+1)^2/(sqrt(psi_p)-1)^2 - 1 for the p = d_lat + n + 300 control: d_lat=20 -> psi_p=41, Delta=0.88; d_lat=40 -> 21, 1.43; d_lat=100 -> 9, 3.00; d_lat=200 -> 5, 5.85. At d_lat=200 the MP band around each 'leading-order edge' is 585% wide, i.e. the notion of an edge has no content. sec-rfnn.tex describes this control as 'the strictest representative setting' shown in the main text; sec-rfnn-bounds.tex nonetheless asserts 'Marchenko-Pastur deformations of relative width O(1/sqrt(psi_p)) around each edge are sub-dominant at our psi_p = 64'.

**Fix:** Either restrict every theorem-vs-figure comparison to the p = 64 d_lat sweep, or restate the theorem for psi_p = psi_p(d_lat) and give the edge locations as deformed-MP quantities valid down to psi_p = O(1). At minimum, add an explicit sentence saying which figures the theorem's asymptotics cover and which they do not.


**Referee:** The main-text four-bulk figure is figures/rfnn_exp2_pndr_main_selected.pdf (ICLR_2026/sec-rfnn.tex:50), i.e. the p = d_lat + n + 300 control, and sec-rfnn.tex:13 says 'the main text shows the strictest representative setting'. For n=500 that gives psi_p = 1 + 800/d_lat: 41 (d_lat=20), 21 (40), 9 (100), 5 (200), so psi_p -> 1, and Delta_MP = (sqrt(psi_p)+1)^2/(sqrt(psi_p)-1)^2 - 1 = 0.88, 1.43, 3.00, 5.85 respectively (I reproduced these). At d_lat=200 the MP band around each 'leading-order edge' is 585% wide, so the edge has no quantitative content there. Meanwhile sec-rfnn-bounds.tex:82-85 asserts the MP deformations 'are sub-dominant at our psi_p = 64', and sec-appendix-fourbulk.tex Remark rem:MP evaluates everything at psi_p = 64. So the theorem's quantitative asymptotics do not cover the figure the paper displays as its primary evidence.


**Referee correction:** Two refinements. (i) There is also an internal inconsistency inside sec-rfnn.tex:13 itself: it calls the THIRD control (fixed-p with sigma_perp^2 ~ 1/(d_lat - d_int)) 'the strictest control', then says the main text shows 'the strictest representative setting' while the displayed figure is the SECOND (pndr) control. That sentence needs fixing regardless. (ii) The scope of the problem is broader than psi_p: n = 500 is fixed across every sweep, so psi_n = n/d_lat runs 100 -> 2.5 even in the p = 64 d_lat sweep. No shown experiment satisfies the theorem's proportional-limit hypotheses; restricting theorem-vs-figure comparisons to the p = 64 d_lat sweep fixes psi_p but not psi_n.


---

## [MAJOR] The status of the four-bulk structure is stated three different ways: hypothesized (sec-rfnn), observed (sec-rfnn), identified with exact boundaries (intro)

`hypothesized-vs-identified-status` · verdict **CONFIRMED** · kind inconsistency · effort hours  

**Where:** ICLR_2026/sec-intro.tex:25 vs ICLR_2026/sec-rfnn.tex:46 and :57

**Claim:** The intro asserts "bulk-count boundaries that land exactly at indices $\dint$ and $\dlat$"; the body says "we hypothesize the sorted spectrum instead splits into four bulks" and downgrades again to "We observe these four populations". The body then explicitly contradicts "exactly": at sigma_perp=0.01 the boundary at d_lat is not a boundary at all.

**Evidence:** sec-rfnn.tex:46: "we \emph{hypothesize} the sorted spectrum instead splits into \emph{four} bulks" ... "We observe these four populations in every configuration we ran". sec-rfnn.tex:57: "The cliff at index $\dint$ ... is sharp at both noise scales; the cliff at $\dlat$ between data and sample bulks is sharp at $\signoise = 0.5$ and degenerates to a smooth transition at $\signoise = 0.01$." A boundary that "degenerates to a smooth transition" cannot simultaneously "land exactly at index $\dlat$".

**Fix:** Pick one status verb and use it everywhere. If the honest status is "predicted analytically at leading order and observed numerically at sigma_perp=0.5 but not resolved at sigma_perp=0.01", say that in the Contributions list too, and delete "land exactly".


**Referee:** The 'land exactly' half is verifiably wrong at sigma_perp=0.01, and I checked it numerically rather than from prose. On the saved spectra in sigma_noise_0.01/exp2_rfnn/raw_data, the eigenvalue ratio across the claimed boundary at index d_lat, lambda_{d_lat}/lambda_{d_lat+1}, is 1.08 (d_lat=10), 1.01 (d_lat=20) and 1.14 (d_lat=40) -- i.e. no boundary at all. At sigma_perp=0.5 the same ratio is 4.4, 7.7, 13.1, 18.0, 12.3 for d_lat=10,20,40,100,200, so the boundary is real there. sec-rfnn.tex:57 says exactly this ('degenerates to a smooth transition at signoise = 0.01') while sec-intro.tex:25 claims boundaries 'land exactly at indices dint and dlat' with no noise-scale qualifier. That is a genuine, checkable contradiction between the Contributions list and the body.


**Referee correction:** The 'hypothesize' vs 'observe' part of the finding is not a defect. sec-rfnn.tex:46 states a hypothesis and then reports 'We observe these four populations in every configuration we ran' -- that is ordinary hypothesis-then-evidence narrative, not three inconsistent status claims. The single substantiated inconsistency is 'land exactly at indices dint and dlat' (intro) versus the unresolved d_lat boundary at sigma_perp=0.01 (body, and confirmed in the raw spectra). Note also a boundary the paper never checks: the predicted sample/rank-null boundary at index d_lat+n shows ratio 1.00-1.01 in every run at BOTH noise scales, so that boundary does not exist empirically anywhere.


---

## [MAJOR] "Boundaries land exactly at d_int and d_lat" is untestable under the paper's own non-adaptive coloring, and the one non-circular test was deleted from the draft

`boundary-claim-is-circular` · verdict **REFUTED** · kind gap · effort days  

**Where:** ICLR_2026/sec-rfnn.tex:57 ("deliberately non-adaptive" defence); code/v3/lib/eigenvalues.py:76-93 (bulk_indices); dropped diagnostic at ICLR_2026/sec-appendix.tex:875-890 (app:bulk-detection), absent from sec-appendix-integrated.tex

**Claim:** The defence -- "when the colored groups separate on the $x$-axis, the claim is stronger than a post-hoc visual partition" -- only escapes circularity for the WEAK claim that eigenvalue magnitude is monotone in index-block. It cannot support the STRONG claim that the boundaries are at d_int and d_lat, because no boundary is ever estimated. The plotted summary is "dashed vertical lines mark within-block median log-eigenvalues", and block medians of any monotonically decreasing spectrum are automatically ordered and distinct -- a smooth power-law spectrum with no gaps whatsoever would produce the identical picture. code/v3/lib/eigenvalues.py:bulk_indices hard-codes slices [0,d_int),[d_int,d_lat),[d_lat,d_lat+n),[d_lat+n,p), and its docstring literally says "Paper's predicted boundaries land at indices d_intrinsic and d_latent".

**Evidence:** sec-rfnn.tex:57: "we sort the eigenvalues of $\Umat$ in decreasing order and then assign labels by the \emph{a priori} index blocks predicted by the data construction ... No clustering or threshold fitting is used to place the colors". The only label-free detector in the project (scipy find_peaks, ICLR_2026/sec-appendix.tex:875, "capped at 4 by prominence") is (a) itself constrained to return four bulks and (b) has been removed from the current draft along with its table.

**Fix:** Add a label-free boundary estimate and report it as a number: for each run, compute the k largest consecutive log-eigenvalue gaps (or a change-point fit) WITHOUT using d_int, d_lat, n, and report the argmax gap indices against the predicted 5 / d_lat / d_lat+n with error bars over seeds. Also report the gap ratio lambda_{d_int}/lambda_{d_int+1} against the median within-block ratio; only that number can support "lands exactly".


**Referee:** The core assertion -- that no label-free test is possible or that a gapless spectrum would produce the same picture -- is false, and I disproved it by running the reviewer's own proposed fix. Taking the saved spectra and computing the largest consecutive log10-eigenvalue gaps WITHOUT using d_int, d_lat or n, index 5 is among the top-5 gaps in every single run at both noise scales (gap ratio lambda_5/lambda_6 = 4.95, 4.41, 3.89, 3.42, 2.72 at sigma_perp=0.5 for d_lat=10,20,40,100,200; 31.2, 50.3, 64.8 at sigma_perp=0.01), and index d_lat is among the top-5 gaps at sigma_perp=0.5 with ratios 4.4-18. A smooth power-law spectrum with no gaps would show ratios of ~1 at those indices, which is precisely what the d_lat boundary at sigma_perp=0.01 does show (1.01-1.14). So the coloring defence is not circular for the strong claim either: the histogram itself (not just the median lines) would show the colored blocks overlapping if the index cuts were in the wrong place, and the label-free numbers confirm they are not.


**Referee correction:** Two residual points survive and are worth making instead. (i) The paper never reports any of these gap-ratio numbers, so the strong claim is supportable but unsupported -- the fix is trivial (one column of numbers) and should be done. (ii) The boundary that genuinely fails the label-free test is the one the finding does not mention: the predicted sample/rank-null cut at index d_lat+n has gap ratio 1.00-1.01 in every configuration at both noise scales, so the fourth bulk boundary is never resolved. That, not the d_int cut, is where 'lands exactly' breaks down.


---

## [MAJOR] The paper's stated meaning of the generalization gap is contradicted by the repo's own synthetic data: gen-gap rises 15x with d_lat while memorization falls

`gen-gap-proxy-anticorrelates-with-memorization` · verdict **CONFIRMED** · kind error · effort days  

**Where:** ICLR_2026/sec-rfnn.tex:69 ("What the gen-gap measures"); data at /Users/ryan/Desktop/latent_space_diffusion_analysis/sigma_noise_0.5/exp2_mlp/raw_data/*/metrics.jsonl

**Claim:** sec-rfnn.tex asserts the gen-gap is a direct readout of sample-bulk absorption and is therefore delayed by the buffer. In the repo's sigma_perp=0.5 d_lat sweep the final gen-gap increases monotonically with d_lat (0.0741 at d=5 to 1.0989 at d=200) while memorization_fraction falls to zero over 10<=d<=100. Gen-gap and memorization ANTI-correlate over exactly the range the buffer is supposed to govern. The step at which gen_gap first exceeds 0.02 -- the tau_mem proxy used in the project's real-data experiments -- is flat in d_lat (5k, 1, 5k, 5k, 5k, 20k, 10k, 10k, 10k, 5k, 5k for d=5..200): no delay whatsoever. At sigma_perp=0.01 the same quantity moves the OTHER way (0.0838 at d=5 down to -0.0008 at d=200), so the gen-gap's sign of d_lat-dependence is not even stable across noise scales.

**Evidence:** Extracted final rows of sigma_noise_0.5/exp2_mlp/raw_data/di5_d*_n500_s42/metrics.jsonl: (d_lat, gen_gap, mem) = (5, 0.0741, 0.0062), (20, 0.2595, 0.0000), (100, 0.9317, 0.0000), (200, 1.0989, 0.0144). sec-rfnn.tex:69: "The gen-gap is therefore a direct measure of how much null structure the readout has absorbed, which is why it opens at $\taumem$ and why the noise-dim buffer also delays its onset".

**Fix:** Either show the gen-gap-vs-d_lat curves in the paper and explain the sigma-dependent sign reversal, or delete the "What the gen-gap measures" paragraph. It currently makes a falsifiable prediction that the project's own data falsifies at sigma_perp=0.5. This matters beyond the paragraph because the gen-gap>0.02 proxy is what generated the delay numbers still quoted in PROJECT_BRIEF.md.


**Referee:** The paragraph at sec-rfnn.tex:69 makes a falsifiable claim ('The gen-gap is therefore a direct measure of how much null structure the readout has absorbed, which is why it opens at taumem and why the noise-dim buffer also delays its onset') and the repo's data falsifies it -- including, decisively, in the protocol the paper actually uses. I verified gen_gap is per-dimension (code/v3/run_experiment.py:105-119 divides both test_loss and train_loss by d_latent), so the trend is not an output-size artifact. In the main-text run (multiseed_runs/exp2_mlp_dlat_sn05_5m_bonnaire_hidden256, hidden=256, 5M steps, 5 seeds) the median first step crossing gen_gap>0.02 is 50,000 for EVERY d_lat from 5 to 40, and the final gen-gap is flat at 0.797-0.839, while over the same sweep the first step with memorization >1% moves from 150k to never. The gen-gap therefore carries zero signal about tau_mem or about the buffer. I also reproduced the reviewer's sigma_perp=0.5 numbers exactly (final gen_gap 0.0741 at d=5 -> 1.0989 at d=200 with mem 0.0062 -> 0.0144 and zero in between) and the sign reversal at sigma_perp=0.01 (0.0838 -> -0.0008).


**Referee correction:** One protocol correction: the 15x-rising gen-gap numbers come from sigma_noise_0.5/exp2_mlp, which is a different sweep (config.json shows hidden=1600=8*d_lat, 300k steps, single seed 42), not the paper's fixed-width-256 5M five-seed protocol. In the paper's own protocol the gen-gap is flat in d_lat rather than anticorrelated. The claim in sec-rfnn.tex:69 is falsified either way -- it is uninformative in the main protocol and anticorrelated in the scaled one -- but the finding should cite the fixed-width run, which is both the relevant protocol and the more damning evidence.


---

## [MAJOR] PROJECT_BRIEF.md's "no loss in image quality" contradicts the abstract, the discussion, sec-real-data, and the repo's own synthetic score-error data

`brief-quality-claim-contradicts-paper-and-data` · verdict **CONFIRMED** · kind inconsistency · effort hours  

**Where:** /Users/ryan/Desktop/latent_space_diffusion_analysis/PROJECT_BRIEF.md (one-liner, "Why this is more than a hunch", "Quick answers") vs ICLR_2026/abstract.tex, sec-real-data.tex:sec:real-mem-fid, sec-discussion.tex:14; data at sigma_noise_0.5/exp2_mlp and sigma_noise_0.01/exp2_mlp

**Claim:** The brief makes quality-neutrality a load-bearing part of the pitch ("with no loss in image quality", "it's free: image quality doesn't drop", "quality is flat across the whole range we tested", "Latent width isn't just a compute knob - it's a free memorization-safety knob"). The paper says the opposite, and so does the raw data. Note the logged score_error is ALREADY per-dimension (code/v3/lib/metrics.py:47 divides by test_x.shape[-1]), so these are the normalized numbers the paper plots, not an artifact of output size.

**Evidence:** abstract.tex: "FID follows a separate quality tradeoff, improving as the VAE bottleneck is relieved and then worsening when the latent score problem becomes too wide." sec-discussion.tex:14: "very wide latents again reduce memorization while often hurting FID." sec-real-data.tex: "worsens again at large $\dlat$, giving a quality tradeoff rather than a monotone benefit". Synthetic per-dim score error, final step, sigma_perp=0.5: 0.1415 (d=5) -> 3.2128 (d=200), monotone, 23x worse. sigma_perp=0.01: 0.1473 (d=5) -> 5.3250 (d=15), 36x worse at d_lat/d_int = 3, which is precisely the ratio the brief recommends as the safe operating point.

**Fix:** Rewrite the brief's one-liner and the "Quick answers" entry to state the tradeoff the paper actually reports. If the brief is a public-facing artifact, this is the highest-risk sentence in the project.


**Referee:** All quotes verified verbatim. PROJECT_BRIEF.md one-liner: 'gives you a free "buffer" that delays memorization -- with no loss in image quality'; 'And it's free: image quality doesn't drop'; Quick answers: 'No -- quality is flat across the whole range we tested'; killer line: 'Latent width isn't just a compute knob - it's a free memorization-safety knob.' The paper says the opposite in three places: abstract.tex ('FID follows a separate quality tradeoff, improving as the VAE bottleneck is relieved and then worsening when the latent score problem becomes too wide'), sec-real-data.tex:45 ('worsens again at large dlat, giving a quality tradeoff rather than a monotone benefit from larger codes'), sec-discussion.tex:14 ('very wide latents again reduce memorization while often hurting FID'), and sec-discussion.tex:23-26 ('The best latent dimension is neither the smallest code nor the largest code'). I confirmed score_error is per-dimension (code/v3/lib/metrics.py:47 and run_experiment.py:114 both divide by d_latent) and reproduced the synthetic numbers: sigma_perp=0.5 final score error 0.1415 (d=5) -> 3.2128 (d=200); sigma_perp=0.01 0.1473 (d=5) -> 5.3250 (d=15). The brief's central selling point is the one claim the paper explicitly retracts.


**Referee correction:** Minor scoping: the synthetic per-dimension score error is a fixed-capacity score-fitting metric, not image quality, so it corroborates rather than proves the point; the decisive contradiction is with the paper's own FID results. Also note the score-error sweeps cited are from the archived 300k/capacity-scaled runs, not the main-text protocol. Neither weakens the finding -- the abstract alone refutes the brief.


---

## [MAJOR] The brief's headline "~8x (MNIST) and ~15x (CelebA)" delay comes from a deleted experiment measured with the discredited gen-gap proxy, on a dataset the paper no longer contains

`brief-8x-15x-numbers-are-orphaned` · verdict **CONFIRMED** · kind unsupported-claim · effort hours  

**Where:** PROJECT_BRIEF.md "Why this is more than a hunch" item 3; source: /Users/ryan/Desktop/latent_space_diffusion_analysis/ICLR_2026/sec-appendix.tex:425-450 (file NOT \input by main.tex)

**Claim:** Those numbers trace to the old real-data appendix ($10k \to 80k$ on MNIST, $30k \to 440k$ on CelebA), which (a) is not in the current draft, (b) contains no MNIST anywhere in the current draft (grep -i mnist over the compile set returns nothing), (c) measured tau_mem by the gen-gap>0.02 proxy explicitly BECAUSE the nearest-neighbor memorization test would not fire, and (d) claimed flat FID, which the current paper now denies. The brief also passes the old text's own "$\sim$$10{\times}$" through as separate 8x and 15x endpoints.

**Evidence:** ICLR_2026/sec-appendix.tex:430-436: "$\taumem$ grows with $\dlat$ on both datasets, going from $10\,k$ to $80\,k$ steps on MNIST and from $30\,k$ to $440\,k$ steps on CelebA across the sweep---a $\sim$$10{\times}$ delay". Same file:420-427: "we use the gen-gap proxy here rather than the \citet{somepalli2023diffusion}-style memorization fraction because the pixel-space nearest-neighbor threshold rarely triggers at $\nsamp = 1000$". Same file:439-441: "minimum FID is roughly flat across the sweep, so the delay benefit of larger $\dlat$ does not come at the cost of generation quality" -- directly contradicted by sec-real-data.tex today.

**Fix:** Delete the 8x/15x numbers from the brief or requote them from the current CelebA/CIFAR-10 runs, and state the metric (final memorization fraction at 5M steps, not time-to-memorize). If a delay-factor number is wanted, it has to be recomputed from the NN-ratio trajectories, not the gen-gap proxy.


**Referee:** Every element verified. ICLR_2026/sec-appendix.tex:429-434 is the source ('going from 10k to 80k steps on MNIST and from 30k to 440k steps on CelebA across the sweep---a ~10x delay'). That file is not \input by main.tex. grep -i mnist over all ten files in the compile set returns nothing. sec-appendix.tex:420-424 states the metric explicitly: 'we use the gen-gap proxy here rather than the somepalli2023diffusion-style memorization fraction because the pixel-space nearest-neighbor threshold rarely triggers at nsamp = 1000' -- and I independently showed above (finding gen-gap-proxy) that in the paper's own fixed-width protocol the gen-gap>0.02 crossing is pinned at 50k for every d_lat while true memorization onset moves 30x, so the proxy does not track tau_mem. sec-appendix.tex:439-441 'minimum FID is roughly flat across the sweep, so the delay benefit of larger dlat does not come at the cost of generation quality' is directly contradicted by sec-real-data.tex:45 today. So the brief's flagship evidence item 3 rests on a deleted experiment, a discredited proxy, a dataset the paper no longer contains, and a quality claim the paper now denies.


**Referee correction:** Drop the last sub-point. The brief's 8x and 15x are the arithmetically correct endpoint ratios (10k->80k = 8x, 30k->440k = 14.7x); the source's '~10x' is the rounding, not the brief's. The defect is provenance and metric, not arithmetic.


---

## [MAJOR] The 2x2 taxonomy in the text names four bulks that are not the four bulks in the figure, the coloring code, or the theorem

`four-bulk-taxonomy-mismatch` · verdict **CONFIRMED** · kind inconsistency · effort hours  

**Where:** ICLR_2026/sec-rfnn.tex:46 (the 2x2) vs sec-rfnn.tex:57 (the coloring), Figure fig:fourbulk-clean caption, code/v3/lib/eigenvalues.py:76-93, and Theorem thm:fourbulk

**Claim:** The prose defines the four bulks as the 2x2 (shared vs. sample-specific) x (signal vs. null): bulk_data_signal, bulk_data_noise, bulk_sample_signal, bulk_sample_noise -- and then immediately collapses two of them ("Together (iii) and (iv) make up Bonnaire's $\rho_1$"). The figure, the code, and the theorem use a different four-way split: signal / noise-dim / sample / RANK-NULL. Rank-null does not appear in the 2x2 at all, and the sample bulk is never split into signal and null halves anywhere except in that sentence. So the paper presents two incompatible four-way decompositions as if they were the same decomposition.

**Evidence:** sec-rfnn.tex:46 lists (i)-(iv) as the 2x2 and then says (iii)+(iv) are one bulk. sec-rfnn.tex:57: "indices $\dlat{+}1{:}\dlat{+}\nsamp$ are sample-specific, and any remaining feature directions are rank-null." Theorem thm:fourbulk's table: Signal / Noise-dim / Sample / Rank-null. code/v3/lib/eigenvalues.py BulkLayout fields: signal, noise_dim, sample, rank_null.

**Fix:** Drop the 2x2 framing or keep it and admit it predicts a FIFTH population (sample-signal vs sample-null should be separable at sigma_sig^2/sigma_perp^2 = 10^4, which would be a genuine falsifiable test the paper could run). As written the taxonomy is decorative and inconsistent with everything downstream.


**Referee:** Real internal inconsistency, verified across all four artifacts. sec-rfnn.tex:46 says the spectrum 'splits into four bulks along the 2x2 axes (shared across all training points vs. unique to individual training points) x (signal vs. null subspace)' and enumerates (i) bulk_data_signal, (ii) bulk_data_noise, (iii) bulk_sample_signal, (iv) bulk_sample_noise -- then immediately says 'Together (iii) and (iv) make up Bonnaire's rho_1', collapsing the enumerated four into three. The actual four-way split used everywhere downstream is signal / noise-dim / sample / rank-null: sec-rfnn.tex:57 ('indices dlat+1:dlat+nsamp are sample-specific, and any remaining feature directions are rank-null'), the fig:fourbulk-clean caption ('red is signal/source, green is latent-null/noise-dimension, blue is sample-specific, and purple is rank-null'), code/v3/lib/eigenvalues.py BulkLayout(signal, noise_dim, sample, rank_null), and Theorem thm:fourbulk's table. Rank-null is in neither cell of the 2x2, and the sample bulk is split into signal/null halves nowhere else in the corpus.


**Referee correction:** Slightly narrow the framing: rank-null is not 'absent from the paper' -- it is defined 11 lines later at sec-rfnn.tex:57 -- so this is one muddled definitional sentence rather than two competing decompositions running through the paper. It matters because that sentence is the one that defines the paper's headline contribution. The suggested falsifiable test is worth keeping: if the 2x2 were meant literally, sample-signal and sample-null should separate at sigsig^2/sigperp^2 = 10^4, and they demonstrably do not (I find no gap at index d_lat+n at either noise scale).


---

## [MAJOR] The mode-absorption clock contains no sample-bulk term, so it cannot instantiate the mechanism it is presented as instantiating

`predictor-omits-the-sample-bulk` · verdict **CONFIRMED** · kind missing-theory · effort weeks  

**Where:** ICLR_2026/sec-spectral-predictor.tex (definition of $M_t(d)$ and $P_d(s)$); ICLR_2026/sec-appendix-integrated.tex app:spectral-predictor-clean

**Claim:** In the theory, tau_mem is set by the SAMPLE bulk, whose eigenvalues come from the higher-Hermite diagonal mass etabar_star and do not appear in the data covariance at all. The predictor's $M_t(d)=e^{-2t}Z_d^\top Z_d/n+(1-e^{-2t})I_d$ is a pure d x d data-covariance object: it has exactly the population + noise-dim modes and no sample bulk. Its d_lat-dependence comes entirely from $w_i=1$ averaging over more small eigenvalues in $P_d(s)$, i.e. the queue intuition is baked into the estimator by construction, not derived. The appendix concedes the missing term.

**Evidence:** sec-appendix-integrated.tex (app:spectral-predictor-clean): "A real $n$-dependence for the 1\% time would require recomputing $M_{t,n}(d)$ on different $n$-sized subsets, or adding an explicit sample-count/sample-bulk term to the predictor." The same appendix notes the Figure~\ref{fig:app-spectral-by-n} curves "are flat: this is not evidence that true memorization dynamics are independent of $n$". But the theory says the memorization bulk has size exactly n -- an estimator whose predictions are n-independent cannot be a model of it.

**Fix:** Add the etabar_star / rank-n term to $M_t(d)$ (or a surrogate for it) and show the predictor still works; or relabel the section as an empirical spectral heuristic and remove the "mode-absorption clock" framing that ties it to the theory.


**Referee:** Verified from the definition itself. sec-spectral-predictor.tex defines M_t(d) = e^{-2t} Z_d^T Z_d/n + (1-e^{-2t}) I_d, a d x d object built entirely from the latent data covariance; its spectrum therefore contains only population and noise-dim modes. In the theory (sec-appendix-fourbulk.tex, Lemma lem:Udiag), the sample bulk is a rank-n object whose eigenvalues come from etabar_star = sum_{k>=3} mu_k^2 E[(||x||^2/dlat)^k], the higher-Hermite diagonal mass, which does not appear in the data covariance at all. So the clock literally cannot represent the bulk that defines tau_mem. With w_i = 1 in P_d(s) = sum_i w_i (1-exp[-kappa lambda_i s])^2 / sum_i w_i, all d-dependence enters through averaging over a larger set of small eigenvalues -- the queue intuition is the estimator's construction, not a derived consequence. The appendix concedes the gap in the words the finding quotes ('adding an explicit sample-count/sample-bulk term to the predictor') and concedes the n-independence is an artifact ('this is not evidence that true memorization dynamics are independent of n').


**Referee correction:** The paper is more careful than the finding implies, so this is a framing fix rather than a false claim. sec-spectral-predictor.tex frames the section as an empirical test ('the test asks whether frozen latent geometry contains the right temporal ordering'), and sec-discussion.tex:58-60 explicitly says the predictor is 'an empirical frozen-VAE latent-spectrum rule, not yet a theorem.' What needs changing is the 'mode-absorption clock' label in the abstract and Contribution 5, which implies the theory's mechanism is being instantiated when the sample-bulk term is absent.


---

## [MAJOR] The spectral predictor's three free parameters are fitted on the same dimensions it is evaluated on; no held-out test is reported

`predictor-is-fitted-not-predicted` · verdict **CONFIRMED** · kind unsupported-claim · effort days  

**Where:** ICLR_2026/abstract.tex (last sentence), sec-intro.tex:29 (contribution 5), sec-spectral-predictor.tex, sec-appendix-integrated.tex app:spectral-predictor-clean

**Claim:** The abstract and contribution 5 present the clock as estimating memorization percentage from the frozen VAE alone. It also requires $(\kappa,a,b)$, which are fitted to the observed five-seed memorization curves of the very dimensions being plotted. There is no leave-one-d_lat-out, no cross-dataset transfer, and no cross-architecture transfer reported, so the figure showing predicted vs. observed hitting times is an in-sample fit with 3 parameters against a handful of dimensions.

**Evidence:** sec-appendix-integrated.tex: "The constants $(\kappa,a,b)$ are fit once per dataset using the dimensions where the VAE already preserves pixel-space identity: $d\ge70$ for CelebA and $d\ge140$ for CIFAR-10." sec-intro.tex:29 says only "we encode the same $n$ images with the frozen VAE, compute the covariance spectrum of the latent codes, and convert that spectrum into a mode-absorption clock that outputs an estimated memorization percentage over training time" -- the calibration step is not mentioned. sec-discussion.tex:59 does concede "an empirical frozen-VAE latent-spectrum rule, not yet a theorem".

**Fix:** Report at least a leave-one-dimension-out fit (fit (kappa,a,b) on all post-threshold d except one, predict the held-out one) and, ideally, transfer of $(\kappa,a,b)$ from CelebA to CIFAR-10. State in the abstract and contribution 5 that a per-dataset calibration is required.


**Referee:** Verified. sec-appendix-integrated.tex: 'The constants (kappa,a,b) are fit once per dataset using the dimensions where the VAE already preserves pixel-space identity: d>=70 for CelebA and d>=140 for CIFAR-10' -- the same post-threshold dimensions that are then plotted in fig:real-spectral-predictor and in the predicted-vs-observed hitting-time panels. grep for 'held-out', 'holdout', 'leave-one', 'cross-validat', 'transfer' across the entire compile set returns exactly one hit, and it is about a DiT pilot, not about the predictor. So no leave-one-dimension-out, no cross-dataset transfer of (kappa,a,b), and no cross-architecture check is reported anywhere. abstract.tex and sec-intro.tex:29 both describe the clock as being built 'from the frozen VAE latent spectrum' with no mention that a three-parameter per-dataset calibration against observed memorization curves is required.


**Referee correction:** Give the paper the credit it is due on one point: sec-spectral-predictor.tex does disclose that ONE shared response is used across all latent widths within a dataset ('The same response is used across latent widths within a dataset, so the test asks whether frozen latent geometry contains the right temporal ordering'), which is a meaningful constraint -- the d_lat-dependence is not fitted per dimension, only the global (kappa,a,b). The two substantiated defects are therefore (i) the abstract and Contribution 5 omit the calibration step entirely, and (ii) no held-out or transfer evaluation exists, so the reported agreement is in-sample.


---

## [MAJOR] sec-mlp explains the MLP result by a mechanism (feature learning collapsing onto the signal subspace) that shrinks the buffer and therefore predicts EARLIER memorization

`mlp-mechanism-self-contradiction` · verdict **PLAUSIBLE** · kind inconsistency · effort days  

**Where:** ICLR_2026/sec-mlp.tex:41 vs ICLR_2026/sec-intro.tex:27 (contribution 3)

**Claim:** Contribution 3 says "We confirm the buffer mechanism in trainable MLP score networks". The body then says the MLP does something else: the first layer learns to project onto the signal subspace instead of spending effort on all modes. If the trainable network discards the null directions, the noise-dim bulk it must traverse is smaller, not larger -- the buffer explanation is voided, and only the sign of the effect coincides. The project's own planning note states the consequence explicitly.

**Evidence:** sec-mlp.tex:41: "The MLP result is qualitatively sharper than the frozen-feature prediction: at high $\dlat$, feature learning appears to find sparser, signal-aligned directions instead of spending equal effort on all random-feature modes. Nevertheless, the direction of the effect agrees with the buffer picture". _next_steps/theory_plan.md (kevin's section): "the trainable first layer learns to project onto the $d_{int}$-dim signal subspace at large $d_{lat}$, effectively shrinking the buffer."

**Fix:** Downgrade contribution 3 to "the sign of the effect replicates in a trainable score network, though the frozen-feature mechanism does not account for its magnitude", and either run the weight-projection diagnostic the plan lists (project the trained first layer onto the ground-truth Q) or drop the sparse-alignment speculation.


**Referee:** The textual tension is real but the finding's central inference is not established. Verified quotes: sec-intro.tex:27 'We confirm the buffer mechanism in trainable MLP score networks'; sec-mlp.tex:41 'at high dlat, feature learning appears to find sparser, signal-aligned directions instead of spending equal effort on all random-feature modes. Nevertheless, the direction of the effect agrees with the buffer picture'; _next_steps/theory_plan.md:99 'the trainable first layer learns to project onto the d_int-dim signal subspace at large d_lat, effectively shrinking the buffer.' However, the step 'shrinking the buffer therefore predicts EARLIER memorization' is the reviewer's own inference and is nowhere established: a first layer that discards null directions could equally reduce the capacity available to fit sample-specific directions. Moreover theory_plan.md:99 offers the shrinking-buffer hypothesis to explain the n-shape in late SCORE ERROR at sigma_perp=0.01 (figs 14/16), not to explain memorization, so the plan does not 'state the consequence explicitly' as claimed. I could not verify the mechanism either way: no weight-projection diagnostic exists anywhere in code/ (theory_plan.md:112 lists it as a to-do), so the sparse-alignment sentence is pure speculation with no supporting measurement.


**Referee correction:** The substantiated version is narrower: Contribution 3's word 'confirm' overstates what sec-mlp.tex:41 itself claims ('the direction of the effect agrees'), and the sparse-alignment explanation offered in that sentence has no supporting diagnostic anywhere in the repo -- the weight-projection probe theory_plan.md:112 specifies was never run. Recommend downgrading Contribution 3 to 'the sign of the effect replicates in a trainable score network' and either running the projection probe or deleting the speculation. Do not assert that the mechanism predicts earlier memorization; that is unverified.


---

## [MAJOR] The real-data decline at large d_lat is not separated from the score problem simply getting harder at fixed hidden width; the paper's own FID curve is the signature of that confound

`underfitting-confound-not-separated` · verdict **—** · kind gap · effort weeks  

**Where:** ICLR_2026/abstract.tex, sec-real-data.tex:sec:real-mem-fid, sec-discussion.tex limitations (i) and (v)

**Claim:** Every real-data run holds hidden width at 256 while d_lat grows to 200+. Under that protocol "memorization falls" and "the model fits its training set worse" are the same observable, and the FID panel showing quality degrading at exactly the widths where memorization declines is what an underfitting explanation predicts. The abstract nevertheless states the buffer reading as the finding. The synthetic evidence points the same way: per-dimension score error degrades 23x (sigma_perp=0.5) over the same sweep in which memorization goes to zero.

**Evidence:** sec-discussion.tex limitation (i): "If hidden width, depth, or training compute are increased with $\dlat$, high-dimensional models may recover better FID and may also memorize more because the extra capacity can fit sample-specific directions". Limitation (v): "high-dimensional VAEs can degrade FID through optimization difficulty. Separating these effects more cleanly is the main next ablation." abstract.tex nonetheless: "once the VAE is wide enough to preserve pixel-space sample identity, increasing latent width reduces the memorization rate."

**Fix:** Run at least one width-scaled arm (hidden width proportional to d_lat, or equal-loss-matched training) at the two extreme d_lat values. If memorization still declines when the model fits equally well, the buffer reading survives; if not, the paper's central real-data claim is the confound. Until then the abstract should say "at fixed score-model capacity".


---

## [MAJOR] Lemma 1 drops the additive (||sigma||^2 - mu_1^2) I_p term that is present in the published expression for Utilde, and that omission is what makes the buffer bound diverge

`missing-identity-floor-in-lemma1` · verdict **—** · kind error · effort days  

**Where:** /Users/ryan/Desktop/latent_space_diffusion_analysis/sec-appendix-fourbulk.tex:113-131 (Lemma 1 and proof), Eq (U-lin) :70-71; vs. Bonnaire SM Eq (85)

**Claim:** Bonnaire's Gaussian equivalent of the population feature covariance is Utilde = (mu_1^2(t)/Gamma_t^2) W Sigma_t W^T/d + (||sigma||^2 - mu_1^2(t)) I_p. The paper's U^lin keeps only the first term. The dropped isotropic floor is O(1) and dominates whenever mu_1^2 beta_t^2 is small -- exactly the sig_perp -> 0 regime the paper cares most about (sig_perp = 0.01, t = 0.01 gives beta_t^2 ~ 0.02). Consequently the noise-dim eigenvalue never falls below the floor, the noise-dim relaxation time is bounded above by 1/(||sigma||^2 - mu_1^2), and the buffer bound's advertised 1/beta_t^2 blow-up as sig_perp -> 0 is an artifact of the omission.

**Evidence:** Bonnaire SM Eq (85): "Utilde = (mu_1^2(t)/Gamma_t^2) (W Sigma_t W^T)/d + (||sigma||^2 - mu_1^2(t)) I_p". sec-appendix-fourbulk.tex Eq (U-lin): "Umat^{lin} = (mu_1^2/(p d)) W M_t W^T" with no I_p term; Lemma 1 then reports edges Theta(mu_1^2 alpha_t^2/psi_p) and Theta(mu_1^2 beta_t^2/psi_p).

**Fix:** Add the (||sigma||^2 - mu_1^2(t)) I_p term, recompute all four edges as floor + lift, and recheck the "Bulk-gap scaling" paragraph and Eq (buffer-bound) with the floor in place. Cite Bonnaire SM Eq (85) rather than re-deriving. Note also that Lemma 1's proof step "the nonzero eigenvalues ... are the eigenvalues of M_t rescaled by psi_p" is inconsistent with Eq (U-lin)'s own 1/(p d_lat) prefactor (that gives a 1/d_lat scale, not a 1/psi_p scale); the published expression fixes this too.


---

## [MAJOR] The entire Hermite/kernel-decomposition derivation and the fixed-W concentration remark can be replaced by two citations (Bonnaire Lemma C.1 under arbitrary Sigma; Goldt et al. 2021 / Hu & Lu 2023 for rigour)

`cite-gep-not-rederive-hermite` · verdict **—** · kind improvement · effort days  

**Where:** /Users/ryan/Desktop/latent_space_diffusion_analysis/sec-appendix-fourbulk.tex:40-100 (kernel decomposition Eq (kernel-decomp), U-split, Remark rem:fixedW)

**Claim:** Eq (kernel-decomp) -- mu_1^2 x^T y/d off-diagonal plus a diagonal-only residual -- is the Gaussian Equivalence Principle, and Bonnaire prove exactly it for arbitrary Sigma in SM Lemma C.1, giving the explicit surrogate U = GG^T/n + b_t^2 WW^T/d + s_t^2 I_p with G = e^{-t} a_t W X'/sqrt(d) + v_t Omega, X' columns ~ N(0, Sigma). This surrogate simultaneously supplies the rank-n "sample" piece the paper struggles to produce (it is the Omega part of G), the anisotropic linear piece, and the s_t^2 floor. Remark rem:fixedW (fixed-W concentration) is precisely what the universality theorems of Hu & Lu (2023, IEEE Trans. Inf. Theory 69(3):1932-1964) and Goldt, Loureiro, Reeves, Krzakala, Mezard, Zdeborova (2021) supply. Roughly 1.5 pages of hand-derivation can be deleted and replaced by a paragraph of citations, freeing the workshop page budget for the part that is actually new.

**Evidence:** Bonnaire SM Lemma C.1 (GEP for U): "the matrix U ... has the same spectrum as its Gaussian equivalent U = GG^T/sqrt(n)sqrt(n) + b_t^2 WW^T/d + s_t^2 I_p where G = e^{-t} a_t W X'/sqrt(d) + v_t Omega, X' in R^{d x n} is a matrix whose columns x'_nu are sampled according to N(0, Sigma)". Bonnaire p.7 lists their standard-RF assumptions citing [41] Peche, [17] Goldt et al., [22] Hu & Lu.

**Fix:** Replace sec-appendix-fourbulk.tex:40-100 with: (a) statement of Bonnaire Lemma C.1 specialized to the two-block Sigma; (b) citations to Goldt et al. 2021 and Hu & Lu 2023 for the universality that upgrades the W-average to a concentration statement. Add all three to references.bib. Note for the authors: the theory_plan.md claim that the sample bulk requires "the Hermite-2 / quadratic correction ... the Pennington-Worah term mu_2" is wrong -- tanh is odd so mu_2 = 0; the correct object is the GEP noise variance v_t^2 (Bonnaire Eq (13)), and the appendix's k>=3 bookkeeping is the right one. The conflict flagged in theory_plan.md resolves in the appendix's favour.


---

## [MAJOR] George & Macris (2026) give exact RFNN-score learning curves for manifold data as a function of psi_D = D/d and report that memorization error shrinks as psi_D decreases -- the paper's headline, in a rigorous framework, published

`george-macris-manifold-near-scoop` · verdict **—** · kind gap · effort hours  

**Where:** /Users/ryan/Desktop/latent_space_diffusion_analysis/ICLR_2026/references.bib (absent); claim staked at ICLR_2026/abstract.tex and sec-intro.tex:26 (contribution 2)

**Claim:** "Asymptotic Learning Curves for Diffusion Models with Random Features Score and Manifold Data" (Anand Jerry George, Nicolas Macris, arXiv:2603.22962v3, June 2026) uses the *same* RFNN score parameterization, a hidden-manifold data model with intrinsic dimension D in ambient dimension d, the same proportional limit with psi_D = D/d, psi_n = n/d, psi_p = p/d, and derives asymptotically exact test, train and score errors. They explicitly report the intrinsic-vs-ambient-dimension effect on memorization: "the error due to memorization is smaller as psi_D decreases for a fixed psi_n" and "lower-dimensional manifolds have more orthogonal directions, which favors a smaller score error." That is the paper's headline claim (excess ambient/latent dimension relative to intrinsic dimension suppresses memorization) obtained rigorously and published before submission. They also show the benefit degrades for nonlinear manifolds -- which is a direct, testable challenge to the paper's CelebA/CIFAR-10 extrapolation from a linear Gaussian mixture.

**Evidence:** arXiv:2603.22962v3, Sec. 3: "For linear manifolds, we observe that the error due to memorization is smaller as psi_D decreases for a fixed psi_n"; "we observe that the sample complexity grows approximately linearly with psi_D, indicating that it is governed by the intrinsic dimension of the data manifold rather than the ambient dimension"; abstract: "within our model, the benefits of low-dimensional structure starts to diminish once we have a non-linear manifold." No matching entry in ICLR_2026/references.bib.

**Fix:** Cite and distinguish in the intro. The honest distinction available: their control axis is psi_n (sample complexity of the ridge/interpolation minimizer at convergence); the paper's is *training time* at fixed n, i.e. where in the eigenvalue-ordered absorption schedule the model is stopped. That is a real difference, but it must be stated, and their nonlinear-manifold negative result must be acknowledged as a limitation on the real-data claims (Sec. sec-real-data, and Limitation (iii) in sec-discussion.tex).


---

## [MAJOR] Achilli, Ambrogioni, Lucibello, Mezard & Ventura already publish an explicit formula for a memorization-collapse time as a function of dataset size and the intrinsic/ambient dimension ratio alpha_D

`achilli-ventura-manifold-memorization` · verdict **—** · kind gap · effort hours  

**Where:** /Users/ryan/Desktop/latent_space_diffusion_analysis/ICLR_2026/references.bib (absent); claim staked at ICLR_2026/abstract.tex and sec-intro.tex:26

**Claim:** "Memorization and Generalization in Generative Diffusion under the Manifold Hypothesis" (arXiv:2502.09578) considers P data points in N dimensions confined to a latent subspace of dimension D = alpha_D N under the Hidden Manifold Model, and derives "an explicit formula for t_c as a function of P and alpha_D" showing "the curse of dimensionality is avoided for structured data (alpha_D << 1), even with nonlinear manifolds." In words, this is "small intrinsic-to-ambient dimension ratio delays memorization" -- verbally identical to the paper's headline. Its companion, Achilli et al., "Losing dimensions: Geometric memorization in generative diffusion" (arXiv:2410.08727), builds a theory of memorization from the eigenvalue spectrum of the empirical score Jacobian and finds spectral gaps signalling progressive dimensional collapse -- the closest existing "spectral account of memorization". Neither is cited. Both are by overlapping author groups with the paper's own headline reference (Mezard, Ambrogioni, Ventura), so a NeurIPS/ICML referee will know them.

**Evidence:** arXiv:2502.09578 abstract: "we consider a set of P data points in N dimensions confined to a latent subspace of dimension D = alpha_D N, following the Hidden Manifold Model (HMM) ... An explicit formula for t_c as a function of P and alpha_D shows that the curse of dimensionality is avoided for structured data (alpha_D << 1), even with nonlinear manifolds." arXiv:2410.08727: "a theory of geometric memorization based on the analysis of the eigenvalue spectrum of the Jacobian of the empirical score function, finding the emergence of spectral gaps." grep over ICLR_2026/references.bib for "achilli", "ventura", "ambrogioni" returns nothing.

**Fix:** Cite both and state the distinction crisply: their t_c is a *backward-diffusion* (generative-process) collapse time computed for the exact empirical score with no learning dynamics; this paper's tau_mem is a *training* time for a learned score. That distinction is real and defensible, but the abstract as written ("excess dimensions ... buffer ... delays memorization") reads as the Achilli et al. claim, and needs one sentence separating them.


---

## [MAJOR] The paper's practical contribution (spectrum-informed early-stopping / checkpoint auditing) is preempted by Favero, Sclocchi & Wyart, uncited

`favero-sclocchi-wyart-early-stopping` · verdict **—** · kind gap · effort days  

**Where:** /Users/ryan/Desktop/latent_space_diffusion_analysis/ICLR_2026/sec-discussion.tex ("Practical implication", "Implications" paragraphs); references.bib (absent)

**Claim:** "Bigger Isn't Always Memorizing: Early Stopping Overparameterized Diffusion Models" (Favero, Sclocchi & Wyart, arXiv:2505.16959) establishes the empirical law that memorization time is proportional to dataset size across image *and* language diffusion models and proposes "a principled early-stopping criterion -- scaling with dataset size -- [that] can effectively optimize generalization while avoiding memorization, with direct implications for hyperparameter transfer and privacy-sensitive applications." sec-discussion.tex proposes exactly that workflow ("Checkpoint audits or early-stopping rules can then be placed near the predicted iso-memorization times") without citing it. The paper's genuine increment is that its criterion is indexed by d_lat rather than n and is computable from a frozen encoder before diffusion training -- but that has to be said against the existing criterion, not in a vacuum.

**Evidence:** arXiv:2505.16959 abstract: "results ranging from image to language diffusion models systematically support the empirical law that memorization time is proportional to the dataset size"; "a principled early-stopping criterion -- scaling with dataset size -- can effectively optimize generalization while avoiding memorization". ICLR_2026/sec-discussion.tex: "Checkpoint audits or early-stopping rules can then be placed near the predicted iso-memorization times." No matching bib entry.

**Fix:** Cite Favero-Sclocchi-Wyart in the intro and in sec-discussion.tex, and reframe the predictor as a *d_lat* axis complementing their *n* axis, computable pre-training from a frozen VAE. Ideally run the head-to-head: their n-scaling criterion vs the frozen-latent clock on the same CelebA sweep.


---

## [MAJOR] The three RMT citations carrying the anisotropic argument (Pennington-Worah, Benigni-Peche, El Karoui) do not cover anisotropic input covariance; the correct references are Silverstein-Bai and Louart-Liao-Couillet

`wrong-rmt-citations-for-anisotropic-spectrum` · verdict **—** · kind unsupported-claim · effort hours  

**Where:** /Users/ryan/Desktop/latent_space_diffusion_analysis/sec-appendix-fourbulk.tex:49 (citep after Eq (kernel-decomp)), :59 (El Karoui claim), :127 (Lemma 1 proof), :179 (Remark rem:fixedW), :269-270 ("Toward a full Stieltjes derivation")

**Claim:** Pennington & Worah (2017) and Benigni & Peche (2021) both analyse M = (1/m) YY* with Y = f(WX) where W and X have i.i.d. centered entries -- i.e. isotropic X. Neither states a result for a general (let alone block-anisotropic) input covariance, so neither supports Lemma 1 or Eq (kernel-decomp) as invoked. El Karoui (2010) analyses n x n kernel matrices with entries f(X_i^T X_j/p) or f(||X_i - X_j||^2/p) -- a different ensemble from the p x p feature second-moment matrix U -- so ":59, "establishes the same off-diagonal/diagonal split at the level of kernel random matrices" overstates the transfer. The results that actually do the job exist and should be named: (a) Silverstein & Bai (1995), "On the empirical distribution of eigenvalues of a class of large dimensional random matrices", J. Multivariate Anal. 54(2):175-192 -- the self-consistent equation for the ESD of (1/n) X^T T X with general population covariance T, which is exactly what turns the two-atom rho_Sigma into the two data-side bulks of W Sigma_t W^T/d (Bonnaire cite it as [48] for precisely this); and (b) Louart, Liao & Couillet (2018), "A random matrix approach to neural networks", Ann. Appl. Probab. 28(2):1190-1248, which gives a deterministic equivalent for the resolvent of (1/T) Sigma^T Sigma with Sigma = sigma(WX) for an *arbitrary deterministic data matrix X of bounded norm* -- the only one of the named papers whose hypotheses admit a block-structured X.

**Evidence:** Louart-Liao-Couillet abstract: "we study the Gram random matrix model G = (1/T) Sigma^T Sigma, Sigma = sigma(WX), ... where X is a data matrix of bounded norm, W is a matrix of independent zero-mean unit variance entries ... the resolvent Q = (G + gamma I_T)^{-1} ... has a similar behavior as that met in sample covariance matrix models, involving notably the moment Phi = (T/n) E[G], which provides a deterministic equivalent for the empirical spectral measure of G." El Karoui 2010 abstract: "n x n matrices whose (i,j)th entry is f(X_i' X_j/p) or f(||X_i - X_j||^2/p)". Bonnaire reference [48] is Silverstein & Bai (1995).

**Fix:** Add silverstein1995, louartliao2018 (and optionally fanmontanari2019, adlampennington2020, meimontanari2022) to references.bib. Re-point :49, :127, :179 and the "Toward a full Stieltjes derivation" paragraph at Bonnaire Lemma C.1 + Silverstein-Bai + Louart-Liao-Couillet. Keep Pennington-Worah and Benigni-Peche only where the isotropic-X hypothesis is genuinely met.


---

## [MAJOR] For this exact data model an exact (non-replica, non-asymptotic) result is available and would be strictly stronger than the four-bulk sketch -- this is the highest-value redirection

`exact-gaussian-redirection` · verdict **—** · kind opportunity · effort weeks  

**Where:** /Users/ryan/Desktop/latent_space_diffusion_analysis/ICLR_2026/sec-theory.tex (empty TODO stub); sec-appendix-fourbulk.tex:261-272 ("Toward a full Stieltjes derivation ... left to follow-up work")

**Claim:** The paper's synthetic setting is a Gaussian mixture with block-diagonal covariance diffused by an OU process -- a setting where the population score is exactly linear in x_t and the gradient-flow dynamics of a *linear* score model are solvable in closed form, with tau_i = 1/lambda_i(Sigma_t) literally rather than asymptotically. Four existing results supply the machinery: Pierret & Galerne, "Diffusion models for Gaussian distributions: Exact solutions and Wasserstein errors" (arXiv:2405.14250, ICML 2025) -- exact backward-SDE and PF-ODE solutions and Wasserstein errors; Wang & Vastola, "The Unreasonable Effectiveness of Gaussian Score Approximation for Diffusion Models" (arXiv:2412.09726) -- closed-form Gaussian score via eigendecomposition of the covariance, with the empirical result that it dominates the early/mid part of sampling; Merger & Goldt, "Generalization Dynamics of Linear Diffusion Models" (arXiv:2505.24769) -- exact treatment of how the data covariance spectrum (including power-law/hierarchical spectra) drives the train-test gap, early stopping and the N < d vs N > d regimes, which is the paper's mode-absorption story done exactly; and Shah, Chen & Klivans (NeurIPS 2023) for provable GMM score learning under the DDPM objective. A half-page exact theorem for a linear score on diag(sig_sig^2 I_dint, sig_perp^2 I_{dlat-dint}) would establish the mode-ordering and the tau_gen/tau_mem separation without any replica step, and would let the RFNN section be honestly demoted to "the same ordering survives a nonlinear random feature map, empirically".

**Evidence:** sec-appendix-fourbulk.tex:261-272 concedes: "A full Stieltjes-transform / replica derivation that produces the bulk edges as exact roots of a polynomial fixed-point equation ... is left to follow-up work." sec-theory.tex is a TODO stub. Merger & Goldt: "analyse linear diffusion models under a Gaussian data assumption, focusing on how data covariance spectra influence learning ... Strong hierarchical structure, regularization, and early stopping mitigate overfitting."

**Fix:** Write sec-theory.tex as an exact result for a linear (or GEP-linearized) score under Sigma_data = diag(sig_sig^2 I_dint, sig_perp^2 I_{dlat-dint}) plus cluster means: state Sigma_t = e^{-2t}Sigma_data + Delta_t I, give the exact per-eigendirection gradient-flow solution, define tau_gen and tau_mem as crossing times of explicitly named functionals, and derive the d_lat dependence exactly. Cite Pierret-Galerne, Wang-Vastola, Merger-Goldt and Shah-Chen-Klivans for the exactly-solvable-Gaussian background. Add all four to references.bib. This converts the weakest section of the paper into its strongest and fits a workshop page budget.


---

## [MAJOR] The "mode-absorption clock" is the standard spectral-bias learnability curve; presenting it as new over-claims, and the right citations would also give it a theoretical backing it currently lacks

`mode-absorption-clock-is-standard` · verdict **—** · kind improvement · effort hours  

**Where:** /Users/ryan/Desktop/latent_space_diffusion_analysis/ICLR_2026/sec-spectral-predictor.tex (definition of P_d(s)); sec-intro.tex:29 ("We introduce a way to estimate memorization...")

**Claim:** P_d(s) = sum_i w_i (1 - exp[-kappa lambda_i(d) s])^2 / sum_i w_i is exactly the per-eigenmode learnability of gradient flow on a quadratic loss, squared and spectrum-averaged. This object is standard: Ali, Kolter & Tibshirani (2019), "A continuous-time view of early stopping for least squares regression" (AISTATS) shows gradient flow at time T is equivalent to ridge at lambda ~ 1/T with exactly the per-mode factor (1 - e^{-lambda_i T}); Bordelon, Canatar & Pehlevan (ICML 2020) and Canatar, Bordelon & Pehlevan (Nat. Commun. 2021) give spectrum-dependent learning curves built from the same mode-wise factors; Bahri et al. (2024), already in references.bib, is the scaling-law version. sec-intro.tex:29 says "We introduce a way to estimate memorization at a given training step", which reads as introducing the clock rather than the application. Citing the sources converts a bare heuristic (currently limitation (ii): "an empirical frozen-VAE latent-spectrum rule, not yet a theorem") into a principled construction with a known meaning.

**Evidence:** sec-spectral-predictor.tex: "P_d(s) = [sum_i w_i(d)(1 - exp[-kappa lambda_i(d) s])^2] / [sum_i w_i(d)]", with M_t(d) = e^{-2t} Z_d^T Z_d/n + (1-e^{-2t}) I_d. sec-discussion.tex limitation (ii): "The spectral predictor is an empirical frozen-VAE latent-spectrum rule, not yet a theorem for the trainable MLP". No entry for Ali/Kolter/Tibshirani, Bordelon, or Canatar in ICLR_2026/references.bib.

**Fix:** Add alikolter2019, bordelon2020, canatar2021 to references.bib; in sec-spectral-predictor.tex, derive P_d(s) as the standard gradient-flow learnability of the linearized score under latent covariance M_t(d) and cite; reword sec-intro.tex:29 to claim the *application* (a pre-training memorization-risk curve from a frozen encoder) rather than the construction.


---

## [MAJOR] mu_1^2 is the wrong linear coefficient: the Hermite coefficients must be those of tanh at pre-activation variance tau, not tau = 1

`mu1-squared-wrong-coefficient` · verdict **CONFIRMED** · kind error · effort hours  

**Where:** sec-appendix-fourbulk.tex, Eq (kernel-decomp) (lines 50-58) and the definition etabar_star = sum_{k>=3} mu_k^2 E[(||x||^2/d_lat)^k] (lines 57, 140-141); propagated into Lemma 1, the theorem table, and the 'Bulk-gap scaling' paragraph (lines 250-255)

**Claim:** Eq (kernel-decomp) writes mu_1^2 (x^T y/d_lat) with mu_k = E[h_k(z) tanh(z)] for a STANDARD normal z, but the actual pre-activations have variance tau = ||x||^2/d_lat, which in every experiment is 0.33-1.5, not 1. The correct linear coefficient is a(tau) = (E[tanh'(sqrt(tau) g)])^2, which tends to 1 as tau -> 0 (tanh is linear there) and equals mu_1^2 only at tau = 1. The series is also missing the 1/k! Hermite normalization and expands in x^T y/d_lat rather than in the correlation rho = (x^T y/d_lat)/tau.

**Evidence:** Price/Stein: for u,v jointly Gaussian with variance tau and covariance c, dK/dc at c=0 equals (E[tanh'(sqrt(tau) g)])^2, so K(x,y) = a(tau) (x^T y/d_lat) + ... and eta_star(x,x) = E[tanh^2(sqrt(tau) g)] - tau a(tau). Empirical test against the saved spectra at sigma_perp=0.5: the measured linear coefficient (mean of the noise-dim bulk divided by psi_p beta_t^2) is 0.306, 0.419, 0.509, 0.546, 0.574, 0.587, 0.613, 0.627 for d_lat = 10..200, while a(tau) predicts 0.281, 0.393, 0.494, 0.541, 0.568, 0.586, 0.612, 0.625 (agreement to 0.3% at large d_lat) and mu_1^2 = 0.367 is off by up to 70%. The full corrected edge psi_p a(tau) beta_t^2 reproduces the measured noise-dim median to within a few percent across the whole d_lat sweep; mu_1^2 does not.

**Fix:** Define mu_k(tau) = E[h_k(g) tanh(sqrt(tau) g)] and write K(x,y) = sum_k (mu_k(tau)^2/k!) rho^k, so the linear coefficient is a(tau) = mu_1(tau)^2/tau = (E tanh'(sqrt(tau) g))^2 and etabar_star = E_x[ E tanh^2(sqrt(tau_x) g) - tau_x a(tau_x) ]. Since tau depends on d_lat, d_int, sigma_sig, sigma_perp, s and t, a(tau) is itself a slowly varying function of the sweep parameters -- this must be stated, not hidden in a constant.


**Referee:** Verified analytically and numerically, and the finding is if anything understated. The paper defines mu_k = E_{z~N(0,1)}[h_k(z) tanh(z)] (line 36-39) but the actual pre-activations have variance tau = ||x||^2/d_lat, which over the sweep runs 1.515 (d=10) down to 0.327 (d=200) - never 1. By Stein/Price the linear coefficient is a(tau) = (E tanh'(sqrt(tau) g))^2 = 0.281, 0.393, 0.494, 0.541, 0.568, 0.586, 0.611, 0.625, versus the paper's constant mu_1^2 = 0.3669. Measured linear coefficient (noise-dim bulk mean / (psi_p beta_t^2)): 0.3055, 0.4185, 0.5083, 0.5449, 0.5735, 0.5869, 0.6126, 0.6272 - matching a(tau) to under 1% at large d_lat, while mu_1^2 is off by up to 71%. The corrected coefficient also predicts the SIGNAL bulk to ~2% across the sweep (psi_p a alpha^2 = 49.7...110.6 vs measured 49.6...112.8), which mu_1^2 cannot. And it makes the trace exact: with etabar = E[tanh^2(sqrt(tau)g)] - tau a(tau), the identity a*tau + etabar = E[tanh^2] reproduces the measured tr(U)/p to 4 digits. The two structural points are also correct on inspection: with probabilists' Hermite polynomials E[h_k^2] = k!, so the kernel is sum_k c_k^2 rho^k/k! and the paper's etabar_star = sum_{k>=3} mu_k^2 E[(||x||^2/d)^k] omits the 1/k! and expands in tau^k rather than in the correlation. Numerically the paper's own etabar formula (mu_3^2 tau^3 = 0.0019 at d=40) is 6.7x below the true etabar = 0.0125.


**Referee correction:** None. The proposed fix is the right one; note additionally that a(tau) drifts by 2.2x across the paper's own d_lat sweep, so treating it as a constant is not a cosmetic simplification - it is the difference between a formula that tracks both linear bulks to a few percent and one that does not track them at all.


---

## [MAJOR] The fixed-W concentration remark's arithmetic is wrong and the quantity it claims vanishes actually diverges

`fixedW-remark-arithmetic` · verdict **CONFIRMED** · kind error · effort days  

**Where:** sec-appendix-fourbulk.tex, Remark \ref{rem:fixedW} (lines 173-188), specifically 'operator-norm corrections of order nsamp/sqrt(pwidth) = sqrt(psi_n/(psi_p pwidth)) -> 0'

**Claim:** The stated identity is false. With n = psi_n d_lat and p = psi_p d_lat, n/sqrt(p) = psi_n sqrt(d_lat/psi_p), which DIVERGES as d_lat -> infinity, whereas sqrt(psi_n/(psi_p p)) = sqrt(psi_n)/(psi_p sqrt(d_lat)) -> 0. The two sides differ by a factor d_lat sqrt(psi_n psi_p). Since this remark is the only argument offered for upgrading the W-averaged kernel to the fixed-W matrix that is actually computed, the fixed-W version of Lemma 2 is unproven.

**Evidence:** Algebra as above. Numerically at the paper's defaults (d_lat=512, psi_p=64, n=500): n/sqrt(p) = 500/181.0 = 2.76, while sqrt(psi_n/(psi_p p)) = sqrt(0.977/(64*32768)) = 6.8e-4 -- a factor of 4000 apart, and the true quantity is not small. Even the sharp bound is not o(1): for an n x n perturbation with independent-ish entries of relative size O(1/sqrt p) the operator norm is O(sqrt(n/p)) = O(sqrt(psi_n/psi_p)), which is a fixed O(1) constant in the proportional limit (0.088 at psi_n=1, psi_p=64), never o(1).

**Fix:** Delete the false identity. The correct route is Gaussian equivalence (Pennington-Worah / Hu-Lu): at fixed W, psi^perp behaves like sqrt(eta_star) times an iid Gaussian matrix independent of W, so U^diag is a scaled Wishart whose n nonzero eigenvalues form an MP band around etabar/n of relative half-width sqrt(psi_n/psi_p) (= 0.20 at d_lat=200, psi_p=64), not an asymptotically orthonormal frame. The MP-band statement is verifiable: the measured sample-bulk top edge at d_lat=200 is 0.173 vs predicted (etabar/n)(1+sqrt(n/p))^2 = 0.166.


**Referee:** Pure algebra, verified by inspection of sec-appendix-fourbulk.tex:185-186 (mirrored live at ICLR_2026/sec-appendix.tex:763-764). With n = psi_n d_lat and p = psi_p d_lat: n/sqrt(p) = psi_n sqrt(d_lat/psi_p), which DIVERGES as d_lat -> infinity, while sqrt(psi_n/(psi_p p)) = sqrt(psi_n)/(psi_p sqrt(d_lat)) -> 0. The claimed identity between them is false, and the two sides differ by d_lat*sqrt(psi_n psi_p). Since this remark is the only bridge offered from the W-averaged kernel (which is all Eq (kernel-decomp) gives) to the fixed-W matrix that compute_U actually assembles, the fixed-W version of Lemma 2 is unproven. The sharp bound is also not o(1): O(sqrt(n/p)) = O(sqrt(psi_n/psi_p)) is a fixed constant in the proportional limit. Independent support for the finding's proposed Gaussian-equivalence route: at d_lat=200 the measured sample-bulk top eigenvalue is 0.1733 and the MP prediction (p*etabar/n)(1+sqrt(n/p))^2 = 0.1713 - a 1% match, so the sample bulk really is an MP band, not an orthonormal frame.


**Referee correction:** Confirmed. Minor: d_lat=512 is not in the actual sweep (which tops out at d_lat=200), so use the largest real run instead - at d_lat=200, p=12800, n=500 the claimed-vanishing quantity is n/sqrt(p) = 4.42 while the stated equivalent is sqrt(psi_n/(psi_p p)) = 1.7e-3. The point stands unchanged.


---

## [MAJOR] The bulk-width remark uses the wrong random matrix: the noise-dim bulk width is set by d_lat/n (Wishart fluctuation of hat-S), not by 1/sqrt(psi_p)

`mp-width-wrong-mechanism` · verdict **CONFIRMED** · kind error · effort days  

**Where:** sec-appendix-fourbulk.tex, Remark \ref{rem:MP} (lines 190-205); consumed by sec-rfnn-bounds.tex Eq (buffer-bound) via lambda_min^{noise-dim}

**Claim:** The remark claims each bulk is an MP band of relative width Delta_MP = 4/sqrt(psi_p) approximately 0.50 at psi_p=64, i.e. max/min about 1.5. The measured noise-dim bulk spans max/min = 12.8 at d_lat=200 and the spread GROWS with d_lat, because the dominant fluctuation is not W's Wishart but hat-S = (1/n) sum xi xi^T, whose noise-block spectrum is sigma_perp^2 times MP with ratio d_lat/n. The buffer bound then identifies lambda_min^{noise-dim} with the bulk edge, which is wrong by an order of magnitude.

**Evidence:** Corrected prediction lambda_code = a(tau) psi_p [e^{-2t} sigma_perp^2 (1 -/+ sqrt(d_lat/n))^2 + Delta_t]. At d_lat=200, n=500, sigma_perp=0.5, t=0.01: lower edge 64*0.625*(0.98*0.25*(1-0.632)^2 + 0.0198) = 2.12 and upper edge 64*0.625*(0.98*0.25*(1+0.632)^2 + 0.0198) = 26.9. Measured from the saved spectrum: ev[199] = 2.124 and ev[5] = 27.17. Exact agreement. At d_lat=40 the same formula gives 4.61/13.38 vs measured 4.90/13.80. The paper's 4/sqrt(psi_p) = 0.50 would predict max/min about 1.5; observed 12.8. Full measured noise-dim (max, min) across the sweep: (6.44, 4.10) d=10, (9.52, 5.16) d=20, (13.80, 4.90) d=40, (16.04, 4.43) d=60, (18.73, 4.17) d=80, (19.75, 3.57) d=100, (23.28, 3.04) d=150, (27.17, 2.12) d=200 -- the minimum falls monotonically while the paper's edge formula says it is constant.

**Fix:** State the noise-dim bulk as an MP band with edges a(tau) [e^{-2t} sigma_perp^2 (1 -/+ sqrt(d_lat/n))^2 + Delta_t]/d_lat (paper units), and note the regime change at d_lat > n where d_lat - n noise-dim modes collapse onto the diffusion floor Delta_t/d_lat. Then recompute lambda_min^{noise-dim} used in Eq (buffer-bound): it is the lower MP edge, not the mean, and it degrades with d_lat/n.


**Referee:** Reproduced the corrected formula against the saved spectra and it is essentially exact. Predicted noise-dim band psi_p a(tau) [e^{-2t} sigma_perp^2 (1 -/+ sqrt(d_lat/n))^2 + Delta_t] vs measured (upper edge ev[d_int], lower edge ev[d_lat-1]) at sigma_perp=0.5: d=40 predicted 13.365/4.607 vs measured 13.797/4.900; d=100 predicted 19.979/3.549 vs measured 19.751/3.568; d=200 predicted 26.921/2.117 vs measured 27.172/2.124. The measured max/min ratio runs 1.57, 1.85, 2.82, 3.62, 4.49, 5.54, 7.65, 12.79 for d_lat=10..200 - growing with d_lat, exactly as the hat-S Wishart mechanism (ratio d_lat/n) predicts, and nowhere near the d_lat-independent 1.5 that Remark rem:MP's Delta_MP ~ 4/sqrt(psi_p) = 0.50 implies. The consequence for Eq (buffer-bound) is confirmed: it sets lambda_min^{noise-dim} = mu_1^2 beta_t^2/psi_p, i.e. 19.43 in code units at d_lat=200, against a measured 2.12 - a factor 9.2, an order of magnitude, in the direction that inflates the bound.


**Referee correction:** Confirmed, plus one additional slip the finding did not catch: the paper's own stated MP width formula (sqrt(psi_p)+1)^2/(sqrt(psi_p)-1)^2 - 1 evaluates to 0.653 at psi_p = 64, not the 0.50 the text reports from its 4/sqrt(psi_p) approximation. Also worth noting for the rewrite: at small d_lat (d=10, d_lat/n = 0.02) the psi_p mechanism and the d_lat/n mechanism give comparable widths, which is presumably why the remark was never caught - it is only wrong once d_lat/n becomes appreciable, which is precisely the regime the paper pushes into.


---

## [MAJOR] The buffer bound multiplies a timescale by a mode count, which gradient flow does not do

`buffer-bound-count-multiplication` · verdict **CONFIRMED** · kind error · effort days  

**Where:** sec-rfnn-bounds.tex, Eq (buffer-bound) (lines 57-63) and the sentence 'the time to traverse the noise-dim bulk is bounded below by its width times the slowest noise-dim timescale'

**Claim:** Under the per-mode decay used throughout (mode i absorbed as exp(-lambda_i tau)), all modes are absorbed CONCURRENTLY, so the time until the last noise-dim mode is absorbed is 1/lambda_min^{noise-dim}, not (d_lat - d_int)/lambda_min^{noise-dim}. Multiplying by the bulk count inflates the claimed lower bound by a factor d_lat - d_int, which is the unsafe direction for a '>=' statement, and it is precisely the factor that carries the paper's 'linear in the buffer width' headline.

**Evidence:** Eq (rfnn-mode-decay) as cited gives independent exponential absorption per eigenmode; there is no sequential-scheduling mechanism anywhere in the derivation that would make absorption times add. The text's own justification ('its width times the slowest noise-dim timescale') is the only support given. Note that after the trace correction (finding trace-violation-psi-p) the true gap is 1/lambda_min^{nd} - 1/lambda_signal = (d_lat/a)(1/beta_eff^2 - 1/alpha_t^2), which still grows linearly in d_lat -- but so does tau_gen = d_lat/(a alpha_t^2), so the growth is a global clock rescaling, not a buffer.

**Fix:** Either replace the bound by tau_mem - tau_gen >= 1/lambda_min^{noise-dim} - 1/lambda_min^{signal} (no count factor), or supply an actual argument that absorption is sequential rather than concurrent. Then re-examine whether the surviving d_lat dependence is a genuine differential delay or just a rescaling of the training clock; the ratio tau_mem/tau_gen is the invariant quantity and it is d_lat-independent under the corrected edges.


**Referee:** Checked the cited equation. ICLR_2026/sec-rfnn.tex:30-35 gives a_i(T) - a_i^star = (a_i(0) - a_i^star) e^{-lambda_i T}, i.e. each eigenmode decays independently and concurrently, and line 35 reads off tau_i = 1/lambda_i per mode. There is no scheduling, no residual-budget coupling, and no sequential mechanism anywhere in the derivation that would make per-mode absorption times ADD. Under that dynamics the time until the last noise-dim mode is absorbed is 1/lambda_min^{noise-dim}, full stop. Multiplying by the count (d_lat - d_int) in Eq (buffer-bound) inflates a lower bound by a factor of up to 195 at d_lat=200, in the unsafe direction for a '>=' claim, and it is exactly the factor that produces the 'linear in the buffer width' headline. The text's only justification is the assertion 'the time to traverse the noise-dim bulk is bounded below by its width times the slowest noise-dim timescale' (sec-rfnn-bounds.tex:54-56), which restates the claim rather than supporting it.


**Referee correction:** Confirmed, and materially worse in scope than stated: this is not confined to the orphaned sec-rfnn-bounds.tex. The identical Eq (buffer-bound) with the (d_lat - d_int) prefactor and the psi_p/(mu_1^2 beta_t^2) edge appears in the LIVE draft at ICLR_2026/sec-appendix.tex:578-586. One amendment to the finding's parenthetical remedy: '1/lambda_min^{nd} - 1/lambda_signal' again uses the noise-dim bulk as the memorization timescale. Memorization is the sample bulk in the paper's own taxonomy, so the corrected statement is tau_mem - tau_gen >= 1/lambda^{sample} - 1/lambda_min^{signal} = n/etabar_star - d_lat/(a(tau) alpha_t^2), with the noise-dim bulk entering only as the ordering argument for why the sample modes come last.


---

## [MAJOR] The noise-dim-to-sample gap is missing a factor psi_n = n/d_lat, so the paper's resolvability criterion is d_lat-independent when the true gap shrinks like 1/d_lat

`bulk-gap-missing-psi-n` · verdict **CONFIRMED** · kind error · effort hours  

**Where:** sec-appendix-fourbulk.tex, 'Bulk-gap scaling' paragraph (lines 250-259) and sec-rfnn-bounds.tex lines 75-82 ('The noise-dim-to-sample ratio mu_1^2 beta_t^2/etabar_star')

**Claim:** Because the linear bulks are mis-normalized by d_lat/psi_p and the sample bulk by n/psi_p (two different factors), the errors do not cancel in the gap ratio. The correct ratio is (a(tau) beta_t^2/d_lat)/(etabar/n) = psi_n a(tau) beta_t^2/etabar, i.e. the paper's expression times psi_n = n/d_lat. Consequently the four-bulk separation degrades as 1/d_lat at fixed n -- exactly the direction the paper wants to push -- while the paper's criterion says it is constant.

**Evidence:** Measured ratio of noise-dim median to sample median at sigma_perp=0.5: 1409, 485, 284, 235, 203, 185, 150, 130 for d_lat = 10, 20, 40, 60, 80, 100, 150, 200 -- monotonically decreasing, roughly as 1/d_lat modulated by etabar's own decay. The paper's mu_1^2 beta_t^2/etabar predicts a d_lat-independent gap. Separately, the paper's merge criterion (beta_t^2 >~ etabar/mu_1^2) fails at sigma_perp=0.01, d_lat=40: it predicts beta^2 = 0.0199 exceeds etabar/a = 0.0082 by 2.4x, i.e. resolvable, yet the saved spectrum shows no cliff at index d_lat at all (ev[37..41] = 0.637, 0.560, 0.553, 0.488, 0.428 -- a smooth decay).

**Fix:** Restate the gap as psi_n a(tau) beta_t^2/etabar_star and give the resolvability threshold as a condition on d_lat/n, not on tanh coefficients alone. Re-test the merge criterion against the sigma_perp=0.01 spectra, which currently contradict it.


**Referee:** The algebra follows directly from the two normalization corrections I confirmed above: the linear bulks are mis-normalized by d_lat/psi_p and the sample bulk by n/psi_p, so the errors do not cancel in the ratio and the correct noise-dim-to-sample gap is (a beta_t^2/d_lat)/(etabar/n) = psi_n a(tau) beta_t^2/etabar_star, i.e. the paper's expression times psi_n = n/d_lat. Empirically the paper's expression fails badly: mean-based measured gap (noise-dim mean / sample mean) = 143, 171, 189, 188, 188, 184, 176, 171 for d_lat=10..200, essentially flat, while the paper's mu_1^2 beta_t^2/etabar_star gives 2.2, 4.1, 7.8, 10.8, 13.2, 15.1, 18.6, 20.8 - off by 8-65x and monotonically rising against a flat measurement. The corrected psi_n a beta_t^2/etabar gives 83, 111, 131, 132, 127, 120, 103, 89 - right order of magnitude and right shape. The sigma_perp=0.01 merge-criterion failure is confirmed independently: the paper's criterion beta_t^2 = 0.0199 > etabar/a = 0.0094 says 'resolvable by 2.1x', yet the saved spectrum at d_lat=40 shows no cliff at index d_lat at all (ev[39..43] = 0.6374, 0.5599, 0.5531, 0.4882, 0.4283, decaying smoothly).


**Referee correction:** The algebraic fix and the failure of the merge criterion are confirmed; two claims in the evidence need correcting. (a) 'The paper's mu_1^2 beta_t^2/etabar_star predicts a d_lat-independent gap' is FALSE - etabar_star depends on d_lat through tau, so the paper's own expression rises 2.2 -> 20.8 over the sweep. The indictment is direction and magnitude, not constancy. (b) 'Measured ratio ... monotonically decreasing, roughly as 1/d_lat' is an artifact of using the sample-bulk MEDIAN, which is heavily depressed at small d_lat because the sample bulk is not concentrated (at d_lat=10, sample mean 0.0362 vs median 0.0037). On the mean the measured gap is nearly flat (143 -> 171), which the corrected psi_n a beta_t^2/etabar reproduces and the paper's expression does not. (c) The quoted sigma_perp=0.01 indices are shifted by two: the values 0.637, 0.560, 0.553, 0.488, 0.428 are ev[39..43], not ev[37..41]; the conclusion is unaffected.


---

## [MAJOR] 'Var(s*_null) x lambda_null = 1' is Stein's identity at its Cauchy-Schwarz equality point; the same relation holds in the signal block and destroys the crossover derivation

`stein-identity-mislabelled-as-key-identity` · verdict **CONFIRMED** · kind error · effort hours  

**Where:** _next_steps/n_shape_heuristic_derivation.md lines 68-70, 76-91 (G_sig, G_null, d_lat* formula)

**Claim:** For any smooth density, integration by parts gives E[x_i partial_i log p] = -1 in every coordinate. Cauchy-Schwarz then gives Var(s_i) lambda_i >= 1 with equality iff s_i is linear in x_i, i.e. iff that marginal is Gaussian. So the celebrated identity is not a special property of the null block; it is the statement that the null marginal is exactly Gaussian. Crucially, the quantity that actually enters the initialisation gradient of a linear network, dL/dW1 ∝ W2^T E[s* x^T] = -W2^T I, is dimension-blind: per-coordinate it is exactly -1 in the signal block too. Under that (correct) reading G_sig = d_int and G_null = d_null, so the crossover is d_lat* = 2 d_int = 10 with no free parameters. The doc's G = d x Var x lambda is instead the Cauchy-Schwarz upper bound on the squared per-mode gradient, and the entire difference between 10 and the reported 14.1 is the non-Gaussianity factor Var(s*_sig) lambda_sig = 1.78 of the mixture's signal block.

**Evidence:** Numerically, 2e5 test points at t=0.1, d_lat=40, sigma_perp=0.01, correct score: E[s_i x_i] = -1.002, -0.999, -1.002, -1.007, -1.003 in the five signal dims and -0.999, -1.002, -1.000, -0.996, -1.000 in the first five null dims. Var(s_sig)=0.719, lambda_sig=2.478, product 1.782; Var(s_null)=5.517, lambda_null=0.1814, product 1.001.

**Fix:** Drop the claim that the identity is derived and special. If a gradient-competition argument is retained, define G as an actual norm of a gradient (or as the per-mode loss-decrease rate lambda_i a_i^{*2}) and state the resulting crossover; note that the parameter-free Stein version predicts 2 d_int = 10 and the loss-rate version predicts d_null* = d_int lambda_null/lambda_sig = 0.37, so the criterion has essentially no discriminating power against a peak that the data locate broadly between d_lat = 10 and 15.


**Referee:** The mathematics checks out numerically. At d_lat=40, sigma_perp=0.01, t=0.1, 2e5 points with the correct score: diag(E[s x^T]) = -1.001,-1.002,-0.999,-0.997,-1.001 in the signal dims and -0.999,-0.998,-1.002,... in the null dims, with max |off-diagonal| 0.022 - i.e. E[s* x^T] = -I exactly, in every coordinate, as integration by parts requires. Var(s_sig) x lambda_sig = 1.775, Var(s_null) x lambda_null = 0.9996. So the doc's 'Key identity' (n_shape_heuristic_derivation.md:70) is Cauchy-Schwarz equality, holding iff that marginal is Gaussian, which the null block is exactly (all cluster means have zero null component, so p_t factorises as [signal mixture] x N(0, lambda_null I)). The substantive error is in what G measures: for s_theta = W2 W1 x at initialization with error ~= s*, dL/dW1 = -2 W2^T E[s* x^T] = +2 W2^T, whose signal-column and null-column mass are proportional to d_int and d_null respectively with no lambda or Var factor. The doc's G = d x Var x lambda is a second moment (the Cauchy-Schwarz upper bound on the squared per-mode gradient), not the mean gradient that accumulates. Under the mean-gradient reading the crossover is d_lat* = 2 d_int = 10, parameter-free, and the entire gap between 10 and 14.1 is the non-Gaussianity factor 1.78 of the signal block.


**Referee correction:** Soften 'destroys the crossover derivation' and 'the same relation holds in the signal block'. The doc does NOT mistakenly assume the identity holds in the signal block - it explicitly uses Var(s*_sig) x lambda_sig = 0.739 x 2.474 = 1.83 there, correctly non-unity. The valid criticisms are narrower: (a) the identity is presented as derived and special when it is Cauchy-Schwarz equality plus exact Gaussianity of the null marginal, and (b) G is defined as a second moment rather than as any actual gradient norm, so its crossover is one of several defensible readings. Note also that materiality is limited: this is a self-declared non-theorem in _next_steps/, and none of it is compiled into main.tex.


---

## [MAJOR] Comparing G_sig to G_null at initialization is not a defensible crossover criterion for an observable measured 300k steps later

`crossover-criterion-not-defensible` · verdict **CONFIRMED** · kind unsupported-claim · effort hours  

**Where:** _next_steps/n_shape_heuristic_derivation.md lines 85-91 and Limitation 2 (lines 274-275)

**Claim:** The criterion (i) uses a 2-layer linear surrogate for a 4-layer GELU network, (ii) is evaluated at t=0 of training, (iii) predicts a quantity (the late-time residual) that is a property of the fixed point, not of the initial gradient, and (iv) contains one empirically fitted input, so it has one free parameter against a single scalar observation. Three defensible choices of the same 'competition' give d* in {0.4, 10, 14.1}; the doc reports agreement of 14.1 with an observed peak at 15 as a 6.7% success. With a broad observed peak (4.95 at d=10, 5.33 at d=15, 3.42 at d=20) this is not evidence for any of them.

**Evidence:** Doc lines 85 ('These gradient magnitudes are evaluated at initialization ... not the steady-state dynamics'), 91 ('order-of-magnitude estimate'), and 280 ('Var(s*_sig) = 0.739 is empirical'). Alternative criteria computed above: Stein/gradient-norm gives d_lat* = 2 d_int = 10; per-mode loss-decrease rate lambda_i (a_i^*)^2 = 1/lambda_i gives d_null* = d_int lambda_null/lambda_sig = 0.37.

**Fix:** Either derive the crossover from the fixed point of the actual dynamics (see `exact-linear-score-model-route`) or demote it to a dimensional-analysis remark with no numerical claim of agreement.


**Referee:** Substantiated on the text. The doc uses a 2-layer linear surrogate for a 4-layer GELU net (line 78, Limitation 6), evaluates gradients at initialization (line 85: 'evaluated at initialization ... not the steady-state dynamics'), predicts a late-time residual at T=300k from that init-time quantity, and Limitation 2 concedes it is 'an order-of-magnitude estimate'. Yet the Pass 3 evaluation counts it as Criterion 1 success ('Peak at d_lat = 14.1 predicted, d_lat = 15 observed (6.7%, within +/-20%)'). Since three defensible readings of the same 'competition' give ~0.4, 10 and 14.1, and the observed peak is broad (4.95 at d=10, 5.33 at d=15, 3.42 at d=20), none of them is distinguishable from the data. And the whole exercise is moot given that the peak it targets is the rotation-mismatch curve, not a learning phenomenon.


**Referee correction:** Drop the 'one free parameter fitted to a single scalar observation' characterisation - it is wrong. Var(s*_sig) = 0.739 is read from the step-1 score error at d_lat=5, which I independently computed as E||s*_ok||^2/d = 0.7241, and it is measured from a different observable than the peak it predicts. It is an independently measured input, not a fit. The defensible version is: the criterion has no discriminating power (three readings span 0.4-14.1 against a peak located only broadly between 10 and 15), and the doc's own caveats already concede everything except the '6.7% agreement' scoring.


---

## [MAJOR] Every MLP is trained with Adam while the tau_i = 1/lambda_i mechanism is a gradient-flow/SGD statement; Adam's preconditioner approximately removes the lambda-dependence

`adam-invalidates-eigenvalue-ordered-absorption` · verdict **REFUTED** · kind missing-theory · effort days  

**Where:** code/experiment_v2.py:435 and code/v3/run_experiment.py:539 (Adam) vs code/experiment_v2_rfnn.py:237 (SGD); _next_steps/n_shape_heuristic_derivation.md lines 118-124; ICLR_2026/sec-theory.tex (empty stub that is supposed to state tau_i = 1/lambda_i)

**Claim:** The whole paper's mechanism — modes absorbed in decreasing-eigenvalue order at rate lambda_i — is a property of gradient flow on a quadratic. Adam divides each coordinate's step by the RMS of its own gradient, which to leading order makes the per-coordinate step size independent of that coordinate's curvature; low-lambda directions are accelerated relative to SGD, which is exactly the regime the buffer argument depends on. The RFNN (SGD, where the ODE is exact) and the MLP (Adam) are therefore not testing the same dynamical statement, and the mode-absorption clock in sec-spectral-predictor is applied to Adam-trained real-data models. The n-shape doc compounds this by computing tau_null = 1/(eta lambda_null) = 55.2k steps for an Adam run, then using that number to explain a sigma_perp = 0.5 vs 0.01 difference.

**Evidence:** Optimizer lines above. Doc line 120 uses tau_null = 1/(eta lambda_null) with eta = 1e-4; doc line 183 concedes 'the prefactor A = 112 encodes the Adam second-moment dynamics ... not captured by the gradient-flow toy'; doc line 374 lists 'a theory of Adam's implicit bias' as one of the three missing ingredients. sec-theory.tex is an empty \todo stub, so the paper never states the assumption the mechanism needs.

**Fix:** Add SGD MLP runs at matched effective learning rates as the primary synthetic evidence (the appendix already promises an 'optimizer choice' control — promote it), and state in sec-theory.tex that tau_i = 1/lambda_i is a gradient-flow statement, with an explicit remark on what adaptive preconditioning does to the ordering. Any claim that the mode ordering survives Adam needs a measurement (e.g. per-mode projection of the residual over training), not an assertion.


**Referee:** Two of the three load-bearing factual claims are false. (1) 'Every MLP is trained with Adam' - the real-data latent-diffusion MLPs, on which the spectral predictor of Sec. 5 is evaluated, use SGD: every run_*_sgd_*.sh passes `--optimizer sgd --lr 0.001 --momentum 0.80` (run_celeba_diverse_bigmlp_sgd_lr001_m08_10k_gpus0123.sh:66-68 and nine sibling scripts), and the draft states this in two places (sec-real-data.tex:40 'Each run uses SGD with learning rate 10^{-3} and momentum 0.80'; sec-appendix-integrated.tex:94). So 'the mode-absorption clock in sec-spectral-predictor is applied to Adam-trained real-data models' is simply wrong. (2) The RFNN, where tau_i = 1/lambda_i actually lives, is full-batch torch.optim.SGD with momentum=0.0 (experiment_v2_rfnn.py:237, config momentum 0.0) - i.e. plain gradient descent, exactly the setting the theory assumes. (3) The claim that the paper 'never states the assumption the mechanism needs' is also wrong: sec-theory.tex is indeed an empty \todo stub, but sec-rfnn.tex:25-38 states it in full - 'Freezing W makes the score-matching loss quadratic in A, so full-batch gradient flow on A is the linear ODE' followed by Eq. (rfnn-gradflow), the per-mode decay Eq. (rfnn-mode-decay), and 'eigenmode i of U is absorbed by the readout with characteristic timescale tau_i = 1/lambda_i'.


**Referee correction:** What genuinely survives is much narrower and is minor rather than major: the SYNTHETIC MLP (experiment_v2.py:435) uses Adam, and the compiled draft never says so - sec-appendix-integrated.tex's 'Synthetic experiment details' paragraph omits the optimizer entirely (only the un-included sec-appendix.tex:67 and :150 mention Adam). So the recommendation reduces to (a) state the synthetic MLP's optimizer in the methods paragraph, and (b) add one sentence noting that tau_i = 1/lambda_i is a gradient-flow statement exactly realised by the RFNN and the real-data SGD runs, and only approximately transferred to the Adam-trained synthetic MLP. The n-shape doc's tau_null = 1/(eta lambda_null) misapplication to an Adam run is real but is confined to a non-compiled planning document.


---

## [MAJOR] The Pass 4/5 post-mortem falsifies the finite-T absorption formula as a predictor of test score error, but its 'decisive' RFNN check is arithmetically and dimensionally broken and compares quantities at different diffusion times

`pass45-falsification-scope-and-errors` · verdict **CONFIRMED** · kind error · effort hours  

**Where:** _next_steps/n_shape_heuristic_derivation.md lines 306-374, especially line 359 (lambda_eff^RFNN) and lines 209-217 (RFNN vs MLP eps_null table)

**Claim:** Three separate problems. (1) The stated rate is lambda_eff = (eta/(d_lat x n)) lambda_noise-dim but the substitution on the same line is 0.01/(Delta_t x n) x 0.813 — d_lat has silently become Delta_t, so the exponent 493 is not the quantity defined. (2) The RFNN score error is evaluated at t_fixed = 0.01 (sigma_noise_0.01/exp2_rfnn/raw_data/*/config.json) while Var(s*_null) = 5.514 and the MLP eps_null are t = 0.1 quantities; at t = 0.01 the correct Var(s*_null) is 1/(e^{-0.02} 10^{-4} + 0.0198) = 50.25, so the '289 >> 5.514' and the '86x-105x MLP/RFNN ratio' tables compare different diffusion times. (3) Both sides of the comparison are the buggy target field, so 'massive anti-learning' is the metric, not the model. What genuinely survives is narrower but still important: the per-mode exponential-absorption formula, applied at the paper's own T and eta, predicts essentially complete absorption of the noise-dim bulk in both the RFNN and the MLP, so it cannot be the source of any residual error — and the paper never says so.

**Evidence:** Line 359: '$\lambda_{eff}^{RFNN} = \frac{\eta}{d_{lat} n}\lambda = \frac{0.01}{\Delta_t n} 0.813$' with Delta_t = 0.0198 for t=0.01 and d_lat = 40. RFNN config t_fixed = 0.01. RFNN final/init score_error: d=5 0.295/0.745 = 0.40; d=8 282.9/286.0 = 0.99; d=20 356.2/383.6 = 0.93; d=40 252.6/290.9 = 0.87 — i.e. the RFNN also barely moves against the wrong target. ICLR_2026 contains no mention of anti-learning, non-convergence, or any failure of the absorption formula (grep for `anti-learn|non-monoton` in ICLR_2026/*.tex returns only appendix n-shape prose).

**Fix:** Fix the arithmetic and the t-matching, then restate the surviving claim honestly in sec-theory.tex: the exponential-absorption formula predicts the ORDER in which modes are fit, not the magnitude of the late-time residual, and at the paper's training budgets it predicts full absorption of every bulk it names. Any quantitative use of the clock (Sec. 5) must be justified separately.


**Referee:** All three sub-claims verified. (1) Line 359 literally reads lambda_eff^RFNN = eta/(d_lat n) x lambda and then substitutes 0.01/(Delta_t x n) x 0.813; with Delta_t = 0.0198 at t = 0.01 and d_lat = 40 these differ by a factor 1010, and the difference is decisive: the literal formula gives 2 lambda_eff T = 0.244 (exp(-0.24) = 0.78, essentially no convergence), the substituted one gives 493. (2) Confirmed t-mismatch: sigma_noise_0.01/exp2_rfnn/raw_data/di5_d40_n500_s42/config.json has t_fixed = 0.01, while Var(s*_null) = 5.514 and all the MLP eps_null numbers are t = 0.1. I computed the correct Var(s*_null) at t=0.01 as 50.253 (and confirmed the true score attains it exactly, per-null-dim 50.234), so both the '289 >> 5.514' verdict and the '86x-105x MLP/RFNN ratio' table compare quantities at different diffusion times. (3) Both sides of the comparison are the buggy field: RFNN d=40 final score_error 252.6 vs mismatch ||s_ok-s_bug||^2/d = 249.1 and correct target norm 44.7 - the RFNN's true error is of order 3.5 per dim, ~8% of target energy, i.e. it fits the score well. 'Massive anti-learning' is entirely the metric. Also verified that ICLR_2026/*.tex contains no mention of anti-learning or absorption-formula failure.


**Referee correction:** The exponent arithmetic needs one further correction the reviewer did not make. Neither 493 nor 0.244 is right. From the code, lr = 0.01 d_lat/Delta_t (experiment_v2_rfnn.py:236, matching the recorded lr = 20.2 at d=40) and the loss carries 1/(d n), giving a per-mode rate of 0.02 lambda/n = 3.25e-5 and 2 x rate x T ~= 9.8 at T = 300k. So the formula does predict near-complete absorption (exp(-9.8) ~ 6e-5) - the surviving conclusion the reviewer identifies is correct, but it should be stated with exponent ~10, not 493 and not 1e-193.


---

## [MAJOR] The structural argument that the frozen-feature RFNN 'has no mechanism to improve per-null-dim accuracy' confuses expressivity with timescale, and is false as stated

`rfnn-cannot-recover-argument-is-wrong` · verdict **CONFIRMED** · kind error · effort hours  

**Where:** _next_steps/n_shape_heuristic_derivation.md lines 197-207 ('p x (null SNR per feature) = 64 lambda_null/lambda_sig = constant in d_lat')

**Claim:** The RFNN readout A is a full d_lat x p matrix acting on features whose Hermite-1 component is mu_1 W x /sqrt(p). Since W has rank d_lat almost surely and p = 64 d_lat, the span of the features contains a subspace on which the map x -> -Sigma_t^{-1} x is representable to O(mu_1) accuracy, and the null score s*_null = -x_null/lambda_null is exactly linear. So the RFNN's population optimum fits the null score well; the barrier is the time to reach it (the noise-dim bulk eigenvalue is small), not the frozen features. The 'null SNR per feature' cancellation is a heuristic about a single feature's signal-to-noise, not about the least-squares solution over p features, and it is contradicted by the very next section (Pass 5) which computes that the RFNN should be fully converged.

**Evidence:** Doc line 205 asserts the cancellation; doc lines 358-364 then compute exp(-493) for the same system, i.e. complete convergence. Both cannot describe the same model. RFNN readout is `self.A = nn.Parameter(torch.zeros(d_latent, p))` with p = 64 d_lat (experiment_v2_rfnn.py:76), so the readout is not rank-limited relative to the target.

**Fix:** Replace the SNR argument with the correct statement: the RFNN's null-block residual at finite T is governed by exp(-2 eta lambda_noise-dim T) times the null target energy, so the difference between RFNN and MLP is that the MLP's first layer can raise the effective eigenvalue of the null directions (feature learning), not that the RFNN cannot represent the target. Whether that is what actually happens must be measured (per-mode residual projections), not asserted.


**Referee:** Verified in code and data. The readout is `self.A = nn.Parameter(torch.zeros(d_latent, p))` with p = config.p_ratio * d_latent = 64 d_lat (experiment_v2_rfnn.py:76, 234), so it is a full d_lat x p matrix over an overcomplete feature set, not rank-limited relative to a d_lat-dimensional linear target. The null score is exactly linear (s*_null = -x_null/lambda_null, verified: per-null-dim E[s^2] = 5.5142 = 1/lambda_null to five figures, Stein = -1.000 in every null coordinate), so the population optimum over 64 d_lat random features represents it well; the 'null SNR per feature' cancellation at doc line 205 is a statement about one feature in isolation, not about the least-squares solution over p of them. The internal contradiction is real and direct: line 205 says the RFNN has 'no mechanism to improve per-null-dim accuracy', while lines 358-364 compute exp(-493) ~ complete convergence for the same system, and Pass 5's conclusion (line 366) reassigns the residual to 'steady-state signal contamination' in flat contradiction with Pass 3's Limitation 1 ('the dominant source is slow null convergence under Adam dynamics').


**Referee correction:** The reviewer's proposed replacement statement ('the RFNN's null-block residual at finite T is governed by exp(-2 eta lambda T) times the null target energy') is right in form but overtaken by the metric result: the RFNN's reported eps_null of 289 at d=40 is not a residual at all. Reported final 252.6 vs mismatch 249.1 vs correct target norm 44.7 means the RFNN has essentially learned the true score to within ~8% of target energy. So there is no RFNN non-recovery phenomenon to explain, and the correct action is to withdraw the RFNN half of the comparison entirely rather than re-derive it. Note also the RFNN's genuine residual source is empirical-risk overfitting (n = 500, p = 64 d_lat >> n, so A* interpolates the training set), which neither the doc nor the proposed replacement mentions.


---

## [MAJOR] The signal-contamination mechanism has no formalization, is falsified by its own prefactor by 444x, and its 1/d_lat scaling depends on h = 8 d_lat, which the main-text sweep does not use

`signal-contamination-story-lacks-formalization-and-contradicts-protocol` · verdict **CONFIRMED** · kind gap · effort days  

**Where:** _next_steps/n_shape_heuristic_derivation.md lines 95-112 (V_spur), 153-159 (A_spur = 0.25), 366-368; ICLR_2026/sec-mlp.tex:8 (main-text sweep uses fixed hidden width 256)

**Claim:** Three problems. (1) The mechanism is never written as an equation with a computable object: 'V_spur = Var(f(x_t^sig))' is defined but f is never characterised, and the only quantitative version (V_spur ~ d_int/(h lambda_sig)) is off by 444x, which the doc itself reports and then keeps using as the mechanism in Pass 5. (2) Its central scaling prediction A = d_lat V_spur = const requires h proportional to d_lat; the main-text and 5M-step sweeps fix h = 256, under which the same mechanism predicts a d_lat-independent per-dim contamination and hence a growing total, i.e. the opposite of the recovery it is invoked to explain. (3) The residual it is meant to explain is the metric artifact, so there is currently no phenomenon requiring the mechanism. Separately, the reframing 'the residual is steady-state contamination rather than unabsorbed modes' would, if true, be a substantive revision of the four-bulk interpretation and would need to appear in sec-theory.tex — it appears nowhere.

**Evidence:** Doc line 159: 'Signal contamination is not the dominant source of A'; doc line 366 nevertheless: 'The failure mode is definitive ... the residual ... is ... steady-state signal contamination'. Doc line 368: 'its predicted prefactor is A_spur = 0.253, which is 444x too small'. sec-mlp.tex:8 'hidden width fixed at 256'. Config multiseed_runs/.../di5_d40_n500_s42/config.json: `"hidden": 256`.

**Fix:** There is a clean formalization available and it should replace the hand-waving: for any score model that is linear in its features, the population risk decomposes exactly as E||s_theta - s*||^2 = tr((B - Sigma_t^{-1}) Sigma_t (B - Sigma_t^{-1})^T) plus the non-Gaussian signal-block term, and 'contamination' is precisely the signal-null off-diagonal block of B - Sigma_t^{-1}, which for the RFNN least-squares solution is computable in closed form from W and Sigma_t. Compute it, do not estimate it. And re-derive the h-scaling for the protocol actually used.


**Referee:** Sub-claims (1) and (3) hold. (1) V_spur is never given a computable form - f is introduced at line 100 and never characterised; the only quantitative version V_spur ~ d_int/(h lambda_sig) yields A_spur = 0.253, which the doc itself reports as '444x too small' at line 368 and declares 'not the dominant source of A' at line 159, yet line 366 then asserts 'The failure mode is definitive ... the residual ... is ... steady-state signal contamination'. That is a direct self-contradiction inside one document. (3) I confirmed the residual it is meant to explain is the rotation-mismatch artifact (see n-shape verdict), so there is currently no phenomenon requiring the mechanism. And sec-theory.tex is verifiably an empty \todo stub, so the four-bulk reinterpretation the reframing implies appears nowhere.


**Referee correction:** Sub-claim (2) is not a defect in the doc and should be dropped or restated as a transferability limit. The doc is explicitly about the T=300k sigma_perp=0.01 sweep with h = 8 d_lat, and both 300k sweeps do use that rule - I confirmed hidden = 64/320/1600 at d_lat = 8/40/200 in both sigma_noise_0.01/exp2_mlp and sigma_noise_0.5/exp2_mlp configs. h=256 is used only by the separate 5M main-text sweep, which the doc never claims to describe. So the correct statement is that the mechanism, even if it were real, would not transfer to the protocol the paper actually reports - not that the doc contradicts its own protocol. The proposed closed-form replacement (population risk tr((B - Sigma_t^{-1}) Sigma_t (B - Sigma_t^{-1})^T) with contamination as the off-diagonal block) is sound and I verified its ingredients.


---

## [MAJOR] probe_ushape.py measures null-direction error in the wrong basis and its sparsity probes have no random-initialization null, so it cannot test the SAE hypothesis as written

`probe-ushape-null-basis-bug-and-missing-baselines` · verdict **—** · kind error · effort hours  

**Where:** /Users/ryan/Desktop/latent_space_diffusion_analysis/code/probe_ushape.py: `null_dims = list(range(d_int, d_lat))` applied to `pred_s`/`true_s` (rotated frame) in `run_experiment`; `sig_mass` return value

**Claim:** `analytic_score_full` correctly returns the score in the rotated data frame (`score_orig @ Q.T`), and `test_xt` is in that frame, so columns d_int..d_lat-1 of `pred_s`/`true_s` are arbitrary mixtures of signal and null directions, not the null subspace. The null subspace is span(Q[:, d_int:]). Because Var(s*_null) = 5.51 and Var(s*_sig) = 0.73 differ by ~8x, every rotated coordinate carries the d_lat-weighted average, so the reported `null_err_per_dim` and `null_ratio` are systematically wrong and their d_lat dependence is spurious. Separately, `sig_mass` is returned unnormalised; at random initialization its expectation is d_int/d_lat, which falls with d_lat, so the raw statistic cannot distinguish 'the network aligns to signal' from 'nothing happened'. No probe output is committed anywhere in the repo, so none of this has been run at the sweep's 300k budget (the script default is 50k).

**Evidence:** probe_ushape.py: `return score_orig @ Q.T` in analytic_score_full; `null_dims = list(range(d_int, d_lat))`; `pred_s = model.forward(test_xt, t_batch_te)`; `null_err_per_dim = np.mean((pred_s[:, null_dims] - true_s[:, null_dims])**2)`. `sig_mass` returns `sig_sq/total_sq` with no baseline. `find . -iname 'probe_results*'` returns nothing. Argparse default `--n_steps 50000` vs the sweep's 300000.

**Fix:** Project into the ground-truth frame before slicing: `pred_o = pred_s @ Q; true_o = true_s @ Q` then index `[:, d_int:]`. Report sig_mass as an enrichment ratio (sig_mass x d_lat / d_int) with a frozen-W1 control at every d_lat. Use the squared-singular-value participation ratio for effective rank rather than exp(entropy of s_i/sum s_j). Then run at the full budget and commit the output.


---

## [MAJOR] Recommended theory route: the exactly solvable linear score model, which the paper never writes down and which subsumes all three candidate frameworks

`exact-linear-score-model-route` · verdict **CONFIRMED** · kind opportunity · effort weeks  

**Where:** _next_steps/theory_plan.md, kevin's section (SAE / NTK reweighting / two-competing-terms); ICLR_2026/sec-theory.tex (empty)

**Claim:** None of the three candidate frameworks is the right first move. The correct first move is the linear score model, because in this data model the null block is exactly a linear-Gaussian problem: s*_null = -x_null/lambda_null holds exactly (the mixture posterior weights depend on x only through the signal block, since all cluster means have zero null component), and it is independent of the signal block. For s_theta(x) = Bx trained by gradient flow on E||Bx - s*||^2, dB/dt = -2(B Sigma_t + I) has the closed-form solution B(T) = -Sigma_t^{-1} + (B_0 + Sigma_t^{-1}) e^{-2 Sigma_t T}, giving a per-mode risk contribution lambda_i (1/lambda_i)^2 e^{-4 lambda_i T} = e^{-4 lambda_i T}/lambda_i. This single formula gives, with no free parameters: (i) the mode-absorption clock tau_i = 1/(4 lambda_i) rigorously; (ii) the buffer statement, since the d_lat - d_int null modes have the smallest lambda and so are absorbed last among population modes; (iii) an exact prediction of total risk vs d_lat and vs T that can be checked against the (fixed) score-error metric; and (iv) an important correction to the paper's intuition, namely that low-lambda modes carry the LARGEST target energy 1/lambda_i, so the noise-dim bulk dominates the late residual rather than merely delaying things. It also yields the correct crossover criterion (compare per-mode loss-decrease rates, not d x Var x lambda) and the exact form of signal-null contamination (the off-diagonal block of B - Sigma_t^{-1}).

**Evidence:** Verified numerically above: E[s_i x_i] = -1 in every coordinate (Stein), and E[s_null^2] lambda_null = 1.001 exactly while E[s_sig^2] lambda_sig = 1.782, confirming the null block is exactly Gaussian/linear and the signal block is not. The RFNN is already exactly this model in feature space (frozen W, zero-initialised readout A, SGD), so the linear solution applies to it verbatim with Sigma_t replaced by U.

**Fix:** Write sec-theory.tex around this: (1) state the exact linear-score-model lemma with the closed-form B(T) and per-mode risk; (2) specialise Sigma_t to the block-anisotropic case to get the four eigenvalue groups and the ordering, which is the honest version of the buffer claim for a trainable linear net; (3) state the RFNN corollary by substituting U; (4) define the MLP gap precisely as the statement that feature learning changes the effective Sigma seen by the readout, and measure it (per-mode residual projections over training) rather than theorising it. Only if that measurement shows a real signal-subspace concentration should Ba-Erdogdu-style single-index / spiked-covariance feature-learning results (one gradient step on W1 creates a rank-d_int spike in the feature covariance) be brought in as the second stage. Mei-Montanari / double-descent is the wrong tool: the observed curve is not a variance peak at an interpolation threshold, and n = 500, p = 64 d_lat never crosses one in the sweep.


**Referee:** Every technical claim checks out. The null block is exactly linear-Gaussian: all cluster means have zero null component (means_full[:, d_int:] = 0 before rotation, experiment_v2.py:111 and v3/lib/data_synthetic.py:43) and Sigma_t is block diagonal in that frame, so p_t factorises and s*_null = -x_null/lambda_null exactly, independent of the signal block. Verified numerically: per-null-dim E[s^2] = 5.5142 = 1/lambda_null exactly at t=0.1 and 50.234 vs 50.253 at t=0.01, Var(s_null) lambda_null = 0.9996 vs Var(s_sig) lambda_sig = 1.775, and E[s x^T] = -I with max off-diagonal 0.022. The ODE is right: dL/dB = 2(B Sigma_t + I), so B(T) = -Sigma_t^{-1} + (B_0 + Sigma_t^{-1}) e^{-2 Sigma_t T} and, with B_0 = 0, the per-mode excess risk is lambda_i (e^{-2 lambda_i T}/lambda_i)^2 = e^{-4 lambda_i T}/lambda_i. Point (iv) is the most valuable: low-lambda modes carry target energy 1/lambda_i, so the noise-dim bulk dominates the late residual rather than merely delaying - a genuine correction to the paper's 'buffer' intuition that appears nowhere in the draft. The RFNN mapping is exact: frozen W, A initialised to zeros (experiment_v2_rfnn.py:76), full-batch SGD with momentum 0 (line 237), and sec-rfnn.tex already writes the same ODE with U in place of Sigma_t.


**Referee correction:** Temper 'subsumes all three candidate frameworks' and point (ii). The population linear model has only Sigma_t's spectrum and therefore no sample bulk - it can order population modes and give tau_gen, but it cannot produce tau_mem or memorization at all, which is the paper's headline claim. Delivering the buffer statement requires the empirical U (with its n sample-specific modes), where the closed form still applies but the eigenvalue ordering is an empirical/random-matrix question, not a consequence of the lemma. So the route is the right first move for sec-theory.tex, but it is the honest version of the mode-ordering claim, not a complete theory of memorization; step (2) as written should be stated for the empirical second-moment matrix, not for Sigma_t.


---

## [MAJOR] Fix the normalization of the bulk edges; the correct scaling changes the buffer law from linear to quadratic in d_lat and changes what the width controls test

`edge-normalization` · verdict **CONFIRMED** · kind missing-theory · effort hours  

**Where:** sec-appendix-fourbulk.tex Theorem thm:fourbulk (edges written as mu_1^2 alpha_t^2/psi_p); sec-rfnn-bounds.tex Eq (buffer-bound) (denominator mu_1^2 beta_t^2/psi_p, i.e. delay proportional to psi_p); code/v3/lib/eigenvalues.py compute_U vs code/v3/lib/models.py RFNNScore

**Claim:** The theorem writes each edge as mu_1^2 alpha_t^2 / psi_p. Its own Lemma proof says the nonzero eigenvalues of W M_t W^T are those of M_t 'rescaled by psi_p', and with phi carrying the 1/sqrt(p) of Eq (U) the correct edge is mu_1^2 alpha_t^2 / d_lat. Since the buffer bound is (d_lat - d_int) / lambda_min^noise-dim, this is the difference between a delay linear in d_lat (as written, at fixed psi_p = 64) and quadratic in d_lat (correct). It also determines what the three width controls test: with the correct normalization the spectrum of U is p-independent, so all three width controls should collapse -- a prediction the paper is already sitting on the data for.

**Evidence:** code/v3/lib/eigenvalues.py: `phi = torch.tanh(x_t @ W_cpu.T); U += phi.T @ phi / n` -- no 1/p, while code/v3/lib/models.py RFNNScore forward returns `features @ self.A.T / math.sqrt(self.p)`. So the plotted eigenvalues are p times the eigenvalues of the U in Eq (U). Measured signal median at d_lat=20, p=1280 is 56.6; mu_1^2 alpha_t^2 * psi_p = 64.9 (15% off), whereas mu_1^2 alpha_t^2 / psi_p = 0.0159 (3500x off). Dividing by p, the paper-normalized edge is mu_1^2 alpha_t^2 / d_lat, so tau_noise-dim = d_lat/(mu_1^2 beta_t^2) and Eq (buffer-bound) should read (d_lat - d_int) * d_lat/(mu_1^2 beta_t^2), not (d_lat - d_int) * psi_p/(mu_1^2 beta_t^2).

**Fix:** Add an explicit normalization convention paragraph to Section 2 stating which U is plotted (the p-unnormalized one) and which enters the ODE (the 1/p one), and rewrite the theorem edge column and Eq (buffer-bound) with /d_lat. Then state the resulting corollary as a prediction: the timescale law depends on d_lat and the M_t spectrum but not on p, so the tau curves from the p=64 d_lat, p=d_lat+n+300, and fixed-p sweeps must superimpose when plotted against d_lat. Verifying that against the existing Appendix Figures app-rfnn-exp2-widths / -pndr / -pfixed costs an afternoon and converts three 'controls' into a quantitative confirmation.


**Referee:** Verified end to end. code/v3/lib/models.py:61 registers W ~ randn(p,d)/sqrt(d), so `tanh(x_t @ W.T)` is exactly $\tanh(Wx/\sqrt{\dlat})$ in paper notation, and forward (line 67) applies the $1/\sqrt p$: `features @ self.A.T / math.sqrt(self.p)`. code/v3/lib/eigenvalues.py:compute_U builds `phi = torch.tanh(x_t @ W_cpu.T); U += phi.T @ phi / n` with no $1/p$. So plotted $\lambda$ = $p\cdot\lambda$ of the $U$ in Eq (U), while the gradient-flow ODE Eq (rfnn-gradflow) runs on the $1/p$ version. The theorem's own proof is the source of the error: lem:Ulin's proof writes $E[W^\top W/\dlat]=\psi_p I$ (true: $W^\top W = pI$) and then concludes 'the eigenvalues of $M_t$ rescaled by $\psi_p$'. Combined with the $\mu_1^2/(p\,\dlat)$ prefactor of Eq (U-lin), the correct result is $\mu_1^2\lambda(M_t)/\dlat$, not $\mu_1^2\lambda(M_t)/\psi_p$ -- an error of $\dlat/\psi_p = \dlat^2/p$. Numerically at $\dlat=20$: measured signal median 56.58; $\mu_1^2\alpha_t^2\psi_p = 64.9$ (15% high); $\mu_1^2\alpha_t^2/\psi_p = 0.0159$ (3560x low). Eq (buffer-bound) inherits the same $\psi_p$ and should carry $\dlat$. theory_plan.md independently flags the worry ('i suspect i'm off by factors of $p/d_{lat}$ from the lift') and its own edge table omits the factor entirely, so the corpus contains three mutually inconsistent normalizations.


**Referee correction:** The stated consequence -- 'the difference between a delay linear in $\dlat$ and quadratic in $\dlat$' -- should not be published, because it inherits the $(\dlat-\dint)\times$ counting factor that finding `buffer-bound-not-derived` shows is not derivable and that the data contradict (observed delay ratio 5.3x vs 39x predicted by counting). Fix the normalization, but state the corrected buffer law through the cumulative-absorption route, not as a quadratic. The second half of the finding is the valuable part and is sound: with the corrected $/\dlat$ normalization the leading-order spectrum of the paper's $U$ is $p$-independent ($U^{\rm lin}$ edges $\mu_1^2\lambda(M_t)/\dlat$, $U^{\rm diag}$ edges $\bar\eta_\star/n$, both free of $p$), so the three width controls ($p=64\dlat$, $p=\dlat+n+300$, fixed $p=1800$) must collapse when $\tau$ is plotted against $\dlat$ -- a real, free, quantitative prediction sitting on data already in the repo.


---

## [MAJOR] No theory for the fixed-null-energy control, which the paper calls its strictest; under that control the buffer is created by the diffusion floor Delta_t, not by the data

`fixed-null-energy-theory` · verdict **REFUTED** · kind missing-theory · effort days  

**Where:** ICLR_2026/sec-rfnn.tex, "Third, the fixed-$\pwidth$ ablation uses fixed total null energy, rescaling $\sigma_\perp^2\propto 1/(\dlat-\dint)$"; sec-appendix-fourbulk.tex (theorem stated only for fixed sigma_perp)

**Claim:** The four-bulk theorem is stated for fixed sigma_perp, so it says nothing about the control the paper leans on hardest. Working it out is short and gives a sharp, previously unstated conclusion: under sigma_perp^2 = E/(d_lat - d_int), the noise-dim block eigenvalue beta_t^2 = e^{-2t} E/(d_lat - d_int) + Delta_t collapses onto the diffusion floor Delta_t, so at large d_lat the buffer exists *only because the forward process injects isotropic variance Delta_t into every latent coordinate*. The buffer is then a property of the noise schedule, not of the data -- which is a stronger and more interesting claim than the paper currently makes, and it comes with a falsification test.

**Evidence:** With the paper's default t=0.01, Delta_t = 1 - e^{-0.02} = 0.0198. Under fixed null energy E, e^{-2t} E/(d_lat-d_int) -> 0 while Delta_t stays 0.0198, so beta_t^2 -> Delta_t and the noise-dim bulk sits at a1(q)^2 Delta_t psi_p, d_lat-independent, while its multiplicity still grows as d_lat - d_int. Simultaneously q = tr(M_t)/d_lat -> Delta_t = 0.0198, so a1(q)^2 -> ~1 (tanh essentially linear) and a_*(q)^2 -> O(q^3) ~ 1e-5, i.e. the sample bulk collapses toward the rank-null floor and the noise-dim-to-sample gap *widens* under this control rather than narrowing.

**Fix:** Add a corollary to Section 2 or the appendix: 'Corollary (fixed null energy). Under sigma_perp^2 = E/(d_lat-d_int), the four bulks persist with the same counts; the noise-dim edge converges to a1(Delta_t)^2 Delta_t psi_p and the sample edge to a_*(Delta_t)^2 (1+sqrt(p/n))^2, so the buffer survives and is asymptotically set by the noise schedule.' Then state the falsifier: the corollary predicts the fixed-null-energy buffer vanishes as t -> 0, because beta_t^2 -> e^{-2t} E/(d_lat-d_int) -> 0. Rerunning the fixed-p / fixed-null-energy sweep at t = 0.001 (Delta_t = 0.002) should destroy the noise-dim/sample separation, while the fixed-sigma_perp=0.5 sweep at the same t should keep it. That single rerun turns the strictest control into a mechanistic test rather than a robustness check.


**Referee:** The premise is wrong: thm:fourbulk is NOT stated for fixed $\signoise$. It is stated with $\beta_t^2 \equiv e^{-2t}\signoise^2+\Delta_t$ as a free parameter (sec-appendix-fourbulk.tex lines 104, 221-222), so substituting $\signoise^2 = E/(\dlat-\dint)$ is a one-line specialization the theorem already covers. 'The four-bulk theorem says nothing about the control the paper leans on hardest' is therefore false. Moreover the substantive physical observation is already in the corpus twice: theory_plan.md's 'shared concerns' says explicitly 'what about $\sigma_\perp\to 0$? the noise-dim bulk's edge collapses to the diffusion floor $\Delta_t$, and at small $t=0.01$ this is $\approx 0.02$. the bulk doesn't fully disappear, it just sits on the diffusion floor'; and sec-appendix-fourbulk.tex's 'Bulk-gap scaling' paragraph (lines 241-248) already states that both bulks are dragged toward $\Delta_t$. The arithmetic checks out ($\Delta_{0.01}=0.0198$; under fixed $E$, $\beta_t^2\to\Delta_t$ while multiplicity still grows), but it is a corollary the authors have already noticed, not a missing theory.


**Referee correction:** Two things worth keeping, restated at their true weight. (a) Presentational, minor: the paper never *writes down* the fixed-null-energy corollary or the sentence 'under this control the buffer is a property of the noise schedule, not of the data' -- and that sentence is a stronger and more interesting claim than the paper currently makes, so it is worth a two-line remark. (b) A real gap the finding missed: the fixed-$p$ control uses $p=1800$ fixed while $\dlat$ sweeps (sec-appendix-integrated.tex:44-46), so $\psi_p=p/\dlat$ is *not* held fixed, which violates the theorem's stated proportional limit ($\psi_p$ fixed, $\psi_p>1+\psi_n$). That is a genuine mismatch between the theorem's hypotheses and the strictest control -- though it dissolves once the $/\dlat$ normalization of finding `edge-normalization` is adopted, since the edges are then $p$-independent. The $t\to 0.001$ falsifier is a good experiment but is the same experiment already proposed, with better arithmetic, under `untested-predictions` P2.


---

## [MAJOR] Minimal feature-learning theory for the MLP: a spiked-W (one-gradient-step) extension of the same four-bulk computation

`trainable-feature-theory` · verdict **CONFIRMED** · kind missing-theory · effort weeks  

**Where:** ICLR_2026/sec-mlp.tex, "The MLP result is qualitatively sharper than the frozen-feature prediction: at high $\dlat$, feature learning appears to find sparser, signal-aligned directions"; _next_steps/theory_plan.md "kevin: U-shape recovery"; _next_steps/n_shape_heuristic_derivation.md (self-declared non-theorem)

**Claim:** Every headline experiment -- synthetic MLP, CelebA, CIFAR-10 -- uses a trainable score network, while the entire mechanism is a frozen-feature result. The paper currently bridges this with the word 'qualitatively'. The internal plan proposes three open-ended frameworks (SAE, NTK reweighting, two-competing-terms) and the resulting document declares itself not a theorem. There is a fourth, much more tractable option that reuses the machinery already written.

**Evidence:** sec-mlp.tex: "The MLP result is qualitatively sharper than the frozen-feature prediction". _next_steps/n_shape_heuristic_derivation.md header: "Status: EMPIRICAL OBSERVATION + HEURISTIC, NOT A THEOREM (Pass 3)" and "Empirically calibrated: A = 112 ... not computable from first principles". theory_plan.md: "i don't actually know if any of these angles lead to clean math".

**Fix:** Take the one-gradient-step feature-learning route (Ba-Erdogdu-Suzuki-Wu-Yang 2022; Damian et al. 2022; Moniri et al. 2023 -- none currently in references.bib). After one large first-layer step the weights become W = W_0 + Delta with Delta a low-rank spike aligned with the leading directions of the score target, i.e. with the d_int-dimensional signal subspace. Everything in the four-bulk computation goes through with W replaced by the spiked W, because the Gaussian-equivalent map only needs W's singular structure: M_t is effectively replaced by a reweighted M_t' in which the signal block is amplified by (1+eta*spike)^2 and the null block is not. Two immediate predictions: (i) the signal bulk moves up and the noise-dim bulk stays put, so tau_gen falls and the signal-to-noise-dim gap widens with training -- directly checkable against the already-saved pre/post eigenvalue files (eigenvalues_pre.npy vs eigenvalues_post.npy exist in every RFNN run directory, and the equivalent MLP measurement is a one-day job); (ii) the effective buffer shrinks from d_lat - d_int toward N_eff - d_int with N_eff determined by how much null mass the spike suppresses, which is the n-shape story stated as a computation rather than a hypothesis. Write it as a Proposition with the spike strength as a free parameter fitted from the measured post-training W, and be explicit that it is a one-step (not full gradient-flow) feature-learning result.


**Referee:** The gap is real and material to claim-chain links (4) and (5): the entire mechanism is a frozen-$W$ result, while the synthetic MLP, CelebA and CIFAR-10 experiments -- i.e. every headline number -- use trainable score networks. Quotes verified verbatim: sec-mlp.tex:41 'The MLP result is qualitatively sharper than the frozen-feature prediction: at high $\dlat$, feature learning appears to find sparser, signal-aligned directions'; n_shape_heuristic_derivation.md line 3 'Status: EMPIRICAL OBSERVATION + HEURISTIC, NOT A THEOREM (Pass 3)' and line 11 'Empirically calibrated: A = 112 ... not computable from first principles'; theory_plan.md 'i don't actually know if any of these angles lead to clean math'. I also confirmed the bibliography claim: references.bib contains bonnaire2025, pennington2017nonlinear, ELKaroui2010spectrum, benigni2021eigenvalue, mei2018meanfield, but grep finds no Ba/Erdogdu/Suzuki, no Damian, no Moniri, and no Hu & Lu or Goldt. The one-gradient-step spiked-$W$ route is a reasonable and much more tractable proposal than the three in theory_plan.md.


**Referee correction:** The finding's claimed free empirical check is refuted, and the cost estimate should go up accordingly. It says prediction (i) is 'directly checkable against the already-saved pre/post eigenvalue files (eigenvalues_pre.npy vs eigenvalues_post.npy exist in every RFNN run directory)'. Those files are from the frozen-$W$ RFNN, where $U$ does not depend on the trained readout $A$ at all, so pre and post are the same matrix up to Monte-Carlo noise -- I checked: max relative eigenvalue difference 3.6% (d=20), 3.0% (d=100), 2.0% (d=200), with the top eigenvalue moving by 0.02% (102.644 -> 102.666 at d=20). By construction they can never show a spike. And `find` over sigma_noise_0.5 returns eigenvalue arrays only under exp2_rfnn, exp3_rfnn and exp2_exp3_dynamics -- there are no MLP eigenvalue files anywhere. So testing prediction (i) requires new measurement of the trained MLP first layer, not a re-read of saved arrays.


---

## [MAJOR] Propose falsifiable predictions the theory has not yet been fitted to; at least two are cheap and orthogonal to the buffer-count story

`untested-predictions` · verdict **CONFIRMED** · kind opportunity · effort days  

**Where:** ICLR_2026/sec-rfnn.tex, sec-mlp.tex, sec-spectral-predictor.tex (all comparisons are post-hoc); _next_steps/NEXT_STEPS.md section 5 "Lower-priority extras"

**Claim:** Everything in the paper is a post-hoc match of a spectrum to data already collected, and the spectral predictor is explicitly a three-parameter fit. A referee will ask for one confirmed prediction. The corrected theory supplies several that the buffer-count story does not, so they discriminate between the two mechanisms.

**Evidence:** sec-spectral-predictor.tex: "The constants $(\kappa,a,b)$ are fit once per dataset using the dimensions where the VAE already preserves pixel-space identity". sec-rfnn.tex: "We observe these four populations in every configuration we ran". No experiment in the repository was run before the corresponding claim.

**Fix:** P1 (activation sweep, ~1 day, RFNN only). tau_mem is set by lambda_max^sample = a_*(q)^2 (1+sqrt(p/n))^2, where a_*(q)^2 = E[f(sqrt(q)z)^2] - (E[f'(sqrt(q)z)])^2 q is a pure property of the activation. The buffer-count story predicts no activation dependence. Prediction: at fixed d_lat=20, n=500, p=1280, sigma_perp=0.5 (q=0.89), the sample-bulk top edge, and hence tau_mem, scales as 1/a_*^2 across f in {identity (a_*=0, no sample bulk at all), tanh, ReLU, sin, and a deliberately Hermite-2-heavy f(u)=u^2-1 (large a_*)}. Compute the four a_*^2 values in advance, register the predicted ordering and ratios, then run. P2 (diffusion-time sweep, ~1 day). All edges depend on d_lat only through q_t(d_lat) = tr(M_t)/d_lat, so configurations with equal q must have identical rescaled spectra. Every RFNN run is at t=0.01. Prediction: at sigma_perp=0.01, beta_t^2 = e^{-2t} sigma_perp^2 + Delta_t is 99.5% diffusion floor at t=0.01 but only 33% at t=0.001, so lowering t to 0.001 should collapse the noise-dim bulk into the sample bulk and destroy the cliff at index d_lat -- while at sigma_perp=0.5 the same change moves the edge by under 8% and the cliff should survive. This is a sign-flip prediction, not a fit. P3 (n sweep, ~2 days, also fills the predictor's admitted hole). lambda_max^sample = a_*^2 (1+sqrt(p/n))^2 predicts that at fixed p=1280, d_lat=20, doubling n from 500 to 1000 raises tau_mem by (1+sqrt(2.56))^2/(1+sqrt(1.28))^2 = 6.76/4.55 = 1.49x. Combined with the bridge in `mode-to-duplicate-bridge`, which adds a d_NN ~ n^{-1/d_int} channel, this gives a two-term prediction for tau_mem(n) at fixed everything else. Register all three before running and report them as pre-registered.


**Referee:** The framing claim is verifiable and correct: sec-spectral-predictor.tex and sec-appendix-integrated.tex:800-802 confirm '(\kappa,a,b)$ are fit once per dataset using the dimensions where the VAE already preserves pixel-space identity: $d\ge70$ for CelebA and $d\ge140$ for CIFAR-10', and sec-rfnn.tex:46 says 'We observe these four populations in every configuration we ran' -- observation, not prediction. Nothing in the corpus is a registered prediction. Given that the paper's central theorem is quantitatively wrong on two counts (fixed $\mu_1$, and the $/\psi_p$ normalization) and its buffer bound is underived, a pre-registered prediction is the cheapest way to convert a post-hoc spectral match into evidence. P3 (the $n$ sweep) is well posed and also fills the predictor's admitted $n$-hole; its arithmetic checks ($6.76/4.54=1.49$).


**Referee correction:** Two of the three predictions need fixing before registration. P2's arithmetic is wrong: at $\signoise=0.01$, $\beta_t^2=e^{-2t}\signoise^2+\Delta_t$ is 99.5% diffusion floor at $t=0.01$ ($\Delta=0.0198$ vs $9.998\times10^{-5}$) and 95.2% -- not 33% -- at $t=0.001$ ($\Delta=0.001998$ vs $9.98\times10^{-5}$). The move from $t=0.01$ to $t=0.001$ drops $\beta_t^2$ by 9.5x (0.0199 -> 0.0021), which may still collapse the noise-dim bulk onto the sample bulk, but it is not the clean sign flip the finding advertises, and lowering $t$ also lowers $q$ and hence $a_*(q)^2$ in the same direction, so the two edges move together. Recompute the ratio $a_1(q)^2\beta_t^2\psi_p / a_*(q)^2$ at both $t$ before registering. P1 should be registered as a ratio test, not an absolute one: the $a_*^2(1+\sqrt{p/n})^2$ formula it rests on predicts 0.159 at the proposed operating point ($\dlat=20$, $q=0.890$, $a_*^2=0.0235$) against a measured top sample eigenvalue of 0.667 -- 4.2x off -- so an absolute-magnitude registration would fail even if the activation ordering is right.


---

## [MAJOR] Theory must predict the magnitude of the generalization gap, not only its onset: the late gen-gap grows 5x with d_lat, which reads as the opposite of the headline claim

`gen-gap-asymptote` · verdict **CONFIRMED** · kind missing-theory · effort days  

**Where:** ICLR_2026/sec-rfnn.tex paragraph "What the gen-gap measures"; data: sigma_noise_0.5/exp2_rfnn/raw_data/*/metrics.jsonl

**Claim:** The paper's narrative is 'the gen-gap is a direct measure of how much null structure the readout has absorbed', and tau_mem is the first crossing of gen-gap > 0.02. Both onset and asymptote are spectral observables, but the theory only ever discusses onset. The measured asymptote moves the *wrong* way for the story: wider latents end training with a substantially larger gap. A referee will read the final-gap column as evidence against the buffer claim unless the theory explains it.

**Evidence:** Final gen-gap (test - train, per-dimension-normalized: step-1 loss is ~1.0 at every d_lat) at sigma_perp=0.5, 300k steps: 0.0188 (d_lat=5), 0.0079 (10), 0.0411 (20), 0.0342 (40), 0.0351 (60), 0.0473 (80), 0.0489 (100), 0.0687 (150), 0.0931 (200). The gen-gap>0.02 crossing time in the same runs is 20k, 15k, 25k, 10k, 10k, 50k, 45k, 60k, 80k -- so onset is later and non-monotone while the asymptote is 5x larger. sec-rfnn.tex claims the gap 'opens at $\taumem$' and that 'widening the buffer pushes back the moment when null structure begins to be learned', with no statement about how much is eventually learned.

**Fix:** Derive the closed form from the same spectrum: G(T) = sum_{i in sample} pi_i (1 - e^{-lambda_i T})^2 (1 - overlap_i) with pi_i the target mass in mode i. Since the score target puts mass 1/lambda_i in mode i (see `continuous-spectrum-effective-dimension`), G(infinity) = sum over sample modes of pi_i (1-overlap_i) grows with the number of low-eigenvalue modes, i.e. with d_lat -- so the theory *predicts* the observed growth once the target weights are included, and the honest statement becomes 'widening d_lat delays the onset of sample fitting but increases the eventual amount of it'. Add this as a corollary and as an explicit qualification to the abstract's and the Discussion's 'delays memorization' framing (the practical-implication paragraph currently reads as an unqualified free lunch). Also state that with a fixed absolute threshold of 0.02 on a quantity whose asymptote grows with d_lat, the crossing-time estimator is biased toward *earlier* crossings at large d_lat, so the measured delay is a lower bound -- which strengthens the paper's claim, and should be said.


**Referee:** Every number reproduces exactly from sigma_noise_0.5/exp2_rfnn/raw_data/*/metrics.jsonl. Final gen-gap at 300k steps: 0.0188, 0.0079, 0.0411, 0.0342, 0.0351, 0.0473, 0.0489, 0.0687, 0.0931 for d_lat = 5,10,20,40,60,80,100,150,200. First gen-gap>0.02 crossing: 20k, 15k, 25k, 10k, 10k, 50k, 45k, 60k, 80k. So the asymptote grows ~5x with $\dlat$ while the paper's narrative (sec-rfnn.tex:69, repeated at sec-appendix.tex:500-513) says only that the gap 'opens at $\taumem$' and that widening the buffer 'pushes back the moment when null structure begins to be learned' -- silent on how much is eventually learned. Since sec-rfnn.tex:69 explicitly calls the gen-gap 'a direct measure of how much null structure the readout has absorbed', the paper has committed itself to an interpretation under which its own data say wide latents absorb 5x more null structure. A referee reading the final-gap column will read it as counter-evidence. The bias argument is also correct: a fixed absolute 0.02 threshold on a quantity whose asymptote grows with $\dlat$ biases the crossing estimator earlier at large $\dlat$, so the measured delay is a lower bound. The proposed closed form $G(T)=\sum_{i\in\rm sample}\pi_i(1-e^{-\lambda_i T})^2(1-{\rm overlap}_i)$ with $\pi_i\propto1/\lambda_i$ does predict growth with the number of low-eigenvalue modes, and is the same functional as the spectral predictor's $P_d(s)$, so it costs the paper nothing to adopt.


**Referee correction:** Add one more thing the finding leaves on the table, because it is the sharper problem. The crossing times are not merely 'later and non-monotone' -- they are non-monotone in a way that contradicts the buffer story over the first half of the sweep: 20k (d=5) -> 15k (d=10) -> 25k (d=20) -> 10k (d=40) -> 10k (d=60) -> 50k (d=80). Memorization onset is *earliest* at $\dlat=40$ and $60$, i.e. at 8-12x the intrinsic dimension, and the claimed monotone delay only holds for $\dlat\ge60$. Combined with the $\taugen$ proxy computed from the same files (1, 1, 5k, 5k, 5k, 5k, 10k, 10k, 10k), the measured $\taumem-\taugen$ is actually smaller at $\dlat=40$ than at $\dlat=5$. The paper should either restrict the claim to $\dlat\gtrsim 12\,\dint$ or explain the dip; on the current single seed (seed 42 throughout, as theory_plan.md's 'shared concerns' notes) it cannot distinguish the dip from seed noise, which is an independent reason to run the multi-seed rerun that document already calls for.


---

## [MAJOR] The theorem gives one scalar per bulk labelled "leading-order edge" but uses it interchangeably as edge, bulk centre and bulk location; the rank-null entry o(1) is vacuous

`edge-vs-bulk-support-conflation` · verdict **—** · kind inconsistency · effort weeks  

**Where:** /Users/ryan/Desktop/latent_space_diffusion_analysis/sec-appendix-fourbulk.tex, table lines 214-227, Lemma lem:Ulin line 118-121 ("eigenvalues Theta(...)"), Lemma lem:Udiag line 138 ("concentrated around"), Remark rem:MP lines 190-205

**Claim:** Three different objects are conflated: Lemma 1 says eigenvalues are Theta(mu_1^2 alpha_t^2/psi_p) (an order-of-magnitude for the whole block), Lemma 2 says they are 'concentrated around' etabar/psi_p (a centre), the theorem's column header says 'leading-order edge' (a support endpoint), and Remark rem:MP then says each is really 'a MP-deformed band around that edge' of relative width ~0.5-0.65. A band cannot be both centred at and edged at the same number.

**Evidence:** Theorem column header 'Leading-order edge' (line 217) vs Lemma 2 'concentrated around etabar/psi_p' (line 138) vs Remark rem:MP 'a Marchenko-Pastur-deformed band around that edge with relative width Delta_MP ~ 4/sqrt(psi_p)' (lines 193-197). Also Delta_MP is printed as 0.50 at psi_p=64 using the 4/sqrt(psi_p) approximation, whereas the exact expression given one line earlier, (sqrt(psi_p)+1)^2/(sqrt(psi_p)-1)^2 - 1, equals 0.653 — which raises the paper's own resolvability threshold from 1.5 to 1.653. The 'Rank-null / o(1)' row conveys nothing: the neglected remainder R_t in Eq (U-split) is itself only bounded by ||R_t||_op = o_d(1), so the entry is exactly the size of the error term and carries no information; empirically that region is not a bulk at all, spanning two decades (at d_lat=80: 9.4e-3 down to 1.2e-4).

**Fix:** Replace the single-number column with [left edge, right edge] per bulk, computed from the Silverstein-Choi support equation for the linear part and from a Marchenko-Pastur/Bai-Silverstein argument for the diagonal part. For rank-null, either state the exact-zero eigenvalue count of U^lin + U^diag and bound the perturbation by an explicit rate for ||R_t|| (not o_d(1)), or drop the row from the theorem and describe it in a remark.


---

## [MAJOR] "Almost surely" is not justified by the cited tools and is not even well-posed as stated

`almost-surely-unsupported` · verdict **—** · kind unsupported-claim · effort days  

**Where:** /Users/ryan/Desktop/latent_space_diffusion_analysis/sec-appendix-fourbulk.tex:211 (theorem), :114 (Lemma 1), :137 (Lemma 2); restated at sec-rfnn-bounds.tex:35

**Claim:** An a.s. statement about a *limit* requires a single probability space carrying the whole sequence (W_d, X_d, eta_d)_{d>=1} plus a summable-error/Borel-Cantelli argument. No such construction is given (the matrices for different d_lat are simply different-sized), and the cited results are in-probability statements.

**Evidence:** The cited tools: El Karoui (2010, Ann. Statist.) proves operator-norm closeness of kernel matrices to their linearization *in probability*; Pennington-Worah (2017) compute the limiting spectral density by the moment method (expectation/probability level); Benigni-Peche (2021) give ESD convergence, not almost-sure control of individual eigenvalues or of gaps. Remark rem:fixedW (lines 173-188) explicitly upgrades from an E_W statement to a concentration statement 'at rate O(1/sqrt(p))' — a rate that is in probability, and not summable in any stated sense. Nowhere is a joint probability space, coupling, or Borel-Cantelli step written down. Note also that even a.s. ESD convergence would not give what the theorem needs: gaps and orderings are not weak-convergence-continuous functionals.

**Fix:** Replace 'almost surely' by 'with probability 1 - o(1)' (or 'with probability at least 1 - C exp(-c d_lat^gamma)' if you carry sub-Gaussian concentration through), and state the conclusion as: for every eps > 0, with probability -> 1, the sorted spectrum has d_int eigenvalues in I_1(eps), d_lat - d_int in I_2(eps), etc., with the I_k disjoint. If you want a.s., construct the sequence on one space and supply summable tail bounds.


---

## [MAJOR] With d_int fixed the signal "bulk" is a finite-rank perturbation carrying zero ESD mass; it needs a BBP outlier analysis with a separation threshold, which the theorem does not have

`signal-block-is-a-spike-not-a-bulk` · verdict **—** · kind missing-theory · effort weeks  

**Where:** /Users/ryan/Desktop/latent_space_diffusion_analysis/sec-appendix-fourbulk.tex, Lemma lem:Ulin line 118 and theorem row 'Signal / d_int'; setup line 27-29

**Claim:** The theorem is stated in a proportional limit but never says whether d_int/d_lat is fixed or -> 0. In every experiment d_int = 5 is *fixed* while d_lat grows to 200-240, so d_int/d_lat -> 0 and the signal block contributes zero mass to the limiting ESD. It is a rank-d_int spike, and spikes only separate from the bulk above a BBP threshold. No threshold appears.

**Evidence:** Setup line 27-29 fixes only psi_p and psi_n: 'We work in the proportional limit d_lat, n, p -> infinity with psi_p = p/d_lat and psi_n = n/d_lat fixed and psi_p > 1 + psi_n.' d_int is never given a scaling. Configs: sigma_noise_0.5/exp2_rfnn sweeps d_lat = 5..200 with d_int = 5. Moreover alpha_t^2 = e^{-2t}(s^2/d_int + sigma_sig^2) + Delta_t is only order-one because s = 3 and d_int = 5 are both fixed; if d_int grew proportionally with d_lat at fixed s, s^2/d_int -> 0 and the cluster structure — the entire signal spike — disappears. The two readings of the limit give qualitatively different theorems.

**Fix:** Commit to a regime and say so: (A) d_int fixed, s fixed — then state the signal block as d_int BBP outliers of the anisotropic MP law, with the explicit condition alpha_t^2 > beta_t^2 (1 + 1/sqrt(psi_p)) (or whatever the correct threshold is for this model) for them to detach; the theorem then has three bulks plus d_int outliers, not four bulks. (B) d_int/d_lat -> kappa in (0,1) — then it is a genuine two-atom H and Silverstein-Choi applies, but s must scale as sqrt(d_int) for alpha_t^2 to stay order-one, which contradicts the fixed s = 3 in every experiment.


---

## [MAJOR] The hypothesis list omits every condition the result actually needs: k >= d_int, the d_int/d_lat regime, the scaling of s, sigma_perp relative to Delta_t, the range of t, and the validity range of the Hermite series

`missing-assumption-set` · verdict **—** · kind gap · effort days  

**Where:** /Users/ryan/Desktop/latent_space_diffusion_analysis/sec-appendix-fourbulk.tex, setup lines 18-30 and 'For data scale s small compared with sqrt(d_lat)' line 32

**Claim:** The only stated hypotheses are the proportional limit with psi_p, psi_n fixed, psi_p > 1 + psi_n, fixed t in (0, infinity), and an informal 's small compared with sqrt(d_lat)'. At minimum the following are also required and are all violated somewhere in the paper's own experiments.

**Evidence:** (i) k >= d_int: Chat = (1/n) sum_mu m_{c(mu)} m_{c(mu)}^T has rank min(k, d_int) and trace s^2, so the signal-block eigenvalue is e^{-2t}(s^2/min(k,d_int) + sigma_sig^2) + Delta_t, not s^2/d_int. With the default k = 10 and the d_int sweep going to d_int = 20 and 40 (ICLR_2026/sec-appendix-integrated.tex:207), the theorem's signal count d_int and its alpha_t^2 are both wrong there. (ii) The Hermite expansion needs v = E||x_t||^2/d_lat in a compact subset of the convergence region; at d_lat = 5 (v = 2.76) the series for etabar evaluates to 1.13e3. (iii) The stated saturation proxy 's/sqrt(d_lat) small' is the wrong quantity — the relevant one is v = e^{-2t}(s^2 + d_int sigma_sig^2 + (d_lat - d_int) sigma_perp^2)/d_lat + Delta_t, which at d_lat = 5 is dominated by the d_int sigma_sig^2 term that s/sqrt(d_lat) does not see. (iv) sigma_perp vs Delta_t is never constrained: at t = 0.01, Delta_t = 0.0198 while sigma_perp^2 = 1e-4, so beta_t^2 = 0.0199 is 99.5% diffusion noise. (v) t is only 'fixed in (0, infinity)', which admits t -> 0 (Delta_t -> 0, M_t singular on the null block when sigma_perp = 0) and t large (all bulks merge).

**Fix:** Write an explicit Assumptions block: (A1) proportional limit with psi_p, psi_n fixed and psi_p - 1 - psi_n >= c > 0; (A2) d_int fixed (spike regime) or d_int/d_lat -> kappa with s^2/d_int -> sigma_c^2 > 0; (A3) k >= d_int and centers in general position so rank(Chat) = d_int; (A4) v_min <= E||x_t||^2/d_lat <= v_max with v_max inside the Hermite convergence radius; (A5) t in [t_min, t_max] with 0 < t_min <= t_max < infinity; (A6) the two separation conditions of finding ordering-is-a-side-condition. Then re-check every experiment against A1-A6.


---

## [MAJOR] A large share of the reported configurations falls outside the theorem's stated hypotheses, including negative predicted bulk counts

`experiments-outside-hypotheses` · verdict **—** · kind inconsistency · effort hours  

**Where:** hypotheses at /Users/ryan/Desktop/latent_space_diffusion_analysis/sec-appendix-fourbulk.tex:27-29; configs at ICLR_2026/sec-rfnn.tex:13, ICLR_2026/sec-appendix-integrated.tex:45,207,313, ICLR_2026/sec-mlp.tex:8

**Claim:** Four distinct violations, each affecting figures the paper leans on.

**Evidence:** (1) psi_p > 1 + psi_n fails at d_lat = 5 under p = 64 d_lat: p = 320 < d_lat + n = 505, so the predicted rank-null count p - d_lat - n = -185. Confirmed: sigma_noise_0.5/exp2_rfnn/raw_data/di5_d5_n500_s42/eigenvalues_pre.npy has length 320. At d_lat = 8 the count is 4, i.e. zero to leading order. (2) The p = d_lat + n + 300 control (ICLR_2026/sec-rfnn.tex:13) gives psi_p - (1 + psi_n) = 300/d_lat -> 0, so the hypothesis holds only with a margin that vanishes in the very limit the theorem is taken in, and the rank-null 'bulk' is a fixed 300 modes — an o(d) count with no ESD mass. This is the control shown in the main-text figure. (3) psi_n = n/d_lat is not fixed anywhere: with n = 500 held constant it runs from 100 (d_lat = 5) to 2.5 (d_lat = 200), and to 2.08 at the d_lat = 240 MLP sweep (sec-appendix-integrated.tex:313). The sweeps move orthogonally to the limit the theorem describes. (4) The Experiment-2 over-intrinsic cases d_int in {25,30,35,40} at d_lat = 20 (sec-appendix-integrated.tex:207,236) give a noise-dim count d_lat - d_int = -20 and a signal count d_int > d_lat exceeding the rank of M_t. (5) The fixed-p control rescales sigma_perp^2 proportional to 1/(d_lat - d_int), which drives beta_t^2 -> Delta_t: under the theorem the noise-dim edge then becomes exactly the diffusion floor and sigma_perp drops out of the prediction entirely.

**Fix:** Add a table mapping every reported configuration to the assumption set, and mark which figures are inside the theorem's scope. At minimum drop d_lat = 5 (and arguably 8) from any figure captioned as testing the theorem, and state that the p = d_lat + n + 300 and fixed-p controls are outside the proportional regime and are reported as empirical robustness only.


---

## [MAJOR] The sigma_perp -> 0 and sigma_perp -> sigma_sig limits are both mis-stated: the noise-dim bulk is a diffusion-noise bulk, and the isotropic case does not collapse to two bulks

`sigma-perp-to-zero-and-isotropic-limit-both-wrong` · verdict **—** · kind inconsistency · effort days  

**Where:** /Users/ryan/Desktop/latent_space_diffusion_analysis/sec-rfnn-bounds.tex:42-49; claim contradicted at ICLR_2026/sec-rfnn.tex:69 ("The buffer does not exist in Bonnaire's isotropic case")

**Claim:** By the theorem's own formulas, (a) as sigma_perp -> 0 the noise-dim edge does not vanish, it saturates at mu_1^2 Delta_t / d_lat, so the 'buffer' persists and is a property of the diffusion noise floor, not of the data anisotropy; (b) as sigma_perp -> sigma_sig the signal and noise-dim bulks do *not* coalesce, because alpha_t^2 retains the cluster term s^2/d_int.

**Evidence:** (a) At t = 0.01 and sigma_perp = 0.01: beta_t^2 = e^{-2t}(1e-4) + 0.0198 = 0.0199, of which 99.5% is Delta_t. The bulk the paper calls 'noise-dimension' is essentially independent of Sigma_data at this operating point. (b) sec-rfnn-bounds.tex:42-45 asserts "Bonnaire's isotropic two-bulk picture is recovered in the limit sigma_perp -> sigma_sig: the signal and noise-dim bulks coalesce into their single population bulk rho_2 at eigenvalue mu_1^2(e^{-2t} sigma_sig^2 + Delta_t)/psi_p." But at sigma_perp = sigma_sig = 1, s = 3, d_int = 5, t = 0.01 the theorem gives alpha_t^2 = 2.764 and beta_t^2 = 1.000, a ratio of 2.76 — well above the paper's own resolvability threshold of 1.5 (or the exact 1.653). Four bulks would be predicted in the isotropic case too, directly contradicting sec-rfnn.tex:69.

**Fix:** Either state the isotropic reduction correctly (it requires s -> 0, i.e. no clusters, not sigma_perp -> sigma_sig), or re-baseline the novelty claim: the noise-dim block exists whenever d_lat > d_int and cluster centers are present, and anisotropy only moves where it sits. Separately, replace 'noise-dimension bulk' language with something that acknowledges the Delta_t floor, and add the sigma_perp -> 0 limit to the theorem as a stated case (beta_t^2 -> Delta_t) rather than leaving it to a NEXT_STEPS bullet.


---

## [MAJOR] The theorem is fixed-t; under the t-distribution the MLP and real models actually train on, the theorem's own separation criterion fails for ~70% of the time mass and fails outright for the t-averaged operator

`t-dependence-not-treated` · verdict **—** · kind gap · effort research-project  

**Where:** theorem hypothesis 'fixed t in (0, infinity)' at /Users/ryan/Desktop/latent_space_diffusion_analysis/sec-appendix-fourbulk.tex:213; training-time distributions at ICLR_2026/sec-appendix.tex:126 (RFNN fixed t = 0.01, MLP t ~ Unif[0.01, 3])

**Claim:** Nothing in the paper treats the t-averaged operator, yet every non-RFNN result (MLP, CelebA/CIFAR latent diffusion, and the mode-absorption clock) is produced by models trained over a distribution of t. The four-bulk structure is not preserved under that average in the regime the experiments use.

**Evidence:** alpha_t^2/beta_t^2 = (1 + 1.8u)/(1 - (1 - sigma_perp^2)u) with u = e^{-2t}. Requiring the ratio to exceed the paper's own resolvability threshold 1.5 gives t < 0.883, i.e. only 29.2% of Unif[0.01, 3] at sigma_perp = 0.5 (31.2% at sigma_perp = 0.01); against the exact threshold 1.653 it is 25.5% / 27.6%. For the t-averaged linear operator, E_t M_t = E[e^{-2t}](Chat + Shat) + E[Delta_t] I with E[e^{-2t}] = 0.1635 over Unif[0.01,3], giving alphabar^2/betabar^2 = 1.475 at sigma_perp = 0.5 and 1.547 at sigma_perp = 0.01 — both below the exact 1.653 threshold, i.e. the two data bulks merge under averaging. The theory plan itself flags this and guesses the opposite: _next_steps/theory_plan.md, 'shared concerns', 'back-of-envelope says yes (it's preserved at every t) but should check'. Preservation at each t does not imply preservation of the mixture, since the bulk locations move with t and the union of the t-indexed supports overlaps.

**Fix:** Three things need proving: (1) identify the correct operator for a time-conditioned network — E_t[U_t] is not it, because t enters the MLP as an input, so the relevant object lives on the joint (x,t) space and its kernel is E_{t,t'}[k_t,t'(x,x')]; (2) if you nonetheless analyze E_t[U_t], note it is again an anisotropic MP problem with population spectrum the two atoms alphabar_t^2, betabar_t^2 and re-run the Silverstein-Choi separation test — which fails for the paper's t-distribution; (3) if you want the four-bulk structure to survive, restrict the training-time distribution (e.g. importance-weight toward small t) and show that restriction empirically preserves the observed delay. Until then, no theoretical statement in the paper applies to the MLP or real-data sections.


---

## [MAJOR] Remark on fixed-W vs W-averaged kernel contains a false identity; the quantity it claims tends to zero actually diverges

`fixed-W-remark-algebra-error` · verdict **—** · kind error · effort days  

**Where:** /Users/ryan/Desktop/latent_space_diffusion_analysis/sec-appendix-fourbulk.tex, Remark rem:fixedW lines 184-187

**Claim:** The remark is the only place where the crucial upgrade from an E_W kernel identity to a statement about the empirical fixed-W matrix U is attempted, and its one quantitative claim is arithmetically false.

**Evidence:** Text: 'the rank-n structure of U^diag therefore survives at fixed W up to operator-norm corrections of order n/sqrt(p) = sqrt(psi_n/(psi_p p)) -> 0'. With n = psi_n d_lat and p = psi_p d_lat, n/sqrt(p) = psi_n sqrt(d_lat/psi_p), which *diverges* as d_lat -> infinity; sqrt(psi_n/(psi_p p)) = sqrt(psi_n/(psi_p^2 d_lat)) -> 0. At the default d_lat = 20, n = 500, p = 1280 the two sides are 13.98 and 0.0175 — three orders of magnitude apart. The remark also concedes this is 'the self-averaging argument referenced implicitly in the Lemma 2 proof', so Lemma 2's concentration step rests on it.

**Fix:** Redo the bound. The cross/off-diagonal contamination of U^diag at fixed W is an n x n Gram matrix of the residual features whose off-diagonal entries are O(1/sqrt(p)) each; the operator-norm bound from a matrix Bernstein / non-commutative Khintchine argument is O(n/sqrt(p)) only for the crude sum bound and O(sqrt(n/p)) with a proper spectral-norm bound. State which you are using and check it against the required gap etabar/n: you need the contamination to be o(etabar/n), which is a much sharper requirement than 'o(1)'.


---

## [MAJOR] Eq. (U-diag) double-counts eta_star and re-applies 1/p; Lemma 2's proof derives eta_star^2/(p n), contradicting the lemma's own eta_star/psi_p

`lemma2-internal-inconsistency` · verdict **CONFIRMED** · kind inconsistency · effort hours  

**Where:** sec-appendix-fourbulk.tex Eq. (U-diag) lines 77-82, versus Lemma 2 statement line 139 and its proof lines 146-155

**Claim:** phi(x) already carries 1/sqrt(p) (line 65), so the prefactor 1/(p nsamp) in Eq. (U-diag) inserts a second 1/p; and phi^perp already has ||phi^perp||^2 = eta_star (line 86), so the explicit eta_star(x_t^mu,x_t^mu) prefactor double-counts it. The proof then computes the eigenvalues as eta_star^2/(p n) (line 154), which is quadratic in eta_star and disagrees with the lemma's stated eta_star/psi_p, which is linear.

**Evidence:** Line 65: 'phi(x) = (1/sqrt(pwidth)) tanh(Wx/sqrt(dlat))'. Line 86: 'E_W[||phi^perp(x)||^2] = eta_star(x,x)+o_d(1)'. Line 78-80: 'U^diag = (1/(pwidth nsamp)) sum_mu eta_star(x_t^mu,x_t^mu) phi^perp(x_t^mu) phi^perp(x_t^mu)^T'. Lines 152-155: 'The eigenvalues are Lambda_mu . ||phi^perp(x_t^mu)||^2 -> eta_star(x_t^mu,x_t^mu)^2/(pwidth nsamp)' versus line 139 'concentrated around bar-eta_star/psi_p'. The three expressions eta_star/psi_p, eta_star^2/(pn), and the correct eta_star/n are mutually inconsistent (at d_lat=20: 2.65e-4, 4.5e-10, 3.40e-5).

**Fix:** Write U^diag = (1/n) sum_mu phi^perp(x_t^mu) phi^perp(x_t^mu)^T with no extra prefactors, and state its nonzero eigenvalues as eta_star/n.


**Referee:** Read directly off the page, no computation needed. Line 65 defines phi(x) = (1/sqrt(pwidth)) tanh(Wx/sqrt(dlat)), so a 1/p is already inside any phi phi^T; line 86 states E_W[||phi^perp(x)||^2] = eta_star(x,x) + o_d(1), so eta_star is already the squared norm of phi^perp. Eq. (U-diag) at lines 77-82 then writes U^diag = (1/(pwidth nsamp)) sum_mu eta_star(x_t^mu,x_t^mu) phi^perp phi^perp^T, inserting both a second 1/p and a second factor of eta_star. Consistently with its own (wrong) prefactor, the proof at lines 152-155 arrives at eigenvalues eta_star^2/(pwidth nsamp) — quadratic in eta_star — while the Lemma statement at line 139 says eta_star/psi_p, linear in eta_star. The three candidate values at d_lat=20 are eta_star/psi_p = 2.65e-4, eta_star^2/(pn) = 4.51e-10, and the correct eta_star/n = 3.40e-5, i.e. mutually inconsistent by six orders of magnitude. The correct object is U^diag = (1/n) sum_mu phi^perp phi^perp^T with nonzero eigenvalues eta_star/n, since U = (1/n) sum_mu phi phi^T already carries all the normalization.


---

## [MAJOR] Lemma 2's concentration claim is refuted by the paper's own spectra: the 'sample bulk' spans two to three decades

`sample-bulk-not-concentrated` · verdict **CONFIRMED** · kind unsupported-claim · effort days  

**Where:** sec-appendix-fourbulk.tex Lemma 2, lines 137-142 ('its nonzero eigenvalues are concentrated around bar-eta_star/psi_p') and Theorem table row 'Sample'

**Claim:** The lemma predicts n nearly-identical eigenvalues at a single edge. The measured sample-bulk region is a smooth multi-decade continuum with no plateau and no gap at either of its claimed boundaries.

**Evidence:** From eigenvalues_pre.npy (divided by p): at sigma_perp=0.5, d_lat=20 the region [d_lat, d_lat+n) runs 5.21e-4 (index 20) down to 1.85e-6 (index 499) — a 280x spread. At sigma_perp=0.01, d_lat=20 the whole spectrum past index 5 is smooth: lambda[19]=4.561e-4, lambda[20]=4.501e-4, lambda[21]=4.400e-4 (no cliff at d_lat) and lambda[519]=2.403e-7, lambda[520]=2.385e-7, lambda[521]=2.381e-7 (no cliff at d_lat+n). The largest log-drop anywhere in [400,800] at sigma_perp=0.01 is a ratio of 1.018 (d_lat=20) — i.e. no detectable bulk boundary at all.

**Fix:** Replace 'concentrated around' with a density statement (the sample bulk is a Marchenko-Pastur-type band, whose width must be derived, not asserted), and stop claiming a four-bulk separation at sigma_perp=0.01 where only the index-d_int cliff is real.


**Referee:** Verified directly on the saved spectra. At sigma_perp=0.5, d_lat=20 (results_rfnn_exp2v3, p=1280, eigenvalues divided by p): lambda[20] = 5.213e-4 down to lambda[499] = 1.869e-6, a 279x spread across the claimed 'sample bulk', with a smooth power-law-like decay and no plateau anywhere; it continues past the claimed rank-null boundary without a break (lambda[519]=1.629e-6, lambda[520]=1.617e-6, lambda[600]=1.077e-6, lambda[1279]=3.52e-8), so the 'rank-null: o(1)' row is not observed either. At sigma_perp=0.01, d_lat=20 the spread over [d_lat, d_lat+n) is 1692x and there is no cliff at either claimed boundary: lambda[19]=4.561e-4, lambda[20]=4.501e-4, lambda[21]=4.400e-4 (ratio 1.013 at index d_lat) and lambda[519]=2.4035e-7, lambda[520]=2.3851e-7, lambda[521]=2.3809e-7. These reproduce the reviewer's quoted values to 4 significant figures. Lemma 2's 'concentrated around bar-eta_star/psi_p' asserts n nearly-identical eigenvalues at a single edge; the data shows a two-to-three decade continuum. The claim is contradicted by the paper's own figures' underlying data.


**Referee correction:** Minor: the finding writes 'the region [d_lat, d_lat+n)' but then quotes index 499, which is inside that interval, not its endpoint — the endpoint at sigma_perp=0.5, d_lat=20 is lambda[519] = 1.629e-6, giving a 320x spread rather than 280x. Also, the finding's claim that there is 'no detectable bulk boundary at all' at sigma_perp=0.01 is right for the [400,800] window it scanned, but the index-d_int cliff is real and large there (lambda[4]/lambda[5] = 50 at d_lat=20), as the finding itself concedes in its proposed fix.


---

## [MAJOR] The quoted mu_3 ~= -0.099 is inconsistent with the paper's own definition of mu_k, in every convention

`mu3-numerical-value-wrong` · verdict **CONFIRMED** · kind error · effort hours  

**Where:** sec-appendix-fourbulk.tex line 39 ('numerically mu_1 ~ 0.606, mu_3 ~ -0.099') and line 253 ('bar-eta_star is dominated by mu_3^2 ~ 0.0097')

**Claim:** With mu_k = E_{z~N(0,1)}[h_k(z) tanh(z)] as defined on line 36, mu_3 = -0.3636, not -0.099. No standard normalization reconciles -0.099 with the simultaneously-quoted mu_1 = 0.606.

**Evidence:** Gauss-Hermite (150 nodes): probabilists' unnormalized mu_1 = 0.6057, mu_3 = -0.3636; orthonormal (/sqrt(k!)) mu_1 = 0.6057, mu_3 = -0.1484; coefficient (/k!) mu_1 = 0.6057, mu_3 = -0.0606. None gives (0.606, -0.099) jointly. Consequently line 253's 'bar-eta_star dominated by mu_3^2 ~ 0.0097' is wrong: the correct total residual mass at tau=1 is 0.0274 (of which the k=3 term is mu_3^2/3! = 0.0220), 2.8x larger.

**Fix:** Recompute and state mu_k in a single declared convention, and recompute bar-eta_star and the noise-dim-to-sample gap ratio from it.


**Referee:** Line 36 fixes the convention as mu_k = E_{z~N(0,1)}[h_k(z) tanh(z)] with probabilists' (unnormalized) h_k, and line 39 then quotes 'mu_1 ~ 0.606, mu_3 ~ -0.099'. By 200-node Gauss-Hermite the value in the paper's own declared convention is mu_3 = -0.36360, and I checked every standard rescaling for a pair consistent with mu_1 = 0.6057: probabilists' unnormalized (0.60571, -0.36360), orthonormal /sqrt(k!) (0.60571, -0.14844), coefficient /k! (0.60571, -0.06060), physicists' H_k unnormalized (1.21141, 4.35970), /sqrt(2^k k!) (0.85660, 0.62927), /(2^k k!) (0.60571, +0.09083). None yields (0.606, -0.099); the closest magnitude has the wrong sign. The error propagates: line 253 says 'bar-eta_star is dominated by mu_3^2 ~ 0.0097' (consistent internally with -0.0985 but with nothing else), whereas in the paper's own convention mu_3^2 = 0.1322 and the correctly normalized k=3 contribution is mu_3^2/3! = 0.02203, with the full residual at tau=1 equal to 0.02741 — 2.8x the quoted 0.0097. That number then feeds the merge threshold at line 254.


---

## [MAJOR] Adjudication: the appendix's mu_0 = 0 argument is CORRECT and theory_plan's mu_2 / Pennington-Worah claim is WRONG — but neither text states the right criterion

`mu0-and-mu2-adjudication` · verdict **CONFIRMED** · kind inconsistency · effort hours  

**Where:** sec-appendix-fourbulk.tex lines 40-47 and Remark lines 159-171, versus _next_steps/theory_plan.md lines 24-28 and line 65

**Claim:** (i) mu_0 = 0 is correct: at fixed x, (Wx)_a/sqrt(d) is exactly symmetric about zero over W, so E_W[tanh] = 0 whatever the cluster centers do to the distribution of x; theory_plan line 65's worry is unfounded. In fact oddness kills every even coefficient a_k(tau) for every tau, so mu_2 = 0 exactly. (ii) theory_plan lines 24-28 are therefore wrong: there is no Hermite-2 term for tanh, and the sample bulk cannot come from mu_2. (iii) However, the appendix's own framing is also imprecise: Pennington-Worah's sample-bulk parameter is eta - zeta = E[sigma^2] - (E[z sigma])^2 = sum_{k>=2} mu_k^2/k!, the whole nonlinear residual, which for odd sigma reduces to the k>=3 odd terms. So the k>=3 route is the right mechanism, but only once the 1/k! and the tau-dependence are put back (see other findings).

**Evidence:** Numerically mu_0 = 0.000000 and mu_2 = 0.000000 to quadrature precision, for tanh against N(0,1) and against N(0,tau^2) for every tau tested (tau^2 in {0.2,...,4.0}). Pennington-Worah's spectrum depends on data only through zeta/eta with zeta = (E[z sigma(z)])^2 = mu_1^2 = 0.3669 and eta = E[sigma(z)^2] = 0.3943, so eta - zeta = 0.0274 = sum_{k>=3,odd} mu_k^2/k!, matching the appendix's mechanism and not theory_plan's.

**Fix:** Delete theory_plan lines 24-28 as the record of a resolved-and-wrong hypothesis; in the appendix, name the residual as eta - zeta in Pennington-Worah's notation, cite the right quantity, and note explicitly that mu_2 = 0 for tanh so the mechanism is carried by k>=3.


**Referee:** I verified all three legs. (i) The appendix's mu_0 = 0 argument (lines 40-47) is correct: at fixed x, the row of W is N(0, I/d_lat), so (Wx)_a is exactly symmetric about zero over W regardless of what the cluster centers do to the marginal law of x, and E_W[tanh] = 0. The worry recorded in _next_steps/theory_plan.md ('mu_0 != 0 for tanh on non-centered inputs (cluster centers shift the input distribution)') is unfounded. (ii) By the same oddness every even coefficient vanishes at every tau; numerically c_0(tau) = c_2(tau) = 0 to quadrature precision (I get 7.7e-34 and 0.0 at tau=1). So theory_plan.md's central proposal — 'the sample bulk as a separate gapped class only appears once you include the Hermite-2 / quadratic correction to tanh, which is the Pennington-Worah term ... the mu_2 piece behaves like an inner-product kernel and contributes a rank-n block' — is wrong for tanh: mu_2 = 0 identically, and the k>=3 route the appendix takes is the right one. (iii) The appendix's framing is nonetheless imprecise: Pennington-Worah's sample-bulk parameter is eta - zeta with eta = E[sigma(z)^2] = 0.394294 and zeta = (E[z sigma(z)])^2 = mu_1^2 = 0.366879, so eta - zeta = 0.027415 = sum_{k>=2} mu_k^2/k!, the full nonlinear residual, which for odd sigma collapses to the odd k>=3 terms. That is exactly the quantity the appendix should name, and naming it makes the missing 1/k! (finding eta-star-divergent-series) unmissable. This is an accurate adjudication of the theory_plan-vs-appendix conflict flagged in the brief.


**Referee correction:** One caveat on the proposed fix: deleting theory_plan lines 24-28 outright would also delete the only place in the corpus that flags the underlying problem — that a linear-only derivation cannot produce a gapped rank-n class. The appendix's own Remark at lines 159-171 covers that, so the deletion is safe, but the replacement text must keep the point. Also, 'the k>=3 route is the right mechanism' should be qualified by finding indicator-not-a-kernel-statement: for this cluster data the k>=3 mass is not confined to the diagonal, so it produces a rank-~k off-diagonal block (top eigenvalue 0.447 vs diagonal 0.017) in addition to the eta_star I term.


---

## [MAJOR] The saturation criterion s/sqrt(d_lat) <~ 2 is the wrong test; the relevant quantity is tau^2 = ||x_t||^2/d_lat, which is 2.76 at d_lat=5

`saturation-criterion-wrong-quantity` · verdict **CONFIRMED** · kind gap · effort hours  

**Where:** sec-appendix-fourbulk.tex lines 32-35 (referring to app:saturation) and ICML (1)/sec-appendix.tex lines 854-864 (Appendix 'RFNN: eigenvalue saturation and the tanh pitfall')

**Claim:** The saturation appendix's criterion s/sqrt(d_lat) <~ 2 omits both the within-cluster variance d_int sig_sig^2/d_lat and the diffusion floor Delta_t. More importantly, 'not fully saturated' is a much weaker condition than 'the N(0,1) Hermite coefficients apply', which needs tau ~= 1. At the paper's smallest d_lat the criterion is passed while tau is far from 1.

**Evidence:** At d_lat=5 the paper's criterion gives s/sqrt(d_lat) = 3/2.236 = 1.34 < 2 ('safe'), but tau^2 = e^{-0.02}(9 + 5*1 + 0)/5 + 0.0198 = 2.764, tau = 1.663, at which a_1(tau)^2 = 0.178 versus mu_1^2 = 0.367 — a 2.06x error in the leading kernel coefficient. At d_lat=40 the error is 1.99x in the other direction (a_1^2 = 0.729). Neither endpoint of the sweep sits near tau=1.

**Fix:** State the operating condition as a band on tau^2 = e^{-2t}(s^2 + d_int sig_sig^2 + (d_lat-d_int) sig_perp^2)/d_lat + Delta_t, report tau^2 per sweep point, and either normalize inputs to tau=1 or carry a_k(tau).


**Referee:** The referral is real: sec-appendix-fourbulk.tex:32-35 says 'For data scale s small compared with sqrt(dlat) (Appendix app:saturation), the pre-activations Wx_t^mu/sqrt(dlat) have order-one variance per coordinate', i.e. app:saturation is being used as the licence for the whole Hermite step. And 'ICML (1)'/sec-appendix.tex:854-864 states that criterion as ||Wx||/sqrt(dlat) >~ 2 collapses the spectrum and that 's/sqrt(dlat) <~ 2 defines the safe operating regime; all reported sweeps satisfy this constraint'. That expression does omit the within-cluster variance d_int sig_sig^2/d_lat and the diffusion floor Delta_t: the actual per-coordinate pre-activation variance is tau^2 = e^{-2t}(s^2 + d_int sig_sig^2 + (d_lat-d_int) sig_perp^2)/d_lat + Delta_t. At d_lat=5 the criterion reports 3/sqrt(5) = 1.342 while tau^2 = 2.764, tau = 1.663. More importantly the finding's main point holds: 'unsaturated' is far weaker than 'the N(0,1) coefficients apply', which requires tau ~= 1; at d_lat=5, a_1(tau)^2 = 0.180 vs mu_1^2 = 0.367 (2.0x error in the leading kernel coefficient), and at d_lat=40, a_1(tau)^2 = 0.602 (1.6x the other way). Neither endpoint of the sweep is near tau=1; tau=1 is crossed once, between d_lat=15 and 20.


**Referee correction:** The criterion is not wrong for its own stated purpose. Even with the omitted terms restored, tau = 1.663 at d_lat=5 is still below the stated threshold of 2, so the sweep really is outside the saturation regime and the saturation appendix's conclusion survives. What is wrong is the use of that appendix at sec-appendix-fourbulk.tex:32-35 as licence for the fixed-mu_k Hermite expansion. Also the 1.6x figure at d_lat=40 replaces the finding's 1.99x, which came from its erroneous a_1(tau)^2 = 0.729 (see wrong-gaussian-measure-tau).


---

## [MAJOR] The theorem's standing hypothesis psi_p > 1 + psi_n fails at d_lat=5 (and is marginal at d_lat=8), yet those panels are used as confirmations

`proportional-limit-hypothesis-violated` · verdict **CONFIRMED** · kind inconsistency · effort hours  

**Where:** sec-appendix-fourbulk.tex lines 27-29 ('psi_p > 1 + psi_n (so the rank-null tail has room to exist)'), versus the d_lat sweep in sigma_noise_0.01/exp2_rfnn and the claim in ICML (1)/sec-appendix.tex app:bulk-detection that 'four bulks are detected in every panel'

**Claim:** With p = 64 d_lat and n = 500, psi_p = 64 is fixed while psi_n = 500/d_lat. The hypothesis psi_p > 1 + psi_n requires d_lat > 500/63 = 7.94. At d_lat=5 it fails outright (64 vs 101) and p = 320 < d_lat + n = 505, so the claimed rank-null bulk cannot exist; at d_lat=5 = d_int the noise-dim bulk also has count 0. At d_lat=8 it holds by 0.5 (p=512 vs d_lat+n=508).

**Evidence:** eigenvalues_pre.npy for di5_d5_n500_s42 has length 320 (= p), while the theorem's four counts sum to d_int + (d_lat-d_int) + n + (p-d_lat-n) = 5 + 0 + 500 + (-185). Also the sweep's smallest three d_lat values are precisely where tau^2 (2.76, 1.74, 1.39) is furthest from 1 and where the proportional limit d_lat -> infinity is least defensible.

**Fix:** State the admissible d_lat range explicitly (d_lat > n/(psi_p - 1) at fixed p/d_lat), and either drop the d_lat = 5 and 8 panels from any claim the theorem is supposed to explain, or label them as outside the theory's regime.


**Referee:** The hypothesis is stated at sec-appendix-fourbulk.tex:27-29 ('psi_p > 1 + psi_n (so the rank-null tail has room to exist)') and repeated in sec-rfnn-bounds.tex:33 and 'ICML (1)'/sec-appendix.tex:604-605. With p = 64 d_lat and n = 500 it requires 64 > 1 + 500/d_lat, i.e. d_lat > 500/63 = 7.937 — it fails outright at d_lat=5 and holds by a margin of 4 features at d_lat=8. I confirmed the file: results_rfnn_exp2_sn001/di5_d5_n500_s42/eigenvalues_pre.npy has length 320 = p, while d_lat + n = 505, so the theorem's four counts sum to 5 + 0 + 500 + (-185); the rank-null bulk cannot exist and the noise-dim bulk has count 0. And those panels are used as confirmations: 'ICML (1)'/sec-appendix.tex:880-885 says 'four bulks are detected in every panel' and explicitly discusses the d_lat = d_int panel, while Table tab:bulk-sizes reports B3 = 27 at d_lat=5 where the theorem predicts d_lat - d_int = 0. So a panel outside the theorem's hypothesis is being read as evidence for a bulk the theorem says has zero modes there. The finding's supporting observation is also right: d_lat = 5, 8, 10 are exactly where tau^2 = 2.76, 1.74, 1.39 is furthest from 1.


---

## [MAJOR] The 'Bulk-gap scaling' merge criterion compares two edges carrying different normalizations, and is 30x off

`gap-criterion-dimensionally-inconsistent` · verdict **CONFIRMED** · kind error · effort days  

**Where:** sec-appendix-fourbulk.tex 'Bulk-gap scaling' paragraph, lines 249-259 ('for beta_t^2 >~ bar-eta_star/mu_1^2 the noise-dim edge sits above the sample edge')

**Claim:** The criterion is obtained by cancelling a common 1/psi_p between the noise-dim edge and the sample edge. Once the normalizations are corrected (noise-dim edge a_1^2 beta_t^2/d_lat, sample edge eta_star/n), the ratio carries an extra factor n/d_lat, so the separation threshold is beta_t^2 >~ (d_lat/n) eta_star/a_1(tau)^2 — a factor n/d_lat = 25 smaller at the default setting.

**Evidence:** Paper's threshold at d_lat=20: bar-eta_star/mu_1^2 = 0.01699/0.3669 = 0.0463, versus beta_t^2 = 0.0199, so the paper's own criterion predicts the noise-dim and sample bulks MERGE at sigma_perp=0.01. The corrected threshold is (20/500)*0.01699/0.446 = 1.52e-3 << beta_t^2 = 0.0199, predicting a 13x separation. Empirically neither is right: the sigma_perp=0.01 spectrum is smooth across index d_lat (lambda[19]/lambda[20] = 1.013), so the observed merging is not explained by either version of the criterion.

**Fix:** Recompute the gap ratio as (n a_1(tau)^2 beta_t^2)/(d_lat eta_star) and re-derive the merge threshold; then explain the observed sigma_perp=0.01 merging by the mechanism that actually causes it (the neglected off-diagonal cluster block plus the finite-p kernel fluctuation, both of which exceed eta_star here).


**Referee:** The internal inconsistency is real. The 'Bulk-gap scaling' paragraph (lines 249-259) forms the noise-dim-to-sample ratio as mu_1^2 beta_t^2/bar-eta_star, which only works because both edges in the theorem table carry the same 1/psi_p. Once the normalizations are corrected per edge-normalization-psi-p and lemma2-internal-inconsistency (noise-dim edge a_1(tau)^2 beta_t^2/d_lat, sample edge eta_star/n), the ratio picks up n/d_lat = 25 at the default setting and the merge threshold becomes beta_t^2 >~ (d_lat/n) eta_star/a_1(tau)^2 = 1.52e-3 rather than eta_star/mu_1^2 = 0.0463. So the criterion as printed is not consistent with a corrected table, and the factor is 25-30x as claimed.


**Referee correction:** The finding's framing significantly understates the paper's position and overstates its own fix. Checked against the data, the paper's criterion makes the correct qualitative call at both noise levels: at sigma_perp=0.01, threshold 0.0463 > beta_t^2 = 0.0199 predicts merge, and the spectrum merges (lambda[19]/lambda[20] = 1.013 at d_lat=20); at sigma_perp=0.5, beta_t^2 = 0.2649 > 0.0463 predicts separation, and the spectrum separates (lambda[19]/lambda[20] = 7.73, rising to 18.5 at d_lat=80). The finding's corrected threshold of 1.52e-3 predicts separation at both, i.e. it gets the sigma_perp=0.01 case wrong. So the finding is right that the criterion is dimensionally inconsistent with corrected edges, but its proposed replacement is empirically worse, and the finding's own last sentence ('neither is right') should be strengthened to: the correct normalization destroys the criterion's apparent predictive success, which is evidence that the observed merge is set by a mechanism outside this calculation — most plausibly the neglected off-diagonal cluster block, whose top eigenvalue 0.447/n = 8.9e-4 is the size of the sigma_perp=0.01 noise-dim bulk (5.3e-4).


---

## [MAJOR] compute_U omits the 1/p in phi, so every reported eigenvalue is p times the theory's U — any claimed edge match is off by that factor

`code-normalization-mismatch` · verdict **CONFIRMED** · kind inconsistency · effort hours  

**Where:** code/v3/lib/eigenvalues.py compute_U (line ~40: 'phi = torch.tanh(x_t @ W_cpu.T)' then 'U += phi.T @ phi / n'), versus ICLR_2026/sec-rfnn.tex Eq. (eq:U) which defines phi(x_t) = (1/sqrt(p)) tanh(W x_t/sqrt(d_lat))

**Claim:** The code's U equals p times the paper's U. The appendix (lines 262-265) claims the theorem's leading-order edges 'match (within Marchenko-Pastur fluctuations) the empirical bulk locations in Table tab:bulk-sizes'; that comparison cannot be valid without dividing the reported eigenvalues by p, and no such rescaling is documented.

**Evidence:** Sum of saved eigenvalues at d_lat=20 is 405.84 with p=1280, i.e. 0.3170 after dividing by p — matching E[tanh^2(tau u)] = 0.332 at tau^2 = 0.706, as tr(U_paper) must. Across the sweep: 174.5/320 = 0.545, 405.8/1280 = 0.317, 544.5/2560 = 0.213, all matching E[tanh^2] at the corresponding tau. The unnormalized numbers exceed 1 by up to 3 orders of magnitude and cannot be compared to any of the theorem's edges directly.

**Fix:** Either add the 1/sqrt(p) to compute_U or state the p-rescaling explicitly wherever theory edges are compared to measured eigenvalues; then redo the edge comparison in Table 3.


**Referee:** Verified in the source and by the trace identity. code/v3/lib/eigenvalues.py compute_U does 'phi = torch.tanh(x_t @ W_cpu.T)' then 'U += phi.T @ phi / n', with no 1/sqrt(p), while ICLR_2026/sec-rfnn.tex Eq. (eq:U) defines phi(x_t) = (1/sqrt(p)) tanh(W x_t/sqrt(d_lat)). So U_code = p * U_paper. The trace check pins it: sum(eigenvalues_pre.npy) = 174.51, 238.38, 280.27, 354.07, 405.84, 486.47, 544.48 for d_lat = 5, 8, 10, 15, 20, 30, 40 with p = 64 d_lat, giving sums/p = 0.5453, 0.4656, 0.4379, 0.3688, 0.3171, 0.2534, 0.2127, against E[tanh^2(tau u)] = 0.5758, 0.4946, 0.4547, 0.3820, 0.3320, 0.2662, 0.2241 at the corresponding tau — agreement to ~5% across the whole sweep, exactly as tr(U_paper) = E[tanh^2] requires. The unnormalized eigenvalues are up to 3 orders of magnitude larger than any theorem edge, so sec-appendix-fourbulk.tex:262-265's claim that the theorem's edges 'match (within Marchenko-Pastur fluctuations) the empirical bulk locations' cannot be checked without an undocumented p-rescaling.


**Referee correction:** Two clarifications. (i) compute_U does not omit the 1/sqrt(d_lat): that factor is folded into the stored W at code/v3/lib/models.py:61 ('torch.randn(p, d_latent) / math.sqrt(d_latent)'). The only missing factor is the 1/sqrt(p) inside phi, hence exactly one factor of p in U. (ii) The claimed edge comparison is worse than 'off by p' — it never happens. Table tab:bulk-sizes ('ICML (1)'/sec-appendix.tex:171-190) reports only bulk *counts*, no eigenvalue locations, so there is no published number for the theorem's edges to match. And those counts do not match the theorem either: at d_lat=20 the table gives B4=5, B3=48, B2=311, B1=916 against the predicted 5, 15, 500, 760, so sec-rfnn-bounds.tex:46-49 ('Predicted bulk counts match the empirical B1...B4 sizes in Table tab:bulk-sizes across the full d_lat sweep at the predicted indices') is false as written.


---

## [MAJOR] Even granting the serial-queue picture, the inequality points the wrong way: lambda_min gives an UPPER bound on traversal time, not a lower bound

`inequality-direction-reversed` · verdict **CONFIRMED** · kind error · effort hours  

**Where:** ICLR_2026/sec-appendix.tex Eq. (buffer-bound) line 578-587; sec-rfnn-bounds.tex lines 54-63

**Claim:** Under the (unjustified) serial reading, traversal time = sum over the noise-dim bulk of 1/lambda_i. Since lambda_i >= lambda_min^{noise-dim} for every mode in that bulk, 1/lambda_i <= 1/lambda_min, so sum_i 1/lambda_i <= (d_lat - d_int)/lambda_min^{noise-dim}. The RHS of Eq. (buffer-bound) is therefore an UPPER bound on the quantity it is asserted to lower-bound. The valid serial lower bound uses the TOP edge of the noise-dim bulk: (d_lat - d_int)/lambda_max^{noise-dim}. So ">=" is the useful direction for the scientific claim but the wrong direction for the algebra as performed.

**Evidence:** sec-rfnn-bounds.tex line 55-59: "bounded below by its width times the slowest noise-dim timescale ... >= (d_lat - d_int) * 1/lambda_min^{noise-dim}". "Slowest timescale" = largest 1/lambda = the maximum term in the sum; a count times the maximum term bounds a sum from above. Direction-corrected numerics ((d_lat-dint)/lambda_max^{noise-dim}, i.e. lambda at index dint) still fail at sigma_perp=0.01: d=10 RHS 3200 vs LHS 887; d=20 22,063 vs 2141; d=40 87,514 vs 4521; d=100 479,082 vs 26,663. At sigma_perp=0.5 the direction-corrected version happens to hold (d=100: 29,729 vs 32,770) but only by 10%, and the exact serial sum sum_i 1/lambda_i = 73,926 still exceeds the measured tau_mem = 32,866. Separately, the two sides are not homogeneous in the tolerance: any threshold-based tau (as in _next_steps/theory_plan.md, "T_sig(eps) = -log eps/lambda_min^signal") carries a log(1/eps) factor on the LHS that is simply absent from the RHS.

**Fix:** If the serial reading is retained anywhere, replace lambda_min by lambda_max in the noise-dim bulk and carry the log(1/eps) tolerance factor on both sides. Then note that the resulting bound is still empirically false at sigma_perp=0.01, which is why the serial reading has to go entirely (see buffer-count-is-a-non-sequitur).


**Referee:** Granting the serial reading for the sake of argument, traversal time = sum_{i in nd} 1/lambda_i, and since 1/lambda_i <= 1/lambda_min^nd for every i in the bulk, (d_lat-d_int)/lambda_min^nd is an upper bound on that sum, not a lower bound. The paper's own words make the error explicit (sec-rfnn-bounds.tex lines 54-59: 'bounded below by its width times the slowest noise-dim timescale ... >= (d_lat-d_int)/lambda_min^{noise-dim}'): width times the slowest timescale is count times the largest summand. I reproduced the direction-corrected numerics ((d_lat-d_int)/lambda_max^nd, i.e. lambda at index d_int): sigma_perp=0.01 d=10 RHS 3254 vs LHS 890; d=20 22,122 vs 2112; d=40 88,081 vs 4889; d=100 473,963 vs 25,990 — still false everywhere. At sigma_perp=0.5, d=100, the corrected version holds only by ~8% (30,802 vs 33,556) while the exact serial sum 73,746 still exceeds the measured tau_mem 33,651. The tolerance-homogeneity point is also right: any threshold-based tau carries log(1/eps) on the LHS (cf. _next_steps/theory_plan.md 'T_sig(eps) = -log eps/lambda_min^signal') and it is absent on the RHS.


**Referee correction:** This finding is a sub-case of buffer-count-is-a-non-sequitur, not an independent defect: fixing the direction does not rescue the bound (it still fails at sigma_perp=0.01 and holds only marginally at one point at sigma_perp=0.5), and the serial premise is itself unjustified. Report it as 'even on its own terms the algebra is backwards' rather than as a separately fixable issue.


---

## [MAJOR] The theorem as stated implies tau_gen exactly d_lat-independent and tau_mem d_lat-dependent only through eta_bar_star -- the opposite attribution from the buffer story

`theorem-implies-opposite-attribution` · verdict **CONFIRMED** · kind inconsistency · effort days  

**Where:** ICLR_2026/sec-appendix.tex Theorem edge table lines 786-805 vs Eq. (buffer-bound) lines 578-591

**Claim:** Read literally at fixed psi_p, the theorem's four edges are: signal mu_1^2 alpha_t^2/psi_p (contains no d_lat), noise-dim mu_1^2 beta_t^2/psi_p (contains no d_lat), sample eta_bar_star/psi_p, rank-null o(1). Only the counts depend on d_lat. Since tau_gen = 1/lambda_signal and tau_mem = 1/lambda_sample are functions of eigenvalues alone (per Eq. rfnn-mode-decay), the ONLY channel by which d_lat can enter either timescale is eta_bar_star = sum_{k>=3} mu_k^2 E[(||x||^2/d_lat)^k], whose argument E[||x||^2]/d_lat = (dint alpha_t^2 + (d_lat - dint) beta_t^2)/d_lat falls with d_lat whenever beta_t^2 < alpha_t^2. So the theorem does supply a d_lat dependence -- but it is a cubic-or-higher suppression of the higher-Hermite diagonal mass driven by falling per-coordinate norm, entirely unrelated to the WIDTH of the noise-dim bulk. The buffer bound attributes the effect to the wrong quantity.

**Evidence:** Quantitative check of the eta_bar_star channel against the measured spectrum (sigma_perp=0.5): E[||x||^2]/d_lat = 1.5145 at d=10 vs 0.3898 at d=100, so eta_bar_star (dominated by the k=3 term) falls by (1.5145/0.3898)^3 = 58.6x, predicting tau_mem to grow 58.6x; the measured 1/lambda_max^sample grows 762.7 -> 32,866, i.e. 43x. Over the same range the corrected signal edge predicts tau_gen to grow 10x and it measures 3.8x. The ratio tau_mem/tau_gen therefore grows roughly linearly in d_lat, which is why the linear-in-(d_lat - d_int) form fits by coincidence over a narrow range.

**Fix:** State the mechanism correctly: the delay comes from dilution of the average per-coordinate feature norm, which suppresses the higher-Hermite (diagonal, sample-bulk) mass super-linearly while the linear (data-bulk) mass falls only as 1/d_lat. Give the prediction tau_mem/tau_gen ~ mu_1^2 alpha_t^2 n /(d_lat eta_bar_star(d_lat)) and test it; drop the mode-count language. Note that this mechanism predicts the effect vanishes or reverses when sigma_perp^2 approaches the per-coordinate signal variance, which is a sharp falsifiable prediction the current story does not make.


**Referee:** The literal reading is correct and I verified the table (ICLR_2026/sec-appendix.tex lines 786-805): the four edges are mu_1^2 alpha_t^2/psi_p, mu_1^2 beta_t^2/psi_p, eta_bar_star/psi_p, o(1). alpha_t^2 = e^{-2t}(s^2/d_int + sigsig^2) + Delta_t and beta_t^2 = e^{-2t} signoise^2 + Delta_t contain no d_lat, so at fixed psi_p the only d_lat channel in the theorem is eta_bar_star = sum_{k>=3} mu_k^2 E[(||x||^2/d_lat)^k], whose argument I measured falling from 1.517 (d=10) to 0.378 (d=100) at sigma_perp=0.5. Since Eq. (rfnn-mode-decay) makes tau_gen and tau_mem functions of eigenvalues alone, the theorem as printed attributes the entire delay to per-coordinate norm dilution of the higher-Hermite diagonal mass, not to the width of the noise-dim bulk — the opposite attribution from Eq. (buffer-bound) three pages earlier. That is a genuine internal inconsistency between two statements in the same appendix.


**Referee correction:** The quantitative half of the finding overstates the case and should not be repeated as written. Approximating eta_bar_star by its k=3 term gives a 59-65x fall over d_lat = 10 -> 100, but the exact Gaussian-equivalent diagonal mass E[tanh^2(sqrt(q)z)] - q E[tanh'(sqrt(q)z)]^2 falls only from 0.0444 to 0.00645, i.e. 6.9x, because at q ~ 1.5 the k>=5 terms carry as much mass as k=3. The measured 43x growth in 1/lambda_max^sample is therefore NOT explained by the eta_bar_star channel alone (the mean sample eigenvalue, sum over the sample bulk / n, tracks eta_bar_star/n to ~1.5x: e.g. d=100, sigma_perp=0.5, measured 8.5e-6 vs predicted 1.29e-5); the extra growth comes from the spread between the top edge of the sample bulk and its bulk mean. So state the mechanism qualitatively (norm dilution suppresses the diagonal/sample mass faster than the linear/data mass falls as 1/d_lat) and drop the cubic-scaling arithmetic and the tau_mem/tau_gen ~ mu_1^2 alpha_t^2 n/(d_lat eta_bar_star) prediction until it is checked — measured tau_mem/tau_gen grows only ~9.7x over a 10x range in d_lat at sigma_perp=0.5.


---

## [MAJOR] The gap tau_mem - tau_gen grows 10x with d_lat in the isotropic case, where the paper explicitly claims the buffer does not exist

`isotropic-control-shows-same-effect` · verdict **CONFIRMED** · kind inconsistency · effort days  

**Where:** ICLR_2026/sec-rfnn.tex line 65 ("The buffer does not exist in Bonnaire's isotropic case"); ICLR_2026/sec-intro.tex line 17

**Claim:** Running the identical construction with sigma_perp = sigma_sig = 1 (Bonnaire's isotropic Sigma_data, the setting in which the paper says the noise-dim buffer is absent) still produces a gap that grows by an order of magnitude with d_lat. The claimed mechanism-specific signature is therefore present when the mechanism is, by the paper's own definition, absent -- so the signature does not identify the mechanism.

**Evidence:** Measured spectra of U with sigma_noise = 1.0, sigma_signal = 1.0, dint=5, n=500, p=64 d_lat, t=0.01: (d_lat, tau_gen = 1/lambda[dint-1], tau_mem = 1/lambda[d_lat], tau_mem - tau_gen) = (10, 23.4, 1006.5, 983.1); (20, 39.5, 2860.5, 2821.1); (40, 63.9, 6288.1, 6224.3); (100, 127.7, 10403.7, 10276.0). A 10.5x growth in the gap with no anisotropy at all.

**Fix:** Either drop the claim that the buffer is absent in the isotropic case, or -- better -- run the isotropic sweep as a control in the paper and confront the fact that it reproduces the headline signature. The discriminating control the paper is missing is one that holds E[||x||^2]/d_lat fixed as d_lat grows; note that none of the three existing width controls (p=64 d_lat, p=d_lat+n+300, fixed-p with sigma_perp^2 ∝ 1/(d_lat-dint)) does this -- all three let the mean per-coordinate norm fall with d_lat, so none of them separates the mode-count story from the norm-dilution story.


**Referee:** I ran the control myself (sigma_noise = sigma_signal = 1.0, d_int=5, n=500, p=64 d_lat, t=0.01, seed 42, S=50) and reproduce the effect: (d_lat, 1/lambda[d_int-1], 1/lambda[d_lat], gap) = (10, 23.8, 984.7, 961), (20, 36.7, 2882, 2845), (40, 63.3, 6339, 6275), (100, 129.6, 10,453, 10,323) — a 10.7x growth in the gap with Sigma_data = I. The claim being contradicted is quoted correctly (sec-rfnn.tex line 65: 'The buffer does not exist in Bonnaire's isotropic case'; sec-intro.tex line 17 makes the same move). The claim is also questionable on the paper's own terms independent of the numerics: with isotropic Sigma_data there are still d_lat - d_int non-mean-carrying data directions sitting between the mean-driven modes and the sample bulk — what anisotropy adds is spectral SEPARATION of that block, not its existence. So the headline signature (a gap that grows with d_lat) is not diagnostic of the claimed mechanism.


**Referee correction:** State precisely what changes and what does not: under isotropic Sigma_data the noise-dim modes still exist and still sit between signal and sample modes, but they merge with the signal bulk into Bonnaire's single rho_2, so the four-bulk STRUCTURE is absent while the count and the growing gap are not. The finding's point about the missing control is the important one: none of the three width controls (p = 64 d_lat, p = d_lat + n + 300, fixed p with signoise^2 ∝ 1/(d_lat-d_int)) holds E||x||^2/d_lat fixed — I measure it falling 1.52 -> 0.38 (sigma_perp=0.5) and 1.39 -> 0.14 (sigma_perp=0.01) across d_lat = 10 -> 100 — so none of them separates mode-count from norm-dilution.


---

## [MAJOR] The memorization observable itself drifts strongly and non-monotonically with d_lat, and in the synthetic sweeps the canonical 1% threshold is crossed only at the LARGEST d_lat

`memorization-metric-dlat-confound` · verdict **PLAUSIBLE** · kind gap · effort days  

**Where:** sigma_noise_0.5/exp2_mlp/raw_data/*/metrics.jsonl and sigma_noise_0.01/exp2_mlp/raw_data/*/metrics.jsonl (mean_nn_ratio, memorization_fraction); ICLR_2026/sec-appendix.tex lines 88-107 (metric definitions)

**Claim:** The Somepalli criterion is a nearest-neighbour DISTANCE RATIO test in R^{d_lat}; its sensitivity is a strong function of d_lat, in opposite directions at the two noise levels. And in the synthetic MLP sweeps the canonical tau_mem is essentially never observed: memorization_fraction crosses 1% in exactly one run out of 22, and that run is d_lat = 200 -- the widest buffer, i.e. the opposite of the predicted ordering. Every reported synthetic and real-data tau_mem therefore rests on the gen-gap > 0.02 proxy, whose "monotonically tracks the canonical tau_mem" justification has one supporting data point.

**Evidence:** mean_nn_ratio (final) across d_lat = 5,8,10,15,20,30,40,50,100,150,200:
 sigma_perp=0.01: 1.06, 1.11, 1.13, 1.21, 1.28, 1.39, 1.48, 1.56, 1.98, 2.34, 2.59 (rises 2.4x with d_lat -- memorization becomes mechanically harder to detect);
 sigma_perp=0.5: 1.06, 1.05, 1.03, 1.02, 1.00, 0.93, 0.88, 0.84, 0.75, 0.73, 0.71 (falls -- easier to detect).
max memorization_fraction over training: sigma_perp=0.5 gives 0.0062, 0.0010, 0.0008, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0002, 0.0028, 0.0144 -- only d_lat=200 exceeds the 1% threshold; sigma_perp=0.01 never exceeds 0.0068 (at d_lat=5) and is identically 0 for d_lat >= 30. The appendix's own justification, line 103-107: "we substitute the gen-gap proxy ... which monotonically tracks the canonical tau_mem on the MLP".

**Fix:** Report the raw mean_nn_ratio(d_lat) curves in the paper and address the metric's d_lat drift head-on (e.g. by calibrating the ratio threshold per d_lat against a held-out-sample null). State plainly that the canonical Somepalli tau_mem is censored in 21/22 synthetic runs and that all synthetic and real-data tau_mem values are the gen-gap proxy. Validate the proxy-vs-canonical correspondence on more than one run before relying on it.


**Referee:** The metric-drift half reproduces exactly. Final mean_nn_ratio across d_lat = 5..200: sigma_perp=0.01 -> 1.065, 1.113, 1.134, 1.214, 1.278, 1.389, 1.481, 1.562, 1.981, 2.341, 2.589; sigma_perp=0.5 -> 1.063, 1.055, 1.030, 1.016, 0.998, 0.933, 0.878, 0.835, 0.755, 0.726, 0.713. Max memorization_fraction at sigma_perp=0.5 -> 0.0062, 0.0010, 0.0008, 0, 0, 0, 0, 0, 0.0002, 0.0028, 0.0144, and never above 0.0068 at sigma_perp=0.01. So in the 300k-step sweeps the finding is exactly right. But the headline generalization ('the canonical taumem is essentially never observed; 1 of 22 runs, at d_lat=200, the opposite of the predicted ordering') is refuted by data already in the corpus: multiseed_runs/exp2_mlp_dlat_sn05_5m_bonnaire_hidden256 (5 seeds x d_lat in {5,8,10,12,15,20,25,30,35,40}, 5M steps — the sweep the appendix says the body timescale figure uses) crosses 1% in 37 of 50 runs, and the crossing time is monotone in exactly the predicted direction: 150k, 250k, 300k, 400k, 600k, 1050k, 2550k for d_lat = 5..25, censored at 30-40. Max memorization fraction there falls 0.299 -> 0.0018. sec-mlp.tex reports final memorization from these runs, not a threshold, so 'every reported synthetic tau_mem rests on the gen-gap proxy' is not accurate either.


**Referee correction:** Corrected finding: the Somepalli nearest-neighbour ratio is itself a strong function of d_lat (final mean_nn_ratio drifts 2.4x with d_lat at sigma_perp=0.01 and in the opposite direction at sigma_perp=0.5 in the 300k sweeps; in the 5M multiseed sweep it drifts 0.52 -> 0.87 as d_lat goes 5 -> 40), so a fixed 1/3 ratio threshold does not measure the same thing at each d_lat and the headline 'memorization falls from ~30% to near zero' is partly confounded with metric sensitivity. The fix stands: report the raw mean_nn_ratio(d_lat) curves and calibrate the threshold per d_lat against a held-out-sample null. Drop the '1 of 22 runs / opposite ordering' framing — it is an artifact of restricting attention to the 300k-step sweeps, and the 5M multiseed sweep shows the canonical metric behaving as the paper predicts (while simultaneously showing that the gen-gap proxy does not track it at all).


---

## [MAJOR] In the RFNN sweep the measured memorization onset is non-monotone in d_lat, and the zero-buffer run (d_lat = d_int = 5) has one of the LATEST onsets

`rfnn-taumem-nonmonotone-and-zero-buffer-latest` · verdict **CONFIRMED** · kind unsupported-claim · effort days  

**Where:** sigma_noise_0.5/exp2_rfnn/raw_data/*/metrics.jsonl and sigma_noise_0.01/exp2_rfnn/raw_data/*/metrics.jsonl; claim in ICLR_2026/sec-rfnn.tex line 65 and sec-intro.tex line 26

**Claim:** The RFNN is the track where the theory is supposed to apply exactly (frozen W, quadratic loss, gradient flow). Recomputing tau_mem there with the paper's own proxy (first sustained gen_gap > 0.02, in steps) yields a non-monotone, essentially unordered sequence, and the d_lat = 5 configuration -- where the noise-dim bulk has width zero and the buffer is by construction absent -- has a LATER onset than seven of the eight wider configurations.

**Evidence:** sigma_noise_0.5/exp2_rfnn, d_lat = 5,10,20,40,60,80,100,150,200 -> tau_mem = 120k, 55k, 75k, 20k, 10k, 60k, 45k, 60k, 90k steps. The buffer prediction is a monotone increase; the observed sequence falls 12x from d_lat=5 to d_lat=60 and then rises. sigma_noise_0.01/exp2_rfnn, d_lat = 5,8,10,15,20,30,40 -> 120k, 45k, never, 40k, never, step 1(!), never -- 3 of 7 never cross and one crosses at initialization, i.e. the proxy is not usable at this noise level.

**Fix:** Show the per-run gen-gap trajectories for the RFNN sweep in the appendix rather than only threshold crossings, and either demonstrate that the non-monotonicity is seed noise (multiple seeds, error bars -- all these runs are seed 42 only) or withdraw the RFNN as empirical support for the delay claim. Given that the RFNN is the only track the theory covers, this is not optional.


**Referee:** Reproduced from the raw metrics. Using 'first two consecutive evals with gen_gap > 0.02' I get exactly the reviewer's numbers: sigma_noise_0.5/exp2_rfnn, d_lat = 5..200 -> 120k, 55k, 75k, 20k, 10k, 60k, 45k, 60k, 90k; sigma_noise_0.01/exp2_rfnn, d_lat = 5..40 -> 120k, 45k, never, 40k, never, step 1, never. The underlying trajectories explain why: the RFNN gen-gap oscillates in sign around zero with amplitude comparable to the 0.02 threshold (e.g. sigma_perp=0.5, d_lat=10: 0.014, -0.000, -0.025, 0.023, -0.007, -0.009, 0.008, ... ), so the crossing time is dominated by noise. All these runs are seed 42 only (raw_data contains a single s42 directory per d_lat), so there are no error bars. Since the RFNN is the only track the theory covers, this is material.


**Referee correction:** The specific sub-claim 'd_lat=5 has a later onset than seven of the eight wider configurations' is definition-dependent and should be restated. Under the paper's own literal definition (first eval crossing 0.02, sec-appendix.tex line 31) I get d_lat = 5..200 -> 20k, 15k, 25k, 10k, 10k, 50k, 45k, 60k, 80k, in which d_lat=5 is mid-pack (three wider configs are earlier). Under 'two consecutive evals' d_lat=5 is the latest of all nine; under 'gap stays above 0.02 for the rest of training' d_lat=5 never qualifies at either noise level. What is robust across all three definitions is (i) non-monotonicity and (ii) that the answer swings by an order of magnitude with an arbitrary smoothing choice — which is the point to make.


---

## [MAJOR] The sample-bulk edge eta_bar_star/psi_p contains no n, which is impossible for a rank-n bulk; it should be eta_bar_star/n

`sample-bulk-edge-missing-n` · verdict **CONFIRMED** · kind error · effort hours  

**Where:** ICLR_2026/sec-appendix.tex Lemma "Rank and edge of U^diag" lines 712-736 and Theorem edge table row "Sample" line 801; identical in sec-appendix-fourbulk.tex lines 134-157, 223

**Claim:** U^diag is a sum of n asymptotically orthogonal rank-one terms with total trace eta_bar_star (since tr(U) = E||phi||^2 = E[tanh^2] = O(1) and the higher-Hermite share of it is eta_bar_star). Spread over exactly n orthonormal directions, each eigenvalue is eta_bar_star/n. The stated eta_bar_star/psi_p would give tr(U^diag) = n eta_bar_star/psi_p = eta_bar_star n d_lat/p, which is wrong by a factor n d_lat/p (7.8x at the default p = 64 d_lat, n = 500) and diverges in the proportional limit at fixed psi_p, psi_n. This matters directly for the buffer corollary: tau_mem = 1/lambda_sample is the quantity the corollary bounds.

**Evidence:** Stated edge at sigma_perp=0.5, d_lat=20: eta_bar_star/psi_p = 2.83e-4. Corrected eta_bar_star/n = 3.63e-5. Measured median sample-bulk eigenvalue = 1.10e-5, measured top of sample bulk = 4.81e-4. At d_lat=100: stated 2.10e-5, corrected 2.68e-6, measured median 7.84e-6. Also note the sample bulk is the only bulk whose edge must scale as 1/n, and the stated form has no n dependence at all, so it predicts the sample bulk does not move when n changes.

**Fix:** Correct the Lemma 2 edge to eta_bar_star/n and propagate through the Theorem's edge table and the bulk-gap-scaling paragraph (the noise-dim-to-sample ratio becomes mu_1^2 beta_t^2 n /(d_lat eta_bar_star), which is where the true d_lat dependence of the gap lives).


**Referee:** Verified, and the defect is worse than stated: Lemma 2's own proof contradicts its own statement.

sec-appendix.tex:713-736 (= sec-appendix-fourbulk.tex Lemma 2). The lemma claims the nonzero eigenvalues of U^diag concentrate at bar_eta_star/psi_p. Its proof sets Lambda_mu = eta_star/(p*n) and computes the eigenvalues as Lambda_mu*||phi^perp||^2 -> eta_star^2/(p*n) — which is neither the lemma's own bar_eta/psi_p nor the correct answer. Eq. (U-diag) itself carries a spurious extra eta_star factor relative to U = (1/n) sum phi phi^T with phi already carrying 1/sqrt p.

Correct derivation (mine): U's nonzero spectrum equals that of K/(n p) with K_{mu,nu} = tanh_mu . tanh_nu. Writing tanh_mu = c1(W x_mu/sqrt d) + r_mu with ||r_mu||^2 ~ p*eta_star gives residual eigenvalues eta_star/n in theory units, i.e. p*eta_star/n in code units. No 1/psi_p, and an explicit 1/n.

Numerics (mine, sigma=0.5, eta_star = E[tanh^2(sqrt q z)] - c1(q)^2 q): theorem value d_lat*bar_eta = 0.389, 0.442, 0.462, 0.488, 0.527, 0.573, 0.623, 0.758, 0.898 versus measured sample-bulk medians 0.00180, 0.00370, 0.01446, 0.02941, 0.03834, 0.04487, 0.05041, 0.06218, 0.07077 — too large by 216x down to 12.7x. Corrected p*eta/n gives 0.0498...0.1149, i.e. within a factor 1.6-2.7 for d_lat>=40. The reviewer's per-sample mean refinement is not something I re-ran, but the two-orders-of-magnitude failure of the published form and the correctness of the 1/n form are both solid.


**Referee correction:** Add: the small-d_lat entries (d_lat = 5, 10) are not a fair test of any sample-edge formula because p = 320, 640 is not much larger than n = 500, so the 'sample bulk' index window d_lat..d_lat+n is truncated or overlaps the tail; the clean comparisons are d_lat >= 40, where the published form is still 12-17x too large. Also note the deeper issue: with the corrected edge eta_star/n = eta_star/(psi_n d_lat), the sample edge is o(1) in the stated proportional limit, so it is no longer separated in ORDER from the 'o(1)' rank-null bulk — the theorem's four-way edge hierarchy does not survive its own asymptotics.


---

## [MAJOR] The d_lat sweep runs directly along the axis the theorem holds fixed, and the theorem's rank-null hypothesis fails outright at the zero-buffer baseline d_lat = 5

`theorem-hypotheses-violated-by-the-sweep` · verdict **CONFIRMED** · kind gap · effort days  

**Where:** ICLR_2026/sec-appendix.tex lines 603-605 (proportional limit, psi_n fixed, psi_p > 1 + psi_n) and Theorem line 786-805; sweep table lines 133-136

**Claim:** The theorem assumes psi_n = n/d_lat FIXED, but every experiment holds n = 500 fixed and sweeps d_lat from 5 to 200, so psi_n varies from 100 to 2.5 -- the sweep is precisely the direction the asymptotic hypothesis forbids. Worse, the hypothesis psi_p > 1 + psi_n requires p > d_lat + n; at d_lat = 5, p = 64*5 = 320 < 505, so the rank-null count p - d_lat - n = -185 is negative and the four-bulk theorem is vacuous at exactly the configuration used as the no-buffer control. At d_lat = 8 the tail is 4 modes.

**Evidence:** sec-appendix.tex line 604-605: "psi_p = p/d_lat and psi_n = n/d_lat fixed and psi_p > 1 + psi_n (so the rank-null tail has room to exist)". Sweep table line 133-134: RFNN d_lat in {5, 8, 10, 15, 20, 30, 40} and {60, 80, 100, 150, 200} with p = 64 d_lat; appendix line 88 fixes n = 500. Raw config confirms: sigma_noise_0.5/exp2_rfnn/raw_data/di5_d5_n500_s42/config.json has d_latent 5, n 500, p_ratio 64.

**Fix:** Either restate the theorem with n fixed and d_lat -> infinity (a different and easier regime, in which the sample bulk is a fixed-rank perturbation), or run the sweep with n proportional to d_lat so psi_n is actually held fixed. Exclude d_lat = 5 (and flag d_lat = 8) from any claim that invokes the four-bulk theorem, or re-run those points with p >= d_lat + n + margin.


**Referee:** Both parts verified in the files. sec-appendix.tex lines 603-605 state the proportional limit with psi_p = p/d_lat and psi_n = n/d_lat FIXED and psi_p > 1 + psi_n 'so the rank-null tail has room to exist'. sec-appendix.tex line 88 fixes n = 500 for every run and the sweep table (lines 133-136) varies d_lat over {5,8,10,15,20,30,40} and {60,80,100,150,200} — so psi_n runs from 100 down to 2.5 along precisely the axis the hypothesis holds fixed. And sigma_noise_0.5/exp2_rfnn/raw_data/di5_d5_n500_s42/config.json confirms d_latent=5, n=500, p_ratio=64, hence p = 320 < d_lat + n = 505: psi_p = 64 < 1 + psi_n = 101, the stated hypothesis fails, and the predicted rank-null count p - d_lat - n = -185 is negative. At d_lat=8, p=512 leaves a 4-mode tail. d_lat=5 is exactly the zero-buffer control the paper leans on (sec-rfnn.tex figure caption: 'at d_lat = d_int there is no green buffer').


**Referee correction:** Worth adding to the fix list: this also undercuts sec-rfnn.tex's claim that the index-block coloring is an a priori prediction rather than a post-hoc partition, since at d_lat=5 and 8 the predicted block boundaries at indices d_lat and d_lat+n do not fit inside p at all. Also, the more honest restatement is probably n fixed with d_lat -> infinity (sample bulk as a fixed-rank perturbation), which is both closer to the experiments and closer to the latent-diffusion regime the paper cares about.


---

## [MAJOR] The buffer bound is derived for gradient flow on a frozen quadratic and then applied to Adam-trained MLPs with time-varying t and to real latent diffusion, with no argument

`buffer-transported-to-adam-and-real-data` · verdict **CONFIRMED** · kind unsupported-claim · effort weeks  

**Where:** ICLR_2026/sec-appendix.tex lines 148-158 (hyperparameter table) and lines 405-440 (real-data metrics and the "~10x delay ... attributable to the noise-dim buffer" claim); ICLR_2026/sec-mlp.tex, sec-real-data.tex

**Claim:** Eq. (rfnn-mode-decay) requires (i) frozen features, (ii) an exactly quadratic loss, (iii) full-batch gradient flow, (iv) a single fixed diffusion time t. The MLP and real-data tracks satisfy none of these: trainable first layer, Adam, batch 256, t ~ Unif[0.01, 3.0]. Under Adam the per-mode rate is not lambda_i (Adam's preconditioner approximately whitens the curvature spectrum, which is exactly the structure the whole argument rests on), and averaging over t mixes four different four-bulk spectra with t-dependent alpha_t^2, beta_t^2. The real-data section nonetheless attributes a 10x tau_mem growth to "the noise-dim buffer".

**Evidence:** sec-appendix.tex lines 150-155: MLP optimizer Adam, lr 1e-4, batch 256, diffusion time t ~ Unif[0.01,3.0]; RFNN full-batch GD, fixed t = 0.01. sec-appendix.tex lines 429-434: "tau_mem grows with d_lat on both datasets, going from 10k to 80k steps on MNIST and from 30k to 440k steps on CelebA across the sweep---a ~10x delay in memorization onset attributable to the noise-dim buffer." Meanwhile, in the synthetic MLP at sigma_perp=0.01 the same sweep shows tau_gen growing 51x (5k -> 255k steps) and tau_mem censored (never crossed) for every d_lat >= 20 -- i.e. the whole run slows down and the "delay" is censoring, not selective buffering.

**Fix:** Restrict every theory-derived claim to the RFNN track. For the MLP and real-data tracks, either present the delay as a phenomenological observation with the tau_gen curve shown alongside (so the reader sees the global slowdown), or supply an argument -- Adam-preconditioned mode dynamics, t-averaged effective spectrum -- for why the RFNN timescale ordering transports. Also report the censoring explicitly: how many d_lat points never crossed the tau_mem threshold within budget.


**Referee:** The hypotheses of Eq. (rfnn-mode-decay) and the settings of the other two tracks are as the finding states, and I verified both from the files: sec-appendix.tex Table (lines ~148-158) gives MLP = Adam, lr 1e-4, batch 256, t ~ Unif[0.01,3.0] versus RFNN = full-batch GD at fixed t = 0.01 with a frozen first layer; sec-appendix.tex lines 429-434 read verbatim 'taumem grows with dlat on both datasets, going from 10k to 80k steps on MNIST and from 30k to 440k steps on CelebA across the sweep---a ~10x delay in memorization onset attributable to the noise-dim buffer.' No argument is offered anywhere for why a per-mode rate lambda_i survives Adam's preconditioner or an average over t (which mixes spectra with different alpha_t^2, beta_t^2 — note alpha_t^2/beta_t^2 collapses to 1 as t grows, by the paper's own 'Bulk-gap scaling' paragraph, so most of Unif[0.01,3] has no four-bulk separation at all). The real-data tau_mem is additionally the gen-gap proxy (sec-appendix.tex lines 411-424), justified by a synthetic correspondence that the corpus contradicts.


**Referee correction:** Two corrections to the supporting argument. (1) sec-mlp.tex does partly acknowledge the mismatch ('The MLP result is qualitatively sharper than the frozen-feature prediction: at high dlat, feature learning appears to find sparser, signal-aligned directions instead of spending equal effort on all random-feature modes'), so the charge is unargued transport of a quantitative bound, not total silence. (2) The sub-claim that 'the delay is censoring, not selective buffering' is only established for sigma_noise_0.01/exp2_mlp; on the 5M multiseed sigma_perp=0.5 MLP sweep the canonical memorization onset genuinely moves 150k -> 2550k across d_lat = 5..25 with 5 seeds and is only censored at d_lat >= 30. Restrict the censoring charge to the sigma_perp=0.01 sweep and to the real-data runs, and separately demand that the censoring counts be reported.


---

## [MAJOR] At sigma_perp = 0.01 there is no detectable noise-dim/sample boundary, contradicting the 'razor-sharp cliff in all cases' claim

`no-cliff-at-sigma001` · verdict **REFUTED** · kind unsupported-claim · effort hours  

**Where:** /Users/ryan/Desktop/latent_space_diffusion_analysis/sigma_noise_0.01/four_bulk/README.md ('sharp cliff at both boundaries', 'Parameters: ... sigma_noise=0.01') and sigma_noise_0.01/exp2_rfnn/README.md ('The cliff at the green line is razor-sharp in all cases')

**Claim:** At sigma_perp = 0.01 the noise-dim bulk is not separated from the sample bulk at all. The contrast at index d_lat is essentially 1, and a blind largest-log-gap detector never places a boundary at d_lat. The four-bulk structure that the buffer mechanism requires is present only at sigma_perp = 0.5.

**Evidence:** r(d_lat) = lambda_{d_lat-1}/lambda_{d_lat} from sigma_noise_0.01/exp2_rfnn/raw_data/*/eigenvalues_pre.npy: 1.108 (d_lat=8), 1.077 (10), 1.044 (15), 1.013 (20), 1.008 (30), 1.138 (40). Compare sigma_perp=0.5: 4.41, 7.73, 13.10, 15.46, 18.51, 17.98, 18.04, 12.26. Blind top-4 log-gap indices at sigma=0.01, d_lat=20: 2, 5, 54, 1279 -- index 20 does not appear; at d_lat=30: 2, 5, 64, 65; at d_lat=40: 2, 5, 75, 2559. The repo's own bulk_summary.json gives gap_noise_to_sample_dec = 0.0036-0.056 decades (factor 1.008-1.14) for every sigma_noise=0.01 run.

**Fix:** Restrict the four-bulk empirical claim to sigma_perp = 0.5 and state explicitly that at sigma_perp = 0.01 the structure degenerates to two bulks -- which the appendix's own 'Bulk-gap scaling' paragraph predicts but the figure READMEs contradict. Report r(d_lat) or the blind-detector output rather than eyeballed cliffs.


**Referee:** The measurements are right but the paper already says exactly this, and the added theoretical claim is overstated.

My own r(d_lat) = lambda_{d_lat-1}/lambda_{d_lat} at sigma_perp=0.01: 1.108, 1.077, 1.044, 1.013, 1.008, 1.138 for d_lat = 8, 10, 15, 20, 30, 40, versus 4.41, 7.73, 13.10, 15.46, 18.51, 17.98, 18.04, 12.25 at sigma_perp=0.5. Blind detector at sigma=0.01 never returns index d_lat. All reproduced.

But ICLR_2026/sec-rfnn.tex:59 already states, in the submitted main text: 'The cliff at index dint between the two data bulks is sharp at both noise scales; the cliff at dlat between data and sample bulks is sharp at signoise = 0.5 and DEGENERATES TO A SMOOTH TRANSITION at signoise = 0.01.' The appendix's Bulk-gap scaling paragraph (sec-appendix.tex:820-838) predicts the same merge from beta_t^2 vs bar_eta_star/mu_1^2, and the finding concedes this. So the paper is already correct; what is stale is two data-folder READMEs (sigma_noise_0.01/four_bulk/README.md:11 and sigma_noise_0.01/exp2_rfnn/README.md:9) that are not part of the submission. That is a housekeeping nit, not a paper-level unsupported claim, and certainly not 'major'.

The extra claim, 'the four-bulk structure that the buffer mechanism requires is present only at sigma_perp = 0.5', is wrong. At sigma=0.01, d_lat=40 the noise-dim bulk median is 0.811 and the sample-bulk median is 0.0030 — 2.4 decades apart, with the signal bulk 2 decades above that (r(d_int) = 32-65). What is missing is a GAP at the boundary index, because the sample bulk's upper tail is broad and overlaps the noise-dim band (bulk_summary.json: gap_noise_to_sample_dec = 0.019-0.032 decades). Overlapping supports is not 'not separated at all'.


**Referee correction:** Reduced finding, severity minor / documentation-only: two data-folder READMEs (sigma_noise_0.01/four_bulk/README.md:11, sigma_noise_0.01/exp2_rfnn/README.md:9) claim a 'sharp'/'razor-sharp' cliff at index d_lat for sigma_noise=0.01 where the measured contrast is r(d_lat) = 1.01-1.14. Fix the READMEs to match the paper, which already states the correct thing. Do NOT restrict the four-bulk claim to sigma_perp = 0.5: at sigma_perp = 0.01 the four populations are still separated by decades in median value; only the boundary gap closes.


---

## [MAJOR] The unexplained 63.8x factor in the repo's self-critique is the missing psi_p normalization, not Marchenko-Pastur spreading

`63x-gap-is-normalization-not-mp` · verdict **CONFIRMED** · kind error · effort hours  

**Where:** /Users/ryan/Desktop/latent_space_diffusion_analysis/_next_steps/n_shape_heuristic_derivation.md:351 (Pass 5); cross-ref ICLR_2026/sec-appendix.tex Remark \ref{rem:MP}

**Claim:** Pass 5 reports the empirical noise-dim bulk eigenvalue at d_lat=40, sigma_perp=0.01 as 0.813 versus formula mu_1^2 lambda_null = 0.013, and attributes the 63.8x gap to 'Marchenko-Pastur spreading of the W-spectrum'. It cannot be MP spreading: the appendix's own Remark on MP puts the relative spreading at Delta_MP ~ 4/sqrt(psi_p) = 0.50 at psi_p = 64, i.e. a factor <= 1.5. The 63.8 is 64 = psi_p: it is exactly the p-normalization mismatch between the theory's U (which carries 1/p) and the code's U (which does not), compounded by the wrong mu_1^2.

**Evidence:** I confirm the measured number: median of lambda[5:40] at sigma_noise_0.01/exp2_rfnn/raw_data/di5_d40_n500_s42/eigenvalues_pre.npy is 0.8109 (bulk mean 0.813, matching the note). Decomposition of the 63.8x: (i) 0.813 / (psi_p * mu_1^2 * beta_t^2) = 0.813/(64*0.367*0.0199) = 0.813/0.467 = 1.74; (ii) further replacing mu_1^2 = 0.367 by c1(q)^2 = 0.6032 at q = 0.361 gives 0.813/0.768 = 1.06. So 63.8 = 64 (units) x 1.06 (residual), with the c1 correction absorbing the rest. The note's own mu_1^2 value is also wrong: it uses 0.64, whereas mu_1 = E[z tanh z] = 0.6055, mu_1^2 = 0.3670 (Monte Carlo, 2e7 samples).

**Fix:** Correct the Pass 5 diagnosis. Everything downstream in that file that carries the 63.8x 'RMT correction' (the rescaled lambda = 7.41 at t=0.1, the exponent 444.5, the RFNN self-consistency check) is built on a units error and must be redone. More importantly, this is direct evidence that the theory and the measurement code disagree on the normalization of U, which should be stated explicitly in the paper.


**Referee:** Fully verified, and there are additional unit errors in the same passage.

_next_steps/n_shape_heuristic_derivation.md Pass 5 reads verbatim: 'the empirical noise-dim bulk eigenvalue is 0.813 ... while the simple formula at t=0.01 gives mu_1^2 lambda_null = 0.64 x 0.020 = 0.013 - a correction factor of 63.8x from random-matrix theory (Marchenko-Pastur spreading of the W-spectrum).'

I confirm the measurement: median of lambda[5:40] in sigma_noise_0.01/exp2_rfnn/raw_data/di5_d40_n500_s42/eigenvalues_pre.npy is 0.8109.

The MP attribution is impossible on the paper's own terms: Remark rem:MP (sec-appendix.tex:768-784) puts Delta_MP ~ 4/sqrt(psi_p) = 0.50 at psi_p = 64, i.e. a factor at most ~1.5, not 63.8.

The decomposition checks out to three digits: 0.813/(psi_p * mu_1^2 * beta_t^2) = 0.813/(64*0.36653*0.019899) = 1.742, and replacing mu_1^2 by c1(q)^2 = 0.6022 at the measured q = 0.363 gives 0.813/0.7669 = 1.060. So 63.8 = 64 (the psi_p normalization mismatch between theory-U and code-U) x 1.06 (residual). The note's mu_1^2 = 0.64 is also wrong: mu_1 = E[z tanh z] = 0.60542, mu_1^2 = 0.36653.


**Referee correction:** Strengthen: the same passage carries a second, independent unit error that the finding misses. Its 'decisive check' computes the RFNN per-mode rate as lambda_eff = (0.01/(Delta_t * n)) * 0.813 = 8.2e-4. From the actual loss and optimizer (code/experiment_v2_rfnn.py:95-99 and lr = 0.01*d/Delta_t at line 236), the correct gradient-flow rate is 2*Delta_t*lr*lambda_code/(d*p) = 0.02*lambda_code/p = 6.35e-6 — 129x smaller. So the exponent 493 and the 'formula fails even in the linear system' verdict rest on two compounding unit errors, not one. Every downstream number in Pass 5 (the rescaled 7.41, the exponents 444.5 and 493, the RFNN self-consistency check) must be redone. Mitigating: this is an internal scratch note whose own conclusion is 'the derivation does not close', so nothing in the submitted draft currently depends on it.


---

## [MAJOR] No RFNN run in the repo is long enough to observe tau_mem; the RFNN dynamics data cannot test the buffer at all

`rfnn-runs-never-reach-taumem` · verdict **CONFIRMED** · kind gap · effort weeks  

**Where:** sigma_noise_0.5/exp2_rfnn/raw_data/*/metrics.jsonl and config.json (total_steps=300000); ICLR_2026/sec-rfnn.tex:69 (gen-gap as tau_mem proxy)

**Claim:** The paper reads tau_mem off the RFNN gen-gap. Using the spectrum from the very same runs plus the run's own learning rate, tau_mem is 4.3e6-9.0e6 steps at sigma_perp=0.5 and 2.5e7-4.3e7 at sigma_perp=0.01, but every run is 3e5 steps. The gen-gap traces are consistent with noise, not memorization, and their ordering in d_lat is the reverse of the claim.

**Evidence:** config.json total_steps=300000 for all exp2_rfnn runs. Predicted tau_mem = 50p/lambda_sample: 8.65e6 (d_lat=10) ... 9.04e6 (200) at sigma=0.5; 3.85e7 (d_lat=8) ... 4.27e7 (40) at sigma=0.01 -- 15x to 140x beyond budget. Measured gen_gap maxima over the whole run are 0.048-0.093 (sigma=0.5) and 0.025-0.065 (sigma=0.01), with negative final values at d_lat=8 (-0.0255) and 15 (-0.0051), i.e. within noise. First crossing of gen_gap > 0.05 (sigma=0.5): 125k (d_lat=5), 170k (10), 215k (20), never (40), 230k (60), 230k (80), 170k (100), 175k (150), 165k (200) -- the widest latent crosses EARLIEST among d_lat >= 20. Also final score_error at sigma=0.01 is 253-560 (vs 0.29 at d_lat=5), so those runs are diverged and carry no timescale information.

**Fix:** Either run the RFNN to ~1e7 steps or drop the RFNN gen-gap as a tau_mem measurement and present the RFNN as a spectrum-only experiment. As it stands the RFNN section presents a dynamical claim that its own runs cannot reach.


**Referee:** The core gap is real and, using the paper's actual threshold, larger than the finding claims.

Every exp2_rfnn config.json has total_steps = 300000 (verified). Predicted tau_mem from the same runs' spectra: 4.35e6-9.04e6 steps at sigma=0.5 and 2.5e7-4.3e7 at sigma=0.01 using the sample-bulk median, or 1.7e4-3.7e6 using the sample-bulk onset. The paper's RFNN tau_mem proxy is not the finding's ad-hoc 0.05 but gen-gap > 0.02 (sec-appendix.tex:31 and 106). Measured first crossing of 0.02, sigma=0.5: 20k, 15k, 25k, 10k, 10k, 50k, 45k, 60k, 80k for d_lat = 5..200 — i.e. the proxy fires at 0.2%-2% of the median-based spectral tau_mem. So the quantity the paper calls the RFNN tau_mem and the quantity the spectral clock predicts differ by two orders of magnitude and cannot both be 'the sample bulk being absorbed'.

The divergence at sigma_perp = 0.01 is also confirmed: final score_error is 253-560 for d_lat = 8..40 versus 0.29 at d_lat = 5, so those runs carry no usable timescale.

Mitigating and worth stating: the submitted ICLR_2026 draft does not currently present RFNN gen-gap data — every RFNN figure in ICLR_2026/figures is a spectrum histogram — so the draft is already close to the finding's proposed fix. The dynamical reading survives only in sec-rfnn.tex:69 ('the gen-gap ... opens at taumem') and in the uncompiled sec-appendix.tex proxy definitions.


**Referee correction:** One sub-claim is wrong and should be dropped: 'their ordering in d_lat is the reverse of the claim'. That holds only at the reviewer's 0.05 threshold. At the paper's own 0.02 threshold the crossings order 10k, 10k, 50k, 45k, 60k, 80k for d_lat = 40..200 — increasing with d_lat, i.e. in the direction the paper predicts. The defensible finding is narrower and cleaner: the RFNN gen-gap proxy fires ~100x earlier than the spectral clock's tau_mem, so whatever the proxy measures, it is not sample-bulk absorption, and no RFNN run is within 15-140x of the horizon needed to observe the predicted event.


---

## [MAJOR] MLP memorization is non-monotone in d_lat and is 2.3x STRONGER at d_lat=200 than at d_lat=5

`mlp-memorization-non-monotone` · verdict **CONFIRMED** · kind inconsistency · effort days  

**Where:** sigma_noise_0.5/exp2_mlp/raw_data/*/metrics.jsonl; sigma_noise_0.5/exp2_mlp/README.md (which already states this); abstract.tex:2 and ICLR_2026/sec-rfnn.tex:65 (monotone-delay framing)

**Claim:** The headline claim is that widening d_lat delays memorization. The repo's own MLP sweep at sigma_perp=0.5 shows memorization disappearing for 15 <= d_lat <= 50 and then REAPPEARING at d_lat = 100, 150, 200, getting both stronger and earlier as d_lat grows further. This U-shape matches the U-shape I derive from the corrected spectrum and contradicts the monotone-buffer story.

**Evidence:** max memorization_fraction and first step with memorization_fraction > 0: d_lat=5: 0.0062 @ 5k; 8: 0.0010 @ 5k; 10: 0.0008 @ 15k; 15,20,30,40,50: 0 (never); 100: 0.0002 @ 265k; 150: 0.0028 @ 190k; 200: 0.0144 @ 170k. So onset moves EARLIER (265k -> 190k -> 170k) and peak memorization grows (0.0002 -> 0.0028 -> 0.0144) as d_lat goes 100 -> 200, ending 2.3x above the d_lat=5 baseline. mean_nn_ratio minima fall monotonically (1.059 -> 0.713) over the same range. The independently derived spectral clock tau_mem = 50p/lambda_sample also has its minimum at d_lat ~ 40 and rises on both sides, so both the theory (corrected) and the MLP data are non-monotone.

**Fix:** Either restrict the claim to a stated d_lat range (roughly d_int < d_lat <= 50 at these settings), or make the U-shape the finding and derive its minimum. Do not state a monotone delay in the abstract when the paper's own README documents the reappearance.


**Referee:** The concern is real, but the finding cites the wrong experiment; I found stronger evidence for the same point.

The finding's numbers from sigma_noise_0.5/exp2_mlp reproduce exactly (max mem 0.0062, 0.0010, 0.0008, 0, 0, 0, 0, 0, 0.0002, 0.0028, 0.0144 and onsets 265k -> 190k -> 170k for d_lat = 100 -> 200; min mean_nn_ratio falls monotonically 1.059 -> 0.713). But that sweep is NOT the paper's MLP evidence: it is 300k steps, one seed, hidden = 8*d_lat. The main-text sweep (sec-mlp.tex:9) is 5M steps, seeds 42-46, hidden FIXED at 256, d_lat in {5..40}. In that data (multiseed_runs/exp2_mlp_dlat_sn05_5m_bonnaire_hidden256) mean max-mem falls cleanly and monotonically 0.309, 0.226, 0.176, 0.131, 0.082, 0.039, 0.018, 0.0085, 0.0037, 0.0016 with onset moving 50k -> ~1.1M. So on the paper's own controlled data the main-text claim holds over its stated range, and the finding as filed does not contradict it.

What DOES contradict it is a sweep the finding never cites: multiseed_runs/exp2_mlp_dlat_sn05_5m_combined — same 5M steps, same 5 seeds, same sigma, differing only in hidden = 8*d_lat. There mean max-mem is 0.0139, 0.0051, 0.0046, 0.0039, 0.0031, 0.0030, 0.0038, 0.0046, 0.0068, 0.0100 for d_lat = 5..40: a clean U-shape with its minimum at d_lat = 20, rising 3.3x by d_lat = 40 — INSIDE the paper's own main-text range. So the sign of the effect at large d_lat depends on whether hidden width scales with d_lat, and the repo's two READMEs take opposite positions on which choice is the confound (sec-mlp.tex fixes width at 256; sigma_noise_0.5/exp2_mlp/README.md says scaling capacity is what 'avoids the fixed-hidden confound').


**Referee correction:** Re-file as: MLP memorization is non-monotone in d_lat whenever hidden width scales with d_lat, and this shows up inside the paper's own main-text range (5M steps, 5 seeds, hidden = 8*d_lat: minimum at d_lat = 20, 3.3x rebound by d_lat = 40). The fixed-width-256 sweep the main text actually plots is monotone over d_lat <= 40, so the abstract is not contradicted by the paper's plotted data — but the result is not width-robust and the paper claims the opposite by presenting only one width policy. Separately: sec-appendix-integrated.tex:313 and 328-329 claim the fixed-width sweep was extended through d_lat = 240 and that the monotone delay 'continues through dlat = 240'; I could find no d_lat > 40 runs for that sweep anywhere in the repo, and clean figures/final/synthetic_mlp_d_summary_bonnaire.csv stops at d_lat = 40. That extended claim is currently unbacked. Drop the assertion that memorization ends '2.3x above the d_lat=5 baseline' as a headline — it comes from a 300k-step single-seed run.


---

## [MAJOR] The theorem's proportional limit fixes psi_n = n/d_lat, but the sweep varies psi_n by 40x and violates psi_p > 1 + psi_n at the baseline

`proportional-limit-violated-along-the-sweep` · verdict **CONFIRMED** · kind gap · effort days  

**Where:** ICLR_2026/sec-appendix.tex:602-606 ('psi_p = p/d_lat and psi_n = n/d_lat fixed and psi_p > 1 + psi_n'); experimental settings in code/experiment_v2_rfnn.py (n=500 fixed, d_lat swept)

**Claim:** Every experiment holds n = 500 fixed and sweeps d_lat, so psi_n = 500/d_lat runs from 100 down to 2.5. The theorem is stated in a limit where psi_n is FIXED, so the sweep moves along the axis the limit forbids moving along. In addition, the theorem's standing hypothesis psi_p > 1 + psi_n fails outright at the paper's baseline configuration.

**Evidence:** psi_n = 500/d_lat = 100, 62.5, 50, 33.3, 25, 16.7, 12.5, 8.3, 6.25, 5.0, 3.3, 2.5 for d_lat = 5, 8, 10, 15, 20, 30, 40, 60, 80, 100, 150, 200. Hypothesis psi_p > 1 + psi_n with psi_p = 64 requires d_lat > 500/63 = 7.94, so it is FALSE at d_lat = 5 and marginal at d_lat = 8 (64 > 63.5). At d_lat = 5 the predicted rank-null count p - d_lat - n = 320 - 5 - 500 = -185 is negative -- a nonsensical output the paper's own Table tab:bulk-sizes silently replaces with B1 = 150. d_lat = 5 is precisely the 'Bonnaire two-bulk recovery' baseline the paper relies on.

**Fix:** Either state the theorem at fixed n with d_lat -> infinity (a different, honest limit, in which the sample bulk's eta_bar/n edge is a fixed-n quantity), or add a sweep that holds psi_n fixed by scaling n with d_lat. At minimum, mark d_lat = 5 and 8 as outside the theorem's stated hypotheses.


**Referee:** The arithmetic is unambiguous. sec-appendix.tex:602-606 states the limit with 'psi_p = pwidth/dlat and psi_n = nsamp/dlat FIXED and psi_p > 1 + psi_n'. Every RFNN run holds n = 500 and sweeps d_lat, so psi_n = 500/d_lat runs 100 -> 2.5, a 40x excursion along the axis the limit declares fixed. The standing hypothesis psi_p > 1 + psi_n with psi_p = 64 requires d_lat > 7.94: it is false at d_lat = 5 (64 vs 101) and marginal at d_lat = 8 (64 vs 63.5). At d_lat = 5 the theorem's rank-null count p - d_lat - n = 320 - 505 = -185 is negative, while Table tab:bulk-sizes reports B1 = 150 there without comment. d_lat = 5 is the d_lat = d_int no-buffer baseline the paper leans on.


**Referee correction:** Downgrade severity from blocking to minor. Stating a proportional-limit theorem and then sweeping one of the ratios is near-universal practice in RMT-flavored ML theory, and the fix is one sentence, not a re-derivation: state the theorem at fixed n with d_lat -> infinity (which is what the experiments actually do), and mark d_lat = 5 and 8 as outside the hypotheses. Note also that the corrected edges of findings 1-3 make the honest fixed-n limit the natural one anyway, since the signal edge c1(q)^2 alpha_t^2/d_lat and the sample edge eta_star/n are both naturally expressed at fixed n.


---

## [MAJOR] The stated tanh operating point ('inputs at ~0.4 on average') is wrong by 4x at the small-d_lat end of the sweep

`saturation-operating-point-misstated` · verdict **CONFIRMED** · kind error · effort hours  

**Where:** /Users/ryan/Desktop/latent_space_diffusion_analysis/eigenvalue_saturation/README.md ('keeps tanh inputs at ~0.4 on average -- well within the linear regime'); ICLR_2026/sec-appendix.tex:193-198 ('Known pitfalls') and the saturation appendix

**Claim:** The justification for treating tanh as linear (and mu_1 as constant) is that preactivations sit at ~0.4. The actual preactivation standard deviation, computed from the generated data, ranges from 1.66 at d_lat=5 down to 0.57 at d_lat=200. The '~0.4' figure only approximately describes the largest-d_lat end, and the small-d_lat end is well into the nonlinear regime -- which is exactly where the effective Hermite coefficient collapses to half its nominal value.

**Evidence:** Preactivation variance q = tr(M_t)/d_lat measured from generate_data(seed 42, sigma_perp=0.5): q = 2.752, 1.507, 0.889, 0.578, 0.474, 0.422, 0.390, 0.349, 0.329 for d_lat = 5,10,20,40,60,80,100,150,200 -> sd = sqrt(q) = 1.659, 1.228, 0.943, 0.760, 0.688, 0.650, 0.625, 0.591, 0.574. The appendix's own pitfall threshold is 'data scale / sqrt(d_lat) exceeds approximately 2'; at d_lat=5 the sd is 1.66, i.e. 83% of that threshold, and c1(q)^2 = 0.181 is already 51% below mu_1^2 = 0.367. Corroborating tr(U)/p = E[tanh^2(sqrt(q) z)] measured vs predicted: 0.5453 vs 0.5748, 0.4560 vs 0.4689, 0.3633 vs 0.3728, 0.2915 vs 0.2975, 0.2327 vs 0.2347, 0.2092 vs 0.2098 -- agreement to 1-5%, confirming q is the controlling parameter.

**Fix:** Replace the '~0.4' claim with the measured q(d_lat) table and state the range. Then either restrict the sweep to a q-range where c1(q)^2 is approximately constant, or (better) carry c1(q)^2 through the theory as in finding mu1-is-q-dependent.


**Referee:** Verified, though it is largely a restatement of the mu_1-vs-c1(q) finding.

eigenvalue_saturation/README.md:15 reads verbatim: 'We use scale=3, sigma_signal=1, sigma_noise=0.5, which keeps tanh inputs at ~0.4 on average -- well within the linear regime'. The preactivation standard deviation is sqrt(q) = sqrt(tr(M_t)/d_lat) = 1.659, 1.228, 0.943, 0.760, 0.688, 0.650, 0.625, 0.591, 0.574 for d_lat = 5..200. '~0.4' describes at best the large-d_lat end (mean |preactivation| = sqrt(2q/pi) = 0.46 at d_lat = 200) and is 4.1x low at d_lat = 5.

The consequence is real: at d_lat = 5 the effective linear coefficient c1(q)^2 = 0.180 is 51% below mu_1^2 = 0.367. And the appendix's own pitfall criterion (sec-appendix.tex:193-198, 'data scale divided by sqrt(dlat) exceeds approximately 2') is the wrong statistic — it gives 3/sqrt(5) = 1.34 at d_lat = 5 and so passes, while the actual controlling quantity sqrt(q) = 1.66 is 83% of that threshold because the criterion omits the sigma_signal and cluster contributions.

I independently confirm the corroborating check: tr(U_code)/p vs E[tanh^2(sqrt q z)] = 0.5453/0.5756, 0.4560/0.4698, 0.3633/0.3729, 0.2915/0.2973, 0.2327/0.2346, 0.2092/0.2092 — agreement to 0-5%.


**Referee correction:** Severity is minor, not major, and it should be merged into mu1-is-q-dependent rather than filed separately: the '~0.4' line is in a data-folder README, not the submission, and no claim in the draft depends on that number directly. The substantive item to carry into the paper is the replacement of the appendix's scale/sqrt(dlat) < 2 pitfall criterion by an explicit q_t = tr(M_t)/d_lat table with its measured range.


---

## [MINOR] The noise-floor estimator beta_d jumps discontinuously across the sweep, corrupting the derived spectral features

`beta-floor-estimator-discontinuous` · verdict **CONFIRMED** · kind error · effort hours  

**Where:** compute_celeba_spectral_predictor_data.py (`bottom_k = ceil(0.30*d); beta = median(eig[:bottom_k])`); columns beta_floor / effective_rank_excess / buffer_proxy plotted in fig:app-spectral-features

**Claim:** beta_d is the median of the smallest 30% of eigenvalues, so it flips character the moment the collapsed-coordinate fraction crosses 30%. Between d=140 and d=160 it drops from 0.7210 to 0.1894, which makes effective_rank_excess jump upward from 60.2 to 89.5 and makes buffer_proxy = d - round(r_eff) NON-monotone (80 at d=140, 71 at d=160) -- the opposite direction from the paper's narrative, and an artifact of the estimator rather than of the data.

**Evidence:** celeba_spectral_features_primary_t.csv, d=120/140/160/180/200: beta_floor = 0.8274 / 0.7210 / 0.1894 / 0.1815 / 0.1814; effective_rank_excess = 52.4 / 60.2 / 89.5 / 90.3 / 87.9; buffer_proxy = 68 / 80 / 71 / 90 / 112. Note effective rank also becomes non-monotone (90.3 at d=180, 87.9 at d=200).

**Fix:** Define the floor structurally as beta_d = Delta_t = 1 - e^{-2t}, which is the exact lower bound of spec(M_t) and requires no estimation, and recompute effective_rank_excess and buffer_proxy against it.


**Referee:** Every number verified against celeba_spectral_features_primary_t.csv. beta_floor at d = 120/140/160/180/200 = 0.82736 / 0.72099 / 0.18945 / 0.18151 / 0.18136; effective_rank_excess = 52.41 / 60.22 / 89.48 / 90.34 / 87.88; buffer_proxy = 68 / 80 / 71 / 90 / 112. The discontinuity is exactly the mechanism described: `bottom_k = max(1, ceil(0.30*d)); beta = median(eig[:bottom_k])` with eigvalsh returning ascending order, so beta flips character once the collapsed-coordinate fraction crosses 15% of d (the median of the bottom 30%). buffer_proxy is non-monotone in the direction opposite to the narrative (80 at d=140, 71 at d=160), and effective_rank_excess is non-monotone at the top (90.34 at d=180, 87.88 at d=200). The proposed fix (beta_d = Delta_t exactly, the analytic lower bound of spec(M_t)) is correct and free.


**Referee correction:** Scope note: these corrupted features feed the diagnostic figure fig:app-spectral-features and the excess-weighted clock variant, but not the primary predictor, which uses w_i = 1 and never reads beta_floor. 'Corrupting the derived spectral features' is accurate; it does not touch the headline fit. Minor severity is right.


---

## [MINOR] The clock's diffusion time t=0.1 is a hidden hyperparameter selected from three computed values with no reported sensitivity

`diffusion-time-t-unjustified` · verdict **CONFIRMED** · kind gap · effort hours  

**Where:** ICLR_2026/sec-spectral-predictor.tex:17; ICLR_2026/sec-appendix-integrated.tex:770-772; compute_celeba_spectral_predictor_data.py `--times 0.01,0.1,1.0 --primary_t 0.1`

**Claim:** Spectra are computed at t in {0.01, 0.1, 1.0} and only t = 0.1 is reported. t sets Delta_t, which is the floor of the entire spectrum and therefore controls the clock's whole dynamic range (see the lambda_max/Delta_t ceiling above): at t=0.01, Delta_t = 0.0198 and the ceiling rises to ~330x; at t=1.0, Delta_t = 0.865 and the ceiling collapses to ~7x. So the single most consequential structural property of the predictor is set by an unreported choice. The synthetic RFNN theory is stated at t=0.01, and the real diffusion models are trained over all t, so 0.1 matches neither.

**Evidence:** compute_celeba_spectral_predictor_data.py argparse defaults: `--times` = '0.01,0.1,1.0', `--primary_t` = 0.1; `load_spectra` filters `features[np.isclose(features['t'], primary_t)]`. sec-rfnn.tex: 'Training is score matching at a single fixed diffusion time t = 0.01'. Neither the main text nor the appendix reports results at t = 0.01 or 1.0.

**Fix:** Report the fit quality and the predicted tau-vs-d ordering at all three t, and say how t was chosen. If the ordering is stable across t, that is a genuine robustness result worth stating; if it is not, t is a fourth fitted parameter.


**Referee:** Verified. compute_celeba_spectral_predictor_data.py defaults are `--times 0.01,0.1,1.0` and `--primary_t 0.1`, and `load_spectra` keeps only the primary_t rows, so two thirds of the computed spectra are discarded before the fit. Neither sec-spectral-predictor.tex nor sec-appendix-integrated.tex reports any result at t = 0.01 or t = 1.0, gives a reason for 0.1, or acknowledges that other t were computed. The consequence the finding identifies is real: t sets Delta_t, which is the exact floor of spec(M_t) and hence sets the clock's entire dynamic range, and 0.1 matches neither the RFNN theory's t = 0.01 (sec-rfnn.tex) nor the all-t training of the real models.


**Referee correction:** The t = 1.0 ceiling is wrong. Recomputing from the same d=200 CelebA Gram (lambda_max = 6.6439 at t = 0.1 implies Gram lambda_max = 7.893): t = 0.01 gives Delta = 0.0198, lambda_max = 7.757, ceiling 392x (not ~330x); t = 1.0 gives Delta = 0.8647, lambda_max = 1.933, ceiling 2.2x (not ~7x). The argument is unaffected and in fact slightly stronger — the dynamic range varies by more than two orders of magnitude across the three computed t, so the unreported choice is even more consequential than stated.


---

## [MINOR] Remark on MP spreading: the quoted Delta_MP = 0.50 at psi_p=64 is the large-psi asymptotic, not the value (0.653), and the resolvability criterion compares the wrong pair of quantities

`mp-remark-arithmetic-and-criterion` · verdict **CONFIRMED** · kind error · effort hours  

**Where:** sec-appendix-fourbulk.tex Remark \ref{rem:MP} (lines 190-205)

**Claim:** Two issues. (i) Arithmetic: (sqrt(64)+1)^2/(sqrt(64)-1)^2 - 1 = 81/49 - 1 = 0.653, not 0.50; 0.50 is the asymptotic 4/sqrt(psi_p), which the remark itself only claims to hold 'at large psi_p'. The derived threshold should then read alpha_t^2/beta_t^2 >~ 1.65 rather than 1.5. (ii) More substantively, the remark budgets only the W-side MP width and ignores the noise-dim bulk's own internal spread from hat S, which is far larger at sigma_perp = 0.5 and is what actually threatens resolvability.

**Evidence:** Remark text: 'Delta_MP = (sqrt(psi_p)+1)^2/(sqrt(psi_p)-1)^2 - 1 approx 4/sqrt(psi_p) at large psi_p. At our default psi_p = 64 this gives Delta_MP approx 0.50, so the two data-side bulks remain spectrally resolvable whenever alpha_t^2/beta_t^2 >~ 1.5'. Exact value 0.6531. And the hat S contribution dominates: at d_lat=200, sigma_perp=0.5 the noise-dim block of M_t already spans 12x internally (finding Mt-empirical-not-population) against a W-side width of 0.65, so the correct criterion is lambda_min(signal block of M_t) > e^{-2t}sigma_perp^2(1+sqrt(d_lat/n))^2 + Delta_t (1.888 vs 0.666 at that point, margin 2.8x and shrinking with d_lat), not alpha_t^2/beta_t^2 > 1 + Delta_MP.

**Fix:** Correct the numeric value and state the resolvability criterion in terms of the two adjacent band *edges* (signal-block minimum vs noise-dim band maximum), including the hat S MP width, rather than the ratio of block centres.


**Referee:** (i) Arithmetic verified: sec-appendix-fourbulk.tex:195-197 writes the exact formula Delta_MP = (sqrt(psi_p)+1)^2/(sqrt(psi_p)-1)^2 - 1 and its large-psi_p asymptotic 4/sqrt(psi_p), then quotes 'At our default psi_p = 64 this gives Delta_MP approx 0.50'. The exact value is 81/49 - 1 = 32/49 = 0.6531; 0.50 is 4/sqrt(64), i.e. the asymptotic the remark itself only claims at large psi_p. The derived threshold should read alpha_t^2/beta_t^2 >~ 1.65, not 1.5. (ii) The substantive half is also right and follows from the Mt-empirical-not-population finding: the remark budgets only the W-side MP width, but at sigma_perp=0.5, d_lat=200 the noise-dim block of M_t already spans 12.1x internally from hat S ([0.0544, 0.6575]) against a W-side relative width of 0.65, so the binding comparison is lambda_min(signal block of M_t) = 1.878 vs the noise-dim upper edge e^{-2t} sigma_perp^2(1+sqrt(d_lat/n))^2 + Delta_t = 0.673 — margin 2.86x and shrinking with d_lat — not the ratio of block centres alpha_t^2/beta_t^2 = 10.4 > 1 + Delta_MP.


**Referee correction:** None. Both halves check out; the arithmetic slip is minor in isolation but the criterion half is the same defect as Mt-empirical-not-population and should be fixed together with it.


---

## [MINOR] 'MP fluctuations of relative size O(1/sqrt(psi_p))' is order-correct for W^T W/p but mischaracterises the structure for anisotropic M_t, and is attributed to the wrong reference

`fluctuation-claim-mischaracterised` · verdict **CONFIRMED** · kind gap · effort hours  

**Where:** sec-appendix-fourbulk.tex Lemma \ref{lem:Ulin} statement and proof, 'operator-norm fluctuations O(1/sqrt(psi_p)) \citep{pennington2017nonlinear}'

**Claim:** The rate is right in order (the MP band of W^T W/p at aspect d_lat/p = 1/psi_p is [(1-1/sqrt(psi_p))^2, (1+1/sqrt(psi_p))^2], relative width 4/sqrt(psi_p)), but two things are wrong around it. (i) The spectrum of M_t^{1/2} W^T W M_t^{1/2}/p is not 'M_t rescaled, up to fluctuations': it is the free multiplicative convolution of the ESD of M_t with MP(1/psi_p), which for a two-block M_t with a large eigenvalue ratio deforms the blocks differently and can merge them — the perturbation is not a uniform multiplicative smear. (ii) The concentration of W^T W is a Bai-Yin / Vershynin fact; pennington2017nonlinear is a nonlinear-random-matrix paper and is not the right citation for it.

**Evidence:** Proof line: 'W has i.i.d. Gaussian entries, so W^T W/d_lat is full rank a.s. for p >= d_lat, with E[W^T W/d_lat] = psi_p I and operator-norm fluctuations O(1/sqrt(psi_p)) \citep{pennington2017nonlinear}. ... which are the eigenvalues of M_t rescaled by psi_p up to those fluctuations.' The deformed-MP structure is visible in the data: the noise-dim band at d_lat=40, sigma_perp=0.5 spans [4.90, 13.80] in code units (2.8x), which is the hat S spread convolved with the W-side MP, not a +/-0.65 relative smear about a single point.

**Fix:** Cite Bai-Yin (or Vershynin's non-asymptotic bound) for the W^T W edge, and state the result as a deformed-MP / free-multiplicative-convolution statement rather than 'rescaled up to fluctuations'. Since the corrected identity lambda_i(U^lin) = mu_1_eff^2 lambda_i(M_t^emp)/d_lat already matches the data to a few percent per mode across the whole sweep, the cleanest fix is to state Lemma 1 in exactly that form and treat the MP deformation as the (small) residual.


**Referee:** Both sub-claims verified, though the finding is somewhat overstated in its worst-case framing. (i) The citation is wrong: sec-appendix-fourbulk.tex:124-127 attributes the concentration of W^T W/d_lat around psi_p I with operator-norm fluctuations O(1/sqrt(psi_p)) to \citep{pennington2017nonlinear}, and ICLR_2026/references.bib:327-336 confirms that entry is Pennington & Worah, 'Nonlinear random matrix theory for deep learning' (NeurIPS 2017) — a nonlinear-RMT paper, not the source for the Gaussian sample-covariance edge. Bai-Yin or Vershynin is the correct reference. (ii) The characterization is also loose: spec(M_t^{1/2} W^T W M_t^{1/2}/p) is the free multiplicative convolution of the ESD of M_t with MP(1/psi_p), which deforms different parts of an anisotropic M_t differently; 'the eigenvalues of M_t rescaled by psi_p up to those fluctuations' (line 128-129) is not that statement.


**Referee correction:** Narrow the finding. The reviewer's escalation that the convolution 'can merge them' is not realized anywhere in this parameter range: at psi_p = 64 the W-side deformation is ~0.65 relative, while alpha_t^2/beta_t^2 >= 4, and the observed noise-dim band spread ([4.90, 13.80] code units at d_lat=40) is inherited from hat S, not from the W-side MP — that spread belongs to the Mt-empirical-not-population finding, not this one. So the practical impact here is nil once Lemma 1 is restated in the form lambda_i(U^lin) = mu_1_eff^2 lambda_i(M_t^emp)/d_lat, which I confirmed matches the data to a few percent per mode across the entire sweep. What remains is (a) a genuinely wrong citation and (b) imprecise wording — real but minor, and correctly labelled 'gap/minor'.


---

## [MINOR] The four-bulk theorem's standing assumption psi_p > 1 + psi_n fails for the smallest d_lat panels that the paper shows as four-bulk examples

`theorem-hypothesis-violated-in-shown-sweeps` · verdict **CONFIRMED** · kind error · effort hours  

**Where:** sec-appendix-fourbulk.tex (setup paragraph, "$\psi_p > 1 + \psi_n$ (so the rank-null tail has room to exist)"); panels at ICLR_2026/sec-appendix-integrated.tex:156 ($\dlat\in\{5,10,20,...\}$, $p=64\dlat$, $n=500$)

**Claim:** With p = 64 d_lat and n = 500, psi_p = 64 and psi_n = 500/d_lat, so psi_p > 1 + psi_n requires d_lat > 7.94. The d_lat = 5 panel (and the d_lat = 5 row of the bulk-size table) is outside the theorem's hypothesis: p = 320 < n = 500, so the predicted rank-null count p - d_lat - n = -185 is negative and the sample bulk cannot even be accommodated. The paper presents that panel as an instance of the four-bulk structure.

**Evidence:** sec-appendix-fourbulk.tex: "We work in the proportional limit ... and $\psi_p > 1 + \psi_n$ (so the rank-null tail has room to exist)." ICLR_2026/sec-appendix-integrated.tex:156: "the swept variable is $\dlat\in\{5,10,20,40,100,200,500,1000\}$" at $p=64\dlat$, $n=500$. The bulk-size table nonetheless reports four bulks at d_lat=5 with B1=150, B2=138, B3=27, B4=5 summing to 320.

**Fix:** Mark the d_lat < 8 panels as outside the theory's regime, or use the p = d_lat + n + 300 rule (which satisfies the hypothesis for all d_lat) for the panels used as evidence for the theorem.


**Referee:** Arithmetic and panel presence both verified. sec-appendix-fourbulk.tex setup: 'psi_p > 1 + psi_n (so the rank-null tail has room to exist)'. With p = 64*d_lat and n = 500, psi_p = 64 and psi_n = 500/d_lat, so the hypothesis requires 63 > 500/d_lat, i.e. d_lat > 7.94. At d_lat = 5 the raw spectrum file sigma_noise_0.5/exp2_rfnn/raw_data/di5_d5_n500_s42/eigenvalues_pre.npy has length 320 < n = 500, so the predicted rank-null count p - d_lat - n = -185 and the rank-n sample bulk cannot be accommodated. The d_lat = 5, p = 64*d_lat panel IS in the compile set: sec-appendix-integrated.tex:156 gives the sweep as dlat in {5,10,20,40,100,200,500,1000} for Figure fig:app-rfnn-exp2-widths, and that caption asserts 'the green block grows in the exact increasing-buffer regime dlat-dint' even though the green block is empty at d_lat = d_int = 5. The bulk-size table's d_lat=5 row likewise reports four bulks summing to 320.


**Referee correction:** Add the mitigation the finding already half-notes: the main-text figure (fig:fourbulk-clean) uses p = d_lat + n + 300, for which psi_p - psi_n = (d_lat+300)/d_lat > 1 for every d_lat, so the hypothesis holds throughout that sweep. The violation is confined to the p=64*d_lat appendix panels, and it only becomes a stated-hypothesis violation if the theorem is wired in. Note also that d_lat=5 is independently problematic for a different reason -- see the tanh-saturation point under edge-scaling.


---

## [MINOR] The theorem's edges have no d_lat dependence at fixed psi_p, but the reported behavior is that two peaks move with d_lat and one does not

`edge-scaling-inconsistent-with-reported-shifts` · verdict **CONFIRMED** · kind inconsistency · effort hours  

**Where:** Theorem thm:fourbulk table in sec-appendix-fourbulk.tex vs final_report.tex:744-752 figure caption; normalization at code/v3/lib/eigenvalues.py:24-43

**Claim:** At p = 64 d_lat (psi_p = 64 fixed), all three nonzero edges mu_1^2 alpha_t^2/psi_p, mu_1^2 beta_t^2/psi_p and etabar_star/psi_p are d_lat-independent (alpha_t^2, beta_t^2 depend only on t, s, d_int, sigmas; ||x||^2/d_lat -> beta_t^2 so etabar_star is also d_lat-independent). The reported behavior is that two of them move and one does not. Separately, compute_U in the code omits the 1/sqrt(p) in phi that Eq. (rfnn) and Eq. (U) define, so the plotted eigenvalues are p = 64*d_lat times the quantity the theorem is about -- a d_lat-dependent rescaling that would by itself slide the whole histogram right as d_lat grows.

**Evidence:** final_report.tex:746-748: "Signal and noise-dim peaks shift right with $\dlat$; sample mode stays fixed." code/v3/lib/eigenvalues.py:38-40: "phi = torch.tanh(x_t @ W_cpu.T); U += phi.T @ phi / n" -- no 1/p factor, while sec-rfnn.tex Eq. (rfnn)/(U) define $\phi(x_t) := \tfrac{1}{\sqrt{\pwidth}}\tanh(\cdot)$ and sec-rfnn.tex:38 asserts "The reported spectra are the eigenvalues of this empirical estimate".

**Fix:** Fix the normalization in compute_U (or state the convention in the paper), then overlay the theorem's predicted edges on the histograms. Right now the theorem's only quantitative content -- the edge magnitudes -- is never compared to a number anywhere in the project.


**Referee:** Both halves verified, and the normalization half is worse than reported. (a) The 1/sqrt(p) is missing from EVERY U computation in the repo, including the current figure pipeline: code/v3/lib/eigenvalues.py:38-42 'phi = torch.tanh(x_t @ W_cpu.T); U += phi.T @ phi / n', clean_figure_suite.py:133-138 and clean_four_bulk_sweep.py:131-136 do the same, while sec-rfnn.tex Eq. (rfnn)/(U) define phi(x_t) := (1/sqrt(p)) tanh(Wx/sqrt(d_lat)) and sec-rfnn.tex:38 asserts 'The reported spectra are the eigenvalues of this empirical estimate.' (The 1/sqrt(d_lat) IS present, folded into W via models.py:61 randn(p,d)/sqrt(d).) (b) The edges are indeed d_lat-independent at fixed psi_p, and the reported motion is real: from the saved spectra at p=64*d_lat, sigma_perp=0.5, the median of the top-5 eigenvalues is 29.7, 40.6, 56.6, 72.8, 77.8, 86.9, 87.6, 96.7, 99.8 for d_lat = 5,10,20,40,60,80,100,150,200. (c) Confirmed the finding's closing claim: no numeric edge prediction is compared to any measurement anywhere in the corpus.


**Referee correction:** I can supply the missing diagnosis, which strengthens the finding considerably. The observed rightward drift of the signal and noise-dim peaks is a tanh-saturation artifact, not a spectral effect: ||x_t||^2/d_lat = [s^2 + d_int*sigsig^2 + (d_lat-d_int)*sigperp^2]*e^{-2t}/d_lat + Delta_t is ~10 at d_lat=5 (pre-activation std 3.16) and falls to ~0.49 at d_lat=200 (std 0.70). The paper's own appendix (sec-appendix.tex 'Known pitfalls' and app:saturation) says tanh saturates once the data scale over sqrt(d_lat) exceeds ~2 and that this 'can fragment [the structure] into extra internal sub-bulks' -- so the d_lat=5 and d_lat=10 panels sit inside the regime the paper tells the reader to avoid, and the peaks 'shift right' mainly because saturation is being relieved as d_lat grows. Separately, the theorem's edge normalization looks wrong on its own terms: Eq. (U-lin) gives edges mu_1^2*alpha_t^2/d_lat, not /psi_p. Fix the phi normalization, restate the edges, then overlay them on the histograms and restrict the panels to the unsaturated regime.


---

## [MINOR] The paper states 500 Monte Carlo noise samples for U; every U computation in the repo uses 50

`protocol-vs-code-mc-samples` · verdict **REFUTED** · kind inconsistency · effort hours  

**Where:** ICLR_2026/sec-rfnn.tex:38 and sec-appendix-integrated.tex:32 vs code/v3/run_experiment.py:156, code/v3/lib/eigenvalues.py:27, code/experiment_v2_rfnn.py:58

**Claim:** Both places in the current draft say 500 MC diffusion-noise samples. Every call site in the repository passes or defaults to 50. The archived table in ICLR_2026/sec-appendix.tex:158 also says "$50$ noise samples per training point", so the 500 appears to be a number that changed in the text without a corresponding run, or a run whose code is not in the repo.

**Evidence:** sec-rfnn.tex:38: "(500~Monte Carlo noise samples; Appendix~\ref{app:methods})". sec-appendix-integrated.tex:32: "500 Monte Carlo diffusion-noise samples used to estimate $U$". code/v3/run_experiment.py:156: "t=model.t_fixed, n_noise_samples=50". ICLR_2026/sec-appendix.tex:158: "$\Umat$ Monte Carlo & n/a& $50$ noise samples per training point".

**Fix:** Reconcile: either commit the code for the 500-sample rerun or correct the two statements to 50. A reviewer checking reproducibility will find this in one grep.


**Referee:** The grep missed the scripts that actually generate the current draft's figures. clean_figure_suite.py -- whose module docstring line 12 reads 'All runs use tanh RFNN features and MC=500 for U' and whose RFNNConfig sets mc_samples: int = 500 (line 53), used in both compute paths at lines 133 and 160 -- produces figures/clean/{p_n_plus_d_plus_r, p64, p_fixed_energy_fixed}, i.e. the three width controls the current sec-rfnn.tex describes. clean_four_bulk_sweep.py likewise sets mc_samples: int = 500 (line 44) and passes mc_samples=500 in every base config (lines 286, 330, 342, 351). The n_noise_samples=50 default lives in the older code/v3 and code/experiment_v2_rfnn.py pipeline, and the ONLY place the paper reports 50 is the archived ICLR_2026/sec-appendix.tex:158, which is not in the compile set and correctly documents that older pipeline. So both current statements (sec-rfnn.tex:38 and sec-appendix-integrated.tex:32) match the code that produced the current figures.


**Referee correction:** No reconciliation is needed in the text. If anything is worth flagging it is the reverse: the archived sec-appendix.tex still documents 50 for a run that the current draft has superseded, and the older 50-sample code/v3 pipeline is the one that produced the orphaned tab:bulk-sizes numbers -- so if that table is ever revived it must be regenerated at MC=500.


---

## [MINOR] PROJECT_BRIEF.md's "d_lat/d_int >= 3" safety rule has no derivation or fit anywhere, and lands on the worst point of the score-error curve

`brief-rule-of-thumb-unsupported` · verdict **CONFIRMED** · kind unsupported-claim · effort hours  

**Where:** PROJECT_BRIEF.md "What a practitioner should do (the recipe)"; data at sigma_noise_0.01/exp2_mlp/raw_data

**Claim:** The brief gives numeric operating thresholds ("$d_{lat}/d_{int} \gtrsim 3$ -> memorization is pushed way off; $\lesssim 1.5$ -> buffer too small to help") that appear in no section, appendix, theorem, or fit in the project. In the sigma_perp=0.01 synthetic sweep the ratio 3 (d_lat=15, d_int=5) is exactly where per-dimension score error PEAKS at 5.325, 36x its d_lat=5 value.

**Evidence:** PROJECT_BRIEF.md: "**Rule of thumb:** `d_lat / d_int >~ 3` -> memorization is pushed way off; `<~ 1.5` -> buffer too small to help." Final-step score_error from sigma_noise_0.01/exp2_mlp/raw_data: d=5: 0.1473, d=8: 3.3662, d=10: 4.9507, d=15: 5.3250, d=20: 3.4228, d=200: 0.5927.

**Fix:** Delete the thresholds or derive them. If a recommendation is wanted, it has to trade off the memorization curve against the score-error/FID curve, and the resulting optimum in the repo's own data is at large d_lat, not at ratio 3.


**Referee:** Verified that the thresholds appear nowhere else. PROJECT_BRIEF.md: '**Rule of thumb:** d_lat / d_int >~ 3 -> memorization is pushed way off; <~ 1.5 -> buffer too small to help.' grep for 'rule of thumb', 'd_lat / d_int' and 'd_{lat}/d_{int}' across all ICLR_2026/*.tex and METHODS.md returns nothing -- no section, appendix, theorem, figure, or fit anywhere derives or calibrates a ratio threshold. I reproduced the score-error numbers from sigma_noise_0.01/exp2_mlp/raw_data: final per-dimension score error 0.1473 (d=5), 3.3662 (d=8), 4.9507 (d=10), 5.3250 (d=15), 3.4228 (d=20), 0.5927 (d=200), peaking at exactly d_lat/d_int = 3.


**Referee correction:** Two qualifications on the score-error argument. Those runs are the archived 300k-step, single-seed, hidden=8*d_lat protocol, not the paper's fixed-width 5M five-seed protocol, so the peak location should not be presented as the paper's result; and per-dimension score error is a score-fitting metric, not a memorization-risk metric, so a peak there does not by itself make ratio 3 a bad operating point. The solid, sufficient finding is that the brief states two numeric operating thresholds with no derivation, fit, or figure anywhere in the project.


---

## [MINOR] "Memorization falls almost monotonically to near zero" is contradicted by the extended-range data in the repo and by the project's own final_report text

`monotone-decline-overstated` · verdict **REFUTED** · kind inconsistency · effort hours  

**Where:** ICLR_2026/sec-mlp.tex Figure fig:synthetic-mlp-main caption and sec-appendix-integrated.tex:312-329; vs sigma_noise_0.5/exp2_mlp/raw_data and final_report.tex:783-789

**Claim:** The caption says memorization "falls almost monotonically from roughly $30\%$ at $\dlat=5$ to near zero by $\dlat=40$" and the appendix says the decline "continues through $\dlat=240$". In the archived sigma_perp=0.5 sweep memorization returns above the d_lat=5 baseline at large width: 0.0062 at d=5, 0.0000 for 10<=d<=100, 0.0022 at d=150, 0.0144 at d=200. final_report.tex says the same thing in words.

**Evidence:** final_report.tex:786-789: "The complementary $\dlat$ sweep at fixed $\dint = 5$ widens the buffer from 0 to 195; in 300k steps, memorization fraction is non-zero only at $\dlat \in \{5, 150, 200\}$ and essentially zero across the interior $8 \le \dlat \le 100$." Final memorization_fraction from sigma_noise_0.5/exp2_mlp/raw_data/di5_d200_n500_s42/metrics.jsonl = 0.0144 > 0.0062 at d=5.

**Fix:** State the range over which the decline holds and report the large-d_lat return if the 5M-step rerun reproduces it. Note the two runs differ (8*d_lat hidden width, 300k steps, one seed vs. fixed 256, 5M steps, five seeds), which is itself worth disclosing rather than leaving two incompatible sweeps in the repo.


**Referee:** The caption is exactly right for the run it describes, and the counterexample is a different experiment. sec-mlp.tex:8 specifies the main-text sweep as dint=5, n=500, sigma_perp=0.5, dlat in {5,8,10,12,15,20,25,30,35,40}, seeds 42-46, 5M steps, hidden width fixed at 256. That run is multiseed_runs/exp2_mlp_dlat_sn05_5m_bonnaire_hidden256 (config.json confirms hidden=256, total_steps=5000000), and its five-seed mean final memorization fraction is 0.3001, 0.2172, 0.1709, 0.1265, 0.0800, 0.0365, 0.0155, 0.0076, 0.0030, 0.0012 for d_lat = 5..40 -- strictly monotone decreasing, ~30% at d_lat=5, ~0.1% by d_lat=40. That is precisely 'falls almost monotonically from roughly 30% at dlat=5 to near zero by dlat=40'. The cited counterexample, sigma_noise_0.5/exp2_mlp/raw_data, has config.json hidden=1600=8*d_lat, total_steps=300000, single seed 42, and peak memorization ~0.6% -- a 50x different regime under a capacity-scaled protocol the current draft does not use. Comparing them is not a contradiction in the paper.


**Referee correction:** There is a real disclosure issue nearby, and it is a better finding. The repo also holds multiseed_runs/exp2_mlp_dlat_sn05_5m_combined -- same 5M horizon and same five seeds, but hidden width scaled with d_lat (hidden=320 at d_lat=40) -- and there final memorization is NON-monotone: 0.0119 (d=5), 0.0037, 0.0037, 0.0028, 0.0020 (d=15, the minimum), 0.0024, 0.0022, 0.0030, 0.0049, 0.0083 (d=40), i.e. it turns around and roughly quadruples from d_lat=15 to d_lat=40. That is the same-horizon, same-seed-count capacity-scaled arm, and it shows the monotone decline is specific to the fixed-width control -- which is exactly the confound sec-discussion.tex limitation (i) flags. Separately, I could find no local data for the extended sweep, so the appendix claim that the decline 'continues through dlat=240' is unverifiable from this repo.


---

## [MINOR] The appendix locates the noise-dim/sample bulk merge at "small sigma_perp at large t", but at large t the gap grows; the observed merge is at small t

`merge-regime-stated-backwards` · verdict **CONFIRMED** · kind error · effort hours  

**Where:** /Users/ryan/Desktop/latent_space_diffusion_analysis/sec-appendix-fourbulk.tex, "Bulk-gap scaling" paragraph

**Claim:** The claimed merge condition is wrong in the t direction. The gap ratio is mu_1^2 beta_t^2 / etabar_star with beta_t^2 = e^{-2t} sigma_perp^2 + Delta_t. As t grows, beta_t^2 -> 1 and ||x||^2/d_lat -> 1 so etabar_star -> sum_{k>=3} mu_k^2 ~ 0.011; the ratio grows to ~33, i.e. the bulks separate MORE, not less. The regime where the ratio actually collapses is small t with small sigma_perp: at t=0.01, sigma_perp=0.01, d_lat=20 the ratio is roughly 0.367*0.0199 / (0.0098*0.72^3) ~ 2, which is exactly the configuration whose "smooth step" the paragraph cites as evidence.

**Evidence:** sec-appendix-fourbulk.tex: "Below that threshold (small $\signoise$ at large $t$) the two merge, which is the regime where the four-bulk cliff at index $\dlat$ becomes a smooth step in Fig.~\ref{fig:appendix-fourbulk-cliffs}." The cited figure is the sigma_perp=0.01, t=0.01 configuration -- small t, not large t.

**Fix:** Recompute the merge threshold as a condition on (sigma_perp, t) and state it correctly; then check it against the sigma_perp=0.5 vs 0.01 panels, which is a cheap and genuinely informative quantitative test of the derivation.


**Referee:** The t-direction is stated backwards, verified analytically and against the data. sec-appendix-fourbulk.tex 'Bulk-gap scaling' says the noise-dim and sample bulks merge at 'small signoise at large t', citing the sigma_perp=0.01 figure. But beta_t^2 = e^{-2t}sigma_perp^2 + Delta_t is INCREASING in t (from sigma_perp^2 toward 1), and etabar_star = sum_{k>=3} mu_k^2 E[(||x_t||^2/dlat)^k] is DECREASING in t whenever ||x||^2/d_lat > 1, so the ratio mu_1^2 beta_t^2/etabar_star grows monotonically with t: as t->infinity it tends to mu_1^2 / sum_{k>=3} mu_k^2 = 0.3676/0.0267 ~ 14 (or ~37 using the appendix's mu_3^2 ~ 0.0097 alone). The bulks separate more at large t, not less. The merge regime is small t with small sigma_perp, which is exactly the cited configuration (t=0.01, sigma_perp=0.01). I confirmed this empirically: the eigenvalue ratio across index d_lat is 1.08/1.01/1.14 at sigma_perp=0.01, t=0.01 (merged) versus 4.4-18 at sigma_perp=0.5, t=0.01 (well separated).


**Referee correction:** The finding's own numeric estimate is wrong in the same direction it is correcting. It uses ||x||^2/d_lat ~ 0.72, but at d_lat=20, sigma_perp=0.01, center scale s=3, sigsig=1, d_int=5 the diffused value is e^{-2t}[s^2 + d_int*sigsig^2 + (d_lat-d_int)*sigperp^2]/d_lat + Delta_t ~ 0.98*(45+5+0.0015)/20 + 0.0198 ~ 2.47. That gives etabar_star >= mu_3^2 * 2.47^3 ~ 0.148 and a ratio of ~0.05, not ~2 -- so the merge at (t=0.01, sigma_perp=0.01) is roughly 40x more severe than the finding computes, and the corrected threshold condition should be recomputed with the mixture-center contribution to ||x||^2 included.


---

## [MINOR] The real-data section has no related work on memorization in latent (as opposed to pixel) diffusion, and two 2025 papers supply a live confound for the pixel-space NN metric and an alternative explanation for the d_lat effect

`latent-diffusion-memorization-related-work` · verdict **—** · kind gap · effort hours  

**Where:** /Users/ryan/Desktop/latent_space_diffusion_analysis/ICLR_2026/sec-real-data.tex and sec-intro.tex:14; references.bib

**Claim:** sec-intro.tex:14 states "What remains unclear is how the transition depends on the dimension of the latent space itself, especially when diffusion is trained in VAE latents but memorization is measured after decoding back to pixel space" -- but cites nothing on latent-space memorization. Two relevant works: (1) "Latent Diffusion Inversion Requires Understanding the Latent Space" (arXiv:2511.20592) reports that latent diffusion exhibits dimension-wise non-uniform memorization and that local distortion in the VAE is the primary cause -- a direct confound for the paper's pixel-space-nearest-neighbour metric, and one it partly rediscovers as the "VAE identity threshold"; (2) "Optimal Stopping in Latent Diffusion Models" (arXiv:2510.08409) shows in a Gaussian/linear-autoencoder framework that "lower-dimensional representations benefit from earlier termination, whereas higher-dimensional latent spaces require later stopping time" -- an alternative, non-memorization-specific explanation for a d_lat-dependent optimal stopping time that the paper's design rule needs to rule out.

**Evidence:** sec-intro.tex:14 as quoted, with no citation on latent-space memorization; ICLR_2026/references.bib contains no entry on latent-diffusion memorization (only somepalli2023, carlini2023, gu2023 -- all pixel-space).

**Fix:** Add both to related work; use (1) to strengthen (not weaken) the VAE-identity-threshold argument by citing an independent finding that VAE distortion drives measured memorization; use (2) to state explicitly why the observed d_lat effect is memorization timing rather than a generic optimal-stopping shift.


---

## [MINOR] Five intrinsic-dimension estimators are cited in the intro but none is used; the real-data pipeline uses a 95%-variance VAE identity threshold instead

`intrinsic-dimension-refs-cited-but-unused` · verdict **—** · kind gap · effort days  

**Where:** /Users/ryan/Desktop/latent_space_diffusion_analysis/ICLR_2026/sec-intro.tex:8 (citations pope2021, brown2022manifold, stanczuk2024manifold, kamkari2024geometric) vs. sec-spectral-predictor.tex and sec-real-data.tex (which use $\widehat d_{\rm ID}^{\rm VAE}(0.95)$)

**Claim:** The intro invokes the intrinsic-dimension literature to motivate d_int, and references.bib carries levina2004 and facco2017twonn as well, but the real-data experiments never estimate d_int with any of these methods. Instead they use a 95%-variance threshold on the VAE latent spectrum, labelled $\widehat d_{\rm ID}^{\rm VAE}(0.95)$, which is a participation-ratio-style quantity, not an intrinsic-dimension estimate, and which is confounded with VAE reconstruction fidelity by construction. Since "the gap between d_lat and d_int" is the paper's independent variable, the absence of an actual d_int estimate on CelebA/CIFAR-10 leaves the real-data claim without its x-axis.

**Evidence:** sec-intro.tex:8: "a representation closer to the data's true intrinsic dimensionality $\dint$~\citep{pope2021, brown2022manifold, stanczuk2024manifold, kamkari2024geometric}". sec-spectral-predictor.tex: "below the empirical VAE identity threshold $\widehat d_{\rm ID}^{\rm VAE}(0.95)$". No use of TwoNN/MLE/Stanczuk/Kamkari estimators anywhere in the pipeline.

**Fix:** Run at least one of TwoNN (facco2017), Levina-Bickel MLE (levina2004), or the diffusion-based estimators (stanczuk2024, kamkari2024) on the CelebA and CIFAR-10 subsets, report d_int_hat, and plot the memorization curves against d_lat/d_int_hat rather than d_lat. If that is out of scope for a workshop paper, rename $\widehat d_{\rm ID}^{\rm VAE}$ so it does not read as an intrinsic-dimension estimate, and soften sec-intro.tex:8.


---

## [MINOR] The RFNN time axis carries a learning rate proportional to d_lat while the MLP uses a fixed Adam rate, so the two tau_mem(d_lat) curves are measured on different clocks

`rfnn-mlp-clocks-not-comparable` · verdict **—** · kind inconsistency · effort hours  

**Where:** /Users/ryan/Desktop/latent_space_diffusion_analysis/code/v3/run_experiment.py:538-544; ICLR_2026/sec-rfnn.tex ("Training dynamics") and sec-mlp.tex

**Claim:** run_experiment.py:543 sets the RFNN learning rate to `0.01 * args.d_latent / delta_t`, i.e. linear in d_lat -- this is Bonnaire's tau = k*eta/d^2 rescaling and it is the right choice, but it is nowhere stated in the paper, and tau_gen/tau_mem are extracted in raw optimizer steps (code/v3/plot.py: `extract_taus` returns `int(steps[...])`). The MLP branch (run_experiment.py:538-539) uses Adam at a fixed 1e-4 with no d_lat scaling. So the RFNN's step axis is a rescaled gradient-flow time and the MLP's is not, and the two "tau_mem increases with d_lat" results are not measured against a common clock. Bonnaire's own theory then predicts, in their rescaled tau, tau_mem ~ psi_n/Delta_t = n/(d_lat Delta_t), i.e. *decreasing* in d_lat at fixed n -- so the direction of the effect depends on which clock is used, and the paper never says which one it is on.

**Evidence:** code/v3/run_experiment.py:543: `lr = args.lr if args.lr is not None else 0.01 * args.d_latent / delta_t`; :538-539: `lr = args.lr if args.lr is not None else 1e-4; return torch.optim.Adam(...)`. Bonnaire Eq (9)-(10): "after rescaling time as tau = k*eta/d^2"; p.9: "tau_mem becomes large and of order psi_n/Delta_t".

**Fix:** State the learning-rate scaling explicitly in sec-rfnn.tex, say which time normalization the reported tau values are in, and report tau_mem in both raw steps and Bonnaire-rescaled tau so the comparison to their prediction tau_mem ~ psi_n/Delta_t is unambiguous. Either match the MLP normalization to the RFNN's or state plainly that the MLP result is a raw-step claim.


---

## [MINOR] rank(U^diag) = n is asserted for a matrix defined with an expectation over diffusion noise, and the claimed o(1) rank-null bulk is continuous with the sample bulk

`rank-n-and-null-tail` · verdict **CONFIRMED** · kind gap · effort hours  

**Where:** sec-appendix-fourbulk.tex, Lemma \ref{lem:Udiag} ('rank = nsamp almost surely', lines 137-138, 150-152) and Theorem \ref{thm:fourbulk} table row 'Rank-null ... o(1)' (line 224) plus its proof (lines 232-238)

**Claim:** U is defined as (1/n) sum_mu E_eta[phi(x_t^mu) phi(x_t^mu)^T], so the per-sample contribution is E_eta[phi^perp phi^perp^T], which is NOT a rank-one outer product; the rank-n claim needs a condition on t (it holds only while the residual features from different noise draws of the same x are nearly collinear). Separately, the fourth bulk is not separated: the bottom of the sample bulk and the top of the rank-null tail are numerically identical, so there is no cliff at index d_lat + n, and the tail is a smooth decay rather than a set of vanishing eigenvalues.

**Evidence:** Collinearity across noise draws: rho = e^{-2t}||x||^2/(e^{-2t}||x||^2 + Delta_t d_lat) = 0.94 at t=0.01, d_lat=200, sigma_perp=0.5, giving rho^3 = 0.83 (nearly rank one, so the lemma survives at t=0.01) but rho -> 0 as t grows, and the lemma states no t condition while the theorem claims validity for all fixed t in (0, infinity). Null tail, sigma_perp=0.5, d_lat=40: ev[539] = 3.55e-3 (last sample index) vs ev[540] = 3.54e-3 (first rank-null index) -- no gap; the tail then decays smoothly to 9.07e-5 at index 2559. Same at d_lat=200: ev[699] = 2.92e-3 vs ev[700] = 2.91e-3. Aggregate tail mass is only 0.25-0.36% of tr(U), so 'o(1) in aggregate' is defensible, but 'o(1) relative to the sample bulk edge' is not.

**Fix:** Add the t-dependence condition (or state the lemma for the single-noise-draw U and note the E_eta version separately), and restate the fourth bulk as 'carrying vanishing spectral mass' rather than 'eigenvalues o(1)', since its top eigenvalue coincides with the sample bulk's bottom.


**Referee:** Both parts verified. (i) Eq (U) at ICLR_2026/sec-rfnn.tex:22-25 and Eq (U-diag) at sec-appendix-fourbulk.tex:77-81 define the per-sample contribution as E_eta[phi(x_t^mu) phi(x_t^mu)^T], an expectation over diffusion noise, so it is NOT a rank-one outer product and 'rank = nsamp almost surely' does not follow from the stated argument; it holds only while residual features from different noise draws of the same x stay nearly collinear, which is a condition on t that the lemma never states and the theorem's 'fixed t in (0,infinity)' explicitly waives. I confirmed the collinearity parameter rho = e^{-2t}||x||^2/(e^{-2t}||x||^2 + Delta_t d_lat) = 0.94 at t=0.01, d_lat=200, sigma_perp=0.5 (so the lemma survives at the paper's operating point) and that it degrades with t. (ii) The fourth bulk is not separated. From the saved spectra at sigma_perp=0.5: d_lat=40 gives ev[539] = 3.546e-3 vs ev[540] = 3.541e-3 (ratio 1.0014); d_lat=200 gives ev[699] = 2.916e-3 vs ev[700] = 2.908e-3. There is no cliff at index d_lat + n anywhere. Aggregate tail mass is 0.006%-0.36% of tr(U), so 'vanishing spectral mass' is defensible but 'eigenvalues o(1) relative to the sample edge' is not.


**Referee correction:** Confirmed, and I would raise the severity from minor to major on the second half. Re-running compute_U at d_lat=40 across t shows the sample bulk and rank-null tail merge into a single smooth continuum as soon as t is moderate: at t=1.0, ev[40] = 0.0668, ev[539] = 0.0365, ev[540] = 0.0365, ev[1000] = 0.0254 - the 'sample bulk' top is only 2.6x its own bottom and the tail is continuous with it. So the theorem's claim of four separated bulks 'for fixed t in (0,infinity)' does not merely lack a t condition, it is false for moderate t, and the fourth bulk is never separated at any t I tested. The proposed rewording ('carrying vanishing spectral mass') is right but should be accompanied by an explicit small-t restriction on the whole theorem.


---

## [MINOR] The 'unexplained anti-learning at d_lat >= 100 for sigma_perp = 0.5' is an artifact of estimating eps_null by subtracting a d_lat-independent signal baseline

`sigma-perp-0.5-anomaly-explained-by-invalid-subtraction` · verdict **REFUTED** · kind error · effort hours  

**Where:** _next_steps/n_shape_heuristic_derivation.md lines 38, 126-147, 278 (Limitation 4)

**Claim:** eps_null/dim is defined as (d_lat x score_error - d_int x eps_sig^(d_lat=5))/(d_lat - d_int), which assumes the per-dim signal-block error is frozen at its d_lat = d_int value for all d_lat. That is never checked, and both `probe_ushape.py` and the main experiment could measure the two blocks directly. Since the signal-block error demonstrably grows with d_lat (the network's capacity is split across more outputs), the subtraction systematically inflates eps_null, and at large d_lat where d_lat - d_int is large the inflation is divided by a large number but the numerator error is amplified by d_lat. The resulting ratio crossing 1.0 at d >= 100 is therefore not evidence of a physical effect.

**Evidence:** Doc line 38 gives the formula; doc lines 137-139 report ratios 1.054, 1.158, 1.270 at d = 100, 150, 200 and line 145 says 'The current model has no quantitative account of this behavior.' Neither the doc nor the codebase reports a directly measured signal-block error at any d_lat > 5. With the correct target, the trivial-predictor baseline at sigma_perp=0.5 is (5*0.730 + (d-5)*2.591)/d = 1.43 at d=8 rising to 2.36 at d=40, i.e. strongly d_lat-dependent — so the flat-baseline subtraction is not even approximately valid.

**Fix:** Measure eps_sig and eps_null directly by projecting the residual onto Q[:, :d_int] and Q[:, d_int:], as probe_ushape.py intends to do (after fixing its basis bug). Delete Limitation 4 as an open puzzle; it is a measurement artifact.


**Referee:** The proposed cause is not the cause, and I can show it quantitatively. At the d_lat values where the anomaly appears the flat-baseline subtraction is nearly inert: eps_null/dim = (d x score_error - 5 x 0.1415)/(d-5) versus the raw score_error gives, at d=200, 3.2915 vs 3.2128 - a 2.5% inflation, which moves the ratio to Var(s*_null)=2.592 from 1.239 to 1.270. Even replacing the frozen signal baseline with the actual d_lat-dependent signal-block error would change eps_null by (S - 5b)/(d-5); with the measured signal-block discrepancy of 10.4 at d=200 that is 9.7/195 = 0.05, i.e. 1.5%. The anomaly is not a subtraction artifact. It also survives the rotation bug: the sigma_perp=0.5 mismatch ||s_ok-s_bug||^2/d falls to 0.149/0.103/0.077 at d=100/150/200, leaving residuals of 2.45/2.80/3.14 against a correct trivial baseline of ~2.55 - so the trained network genuinely does about as badly as, or worse than, predicting zero out there. (The subtraction IS materially wrong at small d_lat, where the doc reports eps_null = 1.943 at d=8 against a raw score_error of 0.817 - a 2.4x inflation - but that is the region the finding does not target.)


**Referee correction:** Corrected version: the eps_null formula's frozen-signal-baseline assumption is unverified and materially inflates eps_null at SMALL d_lat (2.4x at d=8, where the doc's 'anti-learning zone' lives), not at large d_lat. The sigma_perp=0.5 large-d_lat anomaly (Limitation 4) is not explained by the subtraction and is not eliminated by fixing the rotation bug; after both corrections the trained MLP's score error at d_lat >= 100 is still of order the zero-predictor baseline, so Limitation 4 should be kept as a genuine open item rather than deleted. The recommendation to measure eps_sig and eps_null by projecting the residual onto Q[:, :d_int] and Q[:, d_int:] is nonetheless correct and should be adopted.


---

## [MINOR] Curved manifolds and non-Gaussian data: state the theory in terms of M_t only, and give a curvature-leakage bound that says when the two data bulks stop being separate

`curved-manifold-nongaussian` · verdict **CONFIRMED** · kind missing-theory · effort days  

**Where:** ICLR_2026/sec-discussion.tex limitation (iii); _next_steps/NEXT_STEPS.md section 5 and Open Questions ("Does the four-bulk hold when the data manifold is non-linear?")

**Claim:** The paper's limitation section concedes the linear-Gaussian manifold and defers to CelebA. But the four-bulk argument never actually needs Gaussian data -- it needs (a) the second moment M_t of the diffused data, and (b) Gaussian equivalence for the feature map. Saying this explicitly generalizes the theorem for free and isolates exactly what curvature breaks: the block structure of M_t, not the RMT.

**Evidence:** sec-discussion.tex: "(iii)~The synthetic data manifold is a linear Gaussian mixture, so curved manifolds and class hierarchies are tested mainly through CelebA". sec-appendix-fourbulk.tex's derivation uses the data only through M_t = e^{-2t}(hat C + hat S) + Delta_t I, i.e. only through second moments -- the Gaussianity of xi is never used after Eq (Mt).

**Fix:** Two short propositions. (1) Distribution-free restatement: 'Proposition. For any data distribution with finite second moment, the conclusions of Theorem 1 hold with alpha_t^2, beta_t^2 replaced by the eigenvalues of M_t = e^{-2t} Cov(x) + Delta_t I, provided the Gaussian-equivalence conditions on W and the activation hold.' Cite Hu & Lu for the universality condition and flag honestly that GMM data with well-separated centres is a structured (not i.i.d.-coordinate) input, so the universality is the assumption most in need of checking. (2) Curvature leakage: for data on a C^2 manifold of intrinsic dimension d_int with reach rho and diameter R, the covariance eigenvalues beyond index d_int are not zero but bounded by C R^4/rho^2 (second-order Taylor of the embedding). So the cliff at index d_int softens at a rate set by curvature, and the sharp count d_int must be replaced by N_eff from `continuous-spectrum-effective-dimension`. Cheapest empirical companion: rerun the RFNN spectral microscope on a Swiss roll and a mixture-on-sphere at matched d_int and d_lat and show the two data bulks blur into a power-law tail as curvature grows -- this is one figure and it directly supports the effective-dimension reformulation the real-data section needs.


**Referee:** The structural observation is correct and I verified it by reading the derivation. sec-appendix-fourbulk.tex uses the data distribution only through Eq (Mt) $M_t=e^{-2t}(\hat C+\hat S)+\Delta_t I$ (second moments) and through $\eta_\star(x,x)$, which depends on $x$ only via $\|x\|^2/\dlat$. The Gaussianity of $\xi$ is indeed never invoked after Eq (Mt); what is invoked is Gaussianity of $W$ (lem:Ulin proof) and the kernel concentration of benigni2021eigenvalue (rem:fixedW). So the distribution-free restatement is genuinely free, and sec-discussion.tex limitation (iii) ('The synthetic data manifold is a linear Gaussian mixture, so curved manifolds and class hierarchies are tested mainly through CelebA') concedes more than it needs to. Marked minor by the finding, correctly.


**Referee correction:** The proposed Proposition 1 is slightly too strong as written. 'For any data distribution with finite second moment' suffices for the $U^{\rm lin}$ half (signal and noise-dim bulks), because those depend on $\mathrm{Cov}(x)$ alone. It does not suffice for the sample bulk: lem:Udiag's edge is $\bar\eta_\star=E[\sum_{k\ge3}\mu_k^2(\|x\|^2/\dlat)^k]$ and its proof needs the per-sample $\eta_\star(x_t^\mu,x_t^\mu)$ to concentrate, i.e. it needs concentration of $\|x\|^2/\dlat$, not just its mean. Heavy-tailed norms break the sample bulk into a spread rather than a band. State the proposition as 'finite second moment plus $\|x\|^2/\dlat$ concentrating' and the curvature-leakage bound as proposed.


---

## [MINOR] Two numerical constants used to justify bulk separation are wrong: mu_3^2 by 2.3x and alpha_t^2/beta_t^2 at sigma_perp = 0.01 by 72x

`mu3-numeric-and-alpha-beta-ratio-errors` · verdict **—** · kind error · effort hours  

**Where:** /Users/ryan/Desktop/latent_space_diffusion_analysis/sec-appendix-fourbulk.tex:39 (mu_3 ~= -0.099), :252-254 (mu_3^2 ~= 0.0097), :198-200 (alpha_t^2/beta_t^2 >= 10^4 at sigma_perp = 0.01); repeated at sec-rfnn-bounds.tex:79-81

**Claim:** Both constants enter the only quantitative arguments the paper offers for why the four bulks are separated, so the errors are load-bearing, not cosmetic.

**Evidence:** (a) In the convention that makes Eq (kernel-decomp) valid (orthonormal Hermite, so that sum_k mu_k^2 = E[tanh^2(z)] = 0.3943 and mu_1 = 0.6057 as the paper states), numerical quadrature gives mu_3 = -0.1484 and mu_3^2 = 0.02203, not -0.099 and 0.0097 — a factor 2.27 in mu_3^2, which directly inflates the claimed noise-dim-to-sample gap. (b) Remark rem:MP line 199-200: 'the empirical alpha_t^2/beta_t^2 >= 4 at sigma_perp = 0.5 and >= 10^4 at sigma_perp = 0.01 satisfy this comfortably.' Computed from the theorem's own definitions at t = 0.01: alpha_t^2 = 2.7641, beta_t^2 = 0.2450 (sigma_perp = 0.5) giving 10.4, and beta_t^2 = 0.019899 (sigma_perp = 0.01) giving 138.9 — not 10^4. The 10^4 figure is sigma_sig^2/sigma_perp^2, valid only if Delta_t is dropped, and the Bulk-gap-scaling paragraph (line 245-247) makes exactly that invalid approximation at the operating point where Delta_t = 0.0198 dominates sigma_perp^2 = 1e-4 by 200x.

**Fix:** Recompute mu_k from the stated definition and print the values; replace the 10^4 with 138.9 and delete the 'at small diffusion times this is approximately (s^2/d_int + sigma_sig^2)/sigma_perp^2' sentence, or qualify it with the condition e^{-2t} sigma_perp^2 >> Delta_t, which none of the experiments satisfy.


---

## [MINOR] The RFNN learning rate is proportional to d_lat and the logged "tau" axis is inflated by that same factor; the paper documents the learning rate only as "closed-form"

`rfnn-lr-and-tau-axis-undocumented` · verdict **CONFIRMED** · kind gap · effort hours  

**Where:** code/experiment_v2_rfnn.py line 236 (lr = 0.01 * d_latent / delta_t) and line 266 (tau = step * lr); ICLR_2026/sec-appendix.tex line 151 ("Learning rate ... closed-form") and lines 73-78

**Claim:** lr = 0.01 d_lat/Delta_t. Consequently the true gradient-flow time of Eq. (rfnn-gradflow) advances by exactly 0.02 U-units per step, independent of d_lat, whereas the logged and plotted quantity tau = step * lr = 0.505 d_lat * step is stretched by a factor proportional to d_lat. Any dynamics figure plotted against tau therefore shows a d_lat-proportional rightward shift that is pure axis rescaling, on top of any genuine effect. Neither the lr formula nor the tau definition appears in the paper.

**Evidence:** code/experiment_v2_rfnn.py line 236: `lr = config.lr if config.lr > 0 else 0.01 * config.d_latent / delta_t`; line 266: `tau = step * lr  # Bonnaire rescaled time`. Logged confirmation: at d_lat=20, step 5000 -> tau 50501.7 (tau/step = 10.1 = 0.505*20); at d_lat=200, step 1 -> tau 101.0 (= 0.505*200). Derivation of the 0.02: loss = ||sqrt(Delta) A f/sqrt(p) + eps||^2/(d n) gives dA/dstep = -lr(2 Delta/d) A U_paper = -0.02 A U_paper. sec-appendix.tex line 151 states only "closed-form".

**Fix:** State lr = 0.01 d_lat/Delta_t in the hyperparameter table, state that the corresponding gradient-flow time is T = 0.02 * step, and plot RFNN dynamics against T (or against steps), never against step*lr. Re-examine any figure currently on a tau axis.


**Referee:** Verified in code and paper. code/experiment_v2_rfnn.py line 236: lr = config.lr if config.lr > 0 else 0.01 * config.d_latent / delta_t, with lr = -1 (auto) in every config.json I checked; line 266: tau = step * lr with the comment 'Bonnaire rescaled time'. I re-derived the consequence independently: loss = ||sqrt(Delta_t) A phi + eta||^2 summed / (d n) gives dA/dstep = -lr * (2 Delta_t/d) A U = -0.02 A U, so gradient-flow time is T = 0.02 * step, d_lat-independent, while the logged tau = 0.505 * d_lat * step is stretched linearly in d_lat (at Delta_t = 0.0198 for t = 0.01). The paper documents the RFNN learning rate only as 'closed-form' (sec-appendix.tex hyperparameter table, line 151) and never states the tau definition. Both should be in the paper: the d_lat-proportional lr is exactly what makes the step axis comparable across the sweep, which is load-bearing for every claim about steps-to-generalize and steps-to-memorize.


**Referee correction:** The 'any dynamics figure plotted against tau shows a d_lat-proportional rightward shift ... re-examine any figure currently on a tau axis' part is refuted: tau is logged but never plotted. Every dynamics axis in the plotting code is steps (code/v3/plot.py lines 301, 335, 339, 345, 531, 649 all set xlabel 'training step'/'step'), and the eigenvalue figures are indexed by eigenvalue or by d_lat. So this is a documentation gap and a latent trap for anyone who later plots the logged field, not an existing figure artifact — which is why it is correctly filed as minor.


---

## [MINOR] The quoted Hermite coefficient mu_3 = -0.099 (mu_3^2 = 0.0097) is inconsistent with the paper's own definition and understates the higher-Hermite mass by 2.3x

`mu3-numeric-inconsistent` · verdict **CONFIRMED** · kind error · effort hours  

**Where:** ICLR_2026/sec-appendix.tex lines 612-615 (definition and numerics) and the "Bulk-gap scaling" paragraph ("eta_bar_star is dominated by mu_3^2 ~ 0.0097"); identical in sec-appendix-fourbulk.tex lines 36-39 and 252-254

**Claim:** Eq. (kernel-decomp) is written as sum_k mu_k^2 (x.y/d)^k, which requires the NORMALIZED convention mu_k = E[tanh(z) He_k(z)]/sqrt(k!). Under that convention mu_1 = 0.6057 (as quoted, since sqrt(1!) = 1) but mu_3 = -0.1484 and mu_3^2 = 0.02203, not -0.099 / 0.0097. Under the paper's literally stated definition mu_k = E[h_k(z) tanh(z)] the value would be -0.3635. Neither matches -0.099, and the quoted mu_3^2 is 2.3x too small. This feeds directly into the noise-dim-to-sample gap claim mu_1^2 beta_t^2 / eta_bar_star, i.e. into whether the two bulks are resolvable.

**Evidence:** sec-appendix.tex line 612-615: "mu_k = E_{z~N(0,1)}[h_k(z) tanh(z)] ... numerically mu_1 ~ 0.606, mu_3 ~ -0.099"; line 252-254 of sec-appendix-fourbulk.tex: "mu_1^2 ~ 0.367 while eta_bar_star is dominated by mu_3^2 ~ 0.0097". Gauss-Hermite quadrature (200 nodes) with mu_k = E[tanh(z) He_k(z)]/sqrt(k!) gives mu_1 = 0.6057, mu_3 = -0.14839, mu_5 = 0.06253, mu_7 = -0.03136; hence mu_1^2 = 0.3669, mu_3^2 = 0.02203.

**Fix:** Fix the definition to the normalized convention (mu_k = E[f He_k]/sqrt(k!)) so that Eq. (kernel-decomp) is correct as written, and correct the numerics to mu_3 = -0.1484, mu_3^2 = 0.0220; also list mu_5^2 = 0.0039 since eta_bar_star is a sum over all odd k >= 3. Recheck the bulk-gap-resolvability threshold with the corrected values.


**Referee:** Checked by 200-node Gauss-Hermite quadrature. With the paper's literally stated definition (sec-appendix.tex lines 612-615: mu_k = E_{z~N(0,1)}[h_k(z) tanh(z)], probabilists' h_k), mu_1 = 0.60571 and mu_3 = -0.36360. With the normalized convention mu_k = E[tanh(z) He_k(z)]/sqrt(k!) — the one Eq. (kernel-decomp) actually requires for sum_k mu_k^2 (x.y/d)^k to be correct — mu_1 = 0.60571 (unchanged, sqrt(1!) = 1) and mu_3 = -0.14844, mu_3^2 = 0.02203. The quoted mu_3 = -0.099 matches neither, and the derived mu_3^2 ~ 0.0097 used in the 'Bulk-gap scaling' paragraph (sec-appendix.tex line 831, and identically sec-appendix-fourbulk.tex lines 39 and 253) understates the correct 0.02203 by 2.3x. Since the stated mu_1 = 0.606 is the normalized value while the definition given is the unnormalized one, the definition as printed is also internally inconsistent with Eq. (kernel-decomp). This feeds the noise-dim/sample resolvability threshold beta_t^2 >~ eta_bar_star/mu_1^2.


**Referee correction:** Also list mu_5^2 = 0.003912 and mu_7^2 = 0.000989 (I get mu_5 = 0.062548, mu_7 = -0.031445 normalized), because at the paper's operating points eta_bar_star is NOT k=3-dominated: at q = E||x_t||^2/d_lat = 1.51 (d_lat=10, sigma_perp=0.5) the k=5 term contributes as much as k=3, and the k>=3 series only becomes k=3-dominated for q well below 1. The sentence 'eta_bar_star is dominated by mu_3^2' should be replaced by the exact statement eta_bar_star(q) = E[tanh^2(sqrt(q)z)] - q E[tanh'(sqrt(q)z)]^2, which is cheap to evaluate and removes the convention question entirely.


---

## [MINOR] tr(U) is not conserved across the d_lat sweep: it falls as d_lat^{-0.27}, which is the actual source of the edge drift

`trace-not-conserved` · verdict **REFUTED** · kind gap · effort hours  

**Where:** Measured from sigma_noise_0.5/exp2_rfnn and sigma_noise_0.01/exp2_rfnn eigenvalues_pre.npy; no trace statement appears in ICLR_2026/sec-appendix.tex, which is itself the gap

**Claim:** The buffer mechanism implicitly needs to know how spectral mass redistributes when d_lat grows. The theory never states a trace constraint. Measurement shows tr(U) in theory normalization equals E[tanh^2(sqrt(q) z)] exactly and decays as a weak power of d_lat, so mass is NOT conserved and the mean eigenvalue does not scale as 1/d_lat.

**Evidence:** tr(U_code)/p (= tr(U_theory)) at sigma_perp=0.5: 0.5453, 0.4560, 0.3633, 0.2915, 0.2614, 0.2445, 0.2327, 0.2171, 0.2092 for d_lat = 5..200; log-log fit exponent -0.271 (a 1/d_lat law would be -1.000). At sigma_perp=0.01 over d_lat=5..40 the exponent is -0.455. These match E[tanh^2(sqrt(q) z)] = 0.5748, 0.4689, 0.3728, 0.2975, 0.2649, 0.2468, 0.2347, 0.2183, 0.2098 to within 1-5%.

**Fix:** Add tr(U) = E[tanh^2(sqrt(q_t) z)] as an explicit lemma. It is a one-line sum rule that pins the overall scale, makes the q-dependence unavoidable, and gives a free consistency check on any future edge formula (the four bulk edges times their counts must sum to it).


**Referee:** The measurements are correct but no claim in the corpus is being contradicted, so this is a suggestion rather than a finding.

I reproduce the numbers exactly: tr(U_code)/p = 0.5453, 0.4560, 0.3633, 0.2915, 0.2614, 0.2445, 0.2327, 0.2171, 0.2092 at sigma_perp = 0.5, log-log slope -0.2713; slope -0.4551 at sigma_perp = 0.01 over d_lat = 5..40; and these match E[tanh^2(sqrt q z)] to 1-5%.

But the finding is framed against claims the paper never makes. Nowhere in ICLR_2026/sec-appendix.tex, sec-rfnn.tex, sec-appendix-fourbulk.tex or sec-rfnn-bounds.tex is trace conservation asserted, nor a 1/d_lat mean-eigenvalue law, nor does any step of the theorem or of Eq. (buffer-bound) invoke a spectral-mass budget — the theorem gives four edges directly from Lemmas 1 and 2. 'The buffer mechanism implicitly needs to know how spectral mass redistributes' is asserted, not demonstrated. Refuting an unstated premise is not a defect in the paper.


**Referee correction:** Re-file as an opportunity, minor: tr(U) = E_{x~P_t}[tanh^2(sqrt(q_t) z)] is an exact one-line sum rule that holds to 1-5% in every run, and adding it as a lemma would (a) make the q-dependence of the theory unavoidable, closing the door on the mu_1^2-as-constant assumption, and (b) give a free consistency check on any revised edge formula, since the four bulk edges weighted by their counts must sum to it. Note this check immediately rejects the published edges: at d_lat = 40, sigma_perp = 0.5, the theorem's counts times its edges give 5*(40*0.367*2.764) + 35*(40*0.367*0.265) + 500*(40*0.0122) ~ 500, versus the measured tr(U_theory) = 0.29.


---

## [MINOR] The alpha_t^2 = e^{-2t}(s^2/d_int + sigma_sig^2) + Delta_t dependence on d_int IS confirmed at fixed d_lat

`dint-dependence-verified` · verdict **CONFIRMED** · kind opportunity · effort hours  

**Where:** sigma_noise_0.5/exp3_rfnn and sigma_noise_0.01/exp3_rfnn raw_data (d_lat=20, d_int in {2,5,8,12,16,20}); ICLR_2026/sec-appendix.tex:681-686 (Eq. Mt block spectrum)

**Claim:** One quantitative prediction survives cleanly and should be foregrounded, because it is the only edge prediction in the paper that the data confirms without correction: at fixed d_lat, the signal edge tracks alpha_t^2 with a constant prefactor. This works precisely because q is nearly constant along the d_int sweep, which is itself the cleanest possible demonstration that q, not d_lat or psi_p, is the controlling variable.

**Evidence:** lambda_signal (median) / alpha_t^2, d_lat=20, sigma_perp=0.5: 26.27 (d_int=2), 20.47 (5), 21.22 (8), 22.04 (12), 20.97 (16), 17.39 (20); at sigma_perp=0.01: 30.93, 23.19, 23.31, 23.21, 21.48, 17.39. Apart from the d_int=2 and d_int=d_lat endpoints the ratio is constant to +/-4%, and its value ~21 matches psi_p*c1(q)^2 = 64*0.393 = 25 at the sweep's q = 0.89. Raw signal medians fall 142.1 -> 25.07 (sigma=0.5) as d_int goes 2 -> 20, tracking alpha_t^2 = 5.411 -> 1.441.

**Fix:** Present the d_int sweep as the quantitative edge validation (with the psi_p*c1(q)^2 prefactor stated), and present the d_lat sweep separately as the test the current formula fails. This also gives the corrected theory a clean, already-collected confirmation to point at.


**Referee:** The numbers reproduce and the recommendation is sound, though its stated rationale is not.

From sigma_noise_0.5/exp3_rfnn and sigma_noise_0.01/exp3_rfnn (d_lat = 20, d_int in {2,5,8,12,16,20}), lambda_signal_median / alpha_t^2 with alpha_t^2 = e^{-2t}(s^2/d_int + sigsig^2) + Delta_t: 26.27, 20.47, 21.22, 22.04, 20.97, 17.39 at sigma_perp = 0.5 and 30.93, 23.19, 23.31, 23.21, 21.48, 17.39 at sigma_perp = 0.01 — exactly the finding's values. Raw signal medians fall 142.1 -> 25.07 while alpha_t^2 falls 5.411 -> 1.441, so the alpha_t^2 dependence (the d_int-dependence of the theory's M_t block spectrum, sec-appendix.tex:681-686) does track with a roughly constant prefactor across the middle of the sweep. Given that findings 1-3 show the d_lat sweep breaks the edge formula 12-216x, foregrounding the d_int sweep as the surviving quantitative validation is the right editorial call.


**Referee correction:** The stated rationale is wrong and should not be published as-is. q is NOT nearly constant along the d_int sweep: q = 0.780, 0.890, 1.000, 1.147, 1.294, 1.441 for d_int = 2..20, an 1.85x drift, and c1(q)^2 correspondingly falls 0.416 -> 0.288 (1.44x). So the corrected theory predicts the ratio psi_p*c1(q)^2 to fall 26.6 -> 18.4 across the sweep, which is roughly what the endpoints do (26.27 -> 17.39); the '+/-4% constant' holds only over the four interior points d_int = 5..16, where measured 20.5-22.0 sits ~15% below the predicted 19.9-25.2. Correct claim: the signal edge tracks c1(q_t)^2 * alpha_t^2 across the d_int sweep with residual scatter of order 15%, and the sweep is worth foregrounding because alpha_t^2 varies 3.75x while the corrected prefactor varies only 1.4x — not because q is constant. Also drop 'the only edge prediction the paper confirms without correction': the prefactor still needs psi_p (i.e. the 1/d_lat form), so this validates the alpha_t^2 factor, not the published edge.



---

# Completeness critic (ran on the failed synthesizer; contains additional independent checks)

**The PLAN block is literally `null` — no plan was supplied, so there is nothing to critique line-by-line. Below is the list of things any theory-audit plan for this project must contain, each checked against the files.**

## Corpus nobody read

- **`ICLR_2026/sec-appendix.tex` (946 lines) is byte-identical to `ICML (1)/sec-appendix.tex`** (`diff -q` returns identical) and *already sits inside the current draft folder*, containing `\begin{theorem}` `thm:fourbulk` (L787), `lem:Ulin` (L691), `lem:Udiag` (L713), and Eq. `eq:buffer-bound` (L586). `main.tex` L137 inputs `sec-appendix-integrated.tex` instead (0 theorem/lemma/proof envs). The theory is not "orphaned in an old folder" — it is orphaned by one `\input` line. Any plan that says "port the theory in" must first reconcile *three* near-duplicate copies (`ICLR_2026/sec-appendix.tex`, `ICML (1)/sec-appendix.tex`, `sec-appendix-fourbulk.tex`), which differ: the fourbulk drop-in replaces the μ2/Pennington–Worah sample-bulk argument with a μ_{k≥3} one, and the ICLR copy still carries the *old* "Four-bulk intuition" prose (L486–558) that the drop-in was meant to delete.
- **`ICLR_2026/_stash.tex` (100 lines, not input)** contains the deleted "Anisotropic data is the realistic case" motivation and the four-bulk design paragraph that asserts the counts *empirically* ("across every configuration we ran"). It is the only place the count claim is stated as observation rather than theorem.
- **`bulk_summary.json`** (35 runs, root dir) — the only file with measured bulk *eigenvalue magnitudes*. No audit lens appears to have opened it. It is the single most decisive artifact in the repo.
- **`sigma_noise_0.5/exp2_exp3_dynamics/README.md`** — the RFNN training-dynamics record. Contains the sentence that breaks the paper (below).

## Claims nobody checked (each is falsified or unverifiable against the repo's own data)

- **The buffer bound does not follow from the mode-decay ODE, and contradicts the theorem it cites.** Under Eq. `eq:rfnn-mode-decay`, modes decay *in parallel*: τ to tolerance ε is log(|a_i(0)−a_i*|/ε)/λ_i. So τ_mem − τ_gen depends only on the two edge values λ_sample, λ_signal — **the number of modes between them never enters**. Multiplying by (d_lat − d_int) in `eq:buffer-bound` has no derivation. Worse: at fixed ψ_p = 64, all four edges in `thm:fourbulk` are d_lat-independent, so the theorem predicts τ_mem − τ_gen is *constant in d_lat* while its own corollary claims a lower bound growing linearly in d_lat. The RHS eventually exceeds the LHS: the bound is false, not merely unproven. Also A(0)=0, so the amplitude |a_i*| is what makes memorization visible, and it is never modeled.
- **Sample-bulk edge η̄/ψ_p is n-independent ⇒ the theorem contradicts Bonnaire's central result.** Bonnaire's τ_mem grows with dataset size n; the paper claims to recover his two-bulk picture as σ⊥→σ_sig (`sec-rfnn-bounds.tex`). An n-independent sample edge cannot reproduce τ_mem ∝ n. This is a 30-second sanity check nobody ran, and it localizes the bug: from Eq. `eq:U-diag`, U = (1/n)Σ… and E‖φ⊥‖² = η̄, so **λ_sample ≈ η̄/n, not η̄/ψ_p** — which restores τ_mem ∝ n. Same normalization error in Lemma 2's proof, which derives η_star²/(p·n) (η squared, and inconsistent with the theorem's table). Lemma 1 has the mirror error: nonzero eigenvalues of W M_t Wᵀ are ≈ p·eig(M_t), so **λ_signal ≈ μ1²α_t²/d_lat, not μ1²α_t²/ψ_p** (these agree only if p ∝ d_lat²). The corrected form is d_lat-*dependent*, killing "τ_gen is d_lat-independent" as stated.
- **The claim "predicted bulk counts match the empirical B1…B4 sizes in Table `tab:bulk-sizes`" (`sec-rfnn-bounds.tex`) is false against the paper's own table.** `ICLR_2026/sec-appendix.tex` L163–192, d_lat=20, p=1280, n=500: predicted (760, 500, 15, 5), observed B1..B4 = (916, 311, 48, 5). At d_lat=10: predicted (130, 500, 5, 5), observed (295, 305, 35, 5). **No bulk has size 500 at any d_lat.** The caption's "B3 grows linearly with d_lat − d_int" is B3 = 27 at d_lat = d_int = 5, i.e. a large nonzero intercept where the buffer width is exactly zero. Meanwhile `bulk_summary.json` reports sizes exactly (5, d_lat−5, 500, p−d_lat−500) because they are *assigned by index block*, not measured — so the two "count verifications" in the repo are one circular and one contradictory.
- **Edge magnitudes have never been compared to data at all.** `ICLR_2026/sec-appendix.tex` L840–842 claims the edges "match the empirical bulk locations in Table `tab:bulk-sizes` and Fig. `bulk-sizes-distances`" — that table contains only *sizes*, no eigenvalues, and that figure is a find_peaks overlay. Actual check against `bulk_summary.json`, σ⊥=0.5, ψ_p=64 fixed: signal-bulk edge runs 31.9 → 73.9 (2.3×) and noise-dim top edge 6.44 → 27.17 (4.2×) as d_lat goes 10 → 200, while the theorem says both are d_lat-independent. Signal-to-noise-dim gap: theory predicts a constant 1.019 dec; measured 0.695 → 0.434 dec.
- **In the σ⊥ = 0.01 regime the buffer bulk is not between generalization and memorization.** `bulk_summary.json`, `gap_noise_to_sample_dec` = 0.032, 0.019, 0.006, 0.004, 0.056 for d_lat = 10, 15, 20, 30, 40 — factors of 1.01–1.14, i.e. the noise-dim and sample bulks *touch* (d_lat=20: nd_min 0.584 vs samp_max 0.576). Also samp_min ≈ rank_null_max to 3 digits at every d_lat, so the sample "bulk" is a continuum from the noise bulk down to the rank-null floor, not a gapped class. Yet σ⊥=0.01 is the regime the draft calls the clean four-bulk case and where the MLP n-shape story lives.
- **τ_mem was never observed in the RFNN.** `sigma_noise_0.5/exp2_exp3_dynamics/README.md`: "Gen gap … roughly similar across d_latent values. **No clear memorization onset within 300k steps**." Re-deriving from `raw_data_exp2v3/*/metrics.jsonl`: first step with gen_gap > 0.02 is 20k (d=5), 1 (d=8), 15k (d=10), 30k (d=15), 25k (d=20), 1 (d=30), 10k (d=40) — non-monotone in d_lat, gen_gap goes negative at d=8, and max gen_gap *decreases* with d_lat (0.086 → 0.048). The system the theory actually describes provides zero support for the buffer, and the "gen-gap measures null-structure absorption" paragraph in `sec-rfnn.tex` is contradicted by it.
- **Timescale consistency was never checked.** Same runs reach gradient-flow time τ = 3.03e6, while 1/λ for the *slowest* rank-null mode at d_lat=20 is ~500. Every mode is absorbed thousands of e-folds over, yet gen_gap ends at 0.04. Either the U used for the spectrum and the U driving training are on different normalizations, or "absorbing sample modes = memorizing" is wrong. Nothing in the corpus resolves this.

## Alternative explanations nobody considered

- **The tanh operating point, not the mode count, produces the whole d_lat effect.** The Hermite coefficients must be taken at the actual pre-activation variance q_t = E‖x_t‖²/d_lat, which is *not* 1 and sweeps 1.52 → 0.33 over the σ⊥=0.5 d_lat sweep. Both μ1² and η̄ are then strongly d_lat-dependent (computed: μ1²(q) = 0.42 → 0.20; η̄ = 3.3e-2 → 4.8e-3). Predicted noise-dim→sample gap log10(μ1²(q)β_t²/η̄(q)) vs measured `gap_noise_to_sample_dec`: 0.39/0.64 (d=10), 0.58/0.89 (20), 0.76/1.12 (40), 0.94/1.26 (100), 1.03/1.09 (200). **The corrected, count-free formula reproduces the entire observed growth of the spectral gap**; the paper's constant-μ1², constant-η̄ formula predicts a flat gap. Corollary: the RFNN "buffer" may be nothing but the feature map becoming more linear as ‖x‖²/d_lat falls — the *same* quantity the saturation appendix (`app:saturation`, s/√d_lat ≲ 2) already flags as a confound, in the opposite direction. Nobody noticed the d_lat sweep is also a nonlinearity sweep.
- **The noise-dim bulk's width is a rank-d_lat sample-fluctuation (MP with ratio d_lat/n) effect, exactly the one the Remark dismisses.** Measured nd_max/nd_min = 1.84, 2.82, 5.53, 12.8 at d_lat = 20, 40, 100, 200 (n=500); MP prediction ((1+√(d/n))/(1−√(d/n)))² = 2.25, 3.2, 6.9, 19.7. The Remark on rank addition asserts these Ŝ−Σ modes are harmless "MP deformations", but they set λ_min^{noise-dim}, which is the only quantity `eq:buffer-bound` depends on. Also: this makes the paper's "population vs sample structure" dichotomy (the gen-gap argument) incoherent, since the noise-dim bulk is itself largely sample fluctuation.
- **Nobody proposed the one control that separates counts from edges:** sweep **n** at fixed d_lat, σ⊥, p. The theorem says the sample-bulk *count* grows with n at a fixed edge ⇒ τ_mem unchanged; the corrected λ_sample ≈ η̄/n and Bonnaire both say τ_mem ∝ n. That single sweep discriminates the count mechanism from the edge mechanism. It exists nowhere in `code/` or `_next_steps/`.

## Empirical story the theory must explain but the corpus ignores

- **The spectral predictor (contribution 5) is disconnected from contributions 1–4 and nobody flagged it.** `sec-spectral-predictor.tex` builds the clock from M_t(d) = e^{−2t}ZᵀZ/n + (1−e^{−2t})I — the *data* covariance, d modes, with w_i = 1. It contains **no sample bulk, no rank-null bulk, no Hermite factor, no feature lift**, i.e. none of the four-bulk apparatus, and therefore has no object representing memorization at all. It is also monotone in s by construction (P_d(s) ↑, sigmoid ↑), so it cannot express any non-monotone memorization trajectory, and it carries three free parameters (κ, a, b) fit per dataset with no baseline (e.g. a clock from tr(M_t) alone, or a pure d-dependent time rescaling). An audit plan that only covers the RFNN theorem leaves the paper's headline *practical* claim entirely unaudited.
- **Real latent spectra are not block-anisotropic.** The whole theory assumes a two-level Σ_data; VAE latent spectra are graded, so "d_lat − d_int buffer modes" has no referent on CelebA/CIFAR-10. `data_pca_dlat_sweep.pdf` exists but no one derived the general-Σ statement the real-data and predictor sections require.
- **The σ⊥ ∝ 1/(d_lat − d_int) "strictest control" is the one shown in the main text (`sec-rfnn.tex`), and it changes β_t² — i.e. it moves the theorem's edges while claiming to isolate the count.** No control in the paper holds the edges fixed and varies the count. Given the four-bulk construction, none can; that needs saying explicitly rather than being presented as a clean test.
- **The MLP n-shape (`_next_steps/n_shape_heuristic_derivation.md`) uses μ1 ≈ 0.80 for GELU and λ_null on the *diffusion floor* Δ_t at σ⊥ = 0.01 — i.e. it already concedes that at σ⊥ = 0.01 the noise-dim bulk sits at Δ_t, which is precisely why it merges with the sample bulk (see gap ≈ 0 above).** Nobody connected that document's own numbers to the collapse of the buffer ordering in the RFNN spectra at the same σ⊥.

## Corrections to the only extant plan (`_next_steps/theory_plan.md`) — it is wrong in three places

- **Its flagged "hole" is misdiagnosed.** It insists the sample bulk needs the **Hermite-2 / μ2** Pennington–Worah term. tanh is **odd, so μ2 = 0**; there is no μ2 term. The drop-in `sec-appendix-fourbulk.tex` correctly uses μ_{k≥3}. The plan's "conflict" should be resolved *in favor of the appendix*, and the plan's edge table (`μ2² e^{−4t}σ_sig⁴/n`) discarded. A plan that inherits `theory_plan.md`'s framing starts from a false premise.
- **The proportional limit in the theorem is not the limit the experiments take.** `ICLR_2026/sec-appendix.tex` L604–605 fixes ψ_p, ψ_n and requires ψ_p > 1 + ψ_n. The sweeps hold **n = 500 fixed** while d_lat goes 5 → 200, so ψ_n runs 100 → 2.5. At d_lat = 5, ψ_n = 100 > ψ_p = 64 and p = 320 < d_lat + n = 505 — the hypothesis is **violated at the anchor data point** that carries the entire empirical claim (30% memorization at d_lat = d_int), and `bulk_summary.json` confirms it (sample size 315, not 500; no noise_dim block). A fixed-ψ asymptotic theorem cannot speak to this sweep at all; either restate in a d_lat-with-n-fixed regime or exclude d_lat ≤ 8.
- **Numerical errors to fix while rewriting, not "check later":** (i) μ3 is quoted as −0.099 with μ3² ≈ 0.0097 (L615, L831) — under the convention that gives μ1² = 0.367 the correct value is μ3 = −0.148, μ3² = 0.0219, so η̄ is understated ~2.3×, and η̄ is exactly what sets the noise-dim→sample gap; (ii) L774–775 evaluates Δ_MP = (√ψp+1)²/(√ψp−1)² − 1 at ψ_p = 64 as ≈ 0.50 (the large-ψ asymptote) when the stated formula gives 0.653.
- **"Multi-seed rerun by whoever finishes first" is under-scoped.** Every eigenvalue number in `bulk_summary.json` is seed 42, single run. But the seed issue is second-order next to the fact that the quantities being compared (assigned index blocks vs find_peaks bulks) are two different estimators that disagree by 10× on counts. Fix the estimator before adding seeds.
- **Nothing in the plan assigns anyone the reduction check to Bonnaire's isotropic limit**, which is the cheapest possible falsification test and which the current theorem fails (see n-independence above).