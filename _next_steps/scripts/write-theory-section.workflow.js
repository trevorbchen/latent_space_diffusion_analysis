export const meta = {
  name: 'write-theory-section',
  description: 'Derive, adversarially verify, and write the corrected theory (Section 2 + theory appendix) for the latent-dimensionality memorization paper, covering RFNN, synthetic MLP, real-data VAE latents and the spectral predictor (not DiT)',
  phases: [
    { title: 'Derive', detail: 'eight independent theory pieces' },
    { title: 'Verify', detail: 'two adversarial checkers per piece' },
    { title: 'Revise', detail: 'fold verifier findings back in' },
    { title: 'Assemble', detail: 'unified Section 2 + appendix, then consistency critic' },
  ],
}

const ROOT = '/Users/ryan/Desktop/latent_space_diffusion_analysis'

const CONTEXT = `
You are writing THEORY for the paper "How Excess Latent Dimensionality Delays Memorization in Diffusion Models"
(draft in ${ROOT}/ICLR_2026/, target: ICLR main track). Section 2 (${ROOT}/ICLR_2026/sec-theory.tex) is currently
an empty TODO. Your output will become that section plus a theory appendix. Do NOT write files; return LaTeX as text.

======================= NOTATION (macros defined in main.tex; USE THEM) =======================
\\dlat = d_latent, \\dint = d_intrinsic, \\nsamp = n, \\pwidth = p, \\taugen, \\taumem, \\Umat = U,
\\Sigmadata, \\sigsig = sigma_signal, \\signoise = sigma_perp, \\R, \\E, \\diag.
Theorem envs available: theorem, proposition, lemma, corollary, definition, assumption, remark (numbered by section).
Delta_t = 1 - e^{-2t}. RFNN: s_A(x_t) = p^{-1/2} A tanh(W x_t / sqrt(d_lat)), W in R^{p x d_lat} iid N(0,1), A in R^{d_lat x p}
trained from zero by gradient flow on score matching at fixed t. U = (1/n) sum_mu E_xi[phi(x_t^mu) phi(x_t^mu)^T],
phi(x) = p^{-1/2} tanh(Wx/sqrt(d_lat)). Per-mode decay a_i(T) - a_i* = (a_i(0)-a_i*) e^{-lambda_i T}.
Data: x^mu = m_{c(mu)} + xi^mu, k=10 centers of norm s=3 in a d_int=5 subspace, xi ~ N(0, diag(sig_sig^2 I_dint, sig_perp^2 I_{dlat-dint})),
sig_sig=1, sig_perp in {0.01, 0.5}, n=500, then a random orthogonal rotation Q. Diffused x_t = e^{-t}x + sqrt(Delta_t) eta.
M_t = (1/n) sum_mu E[x_t^mu x_t^mu T] = e^{-2t}(hat C + hat S) + Delta_t I. Population block eigenvalues:
alpha_t^2 = e^{-2t}(s^2/d_int + sig_sig^2) + Delta_t (signal), beta_t^2 = e^{-2t} sig_perp^2 + Delta_t (null).
Pre-activation variance q = tr(M_t)/d_lat. Gaussian-equivalent coefficients a_1(q) = E[tanh'(sqrt(q) z)], a_*(q)^2 = E[tanh(sqrt(q) z)^2] - a_1(q)^2 q.

======================= ESTABLISHED FACTS (from a completed audit + direct simulation; treat as ground truth) =======================
F1. rank(U) <= n. Bulk counts are: signal d_int, noise-dim d_lat - d_int, sample n - d_lat, rank-null p - n.
    The spectral cliff is at index n (machine precision), NOT at d_lat + n. The old draft's count "n" and "p - d_lat - n" are wrong.
F2. Bulk edges are p-INDEPENDENT (verified: changing p by 5.8x moves edges by <0.5%) and scale with d_lat.
    Correct leading-order data-side edges: signal ~ a_1(q)^2 alpha_t^2 / d_lat, noise-dim ~ a_1(q)^2 beta_t^2 / d_lat.
    The sample bulk is a Marchenko-Pastur-type bulk of the residual features with total mass a_*(q)^2 and mean eigenvalue ~ a_*(q)^2 / n.
    (Old draft divided all edges by psi_p = p/d_lat: WRONG, violates tr(U)=O(1).) Any edge you state MUST pass a trace check:
    d_int*signal + (d_lat-d_int)*noise + (n-d_lat)*sample_mean ~ a_1(q)^2 q + a_*(q)^2 = E[tanh(sqrt(q)z)^2] = O(1).
F3. mu_1 is NOT constant across the sweep. q falls 2.76 -> 0.33 (sig_perp=0.5, d_lat 5->200), tanh de-saturates, a_1(q)^2 rises 0.180 -> 0.625.
    Measured signal-bulk medians (code normalization = p x theory) 29.7 (d=5) -> 99.8 (d=200); a_1(q)^2 alpha_t^2 psi_p predicts 31.9 -> 110.6.
    Noise-dim medians 5.22 -> 9.22 vs predicted 4.76 -> 10.60. The fixed-mu_1 theory predicts flat 64.9 and 6.22. So tau_gen is NOT d_lat-independent.
F4. The old buffer bound tau_mem - tau_gen >= (d_lat - d_int) * 1/lambda_min^{noise-dim} is a NON-SEQUITUR (modes decay in parallel under a
    diagonal ODE; no count enters a traversal time) and is numerically violated at every d_lat >= 20. Do NOT reproduce it.
F5. What IS true on the repo's own spectra (sig_perp=0.5, p=64 d_lat, onset statistic = largest sample-bulk eigenvalue):
    tau_gen = 1/lambda_min^{signal} grows 788 -> 6413 steps (8.1x) over d_lat = 5..200; tau_mem(onset) grows 221x;
    tau_mem/tau_gen grows MONOTONICALLY 31 -> 576. On the bulk MEDIAN statistic the ratio instead shrinks 7.8x. So the buffer
    claim survives only as a statement about the ONSET (top of sample bulk) relative to the signal edge, and it must be stated that way.
F6. Isotropic control (sig_perp = sig_sig = 1): the spectral gap still grows 10.7x with d_lat. Under isotropy the noise-dim modes
    still exist between signal and sample modes; anisotropy adds spectral SEPARATION of that block, not its existence. So "the buffer
    does not exist in the isotropic case" is false; the four-bulk STRUCTURE is what is absent. State this correctly.
F7. Bonnaire et al. 2025 (arXiv:2505.17638) Theorem 3.1 is stated for ARBITRARY spectral measure rho_Sigma (their SM derives the GEP
    "under arbitrary input covariance"). The four-bulk result is their theorem evaluated at a two-atom rho_Sigma; say so, do not claim to
    "extend" it. George, Veiga & Macris (arXiv:2502.00336) already have a linear pencil with block dims (p, d-D, D, n) for data on a
    D-dim subspace with sig_perp = 0. Cite both. Defensible novelty: strictly positive sig_perp, the training-time (gradient-flow) reading,
    and everything downstream (bridge to memorization, predictor).
F8. Lemma 2 of the old draft is internally inconsistent (three different sample edges) and its orthonormality claim v_mu^T v_nu = delta + o(1)
    is false for clustered data. Do not reuse it. Also eta_star as a divergent series is wrong; use a_*(q)^2 instead.
F9. Real VAE latents (from the repo's own spectral CSVs, t=0.1, n=1000): effective rank ~ 0.45-0.55 x nominal d across the whole sweep;
    lambda_min = Delta_t = 0.181 exactly for d >= 140 (CelebA) / >= 160 (CIFAR-10) -> those coordinates have ZERO data variance (posterior collapse);
    CelebA effective rank SATURATES: 89.5 (d=160), 90.3 (180), 87.9 (200) while memorization keeps falling; q falls 2.92 -> 1.30.
    Real spectra are continuous/power-law-ish, not two-atom.
F10. Memorization is measured as the fraction of generated samples with NN-ratio d(gen,NN1)/d(NN1,NN2) < 1/3 (Somepalli/Bonnaire), in pixel
    space after decoding for real data. The paper reports FRACTION curves and final fractions, NOT hitting times. gen_gap is flat in d_lat
    (0.82 -> 0.80) while the fraction collapses 0.30 -> 0.0012 and train loss falls 0.53 -> 0.20 (synthetic, d_lat 5->40). The NN-ratio has a
    known distance-concentration drift with ambient dimension (mean_nn_ratio 0.52 -> 0.87). Theory should predict a FRACTION curve, and
    a single sample-bulk edge predicts a step function, so a within-bulk eigenvalue spread over training points is required.
F11. The released spectral predictor: M_t(d) = e^{-2t} Z^T Z/n + Delta_t I; P_d(s) = mean_i (1 - exp(-kappa lambda_i s))^2; anchored sigmoid.
    The square is not derived; the clock has NO sample-bulk term (memorization is the sample bulk); as released it fits 1 kappa + 6 monotone
    threshold levels on all d with no holdout.
F12. The null block of the true score is EXACTLY linear: s*_null(x_t) = -x_t,null / beta_t^2 (cluster means have zero null component).
    A linear score model s(x) = Bx under gradient flow on E||Bx - s*||^2 has closed form B(T) = -Sigma_t^{-1} + (B_0 + Sigma_t^{-1}) e^{-2 Sigma_t T}.
F13. Regime boundaries: psi_n = n/d_lat is NOT fixed in the sweeps (d_lat 5..240 at n=500); p = d_lat+n+300 violates psi_p > 1+psi_n;
    d_int=5 fixed so the signal "bulk" is a finite-rank spike (BBP), not a bulk. As d_lat -> n the sample bulk n - d_lat vanishes and
    memorization empirically RETURNS at large d_lat (hidden = 8 d_lat sweep). sig_perp -> 0 puts the noise-dim bulk on the diffusion floor Delta_t.
F14. The synthetic score-error metric had a transpose bug (Q^T D^-1 Q instead of Q D^-1 Q^T), now fixed; all saved synthetic score_error
    numbers and the "n-shape" are artifacts. Do not build theory on them.
F15. Empirical-score memorization: the exact score of the noised empirical measure is s_emp(x,t) = sum_mu w_mu(x)(e^{-t}x^mu - x)/Delta_t with
    softmax weights; reverse dynamics return a training point when w concentrates, which happens for Delta_t < d_NN^2/(2 log n)
    (Biroli et al. 2024 condensation threshold, cited as biroli2024dynamical). Scarvelis et al. 2023 and Gu et al. 2023 are cited for the closed form.

======================= CITATIONS =======================
references.bib is at ${ROOT}/ICLR_2026/references.bib. Existing keys you may use include bonnaire2025, biroli2024dynamical,
scarvelis2023closedform, gu2023memorization, somepalli2023diffusion, kadkhodaie2023learning, pennington2017nonlinear,
benigni2021eigenvalue, ELKaroui2010spectrum. Check the file (grep) before using any other key; for citations that are not present,
use a clearly marked placeholder key like \\citep{george2025dsm} and list it under citations_needed with full bibliographic info.
Likely needed: George, Veiga & Macris 2025 (arXiv:2502.00336); Hu & Lu 2022 (universality of conjugate kernel); Mei & Montanari 2022;
Goldt et al. 2020/2022 (Gaussian equivalence); Ba, Erdogdu, Suzuki, Wu, Yang 2022 (one-step feature learning); Baik-Ben Arous-Peche 2005 (BBP).

======================= STYLE =======================
State every result with explicit hypotheses. Distinguish proved / sketched / conjectured honestly. Every displayed edge or timescale
must carry its d_lat-dependence explicitly. Write for a hostile expert referee who has read Bonnaire and George et al.
Main-text LaTeX should be compact (each piece ~8-25 lines); appendix LaTeX can be full. Use \\label{} on every numbered statement,
with labels prefixed thm:, prop:, lem:, cor:, def:, eq:, and piece-specific suffixes so pieces do not collide.
`

const DERIVE_SCHEMA = {
  type: 'object',
  properties: {
    piece: { type: 'string' },
    title: { type: 'string' },
    statements: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          kind: { type: 'string', enum: ['definition', 'assumption', 'proposition', 'theorem', 'lemma', 'corollary', 'remark'] },
          label: { type: 'string' },
          status: { type: 'string', enum: ['proved', 'sketched', 'conjectured'] },
          latex_statement: { type: 'string' },
          latex_proof: { type: 'string' },
          hypotheses: { type: 'array', items: { type: 'string' } },
        },
        required: ['kind', 'label', 'status', 'latex_statement', 'latex_proof', 'hypotheses'],
      },
    },
    predictions: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          claim: { type: 'string' },
          how_to_test: { type: 'string' },
          already_supported_by: { type: 'string' },
        },
        required: ['claim', 'how_to_test', 'already_supported_by'],
      },
    },
    empirical_anchor: { type: 'string', description: 'which numbers from the repo this piece matches or predicts, with values' },
    citations_needed: { type: 'array', items: { type: 'object', properties: { key: { type: 'string' }, bib: { type: 'string' } }, required: ['key', 'bib'] } },
    open_gaps: { type: 'string' },
    main_text_latex: { type: 'string', description: 'compact Section 2 version of this piece, 8-25 lines' },
    appendix_latex: { type: 'string', description: 'full version with proofs' },
  },
  required: ['piece', 'title', 'statements', 'predictions', 'empirical_anchor', 'citations_needed', 'open_gaps', 'main_text_latex', 'appendix_latex'],
}

const VERIFY_SCHEMA = {
  type: 'object',
  properties: {
    verdict: { type: 'string', enum: ['SOUND', 'FIXABLE', 'UNSOUND'] },
    issues: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          severity: { type: 'string', enum: ['fatal', 'major', 'minor'] },
          where: { type: 'string' },
          problem: { type: 'string' },
          fix: { type: 'string' },
        },
        required: ['severity', 'where', 'problem', 'fix'],
      },
    },
    must_change: { type: 'string', description: 'the single most important change, or "none"' },
    independent_check: { type: 'string', description: 'any algebra or numerics you ran yourself, with results' },
  },
  required: ['verdict', 'issues', 'must_change', 'independent_check'],
}

const PIECES = [
  {
    key: 'P1-setup-definitions',
    prompt: `PIECE 1: Setup, standing assumptions, and formal definitions of tau_gen and tau_mem.
Deliver: (a) an Assumption block listing exactly the hypotheses the theory needs (p > n; saturation/operating-point condition stated in terms of q,
not s/sqrt(d_lat); fixed t; the proportional or non-proportional limit actually used, noting d_int is fixed so the signal block is a finite-rank spike;
which regime of psi_n = n/d_lat); (b) Definition of the residual spectral error E(T) = sum_i pi_i e^{-2 lambda_i T} with pi_i = ||P_i A*||_F^2 the target
mass in mode i (exact for the RFNN from the per-mode decay); (c) Definition 1: tau_gen(eps) as the time the population-supported modes are absorbed
to tolerance eps, and show tau_gen(eps) = (1/(2 lambda_min^{signal})) log(1/eps) + O(1); (d) Definition 2: tau_mem via the spectral generalization gap
G(T) = E_test(T) - E_train(T) restricted to sample modes, and ALSO an onset version tau_mem^{onset} = 1/lambda_max^{sample}; state precisely which
measured proxy each corresponds to (test-loss-within-5%-of-min; gen_gap > 0.02; NN-ratio fraction crossing q). Make explicit that the paper measures
a FRACTION and that a fraction-level definition needs the within-sample-bulk spread (to be supplied by Piece 6). Keep everything as functionals of the
spectrum so the theory is falsifiable by the measurements.`,
  },
  {
    key: 'P2-four-bulk',
    prompt: `PIECE 2: The four-bulk proposition, corrected.
Deliver a Proposition (status: sketched, with the rigorous route named) giving: counts d_int, d_lat - d_int, n - d_lat, p - n (F1); leading-order
edges with EXPLICIT d_lat dependence via the Gaussian-equivalent coefficients a_1(q), a_*(q) at q = tr(M_t)/d_lat (F2, F3); the sample bulk as a
Marchenko-Pastur-type bulk of residual features with mass a_*(q)^2 (derive its edges properly and PASS THE TRACE CHECK in F2 -- show the trace
arithmetic explicitly); the separation conditions under which the four groups are disjoint (signal vs noise-dim: alpha_t^2/beta_t^2 vs MP spread with
ratio d_lat/n, NOT p; noise-dim vs sample: a_1(q)^2 beta_t^2/d_lat vs top MP edge of the sample bulk); the isotropic limit (F6: noise-dim modes persist
but merge with signal into Bonnaire's rho_2); and an honest positioning paragraph relative to Bonnaire Thm 3.1 at a two-atom rho_Sigma and to
George-Veiga-Macris (F7). Proof route: Gaussian equivalence for the conjugate kernel (Hu-Lu; Goldt et al.; Mei-Montanari) giving
phi(x) ~ a_1(q) W x/sqrt(d_lat) + a_*(q) theta with theta iid, then U ~ (a_1^2/(p d_lat)) W M_t W^T + (a_*^2/(p n)) Theta^T Theta + cross terms,
with W-independence of Theta giving the orthogonal-range structure. Explain why rank <= n forces the sample block to n - d_lat.
Include a short table of predicted vs measured medians using the numbers in F3 (state the normalization: code eigenvalues = p x theory).`,
  },
  {
    key: 'P3-buffer-corollary',
    prompt: `PIECE 3: The buffer corollary, restated correctly.
Starting from the per-mode decay and the corrected edges (F2, F3), derive what actually happens to tau_gen and tau_mem^{onset} as d_lat grows at
fixed d_int, n, sig's, t. Show: tau_gen ~ d_lat/(2 a_1(q(d_lat))^2 alpha_t^2) * log(1/eps) is NOT d_lat-independent and give its scaling;
tau_mem^{onset} ~ 1/lambda_max^{sample} with its d_lat dependence through a_*(q) and the MP top edge; therefore the RATIO tau_mem/tau_gen and how it
opens with d_lat. State as a Corollary: "the buffer" = the ratio lambda_min^{signal}/lambda_max^{sample}, with NO mode count. Explain in a Remark why
the old count-times-timescale bound is a non-sequitur under a diagonal linear ODE (F4), and why the median statistic gives the opposite trend (F5),
so the claim is specifically about onset. Reconcile with F6: what the isotropic case does and does not show. Give the empirical anchor from F5
(31 -> 576 on onset; 8.1x growth of tau_gen). State clearly what the theory does NOT predict: it does not by itself predict the final memorization
FRACTION at a fixed step budget, which additionally depends on the within-bulk spread (Piece 6) and on quality being matched.`,
  },
  {
    key: 'P4-linear-score-model',
    prompt: `PIECE 4: The exactly solvable linear score model.
Using F12: write the Lemma that for the null block the true score is exactly linear, s*_null = -x_null/beta_t^2, and that a linear score model
s_B(x) = Bx trained by gradient flow on E||Bx - s*(x)||^2 has closed-form B(T) = -Sigma_t^{-1} + (B_0 + Sigma_t^{-1}) e^{-2 Sigma_t T} in the
population-covariance eigenbasis (prove it: the loss is quadratic, dB/dT = -2(B Sigma_t + I) restricted appropriately; be careful that the signal
block target is NOT linear, so state the lemma for the null block exactly and for the signal block as the best linear approximation). Derive the
per-mode risk contribution e^{-4 lambda_i T}/lambda_i and hence: (i) the mode clock tau_i = 1/(4 lambda_i) rigorously for this model; (ii) the
observation that low-lambda modes carry the LARGEST target energy 1/lambda_i so the noise-dim block DOMINATES the late residual rather than merely
delaying; (iii) the RFNN corollary by substituting Sigma_t -> U (frozen features, zero-init readout). Give the exact total-risk-vs-(d_lat, T) formula
for the block-anisotropic case and say what it predicts qualitatively for the (now bug-free, to be rerun) synthetic score error. Status: proved.`,
  },
  {
    key: 'P5-bridge-lemma',
    prompt: `PIECE 5: The bridge lemma from mode absorption to near-duplicate generation.
Using F15 and F10: write a Lemma/Proposition (status: sketched, with the gap named) that connects the residual spectral error E(T) of a partially
trained score to memorization of generated samples. Route: the exact score of the noised empirical measure s_emp memorizes when the softmax weights
concentrate on one training point, i.e. when Delta_t < d_NN^2/(2 log n) (Biroli condensation). A partially trained score s_theta = s_emp + error with
E||error||^2 = E(T) behaves like the exact empirical score at an effective noise level Delta_eff(T) = Delta_t + c E(T) (make the constant and the
approximation explicit: treating the error as isotropic Gaussian, and say when that is justified). Hence tau_mem = inf{T : E(T) <= [d_NN^2/(2 log n) - Delta_t]/c}.
Derive the consequences: (a) the Somepalli 1/3 threshold becomes a statement about the generated point lying within a third of the NN spacing, so the
relevant d_NN for "fraction crossing q" is the q-quantile of the training-set NN-distance distribution -- this is what turns a single edge into a
FRACTION curve at the data level; (b) tau_mem depends on n and d_int through d_NN ~ n^{-1/d_int} (state the scaling, and note it is a d_int not d_lat
dependence, which is a sharp testable prediction); (c) the pixel-space vs latent-space subtlety for real data (decoder Lipschitz constant).
Name honestly what is missing for a full proof.`,
  },
  {
    key: 'P6-predictor-derivation',
    prompt: `PIECE 6: Deriving the frozen-latent spectral predictor from the theory, and giving it a sample-bulk term and per-sample resolution.
Using F11, F1-F3, F10: (a) derive from the linear model (Piece 4 logic, redo the needed step here) the correct absorbed-mass clock: the residual in
mode i is e^{-lambda_i T}, so absorbed fraction is 1 - e^{-lambda_i T} and absorbed squared error is 1 - e^{-2 lambda_i T}; state which of these the
released (1 - e^{-kappa lambda_i s})^2 corresponds to and whether the square is justified (it is not derived; say what it would take). (b) Explain
why a clock built only from spec(M_t) covers only the data-side bulks and misses the sample bulk that IS memorization; add the sample-bulk term:
n - d_lat modes at the sample-bulk eigenvalues. (c) Per-sample resolution: the sample-bulk eigenvalue attached to training point mu is
Lambda_mu proportional to a_*(q_mu)^2 with q_mu = ||x_t^mu||^2/d_lat (state this precisely from the Gaussian-equivalence structure, and its
distribution over mu computable from the encoded latents), so the predicted memorized FRACTION at step s is the fraction of mu with
Lambda_mu s exceeding the absorption threshold -- combine with Piece 5's quantile-of-d_NN to give a two-source fraction curve. (d) State the
honest parameter count of the resulting predictor (what is derived, what is calibrated: kappa as a unit conversion, one threshold) versus the
released 1 + 6. (e) Give the leave-one-d-out / cross-dataset test that would validate it. Write it so a practitioner can implement it from the
frozen VAE + 1k latents.`,
  },
  {
    key: 'P7-real-data-continuous-spectrum',
    prompt: `PIECE 7: Real-data extension -- continuous spectra, effective dimension, and posterior collapse.
Using F9 and F7: real VAE latent spectra are not two-atom. (a) Restate the four-bulk result for a general empirical spectral measure rho_Sigma of the
latent covariance (this is exactly Bonnaire's Thm 3.1 setting): the "noise-dim bulk" becomes the part of the spectrum near the diffusion floor,
and the quantity that replaces the count d_lat - d_int is an effective excess dimension -- define it (e.g. d_eff = (tr M_t)^2/tr(M_t^2) participation
ratio, and/or the count of eigenvalues above Delta_t(1+delta)), and define the buffer ratio in terms of the spectral density. (b) Posterior collapse:
show that a coordinate with zero data variance has lambda = Delta_t exactly and sits ON the floor; it is still a live coordinate for the score model
(it is noised) but contributes a_1(q)^2 Delta_t/d_lat to the spectrum -- derive what a block of n_dead such coordinates does to q, to a_1(q), to the
data-side edges and to tau_gen. (c) Confront the measured facts: effective rank ~ 0.45-0.55 x d; CelebA effective rank saturates at ~88-90 for d >= 160
while memorization keeps falling; q falls 2.92 -> 1.30. State what the theory predicts should happen to tau_gen, tau_mem^{onset}, and the ratio when
nominal d grows but effective rank is flat -- and therefore which part of the observed memorization decline over d = 160..200 the buffer mechanism
CAN and CANNOT account for (be blunt: if effective dimension is flat, added coordinates change q and the floor block, not the buffer count).
(d) State the corrected x-axis recommendation: report d_eff alongside d_lat. Status: sketched.`,
  },
  {
    key: 'P8-regime-boundaries',
    prompt: `PIECE 8: Regime boundaries and scope -- where the theory applies and what it predicts at the edges.
Using F13, F6, F1: write a Proposition/Remark set covering: (a) d_lat -> n: the sample bulk has n - d_lat modes, so it vanishes as d_lat -> n and the
"population vs sample" distinction dissolves -- derive what happens to tau_mem^{onset} and explain why memorization empirically RETURNS at large d_lat
(the d >> n regime: with n points in d_lat dims the empirical covariance is rank-deficient and every learned direction is partly sample-specific);
give the crossover scale in terms of psi_n = n/d_lat and predict that the rebound point scales with n (testable). (b) sig_perp -> 0: noise-dim bulk
sits at a_1(q)^2 Delta_t/d_lat; derive the merge condition with the sample bulk and state that at sig_perp = 0.01, t = 0.01 the two bulks are NOT
separated (measured gap 0.004-0.056 decades) so the four-bulk picture applies only above a stated sig_perp threshold. (c) psi_p -> 1 (p = d_lat + n + 300):
what breaks and what survives (edges are p-independent per F2, so the spectrum is fine; rank-null tail shrinks). (d) The fixed-null-energy control
sig_perp^2 = c/(d_lat - d_int): derive the predicted edges and show what the theory says this control tests. (e) Feature learning: the theory is a
frozen-feature result; state the one-gradient-step spiked-W extension (Ba et al. 2022) as a Conjecture with its two concrete predictions (signal bulk
moves up, noise-dim stays; effective buffer shrinks) and note it is checkable on eigenvalues_pre/post.npy already saved. (f) List explicitly which
experimental configurations in the paper fall OUTSIDE the stated hypotheses. Status: mixed; label each.`,
  },
]

phase('Derive')
log(`Deriving ${PIECES.length} theory pieces in parallel`)

const results = await pipeline(
  PIECES,
  (pc) => agent(CONTEXT + '\n\n' + pc.prompt + '\n\nReturn the structured derivation.', {
    label: `derive:${pc.key}`, phase: 'Derive', schema: DERIVE_SCHEMA, effort: 'high',
  }),
  async (draft, pc) => {
    if (!draft) return null
    const body = JSON.stringify(draft, null, 1)
    const [math, empirical] = await parallel([
      () => agent(CONTEXT + `\n\nYou are a hostile mathematical referee. Below is a theory piece for this paper. Check every algebraic step,
every hidden assumption, every normalization (the trace check in F2 is mandatory for any edge), every limit statement, and whether each proof actually
proves its statement. Redo key algebra yourself. Default to FIXABLE or UNSOUND unless you have verified it. Be specific: quote the line, say what is wrong,
say exactly what to change.\n\nPIECE:\n${body}`, { label: `verify-math:${pc.key}`, phase: 'Verify', schema: VERIFY_SCHEMA, effort: 'high' }),
      () => agent(CONTEXT + `\n\nYou are a hostile empirical/consistency referee. Below is a theory piece for this paper. Check: does every stated
d_lat-dependence agree with the ESTABLISHED FACTS F1-F15 (which are ground truth)? Does anything reproduce a falsified claim (count-times-timescale bound,
tau_gen d_lat-independent, isotropy premise, wrong bulk counts, /psi_p edges)? Are the empirical anchors the right numbers? Are the predictions actually
testable with the repo's data (${ROOT}) -- go look at what files exist? Does the piece overclaim its status (proved/sketched/conjectured)? Would a referee
who has read Bonnaire 2025 and George-Veiga-Macris 2025 accept the positioning? Be specific and give exact fixes.\n\nPIECE:\n${body}`,
        { label: `verify-empirical:${pc.key}`, phase: 'Verify', schema: VERIFY_SCHEMA, effort: 'high' }),
    ])
    return { draft, math, empirical }
  },
  (v, pc) => {
    if (!v) return null
    const fb = (label, r) => r ? `### ${label}: verdict=${r.verdict}\nMUST CHANGE: ${r.must_change}\nINDEPENDENT CHECK: ${r.independent_check}\n` +
      r.issues.map(i => `- [${i.severity}] ${i.where}: ${i.problem}\n  FIX: ${i.fix}`).join('\n') : `### ${label}: (verifier failed)`
    return agent(CONTEXT + `\n\nRevise the theory piece below in light of two referee reports. Fix every fatal and major issue; fix minors where cheap.
If a referee is wrong, say so in open_gaps and keep the original. Downgrade status honestly if a proof does not close. Keep labels stable.
Return the full revised structured derivation.\n\nORIGINAL PIECE:\n${JSON.stringify(v.draft, null, 1)}\n\n${fb('MATH REFEREE', v.math)}\n\n${fb('EMPIRICAL REFEREE', v.empirical)}`,
      { label: `revise:${pc.key}`, phase: 'Revise', schema: DERIVE_SCHEMA, effort: 'high' })
  }
)

const pieces = results.filter(Boolean)
log(`${pieces.length}/${PIECES.length} pieces survived derive->verify->revise`)


phase('Assemble')

const PIECES_FILE = `${ROOT}/_next_steps/theory_pieces_raw.md`
const SCRATCH = '/private/tmp/claude-502/-Users-ryan-Desktop-latent-space-diffusion-analysis/c16a2edd-ada1-4528-a0b2-279e757294b9/scratchpad'

const WRITE_RULES = `
FILE-WRITING RULES (this stage OVERRIDES the "do not write files" instruction above; you MUST write to disk):
- The eight verified pieces are on disk at ${PIECES_FILE} (markdown; each piece has "Main-text LaTeX (compact)" and
  "Appendix LaTeX (full)" fenced blocks plus statements/proofs). Read it with Read/grep; do not ask for it in the prompt.
- A single model response is capped at 64k output tokens and you WILL hit it if you emit a whole file at once. Therefore:
  write the file in CHUNKS. First Write call = preamble + first section (<= ~150 lines). Then append every further section with
  Bash: cat >> <file> <<'EOF' ... EOF  (or the Edit tool). Never put more than ~8k tokens in one response.
- Use only the macros and theorem environments declared in ${ROOT}/ICLR_2026/main.tex (read its preamble first).
- Do NOT edit main.tex. Do not touch any other file. Disk is nearly full: do not copy large directories.
- Labels: keep the piece labels (they are already unique per piece). If two pieces define the same object, keep one and \\cref the other.
- When done, return the SHORT structured summary only (no LaTeX in the return value).`

const MAIN_SCHEMA = { type: 'object', properties: {
  path: { type: 'string' }, line_count: { type: 'integer' },
  statements_included: { type: 'array', items: { type: 'string' } },
  cref_targets_used: { type: 'array', items: { type: 'string' } },
  notes: { type: 'string' } }, required: ['path', 'line_count', 'statements_included', 'cref_targets_used', 'notes'] }

const META_SCHEMA = { type: 'object', properties: {
  bib_path: { type: 'string' }, prose_edits_path: { type: 'string' },
  what_we_should_be_doing: { type: 'string' }, unresolved: { type: 'string' } },
  required: ['bib_path', 'prose_edits_path', 'what_we_should_be_doing', 'unresolved'] }

log('Assembling: main text, appendix, and meta files in parallel (chunked disk writes)')

const [mainRes, appRes, metaRes] = await parallel([
  () => agent(CONTEXT + WRITE_RULES + `

TASK: write the complete main-text Section 2 to ${ROOT}/ICLR_2026/sec-theory.tex (OVERWRITE it; it is currently a 9-line \\todo stub with nothing to lose).
Structure, in order: \\section{Theory: spectral buffer mechanism}\\label{sec:theory}; setup + standing assumptions (compact);
Definitions 1-2 (tau_gen, tau_mem, onset) from Piece 1; the four-bulk Proposition from Piece 2 with the edge table (explicit d_lat dependence,
trace check stated in one line) and a 3-4 sentence positioning paragraph vs Bonnaire Thm 3.1 (two-atom rho_Sigma) and George-Veiga-Macris;
the buffer Corollary as an edge ratio from Piece 3 (+ one Remark that the old count-times-timescale form is a non-sequitur, one that the claim is
about onset not median); the linear-model Lemma from Piece 4 as the rigorous anchor (one display); the bridge Proposition from Piece 5 (statement
+ the two consequences, proof deferred); one paragraph on the predictor (Piece 6: what changes vs released); one paragraph on real data (Piece 7:
d_eff, collapse, what the buffer can and cannot explain over d=160..200); one paragraph on regime boundaries (Piece 8); a 4-line scope paragraph
naming what is NOT covered (trainable features, curved manifolds, DiT). Each formal statement gets one sentence of "what it buys us".
Every proof is \\cref'd to the appendix labels used in ${ROOT}/ICLR_2026/sec-appendix-theory.tex (another agent is writing it in parallel from the
same pieces; use the piece labels as-is so the crefs resolve). Target 1.5-3 pages of ICML two-column (~180-300 lines of LaTeX). Do not reproduce
any falsified claim (F4, F2's /psi_p, "tau_gen d_lat-independent", the isotropy premise).`,
    { label: 'assemble:main-text', phase: 'Assemble', schema: MAIN_SCHEMA, effort: 'high' }),

  () => agent(CONTEXT + WRITE_RULES + `

TASK: write the complete theory appendix to ${ROOT}/ICLR_2026/sec-appendix-theory.tex (new file). It is \\input after
sec-appendix-integrated.tex inside \\appendix\\onecolumn, so start with \\section{Theory: proofs and derivations}\\label{app:theory} and use
\\subsection per piece in order 1..8. For each piece: its full statements with proofs (the "Appendix LaTeX (full)" block, cleaned), the
predicted-vs-measured table where the piece has one (Pieces 2, 3, 7), the predictions list as a short itemize with how-to-test, and an
"Open gaps" paragraph reproducing the piece's honest gaps. Add a final \\subsection{Configurations outside the hypotheses} from Piece 8.
Unify notation to the macros. Keep labels exactly as in the pieces. This file will be long (600-1200 lines) -- write it in at least 8 chunks,
one per piece, appending with cat >> ... <<'EOF'.`,
    { label: 'assemble:appendix', phase: 'Assemble', schema: MAIN_SCHEMA, effort: 'high' }),

  () => agent(CONTEXT + WRITE_RULES + `

TASK (no LaTeX section writing): produce three things.
(1) Write ${ROOT}/_next_steps/theory_bib_additions.bib: full BibTeX entries for EVERY citation key that appears in the pieces' "Citations needed"
lists or in their LaTeX and is NOT already in ${ROOT}/ICLR_2026/references.bib (grep it). Use the keys the pieces used. Include at minimum
George-Veiga-Macris 2502.00336, Hu-Lu 2009.07669, Mei-Montanari 1908.05355, Goldt et al. 2006.14709, Ba et al. 2205.01445, BBP 2005,
Louart-Liao-Couillet 2018, Damian-Lee-Soltanolkotabi 2206.15144 if referenced. Mark any arXiv id you are not certain of with % VERIFY.
(2) Write ${ROOT}/_next_steps/theory_prose_edits.md: a table of every sentence elsewhere in the draft (abstract.tex, sec-intro.tex, sec-rfnn.tex,
sec-mlp.tex, sec-spectral-predictor.tex, sec-discussion.tex, sec-appendix-integrated.tex, PROJECT_BRIEF.md) that must change to agree with the new
theory: file:line | current text (quoted) | replacement text | why. Go read the files; do not guess line numbers.
(3) Return what_we_should_be_doing: a plain-language summary for the authors (no LaTeX): what the theory now claims, what it explicitly does not
claim, and the ordered list of experiments it demands, marking which can run today on the HF checkpoints
(https://huggingface.co/trevorbchen/diffusion_memorization: 157 last_model.pt + 39 VAEs) or saved spectra, vs need a rerun.`,
    { label: 'assemble:meta', phase: 'Assemble', schema: META_SCHEMA, effort: 'high' }),
])

const critic = await agent(CONTEXT + `

You are the consistency critic AND the build check. Two files were just written: ${ROOT}/ICLR_2026/sec-theory.tex and
${ROOT}/ICLR_2026/sec-appendix-theory.tex. Read both from disk.
(a) BUILD: disk is nearly full, so do NOT cp -R the whole ICLR_2026 directory (it has 22MB of figures). Instead: mkdir -p ${SCRATCH}/build_check,
copy ONLY *.tex *.sty *.cls *.bst *.bib *.bbl into it, and symlink the figures dir (ln -s ${ROOT}/ICLR_2026/figures ${SCRATCH}/build_check/figures).
In THAT COPY ONLY insert \\input{sec-appendix-theory} on the line after \\input{sec-appendix-integrated} in main.tex, append
${ROOT}/_next_steps/theory_bib_additions.bib to the copy's references.bib, then run pdflatex -interaction=nonstopmode main (twice). Report every
LaTeX error and every "undefined reference"/"multiply defined label"/"undefined citation" warning verbatim with file/line. Do not modify the real
ICLR_2026 directory. Delete build_check when done.
(b) CONSISTENCY: label collisions between the two files; any statement contradicting F1-F15; any surviving falsified claim; any edge without
explicit d_lat dependence; any main-text statement whose appendix proof does not match; missing hypotheses; overclaimed status; a sentence a
Bonnaire/George co-author would object to.
Return a terse, actionable list. If the build passes, say so with the page count of the resulting PDF.`,
  { label: 'critic:build+consistency', phase: 'Assemble', effort: 'high' })

return { main: mainRes, appendix: appRes, meta: metaRes, critic, pieces_survived: pieces.length, pieces_total: PIECES.length }
