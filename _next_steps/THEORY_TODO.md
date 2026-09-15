# Theory TODO — everything that has to be done, with sources

Written 2026-09-12. Self-contained: every claim points at a file, a line, a URL, or a number you can check.
Companion file: `_next_steps/theory_pieces_raw.md` (the seven adversarially-checked derivations, with LaTeX).

---

## 0. Where things stand (one paragraph)

The submitted draft has **no theory**. `ICLR_2026/sec-theory.tex` is a 9-line `\todo{}` and it *prints*
(magenta, page 2 of `main.pdf`). `sec-appendix-integrated.tex` (the appendix `main.tex` actually inputs) has zero
theorem environments. The theorem/lemmas/buffer-bound live in `ICLR_2026/sec-appendix.tex` (byte-identical to
`ICML (1)/sec-appendix.tex`) and root `sec-appendix-fourbulk.tex` / `sec-rfnn-bounds.tex` — none of which is
`\input`. Those drafts are also wrong in several load-bearing places (Section 2 below). The empirical phenomenon
(memorization fraction falls with `d_lat` in the fixed-width MLP) is real; the *explanation* is not established.

Full audit with 157 findings: https://claude.ai/code/artifact/a8e70925-877b-45ff-ae73-fce67e37de8d

---

## 1. Housekeeping that blocks theory work

| # | Task | Why | Source |
|---|------|-----|--------|
| 1.1 | **Commit the transpose fix** (currently uncommitted). | `precompute_sigma_t` computed `Qᵀ D⁻¹ Q`; correct is `Q D⁻¹ Qᵀ` because `data = data @ Q.T`. Every saved synthetic `score_error` is wrong; the "n-shape" (`n_shape_heuristic_derivation.md`) is 100% this bug. Inert at `d_lat = d_int`. | `code/v3/lib/true_score.py:63`, `code/experiment_v2.py:215`, `experiment_v2.py:215`, `figures/code/experiment_v2.py:215`; regression test `code/v3/tests/test_true_score.py` (Stein's identity `E[s(x)xᵀ] = −I`; fails 4/5 cases on old code). |
| 1.2 | Rerun the synthetic score-error sweeps, or correct existing numbers by subtracting the model-free mismatch `‖s*_true − s*_bug‖²/d`. | Audit showed `observed − mismatch` = +0.008…+0.073 (flat ~0.05/dim), i.e. the cross-term is small, so subtraction is defensible. Training, memorization, FID, spectra are **unaffected** (DSM loss targets the noise; `nn_ratio_memorization` is pure geometry). | `code/v3/lib/training.py:125`, `code/v3/lib/metrics.py:39-46,61-84` |
| 1.3 | Reconcile the three appendix copies, pick `ICLR_2026/sec-appendix.tex` as canonical, then decide what to keep. | Root `sec-appendix-fourbulk.tex` replaced the μ₂ argument with μ_{k≥3}; the ICLR copy still carries the old "Four-bulk intuition" prose the replacement was meant to delete. | `diff` them. |
| 1.4 | Fix `bulk_indices()` — it hard-codes the sample bulk as `slice(d_latent, d_latent+n)`. Correct: sample `n − d_lat`, rank-null `p − n`. | `rank(U) ≤ n` for a single noise draw; machine-precision cliff at index `n=500` in `eigenvalues_pre.npy`, never at `d_lat+n`. Regenerate Table 3 + bulk overlays. | `code/v3/lib/eigenvalues.py:75` |
| 1.5 | Fix two paper/code mismatches. | Paper says **500** MC noise samples for `U` (`sec-rfnn.tex:38`, `sec-appendix-integrated.tex:32`); code hard-codes **50** (`eigenvalues.py:27`, `run_experiment.py:156`, `experiment_v2_rfnn.py:58`). Appendix claims fixed-width sweep to `d_lat=240` (`sec-appendix-integrated.tex:313,329`); `multiseed_runs/*` stop at 40; `d ∈ {120,140,160,180,220,240}` exist nowhere. | grep the files. |
| 1.6 | Resolve the real-data architecture. | HF README: hidden=1024, depth=5, 10k eval samples; paper (`sec-appendix-integrated.tex:93-99`): hidden 256, depth 3, 1k samples. Run dirs are named `..._bigmlp_..._10k_5m`. | https://huggingface.co/trevorbchen/diffusion_memorization |
| 1.7 | Make the HF repo private. | README says "Status: private"; API returns `private: false`. Under review, under Trevor's name. | same URL |

---

## 2. What is wrong in the existing theory (must not be carried forward)

All file refs are to `sec-appendix-fourbulk.tex` unless noted; identical text is in `ICLR_2026/sec-appendix.tex` and `sec-rfnn-bounds.tex`.

| # | Error | Evidence | Fix |
|---|-------|----------|-----|
| 2.1 | **Edge table divides by `ψ_p = p/d_lat`.** Should be `/d_lat` for the two data bulks and `~1/n` for the sample bulk. | Nonzero eigenvalues of `W M_t Wᵀ` = those of `M_t^{1/2} WᵀW M_t^{1/2}`, `E[WᵀW] = p·I`, so the `p` cancels the `1/(p·d_lat)` prefactor. Trace: `tr(U) = O(1)` by construction, but the published table sums to `~ d_lat·μ₁²β_t²/ψ_p` → diverges. Simulation: at `d_lat=80`, `p=5120` vs `p=880` (5.8× change in `ψ_p`) moves the signal edge 1.737e-2 → 1.744e-2. **Edges are p-independent.** | Corrected table in §4.2. Scripts: scratchpad `rfnn_spectrum.py`, `rfnn_controls.py` (see §9). |
| 2.2 | **`μ₁ = 0.606` is not a constant.** It's the Hermite coefficient at unit variance; actual pre-activation variance `q = tr(M_t)/d_lat` falls 2.76 → 0.33 across the σ⊥=0.5 sweep. | Measured signal-bulk medians (code normalization = p× theory): 29.7 (d=5) → 99.8 (d=200). Fixed-μ₁ predicts flat 64.9. `a₁(q)² α_t² ψ_p` with `a₁(q)=E[tanh′(√q z)]` predicts 31.9 → 110.6. Noise-dim: measured 5.22 → 9.22, predicted 4.76 → 10.60, fixed-μ₁ flat 6.22. | Use Gaussian-equivalent coefficients `a₁(q)`, `a⋆(q)² = E[tanh(√q z)²] − a₁(q)² q`. Cite Hu–Lu, Goldt et al., Mei–Montanari. |
| 2.3 | **Buffer bound `τ_mem − τ_gen ≥ (d_lat − d_int)·ψ_p/(μ₁²β_t²)` is a non-sequitur.** | Eq. (rfnn-mode-decay) is a diagonal linear ODE; modes decay in parallel; no count enters a traversal time. Numerically violated at every `d_lat ≥ 20` on the repo's own spectra (onset statistic, same units): d=20 1.37e5 vs 9.5e4; d=40 5.4e5 vs 3.4e5; d=100 3.26e6 vs 1.61e6. | Withdraw. Restate as ratio of edges (§4.3). Kill the coffee-shop-queue analogy in `PROJECT_BRIEF.md`. |
| 2.4 | **"τ_gen is d_lat-independent" is false.** | On `sigma_noise_0.5/exp2_rfnn/raw_data/*/eigenvalues_pre.npy`: `τ_gen = 1/λ_min^signal` grows 11.0× in flow units (λ_code/p) over d_lat 5→200; in optimizer steps it falls because lr ∝ d_lat. (Corrected 2026-09-15; the earlier "788 → 6413, 8.1×" was mislabeled and does not reproduce.) | Replace with "τ_gen rises with d_lat as tanh de-saturates; τ_mem^onset rises faster; the ratio opens." |
| 2.5 | **Bulk counts wrong:** sample bulk is `n − d_lat`, not `n`; rank-null `p − n`, not `p − d_lat − n`. `U` is a sum of `n` rank-one terms. | Single-draw sim at d=20,n=500,p=1280: `λ[499]=2.4e-8`, `λ[500]=1.1e-17`. Saved spectra: largest log-drop at index 499 for d=40,100,200; ratio at `d_lat+n` is 1.001–1.009. Paper's own Table 3 (`ICLR_2026/sec-appendix.tex:163-192`) contradicts the theorem: d=20 predicted (760,500,15,5) observed (916,311,48,5). | Note: the paper's ξ-averaged `U` (50 draws) has a *soft* cliff at n; state the proposition for the averaged object and remark on the single-copy one. |
| 2.6 | **Lemma 2 contradicts its own proof.** Statement: `η̄⋆/ψ_p`. Proof line: `η⋆²/(pn)`. Eq. (U-diag) already has `1/(pn)` and the proof re-normalizes. Orthonormality `v_μᵀv_ν = δ + o(1)` is false for clustered data (50 pts/cluster). `η⋆ = Σ_{k≥3} μ_k²(‖x‖²/d)^k` is a divergent series. | Read the lemma and proof side by side. | Rewrite with `a⋆(q)²`; the sample bulk is an MP-type bulk of residual features with total mass `a⋆(q)²`. |
| 2.7 | **Novelty premise is false.** "Bonnaire assumes isotropic covariance" (`sec-intro.tex:17`, `sec-rfnn.tex:46`). | Bonnaire et al. Thm 3.1 integrates against arbitrary `ρ_Σ`; their SM derives the GEP "under arbitrary input covariance." George, Veiga & Macris (arXiv:2502.00336) already have a linear pencil with block dims `(p, d−D, D, n)` for data on a D-dim subspace (σ⊥=0). Neither is cited correctly; George isn't cited at all (`references.bib` "george" hit is Danezis on LOGAN). | Restate contribution 1 as *evaluation* of Thm 3.1 at a two-atom `ρ_Σ`. Novelty that survives: strictly positive σ⊥, the gradient-flow/training-time reading, and everything downstream (bridge, predictor). Optionally solve their Eqs (17)–(19) numerically at the two-atom measure and overlay on the spectrum — half a day, strongest possible version. |
| 2.8 | **Isotropic control reproduces the "signature."** `sec-rfnn.tex:65`: "The buffer does not exist in Bonnaire's isotropic case." | At σ⊥=σ_sig=1 the ratio λ_sig_min/λ_samp_max is NON-monotone: 14, 30, 58, 87, 77, 60 at d=5,10,20,40,80,160 (E5, 3 seeds, 2026-09-15). The earlier "gap grows 10.7×" was a code-unit artifact (factor p=64d); in theory units those values rise then fall. Under isotropy the noise-dim modes still exist between signal and sample; anisotropy adds spectral *separation*, not existence. | Say exactly that. The growing gap is not evidence for anisotropy; the separation is. |
| 2.9 | **τ_gen/τ_mem never defined** as functionals of the spectrum. Three inconsistent empirical proxies in use (NN-ratio fraction >1%; gen-gap >0.02; fitted-sigmoid hitting time). | `ICLR_2026/sec-appendix.tex:417`, `ICML (1)/sec-appendix.tex:30`, `sec-spectral-predictor.tex:17`. | §4.1. |
| 2.10 | **Spectral predictor has no sample-bulk term** and a single sample edge predicts a step function, not a fraction curve. Released code fits 1 κ + 6 monotone θ levels on *all* d (no identity-threshold filter, no holdout); paper says 3 params on `d ≥ 70/140`. Main-text figure `real_spectral_predictor_compact_2x2.pdf` is generated by no script in the repo. | `make_celeba_spectral_mem_predictor_figures.py:calibrate`, `calibration.json`; `sec-spectral-predictor.tex:17`; `sec-appendix-integrated.tex:332-334`. | §4.6. |
| 2.11 | Old `_next_steps/theory_plan.md` asks for the Hermite-2 (μ₂) term for the sample bulk. tanh is odd ⇒ μ₂=0; the appendix's μ₀=0 argument is correct. Moot anyway after 2.6. | — | Don't chase μ₂. |

---

## 3. What survives (build on this)

- **Four-bulk ordering** signal > noise-dim > sample > rank-null holds in every simulated config (3 seeds, 3 width controls, both σ⊥).
- **The buffer opens on the onset statistic.** `τ_mem^onset = 1/λ_max^sample`, `τ_gen = 1/λ_min^signal`: ratio grows monotonically 21 → 426 (20×) over d_lat 5→200 at σ⊥=0.5 (recomputed 2026-09-15; the earlier 31→576 does not reproduce). (On the bulk *median* it shrinks ~14× — so the claim is specifically about onset. Say so.)
- **Fixed-width MLP memorization falls monotonically** 0.309 → 0.0016 over d_lat 5→40, 5 seeds (`multiseed_runs/exp2_mlp_dlat_sn05_5m_bonnaire_hidden256`). Not width-robust: hidden=8·d_lat sweep (`..._combined`) has a minimum at d=20 and a 3.3× rebound by 40.
- **Real-data quality/memory tradeoff** (VAE + FID) is untouched by any of this.
- The **fraction-over-training** metric choice is right and already argued (`sec-appendix-integrated.tex:195-201`). Main text still plots endpoints; promote a trajectory.

---

## 4. What to write

Target: Section 2 ≈ 1.5–3 pages ICML two-column for ICLR main (≈1 column-page for a workshop) + a theory appendix
`sec-appendix-theory.tex`. Pay for main-text space by moving the mechanism prose out of `sec-rfnn.tex`
("Mechanism: a noise-dimension buffer", "What the gen-gap measures", "Decomposing the four-bulk structure", ~35 lines).
Draft LaTeX for every item below is in `theory_pieces_raw.md`.

### 4.1 Setup + Definitions (piece P1)
- Assumption block: `p > n`; operating-point condition in terms of `q` (not `s/√d_lat`); fixed `t`; `d_int` fixed so the signal block is a finite-rank spike (BBP), not a bulk; state which `ψ_n = n/d_lat` regime (the sweep varies it 40×).
- `E(T) = Σ_i π_i e^{−2λ_i T}`, `π_i = ‖P_i A*‖²_F` — exact for the RFNN.
- **Def 1** `τ_gen(ε) = inf{T : E_pop(T) ≤ (1+ε)E_pop(∞)}` ⇒ `τ_gen = (1/(2λ_min^signal)) log(1/ε) + O(1)`. Proxy: test loss within 5% of its min.
- **Def 2** `τ_mem` via spectral gen-gap `G(T) = Σ_{i∈sample} π_i(1−e^{−λ_iT})²(1−overlap_i)` (proxy: gen_gap > 0.02) **and** `τ_mem^onset = 1/λ_max^sample`. State that the paper measures a *fraction* and a fraction-level definition needs the within-bulk spread (4.6).

### 4.2 Four-bulk proposition, corrected (piece P2) — status: sketched
| bulk | count | edge (explicit d_lat dependence) |
|---|---|---|
| signal | `d_int` | `a₁(q)² α_t² / d_lat` |
| noise-dim | `d_lat − d_int` | `a₁(q)² β_t² / d_lat` |
| sample | `n − d_lat` | MP-type bulk of residual features, total mass `a⋆(q)²`, mean `~a⋆(q)²/n` |
| rank-null | `p − n` | `o(1)` |

with `q = [e^{−2t}(s² + d_int σ_sig² + (d_lat−d_int)σ⊥²) + Δ_t d_lat]/d_lat`. **Trace check (mandatory):**
`d_int·signal + (d_lat−d_int)·noise + (n−d_lat)·sample_mean ≈ a₁(q)²q + a⋆(q)² = E[tanh(√q z)²] = O(1)`.
Proof route: Gaussian equivalence `φ(x) ≈ a₁(q)·Wx/√d_lat + a⋆(q)·θ`, θ iid ⇒ `U ≈ (a₁²/(p d_lat))W M_t Wᵀ + (a⋆²/(pn))ΘᵀΘ` + cross terms.
Separation conditions: signal/noise-dim needs `α_t²/β_t²` vs MP spread with ratio **`d_lat/n`** (not p); noise-dim/sample needs `a₁(q)²β_t²/d_lat` vs top MP edge of sample bulk. Isotropic limit: noise-dim merges with signal into Bonnaire's ρ₂.
Include the predicted-vs-measured table from 2.2 (it's the validation the old draft *claims* but never shows).

### 4.3 Buffer corollary restated (piece P3)
`buffer := λ_min^signal / λ_max^sample`. No mode count. Derive `τ_gen(d_lat) ~ d_lat/(2a₁(q(d_lat))²α_t²)·log(1/ε)` and `τ_mem^onset(d_lat)`, hence the ratio and why it opens (tanh de-saturation dominates). Remarks: why count×timescale is a non-sequitur; why the median statistic reverses; what isotropy does/doesn't show; **what the theory does not predict** (final fraction at fixed budget; requires 4.6 + matched quality).
Open gap (labeled): at `d_lat ≤ 50` the sample-block top exceeds the MP edge by 4–6.5× on clustered data — no derivation yet (cluster-spike/BBP effect, probably).

### 4.4 Exactly solvable linear score model (piece P4) — status: proved
Null block target is *exactly* linear: `s*_null = −x_null/β_t²` (cluster means have zero null component). For `s_B(x)=Bx` under gradient flow on `E‖Bx − s*‖²`: `B(T) = −M_t⁻¹ + (B₀ + M_t⁻¹)e^{−2M_t T}` (**note: uncentered second moment `M_t`, not `Σ_t`** — referee correction). Per-mode risk `e^{−4λ_iT}/λ_i` ⇒ mode clock `τ_i = 1/(4λ_i)` rigorously; low-λ modes carry the *largest* target energy `1/λ_i`, so the noise-dim block **dominates** the late residual rather than merely delaying. RFNN corollary by `M_t → U`.

### 4.5 Bridge lemma: spectrum → near-duplicate generation (piece P5) — status: sketched
Exact empirical score `s_emp(x,t) = Σ_μ w_μ(x)(e^{−t}x^μ − x)/Δ_t`, softmax weights; memorizes when `w` concentrates: `Δ_t < d_NN²/(2 log n)` (Biroli et al. 2024 condensation). Partially trained score ≈ exact empirical score at effective noise `Δ_eff(T) = Δ_t + c·E(T)` ⇒
`τ_mem = inf{T : E(T) ≤ [d_NN²/(2 log n) − Δ_t]/c}`.
Consequences: (a) Somepalli 1/3 threshold ⇒ relevant `d_NN` for "fraction crossing q" is the q-quantile of the training NN-distance distribution — this turns a single edge into a **fraction curve**; (b) `τ_mem` depends on `n`, `d_int` via `d_NN ~ n^{−1/d_int}` — a **d_int not d_lat** dependence, sharply testable; (c) pixel vs latent: decoder Lipschitz constant.
Missing for a full proof: isotropic-vs-structured residual (G1 in the piece's gaps).

### 4.6 Predictor derived, with a sample-bulk term and per-sample resolution (piece P6)
Absorbed fraction `1−e^{−λT}`; absorbed squared error `1−e^{−2λT}`; the released `(1−e^{−κλs})²` is neither (say what it would take). Add `n−d_lat` sample-bulk modes. Per-sample: `Λ_μ ∝ r_t(q_μ)·a⋆(q_μ)²/n`, `q_μ = ‖x_t^μ‖²/d_lat` (noise-averaged; referee correction) ⇒ predicted memorized fraction at step s = fraction of μ with `Λ_μ s` past threshold; combine with 4.5(a). Honest parameter count: κ (unit conversion) + one threshold, vs released 1+6. Validation: leave-one-d-out and CelebA→CIFAR transfer.

### 4.7 Real data: continuous spectra, effective dimension, posterior collapse (piece P7)
Real latent spectra are not two-atom. Restate for general `ρ_Σ` (Bonnaire's setting): noise-dim bulk = part of spectrum near the floor; replace the count `d_lat − d_int` with `d_eff` (participation ratio `(tr M_t)²/tr(M_t²)` and/or count above `Δ_t(1+δ)`). Dead coordinate ⇒ `λ = Δ_t` exactly, still noised, contributes `a₁(q)²Δ_t/d_lat`.
**Measured** (`clean figures/*/`*`_spectral_features_primary_t.csv`, t=0.1, n=1000): effective rank ≈ 0.45–0.55 × nominal everywhere; `λ_min = 0.181 = Δ_t` for `d ≥ 140` (CelebA) / `≥ 160` (CIFAR) ⇒ zero data variance; **CelebA effective rank saturates** 89.5 (d=160), 90.3 (180), 87.9 (200) while memorization keeps falling; `q` falls 2.92 → 1.30.
Be blunt in the text: if `d_eff` is flat over d=160..200, added coordinates change `q` and the floor block, **not** the buffer count, so the buffer mechanism cannot account for the decline in that range. Report `d_eff` on every real-data x-axis.

### 4.8 Regime boundaries and scope (piece P8)
- `d_lat → n`: sample bulk `n − d_lat` vanishes; population/sample distinction dissolves; memorization **returns** (seen in hidden=8·d_lat sweep at d=100–200). Predict rebound point scales with `n` (testable).
- `σ⊥ → 0`: noise-dim bulk on the floor `a₁(q)²Δ_t/d_lat`; at σ⊥=0.01, t=0.01 the noise-dim and sample bulks are **not separated** (measured gaps 0.004–0.056 dec) — four-bulk picture applies only above a stated σ⊥ threshold.
- `ψ_p → 1` (p = d_lat+n+300): edges p-independent so fine; rank-null tail shrinks; violates the old theorem's `ψ_p > 1+ψ_n`.
- Fixed-null-energy control `σ⊥² = c/(d_lat−d_int)`: derive predicted edges; say what it tests.
- Feature learning: frozen-feature result; one-gradient-step spiked-W extension (Ba et al. 2022) as a Conjecture with two predictions (signal bulk up, noise-dim fixed; effective buffer shrinks) — checkable on `eigenvalues_pre.npy` vs `eigenvalues_post.npy` already saved in every RFNN run dir.
- List configurations outside hypotheses (`d_lat=5` violates `p>d_lat+n` at p=64d; ψ_n not fixed; etc.).
Open gap: isotropic-residual onset formula underestimates measured sample-bulk top by 3–10× on clustered data.

---

## 5. Experiments the theory demands (ordered; what can run today)

| # | Experiment | Discriminates | Can run now? | Where |
|---|-----------|---------------|--------------|-------|
| E1 | **Posterior-collapse probe**: load each VAE, encode 1k subset, per-coordinate variance + KL per dim → `d_eff` vs nominal. | Whether the real-data x-axis is real. | **Yes** — all 39 VAE checkpoints on HF. `huggingface-cli download trevorbchen/diffusion_memorization --include "vae_checkpoints/*"` | `standard_vae.py`, `train_vae_celeba_standard_tar.py` for the arch |
| E2 | **Dimension-controlled memorization**: NN-ratio after projecting gen+train onto ground-truth `d_int` subspace (synthetic, have `Q`) / top-`d_int` PCA (real). | Whether the fraction decline is distance concentration. `gen_gap` flat 0.82→0.80, train loss falls 0.53→0.20, `mean_nn_ratio` drifts 0.52→0.87. | Yes if generated samples were saved; else regenerate from HF checkpoints (real) or retrain (synthetic). | `code/v3/lib/metrics.py:61` |
| E3 | **Activation sweep** (RFNN only): `τ_mem^onset ∝ 1/a⋆(q)²`; identity (a⋆=0 → no sample bulk), tanh, ReLU, sin, `u²−1`. Compute predicted ratios first, register, then run. | De-saturation/GE story (activation-dependent) vs mode-count story (no activation dependence). | Yes, 1 day. | `code/experiment_v2_rfnn.py` |
| E4 | **Norm-fixed control**: hold `E‖x‖²/d_lat` constant as d_lat grows. None of the 3 existing width controls does (per-coord norm falls 1.52→0.38). | Mode count vs de-saturation. | Yes, RFNN spectra only. | same | **DONE 2026-09-15** (`e4_norm_fixed_results.txt`): R 1.09× at fixed q vs 3.17× baseline over d=40→240. |
| E5 | **Isotropic sweep as a reported control** (σ⊥=σ_sig). | Fixes 2.8 honestly. | Yes, minutes. | same | **DONE 2026-09-15** (`e5_isotropic_results.txt`). |
| E6 | **n-sweep**: rebound point of memorization at large d_lat should move ∝ n; also tests 4.5(b) `d_NN ~ n^{−1/d_int}`. | d≫n story vs sparse-rep story. | Retrain. | — |
| E7 | **Sparse-representation probes** on real-data score nets: activation density, effective rank, per-unit example selectivity (max/mean over training set), first-layer mass on signal vs null via `Q`. | Trevor's "extra dims encode individual examples" vs the earlier plan "project onto signal subspace" (same story, opposite sign). | **Yes** — 157 `last_model.pt` on HF. `probe_ushape.py:255,263` has two of three probes. | HF `results/*/d*/last_model.pt` |
| E8 | **Pre/post spectrum shift** in RFNN runs. | Feature-learning conjecture (4.8). | Yes — files exist. | `sigma_noise_*/exp2_rfnn/raw_data/*/eigenvalues_{pre,post}.npy` |
| E9 | Predictor **leave-one-d-out** / CelebA→CIFAR transfer with the 4.6 form. | Whether "predictor" is earned. | Yes, from saved spectra + curves. | `clean figures/*_spectral_mem_predictor/` |
| E10 | Rerun synthetic score-error sweeps post-fix (or subtract mismatch, 1.2). | What the linear model (4.4) has to explain. | Retrain (no synthetic checkpoints saved). | `code/v3/run_experiment.py` |

---

## 6. Prose edits the theory forces (current → replacement)

| file | current | replacement |
|---|---|---|
| `sec-intro.tex:17` | "Their analysis assumes isotropic data covariance" | delete; Bonnaire Thm 3.1 is for arbitrary ρ_Σ |
| `sec-intro.tex:25` | "We extend Bonnaire's two-bulk theorem" | "We evaluate Bonnaire's Theorem 3.1 at a two-atom spectral measure"; cite George–Veiga–Macris |
| `sec-intro.tex:26` | "delaying τ_mem while leaving τ_gen approximately fixed" | no figure reports either; restate in terms of the fraction actually plotted, or report τ's |
| `sec-rfnn.tex:46` | "prove (replica limit, isotropic Σ_data = I)" | same as row 1 |
| `sec-rfnn.tex:65` | "τ_gen, d_lat-independent…"; "The buffer does not exist in Bonnaire's isotropic case" | "τ_gen rises with d_lat as tanh de-saturates; τ_mem^onset rises faster"; "under isotropy the noise-dim modes exist but merge with the signal bulk; the four-bulk structure is absent, the growing gap is not" |
| `abstract.tex` | "this bulk acts as a buffer" | keep, scoped to onset statistic + fixed-width regime |
| `sec-mlp.tex` fig caption | "rule out the simplest failure mode…" | delete until post-fix; MLP is worse than the zero predictor at every d_lat (obs/trivial 3.56 → 1.06, never < 1) |
| `sec-mlp.tex:41` | "feature learning appears to find sparser, signal-aligned directions" | delete or mark conjecture; no probe supports it |
| `sec-spectral-predictor.tex:17` | "one set of (κ,a,b)" | describe released 1+6 or publish the 3-param script |
| `sec-appendix-integrated.tex:332` | "fit … using the dimensions where the VAE already preserves identity" | false about released code (fits all d) |
| `sec-appendix-integrated.tex:32`, `sec-rfnn.tex:38` | "500 Monte Carlo" | 50 |
| `sec-appendix-integrated.tex:313,329` | sweep to d_lat=240 | only 5–40 exist for that sweep |
| `PROJECT_BRIEF.md` | "no loss in image quality"; "~8×/~15× delay" | contradicts abstract + `sec-real-data`; numbers from a deleted experiment; rewrite or retire |

---

## 7. Citations to add (verify arXiv IDs before submitting)

Present in `references.bib`: bonnaire2025, biroli2024dynamical, scarvelis2023closedform, gu2023memorization, somepalli2023diffusion, kadkhodaie2023learning, pennington2017nonlinear, benigni2021eigenvalue, ELKaroui2010spectrum.

To add:
- **Bonnaire, Urfin, Krzakala, Zdeborová 2025** — *Why diffusion models don't memorize: the role of implicit dynamical regularization in training* — arXiv:2505.17638. (already keyed `bonnaire2025`; fix how it's *described*)
- **George, Veiga & Macris 2025** — *Denoising Score Matching with Random Features: Insights on Diffusion Models from Precise Learning Curves* — arXiv:2502.00336 (AISTATS 2026). Block dims (p, d−D, D, n).
- **Hu & Lu 2022** — *Universality laws for high-dimensional learning with random features* — arXiv:2009.07669. (Gaussian equivalence of the conjugate kernel)
- **Mei & Montanari 2022** — *The generalization error of random features regression: precise asymptotics and the double descent curve* — arXiv:1908.05355.
- **Goldt, Loureiro, Reeves, Krzakala, Mézard, Zdeborová 2022** — *The Gaussian equivalence of generative models for learning with shallow neural networks* — arXiv:2006.14709.
- **Ba, Erdogdu, Suzuki, Wu, Wang, Yang 2022** — *High-dimensional asymptotics of feature learning: how one gradient step improves the representation* — arXiv:2205.01445.
- **Damian, Lee, Soltanolkotabi 2022** — *Neural networks can learn representations with gradient descent* — arXiv:2206.15144.
- **Baik, Ben Arous, Péché 2005** — *Phase transition of the largest eigenvalue for nonnull complex sample covariance matrices* — Ann. Probab. 33(5). (BBP)
- **Louart, Liao, Couillet 2018** — *A random matrix approach to neural networks* — Ann. Appl. Probab. 28(2).
- **Biroli, Bonnaire, de Bortoli, Mézard 2024** — *Dynamical regimes of diffusion models* — arXiv:2402.18491 (already keyed; used for the condensation threshold).
- **Achilli, Ambrogioni, Lucibello, Mézard, Ventura 2024** — *Losing dimensions: geometric memorization in generative diffusion* — arXiv:2410.08727 (check ID). Adjacent work; cite in related work.
- **Scarvelis, Borde, Solomon 2023** — *Closed-form diffusion models* — arXiv:2310.12395 (keyed).
- **Gu et al. 2023** — *On memorization in diffusion models* — arXiv:2310.02664 (keyed).

---

## 8. Open problems (genuinely unsolved — label as such in the paper)

1. Sample-block top edge exceeds MP prediction 4–6.5× at `d_lat ≤ 50` on clustered data (cluster-spike effect?). No derivation.
2. Isotropic-residual onset formula off by 3–10×. Same root cause likely.
3. Gaussian equivalence on real (non-Gaussian) VAE latents — assumed, not proved; per-sample CV of `‖x_t‖²/d` ≈ 0.20.
4. Bridge lemma: additive-isotropic residual assumption vs the structured gradient-flow residual.
5. Feature learning beyond one gradient step.
6. Which of {de-saturation, mode count, d≫n, learned per-example VAE capacity} drives the DiT sign flip (memorization *rises* 22%→48% for `d_lat` 5→10, `sec-appendix-integrated.tex` fig `app-spatial-dit-lowd`).

---

## 9. Sources and pointers

**Repo (local, `/Users/ryan/Desktop/latent_space_diffusion_analysis`)**
- Paper: `ICLR_2026/` (identical to `ICLR_2026.zip`, Aug 29). `main.tex` inputs at lines 117–137. Theory stub `sec-theory.tex`.
- Orphaned theory: `ICLR_2026/sec-appendix.tex` §"RFNN: sketch of an analytical derivation" (L483–853); root `sec-appendix-fourbulk.tex`, `sec-rfnn-bounds.tex`.
- Old plans: `_next_steps/theory_plan.md`, `NEXT_STEPS.md`, `n_shape_heuristic_derivation.md` (retract).
- Code: `code/v3/lib/{true_score,eigenvalues,metrics,training,data_synthetic}.py`, `code/experiment_v2_rfnn.py` (`compute_U` L110: **no `1/√p` on φ ⇒ saved eigenvalues = p× theory**; W drawn `/√d_lat` so pre-activation matches Eq. 1).
- Spectra: `sigma_noise_{0.5,0.01}/exp2_rfnn/raw_data/di5_d*_n500_s42/eigenvalues_{pre,post}.npy`, `bulk_summary.json` (root; 35 runs; sizes assigned by index, circular).
- MLP sweeps: `multiseed_runs/exp2_mlp_dlat_sn05_5m_bonnaire_hidden256/` (main text), `..._combined/` (hidden=8·d).
- Real-data spectral features: `clean figures/{celeba,cifar10}_spectral_mem_predictor/*_spectral_features_primary_t.csv` (columns: d, effective_rank_excess, lambda_min, beta_floor, z_normsq_over_d_mean=q, …).
- Regression test: `code/v3/tests/test_true_score.py`.

**Remote**
- GitHub `trevorbchen/latent_space_diffusion_analysis` — at `f35df1c` (2026-06-30); does **not** have `ICLR_2026/`. SSH times out here; fetch via `git fetch https://github.com/trevorbchen/latent_space_diffusion_analysis.git main:refs/remotes/origin/main`.
- HF `trevorbchen/diffusion_memorization` — 678 files, 2026-08-30: 157 `last_model.pt`, 39 VAEs, per-run `metrics.jsonl`. DiT intentionally excluded. **Public despite README saying private.**

**Derivations (adversarially checked, with LaTeX)**
- `_next_steps/theory_pieces_raw.md` (extracted from the workflow journal).
- Raw journal: `~/.claude/projects/-Users-ryan-Desktop-latent-space-diffusion-analysis/c16a2edd-ada1-4528-a0b2-279e757294b9/subagents/workflows/wf_3725403a-9c4/journal.jsonl` — `{"type":"result",...}` lines; pieces have a `statements` key.
- Audit journal (157 findings): same path with `wf_8c9cedcc-180/journal.jsonl`.

**Simulation scripts (scratchpad, copy into repo if useful)**
`/private/tmp/claude-502/-Users-ryan-Desktop-latent-space-diffusion-analysis/c16a2edd-ada1-4528-a0b2-279e757294b9/scratchpad/{rfnn_spectrum.py,rfnn_controls.py}` — independent RFNN spectrum sims that established p-independence and the d_lat exponents (signal ~d^−0.65, noise-dim ~d^−0.73, sample ~flat at p=64d).

**Audit report**: https://claude.ai/code/artifact/a8e70925-877b-45ff-ae73-fce67e37de8d
