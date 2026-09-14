# HANDOFF — theory work on the latent-dimensionality memorization paper

Written 2026-09-14 so any agent or person can continue from cold. Everything referenced is in this repo
unless marked REMOTE. Read this file, then `THEORY_TODO.md`, then whatever section you're working on.

## 1. What this project is

Paper: "How Excess Latent Dimensionality Delays Memorization in Diffusion Models" (draft: `ICLR_2026/`, compile
`main.tex`). Claim: an RFNN/MLP score network's feature-correlation spectrum splits into four bulks; the
`d_lat − d_int` "noise-dim" bulk sits between generalization and memorization and delays the latter.
Authors: Trevor Chen (repo owner, real-data/VAE/DiT), Ryan Shahbaba (theory; git user TheTrueShah),
Avni Garg (buffer corollary), Kevin Peng (n-shape thread — now moot, see §3.1). Caltech. Target: ICLR main.

## 2. State of the repo right now

- `main` == `origin/main` == `f35df1c` (2026-06-30). GitHub does NOT have `ICLR_2026/` (untracked, 22 MB).
  SSH to github.com times out on this machine; fetch with
  `git fetch https://github.com/trevorbchen/latent_space_diffusion_analysis.git main:refs/remotes/origin/main`.
- **Uncommitted**: the transpose fix (§3.1) in 4 files + `code/v3/tests/test_true_score.py`. Commit it.
- `ICLR_2026/sec-theory.tex` is (or was, if the writer finished) a 9-line `\todo{}` stub that PRINTS in the PDF.
  `main.tex` inputs `sec-appendix-integrated.tex` (no theorems); the old theorem lives in `ICLR_2026/sec-appendix.tex`
  (not input; also wrong — see `THEORY_TODO.md` §2).
- Disk was full on 2026-09-13; `code/v3/data/encoded/celeba_train_*.pt` are two 8 GB regenerable caches.
- REMOTE: all real-data checkpoints are on HF `trevorbchen/diffusion_memorization` (157 `last_model.pt`, 39 VAEs,
  2026-08-30). README says private; API says `private: false`. DiT intentionally excluded. HF README says
  hidden=1024/depth=5/10k-eval; paper says 256/3/1k — unresolved.

## 3. What has been done (all in `_next_steps/` unless noted)

### 3.1 Bug fix (verified, uncommitted)
`precompute_sigma_t` / `precompute_score_params` computed `Qᵀ D⁻¹ Q`; data frame is `x = Q·x_orig` so the
correct inverse is `Q D⁻¹ Qᵀ`. Fixed in `code/v3/lib/true_score.py:63`, `code/experiment_v2.py:215`,
`experiment_v2.py:215`, `figures/code/experiment_v2.py:215`. Test: `python3 code/v3/tests/test_true_score.py`
(Stein's identity `E[s(x)xᵀ] = −I`; passes now, fails 4/5 on old code, inert at `d_lat = d_int`).
Consequence: every saved synthetic `score_error` is wrong; the "n-shape" in `n_shape_heuristic_derivation.md`
is 100% this bug (predicted mismatch matches observed to 3 digits). Training/memorization/FID/spectra unaffected.

### 3.2 Audit — `AUDIT_FINDINGS.md`
157 findings, 109 confirmed by adversarial referees. The blocking ones are summarized in `THEORY_TODO.md` §2.

### 3.3 Independent simulations — `scripts/rfnn_spectrum.py`, `scripts/rfnn_controls.py`
Fresh RFNN spectrum sims (not the repo's code path). Established: bulk edges are p-independent (5.8× change in
`ψ_p` moves edges <0.5%); signal ~`d_lat^−0.65`, noise-dim ~`d_lat^−0.73`, sample ~flat at `p = 64·d_lat`;
`tr(U)` falls 0.45→0.24 (never grows). Run: `python3 _next_steps/scripts/rfnn_spectrum.py` (~15 s).

### 3.4 Plan — `THEORY_TODO.md`
Everything that needs doing, with file:line sources, experiments (which run today), prose edits, citations.

### 3.5 Derivations — `theory_pieces_raw.md` (785 KB)
Eight pieces, each derive → 2 hostile referees → revise. 81 formal statements (47 proved / 24 sketched /
10 conjectured, self-labeled), full LaTeX per piece (compact main-text block + full appendix block).
Pieces: P1 setup+definitions · P2 four-bulk proposition · P3 buffer corollary · P4 linear score model ·
P5 bridge lemma · P6 predictor · P7 real data/`d_eff`/collapse · P8 regime boundaries.

### 3.6 Assembly (IN PROGRESS when this was written — check the files)
Three writers were producing, in chunks: `ICLR_2026/sec-theory.tex` (main text), `ICLR_2026/sec-appendix-theory.tex`
(proofs), `_next_steps/theory_bib_additions.bib`, `_next_steps/theory_prose_edits.md`; then a critic compiles.
**Check**: `wc -l ICLR_2026/sec-theory.tex` (>9 lines = writer ran; ~180–300 = done),
`ls ICLR_2026/sec-appendix-theory.tex` (600–1200 lines when done). If a file is missing or truncated, do §4.1 by hand.
The workflow script that produced all of this: `scripts/write-theory-section.workflow.js` (for reference; it is a
Claude-Code workflow and won't run elsewhere — the spec inside its prompts is what matters).

## 4. What to do next, in order

### 4.1 Finish/verify the assembly
If `sec-theory.tex` / `sec-appendix-theory.tex` are incomplete, assemble them from `theory_pieces_raw.md`:
- Main text order: setup+assumptions → Def 1–2 (P1) → four-bulk Prop + edge table + Bonnaire/George positioning (P2)
  → buffer Corollary as edge ratio + 2 remarks (P3) → linear-model Lemma (P4) → bridge Prop (P5) → one para each on
  predictor (P6), real data (P7), regime boundaries (P8) → 4-line scope. Use the "Main-text LaTeX (compact)" blocks.
  Target 180–300 lines. Every proof `\cref`s the appendix.
- Appendix: `\section{Theory: proofs and derivations}\label{app:theory}`, one `\subsection` per piece using the
  "Appendix LaTeX (full)" blocks, keep the piece labels verbatim, include the predicted-vs-measured tables (P2, P3, P7)
  and each piece's "Open gaps".
- Macros/envs: see `ICLR_2026/main.tex` lines 35–75 (`\dlat \dint \nsamp \pwidth \taugen \taumem \Umat \Sigmadata
  \sigsig \signoise`; theorem/proposition/lemma/corollary/definition/assumption/remark).
- Then add ONE line to `main.tex` after line 137: `\input{sec-appendix-theory}`; append `theory_bib_additions.bib`
  to `references.bib`; build: `cd ICLR_2026 && pdflatex main && bibtex main && pdflatex main && pdflatex main`.
  (main.tex currently inputs `main.bbl` directly; regenerate it.)
- Do NOT reproduce the falsified claims (`THEORY_TODO.md` §2): no `/ψ_p` edges, no count×timescale bound,
  no "τ_gen is d_lat-independent", no "Bonnaire assumes isotropy".

### 4.2 Apply the prose edits
`theory_prose_edits.md` (if written) or `THEORY_TODO.md` §6. These make the rest of the draft agree with Section 2.

### 4.3 Housekeeping
`THEORY_TODO.md` §1: commit the fix; fix `bulk_indices()` (`code/v3/lib/eigenvalues.py:75` — sample bulk is
`n − d_lat`, rank-null `p − n`); fix "500 MC" → 50; fix the d=240 sweep claim; resolve the HF architecture
mismatch; make HF private; reconcile the three appendix copies.

### 4.4 Experiments the theory demands
`THEORY_TODO.md` §5. Highest value first: E1 collapse probe (HF VAEs, today), E2 dimension-controlled
memorization (decides whether the headline is real), E3 activation sweep + E4 norm-fixed control (decides the
mechanism: mode count vs tanh de-saturation), E5 isotropic control (minutes).

## 5. Ground truth to not re-derive (details in `THEORY_TODO.md` §2–3)
- Counts: `d_int`, `d_lat − d_int`, `n − d_lat`, `p − n`. Cliff at index `n`.
- Edges ∝ `a₁(q)²α_t²/d_lat`, `a₁(q)²β_t²/d_lat`, sample mean ~`a⋆(q)²/n`; `q = tr(M_t)/d_lat`; trace = `E[tanh(√q z)²]`.
- Code's `compute_U` omits `1/√p` on φ ⇒ saved eigenvalues = `p ×` theory. W is drawn `/√d_lat` (matches Eq. 1).
- Buffer = `λ_min^signal / λ_max^sample` (onset). Opens 31→576 over d_lat 5→200 at σ⊥=0.5. Median statistic reverses.
- `τ_gen` grows 8.1× over the sweep. Isotropic Σ=I still gives 10.7× gap growth.
- Real VAE latents: effective rank ≈ 0.45–0.55·d; `λ_min = Δ_t = 0.181` for d ≥ 140 (dead dims); CelebA `d_eff`
  saturates ~88–90 for d ≥ 160. Source: `clean figures/*/*_spectral_features_primary_t.csv`.
- Bonnaire 2025 (2505.17638) Thm 3.1 is for arbitrary ρ_Σ. George–Veiga–Macris (2502.00336) has block dims (p, d−D, D, n).
- DiT pilot (`sec-appendix-integrated.tex` §Spatial-DiT): memorization RISES 22%→48% for d_lat 5→10. Opposite sign to MLP.

## 6. Don'ts
- Don't push to GitHub without the authors' say-so (the fix changes result semantics; `ICLR_2026/` was never pushed).
- Don't delete `code/v3/data/encoded/*.pt` without asking (regenerable, but 16 GB to rebuild).
- Don't cite `n_shape_heuristic_derivation.md` or any synthetic `score_error` number from before the fix.
- Don't `cp -R ICLR_2026` for build tests (22 MB figures; disk is tight) — copy `*.tex *.sty *.cls *.bst *.bib *.bbl` and symlink `figures/`.
