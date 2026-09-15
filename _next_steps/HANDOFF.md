# HANDOFF — theory work on the latent-dimensionality memorization paper

Written 2026-09-14 so any agent or person can continue from cold. Everything referenced is in this repo
unless marked REMOTE. Read this file, then `THEORY_TODO.md`, then whatever section you're working on.

## 1. What this project is

Paper: "How Excess Latent Dimensionality Delays Memorization in Diffusion Models" (draft: `ICLR_2026/`, compile
`main.tex`). Claim: an RFNN/MLP score network's feature-correlation spectrum splits into four bulks; the
`d_lat − d_int` "noise-dim" bulk sits between generalization and memorization and delays the latter.
Authors: Trevor Chen (repo owner, real-data/VAE/DiT) and Ryan Shahbaba (theory; git user TheTrueShah). Caltech. Target: ICLR main.

## 2. State of the repo right now

- Analysis repo: local `main` is 8 commits ahead of `origin/main` (`f35df1c`), NOT pushed.
- Paper repo: https://github.com/trevorbchen/ICLR-2026 (Overleaf-synced, paper files at repo root). The theory work was pushed
  there 2026-09-15 as branch `theory-section` with PR https://github.com/trevorbchen/ICLR-2026/pull/1. That branch = this repo's `ICLR_2026/` + `_next_steps/` as `theory_notes/`.
  SSH to github.com times out on this machine; fetch with
  `git fetch https://github.com/trevorbchen/latent_space_diffusion_analysis.git main:refs/remotes/origin/main`.
- The transpose fix (§3.1), the test, and all `_next_steps/` files are committed locally as `2bd8d60` (not pushed).
- `ICLR_2026/sec-theory.tex` is the assembled Section 2 (197 lines, ~5.5 pp); `sec-appendix-theory.tex` is the theory appendix (1416 lines).
  As of 2026-09-15 `main.tex` DOES input the appendix, `main.bbl` is regenerated, and `main.pdf` builds in place (108 pp, 0 undefined refs).
  The paper's .tex/.bib/.bbl are now tracked in git (commit `2457f45`); figures/ and the pdf are not.
  `main.tex` inputs `sec-appendix-integrated.tex` (no theorems); the old theorem lives in `ICLR_2026/sec-appendix.tex`
  (not input; also wrong — see `THEORY_TODO.md` §2).
- Disk was full on 2026-09-13; `code/v3/data/encoded/celeba_train_*.pt` are two 8 GB regenerable caches.
- REMOTE: all real-data checkpoints are on HF `trevorbchen/diffusion_memorization` (157 `last_model.pt`, 39 VAEs,
  2026-08-30). README says private; API says `private: false`. DiT intentionally excluded. HF README says
  hidden=1024/depth=5/10k-eval; paper says 256/3/1k — unresolved.

## 3. What has been done (all in `_next_steps/` unless noted)

### 3.1 Bug fix (verified, committed in `2bd8d60`)
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

### 3.6 Assembly (DONE 2026-09-14; see §7 for the critic's findings)
Three writers were producing, in chunks: `ICLR_2026/sec-theory.tex` (main text), `ICLR_2026/sec-appendix-theory.tex`
(proofs), `_next_steps/theory_bib_additions.bib`, `_next_steps/theory_prose_edits.md`; then a critic compiles.
**Check**: `wc -l ICLR_2026/sec-theory.tex` (>9 lines = writer ran; ~180–300 = done),
`ls ICLR_2026/sec-appendix-theory.tex` (600–1200 lines when done). If a file is missing or truncated, do §4.1 by hand.
The workflow script that produced all of this: `scripts/write-theory-section.workflow.js` (for reference; it is a
Claude-Code workflow and won't run elsewhere — the spec inside its prompts is what matters).

## 4. What to do next, in order

### 4.1 Finish/verify the assembly — DONE 2026-09-15 (kept for reference)
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
  (main.tex currently inputs `main.bbl` directly; regenerate it — or use `_next_steps/main.bbl.regenerated`, produced 2026-09-14
  from references.bib + theory_bib_additions.bib with icml2026.bst; 45 entries.) Gotcha: BibTeX has no `%` comment syntax;
  the `% VERIFY` notes were moved to a plain-text header in the .bib file.
- Do NOT reproduce the falsified claims (`THEORY_TODO.md` §2): no `/ψ_p` edges, no count×timescale bound,
  no "τ_gen is d_lat-independent", no "Bonnaire assumes isotropy".

### 4.2 Apply the prose edits — DONE 2026-09-15
All 53 rows of `theory_prose_edits.md` applied (46 by script, 7 by hand), plus PROJECT_BRIEF.md rewritten. Every touched file has a
`.bak-20260915` sibling. Two cross-cutting items remain in `theory_prose_edits.md` §"Cross-cutting" — read that section.

### 4.2b Trim Section 2 — CANDIDATE READY 2026-09-15 (not yet swapped in)
`ICLR_2026/sec-theory-short.tex`: 2,296 words (from 4,583), every cross-referenced label kept, builds clean; Section 3 moves from
p.9 to p.6. Still ~4 pages of column space because 13 display equations survive; going lower means dropping equations.
To use it: change `\input{sec-theory}` to `\input{sec-theory-short}` in main.tex. The full version stays as `sec-theory.tex`.
Note: References currently start on p.15, so the WHOLE main text is ~5 pages over a 9–10 page limit; Section 2 can give back ~2.5 of those.

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
- Buffer = `λ_min^signal / λ_max^sample` (onset) = λ_5/λ_{d+1}. On `sigma_noise_0.5/exp2_rfnn/raw_data` (seed 42, p=64d) it opens
  21.2 → 426.3 over d_lat 5→200 (recomputed 2026-09-15; the pieces' 428 is the same number to rounding). The audit's
  "31→576, τ_gen 8.1×" does NOT reproduce from any file on disk and is retired. Median statistic reverses (shrinks ~14×).
- `τ_gen` = 1/λ_min^signal grows 11.0× in flow units (λ_code/p) over the sweep; in optimizer steps it FALLS because lr = 0.01·d_lat/Δ_t.
- Isotropic control: `_next_steps/e5_isotropic_results.txt` (2026-09-15) — see §7 item 3 for what it settles.
- Fixed-width control (`clean figures/p_fixed_energy_fixed/exp2_main_gmm`, p=1800, n=500): R = 16.7 → 889 over d_lat 5→200 (verified 2026-09-15).
  Beyond d_lat ≥ n the sample block is empty and the index-based R is meaningless (ignore the d=500, 1000 runs for R).
- Real VAE latents: effective rank ≈ 0.45–0.55·d; `λ_min = Δ_t = 0.181` for d ≥ 140 (dead dims); CelebA `d_eff`
  saturates ~88–90 for d ≥ 160. Source: `clean figures/*/*_spectral_features_primary_t.csv`.
- Bonnaire 2025 (2505.17638) Thm 3.1 is for arbitrary ρ_Σ. George–Veiga–Macris (2502.00336) has block dims (p, d−D, D, n).
- DiT pilot (`sec-appendix-integrated.tex` §Spatial-DiT): memorization RISES 22%→48% for d_lat 5→10. Opposite sign to MLP.

## 6. Don'ts
- Don't push to GitHub without the authors' say-so (the fix changes result semantics; `ICLR_2026/` was never pushed).
- Don't delete `code/v3/data/encoded/*.pt` without asking (regenerable, but 16 GB to rebuild).
- Don't cite `n_shape_heuristic_derivation.md` or any synthetic `score_error` number from before the fix.
- Don't `cp -R ICLR_2026` for build tests (22 MB figures; disk is tight) — copy `*.tex *.sty *.cls *.bst *.bib *.bbl` and symlink `figures/`.

## 7. Assembly outcome (2026-09-14) and what the build critic found

The three writers finished: `ICLR_2026/sec-theory.tex` (197 lines, ~5.5 two-column pages — over the 1.5–3 target),
`ICLR_2026/sec-appendix-theory.tex` (1416 lines, 85 statements, ~58 pages), `_next_steps/theory_bib_additions.bib`,
`_next_steps/theory_prose_edits.md`. Mechanical build issues (cref names, 9 duplicate labels, bib duplicates) were
fixed by script the same day. `main.tex` is NOT modified: to build, add `\input{sec-appendix-theory}` after line 137
and regenerate `main.bbl` (main.tex inputs the .bbl directly, so appending to references.bib alone does nothing).

### Substantive issues the critic found — these need an AUTHOR decision, not a script
1. **Two sets of timescale numbers — RESOLVED 2026-09-15.** Recomputed λ_5/λ_{d+1} from the spectra file directly: 21.2→426.3,
   τ_gen 11.0× (flow units). That is the pieces' set. The audit's 31→576 / 8.1× does not reproduce (its τ_gen list was
   shifted one grid point and its d=200 value was wrong). Audit numbers deleted from both tex files.
2. **Single-draw vs ξ-averaged cliff — RESOLVED 2026-09-15.** Every quoted shoulder ratio matches the spectra
   (λ_n/λ_{n+1} = 1.01, 1.01, 1.32, 1.70, 2.08, 1.68, 1.90, 1.73 at d=10..200); the two "series" were the same data at different d.
   One quoted range corrected (1.32–1.73 → 1.3–2.1). `sec-rfnn.tex` now states the released U is a 50-draw average with a soft shoulder.