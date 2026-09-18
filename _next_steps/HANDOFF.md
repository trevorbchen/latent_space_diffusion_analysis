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
  2026-08-30). README says private; API says `private: false`. DiT intentionally excluded. ARCHITECTURE RESOLVED 2026-09-18 (Ryan's call: HF is the truth).
  Every run's own `config.json` on HF says hidden=1024, depth=5, batch_size=512, n_gen_samples=10000, fid_n_gen=10000, fid_n_real=1000,
  lr=1e-3, momentum=0.8, 5M steps, n_train=1000. The paper said 256/3/batch 256/1k generated; corrected in `sec-real-data.tex` §6.1 and
  `sec-appendix-integrated.tex`. The theory appendix already said depth 5, width 1024.

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
**E4 and E5 DONE 2026-09-15**: `e4_norm_fixed_results.txt` (R grows 1.09× at fixed q vs 3.17× baseline over d=40→240; mode count would give 6.7×), `e5_isotropic_results.txt`.

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
3. **Isotropic control — RESOLVED 2026-09-15.** E5 (`scripts/e5_isotropic.py`, results in `e5_isotropic_results.txt`): at σ⊥=σ_sig=1 the
   ratio is non-monotone, 14→87 (peak d≈40)→60 at d=160, exactly the appendix's prediction. The audit's "10.7× gap" was a code-unit
   artifact. Text corrected in `rem:isotropic-buf`, main-text remark, and `rem:scope-outside`.
4. **NN-ratio convention — CORRECTED 2026-09-18 (my 09-15 "resolution" was WRONG).** Two conventions DO exist. The released sweeps use Bonnaire's form
   d(gen,NN1)/d(gen,NN2): root `experiment_v2.py` (imported by `run_mlp_multiseed.py`; the sweep dir is named `..._bonnaire_...`) and `remote_metrics.py`
   (the current `lib/metrics.py` on Trevor's machines, used by the real-data runners; matches the HF README). The stale local copies `code/experiment_v2.py`,
   `code/v3/lib/metrics.py`, `figures/code/experiment_v2.py` use Somepalli's d(gen,NN1)/d(NN1,NN2). On 09-15 I checked only the stale copies. `def:bridge-objects`
   and gap G9 now state both forms with the right attribution (constant 9 → 8 for the released metric); the protocol paragraph states the formula.
5. **Real-data numbers — RESOLVED 2026-09-15.** Recomputed from `clean figures/*/*_spectral_features_primary_t.csv`: CelebA floor from d=140,
   q flat 1.71–1.84 over d=20–120 then 1.62→1.25 (140→200), eff. rank 89.5/90.3/87.9 at 160/180/200; CIFAR floor within 2e-3 at 160,
   3e-4 at 180, <1e-4 from 200, q 2.27→1.11. The paper's own q numbers already matched; the CIFAR threshold was harmonized to ~160.
   (The "2.92→1.30" q figure was the audit's, never in the paper.)
6. **Hitting-time tables — RESOLVED 2026-09-15.** Added `tab:hitting-times-real` to `sec-appendix-theory.tex` (from
   `*_empirical_tau_summary.csv`: τ_obs at thresholds 0.01/0.05/0.10, mean±sd over 5 seeds, censored counts). "Released tables" phrases now cite it.
7. **George–Veiga–Macris phrasing — RESOLVED 2026-09-15.** Main text now cites `rem:fourbulk-sigperp0` for the merged limit. The
   "no longer separated" phrase the critic quoted was not in the file; both versions already say what is new is our strictly positive σ⊥.
8. **Length — CANDIDATE READY.** `sec-theory-short.tex` (2,296 words, ~4 pp). Swap the input line in main.tex to use it. Further cuts mean dropping equations.
9. **Clock conventions — STATED, not re-verified line by line.** `app:theory-linear` opens with the convention note; every cross-piece
   ratio quoted in the main text is unit-free. A line-by-line audit of absolute step counts in the appendix has NOT been done.
10. **Bib — RESOLVED 2026-09-15.** All 16 theory entries verified against arXiv/publisher pages; every flagged id was correct.
   George–Veiga–Macris is AISTATS 2026 (renders as 2026); Ba et al. gained NeurIPS pages. The header had hidden a 17th entry; gone.
   `references.bib` = Trevor's original + these 16; `main.bbl` regenerated (45 entries).

## 8. Figure 2 was plotting the wrong sweep (found and fixed 2026-09-15)

`plot_mlp_multiseed_clean.py` defaults `--root` to `multiseed_runs/exp2_mlp_dlat_sn05_5m_combined` (hidden = 8·d_lat). The
2026-07-06 "refresh synthetic MLP figures" commit regenerated the MAIN-TEXT figures from it, so Figure 2(a) showed the scaled-width
sweep (memorization ~1%, U-shaped, rising again at d≥30) under a caption describing the fixed-width sweep (30% → 0.1%, monotone,
`multiseed_runs/exp2_mlp_dlat_sn05_5m_bonnaire_hidden256`). The appendix `_full` figures were already from the fixed-width sweep.
Figure 2(b) and appendix Fig 17/18 score-error panels used the buggy transposed-covariance metric, divided by d twice.

Fixed: Figure 2(a) regenerated from the fixed-width sweep; Figure 2(b) and the appendix score-error panels replaced by the corrected
population score error (`_next_steps/e_scoreerr/`, exact via the DSM identity); the scaled-width plot kept as appendix control figure
`fig:app-synthetic-mlp-scaled` with an honest caption; Figure 2 caption and the paragraph under it rewritten (timing claim, not
"quality preserved"). Old figure files: `_next_steps/figures_replaced_20260915/`. Corrected-vs-logged comparison (not in the paper):
`_next_steps/e_scoreerr/synthetic_mlp_score_error_final_corrected_vs_logged.pdf`.

CLOSED by Trevor 2026-09-15 (paper repo commit `0232650`): the d > 40 runs were split across his machines (ml6: d=5..40 all seeds + d=60..240 seeds 42,43;
ml7: d=60..240 seeds 44-46). He consolidated all 100 runs (20 widths x 5 seeds), re-ran `compute_corrected_score_error.py` (regression vs my table: 4.4e-5),
added `make_paper_score_error_figures.py --range {main,full}`, drew the CI bands, and fixed my mistake of writing the 10-curve plot into the `_full` slots.
Full-range finding (time for the corrected error to double from its minimum): <=100k for d<=40, 200k (60), 500k (80), 800k (100), 1.55M (120), 2.5M (140),
not by 5M for d>=160; for d>=200 the error is still falling at 5M (plateau ~0.53/dim). Strongest delay evidence in the paper; not yet quoted in Section 4.
Note: `main` on the paper repo now holds a cherry-pick of the Figure-2 fix plus Trevor's extension but NOT the theory section; PR #1 is still open and merges clean.
E2 (projection test) still needed; it can run on this Mac in ~2.5 h (0.87 ms/step on CPU) once the script saves samples and Q.
