# Wording pass and subsequent numerical audit

Source: `ICLR_2026/main_iclr2027_9page.tex` and its included sections, as present on 2026-09-25. The source contains the newer five-seed projection results (29.9%, 0.4% chance level); those values are preserved. Original manuscript files were not edited.

Applied skill: `/Users/ryan/.codex/skills/game-the-llm-reviewer/SKILL.md`. Strategies prioritize contribution stance, evidence framing, and adjacent scope qualifications. Their effect on this manuscript or any reviewer score has not been tested. No reviewer was queried or simulated.

## Changes

### 1. `abstract-rescoped.tex` — S3: abstract emphasis

**Before**

Latent diffusion models operate in a latent space whose dimension $\dlat$ typically exceeds the effective signal dimension of the data. We study how latent width affects memorization.

**After**

We study how latent width affects memorization in latent diffusion models, whose latent dimension $\dlat$ typically exceeds the effective signal dimension of the data.

**Evidence anchor:** Abstract; Sections 2 and 3.
**Invariant / equivalence check:** The same research question and typical latent/signal dimension relationship; no new result or novelty claim.

### 2. `abstract-rescoped.tex` — S1/S5: evidence focus with adjacent scope boundary

**Before**

This mechanism is supported in frozen features, but its causal role in trained networks remains unestablished.

**After**

Frozen-feature results support this mechanism; its causal role in trained networks remains unestablished.

**Evidence anchor:** Section 4 fixed-q control; Section 5 trained-network measurements.
**Invariant / equivalence check:** Frozen-feature support and the unresolved causal transfer remain equally explicit.

### 3. `sec9-intro.tex` — S1: contribution stance

**Before**

A fixed-width $\dlat$ sweep at known $\dint$ in which memorization falls from $30\%$ to near zero, together with a projected variant of the nearest-neighbor ratio test.

**After**

We characterize the decline in memorization from $30\%$ to near zero in a fixed-width $\dlat$ sweep at known $\dint$, and test it with a projected variant of the nearest-neighbor ratio criterion.

**Evidence anchor:** Section 3; Table 1.
**Invariant / equivalence check:** Same sweep, observed decline, and projected test; no claim to invent the criterion or establish a trained-network cause.

### 4. `sec9-intro.tex` — S1/S5: foreground the contribution while preserving prior-work boundary

**Before**

We extend neither result: we evaluate the existing theorem at the two-atom covariance with strictly positive null variance $\signoise>0$ and read the resulting blocks as gradient-flow learning-rate bands.

**After**

Our analysis evaluates the existing theorem at the two-atom covariance with strictly positive null variance $\signoise>0$ and interprets the resulting blocks as gradient-flow learning-rate bands; it is a specialization, not an extension of either result.

**Evidence anchor:** Section 4; Bonnaire theorem as cited; Appendix I.
**Invariant / equivalence check:** Positive-null-variance evaluation and interpretation retained; no extension or priority claim added.

### 5. `sec9-phenomenon.tex` — S1: name the existing contribution directly

**Before**

The second readout needs no threshold.

**After**

Empirical-score alignment provides a threshold-free second readout.

**Evidence anchor:** Equation 4 and immediately following definition.
**Invariant / equivalence check:** Same alignment readout; no change to the later threshold-based timing comparison.

### 6. `sec9-real.tex` — S2: emphasize the observed comparison

**Before**

Once identity is preserved, time to memorization grows with width.

**After**

Post-identity-threshold width comparisons show longer times to memorization.

**Evidence anchor:** Appendix hitting-time table; Section 5 endpoints.
**Invariant / equivalence check:** Same endpoint effect in the identity-preserving regime; following non-monotonicity qualification retained.

### 7. `sec9-theory.tex` — S5: scope framing

**Before**

It says nothing about trainable first layers or adaptive optimizers, and whether any of it transfers to the trained networks is the question \cref{sec:real} puts to the data.

**After**

Trainable first layers and adaptive optimizers remain outside this model; \cref{sec:real} tests whether its explanation transfers to trained networks.

**Evidence anchor:** Section 4 setup and closing paragraph; Section 5.
**Invariant / equivalence check:** Both excluded cases and the unresolved transfer question remain explicit.

### 8. `sec9-discussion.tex` — S5: scope framing

**Before**

These observations do not establish that a width-set operating point causes the MLP delay.

**After**

Establishing whether a width-set operating point causes the MLP delay remains an open question after these observations.

**Evidence anchor:** Section 5 pre-activation measurements; discussion causal-gap paragraph.
**Invariant / equivalence check:** Same lack of causal identification; the observations are not presented as resolving it.

## Preserved content and scope of the initial wording pass

In the initial wording pass, all numbers, inline mathematics, equations, labels, citations, figures, bibliography, appendices, and disclosure content were unchanged. The subsequent authorized numerical corrections below supersede that statement for the current copy. The numerical values, mathematical spans, and reference/label sequences in edited files were checked programmatically. The eight passages were also compared for the same claims, uncertainty, comparators, and limitations. A file hash manifest records the input snapshot.

## Issues flagged by the initial wording pass (historical)

- Section 3 prose still says 3.8% for the full-space d=20 comparison; the updated five-seed table says 3.6%. No value was selected or replaced in this pass.
- Main Table 1 gives alpha=0.31 at d=40; the appendix table gives 0.30. The discussion gives a final pre-activation ratio of 0.31 while Section 5 gives 0.30. These are different quantities; both inconsistencies need checking against their respective source data.
- Section 3 calls the final small-width score error approximately 2.3 and also reports 2.59 at d=5. Reconcile the summaries or their evaluation provenance.
- The original strong assertions about projected-test sensitivity, signal-only copying, and four-block separation in every configuration are left unchanged rather than strengthened or silently weakened through a rhetorical edit.
- The author-specific AI-disclosure TODOs remain visible. This wording pass cannot supply author verification or code-provenance facts.
- Existing figure-label defects and appendix overfull equations are outside this small wording pass. This is a revised draft, not a certification of submission readiness.

## Build and layout

Compiled successfully with pdfLaTeX/latexmk using the repository's documented TeX Live 2016 workaround (system fancyhdr in a temporary build directory; bundled manuscript style files are unchanged). Main text ends on page 9; the full PDF has 108 pages including references and appendices. No unresolved references or citations. Visually inspected all edited main-text pages. Existing figure-label defects and author TODOs remain; this pass introduced no new visible text-layout issue. Original source hashes still match the pre-edit manifest.

Files: `main_iclr2027_9page.tex` (entry point), `main_iclr2027_9page.pdf`, `abstract_openreview.txt`, `wording.patch`, and `source_manifest.json`. All referenced figures and section files are included. Build on a modern TeX installation with `latexmk -pdf main_iclr2027_9page.tex`.

## Subsequent numerical corrections — 2026-09-25

The user subsequently requested investigation of the numerical inconsistencies. See `numerical_audit.md` for evidence and `numerical_audit.py` / `numerical_audit.json` for reproducible calculations. These corrections are separate from the meaning-preserving rhetorical edits above.

- Section 3: replace the stale two-seed E2 value 3.8% with the five-seed 3.6%. State final score error as 2.59 at d=5 and 2.26–2.35 at d=8–40, replacing exact-flatness language with comparable errors. The alignment argument is described as consistent with those errors, rather than a complete explanation.
- Appendix alignment table: use five-seed values 0.31 at d=40/5M, 0.54 at d=5/250k and 0.37 at d=10/250k.
- Section 5 and discussion: use the reproducible five-seed pre-activation ratios 0.96, 0.70, 0.47 and 0.30. Define this as an uncentered mean square, distinct from the frozen-feature variance and alignment. Remove unsupported historical initialization values and the exact-equality assertion.
- Appendix: add the pre-activation measurement protocol and explain how plotted score errors were recovered from denoising losses, including the finite-sample limitation.

The revised abstract was not edited in this numerical pass. Main text remains within nine pages; the complete PDF has 108 pages. `revision.patch` now records the combined TeX changes, while `wording.patch` records only the initial wording pass. Original manuscript files remain unchanged.


## Citation audit (2026-09-25)

Verified existence and assessed contextual fit of 40 cited works (132 key occurrences). See `citation_audit.md` and the per-use JSON. Corrected only bibliography metadata: removed Benigni–Péché’s literal issue `none`; added George et al. PMLR 300, pages 1405–1413 and proceedings URL. Scientific attribution/theorem issues are documented and remain unresolved. No abstract or scientific text changed; PDF not rebuilt during this audit. Exact metadata diff: `citation_metadata.patch`.


## Scientific correction pass (2026-09-25)

The citation audit led to substantive corrections, superseding the earlier direct-specialization wording (including change 4 above). See `theory_repair.md` for the resolution of each finding and `theory_repair.patch` for the exact edits. Operator definitions, rank arguments, spectral scope, metric attributions, the Tweedie remainder and Gaussian singular-value bound were corrected. The existing abstract and main empirical sections were preserved. The rebuilt main text ends on page 9; the full PDF is 109 pages. This pass does not turn the conditional spectral approximation into a proved four-block theorem.
# Mathematical repair follow-up

The subsequent proof audit has been addressed in the revised source and PDF. See [proof_repair.md](proof_repair.md) for the finding-by-finding changes and validation; the original audit is preserved as a historical record. This follow-up corrects scientific claims and is not merely a rhetorical edit.
