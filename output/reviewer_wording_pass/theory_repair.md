# Theory and citation repair

Updated 2026-09-25. This is a scientific correction pass, not a meaning-preserving rhetorical edit. Work is confined to `output/reviewer_wording_pass`; the original manuscript directory matches its saved source manifest.

**Result:** the main text still ends on page 9. The rebuilt manuscript has 109 pages including bibliography and appendices. The empirical results are preserved. The four-block explanation is now an explicitly conditional approximation of measured index groups, rather than a theorem directly inherited from prior work.

## What changed

| Audit finding | Resolution in the revised body | What remains conditional |
|---|---|---|
| Noise-averaged and single-copy operators conflated | Defined U^(1), U^(K), and E[U^(1)] separately. Added exact conditional-mean/covariance decomposition U=B+C and rank bounds. Documented K=50 for released spectra and K=40 for the fixed-q control. | Noise-averaged spectral component counts and finite-Monte-Carlo edge errors are not proved. |
| Bonnaire specialization and floor attribution | Removed direct-specialization claims from introduction, main theory and repeated appendix passages. Distinguished clean covariance from diffused covariance, and source floor weight 1−(n+d)/p from the single-copy rank deficit. | Mixture universality and finite-rank signal outliers need separate arguments. The abstract is frozen; see below. |
| George pencil dimensions | Recorded five blocks (p,d−D,D,n,D) in Appendix B and distinguished pencil dimensions from spectral bands. | Subspace-limit comparison is qualitative. |
| Gaussian equivalence overclaimed | Recast the replacement as a hypothesis for this mixture. Corrected Liao–Couillet overlap attribution; separated risk, bulk, edge and operator claims. | No new universality proof is claimed. |
| Rank and projected-Wishart “proofs” | Replaced the rank lemma with valid interlacing/rank bounds. Separated classical finite-rank BBP limits from a strong-spike projected-Wishart approximation. Updated related appendix conclusions. | Uniform finite-ratio edge/mass accuracy remains open. |
| Training time versus sampling time | Separated those notions in the introduction and removed inevitable-late-memorization wording. | The empirical scope remains the studied settings. |
| Nearest-neighbor metric attribution | Cited Bonnaire for the released first/second-neighbor distance ratio; identified the other ratio as a distinct repository statistic. Kept its conversion assumptions visible. | Alternate-metric formulas are not exact formulas for the released metric. |
| Tweedie equality | Used the learned score only to define a plug-in denoiser, retained r, and added the explicit squared-error correction bound. Strengthened the bridge assumption and limited its conclusion to a second-moment approximation. | Attachment, negligible remainder contribution and multistep reduction still require assumptions; a pointwise concentration event does not establish pathwise attachment. |
| Gaussian singular-value bound | Replaced the book-corollary citation with the verified Corollary 5.35 of Vershynin’s survey and added a positive part before squaring the lower endpoint. | Subsequent nonlinear spectral results remain conditional on their separately stated assumptions. |
| Posterior collapse and edge products | Distinguished vanishing encoder means from full posterior collapse. Replaced claims of exact free-convolution edges with product-bound approximations. | Real-latent surrogate transfer remains unestablished. |
| Other consequences encountered | Corrected mean-iterate versus mean-loss language, false zero-mass implication, conditional tail-mass bounds and the assertion that a linear sampler can never produce an isolated metric-positive draw. | This is not a complete proof audit of every appendix statement. |

The exact gradient-flow solution remains. The measured spectral ratios and fixed-q control remain. The parameter kappa_top is explicitly measured, so the edge formula is a decomposition/approximation, not an independent parameter-free prediction of all edges.

## Checks

- Clean `latexmk` completion; zero overfull boxes, unresolved citations/references or duplicate-label warnings in the final log.
- Main text, including Discussion and limitations, ends on page 9; references begin there after the disclosure. Appendices are excluded from that nine-page count.
- Main empirical sections (`sec9-phenomenon.tex`, `sec9-real.tex`) are byte-identical to their pre-repair copies. No experiment data or figure assets changed.
- The existing abstract is byte-identical to its pre-repair copy; its hash is in `theory_repair_validation.json`.
- All original inputs checked against `source_manifest.json` still match.
- Deterministic algebra checks in `theory_repair_checks.py`: rank 8 for a single draw versus rank 20 after averaging; conditional covariance identity error below 4e−17; plug-in denoiser identity error below 9e−17; retained-remainder bound holds; rank-deficient lower endpoint is zero; projected-Wishart eigenvalue error decreases from about 0.30 to 3.1e−7 as spike strength grows. These checks illustrate the corrected claims and do not prove universality.
- Rendered and reviewed the main text and theory appendix. Reflowed oversized appendix equations and tables without shrinking fonts or changing the conference template.

## Items that remain visible

**Frozen abstract:** its sentence saying that we “evaluate the random-feature spectral theory” at a two-atom covariance remains stronger than the corrected body’s position. It was intentionally not rewritten. The precise four-component implication also needs to be read subject to the body's index-group and approximation qualifications. This pass does not establish what abstract edits the submission system permits, or verify that the local abstract matches the submitted OpenReview text.

**Author disclosure:** the existing AI-disclosure author-verification placeholders are unchanged. They require factual author input; no verification was asserted on the authors’ behalf.

**Theory status:** the body now exposes the missing arguments instead of treating a citation as a proof. A rigorous four-component theorem, trained-network causal transfer and the endpoint sampler reduction remain research questions. The existing appendix also contains explicitly recorded open gaps and legacy figure/block-label qualifications; this is not a submission-readiness certification.

## Files

- `theory_repair.patch`: this pass only, against the pre-repair revised manuscript.
- `revision.patch`: full current TeX/bibliography diff against the original manuscript.
- `theory_repair_checks.py` and `.json`: reproducible targeted checks and results.
- `theory_repair_validation.json`: build, page-count, protected-file and source-manifest checks.
- `citation_audit.md` and its JSON/inventory remain the historical pre-correction audit; their old line numbers/statuses are superseded by this resolution note.

Primary sources used for the central corrections: [Bonnaire](https://arxiv.org/html/2505.17638v2), [George](https://proceedings.mlr.press/v300/george26a.html), [Liao–Couillet](https://proceedings.mlr.press/v80/liao18a.html), [Efron](https://pmc.ncbi.nlm.nih.gov/articles/PMC3325056/), and [Vershynin, Corollary 5.35](https://arxiv.org/pdf/1011.3027).
