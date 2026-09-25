# Numerical consistency audit — 2026-09-25

Scope: resolve the numerical discrepancies flagged by the wording pass. Edits are confined to this revised manuscript copy. The original `ICLR_2026/` files and the abstract in the revised copy are unchanged by this audit. No training was run.

## Findings and corrections

| Quantity | Evidence | Correction |
|---|---|---|
| E2 full-space memorization, d=20, 5M | Seeds 42–43: 3.8000%; seeds 42–46: 3.5800%. Independently recomputing the nearest-neighbor ratio test on all five saved sample arrays reproduces every CSV entry. | Section 3 comparison: **3.6%**, matching the five-seed table. |
| Empirical-score alignment, d=40, 5M | Two-seed mean 0.3000405; five-seed mean 0.3122756. | Appendix table: **0.31**, matching the main table. |
| Alignment, 250k, appendix row | Five-seed means at d=5,10,20,40: 0.537344, 0.366734, 0.217482, 0.090437. | Adjacent stale entries **0.53 → 0.54** and **0.36 → 0.37**. |
| Final corrected score error per coordinate | Five-seed mean: d=5 **2.589536**; d=8–40 **2.259075–2.353194**; d=240 **0.527729**. | State **2.59 at d=5 and 2.26–2.35 at d=8–40**. Replace exact-flatness language with comparable errors. |
| First-layer pre-activation ratio at 5M | Source note gives 0.30; new reproducible five-seed second-moment calculation gives **0.297223**. | Discussion: **0.30**, distinct from alignment 0.31. Define the measurement explicitly and update its trajectory consistently. |

The original main-sweep memorization value 3.7% at d=20 is a **different run collection** from the E2 checkpoint/sampling control's 3.6%; it was not globally replaced. E2 was retrained with the same configuration to retain checkpoints and generated samples.

## Pre-activation definition and provenance

The historical note `_next_steps/E2_RESULTS.md` labels its pre-activation quantity a variance, but gives neither executable measurement code nor the evaluation sample/noise protocol. Its values closely match an **uncentered second moment**, not either common centered-variance estimator. Therefore we do not claim exact reproduction of that historical protocol.

The revised manuscript instead uses a fully specified measurement, `q_M = mean(h**2)`, where `h` is the first linear layer output, including learned biases and Fourier time features. We evaluate existing checkpoints at t=0.1 on 4096 held-out points, population sampling seed 9999, noise seed 123, matching `e2_score_blocks.py`'s evaluation inputs. Average each run's moment across training seeds 42–46, then take the ratio of width means.

| Steps | q_M(d=5) | q_M(d=40) | Ratio |
|---|---:|---:|---:|
| 50,000 | 0.710131 | 0.684528 | 0.963945 |
| 250,000 | 1.118489 | 0.779579 | 0.696993 |
| 1,000,000 | 1.620885 | 0.762130 | 0.470194 |
| 5,000,000 | 2.813372 | 0.836199 | 0.297223 |

For comparison, the five-seed **global centered variances** at 5M are 1.932509 and 0.828331. The second moment cannot be described as that variance. It also is not identical to the frozen-feature theory's covariance-derived operating point. Both distinctions are explicit in the new appendix protocol.

No saved initialization checkpoint was found. The old initialization numbers and exact-equality assertion were removed; the revised claim is limited to similar moments at the first saved checkpoint (50k). This does not establish that an operating-point difference causes the delay or precedes memorization. These are evidence-driven corrections, beyond the earlier meaning-preserving rhetoric pass.

## Sources and computation

- `_next_steps/e2_results.csv`: five-seed projection/control measurements.
- `_next_steps/e2_score_blocks.csv`: squared distances to the population and empirical scores. Per run, alignment is reconstructed by polarization: `(true_sig + ref_sig - emp_sig)/(2*ref_sig)`; then averaged across seeds.
- `_next_steps/e2_runs/di5_d{5,20,40}_n500_s{42..46}/`: saved data, generated samples and model checkpoints used by the independent checks.
- `_next_steps/e_scoreerr/corrected_score_error_paper_h256_5seed_sn05_5M.csv`: five-seed fixed-width score-error series, including the extended sweep. All final rows satisfy the documented DSM recovery formula to numerical precision.
- `_next_steps/e_scoreerr/compute_corrected_score_error.py` and `README.md`: provenance of corrected score estimates. The correction uses an expectation identity; finite-sample cross terms prevent claiming exact checkpoint errors. The revised appendix now explains this limitation and the original diagnostic's rotation-frame bug. Training did not use the buggy analytic-score diagnostic.

Run `python3 output/reviewer_wording_pass/numerical_audit.py` from the repository to reproduce the key aggregates, independently check saved d=20 samples, and re-evaluate the trained-network moments. It writes `numerical_audit.json`, including per-run moments and SHA-256 hashes of the data/code inputs. It does not train models or modify source data. The score-error recovery itself was not rerun; the existing corrected table and its algebraic consistency were checked.

## Scope and validation

This resolves the flagged discrepancies; it is not an exhaustive audit of every numerical or theoretical claim. In particular, other timing assertions and strong claims about projected-test sensitivity were outside this check.

The revised PDF compiles without LaTeX errors or unresolved references/citations. Main text still ends on page 9; the complete PDF remains 108 pages. Original source files still match all hashes in `source_manifest.json`. `revision.patch` contains the complete original-to-revised TeX diff; `wording.patch` remains the historical rhetoric-only diff. Existing author-disclosure TODOs and unrelated figure-label defects remain.
