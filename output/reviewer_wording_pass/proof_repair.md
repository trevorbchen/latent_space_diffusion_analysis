# Repairs following the mathematical audit

The revised manuscript now corrects the audited algebra and normalization errors and removes unsupported conclusions. The submitted abstract and empirical main-text sections are unchanged. This repair does not turn the conditional spectral approximation or sampler model into proved general theorems.

| Audit finding | Resolution in the revised source |
|---|---|
| 1. Missing dimension divisor in optimizer clock | Derived `T = (2 Delta eta/d) steps = 0.02 steps` for default zero-momentum v2 SGD. Main signal e-fold estimates corrected to 788→8663 steps. Recomputed noise-dimension and sample-top e-folds; corrected tail exponents to approximately 0.003–0.011. Removed the tail-mass number evaluated at the wrong horizon and the unsupported inference from late test-loss minima to aligned sample modes. v3 remains explicitly separate. |
| 2. One-sided alignment cutoff | Replaced the false bound with the general absolute-alignment sum. Explained the additional absolute cutoff needed for the smaller bound and the condition for decreasing diagonal population loss. |
| 3. Divergence and interlacing | Replaced divergence with the finite `1/(4 Delta)` limit, included the shifted atom for aspect ratio above one, and retained indexed interlacing rather than a false global bracket. Global U-shape evidence is labeled numerical. |
| 4. Pointwise spectral localization | Kept the exact covariance identity. Individual spectral attachment now explicitly requires Gram/cloud operator bounds and eigenvalue separation. Removed the unsupported semicircle/MP inference and its downstream localization criterion. Point ordering is a consequence of a heuristic predictor, not a theorem about the learned sampler. |
| 5. Hermite coefficient | Included the fifth-Hermite contribution, correcting 50 to `266/5` in the residual expansion. Numerical monotonicity checks are distinguished from proofs. |
| 6. Fan-in and optimizer conventions | Predictor subsection explicitly uses already-scaled code weights with `tanh(Wx)` and corrected linear prefactor. Restored the dimension divisor in per-step rates; parameter redefinitions no longer imply unchanged optimizer dynamics. |
| 7. Gain rescaling | Fixed-q invariance now requires preserving correlation as well as variance. Updated the downstream collapsed-coordinate prediction. |
| 8. Invalid dead-coordinate limit | Restricted conditional spectral claims to `d<n<p`; removed the fixed-n divergence argument and spurious numerical prefactor. Distinguished a linear lower bound from relative growth and retained explicit fan-in dependence. |
| 9. Endpoint ceiling and ties | Recast the supposed ceiling as an ideal-score endpoint baseline; removed universal-bound predictions. Specified the initial covariance for the single-component ODE. Used strict order statistics for empirical fraction crossings, including ties. Removed the claim that observed time ratios alone refute a distance-based explanation. |
| 10. Condensation time factor | Restored `exp(2 t_c)` in the definition and practitioner recipe; retained the condition as a center-based heuristic, not a trajectory guarantee. |

Additional fixes include explicit finite-sample center/noise cross terms, the distinction between full rank and nonzero target activity, the correct direction of the minimum-of-sums inequality, nonzero finite-sample corrections at small noise, and the distinction between mean-iterate loss and expected stochastic loss. The unsmoothed predictor now uses the exact eligible-rate order statistic for a requested fraction; its calibration recipe exponentiates the median log constant.

## Validation

`python3 proof_repair_checks.py` passes the earlier deterministic algebra checks plus 1,000 signed-alignment bound trials, empirical order-statistic cases with ties, stale-claim checks, and the frozen-abstract hash check. The historical audit JSON and source hashes are preserved.

The rebuilt PDF retains nine pages of main text and 109 pages total. The LaTeX log has no overfull boxes or unresolved/multiply-defined references. Main pages and the theory appendix were rendered for layout review, with detailed inspection of the clock, gap bound, predictor, and collapsed-coordinate sections. Fonts, margins, and main layout settings were not reduced to achieve the page limit. Build details are recorded in `proof_repair_validation.json`.

The build uses the same local TeX compatibility workaround as the previous revision: system `fancyhdr.sty` in a scratch directory. The final PDF and LaTeX source are in this directory. `proof_repair.patch` records this repair relative to commit `1e0ce2a`; the earlier `revision.patch` records the earlier revision and is not a cumulative patch for this repair.

## Remaining limitations

No new training runs were performed. Missing operator-norm/localization proofs and the link from frozen features to trained samplers remain assumptions or research gaps, and are now stated that way. Historical empirical diagnostics are not presented as validation of the newly corrected condensation filter. The submitted abstract is preserved at the user's request; its stronger framing and the existing author-disclosure TODOs remain author-level decisions.
