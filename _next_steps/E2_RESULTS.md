# E2 — dimension-controlled memorization (run 2026-09-18, this Mac, CPU)

**Question.** Figure 2(a): memorized fraction 30% → 0.1% over d_lat 5 → 40. Real delay, or the NN-ratio test going blind in 40 dimensions?

**Runs.** `scripts/e2_run.py` wraps the paper's root `experiment_v2.py` unmodified (same Config as
`multiseed_runs/exp2_mlp_dlat_sn05_5m_bonnaire_hidden256`: hidden 256, Adam 1e-4, batch 256, 5M steps, n=500, σ⊥=0.5) and additionally saves
generated samples, checkpoints, the training set, means and Q. d_lat ∈ {5,10,20,40} × seeds {42,43}; ~2h50 wall, one core each.
Raw outputs: `_next_steps/e2_runs/` (≈0.4 GB, git-ignored). Replication check: full-space fraction equals the logged value in every row
and matches the paper's 5-seed sweep (29.5/16.7/3.8/0.1 % vs 30.0/17.1/3.7/0.1 %).

**Metric.** Bonnaire's form, d(gen,NN1)/d(gen,NN2) < 1/3 — what the paper's sweep uses. (`scripts/e2_analyze.py` → `e2_results.csv`)

| at 5M steps | d=5 | 10 | 20 | 40 |
|---|---|---|---|---|
| model, full-space test | 29.5% | 16.7% | 3.8% | 0.1% |
| model, signal-projected test | 29.5% | 16.9% | 5.8% | 1.2% |
| fresh population draws, projected (chance) | 0.5% | 0.5% | 0.5% | 0.5% |
| exact copies (+ terminal noise, all coords), full-space | 57% | 69% | 84% | 97% |
| signal copies (null coords redrawn), full-space | 57% | 0.8% | 0.0% | 0.0% |
| signal copies, projected | 57% | 57% | 58% | 58% |

**Verdict.** The headline survives. The full-space test IS blind to signal-only copies for d ≥ 10, but the models are not making them:
in the signal subspace memorization still falls 25× (29.5% → 1.2%, chance 0.5%). The full-space test overstates the decline mildly
(3.8 vs 5.8% at d=20; 0.1 vs 1.2% at d=40). Null-only test ≈ 0 everywhere: null coordinates are never copied.

**Score-space check** (`scripts/e2_score_blocks.py` → `e2_score_blocks.csv`). α = ⟨s_θ − s*, s_emp − s*⟩ / ‖s_emp − s*‖² in the signal
coordinates at t = 0.1 (0 = population score, 1 = exact fit of the training set):

| steps | d=5 | 10 | 20 | 40 |
|---|---|---|---|---|
| 0.05M | 0.12 | 0.08 | 0.03 | 0.00 |
| 0.25M | 0.53 | 0.36 | 0.22 | 0.09 |
| 1M | 0.66 | 0.49 | 0.36 | 0.24 |
| 5M | 0.76 | 0.54 | 0.39 | 0.30 |

α passes 0.3 at ≈0.14M steps for d=5 and at 5M for d=40 (>30× later). This also resolves the earlier puzzle (final corrected score
error flat at ≈2.3/dim for every width while memorization differs 25×): the per-coordinate gap ‖s_emp − s*‖² grows 3.7 → 13.2 (3.6×)
from d=5 to 40, so a smaller α costs as much population error.

**In the paper:** one paragraph in `sec-mlp.tex` after Figure 2, and `tab:e2-projection` in `sec-appendix-integrated.tex`.
**Caveats:** two seeds (paper sweep has five; agreement with it is within a point); synthetic GMM only — the real-data analogue
(project onto top-k latent PCs using the HF checkpoints) has not been run.
