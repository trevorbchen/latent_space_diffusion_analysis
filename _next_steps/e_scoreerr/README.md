# Corrected analytic score error for the synthetic-MLP sweeps (no retraining)

## What was wrong and what was recovered

The `score_error` column in every synthetic-MLP `metrics.jsonl` compared the model to an
analytic "true score" whose precision matrix was built in the wrong frame
(`Q_t.T @ diag(1/sigma_t) @ Q_t`, i.e. `Qᵀ D⁻¹ Q`, instead of `Q D⁻¹ Qᵀ`; fixed in commit
2bd8d60 in `code/experiment_v2.py::precompute_score_params` and `code/v3/lib/true_score.py::precompute_sigma_t`).
Training never used the true score, and `test_loss` is a valid held-out denoising-score-matching
loss, so the correct score error is recovered from the saved logs through the DSM identity

    E‖s_θ − s_cond‖² = E‖s_θ − s_true‖² + C,    C := E‖s_true(x_t,t) − s_cond‖²,
    s_cond := ∇_{x_t} log p(x_t|x_0) = −ε/√Δ_t,   Δ_t = 1 − e^{−2t},

whose cross term vanishes because E[s_cond | x_t] = s_true(x_t). C is model-free and is
computed here with the FIXED true score (`code/experiment_v2.py` at HEAD, imported directly).

## Exact weighting / time distribution matched (code/experiment_v2.py, HEAD)

The held-out loss is NOT averaged over the training-time distribution. `evaluate()` scores at a
single fixed time `t_eval = config.t_eval = 0.1` (line 317) on a fixed test set:

* line 321-322: `e_neg_t = exp(-t_eval)`, `delta_t = 1 - exp(-2 t_eval)`  (Δ = 0.181269)
* line 336:     `x_t_test = e_neg_t * test_dev + sqrt(delta_t) * test_noise_dev`
* line 339-341: `test_loss = mean_i ‖ sqrt(delta_t)·s_θ(x_t,i, t_eval) + ε_i ‖² / d`
  (sum over dims, mean over the 2048 test points, divided by d_latent)
* line 351-353: `score_error = mean_i ‖ s_θ − s_true_BUGGY ‖² / d` on the same `x_t_test`
* line 415-419: `test_data_fixed = generate_test_samples(means, k, d_int, d_lat, Q, sigma_noise, n=2048, seed=9999, sigma_signal)`
* line 420:     `test_noise_fixed = torch.randn_like(test_data_fixed)` — the first torch RNG draw after
  `torch.manual_seed(config.seed)` (line 395; lines 399-419 use only numpy), so ε is reproducible.
* line 399-403 / 90-139: data = k=10 GMM in d_int=5 signal dims (centres: orthogonal, norm `scale`=3,
  `sigma_signal`=1) plus isotropic null-space noise `sigma_noise` ∈ {0.5, 0.01} in the other d_lat−d_int
  dims, rotated by a Haar-random Q drawn from `np.random.default_rng(seed)` (line 130-132), n=500 train.
* Training (line 260-271) samples t ~ U[0.01, 3.0] and uses the same residual `‖√Δ s + ε‖²` summed over
  dims, mean over batch — but that only affects what was trained, not what `test_loss` measures.

Since `‖√Δ s_θ + ε‖² = Δ‖s_θ − s_cond‖²`, the identity gives, per run and per logged step,

    corrected_error          = test_loss · d / Δ − C_sample            (summed over dims)
    corrected_error_per_dim  = test_loss / Δ − C_sample / d            (directly comparable to score_error)

where `C_sample = mean_i ‖s_true(x_t,i) − s_cond,i‖²` is evaluated on the SAME 2048 (x_test, ε) pairs the
run used (regenerated from `seed` and `seed=9999`). A population estimate `C_pop` (200 000 fresh draws from the
same distribution, relative SE ≤ 0.23 % in every run, ≤ 0.15 % for the paper sweep) is also stored; the
column `corrected_error_per_dim_popC` uses it instead.

The evaluation is therefore at the single time t = 0.1 — the same t (and the same x_t points) at which the
buggy `score_error` was logged — not an average over the training-time distribution.

## Sweeps covered (all produced by `experiment_v2.py`; config keys match its `Config` dataclass exactly)

| name | directory | hidden | d_int / x | seeds | steps (eval every) | σ⊥ |
|---|---|---|---|---|---|---|
| `paper_h256_5seed_sn05_5M` **(paper Sec. 4 fixed-width sweep)** | `multiseed_runs/exp2_mlp_dlat_sn05_5m_bonnaire_hidden256` | 256 | d_lat ∈ {5,8,10,12,15,20,25,30,35,40} | 42–46 | 5M (50k) | 0.5 |
| `scaled_8dlat_5seed_sn05_5M` | `multiseed_runs/exp2_mlp_dlat_sn05_5m_combined` | 8·d_lat | same | 42–46 | 5M (50k) | 0.5 |
| `exp2_h256_sn05_300k` | `results_mlp_exp2_sn05` | 256 | d_lat 5…200 | 42 | 300k (5k) | 0.5 |
| `exp2_scaled_sn05_300k` | `results_mlp_exp2_scaled_sn05` (= `sigma_noise_0.5/exp2_mlp/raw_data`, `figures/...`) | 8·d_lat | d_lat 5…200 | 42 | 300k (5k) | 0.5 |
| `exp2_h256_sn001_300k` | `results_mlp_exp2_sn001` | 256 | d_lat 5…40 | 42 | 300k (5k) | 0.01 |
| `exp2_scaled_sn001_300k` | `results_mlp_exp2_scaled_sn001` (= `sigma_noise_0.01/exp2_mlp/raw_data`) | 8·d_lat | d_lat 5…200 | 42 | 300k (5k) | 0.01 |
| `exp3_dint_h256_sn05_300k` | `results_mlp_exp3_sn05` (= `sigma_noise_0.5/exp3_mlp/raw_data`) | 256 | d_int ∈ {2,5,8,12,16,20}, d_lat=20 | 42 | 300k (5k) | 0.5 |
| `exp3_dint_h256_sn001_300k` | `results_mlp_exp3_sn001` (= `sigma_noise_0.01/exp3_mlp/raw_data`) | 256 | same | 42 | 300k (5k) | 0.01 |

`results_mlp_exp3/` is a partial duplicate of `results_mlp_exp3_sn05` (3 runs, one unfinished) and was skipped.
The appendix's "fixed-width sweep through d_lat = 240" is not on disk (largest fixed-width d_lat is 200, seed 42, 300k steps).
Real-data dirs (`n1k_results/*mlp*`) are out of scope.

## Files

* `compute_corrected_score_error.py` — regenerates data/test set/noise per run, computes C (sample + population, fixed and buggy score), writes CSVs and `constants_<sweep>.json`.
* `make_figures_and_checks.py` — figures, `final_table_<sweep>.csv`, `sanity_checks.json`.
* `corrected_score_error_<sweep>.csv` — one row per (run, logged step): `d_lat, d_int, seed, hidden, sigma_noise, step, is_final, test_loss, train_loss, C, C_per_dim, corrected_error, corrected_error_per_dim, corrected_error_per_dim_popC, logged_score_error_BUGGY, C_bug_per_dim, mismatch_per_dim, memorization_fraction`.
* `fig_final_corrected_vs_<x>_<sweep>.{pdf,png}` — corrected per-dim error at the final step (mean ± 1 sd over seeds) vs the logged buggy curve and the model-free mismatch E‖s_true − s_bug‖²/d.
* `fig_corrected_vs_step_<sweep>.{pdf,png}` — corrected per-dim error vs training step for four widths.

## Sanity checks (all pass; details in `sanity_checks.json`)

1. `corrected_error ≥ 0` for every row of every sweep (smallest total value 0.11, in `exp3_dint_h256_sn001_300k`; smallest in the paper sweep 0.81). No negative rows.
2. Bug inert at d_lat = d_int: `C_bug_sample == C_fix_sample` to 1e-7 relative and mismatch M ≈ 1e-14 at d_lat = 5 (every seed) and at d_int = 20 = d_lat; M/d is 0.36–0.79 elsewhere for σ⊥ = 0.5 and up to 5.5 for σ⊥ = 0.01.
3. Noise replication: `test_loss(step 1) − ‖ε‖²/d` (regenerated ε) has sd 1e-4…1e-3 and max |·| 2.7e-3 across all 151 runs, versus ≈ 1.4e-2 expected if ε were not the run's own draw — the exact test set and noise are recovered.
4. Identity check on the fixed score: E‖s_cond‖² = d/Δ within 0.23 %, and E[(s_true − s_cond)·s_true] is < 0.13 % of C in every run (population MC), i.e. the cross term does vanish for the fixed score.
5. Precision: relative SE of C_pop ≤ 0.23 % (≤ 0.15 % in the paper sweep). The remaining unknowable term in a per-run corrected value is the finite-sample cross term on the 2048 fixed test points, 2·mean_i (s_θ − s_true)·(s_true − s_cond)/d, whose scale is ≈ 2√(e·c/N) ≈ 0.04 per dim for the paper sweep (e ≈ 2.3, c ≈ 3.5) — smaller than the seed-to-seed sd.
6. The audit's claim is confirmed: across d_lat, (logged buggy − corrected) tracks the model-free mismatch M/d with r = 0.96 (paper sweep), 0.95 (scaled 5-seed) and 0.99 for the 300k sweeps.

## Headline result (paper sweep, hidden 256, 5 seeds, 5M steps, t = 0.1)

The corrected per-dim error at 5M steps is essentially flat in d_lat: 2.59 ± 0.22 at d_lat = 5 and 2.26–2.35 for
d_lat = 8…40 (sd over seeds 0.03–0.12). The logged buggy curve's rise to 2.98 at d_lat = 10 and subsequent decay is
the mismatch term, not a model effect. Two caveats for interpretation:

* At 5M steps every model is far past its best: the corrected error is at or above the zero-score baseline
  E‖s_true‖²/d (0.71 at d_lat = 5, 1.4–2.4 for d_lat ≥ 8), consistent with the high memorization fractions
  and test_loss ≥ its step-1 value. The per-width minimum (0.09–0.27 per dim) is at the first logged eval
  (step 50k) for every d_lat, so the descent is not resolved by the 50k eval interval; the subsequent rise
  starts later for larger d_lat (see `fig_corrected_vs_step_paper_h256_5seed_sn05_5M.png`).
* The 300k-step, 5k-interval sweeps (`exp2_*_300k`) resolve the descent; at 300k the scaled-width σ⊥ = 0.01 models
  reach 0.01–0.05 per dim while the logged buggy values were 0.6–5.3 — there the logged curve was almost
  entirely mismatch.

## Final-step tables (mean over seeds; `sd` = population sd over seeds; `min over steps` = min over logged steps of the seed-mean curve)

### `paper_h256_5seed_sn05_5M`  (`multiseed_runs/exp2_mlp_dlat_sn05_5m_bonnaire_hidden256/`, x = d_lat)

Runs: 50; rows: 5050; min corrected_error (total, all rows): 0.813; negative rows: 0; max rel. SE of C_pop: 0.14%; eps-replication residual: mean 1.5e-04, sd 2.4e-04, max|.| 7.4e-04; corr[(buggy − corrected), M/d] = 0.959.

| x | seeds | test_loss | C/d | corrected/d (mean) | sd | zero-score E|s_true|^2/d | min over steps | argmin step | logged BUGGY | mismatch M/d |
|---|---|---|---|---|---|---|---|---|---|---|
| 5 | 5 | 1.344 | 4.822 | 2.590 | 0.217 | 0.709 | 0.216 | 50000 | 2.596 | 0.000 |
| 8 | 5 | 1.177 | 4.139 | 2.353 | 0.115 | 1.415 | 0.259 | 50000 | 2.847 | 0.601 |
| 10 | 5 | 1.121 | 3.873 | 2.312 | 0.080 | 1.650 | 0.274 | 50000 | 2.978 | 0.742 |
| 12 | 5 | 1.094 | 3.727 | 2.309 | 0.072 | 1.806 | 0.262 | 50000 | 2.918 | 0.735 |
| 15 | 5 | 1.057 | 3.574 | 2.259 | 0.073 | 1.963 | 0.235 | 50000 | 2.832 | 0.714 |
| 20 | 5 | 1.031 | 3.398 | 2.288 | 0.061 | 2.120 | 0.185 | 50000 | 2.710 | 0.583 |
| 25 | 5 | 1.020 | 3.313 | 2.311 | 0.042 | 2.213 | 0.162 | 50000 | 2.639 | 0.509 |
| 30 | 5 | 1.012 | 3.247 | 2.337 | 0.070 | 2.278 | 0.135 | 50000 | 2.586 | 0.434 |
| 35 | 5 | 1.003 | 3.191 | 2.342 | 0.049 | 2.323 | 0.113 | 50000 | 2.530 | 0.378 |
| 40 | 5 | 0.999 | 3.159 | 2.350 | 0.028 | 2.356 | 0.094 | 50000 | 2.503 | 0.358 |

### `scaled_8dlat_5seed_sn05_5M`  (`multiseed_runs/exp2_mlp_dlat_sn05_5m_combined/`, x = d_lat)

Runs: 50; rows: 5050; min corrected_error (total, all rows): 0.310; negative rows: 0; max rel. SE of C_pop: 0.14%; eps-replication residual: mean 4.6e-04, sd 5.8e-04, max|.| 2.7e-03; corr[(buggy − corrected), M/d] = 0.947.

| x | seeds | test_loss | C/d | corrected/d (mean) | sd | zero-score E|s_true|^2/d | min over steps | argmin step | logged BUGGY | mismatch M/d |
|---|---|---|---|---|---|---|---|---|---|---|
| 5 | 5 | 1.023 | 4.822 | 0.820 | 0.079 | 0.709 | 0.100 | 100000 | 0.832 | 0.000 |
| 8 | 5 | 0.975 | 4.139 | 1.238 | 0.095 | 1.415 | 0.100 | 50000 | 1.798 | 0.601 |
| 10 | 5 | 0.981 | 3.873 | 1.541 | 0.066 | 1.650 | 0.102 | 50000 | 2.179 | 0.742 |
| 12 | 5 | 0.987 | 3.727 | 1.715 | 0.033 | 1.806 | 0.100 | 50000 | 2.317 | 0.735 |
| 15 | 5 | 0.985 | 3.574 | 1.861 | 0.079 | 1.963 | 0.096 | 50000 | 2.409 | 0.714 |
| 20 | 5 | 1.002 | 3.398 | 2.131 | 0.056 | 2.120 | 0.102 | 50000 | 2.520 | 0.583 |
| 25 | 5 | 1.016 | 3.313 | 2.291 | 0.054 | 2.213 | 0.114 | 50000 | 2.599 | 0.509 |
| 30 | 5 | 1.018 | 3.247 | 2.371 | 0.096 | 2.278 | 0.119 | 50000 | 2.607 | 0.434 |
| 35 | 5 | 1.000 | 3.191 | 2.328 | 0.050 | 2.323 | 0.136 | 50000 | 2.523 | 0.378 |
| 40 | 5 | 0.993 | 3.159 | 2.318 | 0.026 | 2.356 | 0.144 | 50000 | 2.502 | 0.358 |

### `exp2_h256_sn05_300k`  (`results_mlp_exp2_sn05/`, x = d_lat)

Runs: 10; rows: 610; min corrected_error (total, all rows): 0.456; negative rows: 0; max rel. SE of C_pop: 0.14%; eps-replication residual: mean 1.1e-04, sd 2.5e-04, max|.| 5.2e-04; corr[(buggy − corrected), M/d] = 0.990.

| x | seeds | test_loss | C/d | corrected/d (mean) | sd | zero-score E|s_true|^2/d | min over steps | argmin step | logged BUGGY | mismatch M/d |
|---|---|---|---|---|---|---|---|---|---|---|
| 5 | 1 | 1.190 | 4.851 | 1.714 | 0.000 | 0.720 | 0.091 | 25000 | 1.745 | 0.000 |
| 8 | 1 | 1.078 | 4.168 | 1.778 | 0.000 | 1.424 | 0.120 | 20000 | 2.330 | 0.568 |
| 10 | 1 | 1.022 | 3.879 | 1.757 | 0.000 | 1.656 | 0.110 | 20000 | 2.432 | 0.763 |
| 15 | 1 | 0.979 | 3.596 | 1.802 | 0.000 | 1.964 | 0.091 | 10000 | 2.462 | 0.795 |
| 20 | 1 | 0.933 | 3.442 | 1.704 | 0.000 | 2.121 | 0.097 | 15000 | 2.107 | 0.509 |
| 30 | 1 | 0.839 | 3.253 | 1.373 | 0.000 | 2.281 | 0.073 | 20000 | 1.786 | 0.428 |
| 40 | 1 | 0.766 | 3.179 | 1.049 | 0.000 | 2.358 | 0.085 | 30000 | 1.413 | 0.359 |
| 100 | 1 | 0.574 | 3.034 | 0.130 | 0.000 | 2.498 | 0.123 | 220000 | 0.296 | 0.149 |
| 150 | 1 | 0.620 | 2.998 | 0.421 | 0.000 | 2.530 | 0.401 | 210000 | 0.523 | 0.103 |
| 200 | 1 | 0.643 | 2.974 | 0.576 | 0.000 | 2.545 | 0.576 | 300000 | 0.624 | 0.077 |

### `exp2_scaled_sn05_300k`  (`results_mlp_exp2_scaled_sn05/`, x = d_lat)

Runs: 11; rows: 671; min corrected_error (total, all rows): 0.504; negative rows: 0; max rel. SE of C_pop: 0.14%; eps-replication residual: mean 5.0e-04, sd 9.6e-04, max|.| 2.7e-03; corr[(buggy − corrected), M/d] = 0.988.

| x | seeds | test_loss | C/d | corrected/d (mean) | sd | zero-score E|s_true|^2/d | min over steps | argmin step | logged BUGGY | mismatch M/d |
|---|---|---|---|---|---|---|---|---|---|---|
| 5 | 1 | 0.906 | 4.851 | 0.145 | 0.000 | 0.720 | 0.101 | 155000 | 0.141 | 0.000 |
| 8 | 1 | 0.794 | 4.168 | 0.210 | 0.000 | 1.424 | 0.104 | 80000 | 0.817 | 0.568 |
| 10 | 1 | 0.751 | 3.879 | 0.264 | 0.000 | 1.656 | 0.092 | 55000 | 1.047 | 0.763 |
| 15 | 1 | 0.719 | 3.596 | 0.369 | 0.000 | 1.964 | 0.079 | 40000 | 1.159 | 0.795 |
| 20 | 1 | 0.741 | 3.442 | 0.646 | 0.000 | 2.121 | 0.089 | 20000 | 1.134 | 0.509 |
| 30 | 1 | 0.805 | 3.253 | 1.187 | 0.000 | 2.281 | 0.077 | 25000 | 1.608 | 0.428 |
| 40 | 1 | 0.862 | 3.179 | 1.577 | 0.000 | 2.358 | 0.088 | 20000 | 1.902 | 0.359 |
| 50 | 1 | 0.905 | 3.132 | 1.860 | 0.000 | 2.404 | 0.081 | 20000 | 2.064 | 0.275 |
| 100 | 1 | 1.019 | 3.034 | 2.589 | 0.000 | 2.498 | 0.102 | 15000 | 2.602 | 0.149 |
| 150 | 1 | 1.079 | 2.998 | 2.955 | 0.000 | 2.530 | 0.115 | 15000 | 2.907 | 0.103 |
| 200 | 1 | 1.133 | 2.974 | 3.278 | 0.000 | 2.545 | 0.140 | 15000 | 3.213 | 0.077 |

### `exp2_h256_sn001_300k`  (`results_mlp_exp2_sn001/`, x = d_lat)

Runs: 7; rows: 427; min corrected_error (total, all rows): 0.488; negative rows: 0; max rel. SE of C_pop: 0.14%; eps-replication residual: mean 5.5e-05, sd 2.8e-04, max|.| 5.2e-04; corr[(buggy − corrected), M/d] = 0.988.

| x | seeds | test_loss | C/d | corrected/d (mean) | sd | zero-score E|s_true|^2/d | min over steps | argmin step | logged BUGGY | mismatch M/d |
|---|---|---|---|---|---|---|---|---|---|---|
| 5 | 1 | 1.189 | 4.851 | 1.709 | 0.000 | 0.720 | 0.098 | 20000 | 1.747 | 0.000 |
| 8 | 1 | 0.752 | 3.054 | 1.095 | 0.000 | 2.515 | 0.090 | 20000 | 3.723 | 3.358 |
| 10 | 1 | 0.572 | 2.433 | 0.721 | 0.000 | 3.121 | 0.061 | 40000 | 4.968 | 4.888 |
| 15 | 1 | 0.334 | 1.628 | 0.211 | 0.000 | 3.913 | 0.043 | 40000 | 5.079 | 5.359 |
| 20 | 1 | 0.239 | 1.222 | 0.095 | 0.000 | 4.311 | 0.036 | 75000 | 3.287 | 3.417 |
| 30 | 1 | 0.153 | 0.814 | 0.031 | 0.000 | 4.718 | 0.023 | 160000 | 2.912 | 2.891 |
| 40 | 1 | 0.113 | 0.600 | 0.024 | 0.000 | 4.916 | 0.022 | 290000 | 2.507 | 2.412 |

### `exp2_scaled_sn001_300k`  (`results_mlp_exp2_scaled_sn001/`, x = d_lat)

Runs: 11; rows: 671; min corrected_error (total, all rows): 0.508; negative rows: 0; max rel. SE of C_pop: 0.14%; eps-replication residual: mean 5.0e-04, sd 9.7e-04, max|.| 2.6e-03; corr[(buggy − corrected), M/d] = 1.000.

| x | seeds | test_loss | C/d | corrected/d (mean) | sd | zero-score E|s_true|^2/d | min over steps | argmin step | logged BUGGY | mismatch M/d |
|---|---|---|---|---|---|---|---|---|---|---|
| 5 | 1 | 0.907 | 4.851 | 0.156 | 0.000 | 0.720 | 0.102 | 140000 | 0.147 | 0.000 |
| 8 | 1 | 0.572 | 3.054 | 0.104 | 0.000 | 2.515 | 0.076 | 170000 | 3.366 | 3.358 |
| 10 | 1 | 0.454 | 2.433 | 0.073 | 0.000 | 3.121 | 0.057 | 120000 | 4.951 | 4.888 |
| 15 | 1 | 0.304 | 1.628 | 0.048 | 0.000 | 3.913 | 0.039 | 280000 | 5.325 | 5.359 |
| 20 | 1 | 0.230 | 1.222 | 0.048 | 0.000 | 4.311 | 0.038 | 225000 | 3.423 | 3.417 |
| 30 | 1 | 0.152 | 0.814 | 0.027 | 0.000 | 4.718 | 0.024 | 250000 | 2.907 | 2.891 |
| 40 | 1 | 0.113 | 0.600 | 0.021 | 0.000 | 4.916 | 0.019 | 270000 | 2.452 | 2.412 |
| 50 | 1 | 0.092 | 0.489 | 0.017 | 0.000 | 5.037 | 0.017 | 295000 | 1.889 | 1.849 |
| 100 | 1 | 0.048 | 0.248 | 0.014 | 0.000 | 5.278 | 0.014 | 295000 | 1.053 | 1.008 |
| 150 | 1 | 0.033 | 0.161 | 0.021 | 0.000 | 5.356 | 0.019 | 265000 | 0.745 | 0.691 |
| 200 | 1 | 0.028 | 0.120 | 0.032 | 0.000 | 5.396 | 0.032 | 300000 | 0.593 | 0.520 |

### `exp3_dint_h256_sn05_300k`  (`results_mlp_exp3_sn05/`, x = d_int)

Runs: 6; rows: 366; min corrected_error (total, all rows): 1.127; negative rows: 0; max rel. SE of C_pop: 0.07%; eps-replication residual: mean 3.0e-04, sd 6.9e-05, max|.| 4.1e-04; corr[(buggy − corrected), M/d] = 0.823.

| x | seeds | test_loss | C/d | corrected/d (mean) | sd | zero-score E|s_true|^2/d | min over steps | argmin step | logged BUGGY | mismatch M/d |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 1 | 0.779 | 3.164 | 1.133 | 0.000 | 2.383 | 0.056 | 15000 | 1.387 | 0.244 |
| 5 | 1 | 0.942 | 3.442 | 1.753 | 0.000 | 2.121 | 0.097 | 10000 | 2.146 | 0.509 |
| 8 | 1 | 1.096 | 3.690 | 2.355 | 0.000 | 1.877 | 0.101 | 15000 | 2.884 | 0.821 |
| 12 | 1 | 1.317 | 3.995 | 3.268 | 0.000 | 1.558 | 0.100 | 15000 | 3.745 | 0.789 |
| 16 | 1 | 1.599 | 4.331 | 4.491 | 0.000 | 1.241 | 0.128 | 15000 | 4.532 | 0.484 |
| 20 | 1 | 1.766 | 4.656 | 5.089 | 0.000 | 0.923 | 0.107 | 10000 | 5.074 | 0.000 |

### `exp3_dint_h256_sn001_300k`  (`results_mlp_exp3_sn001/`, x = d_int)

Runs: 6; rows: 366; min corrected_error (total, all rows): 0.112; negative rows: 0; max rel. SE of C_pop: 0.22%; eps-replication residual: mean 2.9e-04, sd 6.4e-05, max|.| 4.1e-04; corr[(buggy − corrected), M/d] = 0.917.

| x | seeds | test_loss | C/d | corrected/d (mean) | sd | zero-score E|s_true|^2/d | min over steps | argmin step | logged BUGGY | mismatch M/d |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 1 | 0.090 | 0.487 | 0.008 | 0.000 | 5.013 | 0.006 | 145000 | 1.605 | 1.589 |
| 5 | 1 | 0.239 | 1.222 | 0.097 | 0.000 | 4.311 | 0.037 | 90000 | 3.284 | 3.417 |
| 8 | 1 | 0.457 | 1.933 | 0.590 | 0.000 | 3.631 | 0.055 | 55000 | 5.068 | 5.508 |
| 12 | 1 | 0.871 | 2.825 | 1.982 | 0.000 | 2.725 | 0.078 | 25000 | 5.178 | 5.204 |
| 16 | 1 | 1.351 | 3.752 | 3.700 | 0.000 | 1.824 | 0.124 | 15000 | 4.953 | 2.994 |
| 20 | 1 | 1.766 | 4.656 | 5.089 | 0.000 | 0.923 | 0.107 | 10000 | 5.074 | 0.000 |
