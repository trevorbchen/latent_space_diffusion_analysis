# theory plan / who does what

ok so we need to actually nail down the theory section. right now appendix H is a sketch. it identifies three rank classes from the linearised tanh, lifts through W, and waves at a fourth rank-null class. no actual bulk edge constants, no theorem statement, and the sample-bulk gap is hand-wavy (see note below). this is the part of the paper that needs the most work.

three of us, three pieces. rough idea:
- **me (ryan)**: the four-bulk theorem (rank count + bulk-edge magnitudes). this is the largest piece and the one the rest of the section depends on
- **avni**: tau_mem - tau_gen lower bound corollary + a clean version of the bulk-gap-scaling formulas. depends on my edges being right but starts with the structure as a hypothesis so can be written in parallel
- **kevin**: U-shape recovery (MLP fig 14). more open-ended, take whichever angle (NTK / SAE / mean-field) feels tractable

we should sync before anyone goes too deep. if the rank-count argument stalls we may need to drop the full replica attempt and just state the count + numerically-verified edges.

---

## a thing i want to flag before we split up

the current appendix H derivation lowk has a hole. it does the linearised feature map
$$\phi(x) \approx Wx/\sqrt{pd_{\text{lat}}}$$
which gives $U^{\text{lin}} = W M_t W^\top / (p\, d_{\text{lat}})$, and at the linear level $U^{\text{lin}}$ has at most $d_{\text{lat}}$ nonzero eigenvalues. so:

- signal bulk: $d_{\text{int}}$ modes (inherited from $\Sigma_{\text{data}}$ signal block) ✓
- noise-dim bulk: $d_{\text{lat}} - d_{\text{int}}$ modes (inherited from $\Sigma_{\text{data}}$ null block) ✓
- a "sample fluctuation" contribution that the appendix lists as a third rank class of size $n$; but at the linear level; this just shows up as finite-$n$ noise on top of the two data bulks. it does **not** give a separately gapped sample bulk.

the sample bulk as a separate gapped class only appears once you include the **Hermite-2 / quadratic** correction to tanh, which is the Pennington–Worah term:
$$\phi_a(x)\phi_b(y) \approx \mu_1^2 (Wx/\sqrt{d})_a (Wy/\sqrt{d})_b + \mu_2^2 [(Wx/\sqrt{d})_a(Wy/\sqrt{d})_b]^2 + \cdots$$
(with $\mu_1, \mu_2$ the Hermite coefficients of $\tanh$). the $\mu_2$ piece behaves like an inner-product kernel and contributes a rank-$n$ block, that's the sample bulk. with this in place the rank-null tail size is $p - d_{\text{lat}} - n$, which is the number the appendix already quotes.

so the appendix's rank-count is **right** but the derivation as written doesn't justify it. fixing this is the first thing on my list: either include the $\mu_2$ term explicitly, or be honest that we're appealing to the nonlinear-RMT literature for the rank-$n$ piece.

---

## me: four-bulk theorem (sections 2a + 2c in NEXT_STEPS)

**goal.** turn appendix H into an actual statement we can label `Theorem 1`. needs to give:

1. **counts.** exactly $d_{\text{int}}$ signal modes, $d_{\text{lat}} - d_{\text{int}}$ noise-dim modes, $n$ sample modes, $p - d_{\text{lat}} - n$ rank-null modes. assumes $p > d_{\text{lat}} + n$ which is satisfied for us ($p = 64 d_{\text{lat}}$, $n=500$).

2. **edges, leading order.** explicit asymptotic magnitudes, at minimum

   | bulk | leading scale |
   | --- | --- |
   | signal | $\mu_1^2 \cdot e^{-2t}(s^2/d_{\text{int}} + \sigma_{\text{sig}}^2) + \mu_1^2 \Delta_t$ |
   | noise-dim | $\mu_1^2 \cdot (e^{-2t} \sigma_\perp^2 + \Delta_t)$ |
   | sample | $\mu_2^2 \cdot e^{-4t} \sigma_{\text{sig}}^4 / n$ (or similar, double-check the Pennington-Worah scaling) |
   | rank-null | $0 + O(d_{\text{lat}}/p)$ corrections |

   (these are my back-of-envelope numbers, work them out properly. i suspect i'm off by factors of $p/d_{\text{lat}}$ from the lift.)

3. **assumptions.** linear regime $s/\sqrt{d_{\text{lat}}} \lesssim 2$ (already characterised in appendix I), proportional asymptotic $d_{\text{lat}}, n, p \to \infty$ with $\psi_p = p/d_{\text{lat}}, \psi_n = n/d_{\text{lat}}$ fixed.

**plan of attack.** two routes:

- **route A (simpler, lower payoff).** rank-counting argument + leading-order edge magnitudes from $\mathbb{E}\, U$. requires (i) writing $U = U^{\text{lin}} + U^{\text{quad}} + O(\|Wx\|^6/d^3)$ via Hermite expansion of tanh, (ii) showing $\mathrm{rank}(U^{\text{lin}}) = d_{\text{lat}}$ a.s. and inherits the block structure of $\Sigma_{\text{data}}$, (iii) showing $U^{\text{quad}}$ adds a rank-$n$ block orthogonal (to leading order) to the linear range. concentration via standard sub-Gaussian tools.

- **route B (full).** full Stieltjes-transform / replica derivation, extending Bonnaire's $\Sigma_{\text{data}} = I$ proof to block-diagonal. fixed-point equations à la El Karoui / Pennington–Worah. gives closed-form bulk edges as roots of a polynomial system, not just leading-order. i want to try this first; if it stalls i'll drop to route A and live with leading-order edges.

things for me to do:
- [ ] write up the route-A rank-count proof in full + check the counts against table 3 in the paper (it's already in `code/v3/results/.../bulk_summary.jsonl`)
- [ ] leading-order edge calculation; check against `sigma_noise_0.5/four_bulk/`
- [ ] attempt route B. if the fixed-point eq for block-diagonal $\Sigma_{\text{data}}$ closes in finite form, write it up
- [ ] theorem statement + proof in the paper, replace appendix H

things i'm worried about:
- the $\mu_2$ term in Pennington–Worah is for $\mathbb{E}_W$, not the empirical $U$. our $U$ has $W$ fixed and averages over diffusion noise + training data. need to check whether the rank-$n$ structure survives at fixed $W$ or whether we're implicitly using a self-averaging argument.
- $\mu_0 \neq 0$ for tanh on non-centered inputs (cluster centers shift the input distribution). need to subtract the mean or argue it gets absorbed.
- the "lift" $W^\top W / d$ has its own MP spectrum on $[(\sqrt{\psi_p} - 1)^2, (\sqrt{\psi_p} + 1)^2]$. this *spreads* each bulk edge. what i'm calling a single edge above is actually a Marchenko–Pastur-deformed band of width $O(\sqrt{1/\psi_p})$. probably small in our regime ($\psi_p = 64$) but should be quantified.

---

## avni: tau_mem lower bound (2b) + bulk-gap formulas

**goal.** once the four-bulk structure is established, derive
$$\tau_{\text{mem}} - \tau_{\text{gen}} \;\ge\; \frac{1}{\lambda_{\min}^{\text{noise-dim}}} \cdot (d_{\text{lat}} - d_{\text{int}})$$
(or some cleaner version, where the right-hand side makes explicit that the delay scales linearly in the buffer width times the per-noise-dim-mode timescale).

**plan.**
1. start from the per-mode ODE $a_i(T) - a_i^\star = (a_i(0) - a_i^\star) e^{-\lambda_i T}$.
2. define $\tau_{\text{gen}}$ as the time the signal bulk is absorbed within tolerance $\epsilon$: $T_{\text{sig}}(\epsilon) = -\log\epsilon / \lambda_{\min}^{\text{signal}}$.
3. define $\tau_{\text{mem}}$ as the time the sample bulk starts being absorbed beyond a fixed fraction. some care: memorization is about *fitting* sample modes, generalization is about fitting signal modes. make sure the definitions line up with what we measure empirically (somepalli fraction > 1%, gen-gap > 0.02).
4. the gap is $\tau_{\text{mem}} - \tau_{\text{gen}}$ = time to traverse noise-dim bulk = $\Omega((d_{\text{lat}} - d_{\text{int}}) \cdot \tau_{\text{noise-dim}})$ via a sum-of-exponentials lower bound.

this is a real proof but it's short. depends on my edge estimates for $\lambda_{\min}^{\text{noise-dim}}$, which from the table above should be around $\mu_1^2(e^{-2t}\sigma_\perp^2 + \Delta_t)$. write the bound symbolically first, plug in the edge magnitude later.

bonus: also clean up the **bulk-gap scaling** paragraph at the end of appendix H. currently it says
- signal-to-null gap is $\Theta(s^2 / (d_{\text{int}} \sigma_\perp^2))$ or $\Theta(s^2/(d_{\text{int}}\Delta_t))$ depending on which dominates
- null-to-sample gap is $\Theta(\sqrt n \sigma_\perp^2 / \sigma_{\text{sig}}^2)$

these should be derivable from my edge magnitudes. make it consistent, drop the $\Theta$ for explicit ratios. helpful to also write the gap-closing thresholds: at what $\sigma_\perp$ does the noise-dim bulk merge with the sample bulk? (probably $\sigma_\perp \sim \Delta_t / \sqrt n$ or thereabouts.)

things for avni to do:
- [ ] draft the lower bound as a lemma with my edges as inputs
- [ ] verify numerically against fig 6 / fig 21. does the predicted $\tau_{\text{mem}}$ slope-with-$d_{\text{lat}}$ match the observed slope?
- [ ] write the bulk-gap-scaling paragraph

---

## kevin: U-shape recovery (2d, exploratory)

**goal.** explain figs 14 + 16: late score error vs $d_{\text{lat}}/d_{\text{int}}$ has an n-shape on the MLP at $\sigma_\perp = 0.01$ (peak around $d_{\text{lat}}/d_{\text{int}} = 3$, then recovers) that the RFNN never shows. our hypothesis is sparse-feature compression: the trainable first layer learns to project onto the $d_{\text{int}}$-dim signal subspace at large $d_{\text{lat}}$, effectively shrinking the buffer.

this is the most open-ended thread. i don't actually know if any of these angles lead to clean math. try a few, see what bites.

**candidate frameworks** (in rough order of how much they'd buy us):

1. **SAE / sparse-feature story.** for the linearised MLP with ReLU/GELU first layer, prove (or empirically show) that gradient descent finds a sparse first-layer weight matrix whose row support concentrates on the $d_{\text{int}}$ signal directions when $d_{\text{lat}} \gg d_{\text{int}}$. would give an analytic curve for score-error($d_{\text{lat}}$) with the n-shape. probably needs an inductive bias argument or a specific initialisation / loss-landscape claim. closest references: anthropic SAE papers, sparse-coding-meets-NN literature.

2. **NTK reweighting.** the NTK kernel inherits the four-bulk spectrum, but the trained network reweights modes by an effective importance function. if at large $d_{\text{lat}}$ the trained-MLP NTK upweights signal modes and downweights noise-dim modes, you get the recovery. specific calc: write the score-error as a sum over NTK eigenvalues weighted by target-projection magnitudes; ask where the n-shape comes from.

3. **two-competing-terms heuristic.** model the late score error as $\mathcal{E}(d_{\text{lat}}) = \mathcal{E}_{\text{fit}}(d_{\text{lat}}) + \mathcal{E}_{\text{approx}}(d_{\text{lat}})$ where fit-cost grows in $d_{\text{lat}}$ (more noise-dim modes) and approximation cost falls (more capacity). a single crossover gives one minimum, not an n-shape, but with the right asymptotic forms maybe we get both a peak and a recovery. probably the most hackable angle.

**experimental backing** (avni and i already have some of this, sync with whoever runs sweeps):
- weight-projection: project trained MLP first layer onto ground-truth signal subspace (we have $Q$ from data gen). signal-subspace mass should grow with $d_{\text{lat}}$.
- activation density: fraction of nonzero post-activation units per neuron, by $d_{\text{lat}}$. should drop at high $d_{\text{lat}}$ if SAE story is right.
- L1 / top-k ablation: explicit sparsity should amplify the recovery.

things for kevin to do (loose, since this is exploratory):
- [ ] pick a framework, write the model down, do the back-of-envelope calc to see if it predicts n-shape at all
- [ ] nail down quantitative match against fig 14. if framework predicts n-shape but at wrong $d_{\text{lat}}$, that's still informative
- [ ] write up best-effort version, include even if results are negative (we'll frame it as "the RFNN theory fails to predict the MLP n-shape, here are three frameworks that partially capture it")

if none of these work cleanly, the paper-grade fallback is just to write the SAE story as a hypothesis backed by the empirical probes (weight-projection, activation density) without claiming a theorem.

---

## shared concerns / open questions

- **multi-seed.** everything theoretical assumes the limiting spectrum is a deterministic object. we're single-seed (seed 42). before claiming bulk-edge magnitudes match within X%, need at least 3 seeds. someone needs to babysit a multi-seed rerun, probably whoever finishes first.
- **$t$-dependence.** all of the above is at fixed $t = 0.01$ (RFNN) or averaged over $t \sim U[0.01, 3]$ (MLP). does the four-bulk structure survive averaging? back-of-envelope says yes (it's preserved at every $t$) but should check.
- **what about $\sigma_\perp \to 0$?** the noise-dim bulk's edge collapses to the diffusion floor $\Delta_t$, and at small $t = 0.01$ this is $\approx 0.02$. the bulk doesn't fully disappear, it just sits on the diffusion floor. this is probably where the n-shape lives. my edges should make this clear.

## stuff that's NOT in scope here

- real-data four-bulk verification (separate thread)
- curved manifolds (thread 4 in NEXT_STEPS, later)
- the activation ablation / different noise schedules (lower priority)


im like chud sleepy so if this doesnt make sense my bad. but that's the plan. i'll kick off the rank-count proof right now, ideally avni starts the lemma statement in parallel using placeholder edges, and kevin picks one of the three U-shape frameworks and runs with it. we also don't need like to finish all of this by conference cus we are low on space and its a workshop lol.
