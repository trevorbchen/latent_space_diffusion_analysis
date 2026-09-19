# ICLR 2027 abstract registration — paste-ready (prepared 2026-09-18 16:25 PDT)

**DEADLINE: Sep 18, 2026 11:59 PM AoE = Saturday Sep 19, 4:59 AM Pacific.** Full paper: Sep 25 AoE = Sat Sep 26, 4:59 AM Pacific.
Submit at https://openreview.net/group?id=ICLR.cc/2027/Conference  (official dates: https://iclr.cc/Conferences/2027/Dates)

Rules that bite tonight (https://iclr.cc/Conferences/2027/AuthorGuidelines):
- No authors can be added or removed after the abstract deadline. Order and the list must be final TONIGHT.
- Every author needs an OpenReview profile. Create it with the caltech.edu address: institutional emails activate right away,
  non-institutional profiles can sit in moderation for up to two weeks.
- Title and abstract can normally still be edited until the full-paper deadline; the author list cannot.

## Title
How Excess Latent Dimensionality Delays Memorization in Diffusion Models

## Authors (in the order on the draft)
1. Trevor Chen — tbchen@caltech.edu — California Institute of Technology
2. Ryan Shahbaba — shahbaba@caltech.edu — California Institute of Technology

## TL;DR (one sentence)
Widening a latent diffusion model's latent space beyond the data's intrinsic dimension delays memorization; we establish the effect with dimension-controlled tests on synthetic and image data and explain it in a solvable random-feature model.

## Keywords
diffusion models; memorization; generalization; latent diffusion; random features; random matrix theory; score matching; intrinsic dimension

## Primary area (suggested)
generative models   (alternative: learning theory)

## Abstract — USE THIS ONE (re-scoped 2026-09-18 evening: leads with the phenomenon, presents the theory as a solvable model, does not claim the
## random-feature mechanism explains the trained MLPs). Plain text with OpenReview-safe math; no custom macros.
Latent diffusion models operate in a latent space whose dimension $d_{\mathrm{latent}}$ typically exceeds the intrinsic dimension $d_{\mathrm{intrinsic}}$ of the data. We show that this excess delays memorization. For MLP score networks of fixed width trained on synthetic mixtures with known intrinsic dimension, the fraction of memorized samples after 5M steps falls from 30% at $d_{\mathrm{latent}}=5$ to near zero at $d_{\mathrm{latent}}=40$, and the divergence of the learned score from the population score begins progressively later as the latent widens. The decline is not an artifact of nearest-neighbor tests losing sensitivity in high dimension: it survives a dimension-controlled test restricted to the true signal subspace. On CelebA and CIFAR-10 VAE latents, once the VAE is wide enough to preserve sample identity, the time to memorize grows two- to four-fold with latent width, while FID follows a separate quality tradeoff. To understand the effect we evaluate the random-feature spectral theory of Bonnaire et al. at the two-atom covariance of data embedded in a larger latent space. The feature-correlation spectrum splits into signal, noise-dimension, sample-specific and null blocks; under gradient flow the delay between generalization and memorization is the ratio of two spectral edges, not a count of intervening modes; and this ratio grows with $d_{\mathrm{latent}}$ because extra coordinates lower the pre-activation variance and shrink the sample-specific feature mass, which a norm-fixed control confirms. We then test which parts of this mechanism carry over to trained networks, and use the latent spectrum to build a one-parameter clock for the memorized fraction during training.

## Earlier version (afternoon draft; overclaims the theory-to-MLP link in its fifth sentence — do not use)
We study how the gap between latent and intrinsic dimensionality shapes memorization in diffusion models. Evaluating the random-feature spectral theory of Bonnaire et al. at the two-atom covariance of data embedded in a larger latent space, we find that the $d_{\mathrm{latent}} - d_{\mathrm{intrinsic}}$ excess dimensions form a distinct noise-dimension block in the score network's feature-correlation spectrum, sitting between the $d_{\mathrm{intrinsic}}$ signal modes and the $n - d_{\mathrm{latent}}$ sample-specific modes. Under gradient flow every mode decays in parallel at a rate equal to its eigenvalue, so the delay between generalization and memorization is the ratio of the smallest signal eigenvalue to the largest sample-specific eigenvalue, not a count of intervening modes. We show that this ratio grows with $d_{\mathrm{latent}}$ because extra latent coordinates lower the pre-activation variance of the random features, de-saturating the nonlinearity and shrinking the sample-specific feature mass by more than an order of magnitude; holding that variance fixed removes the growth. Random-feature spectra and trainable-MLP memorization curves on synthetic data isolate this mechanism, and the decline in memorization survives a dimension-controlled test that projects samples onto the true signal subspace. On CelebA and CIFAR-10 latent diffusion, once the VAE is wide enough to preserve pixel-space sample identity, increasing latent width reduces the memorization rate, while FID follows a separate quality tradeoff. Finally, from the frozen VAE latent spectrum we construct a mode-absorption clock that estimates the memorized fraction as a function of training step; the theory identifies the sample-specific term this clock was missing and reduces its free parameters to one.
