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
Widening a diffusion model's latent space beyond the data's intrinsic dimension delays memorization, and we show why: extra latent coordinates de-saturate the score network's features and shrink the sample-specific modes that drive memorization.

## Keywords
diffusion models; memorization; generalization; latent diffusion; random features; random matrix theory; score matching; intrinsic dimension

## Primary area (suggested)
generative models   (alternative: learning theory)

## Abstract (plain text with OpenReview-safe math; no custom macros)
We study how the gap between latent and intrinsic dimensionality shapes memorization in diffusion models. Evaluating the random-feature spectral theory of Bonnaire et al. at the two-atom covariance of data embedded in a larger latent space, we find that the $d_{\mathrm{latent}} - d_{\mathrm{intrinsic}}$ excess dimensions form a distinct noise-dimension block in the score network's feature-correlation spectrum, sitting between the $d_{\mathrm{intrinsic}}$ signal modes and the $n - d_{\mathrm{latent}}$ sample-specific modes. Under gradient flow every mode decays in parallel at a rate equal to its eigenvalue, so the delay between generalization and memorization is the ratio of the smallest signal eigenvalue to the largest sample-specific eigenvalue, not a count of intervening modes. We show that this ratio grows with $d_{\mathrm{latent}}$ because extra latent coordinates lower the pre-activation variance of the random features, de-saturating the nonlinearity and shrinking the sample-specific feature mass by more than an order of magnitude; holding that variance fixed removes the growth. Random-feature spectra and trainable-MLP memorization curves on synthetic data isolate this mechanism, and the decline in memorization survives a dimension-controlled test that projects samples onto the true signal subspace. On CelebA and CIFAR-10 latent diffusion, once the VAE is wide enough to preserve pixel-space sample identity, increasing latent width reduces the memorization rate, while FID follows a separate quality tradeoff. Finally, from the frozen VAE latent spectrum we construct a mode-absorption clock that estimates the memorized fraction as a function of training step; the theory identifies the sample-specific term this clock was missing and reduces its free parameters to one.
