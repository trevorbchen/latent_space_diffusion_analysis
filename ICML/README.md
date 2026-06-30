# HiLD 2026 / ICML 2026 paper scaffold

All paper sources sit at the top level of this folder. The current
`main.tex` uses the **official ICML 2026 template** (`icml2026.sty`); the
HiLD JMLR-based class (`hld2026.cls`) is also present in case the
workshop strictly enforces it.

```
ICML/
├── main.tex              ← compile this
├── abstract.tex
├── sec-intro.tex
├── sec-mlp.tex           ← role of latent space in MLP training
├── sec-rfnn.tex          ← RMT / RFNN training dynamics
├── sec-discussion.tex
├── sec-appendix.tex
├── checklist.tex         ← reproducibility checklist (after appendix)
├── references.bib
├── figures/              ← drop final PNG/PDF figures here
│
├── icml2026.sty          ← official ICML 2026 style (currently used)
├── icml2026.bst
├── algorithm.sty
├── algorithmic.sty
├── fancyhdr.sty
│
├── hld2026.cls           ← official HiLD class (alternative)
├── jmlr.cls
├── jmlrutils.sty
│
├── icml2026_example/     ← unzipped ICML reference template
└── sample.* / HiLD_*.zip ← reference HiLD example, kept for switch-back
```

## Compiling

```bash
pdflatex main
bibtex   main
pdflatex main
pdflatex main
```

## Style: ICML 2026 (current)

`main.tex` uses `\documentclass{article}` + `\usepackage{icml2026}` per
the official `icml2026_example/example_paper.tex`. Key features:

- Two-column layout via `\twocolumn[ \icmltitle{...} ... ]`
- Author block via `\icmlauthorlist`, `\icmlauthor`, `\icmlaffiliation`
- `[anon]`-equivalent (default): the `icml2026` package strips author
  info automatically. Switch to `\usepackage[accepted]{icml2026}` for
  camera-ready.
- Bibliography style: `icml2026.bst`
- Appendix uses `\onecolumn` to escape the two-column layout

## Switching to HiLD class (if required)

If the workshop enforces `hld2026.cls`:

1. In `main.tex`, replace `\documentclass{article}` + `\usepackage{icml2026}`
   block with `\documentclass[anon]{hld2026}`.
2. Replace the `\twocolumn[ \icmltitle{...} \icmlauthorlist{...} ... ]`
   block with `\title[Running Title]{Full Title}` and
   `\hldauthor{\Name{...} \Email{...}\\\addr ...}` per `sample.tex`.
3. Replace `\bibliographystyle{icml2026}` with no styling (HiLD's class
   sets its own).

Section files (`sec-*.tex`) are class-agnostic and need no changes.

## Page limit

HiLD enforces **5 pages** of main text excluding references and
appendices. The current draft slightly overruns; trim the appendix or
compress the empirical section if needed.

## Figures

`main.tex` references three `\figplaceholder{...}` calls that render
labeled gray boxes:

| Where | Replace with (from project root) |
|---|---|
| `sec-intro.tex` overview fig | the four-panel four-bulk eigenvalue plot |
| `sec-mlp.tex` timescales fig | exp2_v2 timescales (4-panel: tau in steps + FLOPs) |
| `sec-rfnn.tex` four-bulk fig | per-d log-y eigenvalue cliff figure |
| `sec-rfnn.tex` timescales fig | RFNN tau_gen / tau_mem scatter |

Replace each `\figplaceholder{description}` with
`\includegraphics[width=\linewidth]{figures/your_file.pdf}` and update
the caption.

## Submission timeline

- **Deadline**: 11 May 2026 (AoE) via OpenReview.
- The 5M-step retraining runs are still in flight on ml6/ml7
  (`~/scaling/results_5M_scaled/`); pull final metrics back here when
  they finish and regenerate the timescale plots.

## Open content TODOs (priority order)

1. Replace the four `\figplaceholder{...}` calls with the final
   PNG/PDF figures.
2. Fill in the `bonnaire2025` BibTeX entry with full author list and
   final venue details once camera-ready proceedings are public.
3. Sketch (or strengthen) the replica-method derivation in
   `sec-appendix.tex` (Appendix B).
4. Decide whether the MLP U-shape stays as a secondary finding
   (current draft) or gets its own subsection.
5. Trim to 5 pages once figures are in place.
