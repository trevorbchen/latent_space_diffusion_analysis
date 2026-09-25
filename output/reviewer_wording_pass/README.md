# Revised manuscript and audits

Start with **[proof_repair.md](proof_repair.md)**. The manuscript/PDF now incorporates the mathematical audit repairs, including corrected training-step conversions. Unsupported spectral and sampler conclusions are explicitly conditional or heuristic; this is not a claim that all remaining research gaps have been proved.

- Manuscript: `main_iclr2027_9page.tex`; compiled snapshot: `main_iclr2027_9page.pdf`.
- Earlier edits: `changes.md`, `revision.patch`, `theory_repair.md`.
- Verification: `citation_audit.md`, `numerical_audit.md`, `proof_audit.md`.
- Reproduce the latest algebra and repair checks with `python3 proof_repair_checks.py` (requires NumPy).

The submitted abstract was preserved. The repaired PDF retains nine pages of main text; references and appendices follow. `proof_audit.md` and its JSON are historical findings against the pre-repair source; `proof_repair.md` maps each finding to its resolution.

The previous build used the system `fancyhdr.sty` in a temporary build directory because the bundled version is incompatible with the local older TeX installation. See `theory_repair_validation.json` for the recorded build checks.
