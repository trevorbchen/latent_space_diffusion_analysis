# Revised manuscript and audits

Start with **[proof_audit.md](proof_audit.md)**. The latest mathematical audit found unresolved substantive errors, including the conversion from flow time to repository training steps. The accompanying manuscript/PDF includes the earlier wording and citation repairs, **not fixes for the new proof-audit findings**.

- Manuscript: `main_iclr2027_9page.tex`; compiled snapshot: `main_iclr2027_9page.pdf`.
- Earlier edits: `changes.md`, `revision.patch`, `theory_repair.md`.
- Verification: `citation_audit.md`, `numerical_audit.md`, `proof_audit.md`.
- Reproduce the latest algebra checks with `python3 proof_audit_checks.py` (requires NumPy).

The submitted abstract was preserved during the latest repair/audit. The PDF was built before the proof audit; no page-limit recheck was needed for this report-only change. It is a working snapshot, not a claim that the mathematical issues have been resolved.

The previous build used the system `fancyhdr.sty` in a temporary build directory because the bundled version is incompatible with the local older TeX installation. See `theory_repair_validation.json` for the recorded build checks.
