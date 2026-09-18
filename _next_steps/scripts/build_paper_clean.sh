#!/bin/bash
# Clean local build of ICLR_2026/main.tex on this Mac (TeX Live 2016).
# The paper bundles fancyhdr.sty v5.2, which needs the LaTeX hook system (kernel >= 2020-10). On TeX Live 2016 it yields
# ~1700 "Undefined control sequence" errors and prints hook names ("para/before...") in every page header.
# Workaround: build in a scratch copy with the bundled file moved aside so the system fancyhdr loads. Sources are untouched.
# Overleaf / any modern TeX should keep the bundled file; do NOT delete it from the repo.
set -e
T="${1:-main}"   # target: main (ICML two-column draft) or main_iclr2027 (ICLR 2027 single-column submission format)
SRC="$(cd "$(dirname "$0")/../../ICLR_2026" && pwd)"; B="$(mktemp -d)"
cp "$SRC"/*.tex "$SRC"/*.sty "$SRC"/*.cls "$SRC"/*.bst "$SRC"/*.bib "$SRC"/*.bbl "$B"/ 2>/dev/null || true
cp "$SRC"/*.pdf "$SRC"/*.png "$B"/ 2>/dev/null || true; rm -f "$B/main.pdf" "$B/main_iclr2027.pdf"; ln -s "$SRC/figures" "$B/figures"
mv "$B/fancyhdr.sty" "$B/fancyhdr.sty.aside"
( cd "$B" && pdflatex -interaction=nonstopmode "$T" >/dev/null 2>&1; grep -q 'bibdata' "$T.aux" && bibtex "$T" >/dev/null 2>&1; pdflatex -interaction=nonstopmode "$T" >/dev/null 2>&1; pdflatex -interaction=nonstopmode "$T" >/dev/null 2>&1 ) || true
echo "errors: $(grep -ac '^! ' "$B/$T.log")   undefined refs/cites + duplicate labels: $(grep -ac 'Reference .* undefined\|Citation .* undefined\|multiply defined' "$B/$T.log")   pages: $(pdfinfo "$B/$T.pdf" | awk '/Pages/{print $2}')"
cp "$B/$T.pdf" "$SRC/$T.pdf"; rm -rf "$B"; echo "wrote $SRC/$T.pdf"
