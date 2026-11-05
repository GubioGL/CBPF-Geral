#!/usr/bin/env bash
# Local build of the manuscript (Overleaf builds main.tex directly).   bash manuscript/build.sh
# Needs pdflatex + bibtex with revtex4-2 and tikz (Debian: texlive-publishers, texlive-pictures,
# texlive-latex-recommended, texlive-latex-extra). On Windows set MIKTEX_BIN to the MiKTeX bin folder.
set -u
cd "$(dirname "$0")"
PDFLATEX=pdflatex; BIBTEX=bibtex
if [[ -n "${MIKTEX_BIN:-}" ]]; then
  # --disable-installer: MiKTeX would otherwise stop and wait for a package-install prompt
  PDFLATEX="$MIKTEX_BIN/pdflatex --disable-installer"; BIBTEX="$MIKTEX_BIN/bibtex --disable-installer"
fi
latex() { timeout 300 $PDFLATEX -interaction=nonstopmode main.tex > /dev/null 2>&1; }
latex
timeout 120 $BIBTEX main > /dev/null 2>&1
latex
latex
latex
grep -n -E "^!" -A3 main.log | head -20
grep -E "LaTeX Warning: (Reference|Citation).*undefined" main.log | head
echo "overfull boxes: $(grep -c Overfull main.log)"
grep "Output written" main.log
