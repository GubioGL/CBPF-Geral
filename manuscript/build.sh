#!/usr/bin/env bash
# Local build of the manuscript (Overleaf builds main.tex directly).
# --disable-installer: MiKTeX would otherwise stop and wait for a package-install prompt.
set -u
cd "$(dirname "$0")"
B="${MIKTEX_BIN:-/c/Users/Gubio/AppData/Local/Programs/MiKTeX/miktex/bin/x64}"
latex() { timeout 300 "$B/pdflatex" --disable-installer -interaction=nonstopmode main.tex > /dev/null 2>&1; }
latex
timeout 120 "$B/bibtex" --disable-installer main > /dev/null 2>&1
latex
latex
grep -n -E "^!" -A3 main.log | head -20
grep -E "LaTeX Warning: (Reference|Citation).*undefined" main.log | head
echo "overfull boxes: $(grep -c Overfull main.log)"
grep "Output written" main.log
