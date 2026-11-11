#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if command -v latexmk >/dev/null; then
  latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
elif command -v tectonic >/dev/null; then
  tectonic --keep-logs main.tex
elif [ -x "$HOME/.local/bin/tectonic" ]; then
  "$HOME/.local/bin/tectonic" --keep-logs main.tex
else
  echo 'Install TeX Live with latexmk, or Tectonic.' >&2
  exit 1
fi
