# PIGeR — CVPR 2027 anonymous manuscript

Open `main.tex` in Overleaf. The official style and preamble remain unchanged.

Build locally with `./scripts/build.sh` (TeX Live/latexmk or Tectonic). This workspace has Tectonic 0.17.0 in `~/.local/bin`; its first build downloads TeX resources. No experiment data is required to compile: tables and a portable evidence snapshot are included.

After compilation, run `python3 scripts/check_manuscript.py` (requires Poppler). The current PDF has seven total pages, with no unresolved references, overfull boxes or font substitutions. See [build validation](docs/BUILD_VALIDATION.json) for checks and the harmless bundled-package encoding warning.

See [paper status](docs/PAPER_STATUS.md) for evidence provenance, verified references, regeneration instructions and research tasks. This is a complete manuscript draft with explicitly pending later-checkpoint experiments and architecture artwork. It is not yet a submission-ready scientific report.

The existing author-kit README and sample supplementary/rebuttal files are retained for reference. They are not included in the paper. GitHub/Overleaf synchronization should use branch `main` and root document `main.tex`. Manuscript development originated on `paper/piger-cvpr2027`.
