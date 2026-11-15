# PIGeR — CVPR 2027 anonymous manuscript

This project is being prepared only for CVPR 2027 submission. Open `main.tex` in Overleaf. The official CVPR style and preamble remain unchanged.

Build with `./scripts/build.sh` (TeX Live/latexmk or Tectonic), then run `python3 scripts/check_manuscript.py` (requires Poppler). The current PDF has eight total pages, including the LLM statement and references; the main text ends on page seven. It has no unresolved references, overfull boxes or font substitutions. See [validation](docs/BUILD_VALIDATION.json) for the existing dependency-package encoding warning.

The current paper includes verified results through 30K and 25K development input interventions. Tables can be regenerated without the original dataset using `python3 scripts/prepare_current_evidence.py`; the portable snapshot is included. The legacy 10K-only generator is retained for historical reference and should not overwrite the current tables.

See [paper status](docs/PAPER_STATUS.md) and the [revision report](docs/editorial/CVPR2027_CHANGE_REPORT_20261115.md) for evidence boundaries, reproducibility and remaining author checks. The [full Chinese translation](docs/editorial/CVPR2027_TRANSLATION_ZH.md) and [first-three-section translation](docs/editorial/FRONT_THREE_TRANSLATION_ZH.md) match the current English source. The manuscript retains the required AI-use disclosure. It still requires author scientific review and confirmation of AI use in other research stages.

Author-kit example supplementary/rebuttal files are retained but are not included in the paper. Architecture screenshots, editable PPT reconstructions and candidate input images are working assets, not observed experimental case studies. The source root is `main.tex`; repository branch `main` is used for synchronization when requested.
