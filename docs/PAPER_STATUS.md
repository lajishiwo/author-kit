# PIGeR manuscript status

Updated 2026-11-15 (Asia/Shanghai). Target: anonymous CVPR 2027 submission only.

## Current manuscript

`main.tex` is the review entry point. The original `cvpr.sty`, `preamble.tex`, and bibliography style remain unchanged. Standard T1 encoding preserves Times fonts with Tectonic. Sections 0--5 contain the paper; four generated result tables cover checkpoint trajectories, terminal reasoning levels, all terminal task scores, and development interventions. The architecture figure is a text schematic of the measured implementation. The illustrative screenshots and reconstructed PPT files are not used as experimental evidence.

Primary subject area: Multimodal Learning.

Keywords: Vision-Language Models; 3D Spatial Reasoning; Explicit Geometric Representation; Camera Pose Conditioning; RGB-D Perception; Multimodal Fusion; Continuous Latent Reasoning; Embodied Intelligence.

## Verified evidence

Experiment root: `/data/djt/3DThinker/outputs/geolatent_cgmf_full25k_v1`.

Current source: `reports/all_nodes_20261114/{RESULTS.json,SOURCES.json,PAIRED_STATISTICS.json,VERIFICATION.json}` plus original metrics, predictions, protocol files, completion markers, checkpoint manifests and the 30K continuation protocol. Older status files are not the final-state authority.

`evidence/current_verified.json` preserves 31 checked records without user-specific absolute paths. This includes all 28 available formal Tiny/Full records at 10K/15K/20K/25K/30K and the three CGMF 25K development input conditions. Stored per-sample scores were reaggregated; the scorer was not rerun. Evaluation identities and questions match across compared conditions. Original metric/prediction hashes were checked. New-run checkpoint MODEL.json hashes were checked; tensor files were not rehashed. Existing bootstrap intervals were retained, not recomputed; point estimates were checked.

B0-20K is missing and not imputed. One empty answer each in B0-15K Full and CGMF-30K Full remains in the denominator. Historical B0 at 10K/15K/25K lacks complete original source-code hash provenance, inherited by its 30K continuation. The additional 5K steps restore optimizer/RNG state and consume 20K samples, for 1.2 epochs total. The continuation is reported as a follow-up rather than a fully prospective schedule.

30K Full Overall: B0 49.4893191, GENERIC 50.3447707, CGMF 48.5304005. CGMF trails both comparators. GENERIC exceeds B0 in this single-seed comparison but differs in both inputs and parameter budget. Do not describe the result as an established benefit of camera-specific conditioning or latent-mediated reasoning.

## Reproduce tables and compile

```bash
python3 scripts/prepare_current_evidence.py
./scripts/build.sh
python3 scripts/check_manuscript.py
```

To recheck original artifacts, add `--source /data/djt/3DThinker/outputs/geolatent_cgmf_full25k_v1` to the first command. The older `scripts/prepare_evidence.py` and `evidence/step10000.json` preserve the historical 10K-only analysis; the old generator would overwrite tables and should not be used for the current manuscript.

## Editorial record

- Pre-edit audit: `docs/editorial/CVPR2027_AUDIT_20261115.md`.
- Evidence-updated baseline for the separate language-only pass: `docs/editorial/CVPR2027_EVIDENCE_UPDATED_BASELINE.json`.
- Chinese change report: `docs/editorial/CVPR2027_CHANGE_REPORT_20261115.md`.
- Full current Chinese translation: `docs/editorial/CVPR2027_TRANSLATION_ZH.md`; the `FRONT_THREE_*` files now match the current first three English sections. Translation source hashes and coverage checks are recorded in `TRANSLATION_VALIDATION_20261115.json`.

Language-only edits preserve all numbers, equations, citation keys, reference keys and paragraph counts relative to the evidence-updated baseline. All original method formulas and bibliography entries are unchanged.

## Bibliography and submission requirements

Existing bibliography metadata was verified against the primary arXiv records on 2026-11-11 and has not changed in this language edit. Sources: 3DThinker 2510.18632; SpaceMind 2511.23075; BIP3D 2411.14869; SPAR 2503.22976; LoRA 2106.09685; rotation representation 1812.07035.

Official CVPR 2027 guidelines checked on 2026-11-15: https://cvpr.thecvf.com/Conferences/2027/AuthorGuidelines . Anonymous review mode and the mandatory LLM Usage Statement before references are retained. Main text ends on page 7; the current PDF has 8 total pages including disclosure and references. No author declarations, consent statements or submission enrollment facts have been invented.

## Remaining author/research work

- Author review of scientific claims and complete disclosure of any generative-AI use outside this editing process.
- Final architecture artwork if replacing the accurate text schematic; do not present synthetic chair-success screenshots as observed predictions.
- Additional training seeds and separate K/V-only / gate-only retraining ablations.
- Larger latent activation interventions distinguishing prefix and latent pathways.
- Appearance-preserving depth/pose interventions, since donor geometry changes more than metric information.
- Stronger historical B0 provenance or a matched rerun, and confirmation of data permissions, split construction and scene overlap checks.

The manuscript editing and translation did not train models, run inference, publish or create a version for another venue. Repository synchronization is performed only on explicit user request; the subsequent authorized synchronization is recorded in Git history. Overleaf synchronization itself is managed through the connected repository, not by an Overleaf UI action in this session.
