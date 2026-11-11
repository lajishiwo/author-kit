# PIGeR manuscript status

Prepared 2026-11-11 (Asia/Shanghai), branch `paper/piger-cvpr2027`.

## Project

`main.tex` is the anonymous review entry point, using the repository's original CVPR 2027 article/style configuration. `cvpr.sty`, `preamble.tex`, and the official bibliography style are unchanged. The entry point selects standard T1 font encoding to retain the kit's Times fonts with both pdfLaTeX and XeTeX/Tectonic. Sections 0–5 contain the paper. `sec/results_main.tex` and `sec/results_tasks.tex` are generated from verified local metrics. No original architecture image was found in `fig/`; Figure 1 is explicitly a textual pending-artwork specification, not a substituted illustration.

Primary subject area: Multimodal Learning.

Keywords: Vision-Language Models; 3D Spatial Reasoning; Explicit Geometric Representation; Camera Pose Conditioning; RGB-D Perception; Multimodal Fusion; Continuous Latent Reasoning; Embodied Intelligence.

## Evidence

Local experiment root: `/data/djt/3DThinker/outputs/geolatent_cgmf_full25k_v1`.

Inspected `reports/CURRENT_MODEL_METHOD.md`, `RESULTS.json`, `STEP10000_COMPARISON.md`, `B0_REUSE_AUDIT.json`, `ENGINEERING_VERIFICATION.json`, and `PAIRED_STATISTICS.json`; inspected SpaceMind's local CGMF implementation. Six original 10K metrics files were compared with the report. CGMF and GENERIC completion/protocol records identify step10000, seed0, greedy BF16 decoding, 230400 pixels, 4096 output tokens, checkpoint hash, and protocol hashes. The protocol checkpoint hash was independently matched to each checkpoint MODEL.json manifest; checkpoint tensor files were not independently rehashed. B0 remains historical with incomplete original source-code provenance.

Portable snapshots `evidence/step10000.json` and `evidence/paired_statistics.json` contain no user-specific source paths. The main snapshot preserves source metrics SHA-256 hashes. The paper uses existing measurements; model training and inference were not run during manuscript preparation.

To recheck original artifacts and regenerate tables:

```bash
python3 scripts/prepare_evidence.py --source /data/djt/3DThinker/outputs/geolatent_cgmf_full25k_v1
```

Verified Full Overall: B0 42.3476468, CGMF 41.8993122, GENERIC 41.4392039. CGMF is not above B0. Both CGMF comparison confidence intervals include zero. Do not describe these results as an established improvement.

## Bibliography and policy sources

Bibliographic titles and complete author lists verified against arXiv metadata on 2026-11-11. Entries intentionally cite identifiable arXiv versions rather than guess publication page ranges or venue metadata:

- 3DThinker: https://arxiv.org/abs/2510.18632
- SpaceMind: https://arxiv.org/abs/2511.23075
- BIP3D: https://arxiv.org/abs/2411.14869
- SPAR: https://arxiv.org/abs/2503.22976
- LoRA: https://arxiv.org/abs/2106.09685
- Rotation representation: https://arxiv.org/abs/1812.07035

CVPR 2027 policy: https://cvpr.thecvf.com/Conferences/2027/AuthorGuidelines . Main text has an eight-page limit; references and the mandatory anonymous LLM Usage Statement are excluded. The statement belongs after the main text and before references. It discloses this Codex drafting process and explicitly awaits author confirmation of any other AI use; no unverified author declarations are asserted.

## Remaining research and submission work

- Complete matched CGMF/GENERIC 15K and 25K evaluations and additional training seeds.
- Scale up the existing six-question/six-scene engineering pilot; its M1 input and M4 latent-substitution checks are not evidence of a task-level gain or a reliable mediation effect. M5/M6 were not started according to the pilot report.
- Run paired correct/wrong pose and geometry tests; distinguish pose-embedding substitution from geometry re-extraction with perturbed extrinsics.
- Train K/V-only and gate-only conditioning ablations.
- Add latent analysis only when linked to answer behavior; no inference from gate magnitude alone.
- Strengthen B0 provenance or rerun the baseline under the exact frozen protocol.
- Replace the clearly marked architecture artwork placeholder with the actual supplied figure.
- Confirm data provenance, usage permissions, split construction and scene overlap checks with the authors; no compliance declarations have been invented.
- Author scientific review and complete LLM usage disclosure before submission.
- Review pending markers and final page layout; Overleaf uses `main.tex`.

No robot/VLA evidence exists. Initial manuscript preparation did not push remotely or synchronize Overleaf. The subsequent repository synchronization targets `origin/main`; Overleaf should use `main.tex` as its root document.
