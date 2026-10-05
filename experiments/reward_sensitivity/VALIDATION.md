# Implementation validation (2026-10-06)

- Full Python suite: **135 passed**.
- PBS syntax and diff whitespace checks: passed.
- Simulator checks: A/B/C each trained 100 steps, saved and reloaded its model,
  evaluated one short scene, and exported analysis with the actual custom weights.
- A and C completed original/no-front/matched-speed-front/far-front short
  counterfactual rollouts. Multi-lane custom weights were restored from metadata.
- Reproducible sampling and all 90 job mappings checked. Default weights retained.
- Existing baseline collector: 45/135 condition records available; the 90 new
  configurations have not been trained. Complete export refuses missing records.
- Collector test distinguishes physical lane change (2.4 s) from action onset
  (1.8 s), retains non-onsets, excludes infinite spacing explicitly, and rejects
  a mismatched training-scenario prefix.
- R-only figure preview: PDF/SVG/PNG/TIFF exports generated using existing
  baseline data, no synthetic coefficient curves. Both PNGs visually inspected:
  labels/captions readable, no clipped content, rate axes bounded by 0/1 ticks.
  Median/IQR and five-seed definition are stated; no significance claims.

The full three-curves-per-panel visual layout will require a second visual
inspection when new training data are available. Short smoke models are not
included in the scientific collection. Formal training and Katana submission
remain to be run with the maintained cluster environment.
