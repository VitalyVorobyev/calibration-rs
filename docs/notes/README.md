# Math notes — the proof-pack standard

Track Q of the production-grade program requires a **proof pack** per
algorithm family before v1.0. A proof pack is the minimum evidence that a
family is *sound* — not merely "tests pass", but "we can state what the
algorithm assumes, what it cannot observe, and how it degrades":

1. **Math note** (`docs/notes/<family>.md`, 1–3 pages): the model
   equations, the optimized cost, identifiability / degeneracy analysis
   (what is unobservable and why), gauge freedoms, and noise sensitivity.
   Written against the code as shipped — cite the modules that implement
   each equation.
2. **Synthetic-GT matrix test**: a ground-truth parameter grid × noise
   levels, every cell built with `vision_calibration::synthetic`
   (deterministic noise via `UniformPixelNoise`,
   `planar::project_views_noisy`) and pushed through the *standard*
   pipeline. Assert parameter recovery within stated tolerances and a
   residual consistent with the injected noise floor. Template:
   `crates/vision-calibration/tests/planar_intrinsics_matrix.rs`.
3. **Property tests** where a real invariant exists (round-trips, gauge
   invariance, rectification `|Δv|` bounds) — not where they would only
   restate the implementation.
4. **Committed regression Fit record + gate** via the bench machinery
   (`calib-bench accept` per-entry gates plus committed baseline records
   and drift gates).

Initialization routines additionally get a **convergence-basin study**
(perturb the seed over a radius grid, measure gate-pass rate), the
empirical evidence behind ADR 0022.

Families (backlog Q1/Q8): planar intrinsics (template, this directory),
Scheimpflug intrinsics (seeded), rig extrinsics, hand-eye, laserline
bundle, ringgrid detection/bias (Q3), rectification (short note — the C4
gate exists), two-view/triangulation.

## Regression baselines (Q2)

- **Fit baselines** live in `crates/vision-calibration-bench/baselines/`
  (committed; reprojection statistics only). `calib-bench accept` compares
  every run against them and fails on drift beyond `--regression-tol`
  (default 5 %); a changed fit is accepted only by refreezing
  (`accept --freeze-baselines`) in a reviewed PR.
- **Performance baselines** use criterion's own mechanism (not committed —
  they are machine-specific):
  `cargo bench -p vision-calibration-linear --bench linear_init -- --save-baseline main`
  (same for `-p vision-calibration-optim --bench ba_iter`), then compare a
  branch with `-- --baseline main`.

## Notes

- [Planar intrinsics](planar-intrinsics.md) — Zhang init + Brown–Conrady
  refinement (the template pack).
- [Scheimpflug intrinsics](scheimpflug-intrinsics.md) — stub pack: model
  summary, the ADR 0022/0023 seeded route, and the Q6 convergence-basin
  study (`calib-bench basin`) with measured basins for `rtv3d_ref` /
  `rtv3d_ringgrid`.
- [Hand-eye](hand-eye.md) — Tsai–Lenz AX=XB init + joint BA;
  motion-diversity identifiability, base-frame gauge, the 180°
  robot-pose-ingestion guard.
- [Rig extrinsics](rig-extrinsics.md) — per-camera Zhang + linear rig init
  + joint BA; reference-camera gauge, co-visibility requirements,
  narrow-vs-wide baseline noise trade.
- [Laserline bundle](laserline-bundle.md) — S² plane block + 1D stripe
  residuals; stripe non-collinearity identifiability, the soft distance
  DOF, rig-axis reuse.
- [Two-view / triangulation](two-view-triangulation.md) — 8/7/5-point
  solvers, pose recovery + cheirality, DLT+GN triangulation; parallax
  degeneracy made quantitative.
- [Rectification](rectification.md) — short pack: Scheimpflug tilt as a
  normalized-plane homography; row-alignment evidence (C4 gate).
- [Ringgrid bias](ringgrid-bias.md) — Q3 close-out: the projective
  ellipse-center bias is already removed inside the `ringgrid` 0.7+
  detector; predicted bias (~0.09 px) is absent from the residual field,
  so the ~0.47 px floor is small-marker localization noise.
- [rtv3d scale](rtv3d-scale.md) — Q5 close-out: the oracle-vs-measured
  scale gap was a pipeline-stage bookkeeping artifact; 5.2 mm cell
  confirmed, joint-BA hexagon matches the oracle to 0.08 %.
