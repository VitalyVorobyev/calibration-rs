# Math notes — the proof-pack standard

Each algorithm family has a **proof pack**: the minimum evidence that the
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

Families: planar intrinsics (template, this directory),
Scheimpflug intrinsics (seeded), rig extrinsics, hand-eye, laserline
bundle, ringgrid detection/bias, rectification (short note — its gate exists), two-view/triangulation.

## Regression baselines

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
- **Solver benchmark** (`calib-bench solver`) measures the non-linear solve
  on deterministic synthetic scenes: all eight problem types (pinhole and
  Scheimpflug rig variants) × scale (small / medium / large) × pixel noise
  (0.1, 0.5 px) × outliers (none, or 5 % displaced by 10–30 px under Huber
  or Cauchy). Per scene it records init and optimize wall time (median of
  `--repeats` timed runs after a warm-up), the backend's iteration count, a
  solver-independent objective `½ Σ ρ(e²)` over the exported target
  residuals, inlier / all reprojection RMS, and ground-truth parameter
  errors. `calib-bench solver run --preset quick --out a.json` takes a
  fast reading (`--preset full` is the whole matrix; `--only` restricts the
  problems; `--md` writes the tables). To judge a change, run the same
  preset before and after and `calib-bench solver compare a.json b.json`:
  it joins scenes by id and flags optimize-time ratio, objective,
  inlier-RMS and ground-truth-error regressions. Timings are
  machine-specific, so compare runs from one machine; the objective and
  error columns are comparable anywhere.

## Notes

- [Planar intrinsics](planar-intrinsics.md) — Zhang init + Brown–Conrady
  refinement (the template pack).
- [Scheimpflug intrinsics](scheimpflug-intrinsics.md) — stub pack: model
  summary, the ADR 0022/0023 seeded route, and the convergence-basin
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
  DOF, rig-axis reuse, how the laser term fixes metric rig scale.
- [Two-view / triangulation](two-view-triangulation.md) — 8/7/5-point
  solvers, pose recovery + cheirality, DLT+GN triangulation; parallax
  degeneracy made quantitative.
- [Rectification](rectification.md) — short pack: Scheimpflug tilt as a
  normalized-plane homography; row-alignment evidence.
- [Ringgrid bias](ringgrid-bias.md) — the projective
  ellipse-center bias is already removed inside the `ringgrid` 0.7+
  detector; predicted bias (~0.09 px) is absent from the residual field,
  so the ~0.47 px floor is small-marker localization noise.
