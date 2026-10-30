# Math notes — the proof-pack standard

Each algorithm family has a **proof pack**: the minimum evidence that the
family is *sound* — not merely "tests pass", but "we can state what the
algorithm assumes, what it cannot observe, and how it degrades":

1. **Math note** (`docs/notes/<family>.md`, 1–3 pages): the model
   equations, the optimized cost, identifiability / degeneracy analysis
   (what is unobservable and why), gauge freedoms, and noise sensitivity.
   Written against the code — cite the modules that implement each
   equation.
2. **Synthetic-GT matrix test**: a ground-truth parameter grid × noise
   levels, every cell built with `vision_calibration::synthetic`
   (deterministic noise via `UniformPixelNoise`,
   `planar::project_views_noisy`) and pushed through the *standard*
   pipeline on both solver backends. Assert parameter recovery within
   stated tolerances and a residual consistent with the injected noise
   floor. Template: `crates/vision-calibration/tests/planar_intrinsics_matrix.rs`.
3. **Property tests** where a real invariant exists (round-trips, gauge
   invariance, rectification `|Δv|` bounds) — not where they would only
   restate the implementation.
4. **Regression gates**: a `calib-bench accept` entry (per-camera
   reprojection gate plus a committed Fit baseline) and/or a public
   regression-suite entry (`regression/suite.toml`, `calib-bench suite
   check`).

Initialization routines additionally get a **convergence-basin study**
(perturb the seed over a radius grid, measure gate-pass rate); the
Scheimpflug seeded init has one (`calib-bench basin`), the empirical
evidence behind ADR 0011.

## Regression baselines

- **Fit baselines** — `crates/vision-calibration-bench/baselines/`
  (committed; reprojection statistics only). `calib-bench accept` fails on
  drift beyond `--regression-tol` (default 5 %); a changed fit is accepted
  only by refreezing (`accept --freeze-baselines`) in a reviewed PR.
- **Performance baselines** — criterion's own mechanism, not committed
  (machine-specific): `cargo bench -p vision-calibration-linear --bench
  linear_init -- --save-baseline main` (likewise `-p
  vision-calibration-optim --bench ba_iter`), then `-- --baseline main` on
  the branch.
- **Solver benchmark** — `calib-bench solver run` / `compare` (see
  `calib-bench solver --help` and the `vision_calibration_bench::solver`
  module docs): every problem type on deterministic synthetic scenes,
  recording timing, a solver-independent objective and ground-truth
  errors. Compare timings only between runs on one machine. The reference
  backend comparison is
  `crates/vision-calibration-bench/results/solver-backends/compare.md`.

## Notes

- [Planar intrinsics](planar-intrinsics.md) — Zhang init + Brown–Conrady
  refinement (the template pack).
- [Scheimpflug intrinsics](scheimpflug-intrinsics.md) — partial pack:
  tilt↔focal↔distortion identifiability, the ADR 0011 seeded solve,
  and the convergence-basin study (`calib-bench basin`).
- [Hand-eye](hand-eye.md) — Tsai–Lenz AX=XB init + joint BA;
  motion-diversity identifiability, base-frame gauge, 180°-safe robot-pose
  ingestion.
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
- [Ringgrid bias](ringgrid-bias.md) — the projective ellipse-center bias
  is removed inside the `ringgrid` detector; the ring-grid reprojection
  floor is small-marker localization noise.
- [Parameter uncertainty](uncertainty.md) — the covariance every
  calibration result carries, across all families.
- [Detector feature covariance on real data](feature-covariance-study.md) —
  pre-registered study: does weighting by the detector's corner covariance
  improve held-out geometry or the honesty of the stated σ?
