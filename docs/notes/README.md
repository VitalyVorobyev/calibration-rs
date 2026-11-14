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
- [Tilt identifiability](tilt-identifiability.md) — synthetic sweep of
  the tilted fit: the tilt is a gauge without distortion
  (`K·H(τ) = K'·R(τ)`) and a decentred distortion with it, so it trades
  with `p1, p2` and the thin prism to first order; the map of σ(tilt) over
  field, distortion and view design, Monte-Carlo honesty, convergence
  basins, four parameterizations, the stopping rule in the valley.
- [Tilt determination](tilt-determination.md) — when the data determine a
  Scheimpflug tilt (the view-scatter-scaled σ), what fixing it costs
  downstream (only with `p1, p2` free is it the first-order equivalent),
  priors right and wrong, the frame each branch reports, and the procedure
  end to end with its decentring seed (ADR 0031).
- [Scheimpflug residual model](scheimpflug-residual-model.md) — what the
  tilted Brown-Conrady model leaves on a real rig: model variants on
  held-out views against the smooth-fit floor, the stopping rule, the
  tilt / principal-point reparameterization, blur dependence.
- [Camera residual field](camera-residual-field.md) — the camera-fixed
  part of the Scheimpflug residual: structured candidates against a
  designed B-spline field on held-out views, the observation-evaluated
  field artefact, orthogonality to the camera, Monte-Carlo σ (ADR 0030).
- [Parametric residual term](parametric-residual-term.md) — whether the
  camera-fixed field is a camera property: defocus, depth, display and
  detector alternatives on held-out views, compact empirical bases, the
  lens terms of decentred optics and their identifiability; why the
  residual-field slot is on hold.
- [Tilt from focus](tilt-from-focus.md) — per-marker blur fits the plane
  of sharp focus; the tilt it implies (direction with no datum, size with
  the pixel pitch and the pupil magnification) against the distortion
  route on exact thick-lens synthetic data and six real cameras; signed
  defocus against depth in the residual.
- [Tilt bias](tilt-bias.md) — the route refitted jointly with
  defocus/depth laws: how much of the focus–distortion tilt disagreement
  the coupling explains, which laws leave the tilt identified, the
  synthetic bias map of an unmodelled displacement, and safeguards for
  deciding when the data determine the tilt.
- [Display target residual](display-target-residual.md) — a PuzzleBoard on
  an e-ink display, six cameras fitted jointly on held-out captures:
  per-point in-plane offsets, board shape, cover-layer refraction and a
  view-angle shift against the camera's image field; the reachable floor.
- [Hand-eye](hand-eye.md) — closed-form init (Tsai–Lenz, Park–Martin,
  Daniilidis, Shah AX=ZB) + joint BA; motion-diversity identifiability,
  the methods' rotation-poor comparison, base-frame gauge, 180°-safe
  robot-pose ingestion.
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
- [MVG measurements](mvg-measurements.md) — object pose, laser points,
  metrology and the dense cloud: the implicit-function σ of each, the
  calibration's covariance as consider parameters, what each holds exact,
  the Monte-Carlo gates, and the drift-gated board measurement of the
  regression suite.
- [Ringgrid bias](ringgrid-bias.md) — the projective ellipse-center bias
  is removed inside the `ringgrid` detector; the ring-grid reprojection
  floor is per-detection localization scatter under defocus blur.
- [Parameter uncertainty](uncertainty.md) — the covariance every
  calibration result carries, across all families.
- [Detector feature covariance on real data](feature-covariance-study.md) —
  pre-registered study: does weighting by the detector's corner covariance
  improve held-out geometry or the honesty of the stated σ? No; it advises
  against it, and finds the stated projection σ 2–5 times short of the real
  split-half scatter for every arm.
