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
   (`calib-bench accept` per-entry gates today; Q2 adds baseline records
   and drift gates).

Initialization routines additionally get a **convergence-basin study**
(perturb the seed over a radius grid, measure gate-pass rate — Q6), the
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
