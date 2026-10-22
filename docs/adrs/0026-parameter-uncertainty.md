# ADR 0026: Parameter Uncertainty as a Standard Result

## Context

Point estimates and reprojection statistics cannot say whether `fx` is known
to 0.1 px or to 10 px, which parameters trade off, or how well a pixel's
ray is determined: a fit to too few or too similar views reprojects well
and extrapolates badly. The LM's own linearization is not a usable source,
because it is loss-corrected differently by each backend.

## Decision

### Estimator

At the optimum, over the free tangent coordinates, the covariance is the
sandwich `Σ = H⁻¹ G H⁻¹` of Huber's M-estimator covariance, with its
moments pooled per residual family (reprojection px, laser point-to-plane
m, laser line px) and the priors' `sqrt_info` as known information. It is
built from a **loss-free** linearization at the optimum, once, with the
same normal-equation assembly for both backends. With one family, no priors
and no loss it is the classical `σ̂² (JᵀJ)⁻¹`. The derivation, its
identifiability analysis and the Monte-Carlo gates are in
`docs/notes/uncertainty.md`.

The weights are the solve's: `Σ` is the sandwich for the weights the solver
used, not the inverse information of an optimally weighted solve.

Each result also reports, per family, the robust noise level
`σ̂_f = 1.4826 · median |r|` as a diagnostic.

### What a result carries

- The full labelled covariance of the global parameters (every block that is
  not per view), and 6×6 marginals for per-view blocks; their
  cross-covariance with the globals is left out so the size stays linear in
  the views.
- Labels come from one source per block kind (`cam0.fx`, `handeye.rx`,
  `plane.nx`, distortion names from `DistortionKind::param_names()`).
  Coordinates follow ADR 0009; laser planes are reported over
  `[nx, ny, nz, d]`.
- The covariance is **conditional on everything fixed** — fixed
  coordinates, fixed blocks and earlier stages the final solve holds — and
  the result lists the fixed labels. Only the final solve feeds it.
- Projection uncertainty (the σ of where a pixel's ray lands, per camera,
  over a grid) is derived on demand from the covariance through
  `CalibratedCameras`, not stored in the export.

### Availability is a type

```text
Uncertainty = Available { covariance, view_marginals, noise, dof, fixed }
            | Unavailable { NotConverged | RankDeficient { unobservable }
                          | AtBound { labels } | NoRedundancy }
```

A solve that stopped on its iteration budget, a rank-deficient
`H` (with the labels spanning the null space), a free coordinate on its
bound, or no redundancy makes the covariance unavailable with a reason; the
calibration itself still succeeds.

### Always on

No config knob: one extra linearization, assembly and factorization, about
1.2–1.6× the solve time.

### Quality verdict

`analysis::assess(export, image_sizes, thresholds)` turns a result into a
`QualityReport`: a `Pass | Warn | Fail` verdict over typed checks, each with
its own status, measured value, unit and the threshold that decided it. It
reads the export only (no re-solve), through `CalibratedCameras`, so every
problem type shares one implementation of each check.

| Check | Fail | Warn |
|---|---|---|
| Observability | `RankDeficient`, `NoRedundancy` (names the labels) | `AtBound` (names the labels) |
| Convergence | `NotConverged` | — |
| Variance factor | σ̂²/σ₀² above the two-sided χ² range | below it |
| Projection σ (per camera, grid max incl. corners) | — | above `max_projection_sigma_px` (1 px) |
| Feature coverage (per camera, 8 × 6 cells) | below 0.3 | below 0.5 |
| View outliers (per camera, mean `\|r\|/σ̂`) | — | a view above 3 (names it) |
| Parameter correlation (per camera) | — | a pair above 0.99 (names it) |

- **Not applicable is not a pass.** A check whose inputs the result lacks
  is `NotApplicable` with the reason and does not enter the verdict: no
  stated σ₀ (it has no universal default), no image size (the manifest's
  regions, else the caller's), no covariance, or a camera whose parameters
  the final solve held fixed (rig results by default). A report with no
  assessed check is `Warn`. Ambiguous inputs (manifest and caller sizes that
  disagree, out-of-range thresholds) are errors, not guesses.
- **Verdict semantics.** `Fail` means the data does not support the result
  (an unobservable parameter, an unconverged solve, residuals the stated
  noise rejects, a quadrant's worth of coverage); `Warn` names a weakness
  that leaves the result usable. Projection σ, outliers and correlations
  only warn: the camera-frame map is conservative for uses that re-estimate
  the pose, a robust loss already bounds an outlier view, and correlation
  alone does not make a parameter undetermined.
- **χ² degrees of freedom.** A family's share of the residual dof, scaled
  by 0.3675, the Gaussian efficiency of the median-based σ̂ — so the test's
  width matches the scatter of the estimator it tests.
- **`fx`/`fy` are exempt** from the correlation check: the board distance
  scales both, so they correlate above 0.99 in every planar calibration
  of the acceptance set while their ratio is well determined.
- Convergence is read from the uncertainty status, the one place every
  export records it (the covariance is analysed only after convergence).
- The defaults leave every acceptance entry passing; they warn where a real
  weakness shows (corner extrapolation on the stereo pair, outlier views on
  dataset 8 and some private entries).

## Consequences

- Every export carries `uncertainty` (ADR 0012); Python exposes `std_dev`,
  `correlation` and the projection map; the app shows `value ± σ`.
- Staged problems whose final stage holds intrinsics fixed report
  conditional, hence optimistic, extrinsic σ; the fixed list says so.
  Propagating earlier-stage covariance is open.
- Defaults do not fix an observable block for speed, since that would make
  the default covariance conditional on an arbitrary choice.
- The regression suite gates parameter changes in units of σ.
- `calib-bench accept` gates on the quality verdict alongside the mean
  reprojection gate: a `Fail` fails the entry, a `Warn` is printed; an entry
  may override thresholds (`accept.quality`).

## Alternatives rejected

- **The LM's loss-corrected linearization** — differs between backends.
- **IRLS covariance with variance components** — overstates σ about twofold
  under Huber with outliers.
- **Per-residual sandwich** — calibrated on average, too noisy per run.
- **Bootstrap or Monte-Carlo at runtime** — tens of solves per result; it is
  the validation method instead.
- **Treating a parameter at its bound as fixed** — silently conditions the
  result.
