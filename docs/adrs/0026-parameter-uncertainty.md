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

### Robot-pose priors

Hand-eye problems that refine the robot poses give every view a correction
`δᵢ ∈ se(3)` with a zero-mean prior of the configured `rot_sigma`,
`trans_sigma`: the robot's pose error, in radians and metres. The
reprojection residuals are in pixels with an unknown noise `σ_px`, so the
prior is weighed in pixels: its residual is `σ_px · L δᵢ`
(`L = diag(1/σ_rot, 1/σ_t)`) and the covariance counts it with standard
deviation `σ_px` (`Se3TangentPrior::residual_sigma`, `G_p = σ_px² A_p`). The
objective is then the maximum-likelihood one scaled by `σ_px²`, and the
sandwich reduces to the inverse information. `σ_px` comes from the data:
one solve at `σ_px = 1`, then one from that solution at the reprojection
`σ̂` (floored at 0.01 px so noise-free data keeps its priors). A second
re-solve moved nothing measurable.

Every view's correction is free. The priors alone separate the corrections
from a hand-eye or target change that moves every view alike (the 6 + 6
directions the corrections share with those poses); fixing one view's
correction instead takes that view's robot pose as exact, and the
covariance then misses its error.

Rejected:

- **Pixels weighed at 1 px against the priors.** With sub-pixel corners the
  priors are relatively too strong: robot error leaks into the camera
  parameters, and with view 0 fixed the hand-eye σ were 2–2.6× too small on
  synthetic scenes.
- **Rescaling the reprojection family by `1/σ̂`.** The same estimate and
  covariance (a unit test holds the two equal), but the residuals, their
  reported `σ̂` and a robust loss's scale would no longer be in pixels.
- **Exact robot poses.** The robot's error enters the corner residuals as an
  error shared by every corner of a view, which the covariance counts as
  independent noise: it is too small by about `√(corners per view)`.

The configured standard deviations are taken as the robot's true accuracy;
estimating them from the corrections is open.

Each result also reports, per family, the robust noise level
`σ̂_f = 1.4826 · median |r|` as a diagnostic.

### Per-point pixel covariance

An observation may state each image point's 2×2 covariance `Σ`
(`PointCovariance`, px², validated finite and positive definite on
construction and deserialization). It replaces the scalar per-point weight,
which had no source and no meaning for the noise estimate; a weight `w` is
the covariance `I/w` (the residual was scaled by `√w`).

- **Whitening, not a weight.** `ReprojPoint` carries a square-root
  information `ReprojSqrtInfo`: `Isotropic { scale }` (`L = scale·I`, the
  identity without a stated covariance, bit-identical to the former `w = 1`
  path) or `Whitened { upper }` (`L` upper triangular, `LᵀL = Σ⁻¹`). The
  residual is `L (u_obs − u_proj)` in both backends, the same generic kernel;
  a family weight (`calib_weight` in the laser bundles) multiplies the
  information, so `L` by `√w`. The laser factors keep scalar weights.
- **All or none per input.** Whitened residuals are dimensionless, pixel
  residuals are not; one family pooling both would estimate a meaningless
  `σ̂`. An input whose views (across every camera) partly state covariances
  is rejected with `core::Error::MixedCovariance` by `stated_covariance`,
  which every problem type's `validate_input` calls before any work;
  `ProblemIR::validate` rejects an IR mixing the two variants, the backstop
  for direct optim callers.
- **No weighting knob.** The solve uses what the observations carry. A
  `Uniform | DetectorCovariance` switch would add a second way to drop
  covariances the caller already chose to state; dropping them is building
  the views without.
- **The variance factor.** With whitened residuals the family's `σ̂` is the
  a-posteriori `σ̂₀` (≈1 when the stated covariances hold) and the sandwich
  uses it exactly as it uses `σ̂` in pixels: the stated covariances are
  trusted up to one common scale. The export types the unit:
  `ResidualNoise.level: NoiseLevel = Sigma(σ̂) | Sigma0(σ̂₀)`, flattened to
  the JSON key `sigma` or `sigma0`, so a result without stated covariances
  serializes as before and a reader cannot take one for the other. The
  robot-pose priors are weighed at `σ̂₀` the same way, since the whitened
  family's residual standard deviation is `σ̂₀`.
- **What stays in pixels.** Reprojection statistics (mean, per camera,
  histograms, `error_px`) are raw pixel distances; each target feature adds
  `error_whitened = √(eᵀ Σ⁻¹ e)`. A robust loss acts on the whitened
  residual, so its scale is in units of each point's σ.
- **Quality report.** With `Sigma0` the variance-factor check tests `σ̂₀²`
  against 1 without `expected_feature_noise_px`; the view-outlier check
  averages `error_whitened / σ̂₀`.
- **Evidence** (`optim/tests/feature_covariance.rs`, `docs/notes/uncertainty.md`):
  planar and two-camera rig scenes, covariances 0.1–1 px with anisotropy up
  to 4:1 in any orientation, noise drawn from them, 1000 seeds: `σ̂₀`
  0.976 ± 0.029 and 0.986 ± 0.020, predicted σ 0.96–1.04 of the
  Monte-Carlo σ, 95 % of normalized errors within 2σ, and RMSE 0.31–0.41
  of the uniformly weighted solve's on the same data.
- **Next-view planning** is unchanged: its scale `σ² = tr(Σ S_A)/n`, with
  `S_A` rebuilt from the export's points without their covariances, is then
  an isotropic pixel variance averaged over the existing points' stated
  ones, and candidates (which state none) are added at it. Untested on
  whitened results.

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

### Next-view planning

`analysis::predict_with_views(export, candidates, target_points, image_size)`
predicts the covariance a planar calibration would have with candidate
views added, without a re-solve; `analysis::rank_candidates` scores
candidates one at a time; `analysis::candidate_poses` generates a
deterministic set (`CandidateGrid`: image regions × {fronto-parallel, ±tilt
about x, ±tilt about y} at one distance). The core API takes caller-supplied
poses; the generator is a convenience for the app.

- **Prediction.** A candidate is a `camera_se3_target` pose. The target's
  points are projected through the estimate; points behind the camera or
  outside the image are dropped. Its information per unit pixel variance is
  `JᵀJ` over the camera parameters and its own 6-DoF pose, the pose
  marginalized by a Schur complement (`optim::view_information`; exact
  forward-mode `J` through the solver's camera kernels; the left
  perturbation spans the solver's right-perturbation tangent, and a Schur
  complement does not depend on how the eliminated block is parametrized).
  `Σ′ = (Σ⁻¹ + Σ_c S_c / σ²)⁻¹` over the parameters the result estimated. A
  candidate with fewer than four visible points (the planar problem's
  per-view minimum) or a pose its points do not determine adds nothing.
- **Current information: `Σ⁻¹`, not a rebuild from the IR.** `Σ` is
  conditional on exactly the result's fixed set, has every existing pose
  marginalized, and accounts for per-point covariances and fixed poses the
  export does not carry; adding nothing returns `Σ` unchanged. Inverted in
  its unit-diagonal form. The two routes agree: with the existing views'
  information `S_A` rebuilt from the export's per-feature target points and
  poses, `Σ S_A / σ² − I` is at most 4e-7–3e-6 over ten seeds, the precision
  of the solve's own inversion.
- **Scale: `σ² = tr(Σ S_A) / n`.** A planar result has one residual family
  and no priors, so its covariance is `σ_eff² A⁻¹` with
  `σ_eff² = s/κ²` the sandwich's scalar (robust loss included). The rebuild
  above recovers that scalar from the export alone, and the prediction is on
  the covariance's own scale. The reported median-based σ̂ is not: it was
  0.93–1.01× `σ_eff` over ten seeds at 0.2 px, which would mis-weight a
  candidate by up to 7 %. The value is reported as `effective_noise_px`.
- **Ranking.** By the largest projection σ on the `projection_uncertainty`
  grid (centre included) after adding the candidate, the figure the quality
  check bounds; ties by candidate index. `ln det Σ − ln det Σ′` is reported
  alongside as the whole-covariance measure: unit-free, where a trace adds
  px² to squared distortion coefficients.
- **Scope.** Planar intrinsics, one camera, any model the solver estimates
  (any distortion; identity or Scheimpflug sensor). The information kernel
  takes a camera, a pose and points, and `ViewPose::{Estimated, Fixed}`, so
  rigs can follow with the per-camera covariance blocks and a shared rig
  pose; not implemented.
- **Evidence** (`analysis/next_view/tests.rs`, release gate): six tilted
  views of a 9 × 7 board over the central part of a 1280 × 960 image,
  0.1–0.3 px Gaussian noise, five seeds per model; for the best-, median-
  and worst-ranked full-board candidates and the best three together, the
  predicted σ of every parameter and the predicted maximum projection σ
  against a re-solve with the noisy view added, the re-solve rescaled to the
  current `σ_eff`. Mean absolute deviation of σ: Brown-Conrady 0.5 %,
  division 0.1 %, Scheimpflug 2.1 % (worst single case 13.5 %); of the
  maximum projection σ 0.5 %, 0.0 %, 1.2 %. The best candidate's real
  reduction exceeds the worst one's in every scene (Brown-Conrady at 0.2 px:
  1.85 against 0.35 px of 4.9 px).
- **Limit: first order.** With six views bunched around the centre (corners
  extrapolated to 60 px σ at 0.3 px noise) the deviation reached 20–33 %,
  and one re-solve's distortion moved far enough that its projection map
  folded over: the current estimate's Jacobians are no longer the
  re-solve's. The prediction is for refining a calibration that already
  determines the camera roughly.

## Consequences

- Every export carries `uncertainty` (ADR 0012); Python exposes `std_dev`,
  `correlation` and the projection map; the app shows `value ± σ`.
- Staged problems refine the intrinsics in their final stage by default
  (rigs, rig hand-eye), so the covariance carries them. A configuration that
  holds them reports conditional, hence optimistic, extrinsic σ; the fixed
  list says so. Propagating earlier-stage covariance is open.
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
