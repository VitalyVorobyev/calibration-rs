# ADR 0035: Comparing Calibrations Reports Evidence, and a Test Only Where One Is Valid

## Context

A project accumulates runs of one device: configuration changes, exclusions,
camera models, recalibrations. The user asks what changed, whether it
matters, and which run to keep. Two runs differ in what they estimate (the
models), in the data (the same observations, a subset, a new capture) and in
the estimator (the configuration). A difference in units of
`√(σa² + σb²)` is the right scale only for independent captures; for runs of
one dataset it overstates the σ of the difference (the errors are almost
fully correlated), and for runs that differ by estimator or model the
difference is a systematic effect, not noise. One number per pair, or a
ranking score, would therefore be wrong for most of the pairs a run history
holds.

Measured in the projection-difference work (ADR 0034): a calibration's
stated projection σ is not the scale of a rotation-compensated projection
difference (it includes the frame rotation the compensation removes and
matched the uncompensated error, about 7× the compensated one); and when one
model lacks a term the projection difference was 3.4–3.8× that model's
reprojection RMS, because each view's pose absorbs part of the field.

## Decision

- **`analysis::compare_runs` returns evidence, not a score.** Runs of one
  workflow against the first, the reference; cameras matched by id. Per run
  (`RunEvidence`): the `assess` verdict, the covariance status and its
  view-scatter factor, the rig camera, and per camera views, features and
  the in-sample residual statistics with the largest projection σ. Per later
  run (`RunDifference`): parameters, projection difference, rig poses,
  hand-eye, laser planes, model and configuration changes.
- **Every difference carries `Δ / σ_ref`, labelled materiality, never a
  test**: how large the change is at the precision the reference claims. A
  pose carries the Mahalanobis length of its tangent in the reference's
  covariance.
- **A test only where the data relation makes one valid**, the relation
  stated by the caller (`DataRelation`; a project reads it from its records):
  - `Independent` (no observation in common): `σ_Δ² = σa² + σb²`, `z`, its
    two-sided p; for a pose the 6-dof `χ² = τᵀ(Σa + Σb)⁻¹τ`.
  - `Nested { smaller }` (one run's observations a strict subset of the
    other's, one configuration): the Hausman form
    `σ_Δ² = σ_smaller² − σ_larger²` (Hausman 1978: under the null the
    efficient estimate is uncorrelated with its difference from the
    inefficient one), each variance divided by its own run's view-scatter
    factor and multiplied by the larger run's, so a scaling that fired in
    one run does not enter the difference. A variance that does not order
    is `NoTest::NestedVarianceNotPositive`, never clamped.
  - Otherwise `NotAvailable { reason }`: `SharedData` (`SameData`,
    `Overlapping`: the correlation is unknown and near 1), `UnknownRelation`,
    `Identical`, `NoCovariance`, `NotEstimated`, `GaugeDiffers`,
    `SingularCovariance`. `Unknown` never yields a test.
  - `fit_comparable` is true only for `Identical` and `SameData`: in-sample
    residuals of different observations do not compare.
- **The data relation of project runs comes from their records, at
  feature level** (`project::compare_recorded_runs`, `Project::compare_runs`;
  a caller-given relation is refused there). Each run records the content
  hash of every image it used (ADR 0033 D10). A feature is
  `(image content, camera id, target point, observed pixel)` with exact
  bits, so a moved or renamed image is the same data and a re-detection with
  another detector is not. No image content in common: `Independent`. Equal
  `inputs` digests, or equal features under equal `config` and `dataset`
  digests: `Identical`. Equal features otherwise: `SameData`. One run's
  features a strict subset of the other's under equal `config` and `dataset`
  digests (the dataset digest covers the robot poses and the target):
  `Nested`. Otherwise `Overlapping`. A run without the record (imported,
  earlier) is `Unknown`. The relation is classified conservatively: a
  dataset that grew (another manifest) is `Overlapping`, not `Nested`. Each
  difference carries the `DataOverlap` and the `ExclusionChanges` behind it.
- **Parameters only between cameras of one model** (the labels name the same
  quantity); otherwise `DifferentModels`, and the projection difference
  (ADR 0034), fitted on the blocks either run's features cover, is the
  comparison. It carries no σ.
- **Poses are compared relative to one camera**, the reference run's rig
  camera: `cam_i_se3_cam_r = cam_i_se3_rig · cam_r_se3_rig⁻¹`, the hand-eye
  as `gripper_se3_cam_r` / `cam_r_se3_base`. The values do not depend on
  which camera each run held fixed. The test needs both runs to have held
  that camera (the covariance is of the run's own rig frame); otherwise
  `GaugeDiffers`. Propagating the other run's covariance through its rig
  camera's mount is not done.
- **Rotations as a geodesic angle, translations as a length and
  components**, and the tangent `Log(P_ref⁻¹ P_other)` in the covariance's
  right-perturbation coordinates. An angle carries no σ (it is
  Rayleigh-distributed near zero); the test is the tangent's `χ²`.
- **Never compared**: per-view poses, the target's pose and robot-pose
  corrections. They describe a capture.
- **Deferred**: the σ of the projection difference (it needs the
  compensated covariance and its own gate), and a residual-map difference
  (on shared data it is the projection difference seen through re-fitted
  poses; on independent captures it is ADR 0030's open question).

## Measured

`crates/vision-calibration-pipeline/tests/compare_runs_gates.rs`
(release-only, in `scripts/uncertainty_gates.py`), synthetic captures under
the null (one device, the model right), a 12 × 9 board, 0.3 px noise,
Brown-Conrady, 400 trials per gate on both backends. Ranges over the eight
camera parameters; the Monte-Carlo error of a rate is ±1.1 %, of a z
standard deviation ±0.035:

| Gate | Rejection rate at 5 % | z std |
|---|---|---|
| Independent captures, 20 views | 2.0–5.5 % (tiny-solver), 3.5–5.2 % (factrs) | 0.93–1.02 |
| Nested, 30 % of the views left out | 3.0–7.2 % | 0.89–1.02 |
| Nested, 20 % of the features left out | 3.7–6.8 % | 0.93–1.08 |
| Nested, views, Huber loss (0.5 px) | 4.0–6.5 % | 0.92–1.02 |
| Nested, features, Huber loss (0.5 px) | 4.3–6.8 % | 0.95–1.10 |
| Rig `right_se3_left` χ², independent | 4.0 % (mean χ² 5.66 of 6) | — |
| Rig `right_se3_left` χ², nested (30 % views) | 6.2 % (mean χ² 5.81) | — |

A sweep (200 trials per cell, tiny-solver) over 8 / 20 / 60 views,
0.05 / 0.3 px, a Kannala-Brandt fisheye and a Scheimpflug sensor, for the
independent and nested-view tests: every rate within 2.9 MC errors of 5 %,
z std 0.85–1.14. The gate allows each rate 3.5 MC errors and each z std
`3.4/√(2n)` around 1 (a family-wise false alarm of about 1 % over eight
parameters); a first bound of [0.9, 1.1] failed on one draw of seeds
(z std 1.101, 20 % of the features left out under Huber) and was replaced by
the multiplicity-aware bound, not by a wider fixed one.

Two real effects inside the bounds:

- **Feature exclusion inflates the nested z by about 7 %** (z std 1.04–1.10
  over two seed sets, rates up to 7.5 %). Each run estimates its own noise
  level; with a fifth of the features left out the variance difference is a
  small part of each variance, and the two noise estimates' sampling error
  enters it. Putting both covariances on one estimated noise level (the
  larger run's) would remove it, but the export does not state the variance
  factor its covariance used (a robust loss's is a moment ratio), so this is
  left as a follow-up rather than approximated from the MAD noise level.
- **A few views make the tests slightly overconfident**: independent
  captures of 8 views reached z std 1.09–1.14.

Under a Huber loss the Hausman form holds because, with i.i.d. residuals of
one family, the sandwich covariance is a scalar multiple of the same
information matrix in both runs; it is not gated for mixed residual
families (hand-eye with robot-pose priors, laser).

## Consequences

- The app and `vcal` show a table whose columns each say what they are;
  nothing ranks runs. The book says which criterion answers which question
  (held-out error for the model, the nested test for an exclusion, the
  independent test and the projection σ for a recalibration).
- The tests are as good as each run's stated σ; on real images the
  view-scatter factor corrects the share it measures. Errors shared by both
  captures (the target's flatness) are in neither σ.

## Alternatives rejected

- **One score or a ranking**: no single criterion is right for model choice,
  exclusions and recalibrations at once.
- **`√(σa² + σb²)` for every pair**: wrong for every pair that shares data.
- **Parameter deltas across models**: the labels name different quantities.
- **A residual-map difference in v1**: see Deferred.
- **Comparing `cam_se3_rig` as stored**: depends on the rig camera each run
  chose.
