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
  stated on the run (`ComparedRun::with_relation`; a project reads it from
  its records, `None` is `Unknown`):
  - `Independent` (no observation in common): `σ_Δ² = σa² + σb²`, `z`, its
    two-sided p; for a pose the 6-dof `χ² = τᵀ(Σa + Σb)⁻¹τ`; and one joint
    `χ²` over every camera parameter both estimated (`ParameterComparison::
    SameModel::joint`), because of eight per-parameter tests at 5 % one
    rejects by chance in about a quarter of null comparisons.
  - `Nested { smaller }` (one run's observations a strict subset of the
    other's, one configuration): the Hausman form on **one noise level**.
    Each export states its variance factor `c` (`ParameterCovariance::
    variance_factor`, `Σ = k · c · (JᵀJ)⁻¹`, `k` the view-scatter factor),
    so each run's information `A⁻¹ = Σ / (k c)` is recovered and
    `σ_Δ² = c_s · k_s · (A_s⁻¹ − A_l⁻¹)` (Hausman 1978: under the null the
    efficient estimate is uncorrelated with its difference from the
    inefficient one). The smaller run's `c` and `k` are consistent under
    both hypotheses and keep the power when the larger run holds the
    outliers: the larger run's view-scatter factor measures the scatter
    those outliers cause (an independent check set `k_l` in 69 of 1000
    outlier trials, `k_s` in 20, and `k_s` rejected 53–56 % against 52–56 %
    for `k_l`); under the null `k` is rarely set, and with `k_s` the null
    cells moved closer to nominal (6 of 20 views out: z std 0.96–1.00 with
    `k_s`, 0.89–0.99 with `k_l`). A variance that does not grow is
    `NestedVarianceNotPositive`, never clamped; for truly nested data it is
    an artifact of evaluating each information at its own estimate (below). A pose's nested `χ²` goes through a generalized inverse over
    the directions whose variance grows. A camera's parameters get no nested
    joint test (`NestedJoint`, measured below). Not offered for a covariance
    of several residual families, robot-pose priors or per-camera noise
    levels (`MixedResiduals`, the premise `G ∝ H` fails and no gate exists),
    nor for an export without the factor (`NoVarianceFactor`).
  - The nested test is **not valid for observations left out because of
    their residuals** (the worst features or views of a fit): the remaining
    fit then agrees with the full one by construction less than chance
    predicts. The classifier cannot know how an exclusion was chosen; the
    rustdoc and the book say so, and a residual-based screening with its own
    test is backlog COMPARE-SCREENING.
  - Otherwise `NotAvailable { reason }`: `SharedData` (`SameData`,
    `Overlapping`: the correlation is unknown and near 1), `UnknownRelation`,
    `Identical`, `NoCovariance`, `NotEstimated`, `GaugeDiffers`,
    `SingularCovariance`. `Unknown` never yields a test.
- **A stated relation the exports contradict is refused**, the difference
  compared as `Overlapping` with `RunDifference::relation_refused`: a stated
  independent pair sharing bit-identical features (the same target point at
  the same observed pixel), a stated nested pair whose smaller run is not a
  strict subset of the larger's features, or whose given configurations
  differ. The in-sample residuals compare like with like only for
  `Identical` and `SameData`; a separate flag would restate the relation, so
  none is kept.
- **The data relation of project runs comes from their records, at
  feature level** (`project::compare_recorded_runs`, `Project::compare_runs`). Each run records the content
  hash of every image it used (ADR 0033 D10). A feature is
  `(image content, camera id, target point, observed pixel)` with exact
  bits, so a moved or renamed image is the same data and a re-detection with
  another detector is not. No image content in common: `Independent`. Equal
  `inputs` digests, or equal features under equal `config`, `dataset` and
  `seed` digests (one estimator): `Identical`. Equal features otherwise:
  `SameData`. One run's features a strict subset of the other's under one
  estimator (the dataset digest covers the robot poses and the target):
  `Nested`. Otherwise `Overlapping`. **The seed is part of the estimator**
  (ADR 0033 D8): it is an initial value, but it changes the solver route in
  two places (a Scheimpflug tilt seed bounds the planar refine around it, a
  laserline device holds its tilt at the seed), so two runs from different
  seeds are not one estimator and get no nested test. A seeded and an
  unseeded run of the same data are `SameData`, and the seed is listed
  among the configuration changes (`seed.source`, `seed.values` with the
  seed digest) [measured: `a_seed_is_part_of_the_estimator`, where the two
  reach the same optimum]. A run without the record (imported,
  earlier) is `Unknown`. The relation is classified conservatively: a
  dataset that grew (another manifest) is `Overlapping`, not `Nested`. Each
  difference carries the `DataOverlap` and the `ExclusionChanges` behind it.
- **Parameters only between cameras of one model** (the labels name the same
  quantity); otherwise `DifferentModels` (a general sensor homography,
  which the solver does not parameterize, is `Unparameterized`), and the projection difference
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
the null (one device, the model right, nothing left out by its residuals), a
12 × 9 board, 0.3 px noise, Brown-Conrady, on one backend (the two reach the
same optimum, so the same seeds on the other repeat the draw). Ranges over
the eight camera parameters. Gate per parameter: rejection rate within 3.5
Monte-Carlo errors of 5 %, z std within `3.4/√(2n)` of 1, at most 1 % of
the tests missing; per cell, the pooled mean `z²` within 3 bootstrap errors
of 1; the joint `χ²` likewise.

| Cell (trials) | Rejection at 5 % | z std | Joint χ² (8 dof) / pose χ² (6 dof) |
|---|---|---|---|
| Independent, 20 views (400) | 3.5–5.25 % | 0.96–1.02 | 5.75 %, mean χ²/dof 1.01 |
| Nested, 6 of 20 views out (1000) | 3.8–5.2 % | 0.96–1.00 | not offered (`NestedJoint`) |
| Nested, 20 % of features out (1000) | 3.9–6.0 % | 0.97–1.04 | not offered |
| Nested, 1 of 8 views out (1000) | 4.7–5.5 % | 0.99–1.02 | not offered |
| Nested, Huber 0.5 px, 6 of 20 views (1000) | 4.5–5.6 % | 0.98–1.00 | not offered |
| Nested, Huber 0.5 px, 20 % of features (1000) | 4.6–6.2 % | 0.98–1.04 | not offered |
| Rig `right_se3_left`, independent (400) | — | — | 4.0 %, mean χ²/dof 0.94 |
| Rig `right_se3_left`, 2 of 8 views out (1000) | — | — | 5.0 %, mean χ²/dof 0.97 |

Missing tests under the null: at most 0.8 % per parameter (`k2`
`NestedVarianceNotPositive`, 1 of 8 views out). A sweep at 200 trials per
cell over 8 / 20 / 60 views, 0.05 / 0.3 px, a Kannala-Brandt fisheye and a
Scheimpflug sensor, independent and 30 % of the views out, passes the same
gate (rates 2.0–9.5 %, z std 0.85–1.13, inside each cell's bounds).

The alternative: 1 % of the larger run's features moved 8 px, left out by
the smaller run. Some parameter rejects in 98.75 % of 400 comparisons
(pooled mean `z²` ≈ 900); 82 % of the per-parameter tests are available,
the rest `NestedVarianceNotPositive` (the left-out share is small). Under a
robust loss that absorbs the left-out outliers the test does not reject,
by design: the larger fit already ignores them (measured in the
independent check: 4–6 %).

**Before the common noise level** (the review's measurement, 400 trials,
1024 × 768, 10 × 7 board): each run's own noise estimate entered the
difference, `V̂ = k(ŝ_s²A_s⁻¹ − ŝ_l²A_l⁻¹)`; its second term dominates for
small exclusions: 1 of 25 views out gave z std up to 1.88 and up to
18/400 `NestedVarianceNotPositive` per parameter, 6 features out z std
1.09–1.69, a rig pose χ² with 2 of 8 views out a mean of 11.76 against 6.03
on one noise level, and the test was unavailable in 99.5–100 % of the
comparisons with 1 % outliers. The gates at 20–30 % exclusions had passed
because the term is small there. The earlier claims of this ADR that "the
feature-exclusion inflation comes from two noise estimates" at 20 % and
that "8 views reach z std 1.09–1.14" were not reproduced (the review's 8
views: 0.93–1.04; here, 0.94–1.09 at 200 trials) and are withdrawn.

**A limitation, pinned** (`nested_small_exclusions_have_heavier_tails`):
with a small left-out share the nested `z` of some parameters still has
heavy tails on one noise level: 1 of 25 views out, `k2` z std 1.15 (MAD
1.00, three `|z| > 4` in 1000); 6 of about 2000 features out, z std up to
1.34 and `k2` 7.7 % rejections with 4.3 % not positive. **The cause is
measured** (independent check): each run's information is evaluated at its
own estimate, and two nearly equal informations then differ mostly by that
mismatch. Evaluating both at one common point removes the excess (1 of 25
views, `k2` z std 1.090 → 1.012, ten `|z| > 4` → none, 7 missing → 0; 6
features, `k2` 1.245 → 0.992, 74 missing → 0), and explains
`NestedVarianceNotPositive` for truly nested data: at one point
`A_s⁻¹ ⪰ A_l⁻¹` by construction. A tighter solver stop (`1e-14`
decrease) and polishing the solves changed nothing. The fix, deferred
(backlog COMPARE-SMALL-EXCLUSIONS):
`V_Δ = c_s · k_s · [(M_l − M_d)⁻¹ − M_l⁻¹]`, `M_l = (Σ_l / (k_l c_l))⁻¹`
from the larger run's export and `M_d` the information of the left-out
observations at `θ̂_l`, each view's pose profiled out
(`vision_calibration_optim::view_information`). It needs the observations,
which `Project::compare_runs` can reach and an export alone cannot, the
same whitening for runs with stated per-feature covariances, and a rig
analogue. A cut on
the relative variance growth was measured and rejected: at 1–2 % it removes
most small-exclusion tests and the 1 %-outlier tests the nested test exists
for, and biases what it keeps (joint mean χ²/dof 0.92). The camera-parameter
joint test through a generalized inverse rejects 7.7 % (1 of 8 views),
9.1 % (1 of 25) and 15.7 % (6 features) and is not offered for nested runs;
the common point does not rescue it for whole-view drops (independent
check: 1 of 25 views 8.7 %, mean χ²/dof 1.46; 1 of 8 views 16.9 %), so
`NestedJoint` stays after that fix.

## Consequences

- The app and `vcal` show a table whose columns each say what they are;
  nothing ranks runs. The book says which criterion answers which question
  (held-out error for the model; the nested test for whether data left out
  without looking at the residuals fit as the rest do; the independent test
  and the projection σ for a recalibration).
- The export gains `ParameterCovariance::variance_factor`, an optional
  field: present for one residual family without priors, its numbers
  unchanged (the regression suite on both backends).
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
- **Each run's own noise level in the nested form**: measured above; the
  difference of the two noise estimates dominates small exclusions.
- **The larger run's variance factor**: inconsistent under the alternative
  when the larger run holds the outliers the smaller left out.
- **The larger run's view-scatter factor**: the same argument, and it is set
  more often by the outliers it should expose (measured above).
- **A cut on the relative variance growth**: measured above.
- **The relation in `CompareRunsOptions`** (one entry per later run): a
  length to check and, in a project, a state to forbid; it is the run's.
