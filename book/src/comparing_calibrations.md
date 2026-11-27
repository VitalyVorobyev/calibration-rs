# Comparing Calibrations

A camera or rig collects calibrations: a configuration changed, a few
images left out, another camera model tried, the device recalibrated a month
later. `compare_runs` puts them side by side against the first, the
reference, and reports what changed and what each change means. It returns
evidence, not a score: no number ranks the runs.

Two runs can differ in three ways, and which comparisons are valid depends
on all three:

- **what they estimate**: the camera models;
- **the data**: the same observations, some of them, or a new capture;
- **the estimator**: the configuration (loss, solver, what is held fixed).

## What a comparison holds

For each run, its own evidence (`RunEvidence`): the quality verdict, whether
it has a covariance (and the view-scatter factor it was scaled by), the
camera that is its rig frame, and per camera the views, features and the
in-sample residual statistics (RMS, median, 95th percentile), with the
largest projection σ over the image.

For each run after the first, against the reference (`RunDifference`),
cameras matched by id:

| Field | What it is | What it is not |
|---|---|---|
| `relation` | how the run's observations relate to the reference's, as stated | a guess: it is given (or, for a project's runs, read from their records) |
| `fit_comparable` | whether the in-sample residuals compare like with like: only for the same observations | — a run that left out its worst features always has a lower residual, which says nothing of its calibration |
| parameter `delta` | `other − reference`, in the parameter's unit | — |
| `delta_over_reference_sigma` | the change in units of the reference's stated σ: whether it matters at the precision the reference claims (materiality) | a test. Two fits of the same images can differ by many σ without disagreeing, and by less than one while disagreeing |
| parameter `test` | a z-test, only where the data relation makes one valid (below), with its p-value | present for every pair: without a valid test it is `not_available`, with the reason |
| `projection` | the [projection difference](diagnostic_maps.md#comparing-two-calibrations) over the image, fitted on the blocks either run's features cover, for any two models | a σ-scaled quantity: a run's projection σ is not its scale |
| `rig` | each camera's pose relative to the reference run's rig camera: rotation angle, translation length and components, the tangent, a 6-dof χ² where valid | dependent on which camera each run held fixed: the poses are relative, so they compare whatever the rig frame |
| `handeye` | the hand-eye expressed for that camera | — |
| `laser_planes` | the angle between the normals and the change of distance, with a test where valid | — |
| `model_changes`, `config_changes` | the cameras whose models differ; the settings whose values differ (when each run's configuration is given) | — |

Parameters are compared only between cameras of one model; across models
their labels name different quantities, and the projection difference is the
comparison. Per-view poses, the target's pose and robot-pose corrections
describe a capture, not the device, and are never compared.

## When a difference has a test

| Relation | What `Δ` is | `σ` of `Δ` |
|---|---|---|
| `independent`: two captures with no observation in common | device change plus each capture's error | `√(σ_ref² + σ_other²)` |
| `nested`: one run's observations a strict subset of the other's, one configuration | the pull of the left-out observations | `√(σ_smaller² − σ_larger²)` |
| `same_data`, `overlapping`, `identical`, `unknown` | an estimator or model effect on shared noise, or not known | none |

The nested form is the specification test of
[Hausman (1978)](references.md#hausman-1978): when the left-out observations
fit the model as the rest do, the estimate from all of them is efficient,
uncorrelated with its difference from the estimate without them, so the
variance of the difference is the difference of the variances. A large `z`
says the left-out observations pulled the estimate further than leaving out
that much data does by chance: they did not fit as the rest do. A small `z`
says leaving them out cost precision and changed nothing else. Each variance
is taken at the larger run's view-scatter factor, so a scaling that fired in
one run only does not enter the difference. When the smaller run's stated
variance is not the larger (`nested_variance_not_positive`), there is no
test.

Runs of the same observations under another configuration have no test:
their errors are almost fully correlated by an amount neither export states.
A small difference there is not agreement and a large one is not a
significance; read it as materiality only.

A rig pose is tested only when both runs held the same camera as the rig
frame (`gauge_differs` otherwise): the other run's covariance is of its own
rig frame's poses.

**Measured.** On synthetic captures under the null (both runs of one
device, the model right), 400 trials each on both solver backends, the
rejection rate at the 5 % level was 2.0–7.2 % per parameter (Monte-Carlo
error ±1.1 %) and the z standard deviation 0.89–1.10: independent captures,
30 % of the views left out, 20 % of the features left out, under plain least
squares and a Huber loss. The 6-dof χ² of a two-camera rig's relative pose
rejected 4.0 % (independent) and 6.2 % (nested). A sweep over 8, 20 and 60
views, two noise levels, a Kannala-Brandt fisheye and a Scheimpflug sensor
stayed within the same bounds. Two effects are real: leaving out a fifth of
the features makes the nested z about 7 % too wide (each run estimates its
own noise level, and their difference enters the small variance
difference), and with 8 views the independent z reached a standard
deviation of 1.14: read a z near the threshold from few views or small
exclusions with that margin.

The tests are as good as the stated σ: on real images a covariance can be
short by the error shared by a view's corners, which the view-scatter
factor corrects for when it measures it (see
[Limits and Unavailable Results](uncertainty_limits.md)). Errors shared by
both captures, such as the target's own flatness, are in neither σ.

## Runs of a project

A project's runs carry what the relation needs: each run's record lists the
images it used with a hash of their content, and the digests of its
configuration and dataset. `Project::compare_runs` (`vcal project compare`,
Python `Project.compare_runs`) reads the relations from them instead of
taking them from the caller; see [Projects](projects.md#comparing-runs) for
the rules. Each difference then also says what the two runs share
(`data`: images and features in common and in one run only) and which
exclusions only one of them made (`exclusion_changes`).

## Which run to keep

No function picks one, because each question has its own criterion:

- **Which camera model?** Held-out error,
  [`compare_camera_models`](distortion_selection.md). In-sample residuals
  always favour the larger model.
- **Was that exclusion right?** The exclusion is justified by what the
  detection shows. The nested test says whether the left-out data were
  inconsistent with the rest (large `z`) or merely noisy (small `z`).
- **Which recalibration to deploy?** Among runs that pass and have a
  covariance, the most precise (smallest projection σ), checked against the
  others with the independent test. A significant disagreement means the
  device changed or a σ is understated, and the comparison says which
  parameters disagree.

## In code

```rust,no_run
use vision_calibration::analysis::{
    CompareRunsOptions, ComparedRun, DataRelation, DifferenceTest, ParameterComparison,
    compare_runs,
};
use vision_calibration::common::calibrated_cameras;

fn main() -> anyhow::Result<()> {
    let monday = calibrated_cameras(&std::fs::read_to_string("monday/calibration.json")?)?;
    let friday = calibrated_cameras(&std::fs::read_to_string("friday/calibration.json")?)?;
    let cam = || vec!["cam0".to_string()];
    let runs = [
        ComparedRun::new("monday", monday.as_ref(), cam(), vec![[1280, 960]]),
        ComparedRun::new("friday", friday.as_ref(), cam(), vec![[1280, 960]]),
    ];
    // Two captures of the device: no image in common.
    let options = CompareRunsOptions::new().with_relations(vec![DataRelation::Independent]);
    let comparison = compare_runs(&runs, &options)?;
    for camera in &comparison.differences[0].cameras {
        if let ParameterComparison::SameModel { parameters } = &camera.parameters {
            for p in parameters {
                if let DifferenceTest::Independent { z, p_value, .. } = p.test {
                    println!("{} {:+.4} (z {z:+.2}, p {p_value:.3})", p.label, p.delta);
                }
            }
        }
    }
    Ok(())
}
```

In Python:

```python
comparison = vc.compare_runs(
    [
        {"name": "monday", "export": monday, "camera_ids": ["cam0"], "image_sizes": [(1280, 960)]},
        {"name": "friday", "export": friday, "camera_ids": ["cam0"], "image_sizes": [(1280, 960)]},
    ],
    relations=[{"kind": "independent"}],
)
for p in comparison["differences"][0]["cameras"][0]["parameters"]["parameters"]:
    print(p["label"], p["delta"], p["test"])
```
