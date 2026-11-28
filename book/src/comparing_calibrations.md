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
- **the estimator**: the configuration (loss, solver, what is held fixed)
  and, for a project's runs, the seed they started from.

## What a comparison holds

For each run, its own evidence (`RunEvidence`): the quality verdict, whether
it has a covariance (with the view-scatter factor it was scaled by and its
variance factor), the camera that is its rig frame, and per camera the
views, features and the in-sample residual statistics (`reprojection`:
mean, RMS, median, 95th percentile), with the largest projection σ over the
image. The residual statistics compare like with like only between runs of
the same observations: a run that left out its worst features always has a
lower residual, which says nothing of its calibration.

For each run after the first, against the reference (`RunDifference`),
cameras matched by id:

| Field | What it is | What it is not |
|---|---|---|
| `relation` | how the run's observations relate to the reference's, as stated on the run (or, for a project's runs, read from their records) | a guess: a relation the exports contradict is refused (`relation_refused`) and the pair compared as overlapping |
| parameter `delta` | `other − reference`, in the parameter's unit | — |
| `delta_over_reference_sigma` | the change in units of the reference's stated σ: whether it matters at the precision the reference claims (materiality) | a test. Two fits of the same images can differ by many σ without disagreeing, and by less than one while disagreeing |
| parameter `test` | a z-test, only where the data relation makes one valid (below), with its p-value | present for every pair: without a valid test it is `not_available`, with the reason |
| `joint` | one χ² of all the camera's parameters both runs estimated, for independent captures | eight per-parameter tests read together: at 5 % each, one of eight rejects by chance in about a quarter of comparisons |
| `projection` | the [projection difference](diagnostic_maps.md#comparing-two-calibrations) over the image, fitted on the blocks either run's features cover, for any two models | a σ-scaled quantity: a run's projection σ is not its scale |
| `rig` | each camera's pose relative to the reference run's rig camera: rotation angle, translation length and components, the tangent, a 6-dof χ² where valid | dependent on which camera each run held fixed: the poses are relative, so they compare whatever the rig frame |
| `handeye` | the hand-eye expressed for that camera | — |
| `laser_planes` | the angle between the normals and the change of distance, with a test where valid | — |
| `model_changes`, `config_changes` | the cameras whose models differ; the settings whose values differ (when both runs' configurations are given) | — |

Parameters are compared only between cameras of one model; across models
their labels name different quantities, and the projection difference is the
comparison. Per-view poses, the target's pose and robot-pose corrections
describe a capture, not the device, and are never compared.

## When a difference has a test

| Relation | What `Δ` is | Variance of `Δ` |
|---|---|---|
| `independent`: two captures with no observation in common | device change plus each capture's error | `σ_ref² + σ_other²` |
| `nested`: one run's observations a strict subset of the other's, one configuration, left out without looking at the residuals | the pull of the left-out observations | `c_smaller · k_smaller · (a_smaller − a_larger)` (below) |
| `same_data`, `overlapping`, `identical`, `unknown` | an estimator or model effect on shared noise, or not known | none |

The nested form is the specification test of
[Hausman (1978)](references.md#hausman-1978): when the left-out observations
fit the model as the rest do, the estimate from all of them is efficient,
uncorrelated with its difference from the estimate without them, so the
variance of the difference is the difference of the variances. Both
variances must be on **one noise level**: each run estimates its own, and
the difference of the two estimates would otherwise dominate a small
exclusion. Each export states its variance factor `c` (`σ̂²` under least
squares) beside its view-scatter factor `k`, so each run's variance per unit
noise `a = σ² / (k c)` is known, and the variance of `Δ` is taken on the
smaller run's noise level, its `c` and `k`: they are read from the
observations both runs share, which fit the model whether or not the
left-out ones do, so outliers in the larger run cannot inflate the scale
that should expose them. A large `z` says the left-out observations
pulled the estimate further than leaving out that much data does by chance;
a small one says leaving them out cost precision and changed nothing else.

The nested test answers one question: *do the two fits agree beyond what
their shared data explain, for observations left out without looking at the
residuals* (a camera's images dropped because of motion blur, a view
excluded because the target was partly hidden, a random half for a
consistency check). It is **not valid for observations chosen by their
residuals** (the worst features or views of a fit): the remaining fit then
agrees with the full one less than chance predicts, and a correct model is
rejected far more often than the level says. There is no test for such an
exclusion, and the comparison cannot tell how an exclusion was chosen.
Under a robust loss that absorbs the left-out outliers, the larger run's
fit already ignores them, so the nested test does not reject: by design, it
asks whether the two fits differ, not whether the data had outliers.

It is also not offered when a run's covariance combines several residual
families or priors (laser and reprojection residuals, robot-pose priors,
per-camera noise levels: `mixed_residuals`), when an export states no
variance factor (`no_variance_factor`), and when the smaller run's variance
does not grow (`nested_variance_not_positive`). A camera's parameters have
no nested joint test (`nested_joint`); each parameter's own test and a
pose's χ² remain.

Runs of the same observations under another configuration have no test:
their errors are almost fully correlated by an amount neither export states.
A small difference there is not agreement and a large one is not a
significance; read it as materiality only.

A rig pose is tested only when both runs held the same camera as the rig
frame (`gauge_differs` otherwise): the other run's covariance is of its own
rig frame's poses.

**Measured.** On synthetic captures under the null (both runs of one
device, the model right), at the 5 % level each parameter's rejection rate
was 3.5–6.2 % (Monte-Carlo error ±0.7 % at 1000 trials) and its z standard
deviation 0.96–1.04 for independent captures, 6 of 20 views, 20 % of the
features and 1 of 8 views left out, under plain least squares and a Huber
loss; the joint χ² of independent captures rejected 5.75 %. The 6-dof χ² of
a two-camera rig's relative pose rejected 4.0 % (independent) and 5.0 % (2
of 8 views left out). A sweep over 8, 20 and 60 views, two noise levels, a
Kannala-Brandt fisheye and a Scheimpflug sensor stayed within the same
bounds. A run whose larger half holds 1 % gross outliers is told apart from
the run without them in 98.75 % of comparisons; under a robust loss that
absorbs those outliers the test rejects 4–6 %, as designed.

When the left-out data are a **small share** of the information (1 of 25
views, a handful of features), the nested z of some parameters has heavier
tails than a standard normal (standard deviation up to 1.34 in a synthetic
test, a rejection rate up to 7.7 % at 5 %): each run's information is
evaluated at its own estimate, and two nearly equal informations then differ
mostly by that mismatch. Evaluating both at one point removes the excess,
but needs the left-out observations, which an export does not hold. The
same mismatch is why a small exclusion's variance sometimes does not grow
(`nested_variance_not_positive`). Read a small exclusion's p-value with that
margin.

The tests are as good as the stated σ: on real images a covariance can be
short by the error shared by a view's corners, which the view-scatter
factor corrects for when it measures it (see
[Limits and Unavailable Results](uncertainty_limits.md)). Errors shared by
both captures, such as the target's own flatness, are in neither σ.

## Runs of a project

A project's runs carry what the relation needs: each run's record lists the
images it used with a hash of their content, and the digests of its
configuration, dataset and seed. `Project::compare_runs` (`vcal project compare`,
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
- **Do the runs agree after leaving data out?** For data left out without
  looking at the residuals, the nested test says whether the fit without
  them disagrees with the fit with them beyond what their shared data
  explain. Whether an exclusion was *right* is what the detection shows;
  outliers chosen by their residuals need a screening test of their own, not
  this one.
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
        // Another capture of the device: no image in common.
        ComparedRun::new("friday", friday.as_ref(), cam(), vec![[1280, 960]])
            .with_relation(DataRelation::Independent),
    ];
    let comparison = compare_runs(&runs, &CompareRunsOptions::new())?;
    for camera in &comparison.differences[0].cameras {
        if let ParameterComparison::SameModel { parameters, joint } = &camera.parameters {
            println!("all parameters: {joint:?}");
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
        {
            "name": "friday",
            "export": friday,
            "camera_ids": ["cam0"],
            "image_sizes": [(1280, 960)],
            "relation": {"kind": "independent"},
        },
    ]
)
for p in comparison["differences"][0]["cameras"][0]["parameters"]["parameters"]:
    print(p["label"], p["delta"], p["test"])
```
