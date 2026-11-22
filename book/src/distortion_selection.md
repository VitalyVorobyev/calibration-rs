# Choosing a Distortion Model

Every workflow that estimates intrinsics picks a lens distortion model and a per-parameter fix list. Get it wrong and you either overfit sparse data (a free `k3` on 20 views chases noise) or underfit a genuinely distorted wide-angle lens. This chapter lists the available models, the default fix lists and why they are the defaults, and when reaching for a richer model actually buys anything.

## The Models

`DistortionKind` (`vision_calibration::core::DistortionKind`) fills the distortion slot of the camera model (`camera.model.distortion` in every `*Config`) and selects the packed coefficient layout the factors compile against:

| Kind | dim | Packed layout | Model |
|---|---:|---|---|
| `None` | 0 | (no distortion block) | no distortion |
| `BrownConrady5` | 5 | `[k1, k2, k3, p1, p2]` | Brown-Conrady radial and tangential (default) |
| `Rational8` | 8 | `[k1, k2, k3, k4, k5, k6, p1, p2]` | rational polynomial |
| `ThinPrism9` | 9 | `[k1, k2, k3, p1, p2, s1, s2, s3, s4]` | Brown-Conrady plus thin prism |
| `Division1` | 1 | `[lambda]` | division model ([Fitzgibbon, 2001](references.md#fitzgibbon-2001)) |

Every workflow calibrates every kind with a straight or a tilted (Scheimpflug) sensor: planar intrinsics, single-camera hand-eye and the laserline device through `camera`, the rigs per camera (`camera` plus `camera_overrides`). The rig laser workflows take each camera of the upstream rig in its own model.

Read the coefficients from `export.camera.distortion`, a `DistortionParams` with one variant per model (see [Serialization and Runtime-Dynamic Types](params.md)).

## The Fix Lists

`camera.fix` (a `CameraFix`) names, per slot, the parameters the non-linear solve holds fixed:

- `intrinsics`: any of `fx, fy, cx, cy`;
- `distortion`: the model's own coefficient names;
- `sensor`: `tilt_x, tilt_y`.

A slot left `None` keeps the camera model's default; an explicit list replaces it; an unknown name is a configuration error raised before any work. In JSON: `"camera": {"model": {...}, "fix": {"intrinsics": ["cx", "cy"], "distortion": ["k3", "p1", "p2"]}}`.

The defaults:

- `BrownConrady5`: `k3` fixed. **`k3` often overfits sparse or noisy corner data; relax it only for wide-angle lenses or high-quality data.** The same rule applies to the linear stage's `IntrinsicsInitConfig::fix_k3`, which defaults to `true` for the same reason.
- The extended models: nothing fixed, since they are chosen for their extra terms.
- Any model under a Scheimpflug sensor: everything but the low-order radial terms (`k1`, `k2`; the division model's `lambda`) fixed. See below.

### Scheimpflug-Specific Defaults

A Scheimpflug sensor (`camera.model.sensor = SensorKind::Scheimpflug`) changes two things, both because the sensor tilt couples with distortion:

- The default distortion fix list keeps only `k1`, `k2` (or `lambda`) free: the tilt can absorb tangential-distortion-like signal, so leaving `p1` and `p2` free overfits.
- The linear stage holds `p1` and `p2` at zero while the tilt is estimated, even when the fix list frees them: a free tangential term is ill-posed next to the tilt in the linear fit. The non-linear solve, which separates them, then moves them.

In every workflow the linear stage fits only the distortion terms the camera's solves estimate: a term the fix list holds stays at zero from the start, rather than at a linear fit the later solves would keep.

The tilt itself is the `sensor` slot of the same fix list (`camera.fix.sensor = Some(vec!["tilt_y".into()])` holds `tilt_y`), since the tilt is a sensor parameter, not a distortion one.

## Configuring a Model

### Do nothing

`PlanarIntrinsicsConfig::default()` selects `BrownConrady5` with `k3` fixed and an untilted sensor. Most datasets should start here.

### Choose an extended model

```rust
use vision_calibration::core::DistortionKind;
use vision_calibration::planar_intrinsics::PlanarIntrinsicsConfig;

let mut config = PlanarIntrinsicsConfig::default();
config.camera.model.distortion = DistortionKind::Rational8;
// The extended models fix nothing by default: k1..k6, p1, p2 are all refined.
// Name what to hold, e.g. the higher-order radial block:
config.camera.fix.distortion = Some(vec!["k4".into(), "k5".into(), "k6".into()]);
```

### Fix intrinsics or distortion terms

```rust
use vision_calibration::planar_intrinsics::PlanarIntrinsicsProblem;
use vision_calibration::session::CalibrationSession;
# fn main() -> anyhow::Result<()> {
# let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();

// Fix cx, cy but optimize fx, fy
session.update_config(|c| {
    c.camera.fix.intrinsics = Some(vec!["cx".into(), "cy".into()]);
})?;

// Fix k3 and the tangential terms
session.update_config(|c| {
    c.camera.fix.distortion = Some(vec!["k3".into(), "p1".into(), "p2".into()]);
})?;
# Ok(())
# }
```

### A tilted sensor, seeded

```rust,no_run
use vision_calibration::common::IntrinsicsManualInit;
use vision_calibration::core::{
    DistortionKind, FxFyCxCy, PlanarDataset, ScheimpflugParams, SensorKind,
};
use vision_calibration::planar_intrinsics::{
    PlanarIntrinsicsConfig, PlanarIntrinsicsProblem, step_init_with_seed, step_optimize,
};
use vision_calibration::session::CalibrationSession;
# fn main() -> anyhow::Result<()> {
# let dataset: PlanarDataset = unimplemented!("the observed views");
# let nominal_k: FxFyCxCy<f64> = unimplemented!("from the datasheet");
# let mount_tilt: ScheimpflugParams = unimplemented!("from the mount");

let mut config = PlanarIntrinsicsConfig::default();
config.camera.model.sensor = SensorKind::Scheimpflug;    // radial-only default fix list
config.camera.model.distortion = DistortionKind::ThinPrism9;

let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
session.set_input(dataset)?;
session.set_config(config)?;

let mut seed = IntrinsicsManualInit::default();
seed.intrinsics = Some(nominal_k);   // datasheet focal length / pixel pitch
seed.sensor = Some(mount_tilt);      // mechanical mount angle
step_init_with_seed(&mut session, seed, None)?;
step_optimize(&mut session, None)?;
# Ok(())
# }
```

The linear stage always produces Brown-Conrady coefficients; they are embedded into the chosen model with any extra degrees of freedom zeroed before the non-linear refinement. A `distortion` seed is a value in the chosen model itself (its kind must match `camera.model.distortion`) and replaces the fitted coefficients as given. See [Seeding with Prior Knowledge](manual_init.md) for the seeding mechanism and [Planar Intrinsics Calibration](planar_intrinsics.md#tilted-scheimpflug-sensor) for why a tilted camera starts from a seed.

## Does a Richer Model Help?

The reprojection error of a calibration is measured on the views it was fitted to, and a richer model always lowers it: a model with more free coefficients contains the simpler one, so it fits the same points at least as well, signal or noise. That number cannot say whether the extra terms describe the lens. The views the fit did *not* see can.

`analysis::compare_camera_models` splits the views into folds (five by default, assigned by a seeded shuffle), fits each candidate camera to all folds but one, and scores the held-out fold: each held-out view's board pose is solved with the fold's camera held fixed, and every point's pixel error counts. Every view is held out exactly once, so each candidate's pooled held-out RMS covers the same points. The candidates are `CameraConfig`s (model and fix list); the rest of the config, the initialization, the solver and the robust loss, is the same for all of them.

Below, the lens has only `k1` and `k2`. Three candidates: `k1` alone (under-modelled), the default Brown-Conrady with `k3` fixed, and `Rational8` with all eight coefficients free.

```rust
use vision_calibration::analysis::{CrossValidationOptions, ModelOutcome, compare_camera_models};
use vision_calibration::common::config::CameraConfig;
use vision_calibration::control::RunControl;
use vision_calibration::core::{CameraModelSpec, DistortionKind, SensorKind};
use vision_calibration::planar_intrinsics::PlanarIntrinsicsConfig;
# use vision_calibration::core::{BrownConrady5, FxFyCxCy, PlanarDataset, View, make_pinhole_camera};
# use vision_calibration::synthetic::{noise::UniformPixelNoise, planar, poses};
# fn main() -> anyhow::Result<()> {
# let lens = make_pinhole_camera(
#     FxFyCxCy { fx: 800.0, fy: 790.0, cx: 640.0, cy: 480.0 },
#     BrownConrady5 { k1: -0.25, k2: 0.08, ..BrownConrady5::default() },
# );
# let board = planar::grid_points(9, 7, 0.03);
# let specs = poses::seeded_board_pose_specs(12, 0.45, 11);
# let mut board_poses = poses::centered_board_poses(&specs, (0.12, 0.09), 0.5);
# for (i, pose) in board_poses.iter_mut().enumerate() {
#     pose.translation.vector.x += [-0.12, 0.0, 0.12][i % 3];
#     pose.translation.vector.y += [-0.1, 0.0, 0.1][(i / 3) % 3];
# }
# // Uniform noise of ±0.52 px: 0.3 px standard deviation per axis.
# let noise = UniformPixelNoise { seed: 5, max_abs_px: 0.52 };
# let views = planar::project_views_noisy(&lens, &board, &board_poses, &noise)?;
# let dataset = PlanarDataset::new(views.into_iter().map(View::without_meta).collect())?;

let brown_conrady = CameraModelSpec::pinhole(DistortionKind::BrownConrady5, SensorKind::Identity);
let mut k1_only = CameraConfig::new(brown_conrady);
k1_only.fix.distortion = Some(vec!["k2".into(), "k3".into(), "p1".into(), "p2".into()]);
let rational = CameraConfig::new(CameraModelSpec::pinhole(
    DistortionKind::Rational8,
    SensorKind::Identity,
));

// The config's own camera (Brown-Conrady, k3 fixed) is candidate 0.
let config = PlanarIntrinsicsConfig::default();
let comparison = compare_camera_models(
    &dataset,
    &config,
    &[k1_only, rational],
    &CrossValidationOptions::new(4, 0),
    &RunControl::default(),
)?;

for score in &comparison.candidates {
    if let ModelOutcome::Scored(s) = &score.outcome {
        println!(
            "{:?}, {} free: training {:.3} px, held-out {:.3} px ({:+.4} ± {:.4} vs best)",
            score.camera.model.distortion,
            score.free_parameters,
            s.training.rms,
            s.held_out.rms,
            s.delta_vs_best_px,
            s.delta_se_px,
        );
    }
}

// Prefer the fewest free parameters among the candidates within about two
// standard errors of the best.
let chosen = comparison
    .ranking
    .iter()
    .copied()
    .filter(|&i| match &comparison.candidates[i].outcome {
        ModelOutcome::Scored(s) => s.delta_vs_best_px <= 2.0 * s.delta_se_px,
        _ => false,
    })
    .min_by_key(|&i| comparison.candidates[i].free_parameters)
    .expect("the best candidate qualifies");
assert_eq!(chosen, 0); // the default model
# let training = |i: usize| match &comparison.candidates[i].outcome {
#     ModelOutcome::Scored(s) => s.training.rms,
#     _ => unreachable!("every candidate is scored"),
# };
# assert!(training(2) <= training(0) && training(0) <= training(1));
# assert_eq!(comparison.ranking.last(), Some(&1));
# Ok(())
# }
```

On this scene the training error falls with every added coefficient (`Rational8` lowest), while on held-out views `k1` alone is clearly worse and `Rational8` is no better than the default: its extra terms fit the noise of the views they saw. The same comparison from Python:

```python
k1_only = vc.CameraConfig(fix=vc.CameraFix(distortion=["k2", "k3", "p1", "p2"]))
rational = vc.CameraConfig(model=vc.CameraModelSpec(distortion="rational8"))
comparison = vc.compare_camera_models(dataset, [k1_only, rational], folds=4)
for candidate in comparison["candidates"]:
    outcome = candidate["outcome"]
    if outcome["status"] == "scored":
        print(candidate["free_parameters"], outcome["held_out"]["rms"], outcome["delta_se_px"])
```

### Reading the Comparison

- **Rank by held-out error, not training error.** `ranking` orders the comparable candidates by pooled held-out RMS. The training figure (`training`) is there to show the gap, not to choose.
- **Prefer the fewest free parameters within about two standard errors of the best.** `delta_vs_best_px` is a candidate's held-out RMS minus the best's, and `delta_se_px` its standard error, paired by fold so the variation the candidates share cancels. A difference inside two standard errors is not evidence; fewer parameters then extrapolate better beyond the image regions the views covered.
- **`free_parameters` shows what a candidate actually estimates.** It counts the parameters the fix lists leave free, not the model's slots. Under a Scheimpflug sensor the default fix list keeps only `k1` and `k2` (or `lambda`) free whatever the distortion model, so `Rational8` or `ThinPrism9` there has the same count as `BrownConrady5`: selecting the model alone changes nothing that is optimized. Free the extra coefficients through `camera.fix.distortion` to compare them.
- **A candidate that could not be fitted** is `ModelOutcome::Failed` with the fold and the error; the others are still scored. A candidate whose camera does not project some held-out points is listed but not ranked.
- The held-out board pose is re-solved for each view, so a camera term that a pose can partly imitate (a principal-point shift, for one) is partly forgiven. The comparison is sharpest for distortion terms, which no rigid pose of a plane can imitate.

### Practical Guidance

- **Default to `BrownConrady5` with `k3` fixed** (or, under a Scheimpflug sensor, the radial-only list) and move to a richer configuration when the held-out comparison says it helps, typically for a very wide field of view or a fisheye-adjacent lens.
- If you reach for an extended model, **free its new coefficients deliberately** via `camera.fix.distortion` when the sensor's default holds them.
- Freeing every coefficient of an extended model can diverge on cameras with few or poorly spread views; such a candidate shows up as failed in the comparison rather than stopping it.


## Special Cases

- **Wide-angle or fisheye-adjacent lenses**: set `camera.fix.distortion = Some(vec![])` and `intrinsics_init.fix_k3 = false` so the linear stage also fits `k3`. This is the one knob the default advice tells you not to flip without a reason.
- **`DistortionKind::None`**: pinhole-only, for pre-rectified or synthetic inputs. It is not accepted with a Scheimpflug sensor: the tilt is estimated jointly with the radial terms, so configuration validation rejects `None` with a typed error.
- **Hand-eye and laser workflows**: the same `camera.model.distortion` choice as planar intrinsics. A rig sets it per camera, and the rig laser workflows inherit each camera's model from the upstream rig.
