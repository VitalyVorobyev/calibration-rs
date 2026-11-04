# vision-calibration

Camera calibration in Rust: one camera or a multi-camera rig, on a robot arm
or not, with a laser plane or not, for every supported camera model, with the
uncertainty of each estimate.

## Which crate do I want?

This one. `vision-calibration` re-exports the whole toolbox under one module
tree, and its module paths are the stable API. The other crates are its
building blocks, published so it can depend on them:

| Crate | Holds |
|---|---|
| `vision-calibration` | everything below, as one API: workflows, dataset manifests, detection, camera models, MVG |
| `vision-calibration-pipeline` | the calibration workflows and the session API |
| `vision-calibration-core` | math types, camera models, synthetic data |
| `vision-calibration-linear` | closed-form initialization (Zhang, hand-eye, laser plane) |
| `vision-calibration-optim` | non-linear least squares and parameter covariance |
| `vision-calibration-dataset` | the dataset manifest and the sample datasets |
| `vision-calibration-detect` | calibration-target detectors and the detection cache |
| `vision-geometry`, `vision-mvg` | two-view and PnP solvers; multiple-view geometry over calibrated cameras |
| `vision-calibration-cli` | the `vcal` program: calibrate a folder of images without code |

From Python, install the `vision-calibration` package from PyPI instead.

## Quick start

```toml
[dependencies]
vision-calibration = { version = "0.16", features = ["synthetic"] }
```

Eight synthetic views of an 8×6 board (40 mm pitch) calibrate a camera:

```rust
use vision_calibration::planar_intrinsics::run_calibration;
use vision_calibration::prelude::*;
use vision_calibration::synthetic::{planar, poses};

fn main() -> anyhow::Result<()> {
    let truth = make_pinhole_camera(
        FxFyCxCy { fx: 800.0, fy: 780.0, cx: 640.0, cy: 360.0 },
        BrownConrady5 { k1: 0.05, k2: -0.02, ..BrownConrady5::default() },
    );
    let board = planar::grid_points(8, 6, 0.04);
    let specs = poses::seeded_board_pose_specs(8, 0.35, 3);
    let poses = poses::centered_board_poses(&specs, (0.14, 0.10), 0.6);
    let views = planar::project_views_all(&truth, &board, &poses)?;
    let dataset = PlanarDataset::new(views.into_iter().map(View::without_meta).collect())?;

    let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
    session.set_input(dataset)?;
    run_calibration(&mut session)?;

    let export = session.export()?;
    println!("fx = {:.2} px", export.params.camera.intrinsics.fx);
    Ok(())
}
```

### On real images

A dataset manifest names the images, the target and the workflow;
`calibrate_dataset` detects the target and runs the workflow. The `fetch`
feature downloads a sample dataset with a ready manifest:

```rust,no_run
use vision_calibration::dataset::{DatasetSpec, samples};
use vision_calibration::control::RunControl;
use vision_calibration::dataset_runner::{Detection, calibrate_dataset};

fn main() -> anyhow::Result<()> {
    let manifest = samples::fetch("opencv_left", &samples::default_cache_dir()?)?;
    let spec = DatasetSpec::from_path(&manifest)?;
    let dir = manifest.parent().expect("the manifest sits in the sample's folder");
    let run = calibrate_dataset(&spec, dir, None, Detection::Uncached, None, &RunControl::default())?;
    println!("{} of {} views used", run.usable_views, run.total_views);
    Ok(())
}
```

## Problem Types

| Problem Type | Steps |
|---|---|
| `PlanarIntrinsicsProblem` (straight or tilted sensor) | `step_init` → `step_optimize` |
| `SingleCamHandeyeProblem` (any camera model) | `step_intrinsics_init` → `step_intrinsics_optimize` → `step_handeye_init` → `step_handeye_optimize` |
| `LaserlineDeviceProblem` (any camera model) | `step_init` → `step_optimize` |
| `RigExtrinsicsProblem` (any camera model per camera) | `step_intrinsics_init_all` → `step_intrinsics_optimize_all` → `step_rig_init` → `step_rig_optimize` |
| `RigHandeyeProblem` | rig extrinsics steps → `step_handeye_init` → `step_handeye_optimize` |
| `RigLaserlineDeviceProblem` | `step_init` → `step_optimize` (frozen rig hand-eye export as input) |
| `RigHandeyeLaserlineProblem` | `run_calibration` (joint solve) |

Each problem type also provides a `run_calibration` convenience function that
runs all steps. The per-module documentation lists the exact step names and
options.

## Module Organization

| Module | Description |
|--------|-------------|
| `session`, `common` | Session framework (`CalibrationSession`, `ProblemType`) and shared step/config types |
| `planar_intrinsics` | Single-camera intrinsics (Zhang's method), for any camera model including a Scheimpflug sensor |
| `single_cam_handeye`, `laserline_device` | Hand-eye and camera + laser plane calibration, for any camera model |
| `rig_extrinsics`, `rig_handeye`, `rig_laserline_device`, `rig_handeye_laserline` | Multi-camera rig workflows |
| `device_seed`, `dataset_runner` | Device-spec seeding and manifest-driven dataset runs |
| `dataset`, `detect` | Dataset manifests and target detection |
| `analysis` | Quality verdict, reprojection reports and per-feature residuals |
| `core`, `optim` | Math types and camera models; the optimization vocabulary (robust losses, solver backends, projection uncertainty) |
| `mvg` | Two-view solvers and N-view MVG (triangulation, bundle adjustment, rectification, dense stereo) |
| `synthetic` | Deterministic synthetic data generation (feature `synthetic`) |
| `prelude` | Convenient re-exports |

## Examples

Runnable examples (`cargo run --release -p vision-calibration --example <name>`)
live in [`examples/`](examples/); the book's workflow chapters walk through them.

## See Also

- [Book](https://vitalyvorobyev.github.io/calibration-rs)
- [API reference](https://vitalyvorobyev.github.io/calibration-rs/api/vision-calibration/index.html)
- [vision-calibration-pipeline](https://crates.io/crates/vision-calibration-pipeline): pipelines and session API
