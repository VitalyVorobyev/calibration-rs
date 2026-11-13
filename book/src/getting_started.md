# Getting Started

This chapter calibrates a camera twice, each time in Rust and in Python. The first run uses synthetic data: a planar board seen from several poses, the two-step calibration (linear initialization, then non-linear refinement), and the intrinsics and reprojection error it recovers. The second run, [Calibrate Your Own Images](#calibrate-your-own-images), starts from a folder of board images and ends with the camera, the uncertainty of each parameter and a quality verdict.

Both interfaces drive the same solvers and expose the same workflow: a session per problem type with its step functions, and a one-call runner per workflow (`run_calibration` in Rust; `planar_intrinsics.run_calibration`, `rig_extrinsics.run_calibration`, ... in Python). The Rust program below steps through a session; the Python program uses the runner, and [Step by Step in Python](#step-by-step-in-python) shows the session.

## Install

### Rust

```toml
[dependencies]
vision-calibration = { version = "0.18", features = ["synthetic"] }
anyhow = "1"
```

`vision-calibration` is the facade crate: it re-exports the calibration workflows, the dataset manifest, the target detectors' cache and the multiple-view geometry tools. The `synthetic` feature adds the synthetic-data generators the example below calibrates; a calibration from real images does not need it.

### Python

```bash
pip install vision-calibration
```

Python 3.10 or newer. To build the extension from a checkout instead, run `maturin develop -m crates/vision-calibration-py/Cargo.toml`. The package is `vision_calibration`; the examples below import it as `vc`.

### Command line

```bash
cargo install vision-calibration-cli
```

The Python package installs the same `vcal` program. It calibrates a folder of images without code: [The Command Line](cli.md).

## A Minimal Calibration

Each program builds a synthetic camera with known intrinsics and Brown-Conrady distortion, projects an 8x6 board (40 mm squares) from several poses, and recovers the camera from those pixels alone.

### Rust

```rust
use anyhow::Result;
use vision_calibration::core::{BrownConrady5, FxFyCxCy, PlanarDataset, View, make_pinhole_camera};
use vision_calibration::planar_intrinsics::{PlanarIntrinsicsProblem, step_init, step_optimize};
use vision_calibration::session::CalibrationSession;
use vision_calibration::synthetic::planar;

fn main() -> Result<()> {
    // 1. Define the ground-truth camera
    let k_gt = FxFyCxCy {
        fx: 800.0, fy: 780.0, cx: 640.0, cy: 360.0
    };
    let dist_gt = BrownConrady5 {
        k1: 0.05, k2: -0.02, k3: 0.0,
        p1: 0.001, p2: -0.001, iters: 8,
    };
    let camera = make_pinhole_camera(k_gt, dist_gt);

    // 2. Generate synthetic observations
    let board = planar::grid_points(8, 6, 0.04); // 8x6 grid, 40 mm squares
    let poses = planar::poses_yaw_y_z(6, -0.2, 0.08, 0.5, 0.05);
    let views = planar::project_views_all(&camera, &board, &poses)?;

    // 3. Create a session and set the input
    let dataset = PlanarDataset::new(
        views.into_iter().map(View::without_meta).collect()
    )?;
    let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
    session.set_input(dataset)?;

    // 4. Run the calibration
    step_init(&mut session, None)?;      // linear initialization
    step_optimize(&mut session, None)?;  // non-linear refinement

    // 5. Inspect the result
    let export = session.export()?;
    let k = export.camera.intrinsics;
    println!("fx={:.1}, fy={:.1}, cx={:.1}, cy={:.1}",
             k.fx, k.fy, k.cx, k.cy);
    println!("Mean reprojection error: {:.4} px",
             export.mean_reproj_error);

    Ok(())
}
```

The library ships this program as an example:

```bash
cargo run -p vision-calibration --example planar_synthetic
```

### Python

```python
import math
import vision_calibration as vc

# 1. The ground-truth camera and a planar board (8x6 grid, 40 mm squares)
fx, fy, cx, cy = 800.0, 780.0, 640.0, 360.0
k1, k2, p1, p2 = 0.05, -0.02, 0.001, -0.001
board = [(i * 0.04, j * 0.04, 0.0) for i in range(8) for j in range(6)]


def project(pt, yaw, tilt, z):
    """Pixel of a board point for a board turned by (yaw, tilt) at depth z."""
    x, y, _ = pt
    cyw, syw, ct, st = math.cos(yaw), math.sin(yaw), math.cos(tilt), math.sin(tilt)
    xc = cyw * x + syw * st * y          # R = Ry(yaw) * Rx(tilt)
    yc = ct * y
    zc = -syw * x + cyw * st * y + z
    xn, yn = xc / zc, yc / zc
    r2 = xn * xn + yn * yn
    radial = 1.0 + k1 * r2 + k2 * r2 * r2
    xd = xn * radial + 2 * p1 * xn * yn + p2 * (r2 + 2 * xn * xn)
    yd = yn * radial + p1 * (r2 + 2 * yn * yn) + 2 * p2 * xn * yn
    return (fx * xd + cx, fy * yd + cy)


# 2. Synthetic observations: varied yaw, tilt and distance
poses = [(-0.3 + 0.1 * v, 0.25 * math.sin(v), 0.45 + 0.04 * v) for v in range(8)]
views = [
    vc.PlanarView(
        observation=vc.Observation(
            points_3d=board,
            points_2d=[project(p, yaw, tilt, z) for p in board],
        )
    )
    for yaw, tilt, z in poses
]

# 3. Build the input and run the calibration (initialization + refinement)
dataset = vc.PlanarIntrinsicsInput(views=views)
result = vc.planar_intrinsics.run_calibration(dataset)

# 4. Inspect the result
k = result.camera.intrinsics
print(f"fx={k.fx:.1f}, fy={k.fy:.1f}, cx={k.cx:.1f}, cy={k.cy:.1f}")
print(f"Mean reprojection error: {result.mean_reproj_error:.4f} px")
```

Real data replaces step 2 in both languages: detect the board corners in each image and pair every corner with its position on the board. [Calibrate Your Own Images](#calibrate-your-own-images) below does that for a folder of images; [Planar Intrinsics with Real Data](planar_real_data.md) shows the detection step by hand.

### Step by Step in Python

The runner above is a session run in one call. The session gives the steps, their results and the init log:

```python
session = vc.PlanarIntrinsicsSession()
session.set_input(dataset)
init = session.step_init()                 # linear initialization
print(f"init fx={init.camera.intrinsics.fx:.1f}")
session.step_optimize()                    # non-linear refinement
result = session.export()                  # the same PlanarIntrinsicsExport

cam = vc.CameraModel(result.camera)        # project with the calibrated camera
print(cam.project([0.0, 0.0, 1.0]))        # the principal point
```

## Calibrate Your Own Images

A folder of photographs of a calibration board is enough to start. Without code, `vcal init <folder>` and `vcal run <folder>/dataset.toml` do what this section does ([The Command Line](cli.md)). The program below takes three steps. It drafts a *dataset manifest* that describes the folder, waits for you to fill in what only you know, and then detects the board, calibrates and judges the result. It works for any [workflow](choose_workflow.md): the manifest's `topology` picks it.

### 1. Draft the manifest

The folder sniffer reads file and folder names, never image content. It groups images into cameras, finds a robot-pose file if there is one and picks the topology. What it cannot know, the board above all, it leaves as a placeholder and lists under `_unresolved`. For a folder of board images from one camera, the draft is:

```json
{
  "version": 1,
  "topology": "planar_intrinsics",
  "cameras": [{ "id": "cam0", "images": { "kind": "glob", "pattern": "*.png" } }],
  "target": { "kind": "chessboard", "rows": 0, "cols": 0, "square_size_m": 0.0 },
  "_unresolved": ["target"]
}
```

### 2. Fill in what only you know

Describe the board and delete `_unresolved`. For a chessboard, `rows` and `cols` count the inner corners, where four squares meet, and `square_size_m` is the measured side of a printed square in metres:

```json
"target": { "kind": "chessboard", "rows": 7, "cols": 11, "square_size_m": 0.03 }
```

The calibration refuses a manifest whose `_unresolved` list is not empty, so a placeholder never reaches a result. ChArUco boards, PuzzleBoards and ring grids are described in [Calibration Targets and Detection](detection.md), and robot poses and the other fields in [The Dataset Manifest](manifest.md).

### 3. Calibrate and judge

#### Rust

```toml
[dependencies]
vision-calibration = "0.18"
anyhow = "1"
serde_json = "1"
```

The real-image route needs no feature flags: detection, the manifest and the dataset runner are part of the default build.

```rust,no_run
use std::path::PathBuf;

use anyhow::{Result, bail};
use vision_calibration::analysis::{CheckStatus, QualityThresholds, assess};
use vision_calibration::common::CalibratedCameras;
use vision_calibration::core::Uncertainty;
use vision_calibration::dataset::{DatasetSpec, sniff_folder};
use vision_calibration::control::RunControl;
use vision_calibration::dataset_runner::{Detection, calibrate_dataset};
use vision_calibration::detect::FsDetectionCache;

fn main() -> Result<()> {
    let dir = PathBuf::from("my_board_images");
    let image_size = [1024, 576]; // width, height in pixels

    // 1. Draft the manifest and stop until it is filled in.
    let manifest = dir.join("dataset.json");
    if !manifest.exists() {
        let draft = sniff_folder(&dir)?;
        std::fs::write(&manifest, serde_json::to_string_pretty(&draft)?)?;
        bail!("fill in {:?} in {}, delete `_unresolved`", draft.unresolved, manifest.display());
    }

    // 2. Detect (cached for the next run) and calibrate.
    let spec = DatasetSpec::from_path(&manifest)?;
    let cache = FsDetectionCache::new(dir.join(".detections"));
    let run = calibrate_dataset(&spec, &dir, None, Detection::Cached(&cache), None, &RunControl::default())?;
    println!("{} of {} views used", run.usable_views, run.total_views);

    // 3. The camera and the uncertainty of its parameters.
    let export = &run.export;
    let k = &export.camera_params()[0].intrinsics;
    println!("fx={:.2} fy={:.2} cx={:.2} cy={:.2}", k.fx, k.fy, k.cx, k.cy);
    if let Uncertainty::Available(cov) = export.parameter_uncertainty() {
        for label in ["fx", "fy", "cx", "cy"] {
            println!("  σ({label}) = {:.3} px", cov.std_dev(label).unwrap_or(f64::NAN));
        }
    }

    // 4. The verdict, and the checks that did not pass.
    let report = assess(export, &[image_size], &QualityThresholds::default())?;
    println!("verdict: {:?}", report.verdict);
    for check in report.checks.iter().filter(|c| c.status != CheckStatus::Pass) {
        println!("  {:?} {:?}: {}", check.kind, check.status, check.message);
    }

    // 5. Keep the export: JSON with a published schema.
    std::fs::write(dir.join("calibration.json"), serde_json::to_string_pretty(export)?)?;
    Ok(())
}
```

`calibrate_dataset` reads the manifest's `topology` and runs that workflow with its default configuration; pass `Some(CalibrationConfig::...)` to change it (see [Configuring a Run](#configuring-a-run)). Its result holds the export, which every analysis accepts whichever workflow produced it, and the detections it left out with the reason for each. The library ships this program, generalized to several cameras, as the `calibrate_folder` example:

```bash
cargo run --release -p vision-calibration --example calibrate_folder -- my_board_images 1024 576
```

#### Python

```python
import json
import sys
from pathlib import Path

import vision_calibration as vc

folder = Path("my_board_images")
manifest = folder / "dataset.json"

# 1. Draft the manifest and stop until it is filled in.
if not manifest.exists():
    draft = vc.sniff_folder(folder)
    manifest.write_text(json.dumps(draft, indent=2))
    sys.exit(f"fill in {draft['_unresolved']} in {manifest}, delete _unresolved")

# 2. Detect (cached for the next run) and calibrate.
cache = vc.FsDetectionCache(folder / ".detections")
result = vc.calibrate_dataset(manifest, cache=cache).export

# 3. The camera and the uncertainty of its parameters.
k, u = result.camera.intrinsics, result.uncertainty
print(f"fx={k.fx:.2f} ± {u.std_dev('fx'):.3f}  fy={k.fy:.2f} ± {u.std_dev('fy'):.3f}")
print(f"cx={k.cx:.2f} ± {u.std_dev('cx'):.3f}  cy={k.cy:.2f} ± {u.std_dev('cy'):.3f}")

# 4. The verdict, and the checks that did not pass.
report = vc.assess(result, image_sizes=[(1024, 576)])
print("verdict:", report.verdict)
for check in report.checks:
    if check.status != "pass":
        print(" ", check.kind, check.status, check.message)
```

### Reading the output

On the 20 left-camera images of the stereo set in [Stereo Rig with Real Data](stereo_real_data.md) (a 7×11 chessboard with 30 mm squares, 1024×576 pixels), both programs print the same numbers; the Python program's output is:

```text
fx=713.85 ± 7.672  fy=724.67 ± 7.821
cx=523.41 ± 1.688  cy=285.95 ± 1.433
verdict: warn
  variance_factor not_applicable no expected feature noise was stated (expected_feature_noise_px)
  projection_sigma warn projection σ up to 7.065 px over the 1024×576 image (centre 1.665 px, corners 7.065 px)
```

Read it from the bottom up. The verdict is `warn`, not `pass`: where a ray lands in the image is uncertain by up to 7 px in the corners, against 1.7 px in the centre. The views did not cover the image corners well enough to pin the focal length down to better than about 1 % (σ(fx) ≈ 7.7 px). More views with the board near the corners and tilted would shrink it; [Planning the Next View](uncertainty_next_view.md) ranks candidate poses by how much each would help. The variance check is not applicable until you state the detector's noise (`expected_feature_noise_px`). [Judging a Result](quality.md) explains every check, and [Reading the Covariance](uncertainty_reading.md) the σ.

### No images at hand: a sample dataset

The library downloads real calibration datasets with a manifest already filled in. Each one comes from its publisher at a pinned revision, every file is checked against its SHA-256, and a second fetch reuses the cache:

| Sample | Workflow | Data | Licence |
|---|---|---|---|
| `opencv_left` | planar intrinsics | 13 views of a 6×9 chessboard, 640×480, OpenCV's stereo sample | Apache-2.0 |
| `opencv_stereo` | rig extrinsics | both cameras of the same stereo sample, the board turned by 90° in most views | Apache-2.0 |
| `opencv_fisheye_stereo` | rig extrinsics, fisheye | 34 views of a 6×8 chessboard, 1280×800, OpenCV's fisheye test set | none stated |
| `kuka_1` | single-camera hand-eye | 30 views of a 17×28 chessboard and the robot poses | none stated |
| `ds8` | rig hand-eye | three cameras on a robot arm, 42 poses of a 10×14 chessboard | US public domain |

Each sample carries the citation its publisher asks for (`citation` in the catalogue). In Rust, fetching needs the `fetch` feature, which adds an HTTPS client:

```toml
vision-calibration = { version = "0.18", features = ["fetch"] }
```

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

```python
manifest = vc.samples.fetch("opencv_left")
result = vc.calibrate_dataset(manifest).export
```

The cache is `vision-calibration` in the platform's cache directory; the `VISION_CALIBRATION_DATA` environment variable, or a directory passed to `fetch`, puts it elsewhere. The example program above runs on a fetched sample's folder as on your own. The fisheye sample needs a fisheye camera model in its configuration ([Fisheye Projection](fisheye.md)).

## Configuring a Run

Every workflow has a `*Config` type with the same grouped shape (`solver`, `camera`, `init`, ...). Rust updates it on the session or hands it to `calibrate_dataset` wrapped in `CalibrationConfig`; Python passes it to the runner, to `calibrate_dataset` or to the session's `set_config`.

### Rust

```rust,no_run
use vision_calibration::optim::RobustLoss;
use vision_calibration::planar_intrinsics::{PlanarIntrinsicsProblem, run_calibration};
use vision_calibration::session::CalibrationSession;
# fn main() -> anyhow::Result<()> {
# let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();

session.update_config(|c| {
    c.solver.max_iters = 80;
    c.optimize.reprojection_loss = RobustLoss::Huber { scale: 1.0 };
})?;
run_calibration(&mut session)?; // the same two steps in one call
# Ok(())
# }
```

From a manifest, the configuration names its workflow, and a configuration for a different problem kind than the manifest's is an error raised before any image is read:

```rust,no_run
use std::path::PathBuf;
use vision_calibration::control::RunControl;
use vision_calibration::dataset::DatasetSpec;
use vision_calibration::dataset_runner::{CalibrationConfig, Detection, calibrate_dataset};
use vision_calibration::detect::FsDetectionCache;
use vision_calibration::planar_intrinsics::PlanarIntrinsicsConfig;
# fn main() -> anyhow::Result<()> {
# let dir = PathBuf::from("my_board_images");
# let spec = DatasetSpec::from_path(dir.join("dataset.json"))?;
# let cache = FsDetectionCache::new(dir.join(".detections"));

let mut config = PlanarIntrinsicsConfig::default();
config.solver.max_iters = 80;
let config = CalibrationConfig::PlanarIntrinsics(config);
let run = calibrate_dataset(&spec, &dir, Some(config), Detection::Cached(&cache), None, &RunControl::default())?;
# Ok(())
# }
```

### Python

```python
config = vc.PlanarIntrinsicsConfig(
    optimize=vc.PlanarOptimizeConfig(reprojection_loss=vc.robust_huber(1.0)),
    solver=vc.SolverConfig(max_iters=80),
)
result = vc.planar_intrinsics.run_calibration(dataset, config)
result = vc.calibrate_dataset("my_board_images/dataset.json", config)  # from a manifest
```

## What Happens in Each Step

### Step 1: Linear Initialization

`step_init` (`session.step_init()` in Python; `planar_intrinsics.run_calibration` runs it first):

1. Computes a homography from each view's 2D-3D correspondences using DLT (direct linear transform)
2. Estimates intrinsics $K$ from the homographies using Zhang's method
3. Estimates lens distortion coefficients from homography residuals
4. Iterates steps 2-3 to refine the joint $K$ + distortion estimate
5. Recovers each view's camera pose from its homography and $K$

After this step, intrinsics are typically within 10-40% of the true values: good enough to start the non-linear optimization.

### Step 2: Bundle Adjustment

`step_optimize` runs Levenberg-Marquardt, minimizing the total reprojection error:

$$\min_{K,\,\mathbf{d},\,\{T_i\}} \sum_{i=1}^{N} \sum_{j=1}^{M_i} \left\| \pi(K, \mathbf{d}, T_i, \mathbf{P}_j) - \mathbf{p}_{ij} \right\|^2$$

where $\pi$ is the camera projection function, $T_i$ is the pose of view $i$, $\mathbf{d}$ are distortion coefficients, $\mathbf{P}_j$ are known 3D board points, and $\mathbf{p}_{ij}$ are observed pixel coordinates.

After optimization, expect under 2% error on intrinsics and under 1 px mean reprojection error on real data. The synthetic data above is noise-free, so both programs recover the ground truth almost exactly.

## Where to Go Next

- [Stereo Rig with Real Data](stereo_real_data.md): a two-camera rig from detected corners, in under a minute.
- [Planar Intrinsics Calibration](planar_intrinsics.md): the full mathematical treatment of this workflow.
- [Parameter Uncertainty](uncertainty.md): every result carries the covariance of its parameters.
- [Architecture Overview](architecture.md): how the crates fit together.
- [The Dataset Manifest](manifest.md): every field of the manifest, robot poses and image pairing included.
