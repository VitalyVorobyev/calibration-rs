<h1>
  <a href="https://vitavision.dev/">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="book/src/img/vv-favicon-dark.svg">
      <img src="book/src/img/vv-favicon-dark.svg" alt="vitavision.dev" height="48" align="left">
    </picture>
  </a>
  &nbsp;vision-calibration
</h1>

[![Crates.io](https://img.shields.io/crates/v/vision-calibration.svg)](https://crates.io/crates/vision-calibration)
[![PyPI](https://img.shields.io/pypi/v/vision-calibration.svg)](https://pypi.org/project/vision-calibration/)
[![Docs.rs](https://docs.rs/vision-calibration/badge.svg)](https://docs.rs/vision-calibration)
[![CI](https://github.com/VitalyVorobyev/calibration-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/VitalyVorobyev/calibration-rs/actions/workflows/ci.yml)
[![Docs](https://github.com/VitalyVorobyev/calibration-rs/actions/workflows/publish-docs.yml/badge.svg)](https://vitalyvorobyev.github.io/calibration/)
[![Audit](https://github.com/VitalyVorobyev/calibration-rs/actions/workflows/audit.yml/badge.svg)](https://github.com/VitalyVorobyev/calibration-rs/actions/workflows/audit.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![MSRV](https://img.shields.io/badge/MSRV-1.93-blue.svg)](https://blog.rust-lang.org/2025/10/30/Rust-1.93.0/)

**Camera calibration in Rust, from images to a validated metric model.** Point
it at a folder of calibration images and get back intrinsics, distortion,
multi-camera rig extrinsics, hand-eye transforms, and laser-plane geometry —
with per-feature residuals you can inspect, JSON checkpoints you can resume
from, and a desktop app to see what the numbers mean.

Available as a Rust crate, a Python package, and a GUI.

## What it calibrates

| Workflow | Solves for | Typical input |
|---|---|---|
| **Planar intrinsics** | `fx, fy, cx, cy, skew` + Brown-Conrady distortion | one camera, N views of a flat target |
| **Scheimpflug intrinsics** | the above + sensor tilt (`tilt_x`, `tilt_y`) | a tilted-sensor camera |
| **Rig extrinsics** | per-camera intrinsics + camera-to-rig poses | 2+ synchronised cameras |
| **Single-camera hand-eye** | intrinsics + camera-to-gripper (or -to-base) transform | camera on/observing a robot arm |
| **Rig hand-eye** | rig extrinsics + hand-eye, eye-in-hand or eye-to-hand | multi-camera head on a robot |
| **Laserline device** | camera + laser-plane geometry | camera + line projector |
| **Rig laserline device** | one laser plane per camera, upstream calibration frozen | multi-camera laser triangulation head |
| **Rig hand-eye + laserline** | all of the above in one joint bundle adjustment | full laser-profiling station |

Each is a *problem type* driven through the same session API: set input, run
init and optimize steps, export. Sessions serialize to JSON, so a long run can
be checkpointed, inspected, and resumed.

**Targets** — chessboard, ChArUco, PuzzleBoard, and coded ring-grid, behind one
detector interface with a content-addressed detection cache (re-running with
unchanged images and unchanged detector parameters costs a filesystem read, not
a re-detection).

**Camera model** — cameras compose as `pixel = K(sensor(distortion(projection(dir))))`,
so a Scheimpflug (tilted-sensor) camera is the pinhole model with a non-identity
sensor homography rather than a separate code path. See
[ADR 0005](docs/adrs/0005-composable-camera-model.md).

## Install

```toml
# Cargo.toml
vision-calibration = "0.8"
```

```bash
pip install vision-calibration
```

The Rust facade is module-first — reach for `vision_calibration::<workflow>::…`
or `vision_calibration::prelude::*` rather than broad top-level imports.

## Quick start

Calibrate a single camera's intrinsics from planar correspondences:

```rust
use vision_calibration::prelude::*;
use vision_calibration::planar_intrinsics::{step_init, step_optimize};

let dataset: PlanarDataset = /* your 2D↔3D correspondences, N views */;

let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
session.set_input(dataset)?;

step_init(&mut session, None)?;      // Zhang's method + linear distortion fit
step_optimize(&mut session, None)?;  // Levenberg-Marquardt bundle adjustment

let result = session.export()?;
println!("{:?} at {:.4} px", result.params.camera, result.mean_reproj_error);
```

The same thing from Python:

```python
import vision_calibration as vc

obs = vc.Observation(
    points_3d=[(0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.1, 0.1, 0.0), (0.0, 0.1, 0.0)],
    points_2d=[(100.0, 100.0), (200.0, 100.0), (200.0, 200.0), (100.0, 200.0)],
)
dataset = vc.PlanarDataset(views=[vc.PlanarView(observation=obs)] * 3)
result = vc.run_planar_intrinsics(
    dataset,
    vc.PlanarCalibrationConfig(solver=vc.SolverConfig(max_iters=80)),
)
print(result.mean_reproj_error)
```

Every workflow also has a `run_calibration` convenience function that runs its
steps in order, for when you do not need to intervene between them.

## Starting from a folder of images

You do not have to assemble correspondences yourself. Describe the dataset once
in a `DatasetSpec` manifest — where the images are, what target is on them, how
views pair across cameras, where robot poses come from — and the *dataset
runner* detects features and builds the problem input for you:

```rust
use std::path::Path;
use vision_calibration::dataset::DatasetSpec;
use vision_calibration::dataset_runner::build_planar_input;
use vision_calibration::detect::FsDetectionCache;

let spec: DatasetSpec = serde_json::from_str(&std::fs::read_to_string("dataset.json")?)?;
let cache = FsDetectionCache::new(".detect-cache");
let run = build_planar_input(&spec, Path::new("."), &cache, false)?;
println!("{} of {} views usable", run.usable_views, run.total_views);

let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
session.set_input(run.dataset)?;
```

`cargo run -p vision-calibration-dataset --features cli --bin generate-manifest`
sniffs a folder and writes a starting manifest; the validator flags anything it
could not determine rather than guessing.

## Desktop app

`app/` is a Tauri 2 + React desktop application that runs calibrations
end-to-end and — the part a terminal cannot give you — *diagnoses* them:
residual overlays on the source images, per-pose and per-camera residual
panels, a 3D rig viewer, epipolar sanity checks, and dense stereo.

```bash
cd app
bun install
bun run tauri dev      # NOT `bun run dev` — that starts Vite without the Tauri APIs
```

It uses **bun** exclusively. See [`app/README.md`](app/README.md).

## Examples

Runnable end-to-end, on synthetic and real data:

```bash
cargo run -p vision-calibration --example planar_synthetic       # planar intrinsics, synthetic
cargo run -p vision-calibration --example planar_real            # planar intrinsics, real images
cargo run -p vision-calibration --example stereo_session         # stereo rig extrinsics
cargo run -p vision-calibration --example stereo_charuco_session # stereo rig, ChArUco target
cargo run -p vision-calibration --example handeye_synthetic      # single-camera hand-eye
cargo run -p vision-calibration --example handeye_session        # hand-eye, KUKA robot data
cargo run -p vision-calibration --example rig_handeye_synthetic  # multi-camera rig hand-eye
cargo run -p vision-calibration --example laserline_device_session  # camera + laser plane
cargo run -p vision-calibration --example mvg_two_view           # two-view geometry
cargo run -p vision-calibration --example dense_stereo_real      # rectification + dense stereo
```

Python counterparts live in `crates/vision-calibration-py/examples/`; the
real-image ones need the optional extras
(`pip install "vision-calibration[examples]"`).

## Documentation

- **[The book](https://vitalyvorobyev.github.io/calibration/)** — the camera
  model, every solver, and a walkthrough per workflow. Start here.
- **[API reference](https://docs.rs/vision-calibration)** — docs.rs.
- **[Tutorials](docs/tutorials/)** — hands-on onboarding.
- **[ADRs](docs/adrs/)** — why the design is the way it is.
- **[CHANGELOG](CHANGELOG.md)** — what changed, and what to do about it.

## Project

A Rust workspace of eleven crates. Nine publish to crates.io — `-core`,
`-linear`, `-optim`, `-pipeline`, `-dataset`, `-detect`, `vision-geometry`,
`vision-mvg`, and the `vision-calibration` facade that re-exports them —
plus `-py` (PyPI) and an internal benchmark crate. Layering is enforced:
solvers do not know about pipelines, and pipelines do not know about the GUI.
See [ADR 0006](docs/adrs/0006-layered-crate-architecture.md) and
[AGENTS.md](AGENTS.md) for the rules, build commands, and contribution
workflow.

The project is pre-1.0 and breaking changes still happen; they are listed in
the CHANGELOG with migration notes. MSRV is 1.93 ([policy](docs/MSRV.md)).

**On how this is built:** development uses AI coding assistants (Codex and
Claude Code) as implementation tools, so not every line is human-reviewed
before merge. The author is a computer-vision engineer and validates
algorithmic behaviour and numerical results against real datasets, with
`fmt`/`clippy`/test/doc gates plus a registry-driven acceptance suite that
hard-gates reprojection error on real calibration data before every release.

Licensed under the [MIT License](LICENSE).
