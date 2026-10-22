# vision-calibration-py

Python bindings for `calibration-rs`.

This crate exposes the calibration workflows of `vision-calibration` to
Python, with the Rust names: a one-call runner and a step-by-step session per
workflow, manifest-driven runs from a folder of images, the target detectors
on numpy images, and calibrated cameras that project numpy arrays.

| Runner | Session | Workflow |
|---|---|---|
| `run_planar_intrinsics` | `PlanarIntrinsicsSession` | Planar intrinsics (pinhole or tilted Scheimpflug sensor) |
| `run_single_cam_handeye` | `SingleCamHandeyeSession` | Single-camera hand-eye |
| `run_laserline_device` | `LaserlineDeviceSession` | Camera + laser plane |
| `run_rig_extrinsics` | `RigExtrinsicsSession` | Multi-camera rig extrinsics |
| `run_rig_handeye` | `RigHandeyeSession` | Rig + hand-eye |
| `run_rig_laserline_device` | `RigLaserlineDeviceSession` | Rig laser plane (frozen rig hand-eye input) |
| `run_rig_handeye_laserline` | `RigHandeyeLaserlineSession` | Joint rig + hand-eye + laser plane |

## Install

```bash
pip install vision-calibration
```

To build from source instead:

```bash
maturin develop -m crates/vision-calibration-py/Cargo.toml
```

## Python package

The Python package name is `vision_calibration`.

```python
import vision_calibration as vc

print(vc.__version__)

# Build Python-native dataset/config objects with docstrings:
obs = vc.Observation(
    points_3d=[(0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.1, 0.1, 0.0), (0.0, 0.1, 0.0)],
    points_2d=[(100.0, 100.0), (200.0, 100.0), (200.0, 200.0), (100.0, 200.0)],
)
dataset = vc.PlanarIntrinsicsInput(views=[vc.PlanarView(observation=obs)] * 3)
config = vc.PlanarIntrinsicsConfig(
    solver=vc.SolverConfig(max_iters=80, robust_loss=vc.robust_huber(1.0)),
)

result = vc.run_planar_intrinsics(dataset, config)
print(result.mean_reproj_error)
```

Tilted (Scheimpflug) sensor: select it in the camera model. By default the
tilt is refined and only the leading distortion terms are free; list parameter
names in `CameraFix` (for example `CameraFix(sensor=["tilt_x"])`) to fix others.

```python
import vision_calibration as vc

obs = vc.Observation(
    points_3d=[(0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.1, 0.1, 0.0), (0.0, 0.1, 0.0)],
    points_2d=[(100.0, 100.0), (200.0, 100.0), (200.0, 200.0), (100.0, 200.0)],
)
dataset = vc.PlanarIntrinsicsInput(views=[vc.PlanarView(observation=obs)] * 3)
config = vc.PlanarIntrinsicsConfig(
    camera=vc.CameraConfig(model=vc.CameraModelSpec(sensor="scheimpflug")),
)
result = vc.run_planar_intrinsics(dataset, config)
print(result.camera.sensor)
```

Every workflow takes its camera model the same way: the single-camera
hand-eye and laserline-device configs have a `camera` field, and the rig
configs take the same `camera` for every rig camera, plus `camera_overrides`
to give individual cameras their own model and fix lists
(`vc.CameraOverride(camera=1, model=vc.CameraModelSpec(sensor="scheimpflug"))`).
Results report each camera as a `PinholeCamera` or, for a tilted sensor, a
`PinholeScheimpflugCamera`, whose `distortion` is the configured model's.

## Sessions and steps

A session runs a workflow step by step, with the Rust step names. Each step
takes its options (a dataclass; `None` fields keep the config) and returns its
result as a dataclass; the `*_with_seed` steps take manual-init seeds. The
problem's seeds and results live in its module (`vision_calibration.rig_extrinsics`,
...).

```python
import vision_calibration as vc
from vision_calibration import planar_intrinsics

session = vc.PlanarIntrinsicsSession()
session.set_input(dataset)
init = session.step_init_with_seed(
    planar_intrinsics.IntrinsicsManualInit(
        intrinsics=vc.PinholeIntrinsics(fx=800.0, fy=800.0, cx=640.0, cy=360.0)
    )
)
session.step_optimize(vc.IntrinsicsOptimizeOptions(max_iters=100))
result = session.export()
for entry in session.log:  # the init log and the solve notes
    print(entry.operation, entry.notes)
saved = session.to_json()  # resume later with PlanarIntrinsicsSession.from_json
```

## Manifests and detection

`calibrate_dataset` runs a dataset manifest (TOML or JSON) end to end:
detection, the problem's input, calibration and export. `sniff_folder` drafts
a manifest from a folder; `FsDetectionCache` keeps detections between runs.
A manifest that leaves a convention open raises `AskUserError` (a `RunError`)
naming the field to set; other runner failures raise `RunError` with a `kind`.

```python
cache = vc.FsDetectionCache("cache/kuka_1")
result = vc.calibrate_dataset("data/kuka_1/dataset.toml", cache=cache)
```

The laser topologies take a `laser_extractor`: any object with a `name` and an
`extract(image, spec)` method returning the laser line's `(N, 2)` pixels.
`detect_target` runs a manifest's target detector on one `(H, W)` `uint8`
image and returns `Features` (`image_xy` and `world_xyz` arrays):

```python
features = vc.detect_target(gray, {"kind": "chessboard", "rows": 17, "cols": 28, "square_size_m": 0.02})
view = vc.PlanarView(observation=features.to_observation())
```

## Cameras and residuals

`CameraModel` builds a calibrated camera (any distortion model, identity or
Scheimpflug sensor) and projects numpy arrays: `project` `(N, 3) -> (N, 2)`,
`unproject` to ray origins and directions, and `undistort` to an ideal pinhole
image with the same intrinsics. Every result's `per_feature_residuals` holds
the per-corner and per-laser-pixel residuals as numpy tables.

```python
cam = vc.CameraModel(result.camera)
pixels = cam.project(points_cam)
res = result.per_feature_residuals
print(res.target.error_px.max(), res.target.observed_px.shape)
```

## Uncertainty

Every result reports how well the data determines its parameters:
`result.uncertainty` holds the covariance of the estimated parameters, the
noise level the residuals imply and, when no covariance can be given, the
reason. `vc.projection_uncertainty` turns it into the standard deviation, in
pixels, with which the camera places a ray in its image. A runnable example
on synthetic views with 0.2 px of corner noise:

```python
import math
import random

import vision_calibration as vc

rng = random.Random(0)
board = [(i * 0.04, j * 0.04, 0.0) for i in range(8) for j in range(6)]


def project(pt, yaw, tilt, z):
    """Noisy pixel of a board point for a board turned by (yaw, tilt) at depth z."""
    x, y, _ = pt
    cy_, sy_, ct, st = math.cos(yaw), math.sin(yaw), math.cos(tilt), math.sin(tilt)
    zc = -sy_ * x + cy_ * st * y + z
    xn = (cy_ * x + sy_ * st * y) / zc
    yn = (ct * y) / zc
    return 800.0 * xn + 640.0 + rng.gauss(0, 0.2), 780.0 * yn + 360.0 + rng.gauss(0, 0.2)


views = [
    vc.PlanarView(
        observation=vc.Observation(
            points_3d=board,
            points_2d=[project(p, -0.3 + 0.1 * v, 0.25 * math.sin(v), 0.45 + 0.04 * v) for p in board],
        )
    )
    for v in range(8)
]
result = vc.run_planar_intrinsics(vc.PlanarIntrinsicsInput(views=views))

u = result.uncertainty
if u.available:
    k = result.camera.intrinsics
    print(f"fx = {k.fx:.1f} +- {u.std_dev('fx'):.2f} px")
    print(f"corr(fx, cx) = {u.correlation('fx', 'cx'):.2f}")
    for n in u.noise:
        print(f"{n.kind} noise sigma = {n.sigma:.3f}")
else:
    print("no covariance:", u.reason, u.reason_labels)

# Where in the image is the camera well determined? Pass the image size.
p = vc.projection_uncertainty(result, (1280, 720))
print(f"projection sigma: centre {p.center:.3f} px, max {p.max:.3f} px")
rows = p.grid()  # 12 rows of 16 values, top row first (None: no ray reaches)
```

`u.std_dev(label)` and `u.correlation(a, b)` take parameter labels such as
`"fx"`, `"k1"` or `"tilt_x"` (rig results prefix a camera, `"cam1.fx"`);
`u.fixed` lists the parameters the solve held fixed, which the covariance is
conditional on. When `u.available` is false, `u.reason` is one of
`"not_converged"`, `"rank_deficient"`, `"at_bound"`, `"no_redundancy"` or
`"not_computed"`, and `u.reason_labels` names the parameters involved.
`projection_uncertainty` raises `ValueError` when the result has no
covariance of the camera's parameters, which includes rig results that held
the intrinsics fixed (the default); the `camera=` argument selects a camera
of a rig. The book's Uncertainty part explains how to read these numbers.

## Runnable Python examples

Workflow examples live in `crates/vision-calibration-py/examples/` and mirror the
Rust examples in `crates/vision-calibration/examples/`. The real-image examples
need the detector extras (`pip install "vision-calibration[examples]"`) and the
datasets under `data/` in the repository.

```bash
for f in crates/vision-calibration-py/examples/*.py; do python "$f"; done
```

`vision_calibration.types` is an advanced-interop surface for raw serde payloads;
prefer the typed dataclasses/models for new code. Its `TypedDict`s are
generated from the JSON Schemas of the Rust types (`cargo xtask emit-schemas`
writes them, and CI fails when they are stale), so each carries its Rust name
and shape: `PlanarIntrinsicsConfig`, `CameraParams`, `Iso3`, and one
`<Enum>_<Variant>` type per variant of an enum with data, such as
`RobustLoss_Huber`.
