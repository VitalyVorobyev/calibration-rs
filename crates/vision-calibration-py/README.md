# vision-calibration-py

Python bindings for `calibration-rs`.

This crate exposes the high-level calibration workflows from `vision-calibration`
as typed Python runners:

| Runner | Workflow |
|---|---|
| `run_planar_intrinsics` | Planar intrinsics (pinhole or tilted Scheimpflug sensor) |
| `run_single_cam_handeye` | Single-camera hand-eye |
| `run_laserline_device` | Camera + laser plane |
| `run_rig_extrinsics` | Multi-camera rig extrinsics |
| `run_rig_handeye` | Rig + hand-eye |
| `run_rig_laserline_device` | Rig laser plane (frozen rig hand-eye input) |
| `run_rig_handeye_laserline` | Joint rig + hand-eye + laser plane |

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
prefer the typed dataclasses/models for new code.
