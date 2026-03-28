# vision-calibration-py

Python bindings for `calibration-rs`.

This crate exposes high-level calibration workflows from `vision-calibration`,
low-level geometric solvers from `vision-geometry`, and multi-view geometry
pipelines from `vision-mvg`.

### Calibration workflows

- planar intrinsics
- single-camera hand-eye
- rig extrinsics
- rig hand-eye
- laserline device
- scheimpflug intrinsics

### Geometry (`vision_calibration.geometry`)

- Fundamental matrix: 7-point, 8-point, RANSAC 8-point
- Essential matrix: 5-point (Nister), decomposition
- Homography: DLT, RANSAC DLT
- Camera matrix: DLT estimation, K/R/t decomposition
- Triangulation: linear DLT

### Multi-view geometry (`vision_calibration.mvg`)

- Relative pose recovery (minimal and RANSAC)
- Essential matrix and homography RANSAC estimation
- Homography decomposition and transfer
- Two-view triangulation with quality metrics
- Scene degeneracy analysis
- Sampson distance and symmetric transfer error

## Build locally

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
dataset = vc.PlanarDataset(views=[vc.PlanarView(observation=obs)] * 3)
config = vc.PlanarCalibrationConfig(
    max_iters=80,
    robust_loss=vc.robust_huber(1.0),
)

result = vc.run_planar_intrinsics(dataset, config)
print(result.mean_reproj_error)
```

Scheimpflug workflow:

```python
import vision_calibration as vc

obs = vc.Observation(
    points_3d=[(0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.1, 0.1, 0.0), (0.0, 0.1, 0.0)],
    points_2d=[(100.0, 100.0), (200.0, 100.0), (200.0, 200.0), (100.0, 200.0)],
)
dataset = vc.PlanarDataset(views=[vc.PlanarView(observation=obs)] * 3)
config = vc.ScheimpflugIntrinsicsCalibrationConfig(
    fix_scheimpflug=vc.ScheimpflugFixMask(tilt_x=False, tilt_y=False),
)
result = vc.run_scheimpflug_intrinsics(dataset, config)
print(result.camera.sensor)
```

Geometry and MVG:

```python
import numpy as np
import vision_calibration as vc

# Estimate a homography from point correspondences
src = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)
dst = np.array([[0.1, 0.1], [1.1, 0.0], [1.2, 1.1], [0.0, 1.0]], dtype=np.float64)
h = vc.geometry.dlt_homography(src, dst)

# RANSAC essential matrix + pose recovery
corrs = np.random.randn(50, 4)  # (N, 4): [x1, y1, x2, y2]
opts = vc.RansacOptions(max_iters=1000, thresh=0.01)
result = vc.mvg.estimate_essential(corrs, opts)
print(result.essential.shape, len(result.inliers))
```

## Migration: hard break to typed high-level API

High-level runner functions now require typed dataset/config objects.
Raw mapping/list payloads are no longer accepted by:

- `run_planar_intrinsics`
- `run_scheimpflug_intrinsics`
- `run_single_cam_handeye`
- `run_rig_extrinsics`
- `run_rig_handeye`
- `run_laserline_device`

Before (no longer supported):

```python
import vision_calibration as vc

result = vc.run_scheimpflug_intrinsics(
    {"views": [...]},
    {"max_iters": 80},
)
```

After (typed high-level API):

```python
import vision_calibration as vc

dataset = vc.PlanarDataset(views=[...])
config = vc.ScheimpflugIntrinsicsCalibrationConfig(max_iters=80)
result = vc.run_scheimpflug_intrinsics(dataset, config)
```

If you still need raw serde payload control for migration/interop, use low-level
helpers from `vision_calibration._api` (for example
`_run_scheimpflug_intrinsics_raw`).

## Runnable Python examples

Python workflow examples live in `crates/vision-calibration-py/examples/` and
mirror the Rust examples from `crates/vision-calibration/examples/`.

Install detector dependencies for real-image examples:

```bash
./.venv/bin/python -m pip install "vision-calibration[examples]"
```

Run all:

```bash
for f in crates/vision-calibration-py/examples/*.py; do ./.venv/bin/python "$f"; done
```

Run individual examples:

```bash
./.venv/bin/python crates/vision-calibration-py/examples/planar_synthetic.py
./.venv/bin/python crates/vision-calibration-py/examples/planar_real.py
./.venv/bin/python crates/vision-calibration-py/examples/stereo_session.py
./.venv/bin/python crates/vision-calibration-py/examples/stereo_charuco_session.py
./.venv/bin/python crates/vision-calibration-py/examples/handeye_synthetic.py
./.venv/bin/python crates/vision-calibration-py/examples/handeye_session.py
./.venv/bin/python crates/vision-calibration-py/examples/rig_handeye_synthetic.py
./.venv/bin/python crates/vision-calibration-py/examples/laserline_device_session.py
```

Notes:

- `planar_real.py`, `stereo_session.py`, `stereo_charuco_session.py`, and
  `handeye_session.py` run detector-based corner extraction from real images
  using `calib-targets`.
- `vision_calibration.types` is low-level compatibility surface for advanced
  interop only; prefer typed dataclasses/models for new code.
