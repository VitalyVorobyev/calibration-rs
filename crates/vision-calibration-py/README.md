# vision-calibration-py

Python bindings for `calibration-rs`.

This crate exposes high-level calibration workflows from `vision-calibration`:

- planar intrinsics
- single-camera hand-eye
- rig extrinsics
- rig hand-eye
- laserline device
- scheimpflug intrinsics

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
    fix_scheimpflug={"tilt_x": False, "tilt_y": False}
)
result = vc.run_scheimpflug_intrinsics(dataset, config)
print(result.camera.sensor)
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
