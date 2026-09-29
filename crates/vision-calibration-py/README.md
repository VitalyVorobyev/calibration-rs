# vision-calibration-py

Python bindings for `calibration-rs`.

This crate exposes the high-level calibration workflows from `vision-calibration`
as typed Python runners:

| Runner | Workflow |
|---|---|
| `run_planar_intrinsics` | Planar intrinsics |
| `run_scheimpflug_intrinsics` | Planar intrinsics with Scheimpflug tilt |
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
