# vision-calibration-py

Python bindings for `calibration-rs`.

This crate exposes high-level calibration workflows from `vision-calibration`:

- planar intrinsics
- single-camera hand-eye
- rig extrinsics
- rig hand-eye
- laserline device

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
- Low-level serde payload schemas remain available in
  `vision_calibration.types` for advanced interop.
