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

# Typed config dictionaries are available via vision_calibration.types.
cfg: vc.RigHandeyeConfig = {
    "handeye_init": {
        "handeye_mode": "EyeInHand",
        "min_motion_angle_deg": 5.0,
    },
    "solver": {
        "max_iters": 80,
        "robust_loss": vc.robust_huber(1.0),
    },
}

# High-level workflow entry points:
# - run_planar_intrinsics
# - run_single_cam_handeye
# - run_rig_extrinsics
# - run_rig_handeye
# - run_laserline_device
```
