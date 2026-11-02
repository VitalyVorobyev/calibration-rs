# Fisheye Calibration: OpenCV's Test Set

This walkthrough calibrates the fisheye stereo pair OpenCV uses to test its
own `fisheye` module, and sets the result next to OpenCV's. The camera is the
[Kannala-Brandt fisheye](fisheye.md): an equidistant projection with
`KannalaBrandt4` distortion, the model `cv::fisheye` calibrates.

## The data

The set is `calib-3_stereo_from_JY` of OpenCV's test data repository
(`opencv_extra`, `testdata/cv/cameracalibration/fisheye/`): 34 synchronized
image pairs, 1280×800, of an 8×6 chessboard with 24.4 mm squares. It ships
the chessboard corners with the images, in `cv::FileStorage` files
(`left.xml`, `right.xml` and `object.xml`), and OpenCV's tests calibrate
those corners rather than detecting them again. The publisher states no
licence for the data.

The regression suite freezes the same corners, so calibration-rs and OpenCV
calibrate exactly the same points. Its three entries are
`opencv_fisheye_left`, `opencv_fisheye_right` (`planar_intrinsics`) and
`opencv_fisheye_stereo` (`rig_extrinsics`); the observations are in
`regression/observations/<id>.json`. `suite fetch` downloads the corner
files from a pinned commit of `opencv_extra` and checks them against their
SHA-256 (see [Regression Suite](regression_suite.md)).

## Rust

The observation file's `input` is the problem's input, so it deserializes
straight into a `PlanarDataset`. The only setting that differs from a pinhole
calibration is the camera model:

```rust
use vision_calibration::core::{
    CameraModelSpec, DistortionKind, PlanarDataset, ProjectionKind, SensorKind,
};
use vision_calibration::planar_intrinsics::{
    PlanarIntrinsicsConfig, PlanarIntrinsicsProblem, step_init, step_optimize,
};
use vision_calibration::session::CalibrationSession;

let text = std::fs::read_to_string("regression/observations/opencv_fisheye_left.json")?;
let mut file: serde_json::Value = serde_json::from_str(&text)?;
let input: PlanarDataset = serde_json::from_value(file["input"].take())?;

let mut config = PlanarIntrinsicsConfig::default();
config.camera.model = CameraModelSpec::new(
    ProjectionKind::Equidistant,
    DistortionKind::KannalaBrandt4,
    SensorKind::Identity,
);
let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
session.set_config(config)?;
session.set_input(input)?;

let init = step_init(&mut session, None)?; // bearing-based fisheye initializer
step_optimize(&mut session, None)?;
let export = session.export()?;
println!("{:?}", export.params.camera.intrinsics);
println!("{:?}", export.params.camera.distortion.values()); // k1..k4
```

`step_init` runs the [bearing-based initializer](fisheye_init.md), which
needs no focal-length guess or image size: here it starts at
`fx = fy = 556.65`, 0.3 % from the final value. The stereo pair is a
`rig_extrinsics` calibration with the same `config.camera.model`; the rig
refines both cameras' intrinsics jointly with the extrinsics, as OpenCV's
`fisheye::stereoCalibrate` does. The complete program, left camera and
stereo pair, is an example:

```bash
cargo run -p vision-calibration --example fisheye_opencv
```

## Python

```python
import json

import vision_calibration as vc

with open("regression/observations/opencv_fisheye_left.json") as f:
    views = json.load(f)["input"]["views"]
dataset = vc.PlanarIntrinsicsInput(
    views=[
        vc.PlanarView(
            observation=vc.Observation(
                points_3d=v["obs"]["points_3d"], points_2d=v["obs"]["points_2d"]
            )
        )
        for v in views
    ]
)
fisheye = vc.CameraModelSpec(projection="equidistant", distortion="kannala_brandt4")
config = vc.PlanarIntrinsicsConfig(camera=vc.CameraConfig(model=fisheye))
result = vc.run_planar_intrinsics(dataset, config)

k, d = result.camera.intrinsics, result.camera.distortion
print(f"fx={k.fx:.3f} fy={k.fy:.3f} cx={k.cx:.3f} cy={k.cy:.3f}")
print(f"k1..k4 = {d.k1:+.6f} {d.k2:+.6f} {d.k3:+.6f} {d.k4:+.6f}")
print(f"mean reprojection error {result.mean_reproj_error:.4f} px")
```

The stereo pair takes a `vc.RigExtrinsicsInput` of `vc.RigExtrinsicsView`s
(one `vc.Observation` per camera) and a
`vc.RigExtrinsicsConfig(camera=vc.CameraConfig(model=fisheye))`, run with
`vc.run_rig_extrinsics`.

## Against OpenCV

OpenCV's `fisheyeTest.Calibration` calibrates the left camera from the same
corners with `CALIB_RECOMPUTE_EXTRINSIC | CALIB_CHECK_COND |
CALIB_FIX_SKEW` and checks the result against stored values. The suite's
OpenCV harness runs `cv2.fisheye.calibrate` with the same flags:

| Left camera | fx | fy | cx | cy | k1 | k2 | k3 | k4 | Mean reprojection |
|---|---|---|---|---|---|---|---|---|---|
| calibration-rs | 558.479 | 560.507 | 620.458 | 381.938 | −0.00150 | −0.00317 | 0.00590 | −0.00367 | 0.2227 px |
| OpenCV 4.12 | 558.478 | 560.507 | 620.459 | 381.939 | −0.00146 | −0.00330 | 0.00606 | −0.00374 | 0.2227 px |
| OpenCV's test values | 558.478 | 560.507 | 620.459 | 381.939 | −0.00146 | −0.00330 | 0.00606 | −0.00374 | — |
| calibration-rs σ | 1.19 | 1.21 | 0.82 | 0.75 | 0.0047 | 0.016 | 0.020 | 0.0084 | |

The two libraries agree to within 0.01 of our standard deviation on every
parameter, and OpenCV's result is the one its own test expects. The
distortion coefficients differ in the third digit, but they are strongly
correlated (`k2`, `k3` and `k4` each have a σ larger than its value), and
both fits reach the same mean reprojection error.

The stereo pair is compared with `fisheye::stereoCalibrate`, which OpenCV's
`fisheyeTest.stereoCalibrate` runs with the same flags. Both refine the two
cameras' intrinsics with the rig and land on the same solution: the
translation from the left camera to the right is
`T = (−0.099403, 0.002708, 0.001293)` m (a 99.45 mm baseline) in both, the
focal lengths and principal points agree to 0.001 px, and the distortion
coefficients within 0.01σ.

`calib-bench suite compare` prints these comparisons for every entry; the
table is in [Regression Suite](regression_suite.md#opencv).
