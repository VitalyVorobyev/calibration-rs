# Serialization and Runtime-Dynamic Types

The generic `Camera<S, P, D, Sm, K>` type provides compile-time composition of camera stages. However, for JSON serialization (session checkpointing, data exchange) and runtime-dynamic camera construction, calibration-rs provides an enum-based parameter system.

## CameraParams

The `CameraParams` struct holds camera parameters as serializable enums:

```rust
pub struct CameraParams {
    pub projection: ProjectionParams,
    pub distortion: DistortionParams,
    pub sensor: SensorParams,
    pub intrinsics: FxFyCxCy<Real>,   // fx, fy, cx, cy
}
```

The projection, distortion and sensor components are enums of supported variants (serialized with a `"type"` tag in snake_case); the intrinsics are the plain `{fx, fy, cx, cy}` struct:

```rust
pub enum ProjectionParams { Pinhole }

pub enum DistortionParams {
    None,
    BrownConrady5 { params: BrownConrady5<Real> },        // k1, k2, k3, p1, p2, iters
    Rational { params: RationalPolynomial<Real> },        // k1..k6, p1, p2, iters
    ThinPrism { params: ThinPrism<Real> },                // Brown-Conrady + s1..s4
    Division { lambda: Real },                            // Fitzgibbon division model
}

pub enum SensorParams {
    Identity,
    Homography { h: [[Real; 3]; 3] },
    Scheimpflug { params: ScheimpflugParams },            // tilt_x, tilt_y
}
```

The `params` payloads are flattened in JSON, so a Brown-Conrady camera serializes as `{"type": "brown_conrady5", "k1": ..., "k2": ..., ...}`.

The enums are `#[non_exhaustive]`: new models arrive in minor releases, so a `match` on one needs a wildcard arm. Code that should work for every model reads the values instead, in the order the model's kind names them:

```rust
let d = &camera.distortion;
for (name, value) in d.kind().param_names().iter().zip(d.values()) {
    println!("{name} = {value}");
}
let adjusted = d.with_values(&new_values)?;                          // same model, new coefficients
let fresh = DistortionParams::from_values(DistortionKind::Rational8, &values)?;
```

`ProjectionParams` and `SensorParams` have the same `kind()`, `values()` and `from_values()`, and `DistortionKind::ALL` (likewise for projections and sensors) lists the supported models.

## Building a Camera from Parameters

The `build()` method constructs a concrete `CameraModel` from serialized parameters. It returns an error if a `SensorParams::Homography` matrix is not invertible (reachable with hand-edited or corrupted files):

```rust
let camera = params.build()?;
```

This is used internally by the session framework to reconstruct cameras from JSON-checkpointed state.

### In Python

`vc.CameraModel` is the built model, made from a result's camera (a `vc.Camera`: intrinsics plus `projection`, `distortion` and `sensor`) or a `CameraParams` payload, with any model in each slot. Its methods are vectorized over numpy arrays and release the GIL for large ones:

```python
import numpy as np
import vision_calibration as vc

cam = vc.CameraModel(result.camera)
pixels = cam.project(points_cam)                # (N, 3) camera-frame points -> (N, 2)
origins, directions = cam.unproject(pixels)     # (N, 3) rays; pinhole directions have z = 1
ideal = cam.undistort(pixels)                   # (N, 2) in a distortion-free pinhole image, same K
```

`undistort` removes the distortion and the sensor tilt, so the output is where each pixel lands in an ideal pinhole image with the same intrinsics. A point behind the camera projects to NaN.

## Reading the Result

The calibrated planar-intrinsics parameters (`PlanarIntrinsicsParams`, `export.params`) hold the camera in their `camera` field and the per-view poses in `camera_se3_target`:

```rust
export.params.camera.intrinsics        // FxFyCxCy<f64>
export.params.camera.distortion        // DistortionParams: match on the model
export.params.build_camera()?          // CameraModel, ready to project
```

## Use in Session Framework

`CameraParams` appears throughout the pipeline:

- **Export records** store calibrated parameters as `CameraParams`
- **Session JSON** serializes intermediate and final camera parameters

This provides a stable, version-safe serialization format decoupled from the generic type system.
