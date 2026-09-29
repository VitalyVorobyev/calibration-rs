# Serialization and Runtime-Dynamic Types

The generic `Camera<S, P, D, Sm, K>` type provides compile-time composition of camera stages. However, for JSON serialization (session checkpointing, data exchange) and runtime-dynamic camera construction, calibration-rs provides an enum-based parameter system.

## CameraParams

The `CameraParams` struct holds camera parameters as serializable enums:

```rust
pub struct CameraParams {
    pub projection: ProjectionParams,
    pub distortion: DistortionParams,
    pub sensor: SensorParams,
    pub intrinsics: IntrinsicsParams,
}
```

Each component is an enum of supported variants (serialized with a `"type"` tag in snake_case):

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

pub enum IntrinsicsParams {
    FxFyCxCySkew { params: FxFyCxCySkew<Real> },
}
```

The `params` payloads are flattened in JSON, so a Brown-Conrady camera serializes as `{"type": "brown_conrady5", "k1": ..., "k2": ..., ...}`.

## Building a Camera from Parameters

The `build()` method constructs a concrete `CameraModel` from serialized parameters. It returns an error if a `SensorParams::Homography` matrix is not invertible (reachable with hand-edited or corrupted files):

```rust
let camera = params.build()?;
```

This is used internally by the session framework to reconstruct cameras from JSON-checkpointed state.

## Convenience Accessors

The calibrated planar-intrinsics parameters (`PlanarIntrinsicsParams`, `export.params`) wrap a `CameraParams` in their `camera` field and provide typed accessors:

```rust
export.params.intrinsics()   // → FxFyCxCySkew<f64>
export.params.distortion()   // → Option<BrownConrady5<f64>>; None for non-Brown-Conrady models
export.params.build_camera()?  // → CameraModel
```

## Use in Session Framework

`CameraParams` appears throughout the pipeline:

- **Export records** store calibrated parameters as `CameraParams`
- **Session JSON** serializes intermediate and final camera parameters

This provides a stable, version-safe serialization format decoupled from the generic type system.
