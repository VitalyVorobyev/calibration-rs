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

Each component is an enum of supported variants:

```rust
pub enum ProjectionParams { Pinhole }

pub enum DistortionParams {
    None,
    BrownConrady5 { k1, k2, k3, p1, p2 },
}

pub enum SensorParams {
    Identity,
    Scheimpflug { tilt_x, tilt_y },
}

pub enum IntrinsicsParams {
    FxFyCxCySkew { fx, fy, cx, cy, skew },
}
```

## Building a Camera from Parameters

The `build()` method constructs a concrete `Camera` from serialized parameters:

```rust
let params = CameraParams { /* ... */ };
let camera = params.build();
```

This is used internally by the session framework to reconstruct cameras from JSON-checkpointed state.

## Convenience Accessors

`CameraParams` provides typed accessors:

```rust
params.intrinsics()   // → &FxFyCxCySkew<f64>
params.distortion()   // → &BrownConrady5<f64> or zero distortion
```

## Use in Session Framework

`CameraParams` appears throughout the pipeline:

- **Export records** store calibrated parameters as `CameraParams`
- **Session JSON** serializes intermediate and final camera parameters
- **Config types** reference `CameraParams` for initial guesses

This provides a stable, version-safe serialization format decoupled from the generic type system.
