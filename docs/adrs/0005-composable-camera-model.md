# ADR 0005: Composable Camera Model Pipeline

- Status: Accepted
- Date: 2026-03-07 (retroactive)

## Context

Camera calibration libraries typically hard-code a single camera model (e.g., OpenCV's pinhole + Brown-Conrady). We need to support multiple projection types, optional distortion, and optional sensor-plane transforms (Scheimpflug tilt) without combinatorial explosion.

## Decision

Model cameras as a composable pipeline of four stages, each a separate generic type:

```
pixel = K(sensor(distortion(projection(direction))))
```

- **Projection**: camera-frame 3D direction to normalized 2D coordinates (e.g., `Pinhole`)
- **Distortion**: warp normalized coordinates (e.g., `BrownConrady5`)
- **Sensor**: apply a homography to model sensor-plane effects (`IdentitySensor` or `ScheimpflugParams`)
- **K (Intrinsics)**: map to pixel coordinates (`FxFyCxCySkew`)

The `Camera<P, D, S, K>` type is generic over all four stages. Concrete type aliases (e.g., `PinholeCamera`) fix common combinations.

## Consequences

- New projection/distortion/sensor models are added by implementing a trait, not modifying existing code.
- Optimization factors must be parameterized per camera variant (trade-off: more factor types, but type-safe).
- Autodiff compatibility requires all stages to be generic over `RealField`.
