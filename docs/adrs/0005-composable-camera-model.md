# ADR 0005: Composable Camera Model

## Context

Calibration libraries usually hard-code one camera model. We need several
projections, distortion models and sensor-plane transforms (Scheimpflug
tilt) without a combinatorial explosion of camera types.

## Decision

A camera is a pipeline of four independent stages:

```
pixel = K(sensor(distortion(projection(dir))))
```

- **Projection** — camera-frame direction to the normalized plane
  (`Pinhole`).
- **Distortion** — warps normalized coordinates (`BrownConrady5` and the
  extended models).
- **Sensor** — a homography for sensor-plane effects (`IdentitySensor`,
  `ScheimpflugParams`).
- **K** — normalized plane to pixels (`FxFyCxCySkew`).

`Camera<S, P, D, Sm, K>` is generic over the scalar `S: RealField` and the
four stage traits; aliases such as `PinholeCamera` fix common combinations.
Code that needs a runtime-polymorphic camera uses the `f64`-only,
dyn-safe `CameraProject` trait.

## Consequences

- A new model is a new stage implementation, not an edit of existing ones.
- Every stage is generic over `RealField`, so the same code runs under
  dual-number autodiff in the optimizer. How the optimizer selects a camera
  model at runtime without multiplying factor types is ADR 0008.
