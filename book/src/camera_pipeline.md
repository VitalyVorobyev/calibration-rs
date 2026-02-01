# Composable Camera Pipeline

A camera model maps 3D points in the camera frame to 2D pixel coordinates. In calibration-rs, this mapping is decomposed into four composable stages, each implemented as a trait:

$$\mathbf{p}_{\text{pixel}} = K \circ S \circ D \circ \Pi(\mathbf{d})$$

where:

- $\Pi$ — **projection**: maps a 3D direction to normalized coordinates on the image plane
- $D$ — **distortion**: warps normalized coordinates to model lens imperfections
- $S$ — **sensor**: applies a homography for tilted sensor planes (identity for standard cameras)
- $K$ — **intrinsics**: scales and translates to pixel coordinates

## The `Camera` Struct

The camera is a generic struct parameterized over the four model traits:

```rust
pub struct Camera<S, P, D, Sm, K>
where
    S: RealField,
    P: ProjectionModel<S>,
    D: DistortionModel<S>,
    Sm: SensorModel<S>,
    K: IntrinsicsModel<S>,
{
    pub proj: P,
    pub dist: D,
    pub sensor: Sm,
    pub k: K,
}
```

The scalar type `S` is generic over `RealField`, enabling the same camera to work with `f64` for evaluation and with dual numbers for automatic differentiation.

## Forward Projection

Given a 3D point $\mathbf{P}_c = [X, Y, Z]^T$ in the camera frame, projection to pixel coordinates proceeds through the four stages:

```rust
pub fn project_point_c(&self, p_c: &Vector3<S>) -> Option<Point2<S>>
```

1. **Projection**: Compute normalized coordinates $\mathbf{n}_u = \Pi(\mathbf{P}_c)$. For pinhole: $\mathbf{n}_u = [X/Z, \; Y/Z]^T$. Returns `None` if the point is behind the camera ($Z \leq 0$).

2. **Distortion**: Apply lens distortion $\mathbf{n}_d = D(\mathbf{n}_u)$. Warps normalized coordinates according to the distortion model (e.g., Brown-Conrady radial + tangential).

3. **Sensor transform**: Apply sensor model $\mathbf{s} = S(\mathbf{n}_d)$. For standard cameras this is identity; for tilted sensors it applies a homography.

4. **Intrinsics**: Map to pixels $\mathbf{p} = K(\mathbf{s})$. Applies focal lengths, principal point, and optional skew.

## Inverse (Back-Projection)

The inverse maps a pixel to a ray in the camera frame:

```rust
pub fn backproject_pixel(&self, px: &Point2<S>) -> Ray<S>
```

1. $\mathbf{s} = K^{-1}(\mathbf{p})$ — pixel to sensor coordinates
2. $\mathbf{n}_d = S^{-1}(\mathbf{s})$ — sensor to distorted normalized
3. $\mathbf{n}_u = D^{-1}(\mathbf{n}_d)$ — undistort (iterative)
4. $\mathbf{d} = \Pi^{-1}(\mathbf{n}_u)$ — normalized to 3D direction

The result is a point on the $Z = 1$ plane in the camera frame, defining a ray from the camera center.

## Common Type Aliases

For the most common configuration (pinhole + Brown-Conrady + no tilt):

```rust
type PinholeCamera = Camera<f64, Pinhole, BrownConrady5<f64>,
                            IdentitySensor, FxFyCxCySkew<f64>>;
```

The `make_pinhole_camera(k, dist)` convenience function constructs this type.

## Projecting World Points

To project a world point through a posed camera, first transform from world to camera frame using the extrinsic pose $T_{C,W} \in \text{SE}(3)$:

$$\mathbf{P}_c = T_{C,W} \cdot \mathbf{P}_w$$

Then apply the camera projection:

$$\mathbf{p} = \text{camera.project\_point\_c}(\mathbf{P}_c)$$

## Trait Composition

Each stage is an independent trait. This design allows mixing and matching:

| Projection | Distortion | Sensor | Intrinsics |
|-----------|-----------|--------|-----------|
| `Pinhole` | `NoDistortion` | `IdentitySensor` | `FxFyCxCySkew` |
| `Pinhole` | `BrownConrady5` | `IdentitySensor` | `FxFyCxCySkew` |
| `Pinhole` | `BrownConrady5` | `ScheimpflugParams` | `FxFyCxCySkew` |

The subsequent sections detail each stage.
