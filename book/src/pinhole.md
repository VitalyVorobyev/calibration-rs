# Pinhole Projection

The pinhole model is the simplest and most widely used camera projection. It models a camera as an ideal point through which all light rays pass — no lens, no aperture, just a geometric center of projection.

## Mathematical Definition

Given a 3D point $\mathbf{P}_c = [X, Y, Z]^T$ in the camera frame, the pinhole projection maps it to **normalized image coordinates**:

$$\mathbf{n} = \Pi(\mathbf{P}_c) = \begin{bmatrix} X / Z \\ Y / Z \end{bmatrix}$$

This is perspective division: the point is projected onto the $Z = 1$ plane. The projection is defined only for points in front of the camera ($Z > 0$).

The inverse (back-projection) maps normalized coordinates to a 3D direction:

$$\Pi^{-1}(\mathbf{n}) = \begin{bmatrix} n_x \\ n_y \\ 1 \end{bmatrix}$$

This defines a ray from the camera center through the image point.

## Camera Frame Convention

calibration-rs uses a right-handed camera frame:

- **$X$**: right
- **$Y$**: down
- **$Z$**: forward (optical axis)

This matches the convention used by OpenCV and most computer vision libraries. A point with positive $Z$ is in front of the camera; negative $Z$ is behind it.

## The `ProjectionModel` Trait

```rust
pub trait ProjectionModel<T: RealField> {
    fn in_domain(&self, p_c: &Vector3<T>) -> bool;
    fn project(&self, p_c: &Vector3<T>) -> Point2<T>;
    fn unproject(&self, n: &Point2<T>) -> Ray<T>;
    fn is_central(&self) -> bool;
}
```

The projection takes the camera-frame point, and back-projection returns a
`Ray` (origin and direction). For `Pinhole`, `in_domain` is `z > 0`,
`project` is `(x / z, y / z)` and `unproject` returns the direction
`(n_x, n_y, 1)` from the camera centre. `project` is total: callers that need
validity check `in_domain` first.

Note the use of `.clone()` — this is required for compatibility with dual numbers used in automatic differentiation (see [Autodiff and Generic Residual Functions](autodiff.md)).

## Limitations

The pinhole model assumes:

- **No lens distortion** — addressed by the [distortion stage](distortion.md)
- **No sensor tilt** — addressed by the [sensor stage](sensor.md)
- **Central projection** — all rays pass through a single point

For wide-angle lenses the distortion model corrects for lens effects after pinhole projection to normalized coordinates. A fisheye lens that sees near or beyond 90° from the axis needs a different projection: see [Fisheye Projection](fisheye.md).
