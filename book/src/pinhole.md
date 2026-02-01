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
pub trait ProjectionModel<S: RealField> {
    fn project_dir(&self, dir_c: &Vector3<S>) -> Option<Point2<S>>;
    fn unproject_dir(&self, n: &Point2<S>) -> Vector3<S>;
}
```

The `Pinhole` struct implements this trait:

```rust
pub struct Pinhole;

impl<S: RealField> ProjectionModel<S> for Pinhole {
    fn project_dir(&self, dir_c: &Vector3<S>) -> Option<Point2<S>> {
        if dir_c.z > S::zero() {
            Some(Point2::new(
                dir_c.x.clone() / dir_c.z.clone(),
                dir_c.y.clone() / dir_c.z.clone(),
            ))
        } else {
            None
        }
    }

    fn unproject_dir(&self, n: &Point2<S>) -> Vector3<S> {
        Vector3::new(n.x.clone(), n.y.clone(), S::one())
    }
}
```

Note the use of `.clone()` — this is required for compatibility with dual numbers used in automatic differentiation (see [Autodiff and Generic Residual Functions](autodiff.md)).

## Limitations

The pinhole model assumes:

- **No lens distortion** — addressed by the [distortion stage](distortion.md)
- **No sensor tilt** — addressed by the [sensor stage](sensor.md)
- **Central projection** — all rays pass through a single point

For cameras with significant distortion (wide-angle, fisheye), the distortion model corrects for lens effects after pinhole projection to normalized coordinates.
