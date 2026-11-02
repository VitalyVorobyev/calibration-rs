# Fisheye Projection

A pinhole projection divides by depth, so it cannot represent a ray at or
beyond 90° from the optical axis, and it stretches the image toward that
angle. A fisheye lens instead maps the **incidence angle** to the image
radius.

## Equidistant projection

For a camera-frame point $\mathbf{P}_c = [X, Y, Z]^T$, with
$\rho = \sqrt{X^2 + Y^2}$ and incidence angle $\theta = \operatorname{atan2}(\rho, Z)$:

$$\mathbf{n} = \theta \begin{bmatrix} X / \rho \\ Y / \rho \end{bmatrix}$$

The normalized point lies in the direction of the point's azimuth, at a
radius equal to $\theta$ in radians, so `fx`, `fy` are pixels per radian.
Every direction but straight back ($\theta < \pi$) projects; points with
$Z \le 0$ are visible. Back-projection returns the unit direction
$(\sin\theta\, n_x / \theta,\ \sin\theta\, n_y / \theta,\ \cos\theta)$.

## Kannala-Brandt distortion

`KannalaBrandt4` is an odd radial polynomial in the radius $r$ of the
normalized point:

$$r_d = r\,(1 + k_1 r^2 + k_2 r^4 + k_3 r^6 + k_4 r^8)$$

On equidistant coordinates $r = \theta$, and the camera is the
Kannala-Brandt fisheye model, the one OpenCV's `fisheye` module calibrates:
`ProjectionKind::Equidistant` with `DistortionKind::KannalaBrandt4`. On
pinhole coordinates the same polynomial is a radial-only distortion with a
fourth term. Undistortion inverts the polynomial by Newton's method, which is
unique where $r_d(r)$ increases: across the field a calibration covers.

```rust
use vision_calibration::core::{
    CameraModelSpec, CameraParams, DistortionKind, DistortionParams, FxFyCxCy, ProjectionKind,
    ProjectionParams, SensorKind, SensorParams,
};

let spec = CameraModelSpec::new(
    ProjectionKind::Equidistant,
    DistortionKind::KannalaBrandt4,
    SensorKind::Identity,
);
let camera = CameraParams {
    projection: ProjectionParams::Equidistant,
    distortion: DistortionParams::from_values(spec.distortion, &[-0.02, 0.002, 0.0, 0.0])?,
    sensor: SensorParams::Identity,
    intrinsics: FxFyCxCy { fx: 300.0, fy: 300.0, cx: 640.0, cy: 400.0 },
};
let model = camera.build()?;
```

In Python the same camera is
`vc.Camera(intrinsics=..., distortion=vc.KannalaBrandt4Distortion(...),
projection=vc.EquidistantProjection())`, and `vc.CameraModel` projects and
back-projects it.

## In the workflows

Every workflow that calibrates a camera accepts an equidistant one: set
`camera.model.projection` to `Equidistant` (a rig can do so per camera).
Its linear stage is the
[bearing-based fisheye initializer](fisheye_init.md) in place of Zhang's
method, so no focal length or image size is needed; the initializer fits
`k1`, `k2` unless the camera holds them, and the refinement frees the
rest.

```rust
use vision_calibration::core::{CameraModelSpec, DistortionKind, ProjectionKind, SensorKind};
use vision_calibration::planar_intrinsics::PlanarIntrinsicsConfig;

let mut config = PlanarIntrinsicsConfig::default();
config.camera.model = CameraModelSpec::new(
    ProjectionKind::Equidistant,
    DistortionKind::KannalaBrandt4,
    SensorKind::Identity,
);
```

Some combinations have no initializer and are configuration errors:

- an equidistant camera with a Scheimpflug sensor;
- the line-distance laser residual (`LineDistNormalized`), which needs the
  laser line's image to be straight: a fisheye laser workflow uses
  `PointToPlane`.

Seeds are values in the camera's own model: a fisheye camera takes seeded
intrinsics and a `KannalaBrandt4` distortion seed, and poses then come from
the bearings of the pixels back-projected through the seeded camera.

