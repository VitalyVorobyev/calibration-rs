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

The calibration workflows initialize a camera from homographies of the
target, which assumes a pinhole projection; they reject an equidistant
camera with a configuration error. `KannalaBrandt4` distortion on a pinhole
camera calibrates in every workflow.
