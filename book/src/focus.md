# Focus and Depth of Field

A lens images one plane sharply: the plane of sharp focus. On an ordinary camera it is parallel to the sensor; on a tilted (Scheimpflug) sensor it is tilted too, which is the point of the tilt. When the detector reports each point's [measured blur](detection.md#measured-blur), a calibrated planar dataset says where that plane is and how deep the sharp zone around it reaches. `analysis::fit_focus` fits it, with the calibrated camera and target poses held fixed.

## The Model

Write a target point in the camera frame as `X` and the sensor normal as `n` (the optical axis for an untilted sensor). The plane of sharp focus is `π·X = 1`, with `π` in 1/m. The point's defocus, its distance from that plane in inverse depth along the sensor normal, is

```text
d = (1 − π·X) / (n·X)        (1/m)
```

and the blur the image shows, the width `σ = √(tr Σ / 2)` of the measured point-spread function, is predicted as

```text
σ = √(σ0² + 1/12 + (κ·s·d)²) + δ
```

- `κ` (px·m) is the blur per unit defocus. For a geometric blur spot from a circular entrance pupil of diameter `D`, read as the per-axis σ of a filled disc, `κ = D·f/4` with `f` the camera's distortion-free scale in pixels (the focal length for an untilted sensor). That reading is of the spot's second moment. A detector that reports the width of the Gaussian whose edges best match the image reports a different width for a disc-shaped spot, so read `κ` and `σ0²` as the detector's; `D` follows from `κ` only as far as the detector's width follows the second moment.
- `s` is the spot's local scale relative to `f`. The entrance pupil lies in the lens plane, square to the camera's `z` axis, so a point's spot is the pupil's shadow, a disc of radius `(D/2)·|1 − π·X|/z` in the normalized coordinates `(x/z, y/z)`. `s` is the calibrated camera's local scale in those coordinates times `(n·X)/z`, over `f`, taken at each point's own projection, never from the observed pixel. It is 1 for an untilted camera without distortion. Distortion changes it by its change of magnification. A tilted sensor sees the pupil obliquely, which changes it at first order in tilt times field angle: 0.99–1.01 across a 720 × 540 image at a 5° tilt, 0.97–1.05 at 11.5° × 5.7°. The calibrated camera frame is taken for the lens's.
- `σ0²` (px²) is the optical blur at focus, and `1/12` the pixel's area integration. The measured blur contains the pixel, so the prediction adds it.
- `δ` (px) is one offset per image, the offsets summing to zero. A detector's blur covariance holds for differences within one image; every point of an image can share an offset it does not contain, and this term takes it. A level common to all images is not a parameter, and it is not harmless. When images cross the plane of sharp focus it moves `κ`, `σ0²` and the depth of field: in a simulated capture, 0.02 px added to every blur moved `κ` by 3 σ and `σ0²` by 13 σ, and the plane by less than 0.2 σ. Without a crossing it moves the plane's position along the sensor normal instead. The plane's direction across the normal does not move. So the plane is the robust output; `κ`, `σ0²` and the depth of field carry whatever common level the detector has.

The law is the geometric (thin-lens) defocus, written in object space: it holds for a thick lens with `D` its entrance pupil. The fit is weighted least squares in `σ`, each point weighted by the inverse variance of its `σ` from the detector's covariance, with the noise scale estimated from the residuals. When no point states a variance (a covariance of zeros) every point weighs the same and `stated_weights` is false; a dataset where some points state one and others do not is an error, not a quiet switch to equal weights.

## What the Views Must Show

Within one image, the defocus is affine in the pixel, so each image's blur is `σ² ≈ σ0² + (αu + βv + γ)²`: a V-shaped valley along the line where the target crosses the plane of sharp focus. The poses' different tilts and distances separate `κ` from the plane's own tilt. The plane's position along the axis needs the valley itself: far from focus, `σ ≈ |αu + βv + γ| + δ`, and the image's offset takes `γ`. So:

- at least one image must hold points on both sides of the plane: a target tilted through the sharp zone. Without one the result is `NotCrossed`, which carries the fit: the plane's direction across the sensor normal and the image line's angle `θ` are determined, but its position along the normal (the in-focus distance, the line's offset `ρ`, the depth of field) is held only by the offsets summing to zero, so a common blur level moves it;
- the targets must vary in depth and tilt. Images all at one distance, facing the camera, cannot separate `κ` from the plane's tilt (`Undetermined`);
- the blur must rise well above the in-focus blur across the field: a wide aperture, a depth of field small against the depth range of the views.

Each image needs at least three points with a blur to enter the fit.

## Running It

```rust,no_run
use vision_calibration::analysis::{FocusDiagnostic, FocusJackknife, FocusOptions, fit_focus};
use vision_calibration::planar_intrinsics::{PlanarIntrinsicsExport, PlanarIntrinsicsInput};
# fn main() -> anyhow::Result<()> {
# let input: PlanarIntrinsicsInput = unimplemented!("views whose points carry blur");
# let export: PlanarIntrinsicsExport = unimplemented!("their planar calibration");

// The depth of field for an acceptable blur of σ ≤ 1.5 px.
// The view jackknife adds a covariance that does not assume independent
// blur errors within an image.
let options = FocusOptions::new(Some(1.5), true);
match fit_focus(&input, &export.camera, &export.cam_se3_target, &options)? {
    FocusDiagnostic::Fitted(fit) => {
        let [px, py, pz] = fit.plane_inv_m;
        let c = fit.covariance;
        println!("π = ({px:.3}, {py:.3}, {pz:.3}) /m, σ(π_z) = {:.3} /m", c[2][2].sqrt());
        println!("in focus at {:.3} m on the optical axis", 1.0 / pz);
        println!("blur fit: {:.3} px RMS, noise scale {:.2}", fit.residual_rms_px, fit.noise_scale);
        if let Some(line) = fit.image_line {
            println!("focus at infinity on u·cos θ + v·sin θ = {:.0} px, θ = {:.1}°",
                line.offset_px, line.angle_rad.to_degrees());
        }
        if let Some(dof) = fit.depth_of_field {
            println!("near plane {:?} /m, far plane {:?} /m", dof.near_plane_inv_m, dof.far_plane_inv_m);
        }
        if let Some(FocusJackknife::Estimated(jackknife)) = fit.view_jackknife {
            println!("σ(π_z): stated {:.3} /m, view jackknife {:.3} /m",
                c[2][2].sqrt(), jackknife.covariance[2][2].sqrt());
        }
    }
    FocusDiagnostic::NotCrossed(fit) => {
        let line = fit.image_line.expect("a plane across the sensor has an image line");
        println!("no view crosses the plane; the focus recedes towards θ = {:.1}°",
            line.angle_rad.to_degrees());
    }
    other => println!("no focus model: {other:?}"),
}
# Ok(())
# }
```

For a rig, call it once per camera with that camera's poses (`cam_se3_rig · rig_se3_target`) and its views.

## Reading the Result

`FocusFit` holds:

- **`plane_inv_m`** `π`, **`blur_scale_px_m`** `κ` and **`in_focus_variance_px2`** `σ0²`, with their 5×5 **`covariance`** (in that order, scaled by the noise scale's square, marginal over the per-image offsets). Along a ray `r` the in-focus distance is `1 / (π·r)`; on the optical axis, `1 / π_z`.
- **`sensor_normal`** `n`, from the camera.
- **`image_line`**: the image line where the plane of sharp focus is at infinity, `u·cos θ + v·sin θ = ρ` in the camera's distortion-free projection, with the covariance of `(θ, ρ)`. On the side `u·cos θ + v·sin θ < ρ` the in-focus distance is finite and grows towards the line; beyond it, the plane lies behind the camera. `θ` is the image direction in which the focus recedes: the tilt direction of the plane. It is absent when the plane is parallel to the sensor.
- **`depth_of_field`**, for the blur limit `σ_max` you set: a point is sharp enough when `|d| ≤ ε`, `ε = √(σ_max² − σ0² − 1/12) / κ`, at the spot scale `s = 1`. The near and far limits are the planes `(π ± ε·n)·X = 1`, given with their covariances. They meet the plane of sharp focus on one line, in the plane through the projection centre parallel to the sensor, so the sharp zone is a wedge opening away from the camera: the view camera's hinge-rule wedge. The planes are exact where `s = 1`; where distortion or a tilted sensor scales the spot by `s`, the limit there is `ε/s`, so the true limits bend slightly away from the planes. Where the far plane's `p·X` is not positive along a ray, the zone reaches infinity. Absent when the limit is not above the in-focus blur `√(σ0² + 1/12)`.
- **`noise_scale`**: `√(χ²/dof)` of the weighted residuals. About 1 when the law explains the blur to the detector's stated precision; larger when the lens departs from it (field curvature, aberrations) or the stated covariance is short.
- **`views`**: per image, the blur points, whether it was fitted, its offset `δ` with σ, the RMS and largest blur residual, and whether it crosses the plane of sharp focus. An image whose residuals stand out is where the law or the detector fails.
- **`points`** used, **`unprojected_points`** (blur points left out because they do not project at their pose) and **`stated_weights`**.
- **`view_jackknife`**, when you ask for it: the delete-one-view jackknife covariance of `(π, κ, σ0²)`, and the image line and depth of field with their covariances from it.

The σ are those of the blur fit with the camera and poses held: the calibration's own uncertainty is not added. They agree with the scatter over noise draws only when the blur errors within an image are independent, as the detector's covariance states. On real captures they need not be: the scatter of the plane over the images has been measured at up to 3 times the stated σ. Treat the stated covariance as a lower bound and ask for the view jackknife, which leaves out one image at a time, refits, and takes the covariance from how the estimate moves. In a simulation with a random blur gradient per image and a common blur level that the model does not have, the stated σ was 2 to 5 times short of the scatter and the jackknife's within 7 %. The jackknife is noisy with few images and errs high when a single image carries a direction alone (the only one that crosses the plane, say).

## What It Does Not Give

The plane of sharp focus fixes the tilt of the lens relative to the sensor only through two numbers it does not contain: the distance from the projection centre to the sensor (the focal length in pixels times the pixel pitch), and the lens's pupil magnification, the ratio of its exit to its entrance pupil. So `fit_focus` reports the plane, its direction in the image and the depth of field, which need neither, and not a lens tilt. A calibration never reads the result: it is a diagnostic you ask for.

The fit takes a pinhole projection and rejects a fisheye one: the law is derived for a perspective projection. A curved field of focus shows as structure in the residuals, largest at the image corners, and a raised noise scale.

In Python, `vision_calibration.fit_focus(dataset, camera, cam_se3_target, blur_limit_px=None, view_jackknife=False)` returns the same result as a dict, its `status` `"fitted"`, `"not_crossed"` (with the same fields) or the reason there is no fit (`"no_blur"`, `"too_few_points"`, `"undetermined"`).
