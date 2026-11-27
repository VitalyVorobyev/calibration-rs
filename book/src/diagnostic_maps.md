# Distortion and Residual Maps

Two dense maps show a calibrated camera over its image: the **distortion
map**, what the lens and the sensor model do to each ray, and the **residual
map**, what the model leaves unexplained in the data. Both sample an
`ImageGrid` (the centres of `stride × stride` pixel blocks, row by row from
the top-left, pixel centres at integer coordinates) and return planes of
`f64`, one value per node, `NaN` where the map has none. They are computed on
demand from a result; no export stores them.

Both use one sign: **actual − reference, in pixels**. A distortion layer is
where the camera puts a ray relative to the simpler model without that
stage; a reprojection residual is `observed − projected`. A distortion the
model lacks therefore shows in the residual map with the sign and size it
would have in a distortion map. `TargetFeatureResidual::residual_px` returns
that vector for one record.

## The distortion map

For a camera `q = K(S(D(P(ray))))` (projection, distortion, sensor,
intrinsics) and the node `u` a ray lands on, with `n = P(ray)`:

| Layer | Value | Reads as |
|---|---|---|
| `distortion` | `K(D(n)) − K(n)` | the lens, as the camera would show it without a sensor tilt |
| `sensor` | `K(S(D(n))) − K(D(n))` | what a tilted (Scheimpflug) sensor adds; zero for an untilted one |
| `total` | `distortion + sensor` | everything; `u − total` is where the plain pinhole `K∘P` puts the ray |

Barrel distortion points inward at the corners. The reference keeps the
projection: for a fisheye (equidistant projection with Kannala-Brandt
distortion, [Kannala & Brandt (2006)](references.md#kannala-brandt-2006)) it
is the `r = f·θ` image, defined beyond 90°, and the distortion layer is the
polynomial's departure from it, `f·θ·(k1θ² + k2θ⁴ + k3θ⁶ + k4θ⁸)` radially.

```rust
use vision_calibration::analysis::{ImageGrid, distortion_map};
use vision_calibration::core::{
    BrownConrady5, CameraParams, DistortionParams, FxFyCxCy, ProjectionParams, SensorParams,
};

fn main() -> anyhow::Result<()> {
    let camera = CameraParams::new(
        ProjectionParams::Pinhole,
        DistortionParams::BrownConrady5 {
            params: BrownConrady5 { k1: -0.12, k2: 0.03, ..BrownConrady5::default() },
        },
        SensorParams::Identity,
        FxFyCxCy { fx: 900.0, fy: 900.0, cx: 639.5, cy: 479.5 },
    );
    let grid = ImageGrid::with_columns([1280, 960], 256)?;
    let map = distortion_map(&camera, &grid)?;
    println!(
        "{} of {} nodes; up to {:.1} px of distortion",
        map.valid_nodes,
        grid.len(),
        map.total.max_magnitude().unwrap_or(0.0)
    );
    Ok(())
}
```

A node has no value where no ray of the lens reaches it: past the fold of a
strong radial polynomial (where the image would fold over), outside a fisheye
polynomial's invertible range, or where the sensor homography sends it to
infinity. The model has no image circle, so a polynomial that stays monotone
past the lens's real field of view extrapolates there; values far from the
observed features are extrapolations of the fit.

**Compare projections, not layers.** The data constrain the focal length and
the distortion coefficients jointly (`fx` trades against `k1`), so two
calibrations that project alike can show different distortion layers. The
[projection difference](#comparing-two-calibrations) compares what they do.
How precisely a calibration places a ray is the
[projection uncertainty](uncertainty_projection.md), which can be evaluated
at the same nodes (`ImageGrid::nodes`).

## Comparing two calibrations

`projection_difference` compares two calibrations of one camera, of the same
model or of different ones, through where they put the same ray. At each node
`u` of the first camera's image:

```text
d(u) = a⁻¹(u)                              the ray a sees through u
Δ(u) = b(R·d(u)) − u                       pixels: b − a at the same ray
R    = argmin over rotations of Σ |b(R·d(u)) − u|²   over the fitted nodes
```

The rotation `R` is the gauge between a camera's intrinsics and its
orientation. A calibration's poses absorb any rotation of its camera frame,
and a shifted principal point is largely a small rotation, so two equally good
calibrations can differ by a near-uniform shift that no use of the camera
with re-estimated poses would see. With the rotation fitted, `Δ` is the part
of the difference no re-orientation removes: the difference in the lens. The
cameras are central, so rays are compared by direction and no translation
enters. The fit minimizes the pixel difference itself, so `rms_fit_px` is the
smallest RMS any rotation reaches, the image-space mapping error of
[Hagemann et al. (2022)](references.md#hagemann-2022). `Compensation::None`
compares the camera frames as calibrated, for a frame used as it is (one
calibration's intrinsics with another's extrinsics).

Fit the rotation where the data constrained the cameras:
`FitRegion::Covered` takes the observed features' pixels and fits on the grid
blocks they fall in, each block once. Over the whole image (the default) the
corners, far from any observation and extrapolated by both models, pull the
rotation. Values outside the fit region are reported (`max_valid_px`) but are
extrapolations.

```rust
use vision_calibration::analysis::{
    Compensation, FitRegion, ImageGrid, ProjectionDifferenceOptions, projection_difference,
};
use vision_calibration::core::{
    BrownConrady5, CameraParams, DistortionParams, FxFyCxCy, ProjectionParams, SensorParams,
};

fn main() -> anyhow::Result<()> {
    let camera = |fx: f64, cx: f64, k1: f64| {
        CameraParams::new(
            ProjectionParams::Pinhole,
            DistortionParams::BrownConrady5 {
                params: BrownConrady5 { k1, k2: 0.03, ..BrownConrady5::default() },
            },
            SensorParams::Identity,
            FxFyCxCy { fx, fy: fx, cx, cy: 479.5 },
        )
    };
    let (a, b) = (camera(900.0, 639.5, -0.12), camera(901.0, 641.0, -0.125));
    // The pixels of the features the calibration observed.
    let observed: Vec<[f64; 2]> = (0..20)
        .flat_map(|j| (0..20).map(move |i| [240.0 + 40.0 * i as f64, 100.0 + 38.0 * j as f64]))
        .collect();
    let grid = ImageGrid::with_columns([1280, 960], 64)?;
    let options =
        ProjectionDifferenceOptions::new(Compensation::Rotation, FitRegion::Covered(observed));
    let diff = projection_difference(&a, &b, &grid, &options)?;
    println!(
        "{:.3} px RMS over {} observed blocks (up to {:.3} px; {:.2} px over the image), \
         rotation {:.4}°",
        diff.rms_fit_px,
        diff.fit_nodes,
        diff.max_fit_px,
        diff.max_valid_px,
        diff.rotation_angle().to_degrees()
    );
    Ok(())
}
```

The grid is the first camera's image, the plane is indexed by its pixels and
its sign is `b − a`. A node is `NaN` where the first camera has no ray (the
distortion map's rules) or the second cannot project the rotated ray: outside
its projection's domain (a fisheye's ray beyond 90° for a pinhole), past its
distortion's fold, or behind its tilted sensor. Those nodes take no part in
the fit. A pixel the second camera puts off the image is a value, not `NaN`.
With cameras of different fields of view, put the one whose rays should be
compared first.

**Reading it.**

- On two calibrations of one capture with models that both fit, the
  difference is small: in a synthetic test with 0.1 px of noise, a
  Brown-Conrady and a rational fit differ by 0.019 px RMS over the observed
  region, half the Brown-Conrady fit's error against the true lens.
- A term one model lacks shows at its full size. The reprojection residual of
  the lacking model understates it, because each view's pose absorbs part of
  the field; in a synthetic test of a thin-prism lens fitted without the
  prism terms, the projection difference was 3.4 times the residual's RMS.
- The difference carries no σ. A calibration's projection uncertainty is not
  its scale: it includes the frame rotation that the compensation removes,
  and in the same test it matched the uncompensated error, about seven times
  the compensated one.

In Python the planes are numpy arrays:

```python
records = run_a.export["diagnostics"]["per_feature_residuals"]["target"]
observed = [r["projected_px"] for r in records if r["projected_px"] is not None]
d = vc.projection_difference(
    run_a.export["camera"], run_b.export["camera"], (1280, 960), fit_points=observed
)
print(d.rms_fit_px, "px RMS;", np.degrees(d.rotation_angle_rad), "deg of rotation")
```

## The residual map

`residual_map` estimates, at each node, the mean of `observed − projected`
over the views: the part of the misfit that stays fixed in the image while
the target moves. Each record sits at its **projection**, not its
observation, so the noise being averaged does not decide where it is
counted.

The estimate is a Gaussian-kernel **local-linear** fit
([Fan & Gijbels (1996)](references.md#fan-gijbels-1996)): the intercept of a
weighted plane through the residuals near the node. A plain kernel or binned
average is biased at the edge of the coverage by about the kernel width
times the field's slope, and the edges are where a distortion misfit is
largest; the local-linear fit removes that first-order term. On a cubic
radial field of 0.3 px at the corners, with a 100 px kernel, the corner
error is about 0.008 px for the local-linear fit and 0.07 px for the kernel
average.

```rust,no_run
use vision_calibration::analysis::{ImageGrid, ResidualMapOptions, residual_map};
# use vision_calibration::planar_intrinsics::PlanarIntrinsicsProblem;
# use vision_calibration::session::CalibrationSession;
# fn main() -> anyhow::Result<()> {
# let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();

let export = session.export()?;
let grid = ImageGrid::with_columns([1280, 960], 64)?;
let map = residual_map(
    &export.diagnostics.per_feature_residuals.target,
    0,
    &grid,
    &ResidualMapOptions::default(),
)?;
let significant = map.significance.iter().filter(|&&chi| chi > 3.03).count();
println!("kernel σ {:.0} px; {significant} significant nodes", map.bandwidth_px);
# Ok(())
# }
```

Each node also carries the local RMS error (`rms`), the standard error of the
mean (`std_error`), a significance (`significance`) and the support behind it
(`effective_samples`, `effective_views`), so a viewer can draw where the map
is supported.

**Support.** A node has a mean when at least 8 effective samples support it
(`min_effective_samples`) and its neighbours are not all on one line. Past
the edge of the coverage the fit extrapolates; with a wide kernel and many
features it can still reach about one kernel width beyond the last feature.

**Standard error and significance.** The residuals of one view share its
pose error, so views, not features, are the independent unit. The standard
error is clustered by view
([Cameron & Miller (2015)](references.md#cameron-miller-2015)) and measures
the scatter itself, so it does not depend on a stated feature noise. It needs
5 effective views (`min_effective_views`); a map of one view has a mean but
no significance. Clustering matters most where many records of every view
support a node, and least where few samples do: in a synthetic capture of
20 views with a pose error per view, a per-feature standard error
understates the scatter 2.6× at the image centre but only 1.1× at a corner.

`significance` is a pointwise test of a zero mean, reported as the `χ` of a
two-degree-of-freedom test, so `χ > 3.03` means `p < 0.01` whatever the
number of views. The covariance is itself estimated from the views, so
`T² = μᵀ V⁻¹ μ` follows Hotelling's distribution with `ν = G − 1` degrees of
freedom (`G` the effective number of views):
`p = (1 + T²/ν)^(−(ν−1)/2)` and `χ = √(−2 ln p)`, which tends to `√T²` with
many views. A raw `√T²` would pass 3.03 at about 3 % of the nodes of a
no-field map with 20 views and over 10 % with 6. Nodes within about two
kernel widths are strongly correlated: count significant regions, not nodes.

Both the standard error and the significance are **conditional on the
fitted camera and poses**: they describe how the mean scatters over the
views of this fit. Over repeated captures, each calibrated anew, the map
scatters less than that, about a quarter of the stated error in a synthetic
test, because the camera parameters absorb part of each capture's
image-fixed noise; the error is conservative there. What it does not cover
is a systematic shift of the fit with the noise: in the same test the
average map of noisy captures sat up to about five standard errors from the
map of a noise-free calibration where the field was weak. A weak field that
is only just significant deserves that caution.

**Bandwidth.** `Bandwidth::Auto` (the default) chooses the kernel's σ by
cross-validation over views on a ladder of 3, 4.5, 7, 10, 15 and 22 % of the
image diagonal, holding out whole views (holding out single features would
fit each view's pose error as a field), and takes the widest kernel whose
held-out error is within one standard error of the best. With no field the
smoothest map wins. With fewer than four views it uses 10 % of the diagonal.
`Bandwidth::Pixels(h)` fixes it.

**Outliers.** Records farther than 8 robust σ from zero (`outlier_factor`,
σ from the median error) are left out and listed by index, so one wrong
detection does not paint a region; a genuine field of a few σ at the corners
stays in.

In Python both maps return numpy planes shaped `(rows, columns)`:

```python
import numpy as np
import vision_calibration as vc

dmap = vc.distortion_map(result.export["camera"], (1280, 960))
rmap = vc.residual_map(result, (1280, 960), columns=64)
print(np.nanmax(dmap.total.magnitude()), "px of distortion at most")
print((rmap.significance > 3.03).sum(), "significant nodes")
```

## Reading the residual map

- **A significant field is not by itself a reason to add a model term.** It
  is the average over this capture's depths, defocus and view geometry, and
  much of such a field can be depth- or defocus-coupled rather than fixed to
  the camera. A term earns its place on held-out views
  ([Choosing a Distortion Model](distortion_selection.md)).
- **A flat map does not prove the model.** At the solution the residuals are
  orthogonal to every direction the free parameters (camera and poses) can
  move them: a missing term the parameters can partly mimic is absorbed into
  biased parameters and does not show.
- **It is not a correction.** Subtracting the map from the observations
  would apply a term fitted to the same data, at the observed pixels.
- **Chain-level residuals** of a hand-eye or rig result come through the
  robot or rig chain and carry its error as a smooth per-view pattern. For a
  diagnosis of the camera model, map the residuals of a fit with a free pose
  per view.
- Views are taken as independent; repeated poses or a drifting session are
  not.

The [calibration report](report.md) draws one residual map per camera.
