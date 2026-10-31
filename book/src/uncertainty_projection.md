# Projection Uncertainty

The covariance of a camera's parameters implies how precisely it places a
ray in the image. For the ray through a pixel, the pixel it projects to
moves with the parameters `θ` as `J = ∂q/∂θ`, so its covariance is the 2×2
`J Σ Jᵀ`. `projection_uncertainty` reports the major-axis standard deviation
of that ellipse, in pixels, on a grid of 16 × 12 nodes from pixel `(0, 0)` to
`(width − 1, height − 1)`, together with its value at the image centre, the
largest of the four corners and the largest on the grid. The export does not
record the image size, so the call takes it:

```rust
use vision_calibration::common::CalibratedCameras;

let export = session.export()?;
let map = export.projection_uncertainty(0, [1280, 1024])?;
println!(
    "σ: centre {:.3} px, corners up to {:.3} px",
    map.center,
    map.corner.unwrap_or(f64::NAN)
);
```

```python
p = vc.projection_uncertainty(result, (1280, 1024), camera=0)
print(f"σ: centre {p.center:.3f} px, max {p.max:.3f} px")
heatmap = p.grid()  # 12 rows of 16 values, top row first
```

`camera` is the camera index (0 for a single camera). A node no ray reaches,
as where a strong distortion folds the image over, has no value.

For the whole ellipse at pixels of your choice, `projection_covariance`
returns `J Σ Jᵀ` itself, in px², one entry per pixel (`None` where no ray
reaches it). It is Rust only:

```rust
let ellipses = export.projection_covariance(0, &[[640.0, 512.0], [10.0, 10.0]])?;
```

Reading the map:

- **Where the views were.** σ is smallest where the board covered the
  image and grows toward corners the views did not reach, where the
  distortion is extrapolated. A large corner value is a reason to add views
  with the board in that corner; [Planning the Next View](uncertainty_next_view.md)
  ranks candidate views by how much each would reduce it.
- **Camera frame.** The ray is held fixed in the camera's own frame and only
  the camera's parameters vary: intrinsics, distortion, sensor tilt. A
  pinhole camera maps every point of a ray to one pixel, so the map does not
  depend on range. It does not include the uncertainty of the camera's pose.
- **Conservative where the pose is re-estimated.** A principal-point shift
  and a small rotation of the camera move the image almost alike. A use that
  estimates the camera's pose again, such as a pose from a known target,
  absorbs that part, so the map overstates the error such uses see. A
  Scheimpflug camera shows this strongly: its sensor tilt and principal point
  trade off, so σ of `cx` and `cy`, and with them the whole map, can be
  several pixels while the corners still reproject to a fraction of one.
- **Conditional.** Parameters the final solve held fixed count as exact.
  The rig and rig hand-eye workflows refine each camera's intrinsics in
  their final solve by default, so each camera has its map; with
  `rig.refine_intrinsics` (or the rig hand-eye's
  `handeye_optimize.refine_intrinsics`) set to `false` the intrinsics are
  held, the result has no covariance to project and the call returns an
  error saying so.
- **The image size matters.** Pass the size of the images the camera was
  calibrated on. Nodes beyond the area the views covered are extrapolated
  through the distortion, and far outside the field of view σ grows without
  bound.
