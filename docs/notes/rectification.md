# Stereo rectification — short note

Family: stereo from a calibration
(`vision-mvg::rectification::rectify(rig, [left, right], opts) ->
StereoRectification`), re-exported as `vision_calibration::mvg::rectification`.
A *short* pack: the tests below pin the family end-to-end and the module doc
carries the geometry. Design: ADR 0028.

## Model

Two virtual pinholes share `K_rect` and an orientation `R₀` whose x axis is the
baseline (centre of the right camera in the left frame), y ⟂ x and ⟂ the left
optical axis, z = x × y; `R₁ = R₀ · R_left_right`. Maps go through each
camera's own model, so no model-specific case exists:

```
rectified = K_rect · normalize(R_side · bearing(raw))      (z > 0)
raw       = camera_side.project(R_sideᵀ · K_rect⁻¹ [u, v, 1])
```

Rectified geometry: `d = x_l − x_r = fx·B/Z`; `Q` is OpenCV's; a disparity
error moves the point along its ray, `δP = −P δd/d`, so
`Σ_d = (σ_d/d)² P Pᵀ` (rank 1, depth σ = `Z²σ_d/(fx B)`). The calibration's
share goes through `StereoRectification::dense_cloud` (a whole map, closed
form, on demand) or `StereoRectification::triangulate` (selected matches, raw
pixels, consider parameters), not through `Σ_d`.

The dense cloud's measurement model, its covariance and the evidence for
both are in [mvg-measurements.md](mvg-measurements.md).

## Gauge

`K_rect` (`RectifyOptions::k_rect`, default the mean of the two cameras'
`fx, fy, cx, cy` times `scale`) fixes the free intrinsic scale of the pair; row
alignment is invariant to it as long as both cameras share it. Points of
`point`/`disparity_covariance` are in the left camera's frame; `dense_cloud`
and `triangulate` return the rig frame.

## Degeneracies and guards

- **Zero baseline**: `Degenerate`. Equal/out-of-range cameras, a bad `scale` or
  `k_rect`: `InvalidInput`, before any work.
- **Baseline along the optical axis**: the y-axis seed switches to the left y
  axis; near-parallel baselines degrade the rectified field of view (a
  property of the construction).
- `triangulate` and `dense_cloud` check the rig (camera count and the two
  poses) against the one rectified. `dense_cloud`: a rig without a covariance
  gives points without one (`CalibrationWithoutCovariance`, never a silent
  disparity-only σ); a partly listed extrinsic is `InvalidInput`; pixels with
  a disparity that is not finite and positive are left out.

## Evidence

- `crates/vision-mvg/tests/stereo.rs`: for a Brown–Conrady pinhole, a KB4
  fisheye and a Scheimpflug rig: row error ≤ 2.3e-13 px, raw round trip
  ≤ 1.7e-13 px, `point`/`Q` recover the 3D point to 1e-14 relative; the depth
  scatter over `Z²σ_d/(fx B)` is 0.97–1.06 over 2000 draws at 0.7/1.5/3 m.
- The covariances of `triangulate` and `dense_cloud` (unit and Monte-Carlo
  gates): [mvg-measurements.md](mvg-measurements.md).
- Throughput (release, one Apple-silicon core): building a 640×480 cloud
  6 ms (51 Mpts/s); `point_covariance` 14.8 µs (68k points/s, the
  dual-number `CameraJacobian` dominates); the joint covariance of 1000
  points 38 ms.
- The app's Depth workspace test `compute_disparity_recovers_board_on_committed_rig`
  (`app/src-tauri/src/disparity.rs`) runs the pipeline on the committed
  `data/stereo` rig.
- **Related**: [two-view-triangulation.md](two-view-triangulation.md),
  [mvg-measurements.md](mvg-measurements.md).
