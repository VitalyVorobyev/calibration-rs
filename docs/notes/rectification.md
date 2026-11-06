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

Dense cloud (`rectification/cloud.rs`). The maps are fixed (built with the
calibration's values), so a match is the fixed raw left pixel `u_L` and the
rectified right column `x − d`; the right row is not measured. The point `X`
(rig frame) solves `F = [π_L(X) − u_L; r(π_R(X) − u_R(d))] = 0`, `r = e₁ᵀM⁻¹`,
`M = ∂raw_R/∂rectified` (fixed), three equations in three unknowns, so
`G = ∂X/∂c = −A⁻¹ ∂F/∂c` with `A = ∂F/∂X` and the camera derivatives from
`CameraJacobian`; `∂X/∂d = −A⁻¹e₃ = −(X − c_L)/d`. `Σ_i = G_i Σ_c G_iᵀ +
σ_d² g_i g_iᵀ`, `Cov(X_i, X_j) = G_i Σ_c G_jᵀ`. The `G_i` are not stored
(3 × ~30 doubles a point); `point_covariance(i)` and `covariance(&indices)`
compute them. It is the limit of `triangulate` with the left pixel exact and
the right one weighted only along `r`; against isotropic raw-pixel noise the
depth σ agrees within 0.5 % and the across-row σ differs up to 40 %
(`triangulate` splits a vertical disparity between the cameras).

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
- `crates/vision-calibration/tests/stereo_uncertainty.rs`: 60 calibrations,
  ten held-out points through `triangulate`: depth scatter/stated 0.95 with the
  calibration's share, 3.2 observations-only.
- `crates/vision-mvg/src/rectification/cloud/tests.rs`: on a three-camera rig
  measured with the pair `[1, 2]` (both extrinsics estimated, a Scheimpflug
  right camera), `G` against central differences of an independent re-solve
  (forward projections and the fixed maps only): worst relative error
  7.9e-7; the joint covariance against `triangulate` with the dense noise
  model (left σ 1e-3 px, right σ_d along `r` and 100 px across): 9.6e-8.
- `crates/vision-calibration/tests/dense_cloud_uncertainty.rs` (`#[ignore]`,
  CI uncertainty gates): 200 calibrations, ten grid pixels of a held-out
  scene at 0.6–0.8 m, the disparity simulated on the estimate's rectified
  images with σ_d = 0.2 px: depth error/stated 0.907 (2.55 mm; 0.93–0.95 on
  three other seed ranges), the disparity's share alone 2.46× too small. At
  σ_d = 0 the calibration share is 2.36 mm for both `dense_cloud` and
  `triangulate` against an error 0.88 times that: the residual
  conservatism is the calibration covariance's, not the dense model's.
- Throughput (release, one Apple-silicon core): building a 640×480 cloud
  6 ms (51 Mpts/s); `point_covariance` 14.8 µs (68k points/s, the
  dual-number `CameraJacobian` dominates); the joint covariance of 1000
  points 38 ms.
- The app's Depth workspace test `compute_disparity_recovers_board_on_committed_rig`
  (`app/src-tauri/src/disparity.rs`) runs the pipeline on the committed
  `data/stereo` rig.
- **Related**: [two-view-triangulation.md](two-view-triangulation.md).
