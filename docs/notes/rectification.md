# Stereo rectification — short note

Family: Scheimpflug-aware stereo rectification
(`vision-mvg::rectification::rectify_stereo_pair(left, right, cam1_se3_cam0,
opts) -> StereoRectification`), re-exported as
`vision_calibration::mvg::rectification`. A *short* pack: the real-data gate
below pins the family end-to-end, and the module doc carries the derivation.

## Model

For a calibrated pair, find homographies `H_left`, `H_right` such that
corresponding points land on the same image row — the precondition for 1-D
disparity search. A pixel in this camera model (ADR 0005) is
`K · H_tilt · x_n`, so the Scheimpflug tilt is a projective factor on the
normalized plane: pre-multiplying each camera's unprojection by `H_tilt⁻¹`
collapses it to an equivalent frontal pinhole, after which the Fusiello /
Bouguet construction applies:

```
H_left  = K_rect · R_rect        · H_tilt0⁻¹ · K0⁻¹
H_right = K_rect · (R_rect · Rᵀ) · H_tilt1⁻¹ · K1⁻¹
```

with `R = R_C1_C0` and `R_rect` the common orientation whose x-axis is the
baseline. Zero tilt reduces exactly to pinhole rectification. Inputs are
**undistorted** pixels — distortion is removed separately, mirroring
OpenCV's `initUndistortRectifyMap` split.

## Degeneracies and guards

- **Zero baseline** (`‖t‖ < 1e-12`): no epipolar geometry — rejected.
- **Degenerate tilt** (`|tilt| ≥ MAX_TILT_RAD = 1.4`, ~80°): `H_tilt` loses
  invertibility; physical Scheimpflug tilts are a few degrees — rejected.
- **Baseline nearly parallel to the optical axis** (forward motion):
  `R_rect` degrades as the baseline leaves the image plane; row alignment
  still holds but the rectified field of view distorts. An exactly parallel
  baseline switches the `R_rect` y-axis seed so the construction stays
  defined; the near-parallel case is not guarded — a property of the
  construction itself.

## Gauge

Camera 0 is the reference: its frame is the rectification's world frame.
`K_rect` (`RectifyOptions::k_rect`, default the average of the two `K`s)
fixes the free intrinsic scale of the rectified pair; row alignment is
invariant to its choice as long as both cameras share it.

## Evidence

- **Unit tests** (`crates/vision-mvg/src/rectification.rs`): rows of
  corresponding points agree to `< 1e-6 px` through the core Scheimpflug
  model (`scheimpflug_pair_rows_align`); the zero-tilt path equals pinhole
  rectification (`zero_tilt_matches_standard_rectification`); baseline
  recovery, explicit `K_rect`, zero-baseline and extreme-tilt rejection.
- **Real-data gate**: the `scheimpflug_rig_ref_rectify` example (examples-private)
  rectifies all oracle camera pairs of the `scheimpflug_rig_ref` Scheimpflug rig (real
  `K`, asymmetric per-camera ~−5° tilts, rig extrinsics) — worst rectified
  row disagreement **3.4e-13 px**.
- **Downstream integration**: the app's Depth workspace rectifies then
  dense-matches (`vision-mvg::dense`); its test
  `compute_disparity_recovers_board_on_committed_rig`
  (`app/src-tauri/src/disparity.rs`) runs the pipeline on the committed
  `data/stereo` rig and checks disparity density and board planarity.
- **Related**: [two-view-triangulation.md](two-view-triangulation.md) (the
  epipolar geometry being rectified).
