# Stereo rectification — short note

Family: Scheimpflug-aware stereo rectification
(`vision-mvg::rectification::rectify_stereo_pair(left, right, cam1_se3_cam0,
opts) -> StereoRectification`), re-exported as `vision_calibration::mvg::rectification`.
This is deliberately a *short* pack: the C4 gate (below) already pins the
family end-to-end, and the module doc carries the full derivation.

## Model

For a calibrated pair, find homographies `H_left`, `H_right` such that
corresponding points land on the same image row — the precondition for 1-D
disparity search. Because a pixel in this project's camera model (ADR 0005)
is `K · H_tilt · x_n`, the Scheimpflug sensor tilt is just a projective
factor on the normalized plane: pre-multiplying each camera's unprojection
by `H_tilt⁻¹` collapses it to an equivalent frontal pinhole, after which the
textbook Fusiello/Bouguet construction applies:

```
H_left  = K_rect · R_rect        · H_tilt0⁻¹ · K0⁻¹
H_right = K_rect · (R_rect · Rᵀ) · H_tilt1⁻¹ · K1⁻¹
```

with `R = R_C1_C0` and `R_rect` the common orientation whose x-axis is the
baseline. Zero tilt reduces exactly to pinhole rectification. Inputs are
**undistorted** pixels — distortion is removed separately, mirroring
OpenCV's `initUndistortRectifyMap` split.

## Degeneracies and guards

- **Zero baseline** (`‖t‖ < 1e-12`): no epipolar geometry to rectify —
  rejected.
- **Degenerate tilt homography** (tilt beyond ~80°, `MAX_TILT_RAD = 1.4`):
  `H_tilt` loses invertibility long before this; physical Scheimpflug tilts
  are a few degrees — rejected.
- **Baseline nearly parallel to the optical axes** (forward motion): the
  standard construction's `R_rect` degrades as the baseline leaves the
  image plane; row alignment still holds but the rectified FOV distorts.
  Not guarded — a property of the Fusiello construction itself.

## Gauge

Camera 0 is the reference: its frame is the rectification's world frame,
and `K_rect` (from `RectifyOptions`) fixes the free intrinsic scale of the
rectified pair. The row-alignment property is invariant to the choice of
`K_rect`.

## Evidence

- **Row-alignment invariant**: 6 synthetic tests through the real core
  Scheimpflug model (`crates/vision-mvg/src/rectification.rs` tests) —
  rectified rows of corresponding points agree to `< 1e-6 px`; the zero-tilt
  path reproduces pinhole rectification exactly.
- **C4 / D4 gate (real data)**: the `rtv3d_ref_rectify` example
  (examples-private) rectifies all oracle camera pairs of the `rtv3d_ref`
  Scheimpflug rig (real `K`, asymmetric per-camera ~−5° tilts, rig
  extrinsics) — worst rectified row disagreement **3.4e-13 px**.
- **Downstream consumer**: the dense block matcher (`vision-mvg::dense`,
  C5) and the app's Depth workspace run on these rectified pairs; the C5
  bench (`data/stereo`) is an implicit integration gate.
- **Related**: ADR 0005 (camera model; `H_tilt` as sensor stage),
  `docs/notes/two-view-triangulation.md` (the epipolar geometry being
  rectified), backlog C4-RECTIFY completion note (PR #74).
