# rtv3d absolute metric scale

**Finding: the rtv3d calibration is metrically consistent with the legacy oracle
(`artifacts.json`) at a 5.2 mm ChArUco cell; there is no cell-size ambiguity.**
Measured on the rig's hexagon-neighbor spacing, the final laser-informed
joint-BA extrinsics give 98.21 ± 0.41 mm, matching the oracle's healthy-camera
hexagon (98.13 ± 1.10 mm) to **0.08 %**. (Restricted to the four cam-0–4
edges on both sides: 98.42 vs 98.13 mm, −0.29 %; both inside the edge-spread
noise.) The hand-eye-stage extrinsics alone are ~10 % too small; the metric
scale is fixed by the laser point-to-plane term of the joint BA, so scale
comparisons must use the joint-BA extrinsics.

## Oracle hexagon

`artifacts.json`'s `extrinsic[i].camera_se3_sensor` is `T_{cam_i}_{rig}` (rig
frame = cam 0, matching our convention). Inverting each and taking distances
between mechanical neighbors (hexagon 0-1-2-3-4-5-0):

| edge | distance (mm) |
|---|---:|
| 0-1 | 96.25 |
| 1-2 | 98.47 |
| 2-3 | 98.85 |
| 3-4 | 98.96 |
| 4-5 | 173.23 (touches cam 5) |
| 5-0 | 144.97 (touches cam 5) |

Cams 0–4 give four consistent edges (mean 98.13 mm, std 1.10 mm). The two edges
touching cam 5 are far off (145–173 mm), consistent with the oracle's defective
cam 5 (`fx = 51`, 127 px reprojection), so only cams 0–4 are used.

## Our hexagon by pipeline stage

| edge | hand-eye stage (mm) | joint-BA stage (mm) | oracle (mm) |
|---|---:|---:|---:|
| 0-1 | 88.95 | 98.84 | 96.25 |
| 1-2 | 88.78 | 98.07 | 98.47 |
| 2-3 | 88.41 | 98.44 | 98.85 |
| 3-4 | 89.59 | 98.31 | 98.96 |
| 4-5 | 88.79 | 97.50 | 173.23 |
| 5-0 | 88.73 | 98.13 | 144.97 |
| **mean ± std** | **88.88 ± 0.36 mm (0.40 %)** | **98.21 ± 0.41 mm (0.41 %)** | 98.13 ± 1.10 mm (clean 4) |

The joint BA rescales the entire rig uniformly by about +10.5 % relative to the
hand-eye stage while preserving its shape (~0.4 % edge spread). That is the
signature of an under-constrained scale direction being resolved by extra metric
information, not of noise or a different local optimum. The likely mechanism is
the Scheimpflug tilt / principal-point / rig-pose valley, which gives the
camera-and-rig-only stage a shallow direction trading scale against tilt and
pose; stage 4 freezes the converged per-camera tilts and adds the laser
point-to-plane term (weight 1e4), which pins the remaining degree of freedom.

## Why not a cell-size error

The dataset carries two cell-size fields for the same 22×22 board:
`config.json` → `target.cellsize_mm = 5.2` (used throughout) and
`board_charuco.json` → `cell_size_mm = 4.8` (wrong; an internal metadata
inconsistency unrelated to the scale). No file contains any other value (in
particular neither 4.75 nor 5.69 mm). A cell-size error would also produce a
fixed ratio (5.2/4.8 = 1.083), whereas the hand-eye-stage/oracle ratio (~10 %)
depends on the pipeline stage and vanishes to 0.08 % at the joint-BA stage.
`artifacts.json`'s `meta` block is empty, so the oracle's capture provenance
cannot be independently confirmed; its geometry (four clean cameras) is
consistent.

An independent mechanical measurement of the camera-to-camera spacing
(calipers/CMM or the drawing dimension) would be the remaining ground truth;
given the agreement above it is a confirmation exercise only.

## Reproduction

```bash
RTV3D_DATA_DIR=privatedata/rtv3d CELL_SIZE_MM=5.2 RTV3D_HANDEYE=eye_to_hand \
  cargo run --manifest-path crates/vision-calibration-examples-private/Cargo.toml \
  --example rtv3d_rig --release
```

`compare_to_oracle()` in
`crates/vision-calibration-examples-private/examples/rtv3d_rig.rs` prints the
hexagon neighbor-edge table (ours vs oracle, mean ± spread) and sources the
extrinsic-scale comparison from the joint-BA extrinsics when laser data is
present. The dataset's mount-translation nominals in `spec.json` are ~10 %
stale in scale but remain a valid bootstrap seed, since seeded and generic
initialization converge to the same optimum.
