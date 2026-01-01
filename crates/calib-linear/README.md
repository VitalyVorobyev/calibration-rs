# calib-linear

Linear and closed-form initialization solvers for camera and rig calibration.

This crate focuses on deterministic, minimal-dependency implementations of
classic solvers. These are intended as **starting points** for non-linear
refinement in `calib-optim` or `calib-pipeline`.

## Algorithms
- Homography estimation (normalized DLT, optional RANSAC)
- Planar intrinsics (Zhang method from multiple homographies)
- Planar pose from homography and intrinsics
- Fundamental matrix: 8-point (normalized) and 7-point minimal solvers
- Essential matrix: 5-point minimal solver + decomposition to (R, t)
- Camera pose (PnP): DLT, P3P, EPnP, and DLT-in-RANSAC
- Camera matrix DLT and RQ decomposition
- Linear triangulation (DLT)
- Multi-camera rig extrinsics
- Hand-eye calibration (Tsai-Lenz)

## Coordinate conventions
- Poses are `T_C_W`: transform from world/board coordinates into the camera
  frame.
- Fundamental matrix solvers accept **pixel coordinates**.
- Essential matrix solvers expect **normalized coordinates** (after applying
  `K^{-1}`).
- P3P and EPnP accept pixel coordinates and intrinsics; they normalize
  internally.

## Usage
Estimate a homography from planar correspondences:

```rust
use calib_linear::HomographySolver;
use calib_core::Pt2;

let world = vec![
    Pt2::new(0.0, 0.0),
    Pt2::new(1.0, 0.0),
    Pt2::new(1.0, 1.0),
    Pt2::new(0.0, 1.0),
];
let image = vec![
    Pt2::new(120.0, 200.0),
    Pt2::new(220.0, 198.0),
    Pt2::new(225.0, 300.0),
    Pt2::new(118.0, 302.0),
];

let h = HomographySolver::dlt(&world, &image)?;
println!("H = {h}");
# Ok::<(), calib_linear::HomographyError>(())
```

For robust estimation, use the RANSAC wrappers and tune `RansacOptions`.

## Relationship to other crates
- `calib-core`: math types, camera models, and RANSAC engine.
- `calib-optim`: non-linear refinement and solvers.
- `calib-pipeline`: high-level calibration pipelines.
