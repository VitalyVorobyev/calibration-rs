# Granular Calibration Functions

This document describes the granular building blocks available for custom calibration workflows. These functions provide fine-grained control compared to the high-level session API.

## When to Use Granular Functions

**Use the Session API when:**
- You have a standard calibration workflow
- You want automatic state management and checkpointing
- Type safety and enforced stage transitions are valuable

**Use granular functions when:**
- You need to inspect intermediate results
- You want to compose custom workflows
- You need to integrate calibration into a larger system
- You want maximum flexibility and control

## Architecture Layers

```
calib-pipeline (high-level)
    ├─ Session API (CalibrationSession) - Stateful, type-safe workflows
    ├─ Pipeline functions (run_*) - All-in-one convenience functions
    └─ Helper functions - Granular operations (Phase 2B)

calib-optim (mid-level)
    └─ Problem builders (optimize_*) - Non-linear refinement

calib-linear (low-level)
    └─ Initialization solvers - Closed-form solutions
```

## Available Functions by Category

### 1. Intrinsics Estimation (calib-linear)

#### Iterative Zhang's Method (with distortion)
```rust
use calib_linear::iterative_intrinsics::{
    IterativeIntrinsicsSolver,
    IterativeCalibView,
    IterativeIntrinsicsOptions,
};

// Prepare views
let views: Vec<IterativeCalibView> = /* ... */;

// Configure options
let opts = IterativeIntrinsicsOptions {
    iterations: 2,  // 1-3 typically sufficient
    distortion_opts: DistortionFitOptions {
        fix_k3: true,
        fix_tangential: false,
        iters: 8,
    },
};

// Estimate intrinsics + distortion
let result = IterativeIntrinsicsSolver::estimate(&views, opts)?;

// Access results
println!("K: fx={}, fy={}", result.intrinsics.fx, result.intrinsics.fy);
println!("Distortion: k1={}, k2={}", result.distortion.k1, result.distortion.k2);
```

**Typical accuracy:** 10-40% error (sufficient for non-linear init)

#### Direct Zhang's Method (no distortion)
```rust
use calib_linear::zhang_intrinsics::estimate_intrinsics_from_homographies;
use nalgebra::Matrix3;

let homographies: Vec<Matrix3<f64>> = /* compute separately */;
let k = estimate_intrinsics_from_homographies(&homographies)?;
```

**Use when:** You have pre-computed homographies or don't need distortion

#### Distortion Fitting
```rust
use calib_linear::distortion_fit::{
    estimate_distortion_from_homographies,
    DistortionFitOptions,
};

let opts = DistortionFitOptions {
    fix_k3: true,
    fix_tangential: false,
    iters: 8,
};

let distortion = estimate_distortion_from_homographies(
    &views,
    &intrinsics,
    opts,
)?;
```

**Use when:** You have K and want to fit distortion separately

### 2. Homography Estimation (calib-linear)

#### DLT Homography (single view)
```rust
use calib_linear::homography::dlt_homography;

let world_points: Vec<Pt3> = /* ... */;
let image_points: Vec<Vec2> = /* ... */;

let H = dlt_homography(&world_points, &image_points)?;
```

#### RANSAC Homography (robust)
```rust
use calib_linear::homography::{dlt_homography_ransac, HomographyRansacOptions};

let opts = HomographyRansacOptions {
    threshold: 2.0,  // Inlier threshold in pixels
    max_iters: 1000,
    confidence: 0.999,
};

let result = dlt_homography_ransac(&world_points, &image_points, opts)?;
println!("Inliers: {}/{}", result.inliers.len(), world_points.len());
```

**Use when:** You have outliers in corner detections

### 3. Pose Estimation (calib-linear)

#### PnP Solvers
```rust
use calib_linear::pnp::PnpSolver;

// DLT (6+ points)
let pose = PnpSolver::dlt(&points_3d, &pixels, &intrinsics)?;

// P3P (minimal 3 points, returns up to 4 solutions)
let solutions = PnpSolver::p3p(&points_3d, &pixels, &intrinsics)?;

// EPnP (efficient N-point)
let pose = PnpSolver::epnp(&points_3d, &pixels, &intrinsics)?;

// RANSAC wrapper
let result = PnpSolver::ransac(&points_3d, &pixels, &intrinsics, opts)?;
```

#### Planar Pose from Homography
```rust
use calib_linear::planar_pose::PlanarPoseSolver;

let H = /* computed homography */;
let K = /* intrinsics matrix */;

// Extract board-to-camera pose
let pose = PlanarPoseSolver::from_homography(&H, &K)?;
```

**Use when:** Converting from homography to 3D pose

### 4. Non-Linear Optimization (calib-optim)

#### Planar Intrinsics
```rust
use calib_core::PlanarDataset;
use calib_optim::{
    optimize_planar_intrinsics, BackendSolveOptions, PlanarIntrinsicsParams,
    PlanarIntrinsicsSolveOptions, RobustLoss,
};

// Prepare dataset
let dataset = PlanarDataset::new(views)?;

// Initial estimates (from linear solver)
let init = PlanarIntrinsicsParams::new_from_components(intrinsics, distortion, poses)?;

// Configure optimization
let opts = PlanarIntrinsicsSolveOptions {
    robust_loss: RobustLoss::Huber { scale: 2.0 },
    fix_poses: vec![0],
    ..Default::default()
};

let backend_opts = BackendSolveOptions {
    max_iters: 50,
    verbosity: 1,
    ..Default::default()
};

// Optimize
let result = optimize_planar_intrinsics(&dataset, &init, opts, backend_opts)?;

// Extract results
let camera = &result.params.camera;
println!("Final cost: {}", result.report.final_cost);
```

**Typical accuracy:** <2% error on intrinsics, <1px reprojection error

#### Linescan (Camera + Laser Plane)
```rust
use calib_optim::problems::linescan_bundle::{
    optimize_linescan,
    LinescanDataset,
    LinescanInit,
    LinescanSolveOptions,
    LaserResidualType,
};

let dataset = LinescanDataset::new_single_plane(views)?;
let init = LinescanInit::new(intrinsics, distortion, poses, planes)?;

let opts = LinescanSolveOptions {
    fix_k3: true,
    fix_poses: vec![0],
    laser_residual_type: LaserResidualType::LineDistNormalized,
    ..Default::default()
};

let result = optimize_linescan(&dataset, &init, &opts, &backend_opts)?;
```

#### Hand-Eye Calibration
```rust
use calib_pipeline::handeye::{
    optimize_handeye,
    CameraViewObservations,
    HandEyeDataset,
    HandEyeInit,
    HandEyeSolveOptions,
    RigViewObservations,
};
use calib_optim::ir::HandEyeMode;

let dataset = HandEyeDataset::new(views, num_cameras, HandEyeMode::EyeInHand)?;
let init = HandEyeInit {
    intrinsics,
    distortion,
    cam_to_rig,
    handeye,
    target_poses,
};

let mut opts = HandEyeSolveOptions::default();
opts.fix_extrinsics = vec![true]; // fix camera 0 for gauge freedom
opts.refine_robot_poses = true;
opts.robot_rot_sigma = 0.5_f64.to_radians(); // radians
opts.robot_trans_sigma = 1.0e-3; // meters

let result = optimize_handeye(dataset, init, opts, backend_opts)?;
```

Notes:
- Default hand-eye optimization assumes a fixed target (single `T_B_T` across views).
- Legacy per-view target relaxation is available via `relax_target_poses = true`.
- Robot pose correction priors use radians (rotation) and meters (translation); delta_0 is fixed to zero.

#### Rig Extrinsics (Multi-Camera)
```rust
use calib_optim::backend::BackendSolveOptions;
use calib_optim::problems::rig_extrinsics::{
    optimize_rig_extrinsics,
    RigExtrinsicsDataset,
    RigExtrinsicsInit,
    RigExtrinsicsSolveOptions,
};

let mut opts = RigExtrinsicsSolveOptions::default();
opts.fix_extrinsics = vec![true, false]; // fix camera 0 extrinsics (rig frame)
opts.fix_rig_poses = vec![0]; // optional gauge fix (fix first target->rig pose)

let backend_opts = BackendSolveOptions::default();
let result = optimize_rig_extrinsics(dataset, init, opts, backend_opts)?;

// Extract baseline for stereo rig
if result.cam_to_rig.len() == 2 {
    // If the rig frame is camera 0, then cam_to_rig[1] is T_R_C1 == T_C0_C1.
    let cam1_to_rig = &result.cam_to_rig[1];
    let cam0_to_cam1 = cam1_to_rig.inverse();
    println!("Baseline (cam0->cam1): {:?}", cam0_to_cam1.translation.vector);
}
```

### 5. Multi-Camera Initialization (calib-linear)

#### Extrinsics from Camera-Target Poses
```rust
use calib_linear::extrinsics::estimate_extrinsics_from_cam_target_poses;

// cam_se3_target[view][cam] = camera->target for that view/camera.
let cam_se3_target: Vec<Vec<Option<Iso3>>> = /* ... */;

// Reference camera index defines the rig frame (rig == reference camera frame).
let extrinsics = estimate_extrinsics_from_cam_target_poses(&cam_se3_target, 0)?;

let cam_to_rig = extrinsics.cam_to_rig; // camera->rig
let rig_from_target = extrinsics.rig_from_target; // target->rig (per view)
```

#### Hand-Eye DLT
```rust
use calib_linear::handeye::estimate_handeye_dlt;

// Robot base-to-end-effector transforms
let base_to_ee: Vec<Iso3> = /* ... */;
// Camera poses in the target frame (invert target-in-camera poses)
let cam_in_target: Vec<Iso3> = /* ... */;

// Solve AX = XB (gripper->camera)
let hand_eye_transform = estimate_handeye_dlt(&base_to_ee, &cam_in_target, 1.0)?;
```

### 6. Epipolar Geometry (calib-linear)

#### Essential Matrix (calibrated cameras)
```rust
use calib_linear::epipolar::essential_matrix_5pt;

let pts1: Vec<Vec2> = /* normalized coordinates */;
let pts2: Vec<Vec2> = /* normalized coordinates */;

let essential_matrices = essential_matrix_5pt(&pts1, &pts2)?;
// Returns up to 10 solutions, use RANSAC to select best
```

#### Fundamental Matrix (uncalibrated)
```rust
use calib_linear::epipolar::{
    fundamental_matrix_7pt,
    fundamental_matrix_8pt,
    fundamental_matrix_8pt_ransac,
};

// 7-point (minimal, up to 3 solutions)
let F_list = fundamental_matrix_7pt(&pts1, &pts2)?;

// 8-point (normalized DLT)
let F = fundamental_matrix_8pt(&pts1, &pts2)?;

// 8-point + RANSAC (robust)
let result = fundamental_matrix_8pt_ransac(&pts1, &pts2, opts)?;
```

#### Decompose Essential Matrix
```rust
use calib_linear::epipolar::decompose_essential;

let E = /* essential matrix */;
let solutions = decompose_essential(&E)?;  // 4 possible (R, t) combinations

// Use cheirality check to select valid solution
```

### 7. Triangulation (calib-linear)

```rust
use calib_linear::triangulation::linear_triangulation;

let pose1 = Iso3::identity();  // Reference camera
let pose2 = /* second camera pose */;

let pt1 = Vec2::new(x1, y1);
let pt2 = Vec2::new(x2, y2);

let point_3d = linear_triangulation(&pose1, &pose2, &pt1, &pt2)?;
```

## Example Custom Workflows

### Workflow 1: Inspect Linear Initialization Before Optimization

```rust
use calib_linear::iterative_intrinsics::IterativeIntrinsicsSolver;
use calib_optim::planar_intrinsics::optimize_planar_intrinsics;

// Step 1: Linear initialization
let linear_result = IterativeIntrinsicsSolver::estimate(&views, linear_opts)?;

// Inspect before committing to optimization
println!("Initial K: {:?}", linear_result.intrinsics);
println!("Initial distortion: {:?}", linear_result.distortion);

// Decide whether to proceed
if linear_result.mean_reproj_error > 10.0 {
    eprintln!("Warning: Poor linear initialization, check corner detection");
}

// Step 2: Non-linear refinement
let dataset = build_dataset_from_views(&views)?;
let init = PlanarIntrinsicsInit::from_linear_result(&linear_result)?;
let final_result = optimize_planar_intrinsics(dataset, init, optim_opts, backend_opts)?;
```

### Workflow 2: Custom Stereo Rig Calibration

```rust
use calib_pipeline::{run_rig_extrinsics, RigExtrinsicsConfig, RigExtrinsicsInput};

// Build multi-camera observations: input.views[view].cameras[camera] = Some(obs)
let input: RigExtrinsicsInput = /* ... */;

// Run per-camera intrinsics init + rig extrinsics init + joint BA
let report = run_rig_extrinsics(&input, &RigExtrinsicsConfig::default())?;

// If the rig frame is camera 0, baseline cam0->cam1 is the inverse of cam1->rig.
if report.cam_to_rig.len() == 2 {
    let cam0_to_cam1 = report.cam_to_rig[1].inverse();
    println!("Baseline (cam0->cam1): {:?}", cam0_to_cam1.translation.vector);
}
```

### Workflow 3: Homography-Based Calibration with Outlier Analysis

```rust
use calib_linear::homography::dlt_homography_ransac;

let mut homographies = Vec::new();
let mut inlier_ratios = Vec::new();

for view in &views {
    let result = dlt_homography_ransac(&view.points_3d, &view.points_2d, ransac_opts)?;

    let inlier_ratio = result.inliers.len() as f64 / view.points_3d.len() as f64;
    inlier_ratios.push(inlier_ratio);

    if inlier_ratio < 0.7 {
        eprintln!("Warning: View {} has low inlier ratio: {:.1}%",
                  view.id, inlier_ratio * 100.0);
    }

    homographies.push(result.model);
}

// Estimate intrinsics from filtered homographies
let k = estimate_intrinsics_from_homographies(&homographies)?;
```

## Best Practices

1. **Always use RANSAC for real data** - Corner detection is noisy
2. **Check intermediate results** - Don't blindly trust pipeline output
3. **Fix gauge freedom** - Always fix at least one pose in optimization
4. **Use robust loss functions** - Huber or Cauchy for handling outliers
5. **Start with fix_k3=true** - Only optimize k3 for wide-angle lenses
6. **Validate reprojection errors** - <1px mean error is expected after optimization
7. **Use proper coordinate conventions**:
   - Poses: `T_C_W` (world-to-camera)
   - Fundamental matrix: takes pixel coordinates
   - Essential matrix: takes normalized coordinates
