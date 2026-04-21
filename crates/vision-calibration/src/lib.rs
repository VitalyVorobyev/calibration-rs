//! High-level entry crate for the `calibration-rs` camera calibration library.
//!
//! This crate provides a unified API for camera calibration workflows:
//! - Single-camera intrinsics calibration (Zhang's method with distortion)
//! - Single-camera intrinsics calibration with Scheimpflug tilt
//! - Single-camera hand-eye calibration (camera on robot arm)
//! - Multi-camera rig extrinsics calibration
//! - Multi-camera rig + hand-eye calibration
//!
//! # Quick Start
//!
//! ```no_run
//! # fn main() -> anyhow::Result<()> {
//! # let dataset = unimplemented!();
//! use vision_calibration::prelude::*;
//! use vision_calibration::planar_intrinsics::{step_init, step_optimize};
//!
//! // Create calibration session
//! let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
//! session.set_input(dataset)?;
//!
//! // Option 1: Step-by-step (recommended for inspection)
//! step_init(&mut session, None)?;
//! step_optimize(&mut session, None)?;
//!
//! // Option 2: Pipeline function (convenience)
//! // run_planar_intrinsics(&mut session)?;
//!
//! // Export results
//! let result = session.export()?;
//! # Ok(())
//! # }
//! ```
//!
//! # Module Organization
//!
//! ## High-Level Calibration Workflows
//!
//! - [`session`] - Session framework (`CalibrationSession`, `ProblemType`)
//! - [`planar_intrinsics`] - Single-camera intrinsics (Zhang's method)
//! - [`single_cam_handeye`] - Single camera + hand-eye calibration
//! - [`rig_extrinsics`] - Multi-camera rig extrinsics
//! - [`rig_handeye`] - Multi-camera rig + hand-eye
//! - [`laserline_device`] - Single camera + laser plane device
//! - [`scheimpflug_intrinsics`] - Single-camera planar intrinsics with Scheimpflug tilt
//!
//! ## Foundation Crates (Advanced Users)
//!
//! - [`core`] - Math types, camera models, RANSAC primitives
//! - [`linear`] - Closed-form initialization algorithms
//! - [`optim`] - Non-linear least-squares optimization
//! - [`synthetic`] - Synthetic data generation for testing
//!
//! # Session API
//!
//! All calibration workflows use the [`session::CalibrationSession`] state container.
//! Each problem type has its own set of step functions — see the table below
//! and the per-module documentation for details.
//!
//! The common pattern is:
//! 1. Create a session for the problem type
//! 2. Set input data with `set_input`
//! 3. Optionally configure with `update_config`
//! 4. Run the problem-specific step functions (or a convenience `run_calibration`)
//! 5. Export results with `export`
//!
//! # Available Problem Types
//!
//! | Problem Type | Input | Steps |
//! |--------------|-------|-------|
//! | [`PlanarIntrinsicsProblem`](planar_intrinsics) | `PlanarDataset` | `step_init` → `step_optimize` |
//! | [`SingleCamHandeyeProblem`](single_cam_handeye) | `SingleCamHandeyeInput` | `step_intrinsics_init` → `step_intrinsics_optimize` → `step_handeye_init` → `step_handeye_optimize` |
//! | [`RigExtrinsicsProblem`](rig_extrinsics) | `RigExtrinsicsInput` | `step_intrinsics_init_all` → `step_intrinsics_optimize_all` → `step_rig_init` → `step_rig_optimize` |
//! | [`RigHandeyeProblem`](rig_handeye) | `RigHandeyeInput` | `step_intrinsics_init_all` → `step_intrinsics_optimize_all` → `step_rig_init` → `step_rig_optimize` → `step_handeye_init` → `step_handeye_optimize` |
//! | [`LaserlineDeviceProblem`](laserline_device) | `LaserlineDeviceInput` | `step_init` → `step_optimize` |
//! | [`ScheimpflugIntrinsicsProblem`](scheimpflug_intrinsics) | `PlanarDataset` | `step_init` → `step_optimize` |

#![deny(missing_docs)]

/// Typed error returned by all public calibration step functions.
pub use vision_calibration_pipeline::Error;

/// Single-camera planar intrinsics with Scheimpflug/tilted sensor refinement.
///
/// This high-level helper mirrors planar intrinsics calibration, but optimizes a
/// Brown-Conrady camera together with two Scheimpflug tilt parameters.
pub mod scheimpflug_intrinsics {
    pub use vision_calibration_pipeline::scheimpflug_intrinsics::{
        IntrinsicsInitOptions, IntrinsicsOptimizeOptions, ScheimpflugFixMask,
        ScheimpflugIntrinsicsConfig, ScheimpflugIntrinsicsExport, ScheimpflugIntrinsicsInput,
        ScheimpflugIntrinsicsParams, ScheimpflugIntrinsicsProblem, ScheimpflugIntrinsicsResult,
        ScheimpflugIntrinsicsState, run_calibration, step_init, step_optimize,
    };
}

// ═══════════════════════════════════════════════════════════════════════════════
// Session Framework
// ═══════════════════════════════════════════════════════════════════════════════

/// Session framework for structured calibration workflows.
///
/// Provides mutable state containers, step functions, and JSON checkpointing.
pub mod session {
    pub use vision_calibration_pipeline::session::{
        CalibrationSession, ExportRecord, InvalidationPolicy, LogEntry, ProblemType,
        SessionMetadata,
    };
}

// ═══════════════════════════════════════════════════════════════════════════════
// Problem-Specific Modules
// ═══════════════════════════════════════════════════════════════════════════════

/// Planar intrinsics calibration (Zhang's method with Brown-Conrady distortion).
///
/// # Steps
/// 1. `step_init` - Zhang's method with iterative distortion estimation
/// 2. `step_optimize` - Non-linear refinement
/// 3. `step_filter` (optional) - Remove outliers by reprojection error
///
/// # Example
/// ```no_run
/// # fn main() -> anyhow::Result<()> {
/// # let dataset = unimplemented!();
/// use vision_calibration::prelude::*;
/// use vision_calibration::planar_intrinsics::{step_init, step_optimize, run_calibration};
///
/// let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
/// session.set_input(dataset)?;
/// run_calibration(&mut session)?;
/// let result = session.export()?;
/// # Ok(())
/// # }
/// ```
pub mod planar_intrinsics {
    pub use vision_calibration_pipeline::planar_intrinsics::{
        // Step options
        FilterOptions,
        IntrinsicsInitOptions,
        IntrinsicsOptimizeOptions,
        // Problem type and config
        PlanarIntrinsicsConfig,
        // Re-exports from vision-calibration-optim
        PlanarIntrinsicsEstimate,
        PlanarIntrinsicsExport,
        PlanarIntrinsicsParams,
        PlanarIntrinsicsProblem,
        PlanarIntrinsicsSolveOptions,
        PlanarState,
        // Step functions
        run_calibration,
        run_calibration_with_filtering,
        step_filter,
        step_init,
        step_optimize,
    };
}

/// Single-camera hand-eye calibration (intrinsics + hand-eye transform).
///
/// For calibrating a camera mounted on a robot arm.
///
/// # Steps
/// 1. `step_intrinsics_init` - Zhang's method
/// 2. `step_intrinsics_optimize` - Non-linear intrinsics refinement
/// 3. `step_handeye_init` - Tsai-Lenz linear estimation
/// 4. `step_handeye_optimize` - Bundle adjustment
///
/// # Example
/// ```no_run
/// # fn main() -> anyhow::Result<()> {
/// # let input = unimplemented!();
/// use vision_calibration::prelude::*;
/// use vision_calibration::single_cam_handeye::{
///     run_calibration, SingleCamHandeyeInput, SingleCamHandeyeProblem
/// };
///
/// let mut session = CalibrationSession::<SingleCamHandeyeProblem>::new();
/// session.set_input(input)?;
/// run_calibration(&mut session)?;
/// let result = session.export()?;
/// # Ok(())
/// # }
/// ```
pub mod single_cam_handeye {
    pub use vision_calibration_pipeline::single_cam_handeye::{
        // Step options
        HandeyeInitOptions,
        // Problem type and config
        HandeyeMeta,
        HandeyeOptimizeOptions,
        IntrinsicsInitOptions,
        IntrinsicsOptimizeOptions,
        SingleCamHandeyeConfig,
        SingleCamHandeyeExport,
        SingleCamHandeyeInput,
        SingleCamHandeyeProblem,
        SingleCamHandeyeState,
        SingleCamHandeyeView,
        // Step functions
        run_calibration,
        step_handeye_init,
        step_handeye_optimize,
        step_intrinsics_init,
        step_intrinsics_optimize,
    };
}

/// Single laserline device calibration (camera + laser plane).
///
/// # Steps
/// 1. `step_init` - Iterative intrinsics + linear laser plane init
/// 2. `step_optimize` - Joint bundle adjustment
///
/// # Example
/// ```no_run
/// # fn main() -> anyhow::Result<()> {
/// # let input = unimplemented!();
/// use vision_calibration::prelude::*;
/// use vision_calibration::laserline_device::{
///     run_calibration, LaserlineDeviceInput, LaserlineDeviceProblem
/// };
///
/// let mut session = CalibrationSession::<LaserlineDeviceProblem>::new();
/// session.set_input(input)?;
/// run_calibration(&mut session, None)?;
/// let result = session.export()?;
/// # Ok(())
/// # }
/// ```
pub mod laserline_device {
    pub use vision_calibration_pipeline::laserline_device::{
        DeviceInitOptions, DeviceOptimizeOptions, LaserlineDeviceConfig, LaserlineDeviceExport,
        LaserlineDeviceInitConfig, LaserlineDeviceInput, LaserlineDeviceOptimizeConfig,
        LaserlineDeviceOutput, LaserlineDeviceProblem, LaserlineDeviceSolverConfig,
        LaserlineDeviceState, run_calibration, step_init, step_optimize,
    };
}

/// Multi-camera rig extrinsics calibration.
///
/// For calibrating a multi-camera rig, estimating per-camera intrinsics
/// and camera-to-rig transforms.
///
/// # Steps
/// 1. `step_intrinsics_init_all` - Per-camera Zhang's method
/// 2. `step_intrinsics_optimize_all` - Per-camera non-linear refinement
/// 3. `step_rig_init` - Linear estimation of camera-to-rig transforms
/// 4. `step_rig_optimize` - Joint bundle adjustment
///
/// # Example
/// ```no_run
/// # fn main() -> anyhow::Result<()> {
/// # let input = unimplemented!();
/// use vision_calibration::prelude::*;
/// use vision_calibration::rig_extrinsics::{
///     run_calibration, RigExtrinsicsInput, RigExtrinsicsProblem
/// };
///
/// let mut session = CalibrationSession::<RigExtrinsicsProblem>::new();
/// session.set_input(input)?;
/// run_calibration(&mut session)?;
/// let result = session.export()?;
/// # Ok(())
/// # }
/// ```
pub mod rig_extrinsics {
    pub use vision_calibration_pipeline::rig_extrinsics::{
        // Step options
        IntrinsicsInitOptions,
        IntrinsicsOptimizeOptions,
        // Problem type and config
        RigExtrinsicsConfig,
        RigExtrinsicsExport,
        RigExtrinsicsInput,
        RigExtrinsicsProblem,
        RigExtrinsicsState,
        RigOptimizeOptions,
        // Step functions
        run_calibration,
        step_intrinsics_init_all,
        step_intrinsics_optimize_all,
        step_rig_init,
        step_rig_optimize,
    };
}

/// Multi-camera rig hand-eye calibration.
///
/// For calibrating a multi-camera rig mounted on a robot arm, including
/// per-camera intrinsics, rig extrinsics, and hand-eye transform.
///
/// # Steps (6 total)
/// 1. `step_intrinsics_init_all` - Per-camera Zhang's method
/// 2. `step_intrinsics_optimize_all` - Per-camera non-linear refinement
/// 3. `step_rig_init` - Linear estimation of camera-to-rig transforms
/// 4. `step_rig_optimize` - Rig bundle adjustment
/// 5. `step_handeye_init` - Tsai-Lenz linear estimation
/// 6. `step_handeye_optimize` - Hand-eye bundle adjustment
///
/// # Example
/// ```no_run
/// # fn main() -> anyhow::Result<()> {
/// # let input = unimplemented!();
/// use vision_calibration::prelude::*;
/// use vision_calibration::rig_handeye::{run_calibration, RigHandeyeInput, RigHandeyeProblem};
///
/// let mut session = CalibrationSession::<RigHandeyeProblem>::new();
/// session.set_input(input)?;
/// run_calibration(&mut session)?;
/// let result = session.export()?;
/// # Ok(())
/// # }
/// ```
pub mod rig_handeye {
    pub use vision_calibration_pipeline::rig_handeye::{
        // Step options
        HandeyeInitOptions,
        HandeyeOptimizeOptions,
        IntrinsicsInitOptions,
        IntrinsicsOptimizeOptions,
        // Problem type and config
        RigHandeyeBaConfig,
        RigHandeyeConfig,
        RigHandeyeExport,
        RigHandeyeInitConfig,
        RigHandeyeInput,
        RigHandeyeIntrinsicsConfig,
        RigHandeyeProblem,
        RigHandeyeRigConfig,
        RigHandeyeSolverConfig,
        RigHandeyeState,
        RigOptimizeOptions,
        // Step functions
        run_calibration,
        step_handeye_init,
        step_handeye_optimize,
        step_intrinsics_init_all,
        step_intrinsics_optimize_all,
        step_rig_init,
        step_rig_optimize,
    };
}

/// Multi-camera rig extrinsics calibration with Scheimpflug-tilted sensors.
///
/// Parallels [`rig_extrinsics`] with per-camera Scheimpflug sensor support.
pub mod rig_scheimpflug_extrinsics {
    pub use vision_calibration_pipeline::rig_scheimpflug_extrinsics::{
        IntrinsicsInitOptions, IntrinsicsOptimizeOptions, RigOptimizeOptions,
        RigScheimpflugExtrinsicsConfig, RigScheimpflugExtrinsicsExport,
        RigScheimpflugExtrinsicsInput, RigScheimpflugExtrinsicsProblem,
        RigScheimpflugExtrinsicsState, run_calibration, step_intrinsics_init_all,
        step_intrinsics_optimize_all, step_rig_init, step_rig_optimize,
    };
}

/// Multi-camera rig hand-eye calibration with Scheimpflug-tilted sensors (EyeInHand).
///
/// Parallels [`rig_handeye`] with per-camera Scheimpflug sensor support.
pub mod rig_scheimpflug_handeye {
    pub use vision_calibration_pipeline::rig_scheimpflug_handeye::{
        HandeyeInitOptions, HandeyeOptimizeOptions, IntrinsicsInitOptions,
        IntrinsicsOptimizeOptions, RigOptimizeOptions, RigScheimpflugHandeyeBaConfig,
        RigScheimpflugHandeyeConfig, RigScheimpflugHandeyeExport, RigScheimpflugHandeyeInitConfig,
        RigScheimpflugHandeyeInput, RigScheimpflugHandeyeIntrinsicsConfig,
        RigScheimpflugHandeyeProblem, RigScheimpflugHandeyeRigConfig,
        RigScheimpflugHandeyeSolverConfig, RigScheimpflugHandeyeState, run_calibration,
        step_handeye_init, step_handeye_optimize, step_intrinsics_init_all,
        step_intrinsics_optimize_all, step_rig_init, step_rig_optimize,
    };
}

/// Rig-level laserline calibration.
///
/// Given an upstream rig calibration ([`rig_scheimpflug_handeye`]), fits one laser
/// plane per camera and reports each plane in the rig frame.
pub mod rig_laserline_device {
    pub use vision_calibration_pipeline::rig_laserline_device::{
        RigLaserlineDeviceConfig, RigLaserlineDeviceExport, RigLaserlineDeviceInput,
        RigLaserlineDeviceProblem, RigLaserlineDeviceState, RigUpstreamCalibration, StepOptions,
        run_calibration, step_init, step_optimize,
    };
}

/// Map a laser pixel in a specific camera to a 3D point in the robot gripper frame.
///
/// Given:
/// - `cam_idx`: which camera of the rig captured the pixel.
/// - `pixel`: observed pixel on the laser line.
/// - `rig_cal`: upstream rig + Scheimpflug hand-eye calibration.
/// - `laser_planes_rig`: laser planes (one per camera) expressed in rig frame.
///
/// Returns the 3D point in gripper (robot flange) frame:
///
/// 1. Undistort `pixel` to a normalized camera-frame ray using the full
///    pinhole + Brown-Conrady + Scheimpflug chain (inverted).
/// 2. Transform the ray into rig frame via `cam_se3_rig[cam_idx].inverse()`.
/// 3. Intersect the ray with `laser_planes_rig[cam_idx]`.
/// 4. Apply `gripper_se3_rig` to obtain the point in gripper frame.
///
/// # Errors
///
/// Returns [`Error`] if `cam_idx` is out of range, if the ray never intersects
/// the plane, or if undistortion fails.
pub fn pixel_to_gripper_point(
    cam_idx: usize,
    pixel: vision_calibration_core::Pt2,
    rig_cal: &rig_scheimpflug_handeye::RigScheimpflugHandeyeExport,
    laser_planes_rig: &[vision_calibration_optim::LaserPlane],
) -> Result<vision_calibration_core::Pt3, Error> {
    use vision_calibration_core::{DistortionModel, Mat3, Pt2, Pt3, SensorModel, Vec3};

    let n_cams = rig_cal.cameras.len();
    if cam_idx >= n_cams {
        return Err(Error::InvalidInput {
            reason: format!("cam_idx {cam_idx} out of range (num_cameras = {n_cams})"),
        });
    }
    if laser_planes_rig.len() != n_cams {
        return Err(Error::InvalidInput {
            reason: format!(
                "laser_planes_rig has {} entries, expected {n_cams}",
                laser_planes_rig.len()
            ),
        });
    }
    if cam_idx >= rig_cal.sensors.len() || cam_idx >= rig_cal.cam_se3_rig.len() {
        return Err(Error::InvalidInput {
            reason: "rig calibration missing per-cam data".to_string(),
        });
    }

    let cam = &rig_cal.cameras[cam_idx];
    let sensor = &rig_cal.sensors[cam_idx];

    // Undistort pixel to a normalized camera-frame direction by inverting the full
    // chain: pixel -> sensor (after Scheimpflug) -> normalized (after distortion) -> ray.
    let k_matrix = Mat3::new(
        cam.k.fx, cam.k.skew, cam.k.cx, 0.0, cam.k.fy, cam.k.cy, 0.0, 0.0, 1.0,
    );
    let k_inv = k_matrix
        .try_inverse()
        .ok_or_else(|| Error::Numerical("intrinsics matrix is singular".to_string()))?;
    let uv_h: Vec3 = Vec3::new(pixel.x, pixel.y, 1.0);
    let sensor_h: Vec3 = k_inv * uv_h;
    if sensor_h.z.abs() < 1e-12 {
        return Err(Error::Numerical(
            "pixel projects to infinity after K^-1".to_string(),
        ));
    }
    let sensor_pt: Pt2 = Pt2::new(sensor_h.x / sensor_h.z, sensor_h.y / sensor_h.z);
    // Invert Scheimpflug sensor (sensor -> distorted normalized).
    let compiled_sensor = sensor.compile();
    let distorted_pt = compiled_sensor.sensor_to_normalized(&sensor_pt);
    // Invert distortion (distorted -> undistorted normalized).
    let normalized = cam.dist.undistort(&distorted_pt);
    // Ray direction in camera frame: (x_n, y_n, 1).
    let dir_cam = Vec3::new(normalized.x, normalized.y, 1.0);

    // Transform ray origin/direction from camera to rig.
    let cam_to_rig = rig_cal.cam_se3_rig[cam_idx].inverse();
    let origin_rig = cam_to_rig.translation.vector;
    let dir_rig = cam_to_rig.rotation.transform_vector(&dir_cam);

    // Intersect with laser plane (in rig frame): n · (o + t d) + d_plane = 0.
    let plane = &laser_planes_rig[cam_idx];
    let n = plane.normal.into_inner();
    let denom = n.dot(&dir_rig);
    if denom.abs() < 1e-12 {
        return Err(Error::Numerical(
            "ray is parallel to laser plane; no intersection".to_string(),
        ));
    }
    let t = -(n.dot(&origin_rig) + plane.distance) / denom;
    let p_rig: Vec3 = origin_rig + t * dir_rig;

    // Apply gripper_se3_rig to get point in gripper frame.
    let p_rig_pt = Pt3::from(p_rig);
    let p_gripper = rig_cal.gripper_se3_rig.transform_point(&p_rig_pt);
    Ok(p_gripper)
}

// ═══════════════════════════════════════════════════════════════════════════════
// Foundation Crates (Advanced Users)
// ═══════════════════════════════════════════════════════════════════════════════

/// Core math types, camera models, and RANSAC primitives.
///
/// Re-exports selected foundational types from `vision_calibration_core`.
pub mod core {
    pub use vision_calibration_core::{
        BrownConrady5, Camera, CameraParams, CorrespondenceView, DistortionFixMask,
        DistortionParams, FxFyCxCySkew, IdentitySensor, IntrinsicsFixMask, IntrinsicsParams, Iso3,
        NoMeta, Pinhole, PinholeCamera, PlanarDataset, ProjectionParams, Pt2, Pt3, Real,
        ReprojectionStats, RigDataset, RigView, RigViewObs, ScheimpflugParams, SensorModel,
        SensorParams, Vec2, Vec3, View, make_pinhole_camera, pinhole_camera_params,
    };
}

/// Closed-form initialization algorithms.
///
/// Includes homography estimation, Zhang's method, PnP solvers,
/// triangulation, hand-eye solvers, and more.
///
/// Re-exports everything from `vision_calibration_linear`.
pub mod linear {
    pub use vision_calibration_linear::*;
}

/// Non-linear optimization with backend-agnostic IR.
///
/// Includes optimization problems, factors, and solver backends.
///
/// Re-exports everything from `vision_calibration_optim`.
pub mod optim {
    pub use vision_calibration_optim::*;
}

/// Deterministic synthetic data generation for testing.
///
/// Provides builders for creating synthetic calibration datasets.
pub mod synthetic {
    pub use vision_calibration_core::synthetic::*;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Prelude (Quick Start)
// ═══════════════════════════════════════════════════════════════════════════════

/// Minimal re-exports for planar "hello world" calibration.
///
/// ```no_run
/// use vision_calibration::prelude::*;
/// ```
pub mod prelude {
    /// Session framework for calibration workflows.
    pub use vision_calibration_pipeline::session::CalibrationSession;

    /// Planar intrinsics problem type for hello-world calibration.
    pub use vision_calibration_pipeline::planar_intrinsics::PlanarIntrinsicsProblem;

    /// Convenience planar calibration runner.
    pub use vision_calibration_pipeline::planar_intrinsics::run_calibration as run_planar_intrinsics;

    /// Core geometry and dataset types used in minimal planar workflows.
    pub use vision_calibration_core::{
        BrownConrady5, CorrespondenceView, FxFyCxCySkew, PlanarDataset, Pt2, Pt3, View,
        make_pinhole_camera,
    };
}
