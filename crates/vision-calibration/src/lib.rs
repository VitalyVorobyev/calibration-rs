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
//! | [`scheimpflug_intrinsics::run_calibration`](scheimpflug_intrinsics) | `PlanarDataset` | Direct function (`run_calibration`) |

/// Single-camera planar intrinsics with Scheimpflug/tilted sensor refinement.
///
/// This high-level helper mirrors planar intrinsics calibration, but optimizes a
/// Brown-Conrady camera together with two Scheimpflug tilt parameters.
pub mod scheimpflug_intrinsics {
    pub use vision_calibration_pipeline::scheimpflug_intrinsics::{
        InitOptions, OptimizeOptions, ScheimpflugFixMask, ScheimpflugIntrinsicsCalibrationConfig,
        ScheimpflugIntrinsicsInput, ScheimpflugIntrinsicsParams, ScheimpflugIntrinsicsProblem,
        ScheimpflugIntrinsicsResult, ScheimpflugIntrinsicsState,
        run_calibration as run_calibration_session, run_calibration_direct as run_calibration,
        step_init, step_optimize,
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
        InitOptions,
        OptimizeOptions,
        // Problem type and config
        PlanarConfig,
        PlanarExport,
        // Re-exports from vision-calibration-optim
        PlanarIntrinsicsEstimate,
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
/// use vision_calibration::single_cam_handeye::{run_calibration, SingleCamHandeyeInput};
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
        HandeyeOptimOptions,
        IntrinsicsInitOptions,
        IntrinsicsOptimOptions,
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
/// use vision_calibration::laserline_device::{run_calibration, LaserlineDeviceInput};
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
        InitOptions, LaserlineDeviceConfig, LaserlineDeviceExport, LaserlineDeviceInitConfig,
        LaserlineDeviceInput, LaserlineDeviceOptimizeConfig, LaserlineDeviceOutput,
        LaserlineDeviceProblem, LaserlineDeviceSolverConfig, LaserlineDeviceState, OptimizeOptions,
        run_calibration, step_init, step_optimize,
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
/// use vision_calibration::rig_extrinsics::{run_calibration, RigExtrinsicsInput};
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
        IntrinsicsOptimOptions,
        // Problem type and config
        RigExtrinsicsConfig,
        RigExtrinsicsExport,
        RigExtrinsicsInput,
        RigExtrinsicsProblem,
        RigExtrinsicsState,
        RigOptimOptions,
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
/// use vision_calibration::rig_handeye::{run_calibration, RigHandeyeInput};
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
        HandeyeOptimOptions,
        IntrinsicsInitOptions,
        IntrinsicsOptimOptions,
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
        RigOptimOptions,
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

/// Convenient re-exports for common use cases.
///
/// ```no_run
/// use vision_calibration::prelude::*;
/// ```
pub mod prelude {
    // Session framework
    pub use vision_calibration_pipeline::session::{CalibrationSession, ProblemType};

    // Problem types
    pub use vision_calibration_pipeline::laserline_device::LaserlineDeviceProblem;
    pub use vision_calibration_pipeline::planar_intrinsics::PlanarIntrinsicsProblem;
    pub use vision_calibration_pipeline::rig_extrinsics::RigExtrinsicsProblem;
    pub use vision_calibration_pipeline::rig_handeye::RigHandeyeProblem;
    pub use vision_calibration_pipeline::single_cam_handeye::SingleCamHandeyeProblem;

    // Pipeline functions
    pub use vision_calibration_pipeline::laserline_device::run_calibration as run_laserline_device;
    pub use vision_calibration_pipeline::planar_intrinsics::run_calibration as run_planar_intrinsics;
    pub use vision_calibration_pipeline::rig_extrinsics::run_calibration as run_rig_extrinsics;
    pub use vision_calibration_pipeline::rig_handeye::run_calibration as run_rig_handeye;
    pub use vision_calibration_pipeline::scheimpflug_intrinsics::run_calibration_direct as run_scheimpflug_intrinsics;
    pub use vision_calibration_pipeline::single_cam_handeye::run_calibration as run_single_cam_handeye;

    // Core types
    pub use vision_calibration_core::{
        BrownConrady5, Camera, CameraParams, CorrespondenceView, FxFyCxCySkew, IdentitySensor,
        Iso3, NoMeta, Pinhole, PinholeCamera, PlanarDataset, Pt2, Pt3, RigDataset, Vec2, Vec3,
        View,
    };

    // Common options
    pub use vision_calibration_optim::{BackendSolveOptions, HandEyeMode, RobustLoss};
}
