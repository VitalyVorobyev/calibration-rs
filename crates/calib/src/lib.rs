//! High-level entry crate for the `calibration-rs` camera calibration library.
//!
//! This crate provides a unified API for camera calibration workflows:
//! - Single-camera intrinsics calibration (Zhang's method with distortion)
//! - Single-camera hand-eye calibration (camera on robot arm)
//! - Multi-camera rig extrinsics calibration
//! - Multi-camera rig + hand-eye calibration
//!
//! # Quick Start
//!
//! ```ignore
//! use calib::prelude::*;
//! use calib::planar_intrinsics::{step_init, step_optimize};
//!
//! // Create calibration session
//! let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
//! session.set_input(dataset)?;
//!
//! // Option 1: Step-by-step (recommended for inspection)
//! step_init(&mut session, None)?;
//! println!("Initial fx: {}", session.state.initial_intrinsics.unwrap().fx);
//! step_optimize(&mut session, None)?;
//!
//! // Option 2: Pipeline function (convenience)
//! // run_planar_intrinsics(&mut session)?;
//!
//! // Export results
//! let result = session.export()?;
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
//! All calibration workflows use the session API with this pattern:
//!
//! ```ignore
//! // 1. Create session for problem type
//! let mut session = CalibrationSession::<ProblemType>::new();
//!
//! // 2. Set input data
//! session.set_input(input)?;
//!
//! // 3. Optionally configure
//! session.update_config(|c| c.max_iters = 100)?;
//!
//! // 4. Run steps or pipeline
//! step_init(&mut session, None)?;
//! step_optimize(&mut session, None)?;
//! // OR: run_calibration(&mut session)?;
//!
//! // 5. Export results
//! let export = session.export()?;
//!
//! // 6. Optionally checkpoint
//! let json = session.to_json()?;
//! ```
//!
//! # Available Problem Types
//!
//! | Problem Type | Input | Steps |
//! |--------------|-------|-------|
//! | `PlanarIntrinsicsProblem` | `PlanarDataset` | init → optimize |
//! | `SingleCamHandeyeProblemV2` | `SingleCamHandeyeInput` | intrinsics_init → intrinsics_optim → handeye_init → handeye_optim |
//! | `RigExtrinsicsProblem` | `RigExtrinsicsInput` | intrinsics_init_all → intrinsics_optim_all → rig_init → rig_optim |
//! | `RigHandeyeProblem` | `RigHandeyeInput` | (all 6 steps) |

// ═══════════════════════════════════════════════════════════════════════════════
// Session Framework
// ═══════════════════════════════════════════════════════════════════════════════

/// Session framework for structured calibration workflows.
///
/// Provides mutable state containers, step functions, and JSON checkpointing.
pub mod session {
    pub use calib_pipeline::session::{
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
/// ```ignore
/// use calib::prelude::*;
/// use calib::planar_intrinsics::{step_init, step_optimize, run_calibration};
///
/// let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
/// session.set_input(dataset)?;
/// run_calibration(&mut session)?;
/// let result = session.export()?;
/// ```
pub mod planar_intrinsics {
    pub use calib_pipeline::planar_intrinsics::{
        // Step functions
        run_calibration,
        run_calibration_with_filtering,
        step_filter,
        step_init,
        step_optimize,
        // Step options
        FilterOptions,
        InitOptions,
        OptimizeOptions,
        // Problem type and config
        PlanarConfig,
        PlanarExport,
        // Re-exports from calib-optim
        PlanarIntrinsicsEstimate,
        PlanarIntrinsicsParams,
        PlanarIntrinsicsProblem,
        PlanarIntrinsicsSolveOptions,
        PlanarState,
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
/// ```ignore
/// use calib::prelude::*;
/// use calib::single_cam_handeye::{run_calibration, SingleCamHandeyeInput};
///
/// let mut session = CalibrationSession::<SingleCamHandeyeProblemV2>::new();
/// session.set_input(input)?;
/// run_calibration(&mut session)?;
/// let result = session.export()?;
/// ```
pub mod single_cam_handeye {
    pub use calib_pipeline::single_cam_handeye::{
        // Step functions
        run_calibration,
        step_handeye_init,
        step_handeye_optimize,
        step_intrinsics_init,
        step_intrinsics_optimize,
        // Step options
        HandeyeInitOptions,
        HandeyeOptimOptions,
        IntrinsicsInitOptions,
        IntrinsicsOptimOptions,
        // Problem type and config
        SingleCamHandeyeConfig,
        SingleCamHandeyeExport,
        SingleCamHandeyeInput,
        SingleCamHandeyeProblemV2,
        SingleCamHandeyeState,
        SingleCamHandeyeView,
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
/// ```ignore
/// use calib::prelude::*;
/// use calib::rig_extrinsics::{run_calibration, RigExtrinsicsInput};
///
/// let mut session = CalibrationSession::<RigExtrinsicsProblem>::new();
/// session.set_input(input)?;
/// run_calibration(&mut session)?;
/// let result = session.export()?;
/// ```
pub mod rig_extrinsics {
    pub use calib_pipeline::rig_extrinsics::{
        // Step functions
        run_calibration,
        step_intrinsics_init_all,
        step_intrinsics_optimize_all,
        step_rig_init,
        step_rig_optimize,
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
/// ```ignore
/// use calib::prelude::*;
/// use calib::rig_handeye::{run_calibration, RigHandeyeInput};
///
/// let mut session = CalibrationSession::<RigHandeyeProblem>::new();
/// session.set_input(input)?;
/// run_calibration(&mut session)?;
/// let result = session.export()?;
/// ```
pub mod rig_handeye {
    pub use calib_pipeline::rig_handeye::{
        // Step functions
        run_calibration,
        step_handeye_init,
        step_handeye_optimize,
        step_intrinsics_init_all,
        step_intrinsics_optimize_all,
        step_rig_init,
        step_rig_optimize,
        // Step options
        HandeyeInitOptions,
        HandeyeOptimOptions,
        IntrinsicsInitOptions,
        IntrinsicsOptimOptions,
        // Problem type and config
        RigHandeyeConfig,
        RigHandeyeExport,
        RigHandeyeInput,
        RigHandeyeProblem,
        RigHandeyeState,
        RigOptimOptions,
    };
}

// ═══════════════════════════════════════════════════════════════════════════════
// Foundation Crates (Advanced Users)
// ═══════════════════════════════════════════════════════════════════════════════

/// Core math types, camera models, and RANSAC primitives.
///
/// Re-exports everything from `calib_core`.
pub mod core {
    pub use calib_core::*;
}

/// Closed-form initialization algorithms.
///
/// Includes homography estimation, Zhang's method, PnP solvers,
/// triangulation, hand-eye solvers, and more.
///
/// Re-exports everything from `calib_linear`.
pub mod linear {
    pub use calib_linear::*;
}

/// Non-linear optimization with backend-agnostic IR.
///
/// Includes optimization problems, factors, and solver backends.
///
/// Re-exports everything from `calib_optim`.
pub mod optim {
    pub use calib_optim::*;
}

/// Deterministic synthetic data generation for testing.
///
/// Provides builders for creating synthetic calibration datasets.
pub mod synthetic {
    pub use calib_core::synthetic::*;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Hand-Eye Calibration Types (Direct Access)
// ═══════════════════════════════════════════════════════════════════════════════

/// Hand-eye calibration types re-exported from calib-optim.
///
/// Use this module for direct access to hand-eye optimization without
/// the session framework.
pub mod handeye {
    pub use calib_optim::{
        optimize_handeye, HandEyeDataset, HandEyeEstimate, HandEyeParams, HandEyeSolveOptions,
        RigViewObs, RobotPoseMeta, View,
    };
}

// ═══════════════════════════════════════════════════════════════════════════════
// Convenience Re-exports (Top-Level)
// ═══════════════════════════════════════════════════════════════════════════════

// Session framework
pub use calib_pipeline::{CalibrationSession, ProblemType};

// Problem types
pub use calib_pipeline::{
    PlanarIntrinsicsProblem, RigExtrinsicsProblem, RigHandeyeProblem, SingleCamHandeyeProblemV2,
};

// Pipeline functions
pub use calib_pipeline::{
    run_planar_intrinsics, run_rig_extrinsics, run_rig_handeye, run_single_cam_handeye,
};

// Core types
pub use calib_core::{
    make_pinhole_camera, pinhole_camera_params, BrownConrady5, Camera, CameraParams,
    CorrespondenceView, FxFyCxCySkew, IdentitySensor, Iso3, NoMeta, Pinhole, PinholeCamera,
    PlanarDataset, Pt2, Pt3, RigDataset, Vec2, Vec3, View,
};

// Common options
pub use calib_optim::{BackendSolveOptions, HandEyeMode, RobustLoss};

// ═══════════════════════════════════════════════════════════════════════════════
// Prelude (Quick Start)
// ═══════════════════════════════════════════════════════════════════════════════

/// Convenient re-exports for common use cases.
///
/// ```ignore
/// use calib::prelude::*;
/// ```
pub mod prelude {
    // Session framework
    pub use crate::session::{CalibrationSession, ProblemType};

    // Problem types
    pub use crate::{
        PlanarIntrinsicsProblem, RigExtrinsicsProblem, RigHandeyeProblem, SingleCamHandeyeProblemV2,
    };

    // Pipeline functions
    pub use crate::{
        run_planar_intrinsics, run_rig_extrinsics, run_rig_handeye, run_single_cam_handeye,
    };

    // Core types
    pub use crate::{
        BrownConrady5, Camera, CameraParams, CorrespondenceView, FxFyCxCySkew, IdentitySensor,
        Iso3, NoMeta, Pinhole, PinholeCamera, PlanarDataset, Pt2, Pt3, RigDataset, Vec2, Vec3,
        View,
    };

    // Common options
    pub use crate::{BackendSolveOptions, HandEyeMode, RobustLoss};
}
