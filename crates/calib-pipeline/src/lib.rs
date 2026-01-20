//! High-level camera calibration pipelines.
//!
//! This crate provides ready-to-use calibration workflows with multiple APIs:
//!
//! ## New Session API (v2) - Recommended
//!
//! The new session API uses a mutable state container with step functions.
//! This is the recommended approach for new code.
//!
//! ```ignore
//! use calib_pipeline::session::v2::CalibrationSession;
//! use calib_pipeline::planar_intrinsics::{
//!     PlanarIntrinsicsProblemV2, step_init, step_optimize, run_calibration,
//! };
//!
//! let mut session = CalibrationSession::<PlanarIntrinsicsProblemV2>::new();
//! session.set_input(dataset)?;
//!
//! // Option 1: Step-by-step control
//! step_init(&mut session, None)?;
//! step_optimize(&mut session, None)?;
//!
//! // Option 2: Pipeline function
//! // run_calibration(&mut session)?;
//!
//! let export = session.export()?;
//! ```
//!
//! ## Legacy Session API (v1) - Deprecated
//!
//! The original artifact-based DAG session API is still available but deprecated.
//!
//! ```ignore
//! use calib_pipeline::session::CalibrationSession;
//! use calib_pipeline::planar_intrinsics::{PlanarIntrinsicsConfig, PlanarIntrinsicsProblem};
//!
//! let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
//! let obs_id = session.add_observations(dataset);
//! let config = PlanarIntrinsicsConfig::default();
//! let init_id = session.run_init(obs_id, config.clone())?;
//! let result_id = session.run_optimize(obs_id, init_id, config)?;
//! ```
//!
//! ## Imperative API
//!
//! Direct function calls without session management.
//!
//! ```ignore
//! use calib_pipeline::planar_intrinsics::{run_planar_intrinsics, PlanarIntrinsicsConfig};
//!
//! let result = run_planar_intrinsics(&dataset, &PlanarIntrinsicsConfig::default())?;
//! ```

// Core session framework
pub mod session;

// Problem-specific modules
pub mod planar_intrinsics;

// ─────────────────────────────────────────────────────────────────────────────
// New Session API (v2) Re-exports - Recommended
// ─────────────────────────────────────────────────────────────────────────────

/// New session API with mutable state container (v2).
///
/// This module re-exports the v2 session infrastructure and planar intrinsics
/// step functions for convenient access.
pub mod v2 {

    // Session infrastructure
    pub use crate::session::v2::{
        CalibrationSession, ExportRecord, InvalidationPolicy, LogEntry, ProblemType,
        SessionMetadata,
    };

    // Planar intrinsics
    pub use crate::planar_intrinsics::{
        run_calibration, run_calibration_with_filtering, step_filter, step_init, step_optimize,
        FilterOptions, InitOptions, OptimizeOptions, PlanarConfig, PlanarExport,
        PlanarIntrinsicsProblemV2, PlanarState,
    };
}

// ─────────────────────────────────────────────────────────────────────────────
// Legacy Session API (v1) Re-exports - Deprecated
// ─────────────────────────────────────────────────────────────────────────────

// Re-export legacy session types (still available for backwards compatibility)
pub use session::{
    Artifact, ArtifactId, ArtifactKind, CalibrationSession, ExportOptions, FilterOptions,
    ProblemType, RunId, RunKind, RunRecord, SessionMetadata,
};

// Re-export planar intrinsics (including both old and new APIs)
pub use planar_intrinsics::{
    planar_init_seed_from_views, run_planar_intrinsics, PlanarIntrinsicsConfig,
    PlanarIntrinsicsEstimate, PlanarIntrinsicsParams, PlanarIntrinsicsProblem,
    PlanarIntrinsicsSolveOptions,
    // New v2 types also at top level for convenience
    PlanarConfig, PlanarIntrinsicsProblemV2, PlanarState,
};

// ─────────────────────────────────────────────────────────────────────────────
// Re-exports from other crates
// ─────────────────────────────────────────────────────────────────────────────

// Re-export from calib-core for convenience
pub use calib_core::{
    make_pinhole_camera, pinhole_camera_params, BrownConrady5, CameraParams, CorrespondenceView,
    FxFyCxCySkew, Iso3, NoMeta, PinholeCamera, PlanarDataset, Pt2, Pt3, View,
};

// Re-export from calib-optim for convenience
pub use calib_optim::{BackendSolveOptions, HandEyeMode, RobustLoss};

// Hand-eye types re-exported from calib-optim
pub mod handeye {
    //! Hand-eye calibration types re-exported from calib-optim.
    pub use calib_optim::{
        optimize_handeye, HandEyeDataset, HandEyeEstimate, HandEyeParams, HandEyeSolveOptions,
        RigViewObs, RobotPoseMeta, View,
    };
}
