//! High-level camera calibration pipelines.
//!
//! This crate provides ready-to-use calibration workflows with two complementary APIs:
//!
//! - **Session API**: Structured workflows with artifact management and checkpointing
//! - **Imperative API**: Direct access to pipeline functions for custom workflows
//!
//! # Session API
//!
//! ```ignore
//! use calib_pipeline::session::CalibrationSession;
//! use calib_pipeline::planar_intrinsics::{
//!     PlanarIntrinsicsProblem, PlanarIntrinsicsObservations,
//! };
//!
//! let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
//! let obs_id = session.add_observations(PlanarIntrinsicsObservations { views });
//! let init_id = session.run_init(obs_id, Default::default())?;
//! let result_id = session.run_optimize(obs_id, init_id, Default::default())?;
//! let report = session.run_export(result_id, Default::default())?;
//! ```

// Core session framework
pub mod session;

// Problem-specific modules
pub mod planar_intrinsics;

// Shared helpers
pub mod helpers;

// Legacy modules - temporarily disabled during refactoring
// TODO: Re-enable and update these modules after core session is stable
// mod handeye_single;
// mod rig_extrinsics;
// mod rig_handeye;

// Re-export session types
pub use session::{
    Artifact, ArtifactId, ArtifactKind, CalibrationSession, ExportOptions, FilterOptions,
    ProblemType, RunId, RunKind, RunRecord, SessionMetadata,
};

// Re-export planar intrinsics
pub use planar_intrinsics::{
    PlanarDataset, PlanarIntrinsicsConfig, PlanarIntrinsicsInitOptions, PlanarIntrinsicsInitial,
    PlanarIntrinsicsObservations, PlanarIntrinsicsOptimOptions, PlanarIntrinsicsOptimized,
    PlanarIntrinsicsParams, PlanarIntrinsicsProblem, PlanarIntrinsicsReport,
    PlanarIntrinsicsSolveOptions, planar_init_seed_from_views, run_planar_intrinsics,
};

// Re-export shared helpers
pub use helpers::{
    initialize_planar_intrinsics, optimize_planar_intrinsics_from_init,
    PlanarIntrinsicsInitResult, PlanarIntrinsicsOptimResult,
};

// Re-export from calib-core for convenience
pub use calib_core::{
    make_pinhole_camera, pinhole_camera_params, BrownConrady5, CameraParams, CorrespondenceView,
    FxFyCxCySkew, Iso3, PinholeCamera,
};

// Re-export from calib-optim for convenience
pub use calib_optim::{BackendSolveOptions, RobustLoss};

// Re-export HandEyeMode from calib-optim
pub use calib_optim::HandEyeMode;

// Re-export handeye types from calib-optim (for tests that use them directly)
pub mod handeye {
    //! Hand-eye calibration types re-exported from calib-optim.
    pub use calib_optim::{
        optimize_handeye, HandEyeDataset, HandEyeEstimate, HandEyeParams, HandEyeSolveOptions,
        RobotPoseMeta, RigViewObs, View,
    };
}
