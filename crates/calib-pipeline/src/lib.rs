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

// Re-export session types
pub use session::{
    Artifact, ArtifactId, ArtifactKind, CalibrationSession, ExportOptions, FilterOptions,
    ProblemType, RunId, RunKind, RunRecord, SessionMetadata,
};

// Re-export planar intrinsics
pub use planar_intrinsics::{
    planar_init_seed_from_views, run_planar_intrinsics, PlanarDataset, PlanarIntrinsicsConfig,
    PlanarIntrinsicsInitOptions, PlanarIntrinsicsInitial, PlanarIntrinsicsObservations,
    PlanarIntrinsicsOptimOptions, PlanarIntrinsicsOptimized, PlanarIntrinsicsParams,
    PlanarIntrinsicsProblem, PlanarIntrinsicsReport, PlanarIntrinsicsSolveOptions,
};

// Re-export from calib-core for convenience
pub use calib_core::{
    make_pinhole_camera, pinhole_camera_params, BrownConrady5, CameraParams, CorrespondenceView,
    FxFyCxCySkew, Iso3, PinholeCamera,
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
