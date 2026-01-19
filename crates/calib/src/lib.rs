//! High-level entry crate for the `calibration-rs` toolbox.
//!
//! This crate provides **two complementary APIs** for camera calibration:
//!
//! ## 1. Session API (Structured Workflows)
//!
//! Use when you want:
//! - Type-safe, structured calibration workflows
//! - Artifact-based state management with branching support
//! - JSON checkpointing for session persistence
//!
//! ```ignore
//! use calib::session::{CalibrationSession, FilterOptions, ExportOptions};
//! use calib::planar_intrinsics::{
//!     PlanarIntrinsicsProblem, PlanarIntrinsicsObservations,
//! };
//! use calib::CorrespondenceView;
//!
//! let views: Vec<CorrespondenceView> = /* load calibration data */;
//!
//! let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
//! let obs_id = session.add_observations(PlanarIntrinsicsObservations { views });
//!
//! // Try different initialization strategies
//! let seed_a = session.run_init(obs_id, Default::default())?;
//!
//! // Optimize
//! let result_id = session.run_optimize(obs_id, seed_a, Default::default())?;
//!
//! // Filter outliers and re-optimize
//! let obs_filtered = session.run_filter_obs(obs_id, result_id, FilterOptions::default())?;
//! let seed_b = session.run_init(obs_filtered, Default::default())?;
//! let result2 = session.run_optimize(obs_filtered, seed_b, Default::default())?;
//!
//! // Export final results
//! let report = session.run_export(result2, ExportOptions::default())?;
//! ```
//!
//! ## 2. Imperative Function API (Custom Workflows)
//!
//! Use when you need:
//! - Full control over calibration workflow
//! - Ability to inspect intermediate results
//! - Custom composition of calibration steps
//!
//! ```ignore
//! use calib::helpers::{initialize_planar_intrinsics, optimize_planar_intrinsics_from_init};
//! use calib_linear::iterative_intrinsics::IterativeIntrinsicsOptions;
//! use calib::BackendSolveOptions;
//! use calib::PlanarIntrinsicsSolveOptions;
//!
//! let views: Vec<CorrespondenceView> = /* load calibration data */;
//!
//! // Step 1: Linear initialization
//! let init_opts = IterativeIntrinsicsOptions::default();
//! let init_result = initialize_planar_intrinsics(&views, &init_opts)?;
//!
//! // Inspect before committing to optimization
//! println!("Initial fx: {}, fy: {}", init_result.intrinsics.fx, init_result.intrinsics.fy);
//!
//! // Step 2: Non-linear optimization
//! let solve_opts = PlanarIntrinsicsSolveOptions::default();
//! let backend_opts = BackendSolveOptions::default();
//! let optim_result = optimize_planar_intrinsics_from_init(
//!     &views, &init_result, &solve_opts, &backend_opts
//! )?;
//!
//! println!("Final reprojection error: {:.2} px", optim_result.mean_reproj_error);
//! ```
//!
//! ## Module Organization
//!
//! - **[`session`]**: Type-safe calibration session framework
//! - **[`helpers`]**: Granular helper functions for common operations
//! - **[`planar_intrinsics`]**: Planar intrinsics calibration (Zhang's method)
//! - **[`core`]**: Math types, camera models, RANSAC primitives
//! - **[`linear`]**: Closed-form initialization algorithms
//! - **[`optim`]**: Non-linear least-squares optimization

/// Type-safe calibration session framework for structured workflows.
///
/// Provides artifact-based state management, branching workflows, and checkpointing.
pub mod session {
    pub use calib_pipeline::session::{
        ArtifactId, ArtifactKind, CalibrationSession, ExportOptions, FilterOptions, ProblemType,
        RunId, RunKind, RunRecord, SessionMetadata,
    };
}

/// Planar intrinsics calibration (Zhang's method with distortion).
pub mod planar_intrinsics {
    pub use calib_pipeline::planar_intrinsics::*;
}

/// Granular helper functions for custom calibration workflows.
pub mod helpers {
    pub use calib_pipeline::helpers::*;
}

/// Hand-eye calibration types.
pub mod handeye {
    pub use calib_pipeline::handeye::*;
}

/// Core math types, camera models, and RANSAC primitives.
pub mod core {
    pub use calib_core::*;
}

/// Deterministic synthetic data generation helpers.
pub mod synthetic {
    pub use calib_core::synthetic::*;
}

/// Closed-form initialization algorithms.
pub mod linear {
    pub use calib_linear::*;
}

/// Non-linear least-squares optimization.
pub mod optim {
    pub use calib_optim::*;
}

// Re-exports for convenience
pub use calib_core::{
    make_pinhole_camera, pinhole_camera_params, BrownConrady5, CameraParams, CorrespondenceView,
    FxFyCxCySkew, Iso3, PinholeCamera,
};

pub use calib_optim::{BackendSolveOptions, HandEyeMode, PlanarIntrinsicsSolveOptions, RobustLoss};

pub use calib_pipeline::{
    planar_init_seed_from_views, run_planar_intrinsics, PlanarDataset, PlanarIntrinsicsConfig,
    PlanarIntrinsicsParams, PlanarIntrinsicsReport,
};

/// Convenient re-exports for common use cases.
pub mod prelude {
    // Core types
    pub use crate::core::{
        BrownConrady5, Camera, CameraParams, FxFyCxCySkew, IdentitySensor, IntrinsicsParams, Iso3,
        Pinhole, Pt2, Pt3, Vec2, Vec3,
    };

    // Session API
    pub use crate::session::{
        ArtifactId, CalibrationSession, ExportOptions, FilterOptions, ProblemType,
    };

    // Planar intrinsics
    pub use crate::planar_intrinsics::{
        PlanarIntrinsicsInitOptions, PlanarIntrinsicsObservations, PlanarIntrinsicsOptimOptions,
        PlanarIntrinsicsProblem, PlanarIntrinsicsReport,
    };

    // Helper functions
    pub use crate::helpers::{
        initialize_planar_intrinsics, optimize_planar_intrinsics_from_init,
        PlanarIntrinsicsInitResult, PlanarIntrinsicsOptimResult,
    };

    // Common types
    pub use crate::{CorrespondenceView, PlanarIntrinsicsConfig};

    // Common options
    pub use crate::linear::distortion_fit::DistortionFitOptions;
    pub use crate::linear::iterative_intrinsics::IterativeIntrinsicsOptions;
    pub use crate::{BackendSolveOptions, PlanarIntrinsicsSolveOptions};
}
