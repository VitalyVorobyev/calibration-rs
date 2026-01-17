//! High-level entry crate for the `calibration-rs` toolbox.
//!
//! This crate provides **two complementary APIs** for camera calibration:
//!
//! ## 1. Session API (Structured Workflows)
//!
//! Use when you want:
//! - Type-safe, structured calibration workflows
//! - Automatic state management and checkpointing
//! - Enforced stage transitions (Uninitialized → Initialized → Optimized → Exported)
//!
//! ```no_run
//! use calib::session::{CalibrationSession, PlanarIntrinsicsProblem, PlanarIntrinsicsObservations};
//! use calib::pipeline::PlanarViewData;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let views: Vec<PlanarViewData> = /* load calibration data */
//! # vec![];
//!
//! let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
//! session.set_observations(PlanarIntrinsicsObservations { views });
//!
//! // Initialize with linear solver
//! session.initialize(Default::default())?;
//!
//! // Can checkpoint here
//! let json = session.to_json()?;
//! std::fs::write("checkpoint.json", json)?;
//!
//! // Optimize with non-linear refinement
//! session.optimize(Default::default())?;
//!
//! // Export final results
//! let report = session.export()?;
//! println!("Final cost: {}", report.report.final_cost);
//! # Ok(())
//! # }
//! ```
//!
//! ## 2. Imperative Function API (Custom Workflows)
//!
//! Use when you need:
//! - Full control over calibration workflow
//! - Ability to inspect intermediate results
//! - Custom composition of calibration steps
//! - Integration into larger systems
//!
//! ### Using High-Level Helper Functions
//!
//! ```no_run
//! use calib::helpers::{initialize_planar_intrinsics, optimize_planar_intrinsics_from_init};
//! use calib::linear::iterative_intrinsics::IterativeIntrinsicsOptions;
//! use calib::linear::distortion_fit::DistortionFitOptions;
//! use calib::optim::planar_intrinsics::PlanarIntrinsicsSolveOptions;
//! use calib::optim::backend::BackendSolveOptions;
//! use calib::pipeline::PlanarViewData;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let views: Vec<PlanarViewData> = /* load calibration data */
//! # vec![];
//!
//! // Step 1: Linear initialization
//! let init_opts = IterativeIntrinsicsOptions {
//!     iterations: 2,
//!     distortion_opts: DistortionFitOptions {
//!         fix_k3: true,
//!         fix_tangential: false,
//!         iters: 8,
//!     },
//!     zero_skew: true,
//! };
//! let init_result = initialize_planar_intrinsics(&views, &init_opts)?;
//!
//! // Inspect before committing to optimization
//! println!("Initial fx: {}, fy: {}", init_result.intrinsics.fx, init_result.intrinsics.fy);
//!
//! // Step 2: Non-linear optimization (if init looks good)
//! let solve_opts = PlanarIntrinsicsSolveOptions::default();
//! let backend_opts = BackendSolveOptions::default();
//! let optim_result = optimize_planar_intrinsics_from_init(
//!     &views,
//!     &init_result,
//!     &solve_opts,
//!     &backend_opts
//! )?;
//!
//! println!("Final reprojection error: {:.2} px", optim_result.mean_reproj_error);
//! # Ok(())
//! # }
//! ```
//!
//! ### Using Low-Level Building Blocks
//!
//! For maximum control, directly access linear and optimization modules:
//!
//! ```no_run
//! use calib::linear::{homography, zhang_intrinsics};
//! use calib::optim::planar_intrinsics::optimize_planar_intrinsics;
//! use calib::core::{Pt2, Pt3};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let world_points: Vec<Pt2> = /* 2D points on planar pattern */
//! # vec![];
//! let image_points: Vec<Pt2> = /* corresponding 2D pixel coordinates */
//! # vec![];
//!
//! // Compute homography (planar pattern uses 2D coordinates)
//! let H = homography::dlt_homography(&world_points, &image_points)?;
//!
//! // Estimate intrinsics from multiple homographies
//! let homographies = vec![H];
//! let K = zhang_intrinsics::estimate_intrinsics_from_homographies(&homographies)?;
//!
//! println!("Estimated K: {:?}", K);
//! # Ok(())
//! # }
//! ```
//!
//! ## Module Organization
//!
//! - **[`session`]**: Type-safe calibration session framework
//! - **[`helpers`]**: Granular helper functions for common operations
//! - **[`pipeline`]**: All-in-one convenience functions (original API)
//! - **[`core`]**: Math types, camera models, RANSAC primitives
//! - **[`linear`]**: Closed-form initialization algorithms
//! - **[`optim`]**: Non-linear least-squares optimization
//! - **[`prelude`]**: Convenient re-exports for common use cases
//!
//! ## Stability
//!
//! The `calib` crate is the public compatibility boundary. Lower-level crates are
//! intended for advanced usage and may evolve more quickly.

/// Type-safe calibration session framework for structured workflows.
///
/// Provides state management, checkpointing, and enforced stage transitions.
pub mod session {
    pub use calib_pipeline::session::{
        CalibrationSession, ProblemType, SessionMetadata, SessionStage,
    };

    // Problem types
    pub mod problem_types {
        pub use calib_pipeline::session::problem_types::*;
    }

    // Re-export common problem types at top level for convenience
    pub use calib_pipeline::session::problem_types::{
        HandEyeModeConfig, HandEyeSingleInitOptions, HandEyeSingleObservations,
        HandEyeSingleOptimOptions, HandEyeSingleProblem, PlanarIntrinsicsInitOptions,
        PlanarIntrinsicsObservations, PlanarIntrinsicsOptimOptions, PlanarIntrinsicsProblem,
        RigExtrinsicsInitOptions, RigExtrinsicsObservations, RigExtrinsicsOptimOptions,
        RigExtrinsicsProblem,
    };
}

/// Granular helper functions for custom calibration workflows.
///
/// These functions bridge between linear initialization and non-linear optimization,
/// allowing you to inspect intermediate results and compose custom workflows.
pub mod helpers {
    pub use calib_pipeline::helpers::*;
}

/// All-in-one convenience functions for standard calibration tasks.
///
/// Use these when you want a simple, single-call solution without managing state.
pub mod pipeline {
    pub use calib_pipeline::{
        handeye, handeye_single, run_planar_intrinsics, run_rig_extrinsics, HandEyeMode,
        PlanarIntrinsicsConfig, PlanarIntrinsicsInput, PlanarIntrinsicsReport, PlanarViewData,
        RigCameraViewData, RigExtrinsicsConfig, RigExtrinsicsInitOptions, RigExtrinsicsInput,
        RigExtrinsicsOptimOptions, RigExtrinsicsReport, RigViewData, RobustLossConfig,
    };
}

/// Core math types, camera models, and RANSAC primitives.
///
/// This module contains the fundamental building blocks used throughout the library.
pub mod core {
    pub use calib_core::*;
}

/// Closed-form initialization algorithms (Zhang, PnP, homography, etc.).
///
/// Use these for linear initialization before non-linear refinement.
pub mod linear {
    pub use calib_linear::*;
}

/// Non-linear least-squares optimization problems and backends.
///
/// Includes planar intrinsics, hand-eye, rig extrinsics, and linescan bundle refinement.
pub mod optim {
    pub use calib_optim::*;
}

/// Convenient re-exports for common use cases.
///
/// Import with `use calib::prelude::*;` to get started quickly.
pub mod prelude {
    // Common types
    pub use crate::core::{
        BrownConrady5, Camera, CameraParams, FxFyCxCySkew, IdentitySensor, IntrinsicsParams, Iso3,
        Pinhole, Pt2, Pt3, Vec2, Vec3,
    };

    // Session API
    pub use crate::session::{
        CalibrationSession, PlanarIntrinsicsObservations, PlanarIntrinsicsProblem, ProblemType,
        RigExtrinsicsObservations, RigExtrinsicsProblem, SessionStage,
    };

    // Helper functions
    pub use crate::helpers::{
        initialize_planar_intrinsics, optimize_planar_intrinsics_from_init,
        PlanarIntrinsicsInitResult, PlanarIntrinsicsOptimResult,
    };

    // Pipeline types
    pub use crate::pipeline::{
        PlanarIntrinsicsConfig, PlanarIntrinsicsInput, PlanarIntrinsicsReport, PlanarViewData,
        RigExtrinsicsConfig, RigExtrinsicsInput, RigExtrinsicsReport, RigViewData,
    };

    // Common options
    pub use crate::linear::distortion_fit::DistortionFitOptions;
    pub use crate::linear::iterative_intrinsics::IterativeIntrinsicsOptions;
    pub use crate::optim::backend::BackendSolveOptions;
    pub use crate::optim::planar_intrinsics::PlanarIntrinsicsSolveOptions;
}
