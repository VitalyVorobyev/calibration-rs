//! Linear and closed-form calibration building blocks.
//!
//! This crate provides deterministic, allocation-light solvers that are commonly
//! used as **initialization** steps before non-linear refinement. The focus is
//! on classic DLT-style methods and minimal solvers with clear numerical
//! assumptions.
//!
//! # Algorithms
//! - Planar intrinsics (Zhang method from multiple homographies)
//! - Distortion estimation (Brown-Conrady from homography residuals)
//! - Iterative intrinsics refinement (alternating K and distortion estimation)
//! - Tilt-aware Scheimpflug planar intrinsics initialization
//! - Planar pose from homography + intrinsics
//! - Camera pose (PnP): DLT, P3P, EPnP, and DLT-in-RANSAC
//! - Multi-camera rig extrinsics and hand-eye calibration
//! - Laserline plane estimation (SVD-based from ray-plane intersections)
//!
//! Two-view geometric solvers (homography, fundamental/essential matrix,
//! camera matrix DLT and RQ decomposition, linear triangulation) now live in
//! the `vision-geometry` crate.
//!
//! # Coordinate conventions
//! - Most solvers accept **pixel coordinates** directly.
//! - P3P/EPnP expect **calibrated** image points; these
//!   functions take intrinsics or assume input is already normalized.
//! - Pose outputs are `T_C_W`: transform from world/board coordinates into the
//!   camera frame.
//!
//! # Example
//! ```no_run
//! use vision_geometry::homography::dlt_homography;
//! use vision_calibration_core::Pt2;
//!
//! let world = vec![
//!     Pt2::new(0.0, 0.0),
//!     Pt2::new(1.0, 0.0),
//!     Pt2::new(1.0, 1.0),
//!     Pt2::new(0.0, 1.0),
//! ];
//! let image = vec![
//!     Pt2::new(120.0, 200.0),
//!     Pt2::new(220.0, 198.0),
//!     Pt2::new(225.0, 300.0),
//!     Pt2::new(118.0, 302.0),
//! ];
//!
//! let h = dlt_homography(&world, &image).expect("homography failed");
//! println!("H = {h}");
//! ```
//!
//! Non-linear refinement and full pipelines live in `vision-calibration-optim` and
//! `vision-calibration-pipeline` and are re-exported via the top-level `vision-calibration` crate.

/// Typed error enum for this crate.
pub mod error;

pub use error::Error;

pub mod distortion_fit;
pub mod extrinsics;
pub mod handeye;
pub mod iterative_intrinsics;
pub mod laserline;
pub mod math;
pub mod planar_pose;
pub mod pnp;
pub mod scheimpflug_init;
pub mod zhang_intrinsics;

/// Minimal imports for common planar initialization workflows.
pub mod prelude {
    pub use crate::distortion_fit::DistortionFitOptions;
    pub use crate::iterative_intrinsics::{
        IterativeIntrinsicsOptions, estimate_intrinsics_iterative,
    };
    pub use crate::planar_pose::estimate_planar_pose_from_h;
    pub use crate::scheimpflug_init::{
        ScheimpflugIntrinsicsInitOptions, ScheimpflugIntrinsicsLinearInit,
        estimate_scheimpflug_intrinsics_iterative,
    };
    pub use crate::zhang_intrinsics::{
        PlanarIntrinsicsLinearInit, estimate_intrinsics_from_homographies,
    };
    pub use vision_geometry::homography::dlt_homography;
}
