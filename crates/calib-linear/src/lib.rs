//! Linear and closed-form calibration building blocks.
//!
//! This crate provides deterministic, allocation-light solvers that are commonly
//! used as **initialization** steps before non-linear refinement. The focus is
//! on classic DLT-style methods and minimal solvers with clear numerical
//! assumptions.
//!
//! # Algorithms
//! - Homography estimation (normalized DLT, optional RANSAC)
//! - Planar intrinsics (Zhang method from multiple homographies)
//! - Distortion estimation (Brown-Conrady from homography residuals)
//! - Iterative intrinsics refinement (alternating K and distortion estimation)
//! - Planar pose from homography + intrinsics
//! - Fundamental matrix: 8-point (normalized) and 7-point
//! - Essential matrix: 5-point minimal solver + decomposition to (R, t)
//! - Camera pose (PnP): DLT, P3P, EPnP, and DLT-in-RANSAC
//! - Camera matrix DLT and RQ decomposition
//! - Linear triangulation (DLT)
//! - Multi-camera rig extrinsics and hand-eye calibration
//! - Linescan laser plane estimation (SVD-based from ray-plane intersections)
//!
//! # Coordinate conventions
//! - Most solvers accept **pixel coordinates** directly.
//! - Essential matrix and P3P/EPnP expect **calibrated** image points; these
//!   functions take intrinsics or assume input is already normalized.
//! - Pose outputs are `T_C_W`: transform from world/board coordinates into the
//!   camera frame.
//!
//! # Example
//! ```no_run
//! use calib_linear::HomographySolver;
//! use calib_core::Pt2;
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
//! let h = HomographySolver::dlt(&world, &image).expect("homography failed");
//! println!("H = {h}");
//! ```
//!
//! Non-linear refinement and full pipelines live in `calib-optim` and
//! `calib-pipeline` and are re-exported via the top-level `calib` crate.

pub mod camera_matrix;
pub mod distortion_fit;
pub mod epipolar;
pub mod extrinsics;
pub mod handeye;
pub mod homography;
pub mod iterative_intrinsics;
pub mod linescan;
pub mod math;
pub mod planar_pose;
pub mod pnp;
pub mod triangulation;
pub mod zhang_intrinsics;

pub use camera_matrix::*;
pub use distortion_fit::*;
pub use epipolar::*;
pub use extrinsics::*;
pub use handeye::*;
pub use homography::*;
pub use iterative_intrinsics::*;
pub use linescan::*;
pub use math::*;
pub use planar_pose::*;
pub use pnp::*;
pub use triangulation::*;
pub use zhang_intrinsics::*;

pub mod prelude {
    pub use crate::homography::dlt_homography;
    pub use crate::iterative_intrinsics::{
        estimate_intrinsics_iterative, IterativeIntrinsicsOptions,
    };
    pub use crate::planar_pose::estimate_planar_pose_from_h;
}
