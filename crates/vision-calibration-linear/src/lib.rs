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
//! - Equidistant (fisheye) planar intrinsics initialization from bearings
//! - Planar pose from homography + intrinsics, or from bearing vectors
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
//!
//! Zhang's method recovers the intrinsics from the homographies of three or
//! more views of a plane:
//!
//! ```
//! use vision_calibration_core::synthetic::{planar, poses};
//! use vision_calibration_core::{BrownConrady5, FxFyCxCy, Pt2, make_pinhole_camera};
//! use vision_calibration_linear::zhang_intrinsics::estimate_intrinsics_from_homographies;
//! use vision_geometry::homography::dlt_homography;
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//!
//! let truth = FxFyCxCy { fx: 800.0, fy: 780.0, cx: 640.0, cy: 360.0 };
//! let camera = make_pinhole_camera(truth, BrownConrady5::default());
//! let board = planar::grid_points(8, 6, 0.04);
//! let specs = poses::seeded_board_pose_specs(6, 0.35, 3);
//! let mut homographies = Vec::new();
//! for pose in poses::centered_board_poses(&specs, (0.14, 0.10), 0.6) {
//!     let view = planar::project_view_all(&camera, &pose, &board)?;
//!     let on_board: Vec<Pt2> = view.points_3d.iter().map(|p| Pt2::new(p.x, p.y)).collect();
//!     homographies.push(dlt_homography(&on_board, &view.points_2d)?);
//! }
//!
//! let k = estimate_intrinsics_from_homographies(&homographies)?;
//! assert!((k.fx - truth.fx).abs() < 1e-6 && (k.cy - truth.cy).abs() < 1e-6);
//! # Ok(())
//! # }
//! ```
//!
//! Non-linear refinement and full pipelines live in `vision-calibration-optim` and
//! `vision-calibration-pipeline` and are re-exported via the top-level `vision-calibration` crate.

/// Typed error enum for this crate.
pub mod error;

pub use error::Error;

pub mod distortion_fit;
pub mod equidistant_init;
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
    pub use crate::equidistant_init::{
        EquidistantInitOptions, EquidistantIntrinsicsLinearInit, estimate_equidistant_intrinsics,
    };
    pub use crate::iterative_intrinsics::{
        IterativeIntrinsicsOptions, estimate_intrinsics_iterative,
    };
    pub use crate::planar_pose::{estimate_planar_pose_from_bearings, estimate_planar_pose_from_h};
    pub use crate::scheimpflug_init::{
        ScheimpflugIntrinsicsInitOptions, ScheimpflugIntrinsicsLinearInit,
        estimate_scheimpflug_intrinsics_iterative,
    };
    pub use crate::zhang_intrinsics::{
        PlanarIntrinsicsLinearInit, estimate_intrinsics_from_homographies,
    };
    pub use vision_geometry::homography::dlt_homography;
}

#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeDoctests;
