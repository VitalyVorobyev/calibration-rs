//! Linear and closed-form calibration building blocks.
//!
//! This crate contains small, self-contained solvers that implement classic
//! linear initialisation steps used in camera and rig calibration, such as:
//! - homography estimation,
//! - Zhang-style planar intrinsics,
//! - planar pose from homography,
//! - multi-camera rig extrinsics,
//! - hand–eye calibration.
//!
//! Non-linear refinement and full pipelines live in `calib-optim` and
//! `calib-pipeline` and are re-exported via the top-level `calib` crate.

pub mod epipolar;
pub mod extrinsics;
pub mod handeye;
pub mod homography;
pub mod planar_pose;
pub mod pnp;
pub mod zhang_intrinsics;

pub use epipolar::*;
pub use extrinsics::*;
pub use handeye::*;
pub use homography::*;
pub use planar_pose::*;
pub use pnp::*;
pub use zhang_intrinsics::*;
