//! Perspective-n-Point (PnP) solvers for camera pose estimation.
//!
//! Includes:
//! - DLT (linear) pose estimation with normalization.
//! - P3P minimal solver (3 points, multiple solutions).
//! - EPnP (control-point formulation) for 4+ points; a planar point set is
//!   solved from its plane homography.
//! - DLT wrapped in RANSAC for outlier rejection.
//!
//! All methods estimate a pose `T_C_W`: transform from world coordinates into
//! the camera frame.

mod dlt;
mod epnp;
mod p3p;
mod planar;
mod pose_utils;
mod ransac;

// Re-export public API
pub use dlt::dlt;
pub use epnp::epnp;
pub use p3p::p3p;
pub use ransac::{PoseEstimate, dlt_ransac};
