//! Multiple-view geometry for calibrated cameras.
//!
//! This crate provides tools for two-view and multi-view geometry:
//!
//! - **Types**: [`Correspondence2D`], [`TriangulatedPoint`], matrix type aliases
//! - **Residuals**: algebraic, Sampson, geometric epipolar residuals and
//!   symmetric transfer error for homographies
//! - **Cheirality**: pose disambiguation from essential matrix decomposition
//! - **Pose recovery**: [`recover_relative_pose`] — end-to-end calibrated 2-view pose
//! - **Triangulation**: two-view triangulation with diagnostics
//!
//! # Low-level solvers (re-exported from `vision-geometry`)
//!
//! The following building-block solvers are re-exported for convenience:
//!
//! - [`essential_5point`] — 5-point essential matrix estimation
//! - [`fundamental_8point`], [`fundamental_7point`] — fundamental matrix estimation
//! - [`decompose_essential`] — essential matrix → candidate (R, t) pairs
//! - [`dlt_homography`] — normalized DLT homography estimation
//! - [`triangulate_point_linear`] — linear DLT triangulation
//!
//! # Coordinate conventions
//!
//! All calibrated functions expect **normalized camera coordinates** (after
//! applying `K⁻¹` to pixel coordinates). Pose outputs follow the `T_C_W`
//! convention: transform from world into camera frame.

pub mod cheirality;
pub mod degeneracy;
pub mod homography;
#[cfg(feature = "refine")]
pub mod refine;
pub mod pose_recovery;
pub mod residuals;
pub mod robust;
pub mod triangulation;
pub mod types;

// Re-export core types for convenience.
pub use homography::{HomographyDecomposition, decompose_homography, homography_from_pose_and_plane, homography_transfer, homography_transfer_inverse};
pub use pose_recovery::{RelativePose, recover_relative_pose};
pub use robust::{EssentialEstimate, HomographyEstimate, RobustRelativePose, estimate_essential, estimate_homography, recover_relative_pose_robust};
pub use types::{
    Correspondence2D, EssentialMatrix, FundamentalMatrix, HomographyMatrix, TriangulatedPoint,
};

// Re-export low-level solvers from vision-geometry.
pub use vision_geometry::camera_matrix::{
    CameraMatrixDecomposition, Mat34, decompose_camera_matrix, dlt_camera_matrix, rq_decompose,
};
pub use vision_geometry::epipolar::{
    decompose_essential, essential_5point, fundamental_7point, fundamental_8point,
    fundamental_8point_ransac,
};
pub use vision_geometry::homography::{dlt_homography, dlt_homography_ransac};
pub use vision_geometry::math::{normalize_points_2d, normalize_points_3d};
pub use vision_geometry::triangulation::triangulate_point_linear;
