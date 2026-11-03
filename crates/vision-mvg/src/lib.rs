//! Multiple-view geometry for calibrated cameras.
//!
//! This crate provides tools for two-view and multi-view geometry:
//!
//! - **Types**: [`PixelMatch`] (a pair of pixels with their noise), [`Correspondence2D`]
//!   (a pair of plain points), [`TriangulatedPoint`], matrix type aliases
//! - **Residuals**: algebraic, Sampson, geometric epipolar residuals and
//!   symmetric transfer error for homographies
//! - **Degeneracy**: pure-rotation, planar-scene and poor-baseline detection
//! - **Pose recovery**: [`recover_relative_pose`] — end-to-end calibrated 2-view pose
//! - **Triangulation**: [`triangulation::triangulate`] — N-view triangulation of
//!   pixel tracks from a calibration, refined on stated pixel noise, with
//!   diagnostics and the points' joint covariance (the calibration's share
//!   included)
//! - **Stereo**: [`rectify`] — rectification of two rig cameras from the
//!   calibration for any camera model, the disparity-to-depth matrix, points
//!   with the covariance of their disparity, and measurements that carry the
//!   calibration's uncertainty
//! - **Measurements**: [`metrology`] — distances, angles, plane fits and
//!   point-to-plane distances over triangulated points, each with the σ from
//!   the points' joint covariance
//! - **Bundle adjustment**: [`bundle_adjust()`](bundle_adjust::bundle_adjust) — capture poses and points
//!   refined together from a calibration, with a stated gauge (capture 0 and
//!   a [`Scale`]), robust losses and the joint covariance; the calibration
//!   enters as consider parameters or under a prior
//!
//! # Low-level solvers (re-exported from `vision-geometry`)
//!
//! The following building-block solvers are re-exported for convenience:
//!
//! - [`essential_5point`] — 5-point essential matrix estimation on bearings
//! - [`fundamental_8point`], [`fundamental_7point`] — fundamental matrix estimation
//! - [`decompose_essential`] — essential matrix → candidate (R, t) pairs
//! - [`dlt_homography`] — normalized DLT homography estimation
//! - [`triangulate_point_linear`] — linear DLT triangulation of bearings
//!
//! # Coordinate conventions
//!
//! Pixels in, bearings inside. The calibrated functions take a
//! [`CalibratedRig`](vision_calibration_core::CalibratedRig) (the cameras, where
//! each sits in the rig, and the uncertainty of their parameters) and the
//! observed **pixels**. Each pixel is unprojected through its camera's own
//! model into a *bearing*, a unit ray, and the closed-form steps work on
//! bearings, so every camera model is served, including a fisheye that sees
//! beyond 90° from its axis, where normalized coordinates do not exist.
//!
//! Poses are named `<frame>_se3_<frame>` and map the right frame into the left:
//! `second_se3_first` takes first-camera coordinates to second-camera
//! coordinates; `cam_se3_world` takes the world into a camera.
//!
//! # Example
//!
//! Two views of twelve points through a camera with lens distortion: the
//! relative pose comes back with its translation known up to scale, and the
//! points triangulated:
//!
//! ```
//! use nalgebra::Rotation3;
//! use vision_calibration_core::{
//!     BrownConrady5, CalibratedRig, FxFyCxCy, Iso3, PointCovariance, Pt3, Uncertainty, Vec3,
//!     make_pinhole_camera, pinhole_camera_params,
//! };
//! use vision_mvg::{PixelMatch, recover_relative_pose};
//!
//! // One 800 px camera with barrel distortion, moved between two captures.
//! let k = FxFyCxCy { fx: 800.0, fy: 800.0, cx: 640.0, cy: 480.0 };
//! let distortion = BrownConrady5 { k1: -0.2, iters: 30, ..Default::default() };
//! let rig = CalibratedRig::single(
//!     pinhole_camera_params(&make_pinhole_camera(k, distortion)),
//!     Uncertainty::default(),
//! )?;
//!
//! // The second view: 20 cm to the right and turned 0.1 rad toward the scene.
//! let r = Rotation3::from_euler_angles(0.0, -0.1, 0.0).into_inner();
//! let t = Vec3::new(-0.2, 0.0, 0.0);
//! let scene: Vec<Pt3> = (0..12)
//!     .map(|i| {
//!         let a = f64::from(i);
//!         Pt3::new((0.7 * a).sin(), 0.5 * (1.3 * a).cos(), 2.0 + 0.1 * a)
//!     })
//!     .collect();
//! let model = rig.model(0).unwrap();
//! // Each pixel is detected to about half a pixel.
//! let noise = PointCovariance::isotropic(0.5)?;
//! let matches: Vec<PixelMatch> = scene
//!     .iter()
//!     .map(|p| {
//!         PixelMatch::new(
//!             model.project_point_c(&p.coords).unwrap(),
//!             noise,
//!             model.project_point_c(&(r * p.coords + t)).unwrap(),
//!             noise,
//!         )
//!     })
//!     .collect();
//!
//! let pose = recover_relative_pose(&rig, [0, 0], &matches)?;
//! assert!((pose.second_se3_first.translation.vector - t.normalize()).norm() < 1e-6);
//! assert_eq!(pose.points.len(), 12);
//! # Ok::<(), vision_mvg::MvgError>(())
//! ```

mod bearing;
pub mod bundle_adjust;
mod cheirality;
pub mod degeneracy;
pub mod dense;
pub mod error;
pub mod homography;
pub mod metrology;
pub mod pose_recovery;
pub mod rectification;
pub mod refine;
pub mod residuals;
pub mod robust;
#[cfg(test)]
mod test_support;
pub mod triangulation;
pub mod types;

// Re-export core types for convenience.
pub use bundle_adjust::{BundleAdjustOptions, BundleAdjustment, Scale, bundle_adjust};
pub use dense::{BlockMatchOptions, DisparityMap, GrayImage, match_block};
pub use error::{MvgError, Result};
pub use homography::{
    HomographyDecomposition, decompose_homography, homography_from_pose_and_plane,
    homography_transfer, homography_transfer_inverse,
};
pub use pose_recovery::{RelativePose, recover_relative_pose};
pub use rectification::{RectifiedMatch, RectifyOptions, StereoRectification, rectify};
pub use robust::{
    EssentialEstimate, HomographyEstimate, RobustRelativePose, estimate_essential,
    estimate_homography, recover_relative_pose_robust,
};
pub use types::{
    Correspondence2D, EssentialMatrix, FundamentalMatrix, HomographyMatrix, PixelMatch,
    TriangulatedPoint,
};

// Re-export low-level solvers from vision-geometry.
pub use vision_calibration_core::linalg::{normalize_points_2d, normalize_points_3d};
pub use vision_geometry::camera_matrix::{
    CameraMatrixDecomposition, Mat34, decompose_camera_matrix, dlt_camera_matrix, rq_decompose,
};
pub use vision_geometry::epipolar::{
    decompose_essential, essential_5point, fundamental_7point, fundamental_8point,
    fundamental_8point_ransac,
};
pub use vision_geometry::homography::{dlt_homography, dlt_homography_ransac};
pub use vision_geometry::triangulation::triangulate_point_linear;

#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeDoctests;
