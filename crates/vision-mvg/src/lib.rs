//! Multiple-view geometry for calibrated cameras.
//!
//! This crate provides tools for two-view and multi-view geometry:
//!
//! - **Types**: [`types::PixelMatch`] (a pair of pixels with their noise), [`types::Correspondence2D`]
//!   (a pair of plain points), [`types::TriangulatedPoint`]
//! - **Residuals**: algebraic, Sampson, geometric epipolar residuals and
//!   symmetric transfer error for homographies
//! - **Degeneracy**: pure-rotation, planar-scene and poor-baseline detection
//! - **Pose recovery**: [`pose_recovery::recover_relative_pose`] — end-to-end calibrated 2-view pose
//! - **Triangulation**: [`triangulation::triangulate`] — N-view triangulation of
//!   pixel tracks from a calibration, refined on stated pixel noise, with
//!   diagnostics and the points' joint covariance (the calibration's share
//!   included)
//! - **Laser points**: [`laser::triangulate_laser`] — laser-line pixels on a
//!   camera's known laser plane, with the covariance of the pixels, the camera,
//!   the plane and the extrinsic
//! - **Stereo**: [`rectification::rectify`] — rectification of two rig cameras from the
//!   calibration for any camera model, the disparity-to-depth matrix, dense
//!   point clouds from a disparity map and triangulated matches, both with the
//!   calibration's uncertainty in their covariance
//! - **Measurements**: [`metrology`] — distances, angles, plane fits and
//!   point-to-plane distances over triangulated points, each with the σ from
//!   the points' joint covariance
//! - **Object pose**: [`pose::estimate_pose`] — the pose of a known object from its
//!   pixels in one or several rig cameras, with optional RANSAC in σ and the
//!   pose's covariance (the calibration's share included)
//! - **Bundle adjustment**: [`bundle_adjust()`](bundle_adjust::bundle_adjust) — capture poses and points
//!   refined together from a calibration, with a stated gauge (capture 0 and
//!   a [`bundle_adjust::Scale`]), robust losses and the joint covariance; the calibration
//!   enters as consider parameters or under a prior
//!
//! The closed-form two-view solvers (5-point essential matrix, fundamental
//! matrix, DLT homography, linear triangulation) live in `vision-geometry`;
//! this crate builds the calibrated pipelines on top of them.
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
//! use vision_mvg::pose_recovery::recover_relative_pose;
//! use vision_mvg::types::PixelMatch;
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
pub mod dense;
mod error;
pub mod homography;
pub mod laser;
pub mod metrology;
pub mod pose;
pub mod pose_recovery;
pub mod rectification;
pub mod refine;
pub mod residuals;
pub mod robust;
mod sensitivity;
#[cfg(test)]
mod test_support;
pub mod triangulation;
pub mod types;

pub use error::MvgError;
pub(crate) use error::Result;

#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeDoctests;
