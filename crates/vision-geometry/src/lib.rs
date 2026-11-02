//! Shared geometric solvers for computer vision.
//!
//! This crate provides deterministic, allocation-light solvers for fundamental
//! geometric problems: epipolar geometry, homography estimation, triangulation,
//! and camera matrix decomposition. These building blocks are used by both
//! calibration workflows and multiple-view geometry pipelines.
//!
//! # Modules
//!
//! - [`epipolar`] — Fundamental and essential matrix estimation, decomposition
//! - [`homography`] — Normalized DLT homography estimation with optional RANSAC
//! - [`triangulation`] — N-view triangulation: linear DLT + Gauss-Newton
//!   reprojection refinement
//! - [`camera_matrix`] — Camera projection matrix estimation and RQ decomposition
//!
//! # Coordinate Conventions
//!
//! - Fundamental matrix solvers accept **pixel coordinates**.
//! - Essential matrix solvers expect **calibrated/normalized coordinates**
//!   (i.e., after applying `K⁻¹` to pixel points).
//! - Pose outputs follow the `T_C_W` convention: transform from world into camera frame.
//!
//! # Example
//!
//! ```
//! use vision_calibration_core::Pt2;
//! use vision_geometry::homography::dlt_homography;
//!
//! // A unit square on the target and where its corners land in the image.
//! let target = [Pt2::new(0.0, 0.0), Pt2::new(1.0, 0.0), Pt2::new(1.0, 1.0), Pt2::new(0.0, 1.0)];
//! let image = [
//!     Pt2::new(120.0, 200.0),
//!     Pt2::new(220.0, 198.0),
//!     Pt2::new(225.0, 300.0),
//!     Pt2::new(118.0, 302.0),
//! ];
//!
//! let h = dlt_homography(&target, &image)?;
//! let centre = h * nalgebra::Vector3::new(0.5, 0.5, 1.0);
//! println!("the square's centre lands at ({:.1}, {:.1})", centre.x / centre.z, centre.y / centre.z);
//! # Ok::<(), vision_geometry::GeometryError>(())
//! ```

pub mod camera_matrix;
pub mod epipolar;
pub mod error;
pub mod homography;
mod math;
pub mod triangulation;

// Items are reached through their owning module — `epipolar::fundamental_8point`,
// `camera_matrix::Mat34` — so that a reader can tell where a solver lives and
// so adding a `pub fn` to a submodule does not silently widen the crate root.
// Only the error type is lifted, because it is shared by all of them.
pub use error::{GeometryError, Result};

#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeDoctests;
