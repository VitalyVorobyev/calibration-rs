//! Shared geometric solvers for computer vision.
//!
//! This crate provides deterministic, allocation-light solvers for fundamental
//! geometric problems: epipolar geometry, homography estimation, triangulation,
//! and camera matrix decomposition. These building blocks are used by both
//! calibration workflows and multiple-view geometry pipelines.
//!
//! # Modules
//!
//! - [`math`] — DLT rank guard, plus a re-export of the shared linear-algebra
//!   primitives (Hartley normalization, polynomial/null-space solvers) from
//!   [`vision_calibration_core::linalg`]
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

pub mod camera_matrix;
pub mod epipolar;
pub mod error;
pub mod homography;
pub mod math;
pub mod triangulation;

// Items are reached through their owning module — `epipolar::fundamental_8point`,
// `camera_matrix::Mat34` — so that a reader can tell where a solver lives and
// so adding a `pub fn` to a submodule does not silently widen the crate root.
// Only the error type is lifted, because it is shared by all of them.
pub use error::{GeometryError, Result};
