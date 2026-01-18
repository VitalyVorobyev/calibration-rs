//! Core math and geometry primitives for `calibration-rs`.
//!
//! This crate provides the foundational building blocks used by all other
//! crates in the workspace:
//!
//! - linear algebra type aliases (`Real`, `Vec2`, `Pt3`, and friends),
//! - composable camera models (projection + distortion + sensor + intrinsics),
//! - a deterministic, model-agnostic RANSAC engine.
//!
//! Camera pipeline (conceptually):
//! `pixel = intrinsics(sensor(distortion(projection(dir))))`
//!
//! The sensor stage supports a Scheimpflug/tilted sensor homography aligned
//! with OpenCV's `computeTiltProjectionMatrix`.
//!
//! # Modules
//!
//! - [`math`]: basic type aliases and homogeneous helpers.
//! - [`models`]: camera model traits and configuration wrappers.
//! - [`ransac`]: generic robust estimation helpers.
//!
//! # Example
//!
//! ```no_run
//! use calib_core::{
//!     CameraParams, DistortionParams, FxFyCxCySkew, IntrinsicsParams, ProjectionParams,
//!     SensorParams,
//! };
//!
//! let params = CameraParams {
//!     projection: ProjectionParams::Pinhole,
//!     distortion: DistortionParams::None,
//!     sensor: SensorParams::Identity,
//!     intrinsics: IntrinsicsParams::FxFyCxCySkew {
//!         params: FxFyCxCySkew {
//!             fx: 800.0,
//!             fy: 800.0,
//!             cx: 640.0,
//!             cy: 360.0,
//!             skew: 0.0,
//!         },
//!     },
//! };
//! let cam = params.build();
//! let px = cam.project_point_c(&nalgebra::Vector3::new(0.1, 0.2, 1.0));
//! assert!(px.is_some());
//! ```

/// Linear algebra type aliases and helpers.
pub mod math;
/// Camera models and distortion utilities.
pub mod models;
/// Generic RANSAC engine and traits.
pub mod ransac;
/// Test utilities for cross-crate calibration testing.
///
/// This module is public to allow usage in integration tests across
/// the workspace, but is not intended for production use.
pub mod test_utils;
/// Common types for observations, results, and options.
pub mod types;

pub use math::*;
pub use models::*;
pub use ransac::*;
pub use types::*;
