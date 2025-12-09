//! Core math and geometry primitives for `calibration-rs`.
//!
//! This crate contains:
//! - linear algebra type aliases (`Real`, `Vec2`, `Pt3`, ...),
//! - basic camera models (`CameraIntrinsics`, `PinholeCamera`),
//! - a generic RANSAC engine (`ransac`, [`Estimator`]).

/// Linear algebra type aliases and helpers.
pub mod math;
/// Camera models and distortion utilities.
pub mod models;
/// Generic RANSAC engine and traits.
pub mod ransac;

pub use math::*;
pub use models::*;
pub use ransac::*;
