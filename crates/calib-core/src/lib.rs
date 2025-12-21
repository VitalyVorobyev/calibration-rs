//! Core math and geometry primitives for `calibration-rs`.
//!
//! This crate contains:
//! - linear algebra type aliases (`Real`, `Vec2`, `Pt3`, ...),
//! - composable camera models (projection + distortion + sensor + intrinsics),
//! - a generic RANSAC engine (`ransac`, [`Estimator`]).
//!
//! Camera pipeline:
//! `pixel = K ∘ sensor ∘ distortion ∘ projection(dir)`
//!
//! The sensor stage supports a Scheimpflug/tilted sensor homography aligned with
//! OpenCV's `computeTiltProjectionMatrix`.

/// Linear algebra type aliases and helpers.
pub mod math;
/// Camera models and distortion utilities.
pub mod models;
/// Generic RANSAC engine and traits.
pub mod ransac;

pub use math::*;
pub use models::*;
pub use ransac::*;
