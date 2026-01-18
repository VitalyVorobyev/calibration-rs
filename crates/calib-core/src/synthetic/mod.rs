//! Deterministic synthetic data generation helpers.
//!
//! This module provides small, reusable building blocks for constructing
//! synthetic calibration problems used in tests and examples:
//! - planar target point grids,
//! - simple pose generators,
//! - projection helpers producing [`crate::CorrespondenceView`],
//! - deterministic pseudo-random noise utilities.
//!
//! The helpers are intentionally lightweight (no heavy dependencies) and
//! deterministic (explicit seeds; stable point ordering).
//!
//! # Example
//!
//! ```no_run
//! use calib_core::{synthetic::planar, BrownConrady5, Camera, FxFyCxCySkew, IdentitySensor, Pinhole};
//!
//! let k = FxFyCxCySkew { fx: 800.0, fy: 800.0, cx: 640.0, cy: 360.0, skew: 0.0 };
//! let dist = BrownConrady5 { k1: 0.0, k2: 0.0, k3: 0.0, p1: 0.0, p2: 0.0, iters: 8 };
//! let cam = Camera::new(Pinhole, dist, IdentitySensor, k);
//!
//! let board = planar::grid_points(6, 5, 0.04);
//! let poses = planar::poses_yaw_y_z(5, -0.3, 0.15, 0.5, 0.1);
//! let views = planar::project_views_all(&cam, &board, &poses).unwrap();
//! assert_eq!(views.len(), 5);
//! ```

pub mod noise;
pub mod planar;
