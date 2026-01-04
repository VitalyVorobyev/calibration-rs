//! Parameter block definitions for camera calibration.
//!
//! Parameter blocks represent the variables being optimized. Each parameter type provides:
//!
//! - **Dimension constant** (`DIM`) - Size of the parameter vector
//! - **Conversion to/from DVector** - For optimization backends
//! - **Conversion to/from calib-core types** - For domain-level usage
//!
//! # Available Parameters
//!
//! - [`intrinsics::Intrinsics4`] - Pinhole camera intrinsics (fx, fy, cx, cy)
//! - [`distortion::BrownConrady5Params`] - Brown-Conrady distortion (k1, k2, k3, p1, p2)
//! - [`pose_se3::iso3_to_se3_dvec`] / [`pose_se3::se3_dvec_to_iso3`] - SE(3) pose conversions
//!
//! # Example
//!
//! ```rust
//! use calib_optim::params::intrinsics::Intrinsics4;
//! use nalgebra::DVector;
//!
//! let intrinsics = Intrinsics4 {
//!     fx: 800.0,
//!     fy: 800.0,
//!     cx: 640.0,
//!     cy: 360.0,
//! };
//!
//! // Convert to optimization vector
//! let vec = intrinsics.to_dvec();
//! assert_eq!(vec.len(), Intrinsics4::DIM);
//!
//! // Reconstruct from vector
//! let restored = Intrinsics4::from_dvec(vec.as_view()).unwrap();
//! assert_eq!(intrinsics, restored);
//! ```

pub mod distortion;
pub mod intrinsics;
pub mod pose_se3;
