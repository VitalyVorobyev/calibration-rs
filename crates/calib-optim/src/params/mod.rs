//! Parameter block definitions for camera calibration.
//!
//! Parameter blocks represent the variables being optimized. Each parameter type provides:
//!
//! - **Dimension constant** - Size of the parameter vector
//! - **Conversion to/from DVector** - For optimization backends
//!
//! # Available Parameters
//!
//! - [`intrinsics::pack_intrinsics`] / [`intrinsics::unpack_intrinsics`] - Pinhole intrinsics
//! - [`distortion::pack_distortion`] / [`distortion::unpack_distortion`] - Brown-Conrady distortion
//! - [`pose_se3::iso3_to_se3_dvec`] / [`pose_se3::se3_dvec_to_iso3`] - SE(3) pose conversions
//! - [`laser_plane::LaserPlane`] - Laser plane (normal + distance)
//!
//! # Example
//!
//! ```rust
//! use calib_optim::params::intrinsics::{pack_intrinsics, unpack_intrinsics, INTRINSICS_DIM};
//! use nalgebra::DVector;
//!
//! let intrinsics = calib_core::FxFyCxCySkew {
//!     fx: 800.0,
//!     fy: 800.0,
//!     cx: 640.0,
//!     cy: 360.0,
//!     skew: 0.0,
//! };
//!
//! // Convert to optimization vector
//! let vec = pack_intrinsics(&intrinsics).unwrap();
//! assert_eq!(vec.len(), INTRINSICS_DIM);
//!
//! // Reconstruct from vector
//! let restored = unpack_intrinsics(vec.as_view()).unwrap();
//! assert_eq!(intrinsics.fx, restored.fx);
//! ```

pub mod distortion;
pub mod intrinsics;
pub mod laser_plane;
pub mod pose_se3;

pub use laser_plane::LaserPlane;
