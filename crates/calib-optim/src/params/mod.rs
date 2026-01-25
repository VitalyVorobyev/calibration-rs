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
//! - Laser plane (normal + distance) (currently internal)

pub mod distortion;
pub mod intrinsics;
pub mod laser_plane;
pub mod pose_se3;
