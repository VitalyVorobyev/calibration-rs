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
//! - [`camera`] - the blocks of a camera of any model (intrinsics, projection,
//!   distortion, sensor), packed with named fix lists
//! - [`distortion::pack_distortion_params`] / [`distortion::unpack_distortion_params`] - distortion of any kind
//! - [`pose_se3::iso3_to_se3_dvec`] / [`pose_se3::se3_dvec_to_iso3`] - SE(3) pose conversions
//! - Laser plane (normal + distance) (currently internal)

pub mod camera;
pub mod distortion;
pub mod intrinsics;
pub mod laser_plane;
pub mod pose_se3;
