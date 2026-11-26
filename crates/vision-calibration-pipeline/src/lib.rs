//! High-level camera calibration pipelines.
//!
//! Most users want the facade crate `vision-calibration`, which re-exports
//! these workflows next to the camera models, detection and dataset
//! manifests.
//!
//! ## Session API
//!
//! A [`CalibrationSession`](session::CalibrationSession) holds a problem's
//! input, config, intermediate state and export; step functions advance it.
//! The `vision-calibration` crate documentation opens with a session
//! calibrating a camera on synthetic data.

mod error;
pub use error::Error;

// Step-option types shared across problem modules.
pub mod common;

// Multi-level reprojection-error analysis (pure library; no I/O).
pub mod analysis;

// Core session framework
mod planar_family;
mod rig_family;
pub mod session;

// Device-spec → manual-init seed derivation
pub mod device_seed;

// Problem-specific modules
pub mod laserline_device;
pub mod planar_intrinsics;
pub mod rig_extrinsics;
pub mod rig_handeye;
pub mod rig_handeye_laserline;
pub mod rig_laserline_device;
pub mod single_cam_handeye;

// Manifest-driven dataset runner: detection and problem input from a DatasetSpec
pub mod dataset_runner;

// Calibration projects: a dataset, its configuration changes, exclusions and runs
pub mod project;

#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeDoctests;
