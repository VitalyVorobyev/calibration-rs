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
//! Here eight synthetic views of an 8×6 board (40 mm pitch) calibrate a camera:
//!
//! ```
//! use vision_calibration_core::synthetic::{planar, poses};
//! use vision_calibration_core::{BrownConrady5, FxFyCxCy, PlanarDataset, View, make_pinhole_camera};
//! use vision_calibration_pipeline::planar_intrinsics::{
//!     PlanarIntrinsicsProblem, step_init, step_optimize,
//! };
//! use vision_calibration_pipeline::session::CalibrationSession;
//! # fn main() -> anyhow::Result<()> {
//!
//! let truth = make_pinhole_camera(
//!     FxFyCxCy { fx: 800.0, fy: 780.0, cx: 640.0, cy: 360.0 },
//!     BrownConrady5 { k1: 0.05, k2: -0.02, ..BrownConrady5::default() },
//! );
//! let board = planar::grid_points(8, 6, 0.04);
//! let specs = poses::seeded_board_pose_specs(8, 0.35, 3);
//! let poses = poses::centered_board_poses(&specs, (0.14, 0.10), 0.6);
//! let views = planar::project_views_all(&truth, &board, &poses)?;
//! let dataset = PlanarDataset::new(views.into_iter().map(View::without_meta).collect())?;
//!
//! let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
//! session.set_input(dataset)?;
//! step_init(&mut session, None)?;
//! step_optimize(&mut session, None)?;
//!
//! let export = session.export()?;
//! assert!((export.camera.intrinsics.fx - 800.0).abs() < 1e-3);
//! # Ok(())
//! # }
//! ```

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

// Device-spec → manual-init seed derivation (ADR 0011)
pub mod device_seed;

// Problem-specific modules
pub mod laserline_device;
pub mod planar_intrinsics;
pub mod rig_extrinsics;
pub mod rig_handeye;
pub mod rig_handeye_laserline;
pub mod rig_laserline_device;
pub mod single_cam_handeye;

// Manifest-driven dataset runner (ADR 0016 + ADR 0016)
pub mod dataset_runner;

#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeDoctests;
