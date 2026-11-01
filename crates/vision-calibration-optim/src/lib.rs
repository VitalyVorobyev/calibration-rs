//! Non-linear least-squares refinement for camera calibration: the
//! optimization engine behind the `vision-calibration` sessions.
//!
//! Each calibration problem is lowered to a backend-agnostic
//! intermediate representation (parameter blocks on their manifolds,
//! autodiff residual factors, robust losses) and solved by one
//! Levenberg–Marquardt loop over two linearization engines, tiny-solver
//! (default) and factrs, selected by [`SolverBackend`].
//!
//! # Public surface
//!
//! This crate's documented items are the vocabulary and results the
//! pipeline's sessions hand to callers:
//!
//! - the solve vocabulary: [`RobustLoss`], [`SolverBackend`], [`HandeyeMode`],
//!   [`LaserlineResidualType`];
//! - each problem's estimate and parameters ([`PlanarIntrinsicsEstimate`],
//!   [`HandeyeEstimate`], [`LaserlineEstimate`], …) with their
//!   [`SolveReport`];
//! - the input types of the laser and hand-eye problems ([`RobotPoseMeta`],
//!   [`LaserlineView`], [`RigLaserlineView`], …) and the [`LaserPlane`];
//! - projection uncertainty ([`projection_uncertainty`]) and the residual
//!   helpers the analysis reports use.
//!
//! The solve entry points (`optimize_*`) and their option structs are public
//! for the workspace's own crates but hidden from these docs and outside the
//! semver promise: they change whenever a pipeline step does. Calibrate
//! through the step functions of `vision-calibration` instead.

mod backend;
mod error;
mod factors;
mod information;
mod ir;
mod params;
mod problems;
mod projection;
mod uncertainty;

pub use error::Error;

// The IR (`ProblemIR` and its factor/parameter/manifold vocabulary), the
// backend dispatch, and the parameter-packing helpers are *implementation*.
// They are how a problem is lowered to a solver, not something a caller
// assembles: the factor kernels `expect` invariants that only this crate's
// builders establish, so a hand-built IR panics inside the solve loop rather
// than failing to compile. Callers reach for the `optimize_*` entry points
// below, which is what every consumer in this workspace does.
pub use crate::backend::{SolveReport, SolverBackend};
pub use crate::ir::{DistortionKind, HandeyeMode, RobustLoss};
pub use crate::params::laser_plane::LaserPlane;
pub use crate::problems::RobotPoseSigma;
pub use crate::projection::{ProjectionUncertainty, camera_parameters, projection_uncertainty};

// The results of each problem: its estimate, parameters and input types.
pub use crate::problems::handeye::{
    HandeyeEstimate, HandeyeParams, RobotPoseMeta, handeye_observer_se3_target,
};
pub use crate::problems::laserline_bundle::{
    LaserResidualRecords, LaserlineDataset, LaserlineEstimate, LaserlineMeta, LaserlineParams,
    LaserlineResidualType, LaserlineStats, LaserlineView, compute_laserline_feature_residuals,
    compute_laserline_stats,
};
pub use crate::problems::laserline_rig_bundle::{
    RigLaserlineDataset, RigLaserlineEstimate, RigLaserlineView,
    compute_rig_laserline_feature_residuals,
};
pub use crate::problems::planar_intrinsics::{PlanarIntrinsicsEstimate, PlanarIntrinsicsParams};
pub use crate::problems::rig_extrinsics::{RigExtrinsicsEstimate, RigExtrinsicsParams};
pub use crate::problems::rig_handeye_laserline_bundle::{
    RigHandeyeLaserlineEstimate, RigHandeyeLaserlineParams, RigHandeyeLaserlinePerCamStats,
    RigHandeyeLaserlineView,
};

// Workspace-internal: the solve entry points and their options, and the
// helpers the pipeline crate builds its steps from. They are public so the
// pipeline (and the workspace's benchmarks) can call them, hidden from the
// documentation, and outside the semver promise: they change whenever a
// pipeline step does. Calibrate through the pipeline's step functions.
#[doc(hidden)]
pub use crate::backend::BackendSolveOptions;
#[doc(hidden)]
pub use crate::information::{ViewInformation, ViewPose, view_information};
#[doc(hidden)]
pub use crate::params::camera::CameraBounds;
#[doc(hidden)]
pub use crate::params::distortion::with_leading_radial;
#[doc(hidden)]
pub use crate::problems::handeye::{
    HandeyeDataset, HandeyeSolveOptions, apply_robot_delta, optimize_handeye,
};
#[doc(hidden)]
pub use crate::problems::laserline_bundle::{LaserlineSolveOptions, optimize_laserline};
#[doc(hidden)]
pub use crate::problems::laserline_rig_bundle::{
    RigLaserlineSolveOptions, RigLaserlineUpstream, optimize_rig_laserline,
};
#[doc(hidden)]
pub use crate::problems::planar_intrinsics::{
    PlanarIntrinsicsSolveOptions, optimize_planar_intrinsics,
};
#[doc(hidden)]
pub use crate::problems::rig_extrinsics::{
    RigExtrinsicsDataset, RigExtrinsicsSolveOptions, optimize_rig_extrinsics,
};
#[doc(hidden)]
pub use crate::problems::rig_handeye_laserline_bundle::{
    RigHandeyeLaserlineDataset, RigHandeyeLaserlineSolveOptions, evaluate_rig_handeye_laserline,
    optimize_rig_handeye_laserline,
};
#[doc(hidden)]
pub use crate::problems::scheimpflug_intrinsics::{
    ScheimpflugStagedInitOptions, optimize_scheimpflug_intrinsics_staged,
};
#[doc(hidden)]
pub use crate::projection::projection_covariance_at;

#[cfg(test)]
mod tests;

#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeDoctests;
