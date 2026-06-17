//! Planar intrinsics calibration with Scheimpflug sensor tilt.
//!
//! This module follows the standard pipeline shape used in this workspace:
//! `problem` + `state` + `steps`.
//!
//! # Recommended workflow (ADR 0022)
//!
//! Seed a coarse prior — the nominal focal (lens spec) and the nominal Scheimpflug
//! mount tilt (`≈ −5°`) — via [`step_init_with_seed`], then optimize. Under the
//! tilt↔focal↔distortion degeneracy, from-scratch [`step_init`] (no seed) is
//! **experimental** and may converge to a wrong tilt/focal basin; it logs a
//! warning when used. The seeded path is gated to ≤ 0.5 px mean reprojection on
//! the private `rtv3d_ref` rig.

mod problem;
mod state;
mod steps;

pub use problem::{
    ScheimpflugFixMask, ScheimpflugIntrinsicsConfig, ScheimpflugIntrinsicsExport,
    ScheimpflugIntrinsicsInput, ScheimpflugIntrinsicsParams, ScheimpflugIntrinsicsProblem,
    ScheimpflugIntrinsicsResult,
};
pub use steps::{
    IntrinsicsInitOptions, IntrinsicsOptimizeOptions, ScheimpflugIntrinsicsInitResult,
    ScheimpflugIntrinsicsOptimizeResult, ScheimpflugManualInit, run_calibration, step_init,
    step_init_with_seed, step_optimize,
};
