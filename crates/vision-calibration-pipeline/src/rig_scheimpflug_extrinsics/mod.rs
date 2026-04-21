//! Multi-camera rig extrinsics calibration with Scheimpflug-tilted sensors.
//!
//! Mirrors [`super::rig_extrinsics`] with an added per-camera Scheimpflug
//! sensor block. The pipeline has four steps:
//!
//! 1. Per-camera Scheimpflug intrinsics initialization (Zhang + zero tilts).
//! 2. Per-camera Scheimpflug intrinsics refinement.
//! 3. Rig extrinsics linear initialization.
//! 4. Joint rig bundle adjustment with Scheimpflug residuals.

mod problem;
mod state;
mod steps;

pub use problem::{
    RigScheimpflugExtrinsicsConfig, RigScheimpflugExtrinsicsExport, RigScheimpflugExtrinsicsInput,
    RigScheimpflugExtrinsicsProblem,
};
pub use state::RigScheimpflugExtrinsicsState;
pub use steps::{
    IntrinsicsInitOptions, IntrinsicsOptimizeOptions, RigOptimizeOptions, run_calibration,
    step_intrinsics_init_all, step_intrinsics_optimize_all, step_rig_init, step_rig_optimize,
};
