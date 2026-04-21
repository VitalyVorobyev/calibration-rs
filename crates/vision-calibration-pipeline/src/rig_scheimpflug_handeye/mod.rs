//! Multi-camera Scheimpflug rig hand-eye calibration.
//!
//! Mirrors [`super::rig_handeye`] with per-camera Scheimpflug sensor support.
//! EyeInHand only.

mod problem;
mod state;
mod steps;

pub use problem::{
    RigScheimpflugHandeyeBaConfig, RigScheimpflugHandeyeConfig, RigScheimpflugHandeyeExport,
    RigScheimpflugHandeyeInitConfig, RigScheimpflugHandeyeInput,
    RigScheimpflugHandeyeIntrinsicsConfig, RigScheimpflugHandeyeProblem,
    RigScheimpflugHandeyeRigConfig, RigScheimpflugHandeyeSolverConfig,
};
pub use state::RigScheimpflugHandeyeState;
pub use steps::{
    HandeyeInitOptions, HandeyeOptimizeOptions, IntrinsicsInitOptions, IntrinsicsOptimizeOptions,
    RigOptimizeOptions, run_calibration, step_handeye_init, step_handeye_optimize,
    step_intrinsics_init_all, step_intrinsics_optimize_all, step_rig_init, step_rig_optimize,
};
