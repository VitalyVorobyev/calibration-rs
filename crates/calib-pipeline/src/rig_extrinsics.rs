//! High-level multi-camera rig extrinsics pipeline.

use crate::session::{problem_types::RigExtrinsicsProblem, CalibrationSession};
use anyhow::Result;
use serde::{Deserialize, Serialize};

pub use crate::session::problem_types::{
    CameraViewData as RigCameraViewData, RigExtrinsicsInitOptions,
    RigExtrinsicsObservations as RigExtrinsicsInput, RigExtrinsicsOptimOptions,
    RigExtrinsicsOptimized as RigExtrinsicsReport, RigViewData,
};

/// End-to-end rig extrinsics configuration (init + non-linear refinement).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RigExtrinsicsConfig {
    #[serde(default)]
    pub init: RigExtrinsicsInitOptions,
    #[serde(default)]
    pub optim: RigExtrinsicsOptimOptions,
}

/// Run the full rig extrinsics pipeline (init + optimize) and return a report.
pub fn run_rig_extrinsics(
    input: &RigExtrinsicsInput,
    config: &RigExtrinsicsConfig,
) -> Result<RigExtrinsicsReport> {
    let mut session = CalibrationSession::<RigExtrinsicsProblem>::new();
    session.set_observations(input.clone());
    session.initialize(config.init.clone())?;
    session.optimize(config.optim.clone())?;
    session.export()
}
