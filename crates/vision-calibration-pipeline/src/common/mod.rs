//! Step-option types shared across problem modules.
//!
//! Each calibration problem exposes step functions (`step_init`, `step_optimize`,
//! …) that accept an optional per-invocation options struct overriding session
//! config. Several of these option structs are identical across problems — the
//! intrinsics init/optimize options are common to every intrinsics-bearing
//! problem, the rig init/optimize options to the two rig problems, and the
//! hand-eye init/optimize options to the two hand-eye problems.
//!
//! This module holds the single canonical definition of each shared struct.
//! Problem modules re-export the types they use (e.g.
//! `planar_intrinsics::IntrinsicsInitOptions`), so existing paths keep
//! resolving — they now all point at one type, and the contract cannot
//! silently diverge per problem.
//!
//! The [`config`] submodule holds the sibling set of shared **config**
//! sub-structs — the grouped building blocks that top-level
//! `*Config` types embed (`init: IntrinsicsInitConfig`, `solver:
//! SolverConfig`, ...). Those are persisted configuration; the types at this
//! module's top level are ephemeral per-call step overrides. Keeping them in
//! separate modules avoids confusing the two.
//!
//! [`ExportKind`] is the shared export vocabulary: the serde
//! discriminator every `*Export` carries. [`CalibratedCameras`] reads any
//! export's cameras and their projection uncertainty.

/// Shared config sub-structs embedded by grouped top-level `*Config` types.
pub mod config;

mod cameras;
mod export_kind;
pub use cameras::{CalibratedCameras, calibrated_cameras};
pub use export_kind::ExportKind;

/// Options for an intrinsics initialization step.
///
/// These options override session config for a single step invocation.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct IntrinsicsInitOptions {
    /// Override the number of iterations for iterative estimation.
    pub iterations: Option<usize>,
}

/// Options for an intrinsics optimization step.
///
/// These options override session config for a single step invocation.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct IntrinsicsOptimizeOptions {
    /// Override the maximum number of iterations.
    pub max_iters: Option<usize>,
    /// Override verbosity level.
    pub verbosity: Option<usize>,
}

/// Options for a hand-eye initialization step.
///
/// These options override session config for a single step invocation.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct HandeyeInitOptions {
    /// Override minimum motion angle (degrees).
    pub min_motion_angle_deg: Option<f64>,
}

/// Options for a hand-eye optimization step.
///
/// These options override session config for a single step invocation.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct HandeyeOptimizeOptions {
    /// Override the maximum number of iterations.
    pub max_iters: Option<usize>,
    /// Override verbosity level.
    pub verbosity: Option<usize>,
}

/// Options for a rig initialization step.
///
/// The linear rig fit has no per-call overrides yet; the struct keeps
/// `step_rig_init` in the same shape as every other step.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct RigInitOptions {}

/// Options for a rig optimization step.
///
/// These options override session config for a single step invocation.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct RigOptimizeOptions {
    /// Override the maximum number of iterations.
    pub max_iters: Option<usize>,
    /// Override verbosity level.
    pub verbosity: Option<usize>,
}
