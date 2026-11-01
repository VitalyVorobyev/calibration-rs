//! Backend-independent intermediate representation of optimization problems.

mod types;

pub use types::{
    Bound, CameraModelSpec, DistortionKind, FactorKind, FixedMask, HandeyeMode, LaserChain,
    ManifoldKind, ProblemIR, ReprojChain, ReprojSqrtInfo, ResidualBlock, RobustLoss, SensorKind,
};

#[cfg(test)]
pub use types::ParamSlotSpec;
pub(crate) use types::{CameraSpecLayout, ParamBlock, ParamId};
