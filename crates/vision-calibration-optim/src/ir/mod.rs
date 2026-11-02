//! Backend-independent intermediate representation of optimization problems.

mod custom;
mod types;

pub(crate) use custom::{CustomFactor, CustomNoise, CustomResidual};

pub use types::{
    Bound, CameraModelSpec, DistortionKind, FactorKind, FixedMask, HandeyeMode, LaserChain,
    ManifoldKind, ProblemIR, ReprojChain, ReprojSqrtInfo, ResidualBlock, RobustLoss, SensorKind,
};

#[cfg(test)]
pub use types::ParamSlotSpec;
pub(crate) use types::{CameraSpecLayout, ParamBlock, ParamId};
