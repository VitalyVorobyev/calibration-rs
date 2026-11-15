//! Backend-independent intermediate representation of optimization problems.

mod custom;
mod types;

pub(crate) use custom::{CustomFactor, CustomNoise, CustomResidual};

pub(crate) use types::{
    Bound, FactorKind, LaserChain, ManifoldKind, ProblemIR, ReprojChain, ReprojSqrtInfo,
    ResidualBlock,
};
pub use types::{CameraModelSpec, DistortionKind, FixedMask, HandeyeMode, RobustLoss, SensorKind};

#[cfg(test)]
pub(crate) use types::ParamSlotSpec;
pub(crate) use types::{CameraSpecLayout, ParamBlock, ParamId};
