//! Backend-independent intermediate representation of optimization problems.

mod types;

pub use types::{
    CameraModelDesc, DistortionKind, FactorKind, FixedMask, HandEyeMode, LaserChain, ManifoldKind,
    ParamSlotSpec, ProblemIR, ProjectionKind, ReprojChain, ResidualBlock, RobustLoss, SensorKind,
};

#[cfg(test)]
pub use types::ParamId;
