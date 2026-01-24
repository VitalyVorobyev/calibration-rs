//! Backend-independent intermediate representation of optimization problems.

mod types;

pub use types::{
    FactorKind, FixedMask, HandEyeMode, ManifoldKind, ProblemIR, ResidualBlock, RobustLoss,
};

#[cfg(test)]
pub use types::ParamId;
