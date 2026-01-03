//! Backend-independent intermediate representation of optimization problems.

mod types;

pub use types::{
    Bound, FactorKind, FixedMask, ManifoldKind, ParamBlock, ParamId, ProblemIR, ResidualBlock,
    RobustLoss,
};
