mod jacobian_ad;
pub mod problem;
pub mod traits;

// Re-export core optimization traits/types at the crate root for ergonomic use.
pub use crate::traits::{NllsProblem, NllsSolverBackend, SolveOptions, SolveReport};

#[cfg(feature = "lm-backend")]
pub mod backend_lm;

pub mod planar_intrinsics;
pub mod robust;

//#[cfg(feature = "ceres-backend")]
// pub mod backend_ceres; // stub for later
