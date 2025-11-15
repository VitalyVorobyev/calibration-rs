// Re-export the core optimization traits and types from the `traits` module.
//
// This keeps `calib_optim::problem::*` as a stable location while the actual
// definitions live in `traits.rs`.
pub use crate::traits::{NllsProblem, NllsSolverBackend, SolveOptions, SolveReport};
