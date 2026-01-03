//! Non-linear optimization utilities and calibration problems using a backend-agnostic IR.
//!
//! This crate focuses on reusable parameter blocks, factors, and problem builders so
//! multiple calibration tasks can share the same projection and residual machinery.
//! Solver backends (tiny-solver today) compile the IR into optimizer-specific graphs.

pub mod backend;
pub mod factors;
pub mod ir;
pub mod math;
pub mod params;
pub mod problems;

pub use crate::backend::{BackendKind, BackendSolution, BackendSolveOptions};
pub use crate::problems::planar_intrinsics;
