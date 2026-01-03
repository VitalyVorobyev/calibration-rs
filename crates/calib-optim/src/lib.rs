//! Non-linear optimization utilities and calibration problems built on tiny-solver.
//!
//! This crate focuses on reusable parameter blocks, factors, and problem builders so
//! multiple calibration tasks can share the same projection and residual machinery.

pub mod factors;
pub mod math;
pub mod params;
pub mod problems;
pub mod solver;

pub use crate::problems::planar_intrinsics;
pub use crate::solver::tiny::TinySolveOptions;
