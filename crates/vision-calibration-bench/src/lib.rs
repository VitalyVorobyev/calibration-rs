//! Benchmarking harness for `calibration-rs`: the dataset registry and
//! acceptance gates, the public regression suite, metric records, and
//! solver and convergence-basin studies.

pub mod baseline;
pub mod basin;
pub mod dense;
#[cfg(feature = "tier-b")]
pub mod detect;
pub mod record;
pub mod registry;
pub mod ringgrid_bias;
pub mod run;
pub mod solver;
pub mod suite;
