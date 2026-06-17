//! Benchmarking harness for `calibration-rs`: dataset registry, metric
//! records, stability testing, and cross-validation runners.

pub mod compare;
pub mod crossval;
#[cfg(feature = "tier-b")]
pub mod detect;
pub mod fixtures;
pub mod params;
pub mod record;
pub mod registry;
pub mod run;
pub mod stability;
