//! A calibration project: a dataset, the changes to its workflow's
//! configuration, what to leave out of it, and the runs calibrated from it.
//!
//! This version holds what a project leaves out of its dataset
//! ([`Exclusions`]) and how a run resolves it for the dataset runner's
//! conversions; the project file and its runs build on it.

mod exclude;

#[allow(
    unused_imports,
    reason = "the project file and its runs name them; until then only tests do"
)]
pub use exclude::{
    ExclusionFault, Exclusions, FeatureExclusion, ImageExclusion, ViewExclusion, ViewRef,
};
