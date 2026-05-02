//! Canonical input-data manifest for calibration-rs.
//!
//! [`DatasetSpec`] is the single on-disk wire format the user authors —
//! either by hand or via AI heuristics that inspect a foreign dataset —
//! describing where images, robot poses, and target metadata live. The
//! manifest is _descriptive_, never prescriptive: data stays where the
//! user put it and the manifest just points at it.
//!
//! See ADR 0016 for the design rationale, and ADR 0019 for the
//! fail-fast-on-ambiguity contract that the [`_unresolved`] field
//! enables.
//!
//! # Tiered fields
//!
//! Every field is tagged either `infer_from_data` (an AI manifest
//! generator is expected to populate it from filenames / folder
//! structure / sample data) or `human_or_doc_required` (the AI is
//! forbidden from guessing — it must read documentation or ask the
//! user). When inference fails, the field is left `null` and the
//! field path is recorded in [`DatasetSpec::_unresolved`]; the runner
//! refuses to proceed until that list is empty.
//!
//! Tier metadata is encoded as the `x-calib-tier` schema extension so
//! both Rust and TypeScript form generators can render the right UX.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod spec;
mod validator;

pub use spec::{
    CameraSource, DatasetSpec, ImagePattern, PoseColumnMap, PoseConvention, PosePairing,
    RobotPoseFormat, RobotPoseSource, RotationFormat, TargetSpec, Topology, TransformConvention,
    TranslationUnits,
};
pub use validator::{ValidationError, validate};
