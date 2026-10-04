//! Canonical input-data manifest for calibration-rs.
//!
//! [`DatasetSpec`] is the single on-disk wire format the user authors —
//! either by hand or starting from the skeleton [`sniff_folder`] infers from a
//! dataset's folder layout — describing where images, robot poses, and target
//! metadata live. The
//! manifest is _descriptive_, never prescriptive: data stays where the
//! user put it and the manifest just points at it.
//!
//! The [`DatasetSpec::unresolved`] field enables a fail-fast-on-ambiguity
//! contract: the runner refuses a manifest until it is cleared.
//!
//! # Tiered fields
//!
//! Every field is tagged either `infer_from_data` (a manifest generator such
//! as [`sniff_folder`] may populate it from filenames / folder structure) or
//! `human_or_doc_required` (never guessed — it must come from documentation
//! or the user). When inference fails, the field is left `null` and the
//! field path is recorded in [`DatasetSpec::unresolved`]; the runner
//! refuses to proceed until that list is empty.
//!
//! Tier metadata is encoded as the `x-calib-tier` schema extension so
//! both Rust and TypeScript form generators can render the right UX.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod device_spec;
mod sniff;
mod spec;
mod validator;

pub use device_spec::{
    CameraDeviceSpec, CameraMountSpec, DEVICE_SPEC_FILENAME, DEVICE_SPEC_VERSION, DeviceSpec,
    DeviceSpecError, HandeyeMountSpec, LaserPlaneSpec, NominalPoseSpec, RigLayoutSpec,
    ScheimpflugMountSpec,
};
pub use sniff::{SniffError, sniff_folder};
pub use spec::{
    CameraSource, ChessCornersDetectorSpec, CornerStrategySpec, DatasetSpec, DetectorSpec,
    ImagePattern, LaserExtractionSpec, LaserScanAxis, PoseColumnMap, PoseConvention, PosePairing,
    RobotPoseFormat, RobotPoseSource, RotationFormat, TargetSpec, Topology, TransformConvention,
    TranslationUnits,
};
pub use validator::{ValidationError, validate};

#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeDoctests;
