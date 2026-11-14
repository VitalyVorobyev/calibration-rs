//! Canonical input-data manifest for calibration-rs.
//!
//! [`DatasetSpec`] is the single on-disk wire format the user authors —
//! either by hand or starting from the skeleton [`sniff_folder`] infers from a
//! dataset's folder layout — describing where images, robot poses, and target
//! metadata live. The
//! manifest is _descriptive_, never prescriptive: data stays where the
//! user put it and the manifest just points at it.
//!
//! # Unresolved fields
//!
//! A manifest generator such as [`sniff_folder`] fills only what it can
//! determine from filenames and folder structure. Values that must come from
//! documentation or the user — the robot-pose convention, the target —
//! are never guessed: the field is left empty and its path is recorded in
//! [`DatasetSpec::unresolved`], and the runner refuses the manifest until
//! that list is empty.
//!
//! # Example
//!
//! The [`samples`] catalogue holds real datasets with ready manifests; each
//! manifest names its images, target and workflow:
//!
//! ```
//! use vision_calibration_dataset::{ProblemKind, samples};
//!
//! let sample = samples::sample("opencv_left")?;
//! let spec = sample.manifest();
//! assert_eq!(spec.topology, ProblemKind::PlanarIntrinsics);
//! assert!(spec.unresolved.is_empty());
//! println!("{}: {} ({})", sample.id, sample.description, sample.licence);
//! # Ok::<(), vision_calibration_dataset::samples::FetchError>(())
//! ```

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod device_spec;
mod load;
pub mod samples;
mod sniff;
mod spec;
mod validator;

pub use device_spec::{
    CameraDeviceSpec, CameraMountSpec, DEVICE_SPEC_FILENAME, DEVICE_SPEC_VERSION, DeviceSpec,
    DeviceSpecError, HandeyeMountSpec, LaserPlaneSpec, NominalPoseSpec, RigLayoutSpec,
    ScheimpflugMountSpec,
};
pub use load::ManifestError;
pub use sniff::{SniffError, sniff_folder};
pub use spec::{
    CameraSource, ChessCornersDetectorSpec, CornerCovarianceSpec, CornerStrategySpec,
    DATASET_SPEC_VERSION, DatasetSpec, DetectorSpec, ImageNoiseSpec, ImagePattern,
    LaserExtractionSpec, LaserScanAxis, PoseColumnMap, PoseConvention, PosePairing, ProblemKind,
    RinggridCenterSpec, RobotPoseFormat, RobotPoseSource, RotationFormat, TargetSpec,
    TransformConvention, TranslationUnits,
};
pub use validator::{ValidationError, validate};

#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeDoctests;
