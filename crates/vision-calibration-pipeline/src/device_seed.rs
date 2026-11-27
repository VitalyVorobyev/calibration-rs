//! Seeds: the device quantities a calibration starts from.
//!
//! A [`DeviceSeed`] holds, by camera id, what belongs to the hardware rather
//! than to a capture: each camera's lens, its mount in the rig, its laser
//! plane, and the rig's hand-eye. It comes from one of two sources:
//!
//! - a datasheet and a drawing, [`DeviceSpec`] ([`DeviceSeed::from_spec`]);
//! - a previous calibration of the device or of one of its type, any
//!   export ([`DeviceSeed::from_calibration`]).
//!
//! Its derivations (`intrinsics_init`, `rig_init`, `handeye_init`, …) give
//! the manual-init seeds the workflows' `*_with_seed` steps take. A seed is
//! an initial value: nothing in it constrains the solve or acts as a prior.
//!
//! The free functions ([`scheimpflug_seed`], [`rig_intrinsics_seed`],
//! [`nominal_cam_se3_rig`], [`rig_layout_seed`], [`handeye_seed`]) derive the
//! same seeds from a [`DeviceSpec`] directly, in the drawing's rig frame.
//!
//! All unit conversion between the datasheet-natural spec (`mm`, `µm`,
//! degrees) and the internal representation (pixels, radians, and
//! **metres** for world-frame translations — the pipeline's world unit)
//! happens here and nowhere else:
//!
//! - `fx = fy = focal_mm · 1000 / pixel_pitch_um`, principal point
//!   defaulting to the resolution center, no skew term;
//! - mount tilt degrees → `ScheimpflugParams` radians;
//! - `rig_se3_cam` mount poses → inverted into the `cam_se3_rig` (`T_C_R`)
//!   the rig problems expect, translations mm → m.
//!
//! The derivations are per-camera-id, never positional: the caller states
//! the camera order it needs and mismatches surface as typed errors.

use serde::{Deserialize, Serialize};
use vision_calibration_core::{
    CameraModelSpec, CameraParams, FxFyCxCy, Handeye, Iso3, LaserPlane, Real, ScheimpflugParams,
    SensorParams,
};
use vision_calibration_dataset::ProblemKind;
use vision_calibration_optim::HandeyeMode;

use crate::common::{
    CalibratedCameras, IntrinsicsManualInit, RigIntrinsicsManualInit, RigManualInit, RigPoses,
};
use crate::laserline_device::DeviceManualInit;
use crate::rig_handeye::HandeyeManualInit;
use crate::rig_laserline_device::PlaneManualInit;
use crate::single_cam_handeye;

// The schema types every consumer of this module needs, re-exported so the
// facade can expose the full device-spec surface from one place.
pub use vision_calibration_dataset::{
    CameraDeviceSpec, CameraMountSpec, DEVICE_SPEC_FILENAME, DEVICE_SPEC_VERSION, DeviceSpec,
    DeviceSpecError, HandeyeMountSpec, LaserPlaneSpec, NominalPoseSpec, RigLayoutSpec,
    ScheimpflugMountSpec,
};

/// Failures when deriving seeds from a [`DeviceSpec`] or a [`DeviceSeed`].
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum DeviceSeedError {
    /// The spec or seed has no camera with the requested id.
    #[error("no camera with id `{id}`")]
    UnknownCameraId {
        /// The unmatched id.
        id: String,
    },

    /// The spec has no rig mechanical layout.
    #[error("device spec has no rig layout")]
    MissingRigLayout,

    /// The spec's rig layout, or the seed, has no hand-eye.
    #[error("no hand-eye mount in the device spec or seed")]
    MissingHandeye,

    /// The spec's rig layout, or the seed, has no mount for the requested
    /// camera.
    #[error("no rig mount for camera `{id}`")]
    MissingCameraMount {
        /// The unmatched id.
        id: String,
    },

    /// The hand-eye mode of the spec or seed contradicts the session's
    /// configured mode — seeding across modes would silently seed the wrong
    /// transform.
    #[error("the hand-eye mount is {spec:?} but the session expects {expected:?}")]
    HandeyeModeMismatch {
        /// Mode stated by the spec or seed.
        spec: HandeyeMode,
        /// Mode the caller's config expects.
        expected: HandeyeMode,
    },

    /// A view has no observed target pose in any camera, so its
    /// `rig_se3_target` cannot be anchored.
    #[error("view {view} has no target pose in any camera")]
    ViewWithoutTargetPose {
        /// The offending view index.
        view: usize,
    },

    /// The device spec itself failed validation (e.g. non-positive focal
    /// length or pixel pitch, or non-finite values).
    #[error(transparent)]
    InvalidSpec(#[from] DeviceSpecError),

    /// A view's per-camera pose list did not match `camera_ids`.
    #[error("view {view} has {got} camera pose entries, expected {expected}")]
    ViewCameraCountMismatch {
        /// The offending view index.
        view: usize,
        /// Number of pose entries supplied.
        got: usize,
        /// Number of camera ids supplied.
        expected: usize,
    },

    /// The seed's format version is not the one this library reads.
    #[error("unsupported device-seed version {found} (this library reads version {supported})")]
    UnsupportedVersion {
        /// The version of the seed.
        found: u32,
        /// [`DEVICE_SEED_VERSION`].
        supported: u32,
    },

    /// The seed, or the list of camera ids asked of it, has no camera.
    #[error("the device seed has no cameras")]
    NoCameras,

    /// Two cameras of the seed, or two ids asked of it, are the same.
    #[error("duplicate camera id `{id}`")]
    DuplicateCameraId {
        /// The repeated id.
        id: String,
    },

    /// A value of the seed is NaN or infinite.
    #[error("{context} must be finite")]
    NonFinite {
        /// Which value.
        context: String,
    },

    /// Some of the seed's cameras have a mount in the rig and others do not:
    /// the rig frame of the others is unknown.
    #[error("some of the seed's cameras have a rig mount and others do not")]
    PartialMounts,

    /// A seed field is set for some of the requested cameras and not for
    /// the others, where the manual-init seed takes all or none.
    #[error("`{field}` is seeded for some of the cameras and not for the others")]
    PartialField {
        /// The field (`distortion`, `laser_plane`, `cam_se3_rig`).
        field: &'static str,
    },

    /// A camera's sensor is a general homography, which no manual-init seed
    /// can express.
    #[error("camera `{camera}` has a homography sensor, which a seed cannot carry")]
    UnseedableSensor {
        /// The camera id.
        camera: String,
    },

    /// A seed of several cameras has a hand-eye but no camera mounts, so the
    /// rig frame the hand-eye is stated in is unknown.
    #[error("the seed has a hand-eye but its cameras have no rig mounts")]
    HandeyeWithoutMounts,

    /// The camera ids or image sizes given with an export are not one per
    /// camera of the export.
    #[error(
        "the export has {export} cameras but {camera_ids} camera ids and {image_sizes} image \
         sizes were given"
    )]
    ExportCameraCount {
        /// Cameras in the export.
        export: usize,
        /// Camera ids given.
        camera_ids: usize,
        /// Image sizes given.
        image_sizes: usize,
    },

    /// The reference camera index is not an index into the camera ids.
    #[error("reference camera {reference} is out of range for {cameras} cameras")]
    ReferenceOutOfRange {
        /// The reference index given.
        reference: usize,
        /// The number of camera ids given.
        cameras: usize,
    },

    /// The seed's cameras are not the dataset's: a seed for a subset of a
    /// device's cameras is made with [`DeviceSeed::select`].
    #[error(
        "the seed's cameras are not the dataset's (missing from the seed: {missing:?}; \
         not in the dataset: {unexpected:?})"
    )]
    CameraIdsDiffer {
        /// The dataset's cameras the seed has no camera for, in the
        /// dataset's order.
        missing: Vec<String>,
        /// The seed's cameras the dataset does not have, in the seed's
        /// order.
        unexpected: Vec<String>,
    },

    /// A calibrated lens of the seed is of another camera model than the
    /// one configured for its camera. [`DeviceSeed::nominal`] keeps only the
    /// intrinsics and the tilt, for a calibration of another model.
    #[error("camera `{camera}`: the seed's model is {seed:?} but the config's is {config:?}")]
    ModelMismatch {
        /// The camera id.
        camera: String,
        /// The model of the seed's lens.
        seed: CameraModelSpec,
        /// The model the config names for the camera.
        config: CameraModelSpec,
    },

    /// A nominal lens of the seed states a sensor tilt, and the model
    /// configured for its camera has no Scheimpflug sensor.
    #[error("camera `{camera}`: the seed states a sensor tilt but the config's model is untilted")]
    TiltForUntiltedModel {
        /// The camera id.
        camera: String,
    },

    /// A seed camera's images are of another size than the dataset's: a
    /// seed for another binning or region of the sensor is for another
    /// camera.
    #[error(
        "camera `{camera}`: the seed is for {seed:?} px images but the dataset's are {dataset:?} px"
    )]
    ImageSizeMismatch {
        /// The camera id.
        camera: String,
        /// The seed camera's `[width, height]`.
        seed: [u32; 2],
        /// The dataset camera's `[width, height]`.
        dataset: [u32; 2],
    },

    /// Nothing in the seed applies to the workflow, so a seeded run would
    /// be an unseeded one.
    #[error("nothing in the seed applies to a {topology:?} calibration")]
    NothingToSeed {
        /// The workflow's topology.
        topology: ProblemKind,
    },
}

/// Seed for a planar intrinsics session whose camera model has a
/// Scheimpflug sensor.
///
/// Intrinsics and sensor tilt are always seeded (both are
/// load-bearing), so the solve trusts the tilt. A camera without a
/// `scheimpflug` entry is frontal *by design*, so it seeds an identity tilt
/// — the spec asserts knowledge of the mount, absence is not ignorance.
/// Distortion and poses stay `None` (auto). The session's config must name
/// a Scheimpflug sensor; a tilt seed for an untilted camera is rejected.
///
/// The same seed as [`DeviceSeed::from_spec`] followed by
/// [`DeviceSeed::intrinsics_init`].
pub fn scheimpflug_seed(
    spec: &DeviceSpec,
    camera_id: &str,
) -> Result<IntrinsicsManualInit, DeviceSeedError> {
    spec.validate()?;
    let cam = lookup(spec, camera_id)?;
    Ok(nominal_lens(cam).intrinsics_init())
}

/// Per-camera intrinsics + sensor seeds for a rig hand-eye session, in the
/// caller's `camera_ids` order.
///
/// `per_cam_sensors` is always populated (zero tilt for frontal cameras):
/// it seeds the tilt of each camera whose model has a Scheimpflug sensor,
/// and an untilted camera accepts its zero tilt.
/// `per_cam_distortion` stays `None` (auto).
///
/// The same seed as [`DeviceSeed::from_spec`] followed by
/// [`DeviceSeed::rig_intrinsics_init`].
pub fn rig_intrinsics_seed(
    spec: &DeviceSpec,
    camera_ids: &[&str],
) -> Result<RigIntrinsicsManualInit, DeviceSeedError> {
    spec.validate()?;
    let lenses = camera_ids
        .iter()
        .map(|id| lookup(spec, id).map(nominal_lens))
        .collect::<Result<Vec<_>, _>>()?;
    rig_intrinsics_init(&lenses.iter().collect::<Vec<_>>())
}

/// Nominal `cam_se3_rig` (`T_C_R`) poses from the mechanical layout, in the
/// caller's `camera_ids` order. Translations are converted to the
/// pipeline's world unit (metres).
///
/// The poses are in the drawing's rig frame; [`DeviceSeed::cam_se3_rig`]
/// gives them re-expressed in the frame of a reference camera.
///
/// This is deliberately *not* a `RigManualInit`: a rig seed (`RigPoses`)
/// couples `cam_se3_rig` with the data-dependent per-view `rig_se3_target`,
/// so combining the nominals with per-view estimates is the caller's job.
pub fn nominal_cam_se3_rig(
    spec: &DeviceSpec,
    camera_ids: &[&str],
) -> Result<Vec<Iso3>, DeviceSeedError> {
    spec.validate()?;
    let rig = spec.rig.as_ref().ok_or(DeviceSeedError::MissingRigLayout)?;
    camera_ids
        .iter()
        .map(|id| {
            // Spec cameras and mounts are validated id-consistent at load;
            // distinguish "unknown camera" from "camera without a mount".
            lookup(spec, id)?;
            let mount = rig.cameras.iter().find(|m| m.id == *id).ok_or_else(|| {
                DeviceSeedError::MissingCameraMount {
                    id: (*id).to_string(),
                }
            })?;
            Ok(cam_se3_rig_from_mount(mount))
        })
        .collect()
}

/// Full rig-stage seed: nominal `cam_se3_rig` from the mechanical layout,
/// combined with per-view `rig_se3_target` anchored on measured per-camera
/// target poses (the two are coupled: a rig seed carries both).
///
/// `cam_se3_target` is `[view][cam] -> Option<cam_se3_target>` in the
/// same camera order as `camera_ids` (the shape returned by
/// `step_intrinsics_init_all*`). For each view, the first camera with an
/// observed pose anchors it:
/// `rig_se3_target = cam_se3_rig[c]⁻¹ · cam_se3_target[c]`.
///
/// The rig frame is the drawing's; [`DeviceSeed::rig_init`] anchors the
/// views on the mounts re-expressed in the frame of a reference camera.
pub fn rig_layout_seed(
    spec: &DeviceSpec,
    camera_ids: &[&str],
    cam_se3_target: &[Vec<Option<Iso3>>],
) -> Result<RigManualInit, DeviceSeedError> {
    let cam_se3_rig = nominal_cam_se3_rig(spec, camera_ids)?;
    anchor_views(cam_se3_rig, cam_se3_target)
}

/// Hand-eye stage seed from the mechanical mount.
///
/// The spec's mount mode must match the session's configured
/// [`HandeyeMode`]; a mismatch is a typed error rather than a silently
/// wrong transform. The target pose stays `None` (data-dependent).
///
/// The hand-eye is in the drawing's rig frame;
/// [`DeviceSeed::handeye_init`] re-expresses it in the frame of a reference
/// camera.
pub fn handeye_seed(
    spec: &DeviceSpec,
    expected_mode: HandeyeMode,
) -> Result<HandeyeManualInit, DeviceSeedError> {
    spec.validate()?;
    let rig = spec.rig.as_ref().ok_or(DeviceSeedError::MissingRigLayout)?;
    let mount = rig
        .handeye
        .as_ref()
        .ok_or(DeviceSeedError::MissingHandeye)?;
    handeye_from_mount(mount).manual_init(expected_mode, &Iso3::identity())
}

// ─────────────────────────────────────────────────────────────────────────────
// The seed value
// ─────────────────────────────────────────────────────────────────────────────

/// The version of the [`DeviceSeed`] format this library reads and writes.
pub const DEVICE_SEED_VERSION: u32 = 1;

/// The device quantities a calibration starts from, by camera id: a
/// datasheet's nominal values or a previous calibration's estimates.
///
/// A seed is an initial value; nothing in it is a constraint or a prior,
/// and it carries no uncertainty. It holds what belongs to the hardware:
/// each camera's lens, its mount in the rig, its laser plane and the rig's
/// hand-eye. Nothing of a capture is in it (no target poses, robot-pose
/// corrections or residuals), and no configuration: the camera model is the
/// config's, and a calibrated lens must be of that model.
///
/// Units are the pipeline's: pixels, radians and metres. A seed is made by
/// [`from_spec`](Self::from_spec) or
/// [`from_calibration`](Self::from_calibration), not transcribed by hand;
/// its JSON form is how it is stored.
///
/// The rig frame of the mounts and the hand-eye is the seed's own (a
/// drawing's frame, or the reference camera of the run it came from). The
/// derivations that take a `reference` camera re-express them so that
/// camera's mount is the identity, as a rig problem's gauge requires; laser
/// planes are in their camera's frame and need no change.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct DeviceSeed {
    /// Format version, [`DEVICE_SEED_VERSION`].
    pub version: u32,
    /// The cameras, ids unique; at least one.
    pub cameras: Vec<SeedCamera>,
    /// The rig on the robot, in the seed's rig frame (see
    /// [`SeedCamera::cam_se3_rig`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub handeye: Option<SeedHandeye>,
    /// Where the seed came from, for its reader.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

/// One camera of a [`DeviceSeed`].
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct SeedCamera {
    /// The camera's id in the dataset manifest.
    pub id: String,
    /// `[width, height]` in pixels of the images the values are for (the
    /// tile, for a camera that is a region of a larger sensor). A seed for
    /// another image size (another binning or region) is for another
    /// camera.
    pub image_size: [u32; 2],
    /// The lens: intrinsics, distortion and sensor tilt.
    pub lens: SeedLens,
    /// Camera-from-rig (`T_C_R`) in the seed's rig frame. Every camera has
    /// one or none does; a seed of one camera without it has the camera's
    /// frame as its rig frame.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(
        feature = "schemars",
        schemars(with = "Option<vision_calibration_core::Iso3Schema>")
    )]
    pub cam_se3_rig: Option<Iso3>,
    /// The camera's laser plane, in the camera's own frame.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub laser_plane: Option<LaserPlane>,
}

/// A seed camera's lens: a whole calibrated camera, or a datasheet's values.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
#[non_exhaustive]
pub enum SeedLens {
    /// A calibrated camera, whole, in its model: intrinsics, distortion and
    /// sensor. Its model must be the one the session is configured with: the
    /// seeded steps refuse a distortion of another kind or a tilt for an
    /// untilted camera, and the derivations do not compare the projection.
    /// [`DeviceSeed::nominal`] keeps only the intrinsics and the tilt, for a
    /// session of another model.
    Calibrated {
        /// The camera's parameters (a sensor homography is not seedable).
        camera: CameraParams,
    },
    /// What a datasheet states: the intrinsics and the mount tilt. The
    /// distortion is not known and starts at zero; the tilt is the stated
    /// one, zero for a frontal mount.
    Nominal {
        /// Pinhole intrinsics, pixels.
        intrinsics: FxFyCxCy<Real>,
        /// Sensor tilt, radians; zero for a frontal mount.
        sensor: ScheimpflugParams,
    },
}

/// Where a seed's rig sits on the robot, in the seed's rig frame.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[serde(tag = "mode", rename_all = "snake_case", deny_unknown_fields)]
#[non_exhaustive]
pub enum SeedHandeye {
    /// The rig rides on the gripper.
    EyeInHand {
        /// The rig's pose on the gripper (`T_G_R`).
        #[cfg_attr(
            feature = "schemars",
            schemars(with = "vision_calibration_core::Iso3Schema")
        )]
        gripper_se3_rig: Iso3,
    },
    /// The rig stands in the robot's base frame.
    EyeToHand {
        /// The base's pose in the rig (`T_R_B`).
        #[cfg_attr(
            feature = "schemars",
            schemars(with = "vision_calibration_core::Iso3Schema")
        )]
        rig_se3_base: Iso3,
    },
}

impl SeedCamera {
    /// Camera `id` with `lens`, for images of `image_size` (`[width,
    /// height]`, pixels), without a rig mount or a laser plane.
    pub fn new(id: impl Into<String>, image_size: [u32; 2], lens: SeedLens) -> Self {
        Self {
            id: id.into(),
            image_size,
            lens,
            cam_se3_rig: None,
            laser_plane: None,
        }
    }
}

impl DeviceSeed {
    /// A seed of the current format version of `cameras`, without a
    /// hand-eye or a description.
    pub fn new(cameras: Vec<SeedCamera>) -> Self {
        Self {
            version: DEVICE_SEED_VERSION,
            cameras,
            handeye: None,
            description: None,
        }
    }

    /// The datasheet seed of a [`DeviceSpec`]: each camera's
    /// [`Nominal`](SeedLens::Nominal) lens, its mount (the drawing's
    /// `rig_se3_cam`, inverted) and the hand-eye mount, converted to
    /// pixels, radians and metres, the drawing's rig frame kept.
    ///
    /// The values are the free functions' ([`scheimpflug_seed`],
    /// [`rig_intrinsics_seed`], [`nominal_cam_se3_rig`], [`handeye_seed`]),
    /// bit for bit. The spec's nominal laser planes are not carried.
    ///
    /// # Errors
    ///
    /// [`DeviceSeedError::InvalidSpec`] when the spec does not validate;
    /// [`DeviceSeedError::PartialMounts`] when its rig layout mounts some
    /// cameras and not others; [`DeviceSeedError::HandeyeWithoutMounts`]
    /// when it has several cameras and a hand-eye mount but no camera
    /// mounts.
    pub fn from_spec(spec: &DeviceSpec) -> Result<Self, DeviceSeedError> {
        spec.validate()?;
        let mounts = spec.rig.as_ref().map_or(&[][..], |rig| &rig.cameras[..]);
        let cameras = spec
            .cameras
            .iter()
            .map(|cam| {
                let mut seed =
                    SeedCamera::new(cam.id.clone(), cam.resolution_px, nominal_lens(cam));
                seed.cam_se3_rig = mounts
                    .iter()
                    .find(|mount| mount.id == cam.id)
                    .map(cam_se3_rig_from_mount);
                seed
            })
            .collect();
        let mut seed = Self::new(cameras);
        seed.handeye = spec
            .rig
            .as_ref()
            .and_then(|rig| rig.handeye.as_ref())
            .map(handeye_from_mount);
        seed.description = spec.description.clone();
        seed.validate()?;
        Ok(seed)
    }

    /// The device quantities of a calibration: every camera whole
    /// ([`Calibrated`](SeedLens::Calibrated)), the mounts of a rig of
    /// several cameras, the laser planes and the hand-eye, as the export
    /// states them. Nothing of the capture is carried (target poses,
    /// robot-pose corrections, residuals), nor the uncertainty.
    ///
    /// An export names neither its cameras nor its image sizes:
    /// `camera_ids` and `image_sizes` give them, one per camera in the
    /// export's camera order. A single-camera export gives a seed without
    /// mounts: the camera's frame is its rig frame.
    ///
    /// # Errors
    ///
    /// [`DeviceSeedError::ExportCameraCount`] when `camera_ids` or
    /// `image_sizes` is not one per camera;
    /// [`DeviceSeedError::UnseedableSensor`] for a camera with a sensor
    /// homography; [`DeviceSeedError::PartialField`] when the export's
    /// mounts or laser planes are not one per camera; any
    /// [`validate`](Self::validate) error (duplicate ids, non-finite
    /// values).
    pub fn from_calibration<E: CalibratedCameras + ?Sized>(
        export: &E,
        camera_ids: &[String],
        image_sizes: &[[u32; 2]],
    ) -> Result<Self, DeviceSeedError> {
        let cameras = export.camera_params();
        let count = cameras.len();
        if camera_ids.len() != count || image_sizes.len() != count {
            return Err(DeviceSeedError::ExportCameraCount {
                export: count,
                camera_ids: camera_ids.len(),
                image_sizes: image_sizes.len(),
            });
        }
        let mounts = export.cam_se3_rig();
        if mounts.len() != count {
            return Err(DeviceSeedError::PartialField {
                field: "cam_se3_rig",
            });
        }
        let planes = export.laser_planes_cam();
        if !planes.is_empty() && planes.len() != count {
            return Err(DeviceSeedError::PartialField {
                field: "laser_plane",
            });
        }
        let mut seed_cameras = Vec::with_capacity(count);
        for (index, camera) in cameras.into_iter().enumerate() {
            let id = &camera_ids[index];
            if matches!(camera.sensor, SensorParams::Homography { .. }) {
                return Err(DeviceSeedError::UnseedableSensor { camera: id.clone() });
            }
            let mut seed = SeedCamera::new(
                id.clone(),
                image_sizes[index],
                SeedLens::Calibrated { camera },
            );
            // One camera is a rig of one: its frame is the rig frame.
            seed.cam_se3_rig = (count > 1).then_some(mounts[index]);
            seed.laser_plane = planes.get(index).cloned();
            seed_cameras.push(seed);
        }
        let mut seed = Self::new(seed_cameras);
        seed.handeye = export.handeye().map(|handeye| match handeye {
            Handeye::EyeInHand { gripper_se3_rig } => SeedHandeye::EyeInHand { gripper_se3_rig },
            Handeye::EyeToHand { rig_se3_base } => SeedHandeye::EyeToHand { rig_se3_base },
        });
        seed.validate()?;
        Ok(seed)
    }

    /// The same seed with every calibrated lens reduced to a
    /// [`Nominal`](SeedLens::Nominal) one (its intrinsics and its tilt,
    /// zero for an untilted sensor): the explicit way to start a session of
    /// another camera model from a calibration. Mounts, laser planes and
    /// the hand-eye are kept. A lens with a sensor homography, which a seed
    /// cannot carry, is left as it is (and fails
    /// [`validate`](Self::validate)).
    #[must_use]
    pub fn nominal(&self) -> Self {
        let mut seed = self.clone();
        for camera in &mut seed.cameras {
            if let SeedLens::Calibrated { camera: params } = &camera.lens {
                let sensor = match &params.sensor {
                    SensorParams::Identity => Some(ScheimpflugParams::default()),
                    SensorParams::Scheimpflug { params } => Some(*params),
                    _ => None,
                };
                if let Some(sensor) = sensor {
                    camera.lens = SeedLens::Nominal {
                        intrinsics: params.intrinsics,
                        sensor,
                    };
                }
            }
        }
        seed
    }

    /// The seed of the cameras `ids` only, in that order, with the rig
    /// frame, the hand-eye and the description unchanged.
    ///
    /// # Errors
    ///
    /// [`DeviceSeedError::UnknownCameraId`],
    /// [`DeviceSeedError::DuplicateCameraId`] or
    /// [`DeviceSeedError::NoCameras`] for `ids`; any
    /// [`validate`](Self::validate) error of this seed.
    pub fn select(&self, ids: &[&str]) -> Result<Self, DeviceSeedError> {
        self.validate()?;
        let cameras = self.cameras_of(ids)?.into_iter().cloned().collect();
        let seed = Self {
            cameras,
            ..self.clone()
        };
        seed.validate()?;
        Ok(seed)
    }

    /// Check the seed: its version, at least one camera, unique ids, finite
    /// values, no sensor homography, mounts on every camera or none, and
    /// mounts for a hand-eye of several cameras.
    ///
    /// # Errors
    ///
    /// The [`DeviceSeedError`] of the first failed check.
    pub fn validate(&self) -> Result<(), DeviceSeedError> {
        if self.version != DEVICE_SEED_VERSION {
            return Err(DeviceSeedError::UnsupportedVersion {
                found: self.version,
                supported: DEVICE_SEED_VERSION,
            });
        }
        if self.cameras.is_empty() {
            return Err(DeviceSeedError::NoCameras);
        }
        let mut seen = std::collections::BTreeSet::new();
        for camera in &self.cameras {
            if !seen.insert(camera.id.as_str()) {
                return Err(DeviceSeedError::DuplicateCameraId {
                    id: camera.id.clone(),
                });
            }
            check_lens(&camera.id, &camera.lens)?;
            if let Some(pose) = &camera.cam_se3_rig {
                check_pose(pose, || format!("camera `{}` cam_se3_rig", camera.id))?;
            }
        }
        let mounted = self
            .cameras
            .iter()
            .filter(|camera| camera.cam_se3_rig.is_some())
            .count();
        if mounted != 0 && mounted != self.cameras.len() {
            return Err(DeviceSeedError::PartialMounts);
        }
        if let Some(handeye) = &self.handeye {
            check_pose(handeye.pose(), || "the hand-eye".to_string())?;
            if mounted == 0 && self.cameras.len() > 1 {
                return Err(DeviceSeedError::HandeyeWithoutMounts);
            }
        }
        Ok(())
    }

    /// The camera `id`, if the seed has it.
    pub fn camera(&self, id: &str) -> Option<&SeedCamera> {
        self.cameras.iter().find(|camera| camera.id == id)
    }

    /// The seed of a one-camera intrinsics stage (planar intrinsics, the
    /// single-camera hand-eye's intrinsics) for camera `camera_id`.
    ///
    /// A calibrated lens seeds the intrinsics, the distortion (of the
    /// lens's model, which must be the configured one) and, for a
    /// Scheimpflug sensor, the tilt; a nominal lens seeds the intrinsics and
    /// the tilt, and the distortion starts at zero. The target poses are
    /// left to the stage.
    ///
    /// A seeded tilt changes how planar intrinsics refines a Scheimpflug
    /// camera: the solve trusts the tilt and runs a warm start. It sweeps
    /// the leading radial coefficient over a fixed set of starting values
    /// (`0`, `-0.20`, `-0.40`) with the intrinsics and the tilt held, so a
    /// seeded `k1` is replaced by the sweep's best start, and then refines
    /// everything within bounds around the seed: `fx` and `fy` within
    /// `[0.75, 1.5]` times the seed, the principal point within `0.15·f`,
    /// the tilt within `0.35` rad. A solution on a bound has no covariance.
    ///
    /// # Errors
    ///
    /// [`DeviceSeedError::UnknownCameraId`], or any
    /// [`validate`](Self::validate) error.
    pub fn intrinsics_init(
        &self,
        camera_id: &str,
    ) -> Result<IntrinsicsManualInit, DeviceSeedError> {
        self.validate()?;
        Ok(self.lookup(camera_id)?.lens.intrinsics_init())
    }

    /// The seed of a rig's per-camera intrinsics stage for the cameras
    /// `camera_ids`, in that order.
    ///
    /// The intrinsics of every camera and the tilts (zero for an untilted
    /// camera) are seeded; the tilts are left unseeded when every lens is a
    /// calibrated one with an untilted sensor. The distortion is seeded when
    /// every lens is a calibrated one and left to start at zero when every
    /// lens is a nominal one.
    ///
    /// # Errors
    ///
    /// [`DeviceSeedError::PartialField`] (`distortion`) when some of the
    /// lenses are calibrated and others nominal: a distortion is not
    /// invented for the nominal ones. [`DeviceSeedError::UnknownCameraId`],
    /// [`DeviceSeedError::DuplicateCameraId`] or
    /// [`DeviceSeedError::NoCameras`] for `camera_ids`; any
    /// [`validate`](Self::validate) error.
    pub fn rig_intrinsics_init(
        &self,
        camera_ids: &[&str],
    ) -> Result<RigIntrinsicsManualInit, DeviceSeedError> {
        self.validate()?;
        let lenses: Vec<&SeedLens> = self
            .cameras_of(camera_ids)?
            .into_iter()
            .map(|camera| &camera.lens)
            .collect();
        rig_intrinsics_init(&lenses)
    }

    /// The cameras' mounts (`cam_se3_rig`, `T_C_R`) in the frame of the
    /// camera `camera_ids[reference]`, in `camera_ids` order: `C_i ·
    /// C_ref⁻¹`, with the reference's exactly the identity, as a rig
    /// problem whose `rig.reference_camera_idx` is `reference` requires. A
    /// single camera's is the identity.
    ///
    /// # Errors
    ///
    /// [`DeviceSeedError::ReferenceOutOfRange`];
    /// [`DeviceSeedError::MissingCameraMount`] when the seed has no mounts
    /// and several cameras are asked; [`DeviceSeedError::UnknownCameraId`],
    /// [`DeviceSeedError::DuplicateCameraId`] or
    /// [`DeviceSeedError::NoCameras`] for `camera_ids`; any
    /// [`validate`](Self::validate) error.
    pub fn cam_se3_rig(
        &self,
        camera_ids: &[&str],
        reference: usize,
    ) -> Result<Vec<Iso3>, DeviceSeedError> {
        self.validate()?;
        let cameras = self.cameras_of(camera_ids)?;
        check_reference(reference, cameras.len())?;
        if cameras.len() == 1 {
            return Ok(vec![Iso3::identity()]);
        }
        let mounts = cameras
            .iter()
            .map(|camera| self.mount(camera))
            .collect::<Result<Vec<_>, _>>()?;
        let Some(rig_se3_ref) = rig_se3_ref(&mounts[reference]) else {
            return Ok(mounts);
        };
        Ok(mounts
            .iter()
            .enumerate()
            .map(|(index, mount)| {
                if index == reference {
                    Iso3::identity()
                } else {
                    mount * rig_se3_ref
                }
            })
            .collect())
    }

    /// The seed of a rig stage: the mounts of
    /// [`cam_se3_rig`](Self::cam_se3_rig) (in the reference camera's frame)
    /// with each view's `rig_se3_target` anchored on its first camera with
    /// a target pose, `rig_se3_target = cam_se3_rig[c]⁻¹ ·
    /// cam_se3_target[c]`.
    ///
    /// `cam_se3_target` is `[view][camera]`, in `camera_ids` order (the
    /// shape of the per-camera intrinsics stage's result).
    ///
    /// # Errors
    ///
    /// [`DeviceSeedError::ViewCameraCountMismatch`],
    /// [`DeviceSeedError::ViewWithoutTargetPose`], and those of
    /// [`cam_se3_rig`](Self::cam_se3_rig).
    pub fn rig_init(
        &self,
        camera_ids: &[&str],
        reference: usize,
        cam_se3_target: &[Vec<Option<Iso3>>],
    ) -> Result<RigManualInit, DeviceSeedError> {
        anchor_views(self.cam_se3_rig(camera_ids, reference)?, cam_se3_target)
    }

    /// The seed of a rig hand-eye stage: the seed's hand-eye in the frame of
    /// the camera `camera_ids[reference]` (eye-in-hand `gripper_se3_rig ·
    /// C_ref⁻¹`, eye-to-hand `C_ref · rig_se3_base`). The target's pose is
    /// left to the stage.
    ///
    /// # Errors
    ///
    /// [`DeviceSeedError::MissingHandeye`];
    /// [`DeviceSeedError::HandeyeModeMismatch`] when the seed's mode is not
    /// `mode`; and those of [`cam_se3_rig`](Self::cam_se3_rig).
    pub fn handeye_init(
        &self,
        camera_ids: &[&str],
        reference: usize,
        mode: HandeyeMode,
    ) -> Result<HandeyeManualInit, DeviceSeedError> {
        self.validate()?;
        let cameras = self.cameras_of(camera_ids)?;
        check_reference(reference, cameras.len())?;
        let handeye = self.handeye.ok_or(DeviceSeedError::MissingHandeye)?;
        let ref_se3_rig = self.mount(cameras[reference])?;
        handeye.manual_init(mode, &ref_se3_rig)
    }

    /// The seed of a single-camera hand-eye stage for camera `camera_id`:
    /// the seed's hand-eye with the camera as the rig (eye-in-hand
    /// `gripper_se3_cam = gripper_se3_rig · cam_se3_rig⁻¹`, eye-to-hand
    /// `cam_se3_base = cam_se3_rig · rig_se3_base`), so one camera of a rig
    /// seed seeds a single-camera hand-eye of it. The target's pose is left
    /// to the stage.
    ///
    /// # Errors
    ///
    /// Those of [`handeye_init`](Self::handeye_init).
    pub fn single_cam_handeye_init(
        &self,
        camera_id: &str,
        mode: HandeyeMode,
    ) -> Result<single_cam_handeye::HandeyeManualInit, DeviceSeedError> {
        use single_cam_handeye::HandeyeManualInit as Single;
        Ok(match self.handeye_init(&[camera_id], 0, mode)? {
            HandeyeManualInit::EyeInHand {
                gripper_se3_rig,
                base_se3_target,
            } => Single::EyeInHand {
                gripper_se3_cam: gripper_se3_rig,
                base_se3_target,
            },
            HandeyeManualInit::EyeToHand {
                rig_se3_base,
                gripper_se3_target,
            } => Single::EyeToHand {
                cam_se3_base: rig_se3_base,
                gripper_se3_target,
            },
        })
    }

    /// The seed of a laserline device for camera `camera_id`: the lens as
    /// [`intrinsics_init`](Self::intrinsics_init) seeds it, and the
    /// camera's laser plane when the seed has one.
    ///
    /// A laserline device holds a Scheimpflug camera's tilt at its seed
    /// unless the config's `camera.fix.sensor` lists the tilts to estimate,
    /// so a seed from a previous calibration fixes that calibration's tilt.
    ///
    /// # Errors
    ///
    /// [`DeviceSeedError::UnknownCameraId`], or any
    /// [`validate`](Self::validate) error.
    pub fn device_init(&self, camera_id: &str) -> Result<DeviceManualInit, DeviceSeedError> {
        self.validate()?;
        let camera = self.lookup(camera_id)?;
        let lens = camera.lens.intrinsics_init();
        Ok(DeviceManualInit {
            intrinsics: lens.intrinsics,
            distortion: lens.distortion,
            sensor: lens.sensor,
            cam_se3_target: None,
            laser_plane: camera.laser_plane.clone(),
        })
    }

    /// The seed of a rig laserline device's planes for the cameras
    /// `camera_ids`, in that order: every camera's laser plane in its own
    /// frame, or `None` (the linear fit) when no camera has one.
    ///
    /// # Errors
    ///
    /// [`DeviceSeedError::PartialField`] (`laser_plane`) when some of the
    /// cameras have a plane and others not;
    /// [`DeviceSeedError::UnknownCameraId`],
    /// [`DeviceSeedError::DuplicateCameraId`] or
    /// [`DeviceSeedError::NoCameras`] for `camera_ids`; any
    /// [`validate`](Self::validate) error.
    pub fn plane_init(&self, camera_ids: &[&str]) -> Result<PlaneManualInit, DeviceSeedError> {
        self.validate()?;
        let planes: Vec<Option<LaserPlane>> = self
            .cameras_of(camera_ids)?
            .into_iter()
            .map(|camera| camera.laser_plane.clone())
            .collect();
        Ok(PlaneManualInit {
            laser_planes_cam: all_or_none(planes, "laser_plane")?,
        })
    }

    /// The camera `id`, or [`DeviceSeedError::UnknownCameraId`].
    fn lookup(&self, id: &str) -> Result<&SeedCamera, DeviceSeedError> {
        self.camera(id)
            .ok_or_else(|| DeviceSeedError::UnknownCameraId { id: id.to_string() })
    }

    /// The cameras `ids`, in that order: at least one, each once.
    fn cameras_of(&self, ids: &[&str]) -> Result<Vec<&SeedCamera>, DeviceSeedError> {
        if ids.is_empty() {
            return Err(DeviceSeedError::NoCameras);
        }
        let mut seen = std::collections::BTreeSet::new();
        ids.iter()
            .map(|id| {
                if !seen.insert(*id) {
                    return Err(DeviceSeedError::DuplicateCameraId {
                        id: (*id).to_string(),
                    });
                }
                self.lookup(id)
            })
            .collect()
    }

    /// `camera`'s mount in the seed's rig frame: its `cam_se3_rig`, or the
    /// identity for the one camera of a seed without mounts.
    fn mount(&self, camera: &SeedCamera) -> Result<Iso3, DeviceSeedError> {
        match camera.cam_se3_rig {
            Some(mount) => Ok(mount),
            None if self.cameras.len() == 1 => Ok(Iso3::identity()),
            None => Err(DeviceSeedError::MissingCameraMount {
                id: camera.id.clone(),
            }),
        }
    }
}

impl SeedLens {
    /// The lens's pinhole intrinsics.
    fn intrinsics(&self) -> FxFyCxCy<Real> {
        match self {
            SeedLens::Calibrated { camera } => camera.intrinsics,
            SeedLens::Nominal { intrinsics, .. } => *intrinsics,
        }
    }

    /// The tilt a one-camera seed sets: a nominal lens's, a calibrated
    /// Scheimpflug camera's, and none for a calibrated untilted one.
    fn sensor_seed(&self) -> Option<ScheimpflugParams> {
        match self {
            SeedLens::Calibrated { camera } => match &camera.sensor {
                SensorParams::Scheimpflug { params } => Some(*params),
                _ => None,
            },
            SeedLens::Nominal { sensor, .. } => Some(*sensor),
        }
    }

    /// The seed of a one-camera intrinsics stage.
    fn intrinsics_init(&self) -> IntrinsicsManualInit {
        IntrinsicsManualInit {
            intrinsics: Some(self.intrinsics()),
            distortion: match self {
                SeedLens::Calibrated { camera } => Some(camera.distortion.clone()),
                SeedLens::Nominal { .. } => None,
            },
            sensor: self.sensor_seed(),
            ..Default::default()
        }
    }
}

impl SeedHandeye {
    /// The pose the hand-eye states.
    fn pose(&self) -> &Iso3 {
        match self {
            SeedHandeye::EyeInHand { gripper_se3_rig } => gripper_se3_rig,
            SeedHandeye::EyeToHand { rig_se3_base } => rig_se3_base,
        }
    }

    /// The rig hand-eye seed of `expected` mode, in the frame of the camera
    /// whose mount in the seed's rig frame is `ref_se3_rig`.
    fn manual_init(
        self,
        expected: HandeyeMode,
        ref_se3_rig: &Iso3,
    ) -> Result<HandeyeManualInit, DeviceSeedError> {
        let mode = match self {
            SeedHandeye::EyeInHand { .. } => HandeyeMode::EyeInHand,
            SeedHandeye::EyeToHand { .. } => HandeyeMode::EyeToHand,
        };
        if mode != expected {
            return Err(DeviceSeedError::HandeyeModeMismatch {
                spec: mode,
                expected,
            });
        }
        let rig_se3_ref = rig_se3_ref(ref_se3_rig);
        Ok(match self {
            SeedHandeye::EyeInHand { gripper_se3_rig } => HandeyeManualInit::EyeInHand {
                gripper_se3_rig: Some(match rig_se3_ref {
                    Some(rig_se3_ref) => gripper_se3_rig * rig_se3_ref,
                    None => gripper_se3_rig,
                }),
                base_se3_target: None,
            },
            SeedHandeye::EyeToHand { rig_se3_base } => HandeyeManualInit::EyeToHand {
                rig_se3_base: Some(match rig_se3_ref {
                    Some(_) => ref_se3_rig * rig_se3_base,
                    None => rig_se3_base,
                }),
                gripper_se3_target: None,
            },
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Conversion helpers
// ─────────────────────────────────────────────────────────────────────────────

fn lookup<'a>(spec: &'a DeviceSpec, id: &str) -> Result<&'a CameraDeviceSpec, DeviceSeedError> {
    spec.camera(id)
        .ok_or_else(|| DeviceSeedError::UnknownCameraId { id: id.to_string() })
}

/// A datasheet camera's lens.
fn nominal_lens(cam: &CameraDeviceSpec) -> SeedLens {
    SeedLens::Nominal {
        intrinsics: intrinsics_from_camera(cam),
        sensor: sensor_from_camera(cam),
    }
}

fn intrinsics_from_camera(cam: &CameraDeviceSpec) -> FxFyCxCy<Real> {
    let f_px = cam.focal_mm * 1000.0 / cam.pixel_pitch_um;
    let [cx, cy] = cam.principal_point_px.unwrap_or([
        f64::from(cam.resolution_px[0]) * 0.5,
        f64::from(cam.resolution_px[1]) * 0.5,
    ]);
    FxFyCxCy {
        fx: f_px,
        fy: f_px,
        cx,
        cy,
    }
}

fn sensor_from_camera(cam: &CameraDeviceSpec) -> ScheimpflugParams {
    match &cam.scheimpflug {
        Some(mount) => ScheimpflugParams {
            tilt_x: mount.tilt_x_deg.to_radians(),
            tilt_y: mount.tilt_y_deg.to_radians(),
        },
        None => ScheimpflugParams::default(),
    }
}

/// A drawing's camera mount as `cam_se3_rig` (the drawing states
/// `rig_se3_cam`).
fn cam_se3_rig_from_mount(mount: &CameraMountSpec) -> Iso3 {
    iso3_from_nominal(&mount.rig_se3_cam).inverse()
}

/// A drawing's hand-eye mount.
fn handeye_from_mount(mount: &HandeyeMountSpec) -> SeedHandeye {
    match mount {
        HandeyeMountSpec::EyeInHand { gripper_se3_rig } => SeedHandeye::EyeInHand {
            gripper_se3_rig: iso3_from_nominal(gripper_se3_rig),
        },
        HandeyeMountSpec::EyeToHand { rig_se3_base } => SeedHandeye::EyeToHand {
            rig_se3_base: iso3_from_nominal(rig_se3_base),
        },
    }
}

/// Spec translations are millimetres (drawing units); derived poses are in
/// the pipeline's world unit, **metres** (target 3D points are built as
/// `mm / 1000`, laser errors/plane distances are metres throughout).
fn iso3_from_nominal(pose: &NominalPoseSpec) -> Iso3 {
    let [roll, pitch, yaw] = pose.rpy_deg.map(f64::to_radians);
    let rotation = nalgebra::UnitQuaternion::from_euler_angles(roll, pitch, yaw);
    let translation = nalgebra::Translation3::new(
        pose.translation_mm[0] * 1e-3,
        pose.translation_mm[1] * 1e-3,
        pose.translation_mm[2] * 1e-3,
    );
    Iso3::from_parts(translation, rotation)
}

/// The per-camera intrinsics seed of `lenses`, in their order.
fn rig_intrinsics_init(lenses: &[&SeedLens]) -> Result<RigIntrinsicsManualInit, DeviceSeedError> {
    let untilted_calibrated = |lens: &&SeedLens| {
        matches!(lens, SeedLens::Calibrated { .. }) && lens.sensor_seed().is_none()
    };
    let per_cam_sensors = if lenses.iter().all(untilted_calibrated) {
        None
    } else {
        Some(
            lenses
                .iter()
                .map(|lens| lens.sensor_seed().unwrap_or_default())
                .collect(),
        )
    };
    let distortions = lenses
        .iter()
        .map(|lens| match lens {
            SeedLens::Calibrated { camera } => Some(camera.distortion.clone()),
            SeedLens::Nominal { .. } => None,
        })
        .collect();
    Ok(RigIntrinsicsManualInit {
        per_cam_intrinsics: Some(lenses.iter().map(|lens| lens.intrinsics()).collect()),
        per_cam_distortion: all_or_none(distortions, "distortion")?,
        per_cam_sensors,
    })
}

/// Every value, `None` when there are none, and an error for some.
fn all_or_none<T>(
    values: Vec<Option<T>>,
    field: &'static str,
) -> Result<Option<Vec<T>>, DeviceSeedError> {
    let present = values.iter().filter(|value| value.is_some()).count();
    if present == 0 {
        Ok(None)
    } else if present == values.len() {
        Ok(Some(values.into_iter().flatten().collect()))
    } else {
        Err(DeviceSeedError::PartialField { field })
    }
}

/// A rig seed of the mounts `cam_se3_rig`, each view anchored on its first
/// camera with a target pose.
pub(crate) fn anchor_views(
    cam_se3_rig: Vec<Iso3>,
    cam_se3_target: &[Vec<Option<Iso3>>],
) -> Result<RigManualInit, DeviceSeedError> {
    for (view, cams) in cam_se3_target.iter().enumerate() {
        if cams.len() != cam_se3_rig.len() {
            return Err(DeviceSeedError::ViewCameraCountMismatch {
                view,
                got: cams.len(),
                expected: cam_se3_rig.len(),
            });
        }
    }
    let rig_se3_target = cam_se3_target
        .iter()
        .enumerate()
        .map(|(view, cams)| {
            cams.iter()
                .zip(&cam_se3_rig)
                .find_map(|(pose, cam_se3_rig)| {
                    pose.map(|cam_se3_target| cam_se3_rig.inverse() * cam_se3_target)
                })
                .ok_or(DeviceSeedError::ViewWithoutTargetPose { view })
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(RigManualInit {
        poses: Some(RigPoses::new(cam_se3_rig, rig_se3_target)),
    })
}

/// The seed's rig frame seen from the reference camera, `ref_se3_rig⁻¹`;
/// `None` when the reference's mount is the identity, where the frames are
/// one and re-expressing a pose changes nothing, bit for bit.
fn rig_se3_ref(ref_se3_rig: &Iso3) -> Option<Iso3> {
    (*ref_se3_rig != Iso3::identity()).then(|| ref_se3_rig.inverse())
}

fn check_reference(reference: usize, cameras: usize) -> Result<(), DeviceSeedError> {
    if reference < cameras {
        Ok(())
    } else {
        Err(DeviceSeedError::ReferenceOutOfRange { reference, cameras })
    }
}

fn check_lens(id: &str, lens: &SeedLens) -> Result<(), DeviceSeedError> {
    let non_finite = || DeviceSeedError::NonFinite {
        context: format!("camera `{id}` lens"),
    };
    let finite = |values: &[Real]| values.iter().all(|v| v.is_finite());
    match lens {
        SeedLens::Calibrated { camera } => {
            if matches!(camera.sensor, SensorParams::Homography { .. }) {
                return Err(DeviceSeedError::UnseedableSensor {
                    camera: id.to_string(),
                });
            }
            if !camera.blocks().iter().all(|block| finite(block)) {
                return Err(non_finite());
            }
        }
        SeedLens::Nominal { intrinsics, sensor } => {
            let k = intrinsics;
            if !finite(&[k.fx, k.fy, k.cx, k.cy, sensor.tilt_x, sensor.tilt_y]) {
                return Err(non_finite());
            }
        }
    }
    Ok(())
}

fn check_pose(pose: &Iso3, context: impl Fn() -> String) -> Result<(), DeviceSeedError> {
    let finite = pose.translation.vector.iter().all(|v| v.is_finite())
        && pose.rotation.coords.iter().all(|v| v.is_finite());
    if finite {
        Ok(())
    } else {
        Err(DeviceSeedError::NonFinite { context: context() })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use vision_calibration_dataset::{CameraMountSpec, RigLayoutSpec, ScheimpflugMountSpec};

    fn camera(id: &str) -> CameraDeviceSpec {
        let mut camera = CameraDeviceSpec::new(id.to_string(), 8.0, 7.0, [720, 540]);
        camera.scheimpflug = Some(ScheimpflugMountSpec::new(-5.0, 0.0));
        camera
    }

    fn rig_spec() -> DeviceSpec {
        let mut rig = RigLayoutSpec::new(vec![
            CameraMountSpec::new(
                "cam0",
                NominalPoseSpec::new([0.0, 0.0, 90.0], [100.0, 0.0, 0.0]),
            ),
            CameraMountSpec::new("cam1", NominalPoseSpec::default()),
        ]);
        rig.handeye = Some(HandeyeMountSpec::EyeToHand {
            rig_se3_base: NominalPoseSpec::new([0.0, 0.0, 0.0], [0.0, 250.0, 0.0]),
        });
        let mut spec = DeviceSpec::new(vec![camera("cam0"), camera("cam1")]);
        spec.rig = Some(rig);
        spec
    }

    #[test]
    fn focal_px_from_datasheet() {
        // f_px = f_mm / pixel_pitch.
        let mut cam = camera("cam0");
        cam.focal_mm = 16.0;
        cam.pixel_pitch_um = 4.8;
        let k = intrinsics_from_camera(&cam);
        assert!((k.fx - 16.0e-3 / 4.8e-6).abs() < 1e-9);
        assert!((k.fx - 3_333.333_333_333).abs() < 1e-6);
        assert_eq!(k.fx, k.fy);

        let k = intrinsics_from_camera(&camera("cam0"));
        assert!((k.fx - 8.0e-3 / 7.0e-6).abs() < 1e-9); // ≈ 1142.857
    }

    #[test]
    fn principal_point_defaults_to_center() {
        let k = intrinsics_from_camera(&camera("cam0"));
        assert_eq!((k.cx, k.cy), (360.0, 270.0));

        let mut cam = camera("cam0");
        cam.principal_point_px = Some([355.5, 275.25]);
        let k = intrinsics_from_camera(&cam);
        assert_eq!((k.cx, k.cy), (355.5, 275.25));
    }

    #[test]
    fn tilt_degrees_to_radians_and_frontal_identity() {
        let seed = scheimpflug_seed(&rig_spec(), "cam0").unwrap();
        let sensor = seed.sensor.unwrap();
        assert!((sensor.tilt_x - (-5.0f64).to_radians()).abs() < 1e-12);
        assert!((sensor.tilt_x - (-0.087_266_46)).abs() < 1e-7);
        assert_eq!(sensor.tilt_y, 0.0);
        assert!(seed.intrinsics.is_some());
        assert!(seed.distortion.is_none());
        assert!(seed.cam_se3_target.is_none());

        let mut spec = rig_spec();
        spec.cameras[0].scheimpflug = None;
        let seed = scheimpflug_seed(&spec, "cam0").unwrap();
        let sensor = seed.sensor.unwrap();
        assert_eq!((sensor.tilt_x, sensor.tilt_y), (0.0, 0.0));
    }

    #[test]
    fn rig_intrinsics_follow_caller_order() {
        let spec = rig_spec();
        let seed = rig_intrinsics_seed(&spec, &["cam1", "cam0"]).unwrap();
        let ks = seed.per_cam_intrinsics.unwrap();
        assert_eq!(ks.len(), 2);
        let sensors = seed.per_cam_sensors.unwrap();
        assert_eq!(sensors.len(), 2);
        assert!(seed.per_cam_distortion.is_none());

        assert!(matches!(
            rig_intrinsics_seed(&spec, &["cam0", "cam9"]),
            Err(DeviceSeedError::UnknownCameraId { .. })
        ));
    }

    #[test]
    fn rpy_convention_fixed_axes_xyz() {
        // roll = 90° about x maps ŷ → ẑ.
        let pose = NominalPoseSpec::new([90.0, 0.0, 0.0], [0.0; 3]);
        let iso = iso3_from_nominal(&pose);
        let mapped = iso * nalgebra::Vector3::y();
        assert!((mapped - nalgebra::Vector3::z()).norm() < 1e-12);
    }

    #[test]
    fn cam_se3_rig_is_inverted_mount() {
        // Mount: Rz(90°), t = [100, 0, 0] mm (camera pose in rig frame).
        // Inverse: R = Rz(−90°), t = −Rz(−90°)·[100,0,0] mm = [0, 100, 0] mm
        // = [0, 0.1, 0] in the pipeline's world unit (metres).
        let spec = rig_spec();
        let poses = nominal_cam_se3_rig(&spec, &["cam0", "cam1"]).unwrap();
        let t = poses[0].translation.vector;
        assert!((t - nalgebra::Vector3::new(0.0, 0.1, 0.0)).norm() < 1e-12);
        // Round-trip: inverse of the inverse is the mount pose (in metres).
        let back = poses[0].inverse();
        assert!((back.translation.vector.x - 0.1).abs() < 1e-12);
        assert!(poses[1].translation.vector.norm() < 1e-12); // identity mount

        assert!(matches!(
            nominal_cam_se3_rig(&spec, &["cam9"]),
            Err(DeviceSeedError::UnknownCameraId { .. })
        ));

        let mut no_mount = rig_spec();
        no_mount.rig.as_mut().unwrap().cameras.pop();
        assert!(matches!(
            nominal_cam_se3_rig(&no_mount, &["cam1"]),
            Err(DeviceSeedError::MissingCameraMount { .. })
        ));

        let mut no_rig = rig_spec();
        no_rig.rig = None;
        assert!(matches!(
            nominal_cam_se3_rig(&no_rig, &["cam0"]),
            Err(DeviceSeedError::MissingRigLayout)
        ));
    }

    #[test]
    fn rig_layout_seed_anchors_views_on_first_available_camera() {
        let spec = rig_spec();
        // cam1's mount is identity, so a view anchored on cam1 has
        // rig_se3_target == cam_se3_target directly.
        let target_pose =
            iso3_from_nominal(&NominalPoseSpec::new([0.0, 0.0, 0.0], [10.0, 20.0, 500.0]));
        // View 0: only cam0 sees the target; view 1: only cam1.
        let poses = vec![vec![Some(target_pose), None], vec![None, Some(target_pose)]];
        let seed = rig_layout_seed(&spec, &["cam0", "cam1"], &poses).unwrap();
        let RigPoses {
            cam_se3_rig,
            rig_se3_target,
        } = seed.poses.unwrap();
        assert_eq!(rig_se3_target.len(), 2);
        // View 1 (identity mount): rig_se3_target == the raw target pose.
        assert!(
            (rig_se3_target[1].translation.vector - target_pose.translation.vector).norm() < 1e-12
        );
        // View 0: rig_se3_target = cam_se3_rig[0]⁻¹ · cam_se3_target —
        // consistency: cam_se3_rig[0] · rig_se3_target[0] == target pose.
        let recomposed = cam_se3_rig[0] * rig_se3_target[0];
        assert!((recomposed.translation.vector - target_pose.translation.vector).norm() < 1e-12);

        // A view with no observation in any camera is a typed error.
        let holey = vec![vec![None, None]];
        assert!(matches!(
            rig_layout_seed(&spec, &["cam0", "cam1"], &holey),
            Err(DeviceSeedError::ViewWithoutTargetPose { view: 0 })
        ));
    }

    #[test]
    fn handeye_seed_mode_checked() {
        let spec = rig_spec();
        let seed = handeye_seed(&spec, HandeyeMode::EyeToHand).unwrap();
        let HandeyeManualInit::EyeToHand {
            rig_se3_base,
            gripper_se3_target,
        } = seed
        else {
            panic!("eye-to-hand seed expected");
        };
        let handeye = rig_se3_base.unwrap();
        // 250 mm in the spec → 0.25 m in the pipeline's world unit.
        assert!((handeye.translation.vector.y - 0.25).abs() < 1e-12);
        assert!(gripper_se3_target.is_none());

        assert!(matches!(
            handeye_seed(&spec, HandeyeMode::EyeInHand),
            Err(DeviceSeedError::HandeyeModeMismatch { .. })
        ));

        let mut no_he = rig_spec();
        no_he.rig.as_mut().unwrap().handeye = None;
        assert!(matches!(
            handeye_seed(&no_he, HandeyeMode::EyeToHand),
            Err(DeviceSeedError::MissingHandeye)
        ));
    }

    #[test]
    fn device_seed_rejects_invalid_spec_and_camera_count_mismatch() {
        let mut bad = rig_spec();
        bad.version = 0;
        assert!(matches!(
            scheimpflug_seed(&bad, "cam0"),
            Err(DeviceSeedError::InvalidSpec(_))
        ));
        assert!(matches!(
            rig_layout_seed(&bad, &["cam0", "cam1"], &[]),
            Err(DeviceSeedError::InvalidSpec(_))
        ));

        let spec = rig_spec();
        assert!(matches!(
            rig_layout_seed(&spec, &["cam0", "cam1"], &[vec![Some(Iso3::identity())]]),
            Err(DeviceSeedError::ViewCameraCountMismatch {
                view: 0,
                got: 1,
                expected: 2,
            })
        ));
    }

    // ── DeviceSeed ──────────────────────────────────────────────────────────

    use serde_json::json;
    use vision_calibration_core::{BrownConrady5, make_pinhole_camera, pinhole_camera_params};

    use crate::common::calibrated_cameras;

    fn pose(rpy: [f64; 3], t: [f64; 3]) -> Iso3 {
        Iso3::from_parts(
            nalgebra::Translation3::new(t[0], t[1], t[2]),
            nalgebra::UnitQuaternion::from_euler_angles(rpy[0], rpy[1], rpy[2]),
        )
    }

    /// A Brown-Conrady camera, tilted when `tilt` is given.
    fn calibrated(fx: f64, k1: f64, tilt: Option<f64>) -> CameraParams {
        let k = FxFyCxCy {
            fx,
            fy: fx * 1.001,
            cx: 640.5,
            cy: 480.25,
        };
        let distortion = BrownConrady5 {
            k1,
            k2: 0.02,
            p1: 1e-4,
            ..Default::default()
        };
        let camera = pinhole_camera_params(&make_pinhole_camera(k, distortion));
        match tilt {
            Some(tilt_x) => camera.with_sensor(SensorParams::Scheimpflug {
                params: ScheimpflugParams {
                    tilt_x,
                    tilt_y: -0.01,
                },
            }),
            None => camera,
        }
    }

    fn plane(d: f64) -> LaserPlane {
        LaserPlane::new(nalgebra::Vector3::new(0.0, 0.6, 0.8), d).unwrap()
    }

    /// A two-camera rig seed: a calibrated tilted camera with a plane and a
    /// nominal one, mounted, on a gripper.
    fn rig_seed() -> DeviceSeed {
        let mut cam0 = SeedCamera::new(
            "cam0",
            [1280, 960],
            SeedLens::Calibrated {
                camera: calibrated(1500.0, -0.3, Some(0.05)),
            },
        );
        cam0.cam_se3_rig = Some(pose([0.1, -0.2, 0.3], [0.05, -0.01, 0.002]));
        cam0.laser_plane = Some(plane(-0.3));
        let mut cam1 = SeedCamera::new(
            "cam1",
            [1280, 960],
            SeedLens::Nominal {
                intrinsics: FxFyCxCy {
                    fx: 1400.0,
                    fy: 1400.0,
                    cx: 640.0,
                    cy: 480.0,
                },
                sensor: ScheimpflugParams::default(),
            },
        );
        cam1.cam_se3_rig = Some(pose([-0.05, 0.4, -0.1], [-0.12, 0.03, 0.01]));
        let mut seed = DeviceSeed::new(vec![cam0, cam1]);
        seed.handeye = Some(SeedHandeye::EyeInHand {
            gripper_se3_rig: pose([0.2, 0.1, -0.3], [0.01, 0.02, 0.15]),
        });
        seed.description = Some("unit 7".to_string());
        seed
    }

    /// Debug text: equal text is equal bits (`-0.0` prints as such).
    fn debug<T: std::fmt::Debug>(value: &T) -> String {
        format!("{value:?}")
    }

    fn assert_pose_near(a: &Iso3, b: &Iso3, what: &str) {
        let dt = (a.translation.vector - b.translation.vector).norm();
        let dr = a.rotation.angle_to(&b.rotation);
        assert!(dt < 1e-12 && dr < 1e-12, "{what}: Δt {dt:e}, Δr {dr:e}");
    }

    /// a seed reads back from JSON and TOML as it was written, with
    /// both lens kinds, both hand-eye modes and planes; unknown fields and
    /// other versions are refused.
    #[test]
    fn seed_round_trips_json_and_toml() {
        let mut eye_to_hand = rig_seed();
        eye_to_hand.handeye = Some(SeedHandeye::EyeToHand {
            rig_se3_base: pose([0.0, 0.3, 1.0], [0.4, -0.2, 0.9]),
        });
        for seed in [rig_seed(), eye_to_hand] {
            let json = serde_json::to_string(&seed).unwrap();
            let back: DeviceSeed = serde_json::from_str(&json).unwrap();
            assert_eq!(debug(&back), debug(&seed));
            let text = toml::to_string(&seed).unwrap();
            let back: DeviceSeed = toml::from_str(&text).unwrap();
            assert_eq!(debug(&back), debug(&seed));
            back.validate().unwrap();
        }

        let value = serde_json::to_value(rig_seed()).unwrap();
        assert_eq!(value["cameras"][0]["lens"]["kind"], "calibrated");
        assert_eq!(value["cameras"][1]["lens"]["kind"], "nominal");
        assert_eq!(value["handeye"]["mode"], "eye_in_hand");
        assert!(value["cameras"][1].get("laser_plane").is_none());

        let unknown: [fn(&mut serde_json::Value); 4] = [
            |v| v["extra"] = json!(1),
            |v| v["cameras"][0]["extra"] = json!(1),
            |v| v["cameras"][1]["lens"]["extra"] = json!(1),
            |v| v["handeye"]["extra"] = json!(1),
        ];
        for (index, add) in unknown.iter().enumerate() {
            let mut bad = value.clone();
            add(&mut bad);
            assert!(
                serde_json::from_value::<DeviceSeed>(bad).is_err(),
                "unknown field {index} is refused"
            );
        }

        let mut newer = rig_seed();
        newer.version = DEVICE_SEED_VERSION + 1;
        assert!(matches!(
            newer.validate(),
            Err(DeviceSeedError::UnsupportedVersion {
                found: 2,
                supported: 1
            })
        ));
    }

    /// every free function's value is the `DeviceSeed` path's, bit
    /// for bit (the reference camera `cam1` is at the drawing's origin).
    #[test]
    fn free_functions_equal_the_seed_path() {
        let spec = rig_spec();
        let seed = DeviceSeed::from_spec(&spec).unwrap();
        let ids = ["cam0", "cam1"];

        for id in ids {
            assert_eq!(
                debug(&scheimpflug_seed(&spec, id).unwrap()),
                debug(&seed.intrinsics_init(id).unwrap())
            );
        }
        assert_eq!(
            debug(&rig_intrinsics_seed(&spec, &ids).unwrap()),
            debug(&seed.rig_intrinsics_init(&ids).unwrap())
        );
        let nominal = nominal_cam_se3_rig(&spec, &ids).unwrap();
        for (id, mount) in ids.iter().zip(&nominal) {
            assert_eq!(
                debug(&seed.camera(id).unwrap().cam_se3_rig.unwrap()),
                debug(mount)
            );
        }
        // cam1 is mounted at the drawing's origin: re-expressing the seed
        // in its frame changes nothing.
        assert_eq!(debug(&seed.cam_se3_rig(&ids, 1).unwrap()), debug(&nominal));
        let target = pose([0.1, 0.2, 0.3], [0.01, 0.02, 0.5]);
        let views = vec![vec![Some(target), None], vec![None, Some(target)]];
        assert_eq!(
            debug(&rig_layout_seed(&spec, &ids, &views).unwrap()),
            debug(&seed.rig_init(&ids, 1, &views).unwrap())
        );
        assert_eq!(
            debug(&handeye_seed(&spec, HandeyeMode::EyeToHand).unwrap()),
            debug(&seed.handeye_init(&ids, 1, HandeyeMode::EyeToHand).unwrap())
        );
        assert_eq!(seed.cameras[0].image_size, [720, 540]);
    }

    #[test]
    fn from_spec_requires_whole_mounts_and_keeps_the_description() {
        let mut spec = rig_spec();
        spec.description = Some("drawing rev C".to_string());
        assert_eq!(
            DeviceSeed::from_spec(&spec).unwrap().description.as_deref(),
            Some("drawing rev C")
        );

        let mut partial = rig_spec();
        partial.rig.as_mut().unwrap().cameras.pop();
        assert!(matches!(
            DeviceSeed::from_spec(&partial),
            Err(DeviceSeedError::PartialMounts)
        ));

        let mut unmounted = rig_spec();
        unmounted.rig.as_mut().unwrap().cameras.clear();
        assert!(matches!(
            DeviceSeed::from_spec(&unmounted),
            Err(DeviceSeedError::HandeyeWithoutMounts)
        ));

        let mut bad = rig_spec();
        bad.cameras[0].focal_mm = -1.0;
        assert!(matches!(
            DeviceSeed::from_spec(&bad),
            Err(DeviceSeedError::InvalidSpec(_))
        ));
    }

    /// Synthetic exports of every problem kind; two cameras for the rigs.
    fn exports() -> Vec<(&'static str, serde_json::Value)> {
        let cam = |fx: f64| serde_json::to_value(calibrated(fx, -0.2, None)).unwrap();
        let iso = |p: Iso3| serde_json::to_value(p).unwrap();
        let mounts = json!([
            iso(Iso3::identity()),
            iso(pose([0.0, 0.1, 0.0], [-0.1, 0.0, 0.0]))
        ]);
        let planes = json!([
            serde_json::to_value(plane(-0.3)).unwrap(),
            serde_json::to_value(plane(-0.4)).unwrap()
        ]);
        let stats = json!({
            "mean_laser_error": 0.1,
            "mean_reproj_error": 0.1,
            "per_view_laser_errors": [0.1],
            "per_view_reproj_errors": [0.1],
        });
        let handeye_pose = iso(pose([0.2, 0.1, -0.3], [0.01, 0.02, 0.15]));
        let target = iso(pose([0.0, 0.0, 0.1], [0.0, 0.0, 1.0]));
        let with = |kind: &str, fields: serde_json::Value| {
            let mut value = json!({
                "kind": kind,
                "schema_version": 1,
                "mean_reproj_error": 0.1,
                "per_cam_reproj_errors": [0.1, 0.1],
                "cam_se3_target": [target],
                "rig_se3_target": [target],
            });
            for (key, field) in fields.as_object().unwrap() {
                value[key] = field.clone();
            }
            (kind_name(kind), value)
        };
        vec![
            with("planar_intrinsics", json!({"camera": cam(800.0)})),
            with(
                "single_cam_handeye",
                json!({
                    "camera": cam(800.0),
                    "handeye": {
                        "mode": "eye_in_hand",
                        "gripper_se3_cam": handeye_pose,
                        "base_se3_target": target,
                    },
                }),
            ),
            with(
                "laserline_device",
                json!({
                    "camera": cam(800.0),
                    "laser_plane": planes[0],
                    "stats": stats,
                }),
            ),
            with(
                "rig_extrinsics",
                json!({"cameras": [cam(800.0), cam(810.0)], "cam_se3_rig": mounts}),
            ),
            with(
                "rig_handeye",
                json!({
                    "cameras": [cam(800.0), cam(810.0)],
                    "cam_se3_rig": mounts,
                    "handeye": {
                        "mode": "eye_to_hand",
                        "rig_se3_base": handeye_pose,
                        "gripper_se3_target": target,
                    },
                }),
            ),
            with(
                "rig_laserline_device",
                json!({
                    "cameras": [cam(800.0), cam(810.0)],
                    "cam_se3_rig": mounts,
                    "laser_planes_cam": planes,
                    "laser_planes_rig": planes,
                    "per_cam_stats": [],
                }),
            ),
            with(
                "rig_handeye_laserline",
                json!({
                    "cameras": [cam(800.0), cam(810.0)],
                    "cam_se3_rig": mounts,
                    "handeye": {
                        "mode": "eye_in_hand",
                        "gripper_se3_rig": handeye_pose,
                        "base_se3_target": target,
                    },
                    "laser_planes_cam": planes,
                    "laser_planes_rig": planes,
                    "per_cam_stats": [],
                }),
            ),
        ]
    }

    fn kind_name(kind: &str) -> &'static str {
        [
            "planar_intrinsics",
            "single_cam_handeye",
            "laserline_device",
            "rig_extrinsics",
            "rig_handeye",
            "rig_laserline_device",
            "rig_handeye_laserline",
        ]
        .into_iter()
        .find(|name| *name == kind)
        .expect("a problem kind")
    }

    /// a calibration seeds exactly its device quantities — the lenses
    /// whole, the mounts of a rig, the planes and the hand-eye — and nothing
    /// of the capture.
    #[test]
    fn from_calibration_carries_the_device_and_nothing_per_view() {
        for (kind, value) in exports() {
            let export =
                calibrated_cameras(&value.to_string()).unwrap_or_else(|e| panic!("{kind}: {e}"));
            let count = export.camera_params().len();
            let ids: Vec<String> = (0..count).map(|i| format!("cam{i}")).collect();
            let sizes = vec![[1280, 960]; count];
            let seed = DeviceSeed::from_calibration(export.as_ref(), &ids, &sizes).unwrap();

            assert_eq!(seed.cameras.len(), count, "{kind}");
            for (camera, params) in seed.cameras.iter().zip(export.camera_params()) {
                let SeedLens::Calibrated { camera: lens } = &camera.lens else {
                    panic!("{kind}: a calibrated lens");
                };
                assert_eq!(debug(lens), debug(&params), "{kind}: the camera, whole");
                assert_eq!(camera.image_size, [1280, 960]);
            }
            let rig = count > 1;
            for (camera, mount) in seed.cameras.iter().zip(export.cam_se3_rig()) {
                match camera.cam_se3_rig {
                    Some(seeded) => assert_eq!(debug(&seeded), debug(&mount), "{kind}"),
                    None => assert!(!rig, "{kind}: a rig carries its mounts"),
                }
            }
            assert_eq!(
                seed.cameras.iter().all(|c| c.laser_plane.is_some()),
                kind.contains("laser"),
                "{kind}: the laser planes"
            );
            assert_eq!(seed.handeye.is_some(), kind.contains("handeye"), "{kind}");

            let text = serde_json::to_string(&seed).unwrap();
            for capture in [
                "cam_se3_target",
                "rig_se3_target",
                "base_se3_target",
                "gripper_se3_target",
                "uncertainty",
                "reproj",
            ] {
                assert!(!text.contains(capture), "{kind}: carries {capture}");
            }
        }
    }

    #[test]
    fn from_calibration_hand_eye_frames() {
        let exports = exports();
        let single = calibrated_cameras(&exports[1].1.to_string()).unwrap();
        let seed =
            DeviceSeed::from_calibration(single.as_ref(), &["cam0".to_string()], &[[640, 480]])
                .unwrap();
        assert!(seed.cameras[0].cam_se3_rig.is_none(), "a rig of one");
        // A single camera is a rig of one: the camera's frame is the rig's.
        let single_cam_handeye::HandeyeManualInit::EyeInHand {
            gripper_se3_cam, ..
        } = seed
            .single_cam_handeye_init("cam0", HandeyeMode::EyeInHand)
            .unwrap()
        else {
            panic!("eye-in-hand");
        };
        assert_eq!(
            debug(&gripper_se3_cam.unwrap()),
            debug(&pose([0.2, 0.1, -0.3], [0.01, 0.02, 0.15]))
        );

        // One camera of a rig seeds a single-camera hand-eye of it:
        // gripper_se3_cam1 = gripper_se3_rig · cam1_se3_rig⁻¹.
        let rig = rig_seed();
        let gripper_se3_rig = *rig.handeye.unwrap().pose();
        let cam1_se3_rig = rig.cameras[1].cam_se3_rig.unwrap();
        let single_cam_handeye::HandeyeManualInit::EyeInHand {
            gripper_se3_cam, ..
        } = rig
            .single_cam_handeye_init("cam1", HandeyeMode::EyeInHand)
            .unwrap()
        else {
            panic!("eye-in-hand");
        };
        // The same camera point lands at the same gripper point both ways.
        let p = nalgebra::Point3::new(0.1, -0.2, 0.7);
        let direct = gripper_se3_cam.unwrap() * p;
        let through_rig = gripper_se3_rig * (cam1_se3_rig.inverse() * p);
        assert!((direct - through_rig).norm() < 1e-14);

        let mut to_hand = rig_seed();
        to_hand.handeye = Some(SeedHandeye::EyeToHand {
            rig_se3_base: pose([0.0, 0.3, 1.0], [0.4, -0.2, 0.9]),
        });
        let single_cam_handeye::HandeyeManualInit::EyeToHand { cam_se3_base, .. } = to_hand
            .single_cam_handeye_init("cam1", HandeyeMode::EyeToHand)
            .unwrap()
        else {
            panic!("eye-to-hand");
        };
        let expected = cam1_se3_rig * *to_hand.handeye.unwrap().pose();
        assert_pose_near(&cam_se3_base.unwrap(), &expected, "cam_se3_base");
    }

    #[test]
    fn from_calibration_refuses_what_a_seed_cannot_hold() {
        let exports = exports();
        let rig = calibrated_cameras(&exports[3].1.to_string()).unwrap();
        assert!(matches!(
            DeviceSeed::from_calibration(rig.as_ref(), &["cam0".to_string()], &[[1, 1]; 2]),
            Err(DeviceSeedError::ExportCameraCount {
                export: 2,
                camera_ids: 1,
                image_sizes: 2
            })
        ));
        let ids = ["a".to_string(), "a".to_string()];
        assert!(matches!(
            DeviceSeed::from_calibration(rig.as_ref(), &ids, &[[1, 1]; 2]),
            Err(DeviceSeedError::DuplicateCameraId { .. })
        ));

        let mut planar = exports[0].1.clone();
        planar["camera"]["sensor"] = json!({
            "type": "homography",
            "h": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        });
        let planar = calibrated_cameras(&planar.to_string()).unwrap();
        assert!(matches!(
            DeviceSeed::from_calibration(planar.as_ref(), &["cam0".to_string()], &[[1, 1]]),
            Err(DeviceSeedError::UnseedableSensor { .. })
        ));
    }

    /// The derivations seed a calibrated lens whole and a nominal lens as
    /// the datasheet path does.
    #[test]
    fn lens_derivations() {
        let seed = rig_seed();
        let init = seed.intrinsics_init("cam0").unwrap();
        let SeedLens::Calibrated { camera } = &seed.cameras[0].lens else {
            unreachable!("cam0 is calibrated")
        };
        assert_eq!(debug(&init.intrinsics.unwrap()), debug(&camera.intrinsics));
        assert_eq!(debug(&init.distortion.unwrap()), debug(&camera.distortion));
        assert_eq!(init.sensor.unwrap().tilt_x, 0.05);
        assert!(init.cam_se3_target.is_none());

        let nominal = seed.intrinsics_init("cam1").unwrap();
        assert!(nominal.distortion.is_none());
        assert_eq!(nominal.sensor.unwrap().tilt_x, 0.0);

        // A calibrated untilted camera seeds no tilt.
        let mut untilted = rig_seed();
        untilted.cameras[0].lens = SeedLens::Calibrated {
            camera: calibrated(1500.0, -0.3, None),
        };
        assert!(untilted.intrinsics_init("cam0").unwrap().sensor.is_none());

        // Device: the lens and the plane.
        let device = seed.device_init("cam0").unwrap();
        assert!(device.distortion.is_some() && device.laser_plane.is_some());
        assert!(seed.device_init("cam1").unwrap().laser_plane.is_none());

        // Rig: distortion all or none, zero tilts for untilted cameras.
        assert!(matches!(
            seed.rig_intrinsics_init(&["cam0", "cam1"]),
            Err(DeviceSeedError::PartialField {
                field: "distortion"
            })
        ));
        let init = seed
            .nominal()
            .rig_intrinsics_init(&["cam1", "cam0"])
            .unwrap();
        assert!(init.per_cam_distortion.is_none());
        let sensors = init.per_cam_sensors.unwrap();
        assert_eq!((sensors[0].tilt_x, sensors[1].tilt_x), (0.0, 0.05));

        let mut both = rig_seed();
        both.cameras[1].lens = SeedLens::Calibrated {
            camera: calibrated(1400.0, -0.1, None),
        };
        let init = both.rig_intrinsics_init(&["cam0", "cam1"]).unwrap();
        assert_eq!(init.per_cam_distortion.unwrap().len(), 2);
        assert_eq!(init.per_cam_sensors.unwrap()[1].tilt_x, 0.0);
        both.cameras[0].lens = SeedLens::Calibrated {
            camera: calibrated(1500.0, -0.3, None),
        };
        let init = both.rig_intrinsics_init(&["cam0", "cam1"]).unwrap();
        assert!(init.per_cam_sensors.is_none(), "no tilt to seed");

        // Planes: all or none.
        assert!(matches!(
            seed.plane_init(&["cam0", "cam1"]),
            Err(DeviceSeedError::PartialField {
                field: "laser_plane"
            })
        ));
        assert!(
            seed.plane_init(&["cam1"])
                .unwrap()
                .laser_planes_cam
                .is_none()
        );
        let planes = seed
            .plane_init(&["cam0"])
            .unwrap()
            .laser_planes_cam
            .unwrap();
        assert_eq!(planes[0].distance(), -0.3);
    }

    #[test]
    fn nominal_keeps_intrinsics_tilt_and_the_device() {
        let seed = rig_seed().nominal();
        let SeedLens::Nominal { intrinsics, sensor } = &seed.cameras[0].lens else {
            panic!("a nominal lens");
        };
        assert_eq!(intrinsics.fx, 1500.0);
        assert_eq!((sensor.tilt_x, sensor.tilt_y), (0.05, -0.01));
        assert!(seed.cameras[0].laser_plane.is_some());
        assert!(seed.cameras[0].cam_se3_rig.is_some());
        assert!(seed.handeye.is_some());
        seed.validate().unwrap();

        let mut untilted = rig_seed();
        untilted.cameras[0].lens = SeedLens::Calibrated {
            camera: calibrated(1500.0, -0.3, None),
        };
        let SeedLens::Nominal { sensor, .. } = untilted.nominal().cameras[0].lens else {
            panic!("a nominal lens");
        };
        assert_eq!((sensor.tilt_x, sensor.tilt_y), (0.0, 0.0));
    }

    #[test]
    fn select_keeps_the_rig_frame() {
        let seed = rig_seed();
        let one = seed.select(&["cam1"]).unwrap();
        assert_eq!(one.cameras.len(), 1);
        assert_eq!(
            debug(&one.cameras[0].cam_se3_rig),
            debug(&seed.cameras[1].cam_se3_rig)
        );
        assert_eq!(debug(&one.handeye), debug(&seed.handeye));
        let swapped = seed.select(&["cam1", "cam0"]).unwrap();
        assert_eq!(swapped.cameras[0].id, "cam1");
        assert!(matches!(
            seed.select(&["cam9"]),
            Err(DeviceSeedError::UnknownCameraId { .. })
        ));
        assert!(matches!(
            seed.select(&["cam0", "cam0"]),
            Err(DeviceSeedError::DuplicateCameraId { .. })
        ));
        assert!(matches!(seed.select(&[]), Err(DeviceSeedError::NoCameras)));
    }

    #[test]
    fn validate_reports_each_defect() {
        assert!(matches!(
            DeviceSeed::new(Vec::new()).validate(),
            Err(DeviceSeedError::NoCameras)
        ));
        let mut duplicate = rig_seed();
        duplicate.cameras[1].id = "cam0".to_string();
        assert!(matches!(
            duplicate.validate(),
            Err(DeviceSeedError::DuplicateCameraId { .. })
        ));
        let mut nan = rig_seed();
        nan.cameras[1].lens = SeedLens::Nominal {
            intrinsics: FxFyCxCy {
                fx: f64::NAN,
                fy: 1.0,
                cx: 1.0,
                cy: 1.0,
            },
            sensor: ScheimpflugParams::default(),
        };
        assert!(matches!(
            nan.validate(),
            Err(DeviceSeedError::NonFinite { .. })
        ));
        let mut nan_mount = rig_seed();
        nan_mount.cameras[0].cam_se3_rig = Some(pose([0.0; 3], [f64::NAN, 0.0, 0.0]));
        assert!(matches!(
            nan_mount.validate(),
            Err(DeviceSeedError::NonFinite { .. })
        ));
        let mut partial = rig_seed();
        partial.cameras[1].cam_se3_rig = None;
        assert!(matches!(
            partial.validate(),
            Err(DeviceSeedError::PartialMounts)
        ));
        let mut unmounted = rig_seed();
        for camera in &mut unmounted.cameras {
            camera.cam_se3_rig = None;
        }
        assert!(matches!(
            unmounted.validate(),
            Err(DeviceSeedError::HandeyeWithoutMounts)
        ));
        unmounted.handeye = None;
        unmounted.validate().unwrap();
        assert!(matches!(
            unmounted.cam_se3_rig(&["cam0", "cam1"], 0),
            Err(DeviceSeedError::MissingCameraMount { .. })
        ));
        let mut homography = rig_seed();
        homography.cameras[0].lens = SeedLens::Calibrated {
            camera: calibrated(1500.0, -0.3, None).with_sensor(SensorParams::Homography {
                h: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            }),
        };
        assert!(matches!(
            homography.validate(),
            Err(DeviceSeedError::UnseedableSensor { .. })
        ));
        // `nominal` does not turn a sensor a seed cannot carry into one.
        assert!(homography.nominal().validate().is_err());
    }

    #[test]
    fn derivation_errors() {
        let seed = rig_seed();
        let ids = ["cam0", "cam1"];
        assert!(matches!(
            seed.cam_se3_rig(&ids, 2),
            Err(DeviceSeedError::ReferenceOutOfRange {
                reference: 2,
                cameras: 2
            })
        ));
        assert!(matches!(
            seed.handeye_init(&ids, 0, HandeyeMode::EyeToHand),
            Err(DeviceSeedError::HandeyeModeMismatch { .. })
        ));
        let mut no_handeye = rig_seed();
        no_handeye.handeye = None;
        assert!(matches!(
            no_handeye.handeye_init(&ids, 0, HandeyeMode::EyeInHand),
            Err(DeviceSeedError::MissingHandeye)
        ));
        assert!(matches!(
            seed.intrinsics_init("cam9"),
            Err(DeviceSeedError::UnknownCameraId { .. })
        ));
        assert!(matches!(
            seed.rig_init(&ids, 0, &[vec![None, None]]),
            Err(DeviceSeedError::ViewWithoutTargetPose { view: 0 })
        ));
        // A single camera is its own reference.
        assert_eq!(
            debug(&seed.cam_se3_rig(&["cam1"], 0).unwrap()),
            debug(&vec![Iso3::identity()])
        );
    }

    /// a seed's derivations do not depend on its rig frame. The same
    /// device stated in its own frame and in the frame of `cam1` gives the
    /// same manual-init seeds for either reference camera.
    #[test]
    fn derivations_are_gauge_free() {
        let original = rig_seed();
        // The same device with cam1 at the origin.
        let c1 = original.cameras[1].cam_se3_rig.unwrap();
        let mut moved = original.clone();
        for camera in &mut moved.cameras {
            camera.cam_se3_rig = camera.cam_se3_rig.map(|c| c * c1.inverse());
        }
        let gripper_se3_rig = *original.handeye.unwrap().pose();
        moved.handeye = Some(SeedHandeye::EyeInHand {
            gripper_se3_rig: gripper_se3_rig * c1.inverse(),
        });
        let rig_se3_base = pose([0.0, 0.3, 1.0], [0.4, -0.2, 0.9]);
        let mut original_to_hand = original.clone();
        original_to_hand.handeye = Some(SeedHandeye::EyeToHand { rig_se3_base });
        let mut moved_to_hand = moved.clone();
        moved_to_hand.handeye = Some(SeedHandeye::EyeToHand {
            rig_se3_base: c1 * rig_se3_base,
        });

        let ids = ["cam0", "cam1"];
        let target = pose([0.1, 0.2, 0.3], [0.01, 0.02, 0.5]);
        let views = vec![vec![Some(target), None], vec![None, Some(target)]];
        let handeye_pose = |init: HandeyeManualInit| match init {
            HandeyeManualInit::EyeInHand {
                gripper_se3_rig, ..
            } => gripper_se3_rig.unwrap(),
            HandeyeManualInit::EyeToHand { rig_se3_base, .. } => rig_se3_base.unwrap(),
        };
        for reference in 0..2 {
            let a = original.cam_se3_rig(&ids, reference).unwrap();
            let b = moved.cam_se3_rig(&ids, reference).unwrap();
            assert_pose_near(&a[reference], &Iso3::identity(), "the reference");
            for (a, b) in a.iter().zip(&b) {
                assert_pose_near(a, b, "cam_se3_rig");
            }
            let poses = |seed: &DeviceSeed| {
                seed.rig_init(&ids, reference, &views)
                    .unwrap()
                    .poses
                    .unwrap()
                    .rig_se3_target
            };
            for (a, b) in poses(&original).iter().zip(&poses(&moved)) {
                assert_pose_near(a, b, "rig_se3_target");
            }
            for (a, b, mode) in [
                (&original, &moved, HandeyeMode::EyeInHand),
                (&original_to_hand, &moved_to_hand, HandeyeMode::EyeToHand),
            ] {
                assert_pose_near(
                    &handeye_pose(a.handeye_init(&ids, reference, mode).unwrap()),
                    &handeye_pose(b.handeye_init(&ids, reference, mode).unwrap()),
                    "hand-eye",
                );
            }
            // The chain is kept: camera `reference` sits on the gripper
            // where the original rig puts it.
            let gripper_se3_ref = handeye_pose(
                original
                    .handeye_init(&ids, reference, HandeyeMode::EyeInHand)
                    .unwrap(),
            );
            let expected =
                gripper_se3_rig * original.cameras[reference].cam_se3_rig.unwrap().inverse();
            assert_pose_near(&gripper_se3_ref, &expected, "gripper_se3_ref");
        }
    }
}
