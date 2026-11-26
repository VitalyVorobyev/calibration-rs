//! One call from a manifest to an export: [`calibrate_dataset`] picks the
//! converter and the problem type by the manifest's topology.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use thiserror::Error;
use vision_calibration_core::{
    CameraParams, FrameKind, FrameRef, ImageManifest, Iso3, LaserPlane, PixelRect, RunControl,
    Uncertainty, data_relative_path,
};
use vision_calibration_dataset::{DatasetSpec, ProblemKind};

use super::{
    Detection, HandeyeRunResult, LaserPixelExtractor, LaserlineRunResult, PlanarRunResult,
    RejectedDetection, RigHandeyeLaserlineRunResult, RigLaserlineRunResult, RigRunResult, RunError,
    build_laserline_device_input, build_planar_input, build_rig_extrinsics_input,
    build_rig_handeye_input, build_rig_handeye_laserline_input, build_rig_laserline_device_input,
    build_single_cam_handeye_input, camera_image_sizes, require_pose_pairing,
};
use crate::Error;
use crate::common::{CalibratedCameras, ExportDiagnostics};
use crate::laserline_device::{
    LaserlineDeviceConfig, LaserlineDeviceExport, LaserlineDeviceProblem,
};
use crate::planar_intrinsics::{
    PlanarIntrinsicsConfig, PlanarIntrinsicsExport, PlanarIntrinsicsProblem,
};
use crate::rig_extrinsics::{RigExtrinsicsConfig, RigExtrinsicsExport, RigExtrinsicsProblem};
use crate::rig_handeye::{RigHandeyeConfig, RigHandeyeExport, RigHandeyeProblem};
use crate::rig_handeye_laserline::{
    RigHandeyeLaserlineConfig, RigHandeyeLaserlineExport, RigHandeyeLaserlineProblem,
};
use crate::rig_laserline_device::{
    RigLaserlineDeviceConfig, RigLaserlineDeviceExport, RigLaserlineDeviceProblem,
};
use crate::session::{CalibrationSession, ProblemType};
use crate::single_cam_handeye::{
    SingleCamHandeyeConfig, SingleCamHandeyeExport, SingleCamHandeyeProblem,
};

/// The configuration of one calibration, tagged by the manifest topology it
/// applies to.
///
/// In JSON the variant is the `topology` field and the problem's config is
/// the `config` field:
/// `{"topology": "planar_intrinsics", "config": {...}}`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "topology", content = "config", rename_all = "snake_case")]
#[non_exhaustive]
pub enum CalibrationConfig {
    /// For a `planar_intrinsics` manifest.
    PlanarIntrinsics(PlanarIntrinsicsConfig),
    /// For a `single_cam_handeye` manifest.
    SingleCamHandeye(SingleCamHandeyeConfig),
    /// For a `laserline_device` manifest.
    LaserlineDevice(LaserlineDeviceConfig),
    /// For a `rig_extrinsics` manifest.
    RigExtrinsics(RigExtrinsicsConfig),
    /// For a `rig_handeye` manifest.
    RigHandeye(RigHandeyeConfig),
    /// For a `rig_laserline_device` manifest.
    RigLaserlineDevice(RigLaserlineDeviceConfig),
    /// For a `rig_handeye_laserline` manifest.
    RigHandeyeLaserline(RigHandeyeLaserlineConfig),
}

impl CalibrationConfig {
    /// Check the configuration on its own, as its problem's `set_config`
    /// does; [`calibrate_dataset`] runs it before any detection.
    fn validate(&self) -> Result<(), Error> {
        match self {
            Self::PlanarIntrinsics(c) => PlanarIntrinsicsProblem::validate_config(c),
            Self::SingleCamHandeye(c) => SingleCamHandeyeProblem::validate_config(c),
            Self::LaserlineDevice(c) => LaserlineDeviceProblem::validate_config(c),
            Self::RigExtrinsics(c) => RigExtrinsicsProblem::validate_config(c),
            Self::RigHandeye(c) => RigHandeyeProblem::validate_config(c),
            Self::RigLaserlineDevice(c) => RigLaserlineDeviceProblem::validate_config(c),
            Self::RigHandeyeLaserline(c) => RigHandeyeLaserlineProblem::validate_config(c),
        }
    }

    /// The default configuration of the problem type that `topology`
    /// selects.
    pub fn default_for(topology: ProblemKind) -> Self {
        match topology {
            ProblemKind::PlanarIntrinsics => Self::PlanarIntrinsics(Default::default()),
            ProblemKind::SingleCamHandeye => Self::SingleCamHandeye(Default::default()),
            ProblemKind::LaserlineDevice => Self::LaserlineDevice(Default::default()),
            ProblemKind::RigExtrinsics => Self::RigExtrinsics(Default::default()),
            ProblemKind::RigHandeye => Self::RigHandeye(Default::default()),
            ProblemKind::RigLaserlineDevice => Self::RigLaserlineDevice(Default::default()),
            ProblemKind::RigHandeyeLaserline => Self::RigHandeyeLaserline(Default::default()),
            other => unreachable!("{other:?} has no problem type in this version"),
        }
    }

    /// The manifest topology this configuration applies to.
    pub fn topology(&self) -> ProblemKind {
        match self {
            Self::PlanarIntrinsics(_) => ProblemKind::PlanarIntrinsics,
            Self::SingleCamHandeye(_) => ProblemKind::SingleCamHandeye,
            Self::LaserlineDevice(_) => ProblemKind::LaserlineDevice,
            Self::RigExtrinsics(_) => ProblemKind::RigExtrinsics,
            Self::RigHandeye(_) => ProblemKind::RigHandeye,
            Self::RigLaserlineDevice(_) => ProblemKind::RigLaserlineDevice,
            Self::RigHandeyeLaserline(_) => ProblemKind::RigHandeyeLaserline,
        }
    }
}

/// The export of one calibration, whichever problem type produced it.
///
/// It serializes as the problem's own export, whose `kind` field names the
/// problem type.
#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
#[non_exhaustive]
pub enum CalibrationExport {
    /// A planar intrinsics calibration.
    PlanarIntrinsics(PlanarIntrinsicsExport),
    /// A single-camera hand-eye calibration.
    SingleCamHandeye(SingleCamHandeyeExport),
    /// A laserline device calibration.
    LaserlineDevice(LaserlineDeviceExport),
    /// A rig extrinsics calibration.
    RigExtrinsics(RigExtrinsicsExport),
    /// A rig hand-eye calibration.
    RigHandeye(RigHandeyeExport),
    /// A rig laserline device calibration.
    RigLaserlineDevice(RigLaserlineDeviceExport),
    /// A joint rig hand-eye and laser-plane calibration.
    RigHandeyeLaserline(RigHandeyeLaserlineExport),
}

impl CalibrationExport {
    fn cameras(&self) -> &dyn CalibratedCameras {
        match self {
            Self::PlanarIntrinsics(e) => e,
            Self::SingleCamHandeye(e) => e,
            Self::LaserlineDevice(e) => e,
            Self::RigExtrinsics(e) => e,
            Self::RigHandeye(e) => e,
            Self::RigLaserlineDevice(e) => e,
            Self::RigHandeyeLaserline(e) => e,
        }
    }

    /// The export's diagnostics, to change before it is written: for
    /// instance its image manifest's root, restated with
    /// [`ImageManifest::rebased`] for the folder the export goes to.
    pub fn diagnostics_mut(&mut self) -> &mut ExportDiagnostics {
        match self {
            Self::PlanarIntrinsics(e) => &mut e.diagnostics,
            Self::SingleCamHandeye(e) => &mut e.diagnostics,
            Self::LaserlineDevice(e) => &mut e.diagnostics,
            Self::RigExtrinsics(e) => &mut e.diagnostics,
            Self::RigHandeye(e) => &mut e.diagnostics,
            Self::RigLaserlineDevice(e) => &mut e.diagnostics,
            Self::RigHandeyeLaserline(e) => &mut e.diagnostics,
        }
    }

    /// The problem type that produced this export.
    pub fn kind(&self) -> ProblemKind {
        match self {
            Self::PlanarIntrinsics(_) => ProblemKind::PlanarIntrinsics,
            Self::SingleCamHandeye(_) => ProblemKind::SingleCamHandeye,
            Self::LaserlineDevice(_) => ProblemKind::LaserlineDevice,
            Self::RigExtrinsics(_) => ProblemKind::RigExtrinsics,
            Self::RigHandeye(_) => ProblemKind::RigHandeye,
            Self::RigLaserlineDevice(_) => ProblemKind::RigLaserlineDevice,
            Self::RigHandeyeLaserline(_) => ProblemKind::RigHandeyeLaserline,
        }
    }
}

impl crate::common::sealed::Sealed for CalibrationExport {}

/// Every problem's export reports its cameras, so [`assess`](crate::analysis::assess)
/// and the other analyses take a `CalibrationExport` directly.
impl CalibratedCameras for CalibrationExport {
    fn camera_params(&self) -> Vec<CameraParams> {
        self.cameras().camera_params()
    }

    fn cam_se3_rig(&self) -> Vec<Iso3> {
        self.cameras().cam_se3_rig()
    }

    fn laser_planes_cam(&self) -> Vec<LaserPlane> {
        self.cameras().laser_planes_cam()
    }

    fn handeye(&self) -> Option<vision_calibration_core::Handeye> {
        self.cameras().handeye()
    }

    fn parameter_uncertainty(&self) -> &Uncertainty {
        self.cameras().parameter_uncertainty()
    }

    fn diagnostics(&self) -> &ExportDiagnostics {
        self.cameras().diagnostics()
    }
}

/// The result of [`calibrate_dataset`]: the export and what detection kept.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct DatasetCalibration {
    /// The calibration's export.
    pub export: CalibrationExport,
    /// Each camera's image size in pixels, `[width, height]`, in camera
    /// order ([`camera_image_sizes`](super::camera_image_sizes)): the
    /// `image_sizes` that [`assess`](crate::analysis::assess) takes.
    pub image_sizes: Vec<[u32; 2]>,
    /// Views that entered the calibration.
    pub usable_views: usize,
    /// Views the manifest's images formed before screening.
    pub total_views: usize,
    /// Detections left out, each with its camera, image and reason.
    pub rejected: Vec<RejectedDetection>,
}

/// Why [`calibrate_dataset`] failed.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum CalibrateDatasetError {
    /// The manifest is invalid, or detection or pairing failed, before any
    /// solve. [`RunError::AskUser`] is the case where the manifest leaves a
    /// choice open.
    #[error(transparent)]
    Run(RunError),
    /// The configuration is for a different topology than the manifest's.
    #[error("the config is for a {config:?} dataset but the manifest's topology is {manifest:?}")]
    ConfigTopologyMismatch {
        /// The manifest's topology.
        manifest: ProblemKind,
        /// The topology the configuration applies to.
        config: ProblemKind,
    },
    /// A laser topology was given no laser-line extractor.
    #[error("a {topology:?} manifest needs a laser-line extractor")]
    LaserExtractorRequired {
        /// The manifest's topology.
        topology: ProblemKind,
    },
    /// The configuration is invalid, the problem rejected the detected
    /// input, or the calibration itself failed.
    #[error(transparent)]
    Calibration(Error),
    /// The run control's token was cancelled, during detection or the
    /// calibration.
    #[error("cancelled")]
    Cancelled,
}

impl From<RunError> for CalibrateDatasetError {
    fn from(err: RunError) -> Self {
        match err {
            RunError::Cancelled => Self::Cancelled,
            other => Self::Run(other),
        }
    }
}

impl From<Error> for CalibrateDatasetError {
    fn from(err: Error) -> Self {
        match err {
            Error::Cancelled => Self::Cancelled,
            other => Self::Calibration(other),
        }
    }
}

/// Calibrates the dataset a manifest describes.
///
/// The manifest's `topology` selects the converter (detection, pairing and
/// screening) and the problem type; `config` is that problem's
/// configuration, or `None` for its defaults. The problem's
/// `run_calibration` runs every step and the session's export is returned.
/// Relative paths in the manifest resolve against `base_dir`, usually the
/// directory that holds it.
///
/// The export's `diagnostics.image_manifest` names the image of each kept
/// `(view, camera)` slot, and of each laser frame, in the input's view
/// order: the identity of every view the calibration used. Its root is
/// `.`, standing for `base_dir`, and each frame's path is relative to
/// `base_dir` (absolute for an image outside it). An export written to
/// another folder restates the root for it with
/// [`ImageManifest::rebased`] through
/// [`CalibrationExport::diagnostics_mut`].
///
/// `laser_extractor` is required by the laser topologies
/// (`laserline_device`, `rig_laserline_device`, `rig_handeye_laserline`) and
/// unused by the others.
///
/// `control` reports each image read ([`Progress::Detection`],
/// [`Progress::LaserExtraction`]), each calibration step
/// ([`Progress::Step`]) and each solver iteration
/// ([`Progress::Iteration`]), and is polled at each of them;
/// `&RunControl::default()` reports nothing and never stops early.
///
/// [`Progress::Detection`]: vision_calibration_core::Progress::Detection
/// [`Progress::LaserExtraction`]: vision_calibration_core::Progress::LaserExtraction
/// [`Progress::Step`]: vision_calibration_core::Progress::Step
/// [`Progress::Iteration`]: vision_calibration_core::Progress::Iteration
///
/// # Errors
///
/// - [`CalibrateDatasetError::ConfigTopologyMismatch`] and
///   [`CalibrateDatasetError::LaserExtractorRequired`] before any image is
///   read.
/// - [`CalibrateDatasetError::Run`]: an invalid manifest, a camera whose
///   images differ in size ([`RunError::ImageSizeMismatch`]), a choice the
///   manifest leaves open ([`RunError::AskUser`]), or detection and pairing
///   failures.
/// - [`CalibrateDatasetError::Calibration`] from the session.
/// - [`CalibrateDatasetError::Cancelled`] once `control`'s token is
///   cancelled.
///
/// # Example
///
/// ```no_run
/// use std::path::Path;
/// use vision_calibration_dataset::DatasetSpec;
/// use vision_calibration_detect::FsDetectionCache;
/// use vision_calibration_core::RunControl;
/// use vision_calibration_pipeline::dataset_runner::{Detection, calibrate_dataset};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let dir = Path::new("my_dataset");
/// let spec = DatasetSpec::from_path(dir.join("dataset.toml"))?;
/// let cache = FsDetectionCache::new(dir.join(".detections"));
/// let control = RunControl::default();
/// let run = calibrate_dataset(&spec, dir, None, Detection::Cached(&cache), None, &control)?;
/// println!("{} of {} views used", run.usable_views, run.total_views);
/// println!("{}", serde_json::to_string_pretty(&run.export)?);
/// # Ok(())
/// # }
/// ```
pub fn calibrate_dataset(
    spec: &DatasetSpec,
    base_dir: &Path,
    config: Option<CalibrationConfig>,
    detection: Detection<'_>,
    laser_extractor: Option<&dyn LaserPixelExtractor>,
    control: &RunControl,
) -> Result<DatasetCalibration, CalibrateDatasetError> {
    let topology = spec.topology;
    let config = config.unwrap_or_else(|| CalibrationConfig::default_for(topology));
    if config.topology() != topology {
        return Err(CalibrateDatasetError::ConfigTopologyMismatch {
            manifest: topology,
            config: config.topology(),
        });
    }
    // An invalid config fails here, before any image is read.
    config.validate()?;
    let laser =
        || laser_extractor.ok_or(CalibrateDatasetError::LaserExtractorRequired { topology });
    let needs_laser = matches!(
        config,
        CalibrationConfig::LaserlineDevice(_)
            | CalibrationConfig::RigLaserlineDevice(_)
            | CalibrationConfig::RigHandeyeLaserline(_)
    );
    if needs_laser {
        laser()?;
    }
    // An open choice in the manifest is asked before any image is read.
    vision_calibration_dataset::validate(spec).map_err(RunError::from)?;
    require_pose_pairing(spec)?;
    // A camera whose images differ in size fails here, before any detection.
    let image_sizes = camera_image_sizes(spec, base_dir)?;

    // Each arm builds its problem's input from the manifest, runs every step
    // and wraps the export. `config.topology() == topology` was checked
    // above, so each arm sees the config variant of its own topology.
    macro_rules! calibrate {
        ($config:ident, $variant:ident, $module:ident, $build:expr) => {{
            let run = $build?;
            let image_manifest = run.image_manifest(spec, base_dir);
            let mut export = solve(run.input, $config, crate::$module::run_calibration, control)?;
            export.diagnostics.image_manifest = Some(image_manifest);
            Ok(DatasetCalibration {
                export: CalibrationExport::$variant(export),
                image_sizes,
                usable_views: run.usable_views,
                total_views: run.total_views,
                rejected: run.rejected,
            })
        }};
    }
    match config {
        CalibrationConfig::PlanarIntrinsics(c) => calibrate!(
            c,
            PlanarIntrinsics,
            planar_intrinsics,
            build_planar_input(spec, base_dir, detection, control)
        ),
        CalibrationConfig::SingleCamHandeye(c) => calibrate!(
            c,
            SingleCamHandeye,
            single_cam_handeye,
            build_single_cam_handeye_input(spec, base_dir, detection, control)
        ),
        CalibrationConfig::LaserlineDevice(c) => calibrate!(
            c,
            LaserlineDevice,
            laserline_device,
            build_laserline_device_input(spec, base_dir, detection, laser()?, control)
        ),
        CalibrationConfig::RigExtrinsics(c) => calibrate!(
            c,
            RigExtrinsics,
            rig_extrinsics,
            build_rig_extrinsics_input(spec, base_dir, detection, control)
        ),
        CalibrationConfig::RigHandeye(c) => calibrate!(
            c,
            RigHandeye,
            rig_handeye,
            build_rig_handeye_input(spec, base_dir, detection, control)
        ),
        CalibrationConfig::RigLaserlineDevice(c) => calibrate!(
            c,
            RigLaserlineDevice,
            rig_laserline_device,
            build_rig_laserline_device_input(spec, base_dir, detection, laser()?, control)
        ),
        CalibrationConfig::RigHandeyeLaserline(c) => calibrate!(
            c,
            RigHandeyeLaserline,
            rig_handeye_laserline,
            build_rig_handeye_laserline_input(spec, base_dir, detection, laser()?, control)
        ),
    }
}

/// A converter's run result as the export's image manifest: one frame per
/// kept `(view, camera)` slot, in the input's view order, and one laser
/// frame per slot with laser pixels. The root is `.`, the dataset folder
/// `base_dir`; each path is relative to it ([`data_relative_path`]).
trait RunFrames {
    fn image_manifest(&self, spec: &DatasetSpec, base_dir: &Path) -> ImageManifest;
}

/// Frames of a one-camera run: the kept views' target images, then the
/// laser images aligned with them.
fn one_camera_manifest(
    spec: &DatasetSpec,
    base_dir: &Path,
    view_paths: &[PathBuf],
    laser_paths: &[PathBuf],
) -> ImageManifest {
    let roi = spec.cameras.first().and_then(|c| c.roi_xywh);
    let target = view_paths
        .iter()
        .enumerate()
        .map(|(view, path)| frame(base_dir, view, 0, path, roi, FrameKind::Target));
    let laser = laser_paths
        .iter()
        .enumerate()
        .map(|(view, path)| frame(base_dir, view, 0, path, roi, FrameKind::Laser));
    ImageManifest::new(".", target.chain(laser).collect())
}

/// Frames of a rig run, `paths[view][camera]`: the target slots, then the
/// laser slots.
fn rig_manifest(
    spec: &DatasetSpec,
    base_dir: &Path,
    view_paths: &[Vec<Option<PathBuf>>],
    laser_paths: &[Vec<Option<PathBuf>>],
) -> ImageManifest {
    let slots = |paths: &'_ [Vec<Option<PathBuf>>], kind: FrameKind| {
        paths
            .iter()
            .enumerate()
            .flat_map(move |(view, cameras)| {
                cameras
                    .iter()
                    .enumerate()
                    .filter_map(move |(camera, path)| {
                        let roi = spec.cameras.get(camera).and_then(|c| c.roi_xywh);
                        path.as_ref()
                            .map(|path| frame(base_dir, view, camera, path, roi, kind))
                    })
            })
            .collect::<Vec<_>>()
    };
    let mut frames = slots(view_paths, FrameKind::Target);
    frames.extend(slots(laser_paths, FrameKind::Laser));
    ImageManifest::new(".", frames)
}

fn frame(
    base_dir: &Path,
    view: usize,
    camera: usize,
    path: &Path,
    roi: Option<[u32; 4]>,
    kind: FrameKind,
) -> FrameRef {
    let mut frame = FrameRef::new(view, camera, data_relative_path(base_dir, path));
    frame.roi = roi.map(|[x, y, w, h]| PixelRect::new(x, y, w, h));
    frame.kind = kind;
    frame
}

impl RunFrames for PlanarRunResult {
    fn image_manifest(&self, spec: &DatasetSpec, base_dir: &Path) -> ImageManifest {
        one_camera_manifest(spec, base_dir, &self.view_paths, &[])
    }
}

impl RunFrames for HandeyeRunResult {
    fn image_manifest(&self, spec: &DatasetSpec, base_dir: &Path) -> ImageManifest {
        one_camera_manifest(spec, base_dir, &self.view_paths, &[])
    }
}

impl RunFrames for LaserlineRunResult {
    fn image_manifest(&self, spec: &DatasetSpec, base_dir: &Path) -> ImageManifest {
        one_camera_manifest(spec, base_dir, &self.view_paths, &self.laser_paths)
    }
}

impl<Meta> RunFrames for RigRunResult<Meta> {
    fn image_manifest(&self, spec: &DatasetSpec, base_dir: &Path) -> ImageManifest {
        rig_manifest(spec, base_dir, &self.view_paths, &[])
    }
}

impl RunFrames for RigLaserlineRunResult {
    fn image_manifest(&self, spec: &DatasetSpec, base_dir: &Path) -> ImageManifest {
        rig_manifest(spec, base_dir, &self.view_paths, &self.laser_paths)
    }
}

impl RunFrames for RigHandeyeLaserlineRunResult {
    fn image_manifest(&self, spec: &DatasetSpec, base_dir: &Path) -> ImageManifest {
        rig_manifest(spec, base_dir, &self.view_paths, &self.laser_paths)
    }
}

fn solve<P: ProblemType>(
    input: P::Input,
    config: P::Config,
    run: fn(&mut CalibrationSession<P>) -> Result<(), Error>,
    control: &RunControl,
) -> Result<P::Export, Error> {
    let mut session = CalibrationSession::<P>::new();
    session.set_input(input)?;
    session.set_config(config)?;
    session.set_control(control.clone());
    run(&mut session)?;
    session.export()
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL: [ProblemKind; 7] = [
        ProblemKind::PlanarIntrinsics,
        ProblemKind::SingleCamHandeye,
        ProblemKind::LaserlineDevice,
        ProblemKind::RigExtrinsics,
        ProblemKind::RigHandeye,
        ProblemKind::RigLaserlineDevice,
        ProblemKind::RigHandeyeLaserline,
    ];

    #[test]
    fn default_config_matches_its_topology_and_round_trips() {
        for topology in ALL {
            let config = CalibrationConfig::default_for(topology);
            assert_eq!(config.topology(), topology);
            let json = serde_json::to_value(&config).unwrap();
            assert_eq!(json["topology"], serde_json::to_value(topology).unwrap());
            let back: CalibrationConfig = serde_json::from_value(json).unwrap();
            assert_eq!(back.topology(), topology);
        }
    }

    #[test]
    fn config_for_another_topology_fails_before_reading_images() {
        let spec: DatasetSpec = serde_json::from_value(serde_json::json!({
            "version": 1,
            "topology": "planar_intrinsics",
            "cameras": [{ "id": "cam0", "images": { "kind": "glob", "pattern": "missing/*.png" } }],
            "target": { "kind": "chessboard", "rows": 6, "cols": 8, "square_size_m": 0.02 },
        }))
        .unwrap();
        let err = calibrate_dataset(
            &spec,
            Path::new("."),
            Some(CalibrationConfig::default_for(ProblemKind::RigExtrinsics)),
            Detection::Uncached,
            None,
            &RunControl::default(),
        )
        .unwrap_err();
        assert!(matches!(
            err,
            CalibrateDatasetError::ConfigTopologyMismatch {
                manifest: ProblemKind::PlanarIntrinsics,
                config: ProblemKind::RigExtrinsics,
            }
        ));
    }

    /// An invalid config is refused before any image is read: the glob
    /// matches nothing, so reaching detection would be an `EmptyImageMatch`.
    #[test]
    fn an_invalid_config_fails_before_reading_images() {
        let spec: DatasetSpec = serde_json::from_value(serde_json::json!({
            "version": 1,
            "topology": "planar_intrinsics",
            "cameras": [{ "id": "cam0", "images": { "kind": "glob", "pattern": "missing/*.png" } }],
            "target": { "kind": "chessboard", "rows": 6, "cols": 8, "square_size_m": 0.02 },
        }))
        .unwrap();
        let mut config = PlanarIntrinsicsConfig::default();
        config.solver.max_iters = 0;
        let err = calibrate_dataset(
            &spec,
            Path::new("."),
            Some(CalibrationConfig::PlanarIntrinsics(config)),
            Detection::Uncached,
            None,
            &RunControl::default(),
        )
        .unwrap_err();
        assert!(
            matches!(
                err,
                CalibrateDatasetError::Calibration(Error::InvalidInput { .. })
            ),
            "{err:?}"
        );
    }

    /// An open choice in the manifest is asked before any image is read: the
    /// glob matches nothing, so reading images would fail first.
    #[test]
    fn an_open_pose_pairing_is_asked_before_reading_images() {
        let spec: DatasetSpec = serde_json::from_value(serde_json::json!({
            "version": 1,
            "topology": "single_cam_handeye",
            "cameras": [{ "id": "cam0", "images": { "kind": "glob", "pattern": "missing/*.png" } }],
            "target": { "kind": "chessboard", "rows": 6, "cols": 8, "square_size_m": 0.02 },
            "robot_poses": { "path": "poses.txt", "format": "rowmajor4x4" },
            "pose_convention": {
                "transform": "t_base_tcp",
                "rotation_format": "matrix4x4_row_major",
                "translation_units": "m"
            },
        }))
        .unwrap();
        let err = calibrate_dataset(
            &spec,
            Path::new("."),
            None,
            Detection::Uncached,
            None,
            &RunControl::default(),
        )
        .unwrap_err();
        assert!(
            matches!(
                err,
                CalibrateDatasetError::Run(RunError::AskUser { ref field, .. }) if field == "pose_pairing"
            ),
            "{err:?}"
        );
    }

    #[test]
    fn laser_topology_without_extractor_is_a_typed_error() {
        let spec: DatasetSpec = serde_json::from_value(serde_json::json!({
            "version": 1,
            "topology": "laserline_device",
            "cameras": [{
                "id": "cam0",
                "images": { "kind": "glob", "pattern": "missing/*.png" },
                "laser_images": { "kind": "glob", "pattern": "missing/laser_*.png" },
            }],
            "target": { "kind": "chessboard", "rows": 6, "cols": 8, "square_size_m": 0.02 },
        }))
        .unwrap();
        let err = calibrate_dataset(
            &spec,
            Path::new("."),
            None,
            Detection::Uncached,
            None,
            &RunControl::default(),
        )
        .unwrap_err();
        assert!(matches!(
            err,
            CalibrateDatasetError::LaserExtractorRequired {
                topology: ProblemKind::LaserlineDevice
            }
        ));
    }

    /// A planar dataset of eight synthetic views of a 8 × 6 board under
    /// `dir/img`: each image a distinct blank PNG whose detection the cache
    /// holds, the board's corners projected by an ideal camera. Image 3's
    /// detection holds three corners, so the run leaves it out.
    fn planar_dataset(
        dir: &Path,
        cache: &vision_calibration_detect::FsDetectionCache,
    ) -> DatasetSpec {
        use vision_calibration_core::synthetic::{planar, poses};
        use vision_calibration_core::{BrownConrady5, FxFyCxCy, make_pinhole_camera};
        use vision_calibration_detect::{
            CacheKey, CachedFeatures, ChessboardConfig, DetectionCache, DetectorConfig, Feature,
        };

        let camera = make_pinhole_camera(
            FxFyCxCy {
                fx: 800.0,
                fy: 800.0,
                cx: 320.0,
                cy: 240.0,
            },
            BrownConrady5::default(),
        );
        let board = planar::grid_points(8, 6, 0.03);
        let specs = poses::seeded_board_pose_specs(8, 0.3, 7);
        let poses = poses::centered_board_poses(&specs, (0.105, 0.075), 0.6);
        let views = planar::project_views_all(&camera, &board, &poses).unwrap();
        let config = DetectorConfig::Chessboard(ChessboardConfig::new(6, 8, 0.03));
        std::fs::create_dir_all(dir.join("img")).unwrap();
        let mut paths = Vec::new();
        for (i, view) in views.iter().enumerate() {
            let rel = format!("img/view{i:02}.png");
            let mut image = image::GrayImage::new(640, 480);
            image.put_pixel(i as u32, 0, image::Luma([255]));
            image.save(dir.join(&rel)).unwrap();
            let bytes = std::fs::read(dir.join(&rel)).unwrap();
            let mut features: Vec<Feature> = view
                .points_3d
                .iter()
                .zip(&view.points_2d)
                .map(|(p, q)| Feature::new([q.x, q.y], [p.x, p.y, p.z]))
                .collect();
            if i == 3 {
                features.truncate(3);
            }
            let key = CacheKey::for_detection(&bytes, &config, None);
            cache.put(&key, &CachedFeatures::new(features)).unwrap();
            paths.push(rel);
        }
        serde_json::from_value(serde_json::json!({
            "version": 1,
            "topology": "planar_intrinsics",
            "cameras": [{ "id": "cam0", "images": { "kind": "list", "paths": paths } }],
            "target": { "kind": "chessboard", "rows": 6, "cols": 8, "square_size_m": 0.03 },
        }))
        .unwrap()
    }

    /// The export names the image of every view it used, relative to the
    /// dataset folder, in the input's view order: the views' identity.
    #[test]
    fn the_export_records_each_kept_view_s_image() {
        let tmp = tempfile::tempdir().unwrap();
        let cache = vision_calibration_detect::FsDetectionCache::new(tmp.path().join("cache"));
        let spec = planar_dataset(tmp.path(), &cache);
        let run = calibrate_dataset(
            &spec,
            tmp.path(),
            None,
            Detection::Cached(&cache),
            None,
            &RunControl::default(),
        )
        .unwrap();
        assert_eq!((run.usable_views, run.total_views), (7, 8));
        let manifest = run
            .export
            .diagnostics()
            .image_manifest
            .clone()
            .expect("the runner fills the image manifest");
        assert_eq!(manifest.root, Path::new("."));
        let frames: Vec<(usize, usize, PathBuf)> = manifest
            .frames
            .iter()
            .map(|f| (f.view, f.camera, f.path.clone()))
            .collect();
        let expected: Vec<(usize, usize, PathBuf)> = [0, 1, 2, 4, 5, 6, 7]
            .iter()
            .enumerate()
            .map(|(view, i)| (view, 0, PathBuf::from(format!("img/view{i:02}.png"))))
            .collect();
        assert_eq!(frames, expected);
        // Every residual's view has its frame.
        for r in &run.export.diagnostics().per_feature_residuals.target {
            assert!(manifest.frame(r.view, r.camera).is_some());
        }

        // Rebased to an output folder inside the dataset, it still finds them.
        let out = tmp.path().join("out");
        std::fs::create_dir_all(&out).unwrap();
        let rebased = manifest.rebased(tmp.path(), &out, tmp.path());
        assert_eq!(rebased.root, Path::new(".."));
        assert!(
            out.join(&rebased.root)
                .join(&rebased.frames[0].path)
                .is_file()
        );
    }

    #[test]
    fn calibration_export_handeye_forwards_variant() {
        use nalgebra::{Translation3, UnitQuaternion};
        use vision_calibration_core::{
            BrownConrady5, CameraParams, DistortionParams, FxFyCxCy, Iso3, ProjectionParams,
            SensorParams, Uncertainty,
        };
        use vision_calibration_optim::HandeyeMode;

        use crate::common::{CalibratedCameras, ExportDiagnostics, ProblemKind, VersionedExport};
        use crate::single_cam_handeye::{SingleCamHandeyeExport, SingleCamHandeyeSolution};

        let camera = CameraParams::new(
            ProjectionParams::Pinhole,
            DistortionParams::BrownConrady5 {
                params: BrownConrady5::default(),
            },
            SensorParams::Identity,
            FxFyCxCy {
                fx: 800.0,
                fy: 800.0,
                cx: 320.0,
                cy: 240.0,
            },
        );
        let he = Iso3::from_parts(
            Translation3::new(0.1, -0.2, 0.3),
            UnitQuaternion::identity(),
        );
        let export = CalibrationExport::SingleCamHandeye(SingleCamHandeyeExport {
            kind: ProblemKind::SingleCamHandeye,
            schema_version: SingleCamHandeyeExport::SCHEMA_VERSION,
            camera,
            handeye: SingleCamHandeyeSolution::new(HandeyeMode::EyeInHand, he, Iso3::identity()),
            robot_refinement: None,
            mean_reproj_error: 0.1,
            per_cam_reproj_errors: vec![0.1],
            diagnostics: ExportDiagnostics::default(),
            uncertainty: Uncertainty::default(),
        });
        assert!(export.handeye().is_some());
        assert_eq!(export.camera_params().len(), 1);
    }
}
