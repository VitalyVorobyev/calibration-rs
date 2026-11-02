//! One call from a manifest to an export: [`calibrate_dataset`] picks the
//! converter and the problem type by the manifest's topology.

use std::path::Path;

use serde::{Deserialize, Serialize};
use thiserror::Error;
use vision_calibration_core::{CameraParams, Iso3, Uncertainty};
use vision_calibration_dataset::{DatasetSpec, Topology};

use super::{
    Detection, LaserPixelExtractor, RejectedDetection, RunError, build_laserline_device_input,
    build_planar_input, build_rig_extrinsics_input, build_rig_handeye_input,
    build_rig_handeye_laserline_input, build_rig_laserline_device_input,
    build_single_cam_handeye_input, camera_image_sizes,
};
use crate::Error;
use crate::common::{CalibratedCameras, ExportDiagnostics, ExportKind};
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
    /// The default configuration of the problem type that `topology`
    /// selects.
    pub fn default_for(topology: Topology) -> Self {
        match topology {
            Topology::PlanarIntrinsics => Self::PlanarIntrinsics(Default::default()),
            Topology::SingleCamHandeye => Self::SingleCamHandeye(Default::default()),
            Topology::LaserlineDevice => Self::LaserlineDevice(Default::default()),
            Topology::RigExtrinsics => Self::RigExtrinsics(Default::default()),
            Topology::RigHandeye => Self::RigHandeye(Default::default()),
            Topology::RigLaserlineDevice => Self::RigLaserlineDevice(Default::default()),
            Topology::RigHandeyeLaserline => Self::RigHandeyeLaserline(Default::default()),
        }
    }

    /// The manifest topology this configuration applies to.
    pub fn topology(&self) -> Topology {
        match self {
            Self::PlanarIntrinsics(_) => Topology::PlanarIntrinsics,
            Self::SingleCamHandeye(_) => Topology::SingleCamHandeye,
            Self::LaserlineDevice(_) => Topology::LaserlineDevice,
            Self::RigExtrinsics(_) => Topology::RigExtrinsics,
            Self::RigHandeye(_) => Topology::RigHandeye,
            Self::RigLaserlineDevice(_) => Topology::RigLaserlineDevice,
            Self::RigHandeyeLaserline(_) => Topology::RigHandeyeLaserline,
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

    /// The problem type that produced this export.
    pub fn kind(&self) -> ExportKind {
        match self {
            Self::PlanarIntrinsics(_) => ExportKind::PlanarIntrinsics,
            Self::SingleCamHandeye(_) => ExportKind::SingleCamHandeye,
            Self::LaserlineDevice(_) => ExportKind::LaserlineDevice,
            Self::RigExtrinsics(_) => ExportKind::RigExtrinsics,
            Self::RigHandeye(_) => ExportKind::RigHandeye,
            Self::RigLaserlineDevice(_) => ExportKind::RigLaserlineDevice,
            Self::RigHandeyeLaserline(_) => ExportKind::RigHandeyeLaserline,
        }
    }
}

/// Every problem's export reports its cameras, so [`assess`](crate::analysis::assess)
/// and the other analyses take a `CalibrationExport` directly.
impl CalibratedCameras for CalibrationExport {
    fn camera_params(&self) -> Vec<CameraParams> {
        self.cameras().camera_params()
    }

    fn cam_se3_rig(&self) -> Vec<Iso3> {
        self.cameras().cam_se3_rig()
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
    Run(#[from] RunError),
    /// The configuration is for a different topology than the manifest's.
    #[error("the config is for a {config:?} dataset but the manifest's topology is {manifest:?}")]
    ConfigTopologyMismatch {
        /// The manifest's topology.
        manifest: Topology,
        /// The topology the configuration applies to.
        config: Topology,
    },
    /// A laser topology was given no laser-line extractor.
    #[error("a {topology:?} manifest needs a laser-line extractor")]
    LaserExtractorRequired {
        /// The manifest's topology.
        topology: Topology,
    },
    /// The configuration is invalid, the problem rejected the detected
    /// input, or the calibration itself failed.
    #[error(transparent)]
    Calibration(#[from] Error),
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
/// `laser_extractor` is required by the laser topologies
/// (`laserline_device`, `rig_laserline_device`, `rig_handeye_laserline`) and
/// unused by the others.
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
///
/// # Example
///
/// ```no_run
/// use std::path::Path;
/// use vision_calibration_dataset::DatasetSpec;
/// use vision_calibration_detect::FsDetectionCache;
/// use vision_calibration_pipeline::dataset_runner::{Detection, calibrate_dataset};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let dir = Path::new("my_dataset");
/// let spec = DatasetSpec::from_path(dir.join("dataset.toml"))?;
/// let cache = FsDetectionCache::new(dir.join(".detections"));
/// let run = calibrate_dataset(&spec, dir, None, Detection::Cached(&cache), None)?;
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
) -> Result<DatasetCalibration, CalibrateDatasetError> {
    let topology = spec.topology;
    let config = config.unwrap_or_else(|| CalibrationConfig::default_for(topology));
    if config.topology() != topology {
        return Err(CalibrateDatasetError::ConfigTopologyMismatch {
            manifest: topology,
            config: config.topology(),
        });
    }
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
    // A camera whose images differ in size fails here, before any detection.
    let image_sizes = camera_image_sizes(spec, base_dir)?;

    // `config.topology() == topology` was checked above, so each arm sees the
    // config variant of its own topology.
    match config {
        CalibrationConfig::PlanarIntrinsics(config) => {
            let run = build_planar_input(spec, base_dir, detection)?;
            let export = solve::<PlanarIntrinsicsProblem>(
                run.input,
                config,
                crate::planar_intrinsics::run_calibration,
            )?;
            Ok(finish(
                image_sizes,
                CalibrationExport::PlanarIntrinsics(export),
                run.usable_views,
                run.total_views,
                run.rejected,
            ))
        }
        CalibrationConfig::SingleCamHandeye(config) => {
            let run = build_single_cam_handeye_input(spec, base_dir, detection)?;
            let export = solve::<SingleCamHandeyeProblem>(
                run.input,
                config,
                crate::single_cam_handeye::run_calibration,
            )?;
            Ok(finish(
                image_sizes,
                CalibrationExport::SingleCamHandeye(export),
                run.usable_views,
                run.total_views,
                run.rejected,
            ))
        }
        CalibrationConfig::LaserlineDevice(config) => {
            let run = build_laserline_device_input(spec, base_dir, detection, laser()?)?;
            let export = solve::<LaserlineDeviceProblem>(
                run.input,
                config,
                crate::laserline_device::run_calibration,
            )?;
            Ok(finish(
                image_sizes,
                CalibrationExport::LaserlineDevice(export),
                run.usable_views,
                run.total_views,
                run.rejected,
            ))
        }
        CalibrationConfig::RigExtrinsics(config) => {
            let run = build_rig_extrinsics_input(spec, base_dir, detection)?;
            let export = solve::<RigExtrinsicsProblem>(
                run.input,
                config,
                crate::rig_extrinsics::run_calibration,
            )?;
            Ok(finish(
                image_sizes,
                CalibrationExport::RigExtrinsics(export),
                run.usable_views,
                run.total_views,
                run.rejected,
            ))
        }
        CalibrationConfig::RigHandeye(config) => {
            let run = build_rig_handeye_input(spec, base_dir, detection)?;
            let export =
                solve::<RigHandeyeProblem>(run.input, config, crate::rig_handeye::run_calibration)?;
            Ok(finish(
                image_sizes,
                CalibrationExport::RigHandeye(export),
                run.usable_views,
                run.total_views,
                run.rejected,
            ))
        }
        CalibrationConfig::RigLaserlineDevice(config) => {
            let run = build_rig_laserline_device_input(spec, base_dir, detection, laser()?)?;
            let export = solve::<RigLaserlineDeviceProblem>(
                run.input,
                config,
                crate::rig_laserline_device::run_calibration,
            )?;
            Ok(finish(
                image_sizes,
                CalibrationExport::RigLaserlineDevice(export),
                run.usable_views,
                run.total_views,
                run.rejected,
            ))
        }
        CalibrationConfig::RigHandeyeLaserline(config) => {
            let run = build_rig_handeye_laserline_input(spec, base_dir, detection, laser()?)?;
            let export = solve::<RigHandeyeLaserlineProblem>(
                run.input,
                config,
                crate::rig_handeye_laserline::run_calibration,
            )?;
            Ok(finish(
                image_sizes,
                CalibrationExport::RigHandeyeLaserline(export),
                run.usable_views,
                run.total_views,
                run.rejected,
            ))
        }
    }
}

fn solve<P: ProblemType>(
    input: P::Input,
    config: P::Config,
    run: fn(&mut CalibrationSession<P>) -> Result<(), Error>,
) -> Result<P::Export, Error> {
    let mut session = CalibrationSession::<P>::new();
    session.set_input(input)?;
    session.set_config(config)?;
    run(&mut session)?;
    session.export()
}

fn finish(
    image_sizes: Vec<[u32; 2]>,
    export: CalibrationExport,
    usable_views: usize,
    total_views: usize,
    rejected: Vec<RejectedDetection>,
) -> DatasetCalibration {
    DatasetCalibration {
        export,
        image_sizes,
        usable_views,
        total_views,
        rejected,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL: [Topology; 7] = [
        Topology::PlanarIntrinsics,
        Topology::SingleCamHandeye,
        Topology::LaserlineDevice,
        Topology::RigExtrinsics,
        Topology::RigHandeye,
        Topology::RigLaserlineDevice,
        Topology::RigHandeyeLaserline,
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
            Some(CalibrationConfig::default_for(Topology::RigExtrinsics)),
            Detection::Uncached,
            None,
        )
        .unwrap_err();
        assert!(matches!(
            err,
            CalibrateDatasetError::ConfigTopologyMismatch {
                manifest: Topology::PlanarIntrinsics,
                config: Topology::RigExtrinsics,
            }
        ));
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
        let err =
            calibrate_dataset(&spec, Path::new("."), None, Detection::Uncached, None).unwrap_err();
        assert!(matches!(
            err,
            CalibrateDatasetError::LaserExtractorRequired {
                topology: Topology::LaserlineDevice
            }
        ));
    }
}
