//! The seed a project's runs start from: [`SeedSource`], what the project
//! file's `[seed]` names; [`RunSeed`], what a run's record keeps of it; and
//! [`DeviceSeed::from_run`], the seed of a run.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use vision_calibration_core::{ImageManifest, lexical_absolute};
use vision_calibration_dataset::ProblemKind;

use super::run::{read_run, sizes_of};
use super::{EXPORT_FILE, ProjectError, RunSource, RunState};
use crate::common::calibrated_cameras;
use crate::device_seed::{DeviceSeed, DeviceSpec};

/// Where the seed a project's runs start from comes from: the project
/// file's `[seed]`, one key.
///
/// ```toml
/// [seed]
/// run = "../unit-0007/runs/run-003"   # or: device_spec = "spec.json"
///                                     # or: file = "seeds/type-x.json"
/// ```
///
/// A path is relative to the project folder unless absolute. Every run of
/// the project starts from the seed: a run's record keeps the seed's
/// values as they were resolved ([`RunSeed`]), so a seed file that changes
/// later does not change what a run says it started from. A seed is an
/// initial value: the calibration still estimates everything.
///
/// The seed is resolved and checked against the dataset and the
/// configuration before any image is read; a seed that is missing, cannot
/// be read or does not fit is an error, never a run without it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum SeedSource {
    /// A calibrated run: its folder (`runs/run-003`, of this project or
    /// another), or the `export.json` in it. Its record names the cameras
    /// and their image sizes ([`DeviceSeed::from_run`]).
    Run(PathBuf),
    /// A device specification (`spec.json`): the datasheet seed
    /// ([`DeviceSeed::from_spec`]).
    DeviceSpec(PathBuf),
    /// A [`DeviceSeed`] file, JSON (TOML by its `.toml` extension): one made
    /// by `vcal seed`, or an aggregate of a device type
    /// ([`DeviceSeed::aggregate`]).
    File(PathBuf),
}

impl SeedSource {
    /// The path the source names, as written.
    pub(crate) fn path(&self) -> &Path {
        match self {
            Self::Run(path) | Self::DeviceSpec(path) | Self::File(path) => path,
        }
    }
}

/// The seed a calibrated run started from, as its record keeps it.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[non_exhaustive]
pub struct RunSeed {
    /// Where the seed came from, as the project file (or the run's options)
    /// named it.
    pub source: SeedSource,
    /// The seed as it was resolved: the values the run started from, in
    /// the seed's own rig frame.
    pub value: DeviceSeed,
    /// What the workflow took from the seed: `<camera>.lens`,
    /// `<camera>.cam_se3_rig`, `<camera>.laser_plane` and `handeye`, in the
    /// seed's camera order.
    pub applied: Vec<String>,
    /// What the seed holds that the workflow does not use, labelled the
    /// same way.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub ignored: Vec<String>,
}

impl DeviceSeed {
    /// The seed of a project's run: [`from_calibration`](Self::from_calibration)
    /// of its export, the cameras' ids and image sizes read from its
    /// record. `run` is the run's folder or the `export.json` in it.
    ///
    /// A run written before its record stated the image sizes has them
    /// read from the headers of the images its export names, when they are
    /// reachable. The seed's description names the run's folder, not its
    /// path.
    ///
    /// # Errors
    ///
    /// [`ProjectError::SeedUnresolved`] when `run` is not a run, its record
    /// cannot be read or does not name its cameras (a run of the desktop
    /// app's earlier format), the image sizes are neither recorded nor
    /// readable, or the export does not make a seed
    /// ([`from_calibration`](Self::from_calibration)'s errors).
    pub fn from_run(run: impl AsRef<Path>) -> Result<Self, ProjectError> {
        let path = lexical_absolute(run.as_ref());
        let unresolved = |message: String| ProjectError::SeedUnresolved {
            path: path.clone(),
            message,
        };
        let dir = if path.is_file() {
            path.parent().map(Path::to_path_buf).unwrap_or_default()
        } else {
            path.clone()
        };
        let id = dir
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        let info = read_run(&dir, id.clone()).ok_or_else(|| {
            unresolved("not a run: the folder holds neither run.json nor export.json".into())
        })?;
        let record = match &info.state {
            RunState::Recorded(record) => record,
            RunState::Unrecorded(_) => {
                return Err(unresolved(
                    "the run has no record (run.json) naming its cameras; make the seed from \
                     its export with the cameras' ids and image sizes"
                        .into(),
                ));
            }
            RunState::Unreadable { message } => {
                return Err(unresolved(format!("its record cannot be read: {message}")));
            }
        };
        let export_path = dir.join(EXPORT_FILE);
        let text = std::fs::read_to_string(&export_path)
            .map_err(|e| unresolved(format!("{}: {e}", export_path.display())))?;
        let export =
            calibrated_cameras(&text).map_err(|e| unresolved(format!("its export: {e}")))?;
        let count = export.camera_params().len();
        let cameras = &record.summary.cameras;
        if cameras.len() != count {
            return Err(unresolved(format!(
                "its record names {} cameras and its export has {count}",
                cameras.len()
            )));
        }
        let ids: Vec<String> = cameras.iter().map(|c| c.id.clone()).collect();
        let recorded: Option<Vec<[u32; 2]>> = cameras.iter().map(|c| c.image_size).collect();
        let image_sizes = match recorded {
            Some(sizes) => sizes,
            None => {
                let sizes = export
                    .diagnostics()
                    .image_manifest
                    .as_ref()
                    .map(|images: &ImageManifest| sizes_of(images, &dir, count))
                    .unwrap_or_default();
                if sizes.len() != count {
                    return Err(unresolved(
                        "its record does not state the cameras' image sizes and its images \
                         cannot be read"
                            .into(),
                    ));
                }
                sizes
            }
        };
        let mut seed = Self::from_calibration(export.as_ref(), &ids, &image_sizes)
            .map_err(|e| unresolved(e.to_string()))?;
        let how = match &record.source {
            RunSource::Calibrated(_) => "calibrated",
            _ => "imported",
        };
        seed.description = Some(format!("the {how} run {id}"));
        Ok(seed)
    }
}

/// The seed `source` names, its relative path resolved against the
/// project folder `dir`.
pub(crate) fn resolve(dir: &Path, source: &SeedSource) -> Result<DeviceSeed, ProjectError> {
    let path = lexical_absolute(&dir.join(source.path()));
    let unresolved = |message: String| ProjectError::SeedUnresolved {
        path: path.clone(),
        message,
    };
    match source {
        SeedSource::Run(_) => DeviceSeed::from_run(&path),
        SeedSource::DeviceSpec(_) => {
            let spec = DeviceSpec::from_path(&path).map_err(|e| unresolved(e.to_string()))?;
            DeviceSeed::from_spec(&spec).map_err(|e| unresolved(e.to_string()))
        }
        SeedSource::File(_) => {
            let text = std::fs::read_to_string(&path).map_err(|e| unresolved(e.to_string()))?;
            let toml = path
                .extension()
                .is_some_and(|e| e.eq_ignore_ascii_case("toml"));
            let seed: DeviceSeed = if toml {
                toml::from_str(&text).map_err(|e| unresolved(e.to_string()))?
            } else {
                serde_json::from_str(&text).map_err(|e| unresolved(e.to_string()))?
            };
            seed.validate().map_err(|e| unresolved(e.to_string()))?;
            Ok(seed)
        }
    }
}

/// What a `topology` run takes from `seed` and what it leaves, by label
/// ([`RunSeed::applied`], [`RunSeed::ignored`]).
pub(crate) fn seed_use(seed: &DeviceSeed, topology: ProblemKind) -> (Vec<String>, Vec<String>) {
    use ProblemKind as K;
    let lens = !matches!(topology, K::RigLaserlineDevice);
    let rig = matches!(
        topology,
        K::RigExtrinsics | K::RigHandeye | K::RigHandeyeLaserline
    );
    let planes = matches!(
        topology,
        K::LaserlineDevice | K::RigLaserlineDevice | K::RigHandeyeLaserline
    );
    let handeye = matches!(
        topology,
        K::SingleCamHandeye | K::RigHandeye | K::RigHandeyeLaserline
    ) && seed.handeye.is_some();
    // One camera's hand-eye is re-expressed through its mount.
    let mounts = rig || (handeye && topology == K::SingleCamHandeye);
    let mut applied = Vec::new();
    let mut ignored = Vec::new();
    let mut note = |used: bool, label: String| {
        if used {
            applied.push(label);
        } else {
            ignored.push(label);
        }
    };
    for camera in &seed.cameras {
        note(lens, format!("{}.lens", camera.id));
        if camera.cam_se3_rig.is_some() {
            note(mounts, format!("{}.cam_se3_rig", camera.id));
        }
        if camera.laser_plane.is_some() {
            note(planes, format!("{}.laser_plane", camera.id));
        }
    }
    if seed.handeye.is_some() {
        note(handeye, "handeye".to_string());
    }
    (applied, ignored)
}

#[cfg(test)]
mod tests {
    use super::*;
    use vision_calibration_core::{BrownConrady5, FxFyCxCy, Iso3, make_pinhole_camera};

    fn seed(cameras: usize, handeye: bool, planes: bool) -> DeviceSeed {
        let camera = vision_calibration_core::pinhole_camera_params(&make_pinhole_camera(
            FxFyCxCy {
                fx: 800.0,
                fy: 800.0,
                cx: 320.0,
                cy: 240.0,
            },
            BrownConrady5::default(),
        ));
        let mut seed = DeviceSeed::new(
            (0..cameras)
                .map(|i| {
                    let mut c = crate::device_seed::SeedCamera::new(
                        format!("cam{i}"),
                        [640, 480],
                        crate::device_seed::SeedLens::Calibrated {
                            camera: camera.clone(),
                        },
                    );
                    if cameras > 1 {
                        c.cam_se3_rig = Some(Iso3::identity());
                    }
                    if planes {
                        c.laser_plane = Some(
                            vision_calibration_core::LaserPlane::new(
                                nalgebra::Vector3::new(0.0, 0.6, 0.8),
                                0.3,
                            )
                            .unwrap(),
                        );
                    }
                    c
                })
                .collect(),
        );
        if handeye {
            seed.handeye = Some(crate::device_seed::SeedHandeye::EyeInHand {
                gripper_se3_rig: Iso3::identity(),
            });
        }
        seed
    }

    #[test]
    fn each_workflow_takes_what_it_uses() {
        let full = seed(2, true, true);
        let (applied, ignored) = seed_use(&full, ProblemKind::RigHandeyeLaserline);
        assert_eq!(
            applied,
            [
                "cam0.lens",
                "cam0.cam_se3_rig",
                "cam0.laser_plane",
                "cam1.lens",
                "cam1.cam_se3_rig",
                "cam1.laser_plane",
                "handeye"
            ]
        );
        assert!(ignored.is_empty());

        let (applied, ignored) = seed_use(&full, ProblemKind::RigExtrinsics);
        assert_eq!(
            applied,
            [
                "cam0.lens",
                "cam0.cam_se3_rig",
                "cam1.lens",
                "cam1.cam_se3_rig"
            ]
        );
        assert_eq!(ignored, ["cam0.laser_plane", "cam1.laser_plane", "handeye"]);

        let (applied, ignored) = seed_use(&full, ProblemKind::RigLaserlineDevice);
        assert_eq!(applied, ["cam0.laser_plane", "cam1.laser_plane"]);
        assert_eq!(
            ignored,
            [
                "cam0.lens",
                "cam0.cam_se3_rig",
                "cam1.lens",
                "cam1.cam_se3_rig",
                "handeye"
            ]
        );

        let one = seed(1, true, true);
        let (applied, ignored) = seed_use(&one, ProblemKind::SingleCamHandeye);
        assert_eq!(applied, ["cam0.lens", "handeye"]);
        assert_eq!(ignored, ["cam0.laser_plane"]);
        let (applied, ignored) = seed_use(&one, ProblemKind::LaserlineDevice);
        assert_eq!(applied, ["cam0.lens", "cam0.laser_plane"]);
        assert_eq!(ignored, ["handeye"]);
        let (applied, ignored) = seed_use(&seed(1, false, false), ProblemKind::PlanarIntrinsics);
        assert_eq!(applied, ["cam0.lens"]);
        assert!(ignored.is_empty());
    }

    #[test]
    fn a_seed_source_is_one_key_in_toml_and_json() {
        for (text, expected) in [
            (
                "run = \"../a/runs/run-003\"",
                SeedSource::Run("../a/runs/run-003".into()),
            ),
            (
                "device_spec = \"spec.json\"",
                SeedSource::DeviceSpec("spec.json".into()),
            ),
            (
                "file = \"seeds/x.json\"",
                SeedSource::File("seeds/x.json".into()),
            ),
        ] {
            let source: SeedSource = toml::from_str(text).unwrap();
            assert_eq!(source, expected);
            let json = serde_json::to_value(&source).unwrap();
            assert_eq!(
                serde_json::from_value::<SeedSource>(json).unwrap(),
                expected
            );
            assert_eq!(
                toml::from_str::<SeedSource>(&toml::to_string(&source).unwrap()).unwrap(),
                expected
            );
        }
        assert!(toml::from_str::<SeedSource>("run = \"a\"\nfile = \"b\"").is_err());
        assert!(toml::from_str::<SeedSource>("path = \"a\"").is_err());
    }
}
