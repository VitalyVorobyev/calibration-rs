//! Freeze an entry: run detection on the images its dataset manifest names,
//! or read the corner files its publisher released, and write the resulting
//! [`ObservationSet`]. The images are read from the manifest's directory, or
//! from the cache `suite fetch` fills; corner files only from that cache.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};
use vision_calibration_dataset::{DatasetSpec, Topology};
use vision_calibration_detect::FsDetectionCache;
use vision_calibration_pipeline::dataset_runner::{
    Detection as DetectionMode, build_planar_input, build_rig_extrinsics_input,
    build_rig_handeye_input, build_single_cam_handeye_input,
};

use super::datasets::{fetched_dir, image_source};
use super::{OBSERVATION_SET_VERSION, ObservationSet, ObservedCamera, ProblemInput, Suite};

/// Build the observations of the dataset manifest at `manifest_path`
/// (TOML), with the manifest's paths resolved against `data_root`, caching
/// detections under `cache_dir`.
///
/// # Errors
///
/// Fails on an invalid manifest, a topology that needs a laser-line
/// extractor, a manifest with an ROI (the observation format has no
/// crop offset), detection errors, or a camera without a usable view.
pub fn freeze_manifest(
    manifest_path: &Path,
    data_root: &Path,
    cache_dir: &Path,
) -> Result<ObservationSet> {
    let spec = DatasetSpec::from_path(manifest_path)?;
    ensure!(
        spec.cameras.iter().all(|c| c.roi_xywh.is_none()),
        "{}: manifests with a camera ROI cannot be frozen",
        manifest_path.display()
    );
    let cache = FsDetectionCache::new(cache_dir);

    // First accepted image per camera, for the camera's image size.
    let (input, first_paths): (ProblemInput, Vec<Option<PathBuf>>) = match spec.topology {
        Topology::PlanarIntrinsics => {
            let run = build_planar_input(&spec, data_root, DetectionMode::Cached(&cache))?;
            let first = vec![run.view_paths.first().cloned()];
            (ProblemInput::PlanarIntrinsics(run.input), first)
        }
        Topology::SingleCamHandeye => {
            let run =
                build_single_cam_handeye_input(&spec, data_root, DetectionMode::Cached(&cache))?;
            let first = vec![run.view_paths.first().cloned()];
            (ProblemInput::SingleCamHandeye(run.input), first)
        }
        Topology::RigExtrinsics => {
            let run = build_rig_extrinsics_input(&spec, data_root, DetectionMode::Cached(&cache))?;
            let first = first_per_camera(&run.view_paths, spec.cameras.len());
            (ProblemInput::RigExtrinsics(run.input), first)
        }
        Topology::RigHandeye => {
            let run = build_rig_handeye_input(&spec, data_root, DetectionMode::Cached(&cache))?;
            let first = first_per_camera(&run.view_paths, spec.cameras.len());
            (ProblemInput::RigHandeye(run.input), first)
        }
        Topology::LaserlineDevice
        | Topology::RigLaserlineDevice
        | Topology::RigHandeyeLaserline => {
            bail!("freezing {:?} needs a laser-line extractor", spec.topology)
        }
    };

    let cameras = spec
        .cameras
        .iter()
        .zip(first_paths)
        .map(|(camera, path)| {
            let path =
                path.with_context(|| format!("camera {:?} has no usable view", camera.id))?;
            let (width, height) = image::image_dimensions(&path)
                .with_context(|| format!("read image size of {}", path.display()))?;
            Ok(ObservedCamera {
                id: camera.id.clone(),
                image_size: [width, height],
            })
        })
        .collect::<Result<_>>()?;

    Ok(ObservationSet {
        version: OBSERVATION_SET_VERSION,
        cameras,
        input,
    })
}

fn first_per_camera(view_paths: &[Vec<Option<PathBuf>>], cameras: usize) -> Vec<Option<PathBuf>> {
    (0..cameras)
        .map(|cam| {
            view_paths
                .iter()
                .find_map(|view| view.get(cam).cloned().flatten())
        })
        .collect()
}

/// Freeze one suite entry: write `observations/<id>.json` (compact JSON)
/// and refresh `SHA256SUMS`.
///
/// The images are read from the dataset manifest's directory, or, with
/// `images`, from the entry's directory in that `suite fetch` cache, after
/// checking every file of its fetch list. An entry with published corners
/// is read from that cache only.
///
/// # Errors
///
/// Fails when the entry has neither a dataset manifest nor published
/// corners, when `images` is given for an entry without a fetch source,
/// when its cached files are missing or changed (or, for corners, `images`
/// is not given), or see [`freeze_manifest`] and
/// [`PublishedCorners::observations`](super::corners::PublishedCorners::observations).
pub fn freeze_entry(
    suite: &Suite,
    id: &str,
    cache_dir: &Path,
    images: Option<&Path>,
) -> Result<ObservationSet> {
    let entry = suite
        .manifest
        .entries
        .iter()
        .find(|e| e.id == id)
        .with_context(|| format!("no suite entry {id:?}"))?;
    let set = match &entry.corners {
        Some(corners) => {
            let images = images.with_context(|| {
                format!(
                    "entry {id:?} is frozen from its publisher's corner files: run \
                     `suite fetch --only {id}` and pass --images"
                )
            })?;
            corners.observations(&fetched_dir(suite, entry, images)?)?
        }
        None => {
            let (manifest, data_root) = image_source(suite, entry, images)?;
            freeze_manifest(&manifest, &data_root, cache_dir)?
        }
    };
    super::write_compact(&suite.observations_path(id), &set)?;
    suite.write_checksums()?;
    Ok(set)
}
