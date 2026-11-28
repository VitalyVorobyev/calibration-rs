//! A project's runs: calibrating one ([`Project::calibrate`]), importing
//! an export ([`Project::import_export`]), listing, notes and deletion.
//!
//! A run is written whole or not at all: every file goes into a staging
//! folder under `runs/`, which is then renamed to the run's id. The id is
//! `run-NNN`, one more than the largest number among the runs; a rename
//! onto a run another writer claimed first fails, and the next number is
//! tried, so the desktop app, `vcal` and Python can write runs into one
//! project at once.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::time::Instant;

use serde::Deserialize;
use vision_calibration_core::{ImageManifest, RunControl, data_relative_path, lexical_absolute};
use vision_calibration_dataset::{DatasetSpec, ProblemKind};
use vision_calibration_detect::FsDetectionCache;

use super::digest::{DigestInputs, RecordingCache};
use super::fs::{TempDir, json_text, read, unique_name, write_atomic};
use super::{
    CalibratedRun, CameraSummary, DatasetSource, EXPORT_FILE, FeatureExclusion, LeftOutImage,
    Project, ProjectError, QUALITY_FILE, REPORT_FILE, RUN_NOTES_FILE, RUN_RECORD_FILE,
    RUN_RECORD_VERSION, RUNS_DIR, RunInfo, RunInputs, RunNotes, RunRecord, RunSeed, RunSource,
    RunState, RunSummary, SeedSource, UnrecordedRun, UsedImage,
};
use crate::analysis::{QualityThresholds, ReportOptions, assess, html_report};
use crate::common::{CalibratedCameras, calibrated_cameras};
use crate::dataset_runner::{
    CalibrateDatasetError, CalibrationConfig, ConfigPatch, ConvertOptions, Detection,
    LaserPixelExtractor, RunError, calibrate_dataset_with, check_seed, require_pose_pairing,
};
use crate::device_seed::DeviceSeed;

/// How many run ids a writer tries before it gives up: each failed try is
/// a run another writer claimed in between.
const MAX_CLAIMS: usize = 64;

/// The detection cache's `.gitignore`: the cache is never committed.
const CACHE_GITIGNORE: &str = "*\n";

/// The options of [`Project::calibrate`].
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct ProjectRunOptions {
    /// Notes written with the run, when not empty.
    pub notes: RunNotes,
    /// Configuration changes for this run only, over the project's; the
    /// record keeps them.
    pub config_patch: ConfigPatch,
    /// Detect every image again instead of reading the detection cache.
    pub redetect: bool,
    /// Also write `report.html`, the run as one self-contained page.
    pub report_html: bool,
    /// The program making the run, as the record names it
    /// (`vcal 1.1.0`).
    pub tool: String,
    /// A seed for this run only, in place of the project's `[seed]`; its
    /// path relative to the project folder unless absolute. The record
    /// keeps it.
    pub seed: Option<SeedSource>,
}

impl ProjectRunOptions {
    /// The options of a run made by `tool`: no notes, no configuration
    /// changes or seed of its own, the detection cache read, no page.
    pub fn new(tool: impl Into<String>) -> Self {
        Self {
            tool: tool.into(),
            ..Self::default()
        }
    }
}

/// A project's inputs to a run, resolved and checked.
pub(crate) struct Prepared {
    pub(crate) dataset: DatasetSpec,
    pub(crate) data_root: PathBuf,
    pub(crate) patch: ConfigPatch,
    pub(crate) config: CalibrationConfig,
    pub(crate) quality: QualityThresholds,
    pub(crate) options: ConvertOptions,
    /// The seed the run starts from: where it came from and its values.
    pub(crate) seed: Option<(SeedSource, DeviceSeed)>,
}

impl Project {
    /// Resolve and check what a run of the project with the configuration
    /// changes `extra` would be given; no image is read.
    pub(crate) fn prepare(&self, extra: &ConfigPatch) -> Result<Prepared, ProjectError> {
        self.prepare_with(extra, None)
    }

    /// [`prepare`](Self::prepare), with the seed `seed` in place of the
    /// project's when one is given. The seed is resolved and checked
    /// against the dataset and the configuration.
    pub(crate) fn prepare_with(
        &self,
        extra: &ConfigPatch,
        seed: Option<&SeedSource>,
    ) -> Result<Prepared, ProjectError> {
        super::file::validate(self.spec(), self.file())?;
        let dataset = self.dataset()?;
        let data_root = self.data_root()?;
        vision_calibration_dataset::validate(&dataset).map_err(RunError::from)?;
        require_pose_pairing(&dataset)?;
        let patch = self.spec().config.then(extra);
        let config =
            CalibrationConfig::patched(dataset.topology, &patch).map_err(ProjectError::Config)?;
        let quality = self.spec().quality.unwrap_or_default();
        let options = self.spec().exclude.resolve(&dataset, &data_root)?;
        let seed = match seed.or(self.spec().seed.as_ref()) {
            Some(source) => {
                let value = super::seed::resolve(self.dir(), source)?;
                check_seed(&value, &dataset, &config)
                    .map_err(|e| ProjectError::Calibrate(CalibrateDatasetError::Seed(e)))?;
                Some((source.clone(), value))
            }
            None => None,
        };
        Ok(Prepared {
            dataset,
            data_root,
            patch,
            config,
            quality,
            options,
            seed,
        })
    }

    /// Calibrate the project's dataset and write the run to
    /// `runs/<run-id>/`: `export.json`, `quality.json` and `run.json`, with
    /// `notes.json` and `report.html` when asked for.
    ///
    /// Everything [`check`](Self::check) checks fails before any image is
    /// read, the seed included: the project's `[seed]`, or
    /// [`ProjectRunOptions::seed`] in its place, is resolved, checked
    /// against the dataset and the configuration, and the run starts from
    /// it as [`calibrate_dataset_with_seed`] does; the record keeps it
    /// ([`RunInputs::seed`]). Detection goes through the project's cache
    /// ([`detection_cache_dir`](Self::detection_cache_dir)). The export's
    /// image manifest is restated for the run folder: relative when the data
    /// lies inside the project, the absolute folder otherwise.
    ///
    /// `laser_extractor` is required by the laser topologies. `control`
    /// reports and stops the run as [`calibrate_dataset`] describes; a
    /// cancelled run writes nothing.
    ///
    /// [`calibrate_dataset`]: crate::dataset_runner::calibrate_dataset
    /// [`calibrate_dataset_with_seed`]: crate::dataset_runner::calibrate_dataset_with_seed
    ///
    /// # Errors
    ///
    /// What [`check`](Self::check) returns, for the run's own seed too;
    /// [`ProjectError::Calibrate`] when detection or the calibration fails,
    /// or a seeded camera's images are of another size than its seed's;
    /// [`ProjectError::Cancelled`];
    /// [`ProjectError::Io`] when the run cannot be written.
    pub fn calibrate(
        &self,
        options: &ProjectRunOptions,
        laser_extractor: Option<&dyn LaserPixelExtractor>,
        control: &RunControl,
    ) -> Result<RunInfo, ProjectError> {
        // For display only: never part of a digest, the export or the report.
        let started = Instant::now();
        let prepared = self.prepare_with(&options.config_patch, options.seed.as_ref())?;
        let cache_dir = self.detection_cache_dir();
        ensure_cache_dir(&cache_dir)?;
        let fs_cache = FsDetectionCache::new(&cache_dir);
        let cache = RecordingCache::new(&fs_cache);
        let detection = if options.redetect {
            Detection::Refresh(&cache)
        } else {
            Detection::Cached(&cache)
        };
        let mut run = calibrate_dataset_with(
            &prepared.dataset,
            &prepared.data_root,
            Some(prepared.config.clone()),
            detection,
            laser_extractor,
            control,
            &prepared.options,
            prepared.seed.as_ref().map(|(_, seed)| seed),
        )?;
        // Cancelled after the last check: still stop before writing.
        if control.is_cancelled() {
            return Err(ProjectError::Cancelled);
        }
        let ids: Vec<String> = prepared
            .dataset
            .cameras
            .iter()
            .map(|c| c.id.clone())
            .collect();
        let images = used_images(&run.export, &prepared.data_root, &ids, &cache.images());
        if let Some(images) = run.export.diagnostics_mut().image_manifest.as_mut() {
            *images = images.rebased(&prepared.data_root, &self.run_dir_model(), self.dir());
        }
        let quality = assess(&run.export, &run.image_sizes, &prepared.quality)
            .map_err(|e| ProjectError::Calibrate(CalibrateDatasetError::Calibration(e)))?;

        let summary = RunSummary {
            kind: run.export.kind(),
            verdict: Some(quality.verdict),
            cameras: camera_summaries(&run.export, &ids, &run.image_sizes),
        };
        let left_out = run
            .rejected
            .iter()
            .map(|r| LeftOutImage {
                camera: camera_id(&ids, r.camera),
                image: slash_path(&data_relative_path(&prepared.data_root, &r.path)),
                reason: r.reason,
            })
            .collect();
        let unmatched_exclusions = self.unmatched_exclusions(&prepared);
        let detections = cache.keys();
        let digest = DigestInputs {
            dataset: &prepared.dataset,
            data_root: &prepared.data_root,
            exclusions: &self.spec().exclude,
            config: &prepared.config,
            quality: &prepared.quality,
            detections: &detections,
            seed: prepared.seed.as_ref().map(|(_, seed)| seed),
        }
        .digest()?;
        let seed = prepared.seed.as_ref().map(|(source, value)| {
            let (applied, ignored) = super::seed::seed_use(value, prepared.dataset.topology);
            RunSeed {
                source: source.clone(),
                value: value.clone(),
                applied,
                ignored,
            }
        });
        let record = RunRecord {
            version: RUN_RECORD_VERSION,
            timestamp: unix_now(),
            tool: options.tool.clone(),
            library_version: env!("CARGO_PKG_VERSION").to_string(),
            source: RunSource::Calibrated(Box::new(CalibratedRun {
                inputs: RunInputs {
                    dataset: prepared.dataset.clone(),
                    data_root: self.stated_data_root(),
                    exclusions: self.spec().exclude.clone(),
                    config_patch: prepared.patch.clone(),
                    config: prepared.config.clone(),
                    quality: prepared.quality,
                    seed,
                },
                digest,
                usable_views: run.usable_views,
                total_views: run.total_views,
                left_out,
                unmatched_exclusions,
                images,
                redetected: options.redetect,
                duration_ms: u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX),
            })),
            summary,
        };

        let runs_dir = self.dir().join(RUNS_DIR);
        let mut files = vec![
            (EXPORT_FILE, json_text(&run.export, &runs_dir)?),
            (QUALITY_FILE, json_text(&quality, &runs_dir)?),
            (RUN_RECORD_FILE, json_text(&record, &runs_dir)?),
        ];
        if options.report_html {
            let mut report = ReportOptions::new();
            report.title = format!(
                "{}: {}",
                self.display_name(),
                kind_name(record.summary.kind)
            );
            report.camera_names = ids.clone();
            report.thresholds = prepared.quality;
            let page = html_report(&run.export, &run.image_sizes, &report)
                .map_err(|e| ProjectError::Calibrate(CalibrateDatasetError::Calibration(e)))?;
            files.push((REPORT_FILE, page));
        }
        if !options.notes.is_empty() {
            files.push((RUN_NOTES_FILE, json_text(&options.notes, &runs_dir)?));
        }
        let (id, dir) = self.write_run(&files)?;
        Ok(RunInfo {
            id,
            dir,
            state: RunState::Recorded(Box::new(record)),
            notes: options.notes.clone(),
        })
    }

    /// Add an export calibrated elsewhere as a run: a calibration export
    /// file, or a folder `vcal run -o` wrote (its `calibration.json`) or a
    /// run folder (its `export.json`).
    ///
    /// The export is written with its image manifest's root restated for
    /// the run folder; nothing else changes. Its quality is assessed with
    /// the default thresholds or the project's, each camera's image size
    /// read from the images its manifest names (without them, the checks
    /// that need a size are not applicable); an export whose quality cannot
    /// be assessed gets no `quality.json` and no verdict.
    ///
    /// # Errors
    ///
    /// [`ProjectError::Io`] and [`ProjectError::Parse`] when the file cannot
    /// be read as an export; [`ProjectError::ImportMismatch`] when its
    /// workflow or its number of cameras differs from the project's
    /// dataset; [`ProjectError::Manifest`] as
    /// [`dataset`](Self::dataset).
    pub fn import_export(
        &self,
        export: impl AsRef<Path>,
        notes: RunNotes,
    ) -> Result<RunInfo, ProjectError> {
        let source = lexical_absolute(export.as_ref());
        let file = if source.is_dir() {
            let vcal = source.join("calibration.json");
            if vcal.is_file() {
                vcal
            } else {
                source.join(EXPORT_FILE)
            }
        } else {
            source
        };
        let bytes = read(&file)?;
        let text = String::from_utf8(bytes).map_err(|e| ProjectError::parse(&file, e))?;
        let cameras = calibrated_cameras(&text).map_err(|e| ProjectError::parse(&file, e))?;
        let mut value: serde_json::Value =
            serde_json::from_str(&text).map_err(|e| ProjectError::parse(&file, e))?;
        let kind: ProblemKind = serde_json::from_value(value["kind"].clone())
            .map_err(|e| ProjectError::parse(&file, e))?;
        let dataset = self.dataset()?;
        if kind != dataset.topology {
            return Err(ProjectError::ImportMismatch {
                message: format!(
                    "the export is a {} calibration; the project's dataset is {}",
                    kind_name(kind),
                    kind_name(dataset.topology)
                ),
            });
        }
        let count = cameras.camera_params().len();
        if count != dataset.cameras.len() {
            return Err(ProjectError::ImportMismatch {
                message: format!(
                    "the export has {count} cameras; the project's dataset has {}",
                    dataset.cameras.len()
                ),
            });
        }

        let source_dir = file
            .parent()
            .map_or_else(|| self.dir().to_path_buf(), Path::to_path_buf);
        let mut image_sizes = Vec::new();
        if let Some(manifest) = value.pointer_mut("/diagnostics/image_manifest")
            && let Ok(images) = serde_json::from_value::<ImageManifest>(manifest.clone())
        {
            image_sizes = sizes_of(&images, &source_dir, count);
            let rebased = images.rebased(&source_dir, &self.run_dir_model(), self.dir());
            manifest["root"] =
                serde_json::Value::String(slash_path(&rebased.root).to_string_lossy().into_owned());
        }
        let thresholds = self.spec().quality.unwrap_or_default();
        let quality = assess(cameras.as_ref(), &image_sizes, &thresholds).ok();
        let ids: Vec<String> = dataset.cameras.iter().map(|c| c.id.clone()).collect();
        let record = RunRecord {
            version: RUN_RECORD_VERSION,
            timestamp: unix_now(),
            tool: format!("vision-calibration {}", env!("CARGO_PKG_VERSION")),
            library_version: env!("CARGO_PKG_VERSION").to_string(),
            source: RunSource::Imported {
                file_name: file
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_default(),
            },
            summary: RunSummary {
                kind,
                verdict: quality.as_ref().map(|q| q.verdict),
                cameras: camera_summaries(cameras.as_ref(), &ids, &image_sizes),
            },
        };
        let runs_dir = self.dir().join(RUNS_DIR);
        let mut files = vec![
            (EXPORT_FILE, json_text(&value, &runs_dir)?),
            (RUN_RECORD_FILE, json_text(&record, &runs_dir)?),
        ];
        if let Some(quality) = &quality {
            files.push((QUALITY_FILE, json_text(quality, &runs_dir)?));
        }
        if !notes.is_empty() {
            files.push((RUN_NOTES_FILE, json_text(&notes, &runs_dir)?));
        }
        let (id, dir) = self.write_run(&files)?;
        Ok(RunInfo {
            id,
            dir,
            state: RunState::Recorded(Box::new(record)),
            notes,
        })
    }

    /// The project's runs: numbered runs by number, then any others by
    /// name. A folder under `runs/` with neither `run.json` nor
    /// `export.json` is not a run, nor is one whose name starts with `.`.
    ///
    /// # Errors
    ///
    /// [`ProjectError::Io`] when `runs/` cannot be listed.
    pub fn runs(&self) -> Result<Vec<RunInfo>, ProjectError> {
        let runs_dir = self.dir().join(RUNS_DIR);
        let entries = match std::fs::read_dir(&runs_dir) {
            Ok(entries) => entries,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(err) => return Err(ProjectError::io(&runs_dir, err)),
        };
        let mut runs: Vec<RunInfo> = entries
            .filter_map(Result::ok)
            .filter(|e| e.file_type().is_ok_and(|t| t.is_dir()))
            .filter_map(|e| {
                let id = e.file_name().to_string_lossy().into_owned();
                if id.starts_with('.') {
                    return None;
                }
                read_run(&e.path(), id)
            })
            .collect();
        runs.sort_by(|a, b| run_order(&a.id).cmp(&run_order(&b.id)));
        Ok(runs)
    }

    /// The run `id`.
    ///
    /// # Errors
    ///
    /// [`ProjectError::RunNotFound`] when the project has no such run.
    pub fn run(&self, id: &str) -> Result<RunInfo, ProjectError> {
        let not_found = || ProjectError::RunNotFound { id: id.to_string() };
        if !is_plain_name(id) {
            return Err(not_found());
        }
        read_run(&self.dir().join(RUNS_DIR).join(id), id.to_string()).ok_or_else(not_found)
    }

    /// Replace the notes of run `id`, `notes.json`, in one step; empty
    /// notes remove the file.
    ///
    /// # Errors
    ///
    /// [`ProjectError::RunNotFound`]; [`ProjectError::Io`].
    pub fn set_run_notes(&self, id: &str, notes: &RunNotes) -> Result<(), ProjectError> {
        let run = self.run(id)?;
        let path = run.dir.join(RUN_NOTES_FILE);
        if notes.is_empty() {
            return match std::fs::remove_file(&path) {
                Ok(()) => Ok(()),
                Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(()),
                Err(err) => Err(ProjectError::io(&path, err)),
            };
        }
        write_atomic(&path, json_text(notes, &path)?.as_bytes())
    }

    /// Delete run `id` with every file in it. The folder is first renamed
    /// out of the listing, so no reader sees half of it.
    ///
    /// # Errors
    ///
    /// [`ProjectError::RunNotFound`]; [`ProjectError::Io`].
    pub fn delete_run(&self, id: &str) -> Result<(), ProjectError> {
        let run = self.run(id)?;
        let doomed = self
            .dir()
            .join(RUNS_DIR)
            .join(unique_name(&format!(".deleted-{id}")));
        std::fs::rename(&run.dir, &doomed).map_err(|e| ProjectError::io(&run.dir, e))?;
        std::fs::remove_dir_all(&doomed).map_err(|e| ProjectError::io(&doomed, e))
    }

    /// A folder at a run's depth under the project: what an export's image
    /// root is restated from, whichever id the run gets.
    fn run_dir_model(&self) -> PathBuf {
        self.dir().join(RUNS_DIR).join("run")
    }

    /// The data root as the project states it: the inline manifest's
    /// `root`, or the folder of the manifest file's path.
    fn stated_data_root(&self) -> PathBuf {
        match &self.spec().dataset {
            DatasetSource::Inline { root, .. } => root.clone(),
            DatasetSource::File { path } => match path.parent() {
                Some(parent) if !parent.as_os_str().is_empty() => parent.to_path_buf(),
                _ => PathBuf::from("."),
            },
        }
    }

    /// The project's name, or its folder's.
    pub(crate) fn display_name(&self) -> String {
        self.spec().name.clone().unwrap_or_else(|| {
            self.dir()
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default()
        })
    }

    /// The feature exclusions with the points that matched no feature of
    /// the detection they were applied to.
    fn unmatched_exclusions(&self, prepared: &Prepared) -> Vec<FeatureExclusion> {
        self.spec()
            .exclude
            .features
            .iter()
            .filter_map(|e| {
                let camera = prepared
                    .dataset
                    .cameras
                    .iter()
                    .position(|c| c.id == e.camera)?;
                let path = prepared.data_root.join(&e.image);
                let points: Vec<[f64; 3]> = e
                    .points
                    .iter()
                    .filter(|p| prepared.options.point_matched(camera, &path, p) == Some(false))
                    .copied()
                    .collect();
                (!points.is_empty()).then(|| FeatureExclusion {
                    camera: e.camera.clone(),
                    image: e.image.clone(),
                    points,
                    note: e.note.clone(),
                })
            })
            .collect()
    }

    /// Write `files` as a new run: into a staging folder, then renamed to
    /// the next free id. Returns the id and the run's folder.
    fn write_run(&self, files: &[(&str, String)]) -> Result<(String, PathBuf), ProjectError> {
        let runs_dir = self.dir().join(RUNS_DIR);
        std::fs::create_dir_all(&runs_dir).map_err(|e| ProjectError::io(&runs_dir, e))?;
        let staging = TempDir::create(runs_dir.join(unique_name(".staging")))?;
        for (name, text) in files {
            let path = staging.path().join(name);
            std::fs::write(&path, text).map_err(|e| ProjectError::io(&path, e))?;
        }
        for _ in 0..MAX_CLAIMS {
            let id = next_run_id(&runs_dir)?;
            let target = runs_dir.join(&id);
            match std::fs::rename(staging.path(), &target) {
                Ok(()) => {
                    staging.kept();
                    return Ok((id, target));
                }
                // Another writer claimed the id first: take the next one.
                Err(_) if target.exists() => continue,
                Err(err) => return Err(ProjectError::io(&target, err)),
            }
        }
        Err(ProjectError::io(
            &runs_dir,
            std::io::Error::other(format!("no free run id after {MAX_CLAIMS} tries")),
        ))
    }
}

/// Create the detection cache folder with a `.gitignore` of everything.
fn ensure_cache_dir(dir: &Path) -> Result<(), ProjectError> {
    std::fs::create_dir_all(dir).map_err(|e| ProjectError::io(dir, e))?;
    let ignore = dir.join(".gitignore");
    if !ignore.exists() {
        std::fs::write(&ignore, CACHE_GITIGNORE).map_err(|e| ProjectError::io(&ignore, e))?;
    }
    Ok(())
}

/// The number of a run id `run-<digits>`.
fn run_number(id: &str) -> Option<u64> {
    let digits = id.strip_prefix("run-")?;
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    digits.parse().ok()
}

/// Listing order: numbered runs by number, then the others by name.
fn run_order(id: &str) -> (u8, u64, &str) {
    match run_number(id) {
        Some(n) => (0, n, id),
        None => (1, 0, id),
    }
}

/// One more than the largest run number under `runs_dir`, as `run-NNN`.
fn next_run_id(runs_dir: &Path) -> Result<String, ProjectError> {
    let entries = std::fs::read_dir(runs_dir).map_err(|e| ProjectError::io(runs_dir, e))?;
    let largest = entries
        .filter_map(Result::ok)
        .filter_map(|e| run_number(&e.file_name().to_string_lossy()))
        .max()
        .unwrap_or(0);
    Ok(format!("run-{:03}", largest + 1))
}

/// A run id names one folder: not empty, not starting with `.`, no
/// separator.
fn is_plain_name(id: &str) -> bool {
    !id.is_empty() && !id.starts_with('.') && !id.contains(['/', '\\'])
}

/// The run in `dir`, as its files say; `None` when the folder holds no
/// run.
pub(crate) fn read_run(dir: &Path, id: String) -> Option<RunInfo> {
    let record_path = dir.join(RUN_RECORD_FILE);
    let state = if record_path.is_file() {
        match std::fs::read(&record_path)
            .map_err(|e| e.to_string())
            .and_then(|b| serde_json::from_slice::<RunRecord>(&b).map_err(|e| e.to_string()))
        {
            Ok(record) => RunState::Recorded(Box::new(record)),
            Err(message) => RunState::Unreadable { message },
        }
    } else if dir.join(EXPORT_FILE).is_file() {
        RunState::Unrecorded(unrecorded_run(dir))
    } else {
        return None;
    };
    let notes = std::fs::read(dir.join(RUN_NOTES_FILE))
        .ok()
        .and_then(|b| serde_json::from_slice(&b).ok())
        .unwrap_or_default();
    Some(RunInfo {
        id,
        dir: dir.to_path_buf(),
        state,
        notes,
    })
}

/// A run of the desktop app's earlier format: its export, the manifest it
/// used and its `diagnostics.json`.
fn unrecorded_run(dir: &Path) -> UnrecordedRun {
    #[derive(Deserialize)]
    struct Diagnostics {
        usable_views: Option<usize>,
        total_views: Option<usize>,
    }
    let diagnostics: Option<Diagnostics> = std::fs::read(dir.join("diagnostics.json"))
        .ok()
        .and_then(|b| serde_json::from_slice(&b).ok());
    let ids: Vec<String> = std::fs::read(dir.join("manifest.json"))
        .ok()
        .and_then(|b| serde_json::from_slice::<DatasetSpec>(&b).ok())
        .map(|m| m.cameras.into_iter().map(|c| c.id).collect())
        .unwrap_or_default();
    let summary = std::fs::read_to_string(dir.join(EXPORT_FILE))
        .ok()
        .and_then(|text| {
            let kind: ProblemKind = serde_json::from_str::<serde_json::Value>(&text)
                .ok()
                .and_then(|v| serde_json::from_value(v["kind"].clone()).ok())?;
            let cameras = calibrated_cameras(&text).ok()?;
            Some(RunSummary {
                kind,
                verdict: None,
                cameras: camera_summaries(cameras.as_ref(), &ids, &[]),
            })
        });
    UnrecordedRun {
        summary,
        usable_views: diagnostics.as_ref().and_then(|d| d.usable_views),
        total_views: diagnostics.as_ref().and_then(|d| d.total_views),
    }
}

/// Each frame of the export's image manifest (its root still the data root)
/// with the content hash its detection read; empty unless every frame has
/// one.
fn used_images(
    export: &dyn CalibratedCameras,
    data_root: &Path,
    ids: &[String],
    read: &std::collections::BTreeMap<PathBuf, u64>,
) -> Vec<UsedImage> {
    let Some(manifest) = export.diagnostics().image_manifest.as_ref() else {
        return Vec::new();
    };
    let root = data_root.join(&manifest.root);
    manifest
        .frames
        .iter()
        .map(|frame| {
            let hash = read.get(&lexical_absolute(&root.join(&frame.path)))?;
            Some(UsedImage {
                camera: camera_id(ids, frame.camera),
                image: slash_path(&frame.path),
                content: format!("{hash:016x}"),
            })
        })
        .collect::<Option<Vec<_>>>()
        .unwrap_or_default()
}

/// Each camera's summary from the export's residuals.
fn camera_summaries(
    export: &dyn CalibratedCameras,
    ids: &[String],
    image_sizes: &[[u32; 2]],
) -> Vec<CameraSummary> {
    let residuals = &export.diagnostics().per_feature_residuals.target;
    (0..export.camera_params().len())
        .map(|camera| {
            let mine: Vec<_> = residuals.iter().filter(|r| r.camera == camera).collect();
            let views: BTreeSet<usize> = mine.iter().map(|r| r.view).collect();
            let errors: Vec<f64> = mine.iter().filter_map(|r| r.error_px).collect();
            let rms_px = (!errors.is_empty())
                .then(|| (errors.iter().map(|e| e * e).sum::<f64>() / errors.len() as f64).sqrt());
            CameraSummary {
                id: camera_id(ids, camera),
                rms_px,
                features: mine.len(),
                views: views.len(),
                image_size: image_sizes.get(camera).copied(),
            }
        })
        .collect()
}

fn camera_id(ids: &[String], camera: usize) -> String {
    ids.get(camera)
        .cloned()
        .unwrap_or_else(|| format!("cam{camera}"))
}

/// Each camera's image size, from the first target frame of the camera the
/// manifest names (its region, when it has one); empty unless every camera
/// has one.
pub(crate) fn sizes_of(images: &ImageManifest, source_dir: &Path, cameras: usize) -> Vec<[u32; 2]> {
    let root = source_dir.join(&images.root);
    let sizes: Option<Vec<[u32; 2]>> = (0..cameras)
        .map(|camera| {
            let frame = images.target_frames().find(|f| f.camera == camera)?;
            match frame.roi {
                Some(roi) => Some([roi.w, roi.h]),
                None => image::image_dimensions(root.join(&frame.path))
                    .ok()
                    .map(|(w, h)| [w, h]),
            }
        })
        .collect();
    sizes.unwrap_or_default()
}

/// A recorded run's camera image sizes: the ones its summary records, else
/// read from the headers of the images its export names (a record written
/// before the summary held them); `None` when neither gives every camera.
pub(crate) fn recorded_sizes(
    cameras: &[CameraSummary],
    export: &dyn CalibratedCameras,
    dir: &Path,
) -> Option<Vec<[u32; 2]>> {
    let count = cameras.len();
    if let Some(sizes) = cameras.iter().map(|c| c.image_size).collect() {
        return Some(sizes);
    }
    let sizes = export
        .diagnostics()
        .image_manifest
        .as_ref()
        .map(|images| sizes_of(images, dir, count))
        .unwrap_or_default();
    (sizes.len() == count).then_some(sizes)
}

/// A path with `/` separators, as the project's files write them.
fn slash_path(path: &Path) -> PathBuf {
    PathBuf::from(path.to_string_lossy().replace('\\', "/"))
}

/// The name a workflow serializes to (`planar_intrinsics`).
fn kind_name(kind: ProblemKind) -> String {
    match serde_json::to_value(kind) {
        Ok(serde_json::Value::String(name)) => name,
        _ => format!("{kind:?}"),
    }
}

/// Seconds since the Unix epoch, for the record's timestamp; 0 on a clock
/// set before it.
fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

#[cfg(test)]
pub(crate) mod tests;
