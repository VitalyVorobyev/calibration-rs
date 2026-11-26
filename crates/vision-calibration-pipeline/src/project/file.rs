//! [`Project`]: a project folder opened, its file read, edited and saved.

use std::io::Write as _;
use std::path::{Path, PathBuf};

use serde::Deserialize;
use vision_calibration_core::lexical_absolute;
use vision_calibration_dataset::DatasetSpec;

use super::fs::{read, sha256, write_atomic};
use super::{
    DETECTIONS_DIR, DatasetSource, PROJECT_FILE, PROJECT_FILE_JSON, PROJECT_VERSION, ProjectError,
    ProjectSpec, RUNS_DIR,
};
use crate::dataset_runner::{CalibrationConfig, ConfigPatch, PairedViews, pair_dataset_views};

/// The manifest file of a version 1 project, beside its project file.
const V1_MANIFEST_FILE: &str = "dataset.json";

/// A calibration project: a folder holding the project file, the runs
/// calibrated from it (`runs/`) and the detection cache (`.detections/`).
///
/// [`open`](Self::open) reads the project file and never writes;
/// [`save`](Self::save) writes [`spec`](Self::spec) back, refusing when the
/// file changed on disk since it was read. A project of version 1 (the
/// desktop app's earlier `calibration-project.json` with its
/// `dataset.json`) opens as a version 2 project whose manifest is written
/// in the project file; [`file_version`](Self::file_version) stays 1 until
/// it is saved, which writes it as version 2 and removes `dataset.json`.
#[derive(Debug, Clone)]
pub struct Project {
    /// The project folder, absolute.
    dir: PathBuf,
    /// The project file in it.
    file: PathBuf,
    format: Format,
    spec: ProjectSpec,
    file_version: u32,
    /// The content revision of the files this value last read or wrote.
    revision: String,
}

/// The project file's format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Format {
    Toml,
    Json,
}

/// The version every project file states, read before the rest.
#[derive(Deserialize)]
struct VersionProbe {
    version: u32,
}

/// A version 1 project file: the folder its manifest's paths resolve
/// against, relative to the project unless absolute.
#[derive(Deserialize)]
struct V1File {
    dataset: PathBuf,
}

/// What reading a project folder gives.
struct Loaded {
    spec: ProjectSpec,
    file_version: u32,
    revision: String,
}

impl Project {
    /// Open the project in the folder `path`, or of the project file `path`.
    ///
    /// Nothing is written.
    ///
    /// # Errors
    ///
    /// - [`ProjectError::NotAProject`] for a folder without a project file,
    ///   or a file that is not one (a bare manifest included: nothing is
    ///   guessed from a folder's contents).
    /// - [`ProjectError::AmbiguousProject`] when the folder holds both
    ///   project file names.
    /// - [`ProjectError::UnsupportedVersion`], [`ProjectError::Parse`] and
    ///   [`ProjectError::Invalid`] for a project file this version does not
    ///   read; [`ProjectError::Io`] when it cannot be read.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, ProjectError> {
        let path = lexical_absolute(path.as_ref());
        let (dir, file) = if path.is_dir() {
            (path.clone(), marker(&path)?)
        } else {
            let named = path
                .file_name()
                .is_some_and(|n| n == PROJECT_FILE || n == PROJECT_FILE_JSON);
            let dir = path.parent().map(Path::to_path_buf).unwrap_or_default();
            if !path.is_file() {
                return Err(ProjectError::io(
                    &path,
                    std::io::Error::from(std::io::ErrorKind::NotFound),
                ));
            }
            if !named {
                return Err(ProjectError::NotAProject { dir: path });
            }
            // The other name beside it makes the folder ambiguous too.
            marker(&dir)?;
            (dir, path)
        };
        let format = format_of(&file);
        let loaded = load(&dir, &file, format)?;
        Ok(Self {
            dir,
            file,
            format,
            spec: loaded.spec,
            file_version: loaded.file_version,
            revision: loaded.revision,
        })
    }

    /// Make the folder `dir` (created when missing) a project of `spec`:
    /// writes `calibration-project.toml` and the `runs/` folder.
    ///
    /// # Errors
    ///
    /// [`ProjectError::ProjectExists`] when `dir` already holds a project
    /// file; [`ProjectError::Invalid`] for a spec that cannot be saved
    /// (another version, an empty device id); [`ProjectError::Parse`] when
    /// it cannot be written as TOML (a `null` in the configuration
    /// changes); [`ProjectError::Io`].
    pub fn create(dir: impl AsRef<Path>, spec: ProjectSpec) -> Result<Self, ProjectError> {
        let dir = lexical_absolute(dir.as_ref());
        let file = dir.join(PROJECT_FILE);
        validate(&spec, &file)?;
        let text = toml::to_string_pretty(&spec).map_err(|e| ProjectError::parse(&file, e))?;
        std::fs::create_dir_all(&dir).map_err(|e| ProjectError::io(&dir, e))?;
        if dir.join(PROJECT_FILE_JSON).exists() {
            return Err(ProjectError::ProjectExists { dir });
        }
        // `create_new` refuses an existing file, so two creators never
        // overwrite each other.
        let mut out = match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&file)
        {
            Ok(out) => out,
            Err(err) if err.kind() == std::io::ErrorKind::AlreadyExists => {
                return Err(ProjectError::ProjectExists { dir });
            }
            Err(err) => return Err(ProjectError::io(&file, err)),
        };
        out.write_all(text.as_bytes())
            .and_then(|()| out.sync_all())
            .map_err(|e| ProjectError::io(&file, e))?;
        let runs = dir.join(RUNS_DIR);
        std::fs::create_dir_all(&runs).map_err(|e| ProjectError::io(&runs, e))?;
        Ok(Self {
            dir,
            file,
            format: Format::Toml,
            spec,
            file_version: PROJECT_VERSION,
            revision: sha256(text.as_bytes()),
        })
    }

    /// Write [`spec`](Self::spec) to the project file, in its format, in one
    /// step. A version 1 project is written as version 2 and its
    /// `dataset.json` removed. A hand-written TOML file loses its comments.
    ///
    /// # Errors
    ///
    /// [`ProjectError::ChangedOnDisk`] when the project file (or a version 1
    /// project's `dataset.json`) is not what this value last read or
    /// wrote: [`reload`](Self::reload) and apply the edit again.
    /// [`ProjectError::Invalid`] for a spec that cannot be saved;
    /// [`ProjectError::Parse`] when it cannot be written as TOML;
    /// [`ProjectError::Io`].
    pub fn save(&mut self) -> Result<(), ProjectError> {
        validate(&self.spec, &self.file)?;
        let text = match self.format {
            Format::Toml => toml::to_string_pretty(&self.spec)
                .map_err(|e| ProjectError::parse(&self.file, e))?,
            Format::Json => super::fs::json_text(&self.spec, &self.file)?,
        };
        if disk_revision(&self.dir, &self.file, self.file_version).as_deref()
            != Some(self.revision.as_str())
        {
            return Err(ProjectError::ChangedOnDisk {
                path: self.file.clone(),
            });
        }
        write_atomic(&self.file, text.as_bytes())?;
        if self.file_version == 1 {
            // Version 2 never reads it; a crash before this leaves a valid
            // project beside an ignored file.
            let manifest = self.dir.join(V1_MANIFEST_FILE);
            match std::fs::remove_file(&manifest) {
                Ok(()) => {}
                Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
                Err(err) => return Err(ProjectError::io(&manifest, err)),
            }
            self.file_version = PROJECT_VERSION;
        }
        self.revision = sha256(text.as_bytes());
        Ok(())
    }

    /// Read the project file again, dropping unsaved edits.
    ///
    /// # Errors
    ///
    /// As [`open`](Self::open).
    pub fn reload(&mut self) -> Result<(), ProjectError> {
        let loaded = load(&self.dir, &self.file, self.format)?;
        self.spec = loaded.spec;
        self.file_version = loaded.file_version;
        self.revision = loaded.revision;
        Ok(())
    }

    /// The project folder, absolute.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// The project file.
    pub fn file(&self) -> &Path {
        &self.file
    }

    /// The version of the project file on disk: 1 for an app project of
    /// the earlier format until it is saved, else [`PROJECT_VERSION`].
    pub fn file_version(&self) -> u32 {
        self.file_version
    }

    /// The content revision of the project's files as this value last read
    /// or wrote them, `sha256:<hex>`. A caller that keeps no `Project`
    /// between an edit and its save (a desktop app's back end, say) keeps
    /// the revision instead: when a freshly opened project's revision
    /// differs, the file changed in between and the edit is stale.
    pub fn revision(&self) -> &str {
        &self.revision
    }

    /// The project's definition, with any unsaved edits.
    pub fn spec(&self) -> &ProjectSpec {
        &self.spec
    }

    /// The project's definition, to edit before [`save`](Self::save).
    pub fn spec_mut(&mut self) -> &mut ProjectSpec {
        &mut self.spec
    }

    /// The folder the manifest's relative paths resolve against, absolute:
    /// the inline manifest's `root` against the project folder, or a
    /// manifest file's own folder.
    ///
    /// # Errors
    ///
    /// None in this version; the signature leaves room for a check of the
    /// folder.
    pub fn data_root(&self) -> Result<PathBuf, ProjectError> {
        Ok(match &self.spec.dataset {
            DatasetSource::Inline { root, .. } => lexical_absolute(&self.dir.join(root)),
            DatasetSource::File { path } => {
                let manifest = lexical_absolute(&self.dir.join(path));
                manifest
                    .parent()
                    .map_or_else(|| self.dir.clone(), Path::to_path_buf)
            }
        })
    }

    /// The dataset manifest: the inline one, or the referenced file read
    /// now.
    ///
    /// # Errors
    ///
    /// [`ProjectError::Manifest`] when the referenced file cannot be read.
    pub fn dataset(&self) -> Result<DatasetSpec, ProjectError> {
        match &self.spec.dataset {
            DatasetSource::Inline { spec, .. } => Ok(spec.as_ref().clone()),
            DatasetSource::File { path } => Ok(DatasetSpec::from_path(lexical_absolute(
                &self.dir.join(path),
            ))?),
        }
    }

    /// The configuration of the dataset's workflow: its defaults with the
    /// project's changes, checked.
    ///
    /// # Errors
    ///
    /// [`ProjectError::Manifest`] as [`dataset`](Self::dataset);
    /// [`ProjectError::Config`] when the changes do not make a valid
    /// configuration.
    pub fn config(&self) -> Result<CalibrationConfig, ProjectError> {
        let dataset = self.dataset()?;
        CalibrationConfig::patched(dataset.topology, &self.spec.config)
            .map_err(ProjectError::Config)
    }

    /// The views the dataset's images pair into, without reading an image:
    /// what an exclusion names, and what a review lists.
    ///
    /// # Errors
    ///
    /// [`ProjectError::Manifest`] as [`dataset`](Self::dataset);
    /// [`ProjectError::Calibrate`] when the images cannot be listed or
    /// paired.
    pub fn views(&self) -> Result<PairedViews, ProjectError> {
        let dataset = self.dataset()?;
        Ok(pair_dataset_views(&dataset, &self.data_root()?)?)
    }

    /// Check everything a run checks before it reads an image: the project
    /// file's fields, the manifest (complete and valid, its pairing
    /// stated), the configuration changes, the quality thresholds, and that
    /// every exclusion names an image, view or camera of the dataset. The
    /// image folders are listed; no image is read.
    ///
    /// # Errors
    ///
    /// [`ProjectError::Invalid`] for a field the project cannot have (an
    /// empty device id, a threshold out of range); [`ProjectError::Manifest`]
    /// as [`dataset`](Self::dataset); [`ProjectError::Calibrate`] for an
    /// invalid manifest or images that cannot be listed or paired;
    /// [`ProjectError::Config`]; [`ProjectError::Exclusion`] for a stale
    /// exclusion.
    pub fn check(&self) -> Result<(), ProjectError> {
        self.prepare(&ConfigPatch::default()).map(drop)
    }

    /// The project's detection cache, `.detections/` in its folder.
    pub fn detection_cache_dir(&self) -> PathBuf {
        self.dir.join(DETECTIONS_DIR)
    }
}

/// The project file of the folder `dir`.
fn marker(dir: &Path) -> Result<PathBuf, ProjectError> {
    let toml = dir.join(PROJECT_FILE);
    let json = dir.join(PROJECT_FILE_JSON);
    match (toml.is_file(), json.is_file()) {
        (true, true) => Err(ProjectError::AmbiguousProject {
            dir: dir.to_path_buf(),
        }),
        (true, false) => Ok(toml),
        (false, true) => Ok(json),
        (false, false) => Err(ProjectError::NotAProject {
            dir: dir.to_path_buf(),
        }),
    }
}

fn format_of(file: &Path) -> Format {
    if file
        .extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("toml"))
    {
        Format::Toml
    } else {
        Format::Json
    }
}

/// The revision of a set of file contents: the hash of the one file, or
/// of the two files' hashes.
fn revision_of(project: &[u8], v1_manifest: Option<&[u8]>) -> String {
    match v1_manifest {
        None => sha256(project),
        Some(manifest) => {
            let joined = format!("{}\n{}", sha256(project), sha256(manifest));
            sha256(joined.as_bytes())
        }
    }
}

/// The revision of the files on disk now; `None` when one is missing.
fn disk_revision(dir: &Path, file: &Path, file_version: u32) -> Option<String> {
    let project = std::fs::read(file).ok()?;
    if file_version == 1 {
        let manifest = std::fs::read(dir.join(V1_MANIFEST_FILE)).ok()?;
        Some(revision_of(&project, Some(&manifest)))
    } else {
        Some(revision_of(&project, None))
    }
}

fn load(dir: &Path, file: &Path, format: Format) -> Result<Loaded, ProjectError> {
    let bytes = read(file)?;
    let text = std::str::from_utf8(&bytes).map_err(|e| ProjectError::parse(file, e))?;
    let probe: VersionProbe = match format {
        Format::Toml => toml::from_str(text).map_err(|e| ProjectError::parse(file, e))?,
        Format::Json => serde_json::from_str(text).map_err(|e| ProjectError::parse(file, e))?,
    };
    match (format, probe.version) {
        (_, PROJECT_VERSION) => {
            let spec: ProjectSpec = match format {
                Format::Toml => toml::from_str(text).map_err(|e| ProjectError::parse(file, e))?,
                Format::Json => {
                    serde_json::from_str(text).map_err(|e| ProjectError::parse(file, e))?
                }
            };
            validate(&spec, file)?;
            Ok(Loaded {
                spec,
                file_version: PROJECT_VERSION,
                revision: revision_of(&bytes, None),
            })
        }
        (Format::Json, 1) => {
            let v1: V1File =
                serde_json::from_str(text).map_err(|e| ProjectError::parse(file, e))?;
            let manifest_path = dir.join(V1_MANIFEST_FILE);
            let manifest = read(&manifest_path)?;
            let dataset: DatasetSpec = serde_json::from_slice(&manifest)
                .map_err(|e| ProjectError::parse(&manifest_path, e))?;
            let mut spec = ProjectSpec::new(DatasetSource::inline(v1.dataset, dataset));
            spec.name = dir.file_name().map(|n| n.to_string_lossy().into_owned());
            Ok(Loaded {
                spec,
                file_version: 1,
                revision: revision_of(&bytes, Some(&manifest)),
            })
        }
        (_, version) => Err(ProjectError::UnsupportedVersion {
            path: file.to_path_buf(),
            version,
        }),
    }
}

/// What a project file must hold to be written or used.
pub(crate) fn validate(spec: &ProjectSpec, file: &Path) -> Result<(), ProjectError> {
    if spec.version != PROJECT_VERSION {
        return Err(ProjectError::invalid(
            file,
            "version",
            format!("must be {PROJECT_VERSION}, not {}", spec.version),
        ));
    }
    if let Some(quality) = &spec.quality {
        quality
            .validate()
            .map_err(|e| ProjectError::invalid(file, "quality", e.to_string()))?;
    }
    if let Some(device) = &spec.device {
        if device.id.trim().is_empty() {
            return Err(ProjectError::invalid(file, "device.id", "is empty"));
        }
        if device.id.trim() != device.id {
            return Err(ProjectError::invalid(
                file,
                "device.id",
                "has leading or trailing white space",
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::project::{DeviceIdentity, ExclusionFault, ImageExclusion};
    use serde_json::json;

    fn manifest() -> serde_json::Value {
        json!({
            "version": 1,
            "topology": "planar_intrinsics",
            "cameras": [{ "id": "cam0", "images": { "kind": "glob", "pattern": "img/*.png" } }],
            "target": { "kind": "chessboard", "rows": 6, "cols": 8, "square_size_m": 0.03 },
        })
    }

    fn inline() -> ProjectSpec {
        ProjectSpec::new(DatasetSource::inline(
            ".",
            serde_json::from_value(manifest()).unwrap(),
        ))
    }

    fn value(v: &impl serde::Serialize) -> serde_json::Value {
        serde_json::to_value(v).unwrap()
    }

    #[test]
    fn create_then_open_round_trips_the_spec() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("bench");
        let mut spec = inline();
        spec.device = Some(DeviceIdentity::new("SN-1"));
        let created = Project::create(&dir, spec.clone()).unwrap();
        assert!(dir.join(PROJECT_FILE).is_file());
        assert!(dir.join(RUNS_DIR).is_dir());
        for path in [dir.clone(), dir.join(PROJECT_FILE)] {
            let opened = Project::open(&path).unwrap();
            assert_eq!(value(opened.spec()), value(&spec));
            assert_eq!(opened.revision(), created.revision());
            assert_eq!(opened.file_version(), PROJECT_VERSION);
            assert_eq!(opened.dir(), lexical_absolute(&dir));
        }
        let err = Project::create(&dir, spec).unwrap_err();
        assert!(matches!(err, ProjectError::ProjectExists { .. }), "{err}");
    }

    #[test]
    fn a_folder_without_a_project_file_is_not_a_project() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("dataset.toml"), "version = 1").unwrap();
        for path in [tmp.path().to_path_buf(), tmp.path().join("dataset.toml")] {
            let err = Project::open(&path).unwrap_err();
            assert!(matches!(err, ProjectError::NotAProject { .. }), "{err}");
        }
        assert!(matches!(
            Project::open(tmp.path().join("missing")).unwrap_err(),
            ProjectError::Io { .. }
        ));
    }

    #[test]
    fn both_project_files_make_the_folder_ambiguous() {
        let tmp = tempfile::tempdir().unwrap();
        Project::create(tmp.path(), inline()).unwrap();
        std::fs::write(tmp.path().join(PROJECT_FILE_JSON), "{}").unwrap();
        for path in [tmp.path().to_path_buf(), tmp.path().join(PROJECT_FILE)] {
            let err = Project::open(&path).unwrap_err();
            assert!(
                matches!(err, ProjectError::AmbiguousProject { .. }),
                "{err}"
            );
        }
    }

    #[test]
    fn a_newer_version_is_refused() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(
            tmp.path().join(PROJECT_FILE),
            "version = 3\n[dataset]\nmanifest = \"d.toml\"\n",
        )
        .unwrap();
        let err = Project::open(tmp.path()).unwrap_err();
        assert!(
            matches!(err, ProjectError::UnsupportedVersion { version: 3, .. }),
            "{err}"
        );
    }

    #[test]
    fn save_writes_the_edit_and_refuses_a_file_changed_on_disk() {
        let tmp = tempfile::tempdir().unwrap();
        let mut project = Project::create(tmp.path(), inline()).unwrap();
        let before = project.revision().to_string();
        project.spec_mut().name = Some("Bench".into());
        project.save().unwrap();
        assert_ne!(project.revision(), before);
        assert_eq!(
            Project::open(tmp.path()).unwrap().spec().name.as_deref(),
            Some("Bench")
        );

        // Another writer edits the file: this value's save is refused,
        // and a reload brings the other edit in.
        let mut other = Project::open(tmp.path()).unwrap();
        other.spec_mut().description = Some("from elsewhere".into());
        other.save().unwrap();
        project.spec_mut().name = Some("Mine".into());
        let err = project.save().unwrap_err();
        assert!(matches!(err, ProjectError::ChangedOnDisk { .. }), "{err}");
        project.reload().unwrap();
        assert_eq!(
            project.spec().description.as_deref(),
            Some("from elsewhere")
        );
        assert_eq!(project.spec().name.as_deref(), Some("Bench"));
        project.spec_mut().name = Some("Mine".into());
        project.save().unwrap();
    }

    #[test]
    fn a_json_project_is_saved_as_json() {
        let tmp = tempfile::tempdir().unwrap();
        let text = serde_json::to_string(&json!({
            "version": 2, "dataset": { "manifest": manifest() }
        }))
        .unwrap();
        std::fs::write(tmp.path().join(PROJECT_FILE_JSON), text).unwrap();
        let mut project = Project::open(tmp.path()).unwrap();
        project.spec_mut().name = Some("j".into());
        project.save().unwrap();
        let saved: serde_json::Value =
            serde_json::from_slice(&std::fs::read(tmp.path().join(PROJECT_FILE_JSON)).unwrap())
                .unwrap();
        assert_eq!(saved["name"], "j");
        assert!(!tmp.path().join(PROJECT_FILE).exists());
    }

    /// A project the desktop app wrote in its first format, exactly as its
    /// writer laid it out: `{"version": 1, "dataset": "."}` and the
    /// manifest in `dataset.json`, both pretty JSON with a final newline.
    fn write_v1(dir: &Path, dataset: &str) {
        let project = json!({ "version": 1, "dataset": dataset });
        std::fs::write(
            dir.join(PROJECT_FILE_JSON),
            serde_json::to_string_pretty(&project).unwrap() + "\n",
        )
        .unwrap();
        std::fs::write(
            dir.join(V1_MANIFEST_FILE),
            serde_json::to_string_pretty(&manifest()).unwrap() + "\n",
        )
        .unwrap();
    }

    #[test]
    fn a_version_1_project_opens_and_saves_as_version_2() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("bench-cam");
        std::fs::create_dir_all(dir.join("images")).unwrap();
        write_v1(&dir, "images");
        let before = std::fs::read(dir.join(PROJECT_FILE_JSON)).unwrap();

        let mut project = Project::open(&dir).unwrap();
        // Opening writes nothing.
        assert_eq!(std::fs::read(dir.join(PROJECT_FILE_JSON)).unwrap(), before);
        assert_eq!(project.file_version(), 1);
        assert_eq!(project.spec().version, PROJECT_VERSION);
        assert_eq!(project.spec().name.as_deref(), Some("bench-cam"));
        assert_eq!(
            project.data_root().unwrap(),
            lexical_absolute(&dir.join("images"))
        );
        assert_eq!(
            value(&project.dataset().unwrap()),
            value(&serde_json::from_value::<DatasetSpec>(manifest()).unwrap())
        );
        assert!(project.spec().config.is_empty());

        project.save().unwrap();
        assert_eq!(project.file_version(), PROJECT_VERSION);
        assert!(!dir.join(V1_MANIFEST_FILE).exists());
        let saved: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.join(PROJECT_FILE_JSON)).unwrap()).unwrap();
        assert_eq!(saved["version"], 2);
        assert_eq!(saved["dataset"]["root"], "images");
        assert_eq!(
            saved["dataset"]["manifest"]["topology"],
            "planar_intrinsics"
        );
        let reopened = Project::open(&dir).unwrap();
        assert_eq!(reopened.file_version(), PROJECT_VERSION);
        assert_eq!(value(reopened.spec()), value(project.spec()));
    }

    #[test]
    fn a_version_1_project_s_manifest_edit_outside_is_a_change_on_disk() {
        let tmp = tempfile::tempdir().unwrap();
        write_v1(tmp.path(), ".");
        let mut project = Project::open(tmp.path()).unwrap();
        std::fs::write(tmp.path().join(V1_MANIFEST_FILE), "{}").unwrap();
        assert!(matches!(
            project.save().unwrap_err(),
            ProjectError::ChangedOnDisk { .. }
        ));
    }

    #[test]
    fn a_referenced_manifest_is_read_from_its_file_and_its_folder_is_the_root() {
        let tmp = tempfile::tempdir().unwrap();
        let data = tmp.path().join("captures");
        std::fs::create_dir_all(&data).unwrap();
        std::fs::write(
            data.join("dataset.json"),
            serde_json::to_string(&manifest()).unwrap(),
        )
        .unwrap();
        let project = Project::create(
            tmp.path().join("proj"),
            ProjectSpec::for_manifest("../captures/dataset.json"),
        )
        .unwrap();
        assert_eq!(project.data_root().unwrap(), lexical_absolute(&data));
        assert_eq!(project.dataset().unwrap().cameras[0].id, "cam0");
        assert!(matches!(
            project.config().unwrap(),
            CalibrationConfig::PlanarIntrinsics(_)
        ));
        std::fs::remove_file(data.join("dataset.json")).unwrap();
        assert!(matches!(
            project.dataset().unwrap_err(),
            ProjectError::Manifest(_)
        ));
    }

    #[test]
    fn an_invalid_spec_is_not_written() {
        let tmp = tempfile::tempdir().unwrap();
        let mut spec = inline();
        spec.device = Some(DeviceIdentity::new(" SN-1"));
        let err = Project::create(tmp.path(), spec).unwrap_err();
        assert!(
            matches!(&err, ProjectError::Invalid { field, .. } if field == "device.id"),
            "{err}"
        );
        assert!(!tmp.path().join(PROJECT_FILE).exists());
        let mut bad = inline();
        bad.config =
            crate::dataset_runner::ConfigPatch::from_json_str(r#"{"solver": {"max_iters": "x"}}"#)
                .unwrap();
        let project = Project::create(tmp.path(), bad).unwrap();
        assert!(matches!(
            project.config().unwrap_err(),
            ProjectError::Config(_)
        ));
    }

    #[test]
    fn views_and_exclusions_resolve_without_reading_an_image() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(tmp.path().join("img")).unwrap();
        for i in 0..3 {
            std::fs::write(tmp.path().join(format!("img/{i}.png")), b"-").unwrap();
        }
        let mut project = Project::create(tmp.path(), inline()).unwrap();
        assert_eq!(project.views().unwrap().paths.len(), 3);
        project.check().unwrap();
        project
            .spec_mut()
            .exclude
            .images
            .push(ImageExclusion::new("cam0", "img/1.png"));
        project.check().unwrap();
        project
            .spec_mut()
            .exclude
            .images
            .push(ImageExclusion::new("cam0", "img/7.png"));
        let err = project.check().unwrap_err();
        assert!(
            matches!(&err, ProjectError::Exclusion { entry, fault: ExclusionFault::ImageNotFound } if entry == "exclude.images[1]"),
            "{err}"
        );
        project.spec_mut().exclude = Default::default();
        let thresholds = project
            .spec_mut()
            .quality
            .insert(crate::analysis::QualityThresholds::default());
        thresholds.coverage_warn_below = 2.0;
        assert!(matches!(
            project.check().unwrap_err(),
            ProjectError::Invalid { field, .. } if field == "quality"
        ));
    }

    #[test]
    fn exclusion_errors_name_their_entry() {
        let tmp = tempfile::tempdir().unwrap();
        let mut project = Project::create(tmp.path(), inline()).unwrap();
        project
            .spec_mut()
            .exclude
            .images
            .push(ImageExclusion::new("cam9", "img/7.png"));
        let dataset = project.dataset().unwrap();
        let err: ProjectError = project
            .spec()
            .exclude
            .resolve(&dataset, &project.data_root().unwrap())
            .unwrap_err()
            .into();
        assert!(
            matches!(&err, ProjectError::Exclusion { entry, fault: ExclusionFault::UnknownCamera } if entry == "exclude.images[0]"),
            "{err}"
        );
    }
}
