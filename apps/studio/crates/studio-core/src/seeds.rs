//! Seeds as the back end reads them: the paths a project's `[seed]` names,
//! judged against the roots before the library follows them, and the seed of
//! a run ([`seed_of_run`], the library's `DeviceSeed::from_run` behind the
//! same checks). The seed logic itself is the library's
//! (`vision_calibration::device_seed`, `project::SeedSource`).

use std::path::{Path, PathBuf};

use vision_calibration::device_seed::DeviceSeed;
use vision_calibration::project::{EXPORT_FILE, Project, SeedSource};

use crate::api::{ErrorKind, StudioError};
use crate::scope::RootScope;

/// The path `source` names, as written in the project file.
pub fn seed_path(source: &SeedSource) -> Option<&Path> {
    match source {
        SeedSource::Run(path) | SeedSource::DeviceSpec(path) | SeedSource::File(path) => Some(path),
        // A source of a later library version: nothing this app can judge.
        _ => None,
    }
}

impl RootScope {
    /// The path the seed `source` of a project in `dir` names (relative to
    /// the project folder unless absolute), inside a root, lexically: what
    /// opening a project checks before anything reads the seed.
    pub fn check_seed_path(&self, dir: &Path, source: &SeedSource) -> Result<PathBuf, StudioError> {
        let Some(path) = seed_path(source) else {
            return Err(
                StudioError::invalid("a seed source this app does not know").on_field("seed")
            );
        };
        let target = dir.join(path);
        self.check_lexical(&target).map_err(|_| {
            StudioError::new(
                ErrorKind::ForbiddenPath,
                format!(
                    "the project's seed {} is outside the folders this back end serves",
                    target.display()
                ),
            )
            .at(target.to_string_lossy())
            .on_field("seed")
        })
    }

    /// [`check_seed_path`](Self::check_seed_path), and for a run, every
    /// image its export's image manifest names (the library reads their
    /// headers when the run's record does not state its image sizes):
    /// what is checked before the library resolves the seed.
    pub fn check_seed_inputs(&self, dir: &Path, source: &SeedSource) -> Result<(), StudioError> {
        let path = self.check_seed_path(dir, source)?;
        if let SeedSource::Run(_) = source {
            let export = if path.is_file() {
                path
            } else {
                path.join(EXPORT_FILE)
            };
            if export.is_file() {
                self.check_export_images(&self.check(&export)?)?;
            }
        }
        Ok(())
    }

    /// The seed paths of `project`'s `[seed]`, when it has one
    /// ([`check_seed_inputs`](Self::check_seed_inputs)).
    pub fn check_project_seed(&self, project: &Project) -> Result<(), StudioError> {
        match &project.spec().seed {
            Some(source) => self.check_seed_inputs(project.dir(), source),
            None => Ok(()),
        }
    }

    /// The images an export's image manifest names, judged against the
    /// roots: an export may name any folder.
    pub fn check_export_images(&self, file: &Path) -> Result<(), StudioError> {
        let Ok(text) = std::fs::read_to_string(file) else {
            // The library reports what it cannot read.
            return Ok(());
        };
        let Ok(export) = serde_json::from_str::<serde_json::Value>(&text) else {
            return Ok(());
        };
        let Some(manifest) = export.pointer("/diagnostics/image_manifest") else {
            return Ok(());
        };
        let base = file.parent().map(Path::to_path_buf).unwrap_or_default();
        let root = base.join(manifest["root"].as_str().unwrap_or("."));
        let mut paths: Vec<PathBuf> = vec![root.clone()];
        for frame in manifest["frames"].as_array().into_iter().flatten() {
            if let Some(path) = frame["path"].as_str() {
                paths.push(root.join(path));
            }
        }
        for path in paths {
            self.check_lexical(&path).map_err(|_| {
                StudioError::new(
                    ErrorKind::ForbiddenPath,
                    format!(
                        "the export names {}, which is outside the folders this back end serves",
                        path.display()
                    ),
                )
                .at(path.to_string_lossy())
            })?;
        }
        Ok(())
    }
}

/// The seed of the run in the folder `run` (the library's
/// `DeviceSeed::from_run`): the folder inside a root, and the images its
/// export names judged first.
pub fn seed_of_run(scope: &RootScope, run: &Path) -> Result<DeviceSeed, StudioError> {
    let dir = scope.check(run)?;
    let export = dir.join(EXPORT_FILE);
    if export.is_file() {
        scope.check_export_images(&scope.check(&export)?)?;
    }
    Ok(DeviceSeed::from_run(&dir)?)
}
