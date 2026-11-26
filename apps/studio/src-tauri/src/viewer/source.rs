//! The run a viewer serves, found from the path given on its command line.

use std::path::{Path, PathBuf};

use serde::Serialize;

/// The export `vcal run -o <dir>` writes, next to `quality.json` and,
/// with `--report`, `report.html`.
const VCAL_EXPORT: &str = "calibration.json";
/// The export of a run the desktop app saved, `runs/<name>/export.json`,
/// next to `manifest.json`, `config.json` and `diagnostics.json`.
const PROJECT_RUN_EXPORT: &str = "export.json";
/// The file that marks an app project folder (whose runs are under `runs/`).
const PROJECT_FILE: &str = "calibration-project.json";

/// How the run's files are laid out.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RunLayout {
    /// A `vcal run -o <dir>` folder: `calibration.json`, `quality.json`.
    Vcal,
    /// A run the desktop app saved in a project: `runs/<name>/export.json`.
    ProjectRun,
    /// One export file, wherever it is.
    ExportFile,
}

/// A run to serve: its folder and its export file, both canonical.
#[derive(Debug, Clone)]
pub struct RunSource {
    pub layout: RunLayout,
    /// The run folder (an export file's own folder for [`RunLayout::ExportFile`]).
    pub dir: PathBuf,
    /// The export file.
    pub export: PathBuf,
}

/// The run at `path`: an export file, a `vcal run` folder or a saved app
/// run. Anything else (a project folder, a folder with both exports, a
/// folder with neither) is an error that says what to pass instead.
pub fn find_run(path: &Path) -> Result<RunSource, String> {
    let canonical = dunce::canonicalize(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if canonical.is_file() {
        let dir = canonical
            .parent()
            .ok_or_else(|| format!("{} has no parent folder", canonical.display()))?
            .to_path_buf();
        return Ok(RunSource {
            layout: RunLayout::ExportFile,
            dir,
            export: canonical,
        });
    }
    let vcal = canonical.join(VCAL_EXPORT);
    let project_run = canonical.join(PROJECT_RUN_EXPORT);
    let (layout, export) = match (vcal.is_file(), project_run.is_file()) {
        (true, false) => (RunLayout::Vcal, vcal),
        (false, true) => (RunLayout::ProjectRun, project_run),
        (true, true) => {
            return Err(format!(
                "{} holds both {VCAL_EXPORT} and {PROJECT_RUN_EXPORT}; pass the export file to view",
                canonical.display()
            ));
        }
        (false, false) if canonical.join(PROJECT_FILE).is_file() => {
            return Err(format!(
                "{} is a project folder; pass one of its runs, runs/<name>",
                canonical.display()
            ));
        }
        (false, false) => {
            return Err(format!(
                "{} is not a run folder: expected {VCAL_EXPORT} (from `vcal run -o`) or \
                 {PROJECT_RUN_EXPORT} (a run saved by the app), or pass an export file",
                canonical.display()
            ));
        }
    };
    Ok(RunSource {
        layout,
        dir: canonical,
        export,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn touch(path: &Path) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, "{}").unwrap();
    }

    #[test]
    fn a_vcal_folder_serves_its_calibration_json() {
        let tmp = tempfile::tempdir().unwrap();
        touch(&tmp.path().join("calibration.json"));
        touch(&tmp.path().join("quality.json"));
        let run = find_run(tmp.path()).unwrap();
        assert_eq!(run.layout, RunLayout::Vcal);
        assert_eq!(run.dir, dunce::canonicalize(tmp.path()).unwrap());
        assert_eq!(run.export, run.dir.join("calibration.json"));
    }

    #[test]
    fn a_saved_app_run_serves_its_export_json() {
        let tmp = tempfile::tempdir().unwrap();
        let run_dir = tmp.path().join("runs/run-001");
        for file in [
            "manifest.json",
            "config.json",
            "export.json",
            "diagnostics.json",
        ] {
            touch(&run_dir.join(file));
        }
        let run = find_run(&run_dir).unwrap();
        assert_eq!(run.layout, RunLayout::ProjectRun);
        assert_eq!(run.export, run.dir.join("export.json"));
    }

    #[test]
    fn an_export_file_is_served_from_its_folder() {
        let tmp = tempfile::tempdir().unwrap();
        let file = tmp.path().join("my_export.json");
        touch(&file);
        let run = find_run(&file).unwrap();
        assert_eq!(run.layout, RunLayout::ExportFile);
        assert_eq!(run.dir, dunce::canonicalize(tmp.path()).unwrap());
        assert_eq!(run.export, dunce::canonicalize(&file).unwrap());
    }

    #[test]
    fn ambiguous_and_foreign_folders_are_errors_that_say_what_to_pass() {
        let both = tempfile::tempdir().unwrap();
        touch(&both.path().join("calibration.json"));
        touch(&both.path().join("export.json"));
        assert!(find_run(both.path()).unwrap_err().contains("both"));

        let project = tempfile::tempdir().unwrap();
        touch(&project.path().join("calibration-project.json"));
        assert!(
            find_run(project.path())
                .unwrap_err()
                .contains("runs/<name>")
        );

        let empty = tempfile::tempdir().unwrap();
        assert!(
            find_run(empty.path())
                .unwrap_err()
                .contains("not a run folder")
        );

        assert!(find_run(&empty.path().join("missing")).is_err());
    }
}
