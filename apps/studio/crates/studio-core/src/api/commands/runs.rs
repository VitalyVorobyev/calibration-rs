//! A project's runs: calibrating the next one (`calibrate`, a job over the
//! library's `Project::calibrate`) and reading one (its listing entry with
//! the record, its export, its quality report). A run is named by its
//! project folder and id ([`RunRef`]).

use std::sync::Mutex;
use std::time::{Duration, Instant};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use vision_calibration::analysis::QualityReport;
use vision_calibration::control::{Progress, RunControl};
use vision_calibration::dataset_runner::ConfigPatch;
use vision_calibration::project::{ProjectRunOptions, RunInfo, RunNotes};

use crate::api::{Ctx, ErrorKind, JobCx, StudioError};
use crate::runs::{RunRef, open_project, resolve, slash};
use crate::scope::UserPath;

/// Calibrate the project as it is on disk and write the run as its next
/// `runs/run-NNN/` (export, quality report, record, the HTML report, and
/// `notes` when given). `config_patch` changes the configuration for this
/// run only, over the project's, and is recorded with it; `redetect`
/// detects every image again instead of reading the detection cache. Every
/// file the project names must lie inside a root. Its events are the
/// library's progress reports; a cancelled run writes nothing and ends the
/// job `cancelled`.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CalibrateArgs {
    /// The project folder.
    pub dir: UserPath,
    #[serde(default)]
    pub notes: RunNotes,
    #[serde(default)]
    pub config_patch: ConfigPatch,
    #[serde(default)]
    pub redetect: bool,
}

pub fn calibrate(
    ctx: &Ctx,
    args: CalibrateArgs,
    job: &JobCx<Progress>,
) -> Result<RunInfo, StudioError> {
    let project = open_project(ctx, &args.dir)?;
    ctx.scope.check_project(&project)?;
    let mut options = ProjectRunOptions::new(format!(
        "calibration-studio {} ({})",
        env!("CARGO_PKG_VERSION"),
        ctx.shell
    ));
    options.notes = args.notes;
    options.config_patch = args.config_patch;
    options.redetect = args.redetect;
    options.report_html = true;
    let reports = job.clone();
    let control = run_control(job, move |report| reports.emit(&report));
    let run = project.calibrate(&options, Some(&crate::laser::VmLaserExtractor), &control)?;
    ctx.events.project_changed(project.dir());
    Ok(run)
}

/// Solver iterations forwarded at most this often: a small solve iterates
/// far faster than a page redraws. Every other report is forwarded.
const ITERATION_PERIOD: Duration = Duration::from_millis(50);

/// The run control of a job: its cancel token stops the run, and every
/// report goes to `emit` (solver iterations no faster than
/// [`ITERATION_PERIOD`]). The clock only throttles the display; nothing it
/// reads enters a result.
fn run_control<E: Serialize>(
    job: &JobCx<E>,
    emit: impl Fn(Progress) + Send + Sync + 'static,
) -> RunControl {
    let last_iteration: Mutex<Option<Instant>> = Mutex::new(None);
    RunControl::new()
        .with_cancel(job.token())
        .with_observer(move |report| {
            if let Progress::Iteration { iter, .. } = report {
                let now = Instant::now();
                let Ok(mut last) = last_iteration.lock() else {
                    return;
                };
                if iter > 1 && last.is_some_and(|t| now - t < ITERATION_PERIOD) {
                    return;
                }
                *last = Some(now);
            }
            emit(report);
        })
}

/// A run's listing entry: its state (with the record of a run this library
/// wrote) and its notes.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RunInfoArgs {
    pub run: RunRef,
}

pub fn run_info(ctx: &Ctx, args: RunInfoArgs) -> Result<RunInfo, StudioError> {
    Ok(resolve(ctx, &args.run)?.info)
}

/// A run's export, with the folder its image manifest's frames resolve
/// against.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RunExportArgs {
    pub run: RunRef,
}

/// A run's export.
#[derive(Debug, Serialize, JsonSchema)]
pub struct RunExport {
    /// The export JSON; the frontend narrows it by its `kind`.
    pub export: serde_json::Value,
    /// The folder the image manifest's frame paths resolve against
    /// (absolute, `/`-separated): the project's data folder for a run it
    /// calibrated, else the manifest's root as the run folder states it.
    /// `null` without an image manifest.
    pub image_root: Option<String>,
}

pub fn run_export(ctx: &Ctx, args: RunExportArgs) -> Result<RunExport, StudioError> {
    let run = ctx.runs.get(ctx, &args.run)?;
    Ok(RunExport {
        export: run.export.clone(),
        image_root: run.image_root.as_deref().map(slash),
    })
}

/// A run's quality report; `not found` (`io`) for an imported export whose
/// quality could not be assessed.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RunQualityArgs {
    pub run: RunRef,
}

pub fn run_quality(ctx: &Ctx, args: RunQualityArgs) -> Result<QualityReport, StudioError> {
    let run = resolve(ctx, &args.run)?;
    let path = ctx.scope.check(&run.info.quality_path())?;
    let text = std::fs::read_to_string(&path).map_err(|e| {
        StudioError::new(ErrorKind::Io, format!("{}: {e}", path.display())).at(slash(&path))
    })?;
    serde_json::from_str(&text).map_err(|e| {
        StudioError::new(ErrorKind::Parse, format!("{}: {e}", path.display())).at(slash(&path))
    })
}
