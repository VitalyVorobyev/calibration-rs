//! `vcal`: calibrate cameras from a folder of images on the command line.
//!
//! ```text
//! vcal init <folder>              draft <folder>/dataset.toml from the file names
//! vcal check <manifest>           check a manifest and its images, without detecting
//! vcal run <manifest> [-o <dir>]  detect, calibrate, write calibration.json and quality.json
//!                                 (--report adds report.html)
//! vcal sample list                list the sample datasets
//! vcal sample fetch <id>          download a sample dataset; prints its manifest's path
//! ```
//!
//! Every command is a thin layer over the library: `init` is
//! [`sniff_folder`], `check` is [`camera_image_sizes`], `run` is
//! [`calibrate_dataset`] followed by [`assess`], and `sample` is
//! [`samples`]. The same commands ship with the Python package as its `vcal`
//! console script.
//!
//! [`run`] is the whole program, for an embedding (the Python console script
//! calls it); the `vcal` binary only passes it the process arguments and a
//! [`RunControl`] whose token its Ctrl-C handler cancels.
//!
//! `run` shows its progress (the images detected, the step and the solver
//! iteration) on one line of standard error when standard error is a
//! terminal, and nothing otherwise.
//!
//! # Exit status
//!
//! | Status | Meaning |
//! |---:|---|
//! | 0 | done; for `run`, the quality verdict is pass or warn |
//! | 1 | an error, printed to standard error |
//! | 2 | the command line is wrong |
//! | 3 | `run` calibrated and wrote its files, but the quality verdict is fail |
//! | 130 | interrupted (Ctrl-C): the run stopped and wrote nothing |

use std::ffi::OsString;
use std::io::{IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use serde::Serialize;
use serde_json::Value;
use vision_calibration::analysis::{
    CheckStatus, QualityThresholds, ReportOptions, Verdict, assess, html_report,
};
use vision_calibration::common::CalibratedCameras;
use vision_calibration::control::{Progress, RunControl};
use vision_calibration::core::Uncertainty;
use vision_calibration::dataset::{DatasetSpec, Topology, samples, sniff_folder};
use vision_calibration::dataset_runner::{
    CalibrateDatasetError, CalibrationConfig, Detection, calibrate_dataset, camera_image_sizes,
};
use vision_calibration::detect::FsDetectionCache;

/// Writes a line to standard output. A failed write is ignored: a reader
/// that stops early (`vcal sample list | head -1`) is not an error.
macro_rules! say {
    ($($arg:tt)*) => {{
        let _ = writeln!(std::io::stdout(), $($arg)*);
    }};
}

/// Exit status of a `run` whose quality verdict is fail.
const VERDICT_FAIL: u8 = 3;
/// Exit status of a run stopped by its control's token: 128 + SIGINT, the
/// shell's status for a process interrupted by Ctrl-C.
pub const INTERRUPTED: u8 = 130;

/// The manifest file `init` writes into the folder.
const MANIFEST_FILE: &str = "dataset.toml";

/// The detection cache directory, next to the manifest.
const CACHE_DIR: &str = ".detections";

#[derive(Debug, Parser)]
#[command(
    name = "vcal",
    version,
    about = "Calibrate cameras from a folder of images",
    long_about = "Calibrate cameras from a folder of images.\n\n\
                  Start with `vcal init <folder>`, describe the board in the manifest it \
                  drafts, then `vcal run <folder>/dataset.toml`. Or try a sample: \
                  `vcal run \"$(vcal sample fetch opencv_left)\"`."
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Draft a manifest from a folder's file names. The fields the names do
    /// not tell, such as the board, are listed under `_unresolved`: fill
    /// them in and delete the list before `vcal run`.
    Init {
        /// The folder of images.
        folder: PathBuf,
        /// Where to write the manifest; `-` prints it [default:
        /// `<folder>/dataset.toml`].
        #[arg(short, long)]
        output: Option<String>,
        /// Replace an existing manifest.
        #[arg(long)]
        force: bool,
    },
    /// Check a manifest and its images without detecting anything: the
    /// manifest is complete and valid, every camera's images are found and
    /// share one size.
    Check {
        /// The manifest (TOML or JSON).
        manifest: PathBuf,
    },
    /// Detect the target, calibrate, and write the export
    /// (`calibration.json`), the quality report (`quality.json`) and, with
    /// `--report`, a page to read (`report.html`).
    Run {
        /// The manifest (TOML or JSON).
        manifest: PathBuf,
        /// Directory for the output files [default: the manifest's
        /// directory].
        #[arg(short, long)]
        out: Option<PathBuf>,
        /// Changes to the workflow's default configuration, a TOML or JSON
        /// file: what it names replaces the default, what it leaves out keeps
        /// it. For a fisheye camera:
        /// `camera.model = { projection = "equidistant", distortion =
        /// "kannala_brandt4", sensor = "identity" }`.
        #[arg(short, long)]
        config: Option<PathBuf>,
        /// Detect every image again instead of reusing the detection cache
        /// (`.detections` next to the manifest).
        #[arg(long)]
        redetect: bool,
        /// Also write `report.html`: the verdict, each camera's parameters
        /// with σ, and the residual and projection-σ figures, in one page
        /// that opens offline.
        #[arg(long)]
        report: bool,
    },
    /// List or download the sample datasets.
    Sample {
        #[command(subcommand)]
        command: SampleCommand,
    },
}

#[derive(Debug, Subcommand)]
enum SampleCommand {
    /// List the sample datasets.
    List,
    /// Download a sample dataset, verify it and print its manifest's path.
    Fetch {
        /// The sample's id, from `vcal sample list`.
        id: String,
        /// The cache directory [default: `VISION_CALIBRATION_DATA`, else the
        /// platform's cache directory].
        #[arg(long)]
        cache: Option<PathBuf>,
    },
}

/// Runs `vcal` on `args` (the program name first, as in
/// [`std::env::args_os`]) and returns its exit status.
///
/// Output goes to standard output, errors to standard error. `vcal run`
/// reports to `control` and polls its token: once it is cancelled the run
/// stops, prints `interrupted`, writes no files and returns
/// [`INTERRUPTED`].
pub fn run<I, T>(args: I, control: &RunControl) -> u8
where
    I: IntoIterator<Item = T>,
    T: Into<OsString> + Clone,
{
    let cli = match Cli::try_parse_from(args) {
        Ok(cli) => cli,
        Err(err) => {
            // `--help` and `--version` print to standard output and succeed.
            let _ = err.print();
            return u8::try_from(err.exit_code()).unwrap_or(2);
        }
    };
    let outcome = match cli.command {
        Command::Init {
            folder,
            output,
            force,
        } => init(&folder, output.as_deref(), force).map(|()| 0),
        Command::Check { manifest } => check(&manifest).map(|()| 0),
        Command::Run {
            manifest,
            out,
            config,
            redetect,
            report,
        } => calibrate(
            &manifest,
            out.as_deref(),
            config.as_deref(),
            redetect,
            report,
            control,
        ),
        Command::Sample { command } => sample(command).map(|()| 0),
    };
    outcome.unwrap_or_else(|err| {
        if matches!(
            err.downcast_ref::<CalibrateDatasetError>(),
            Some(CalibrateDatasetError::Cancelled)
        ) {
            eprintln!("interrupted");
            return INTERRUPTED;
        }
        eprintln!("error: {}", describe(&err));
        1
    })
}

/// One line of standard error that shows the latest progress report,
/// redrawn at most every [`StatusLine::PERIOD`] (a new step always).
#[derive(Clone)]
struct StatusLine(Arc<Mutex<StatusState>>);

#[derive(Default)]
struct StatusState {
    /// The step under way, shown with its iterations.
    step: Option<&'static str>,
    /// When the line was last drawn; display only.
    drawn: Option<Instant>,
    /// Whether the line holds text to clear.
    dirty: bool,
}

impl StatusLine {
    const PERIOD: Duration = Duration::from_millis(50);

    fn new() -> Self {
        Self(Arc::new(Mutex::new(StatusState::default())))
    }

    fn show(&self, progress: Progress) {
        let Ok(mut state) = self.0.lock() else {
            return;
        };
        let text = match progress {
            Progress::Detection { done, total, .. } => format!("detecting {done}/{total}"),
            Progress::LaserExtraction { done, total, .. } => {
                format!("extracting laser lines {done}/{total}")
            }
            Progress::Step { name, .. } => {
                state.step = Some(name);
                state.drawn = None;
                name.to_string()
            }
            Progress::Iteration {
                iter,
                max_iters,
                cost,
                ..
            } => format!(
                "{}: iteration {iter}/{max_iters}, cost {cost:.4e}",
                state.step.unwrap_or("solve")
            ),
            _ => return,
        };
        let now = Instant::now();
        if state.drawn.is_some_and(|t| now - t < Self::PERIOD) {
            return;
        }
        state.drawn = Some(now);
        state.dirty = true;
        // A failed write to a terminal has nowhere better to go.
        let _ = write!(std::io::stderr(), "\r\x1b[2K{text}");
    }

    fn clear(&self) {
        if let Ok(mut state) = self.0.lock()
            && state.dirty
        {
            state.dirty = false;
            let _ = write!(std::io::stderr(), "\r\x1b[2K");
        }
    }
}

/// The error and its causes on one line. A cause whose text the message
/// already holds is left out: most errors quote their source.
fn describe(err: &anyhow::Error) -> String {
    let mut text = String::new();
    for cause in err.chain() {
        let cause = cause.to_string();
        if !text.contains(&cause) {
            if !text.is_empty() {
                text.push_str(": ");
            }
            text.push_str(&cause);
        }
    }
    text
}

fn init(folder: &Path, output: Option<&str>, force: bool) -> Result<()> {
    let spec = sniff_folder(folder)
        .with_context(|| format!("drafting a manifest for {}", folder.display()))?;
    let text = format!(
        "# Drafted by `vcal init` from the file names. Review every field, fill\n\
         # in what `_unresolved` lists and delete it: `vcal run` refuses a\n\
         # manifest that still has it.\n\n{}",
        toml::to_string_pretty(&spec).context("writing the manifest as TOML")?
    );
    if output == Some("-") {
        let _ = write!(std::io::stdout(), "{text}");
        print_unresolved(&spec.unresolved, &mut std::io::stderr());
        return Ok(());
    }
    let path = output.map_or_else(|| folder.join(MANIFEST_FILE), PathBuf::from);
    if path.exists() && !force {
        bail!("{} exists; pass --force to replace it", path.display());
    }
    std::fs::write(&path, text).with_context(|| format!("writing {}", path.display()))?;
    say!("wrote {}", path.display());
    print_unresolved(&spec.unresolved, &mut std::io::stdout());
    Ok(())
}

fn print_unresolved(unresolved: &[String], out: &mut dyn Write) {
    if unresolved.is_empty() {
        return;
    }
    // A write error on a terminal or pipe has nowhere better to go.
    let _ = writeln!(out, "fill in, then delete `_unresolved`:");
    for field in unresolved {
        let _ = writeln!(out, "  {field}");
    }
}

/// The manifest at `path` and the directory its relative paths resolve
/// against.
fn load(path: &Path) -> Result<(DatasetSpec, PathBuf)> {
    // The error names the manifest.
    let spec = DatasetSpec::from_path(path)?;
    let base = path.parent().map(Path::to_path_buf).unwrap_or_default();
    Ok((spec, base))
}

fn check(manifest: &Path) -> Result<()> {
    let (spec, base) = load(manifest)?;
    let sizes = camera_image_sizes(&spec, &base)?;
    say!("{}: {}", manifest.display(), name(&spec.topology));
    for (camera, [width, height]) in spec.cameras.iter().zip(sizes) {
        say!("  camera {}: {width} × {height} px", camera.id);
    }
    say!("ok");
    Ok(())
}

fn calibrate(
    manifest: &Path,
    out: Option<&Path>,
    config: Option<&Path>,
    redetect: bool,
    html: bool,
    control: &RunControl,
) -> Result<u8> {
    let (spec, base) = load(manifest)?;
    let config = config
        .map(|path| configuration(spec.topology, path))
        .transpose()?;
    let cache = FsDetectionCache::new(base.join(CACHE_DIR));
    let detection = if redetect {
        Detection::Refresh(&cache)
    } else {
        Detection::Cached(&cache)
    };
    // The CLI has no laser-line extractor; a laser topology fails before any
    // detection with an error that says so.
    let status = std::io::stderr().is_terminal().then(StatusLine::new);
    let control = match &status {
        Some(line) => {
            let line = line.clone();
            control
                .clone()
                .with_observer(move |progress| line.show(progress))
        }
        None => control.clone(),
    };
    let run = calibrate_dataset(&spec, &base, config, detection, None, &control);
    if let Some(line) = &status {
        line.clear();
    }
    let run = run?;
    // Interrupted after the last check: still stop before writing anything.
    if control.is_cancelled() {
        return Err(CalibrateDatasetError::Cancelled.into());
    }

    say!(
        "{}: {} of {} views used",
        name(&spec.topology),
        run.usable_views,
        run.total_views
    );
    for rejected in &run.rejected {
        say!(
            "  left out {}: {}",
            rejected.path.display(),
            rejected.reason
        );
    }

    let export = &run.export;
    let cameras = export.camera_params();
    let uncertainty = export.parameter_uncertainty();
    for (i, (camera, [width, height])) in cameras.iter().zip(&run.image_sizes).enumerate() {
        let id = spec.cameras.get(i).map_or("", |c| c.id.as_str());
        say!("camera {i} ({id}), {width} × {height} px:");
        let k = &camera.intrinsics;
        for (label, value) in [("fx", k.fx), ("fy", k.fy), ("cx", k.cx), ("cy", k.cy)] {
            // A rig's covariance labels carry the camera's index.
            let key = if cameras.len() > 1 {
                format!("cam{i}.{label}")
            } else {
                label.to_string()
            };
            match uncertainty {
                Uncertainty::Available(cov) => match cov.std_dev(&key) {
                    Some(sigma) => say!("  {label} = {value:.3} ± {sigma:.3} px"),
                    None => say!("  {label} = {value:.3} px (held fixed)"),
                },
                Uncertainty::Unavailable { .. } => say!("  {label} = {value:.3} px"),
            }
        }
    }
    if let Uncertainty::Unavailable { reason } = uncertainty {
        say!("no parameter covariance: {}", name(reason));
    }

    let report = assess(export, &run.image_sizes, &QualityThresholds::default())?;
    say!("verdict: {}", name(&report.verdict));
    for check in report
        .checks
        .iter()
        .filter(|c| c.status != CheckStatus::Pass)
    {
        let camera = check
            .camera
            .map_or(String::new(), |i| format!(" camera {i}"));
        say!(
            "  {} {}{camera}: {}",
            name(&check.status),
            name(&check.kind),
            check.message
        );
    }

    let out = out.map_or(base.clone(), Path::to_path_buf);
    std::fs::create_dir_all(&out).with_context(|| format!("creating {}", out.display()))?;
    let export_path = out.join("calibration.json");
    let report_path = out.join("quality.json");
    write_json(&export_path, export)?;
    write_json(&report_path, &report)?;
    say!(
        "wrote {} and {}",
        export_path.display(),
        report_path.display()
    );
    if html {
        let mut options = ReportOptions::default();
        let what = spec
            .description
            .clone()
            .unwrap_or_else(|| manifest.display().to_string());
        options.title = format!("{}: {what}", name(&spec.topology));
        options.camera_names = spec.cameras.iter().map(|c| c.id.clone()).collect();
        let page = html_report(export, &run.image_sizes, &options)?;
        let page_path = out.join("report.html");
        std::fs::write(&page_path, page)
            .with_context(|| format!("writing {}", page_path.display()))?;
        say!("wrote {}", page_path.display());
    }
    Ok(if report.verdict == Verdict::Fail {
        VERDICT_FAIL
    } else {
        0
    })
}

/// The default configuration of `topology` with the changes the file at
/// `path` makes.
fn configuration(topology: Topology, path: &Path) -> Result<CalibrationConfig> {
    let context = || format!("reading the configuration {}", path.display());
    let text = std::fs::read_to_string(path).with_context(context)?;
    let changes: Value = if path.extension().is_some_and(|e| e == "toml") {
        toml::from_str(&text).with_context(context)?
    } else {
        serde_json::from_str(&text).with_context(context)?
    };
    // The tagged form: {"topology": ..., "config": {...}}.
    let mut config = serde_json::to_value(CalibrationConfig::default_for(topology))?;
    merge(&mut config["config"], &changes);
    serde_json::from_value(config).with_context(context)
}

/// Merges `changes` into `base`: objects merge key by key, any other value
/// replaces the base value.
fn merge(base: &mut Value, changes: &Value) {
    match (base, changes) {
        (Value::Object(base), Value::Object(changes)) => {
            for (key, value) in changes {
                match base.get_mut(key) {
                    Some(slot) => merge(slot, value),
                    None => {
                        base.insert(key.clone(), value.clone());
                    }
                }
            }
        }
        (slot, value) => *slot = value.clone(),
    }
}

fn write_json(path: &Path, value: &impl Serialize) -> Result<()> {
    let text = serde_json::to_string_pretty(value)?;
    std::fs::write(path, text + "\n").with_context(|| format!("writing {}", path.display()))
}

fn sample(command: SampleCommand) -> Result<()> {
    match command {
        SampleCommand::List => {
            for sample in samples::catalogue() {
                say!("{:<24} {}", sample.id, sample.description);
                say!(
                    "{:<24} licence {}, {}",
                    "",
                    sample.licence,
                    sample.source_url
                );
            }
        }
        SampleCommand::Fetch { id, cache } => {
            let cache = match cache {
                Some(cache) => cache,
                None => samples::default_cache_dir()?,
            };
            let manifest = samples::fetch(&id, &cache)
                .with_context(|| format!("fetching the sample {id:?}"))?;
            say!("{}", manifest.display());
        }
    }
    Ok(())
}

/// The name a value serializes to (`planar_intrinsics`, `warn`): the
/// spelling of the JSON files and the manifest.
fn name(value: &impl Serialize) -> String {
    match serde_json::to_value(value) {
        Ok(serde_json::Value::String(name)) => name,
        Ok(other) => other.to_string(),
        Err(_) => "?".to_string(),
    }
}
