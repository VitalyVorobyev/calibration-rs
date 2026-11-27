//! `vcal`: calibrate cameras from a folder of images on the command line.
//!
//! ```text
//! vcal init <folder>              draft <folder>/dataset.toml from the file names
//! vcal check <manifest>           check a manifest and its images, without detecting
//! vcal run <manifest> [-o <dir>]  detect, calibrate, write calibration.json and quality.json
//!                                 (--report adds report.html; --seed or --device-spec
//!                                 start it from a previous calibration or a datasheet)
//! vcal run <project>              the same into the project's next runs/run-NNN/
//! vcal project init <folder>      make a folder a calibration project
//! vcal project runs <project>     list a project's runs
//! vcal project import <project> <export>   add an export calibrated elsewhere as a run
//! vcal project note <project> <run>        label, annotate or star a run
//! vcal project scan <root>...     the projects under folders, grouped by device
//! vcal sample list                list the sample datasets
//! vcal sample fetch <id>          download a sample dataset; prints its manifest's path
//! ```
//!
//! Every command is a thin layer over the library: `init` is
//! [`sniff_folder`], `check` is [`camera_image_sizes`], `run` is
//! [`calibrate_dataset`] followed by [`assess`], `project` is
//! [`Project`] and [`scan_library`], and `sample` is [`samples`]. The same
//! commands ship with the Python package as its `vcal` console script.
//!
//! `check` and `run` take a project when their path is a project folder (one
//! holding `calibration-project.toml` or `calibration-project.json`) or a
//! project file, and read it as a manifest otherwise: the project file
//! decides, nothing is guessed.
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
//! | 2 | the command line is wrong (`-o`, `--seed` or `--device-spec` with a project, `--label` with a manifest included) |
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
use vision_calibration::analysis::{
    CheckStatus, QualityReport, QualityThresholds, ReportOptions, Verdict, assess, html_report,
};
use vision_calibration::common::CalibratedCameras;
use vision_calibration::control::{Progress, RunControl};
use vision_calibration::core::Uncertainty;
use vision_calibration::dataset::{DatasetSpec, ProblemKind, samples, sniff_folder};
use vision_calibration::dataset_runner::{
    CalibrateDatasetError, CalibrationConfig, ConfigPatch, Detection, calibrate_dataset,
    calibrate_dataset_with_seed, camera_image_sizes,
};
use vision_calibration::detect::FsDetectionCache;
use vision_calibration::device_seed::{DeviceSeed, DeviceSpec};
use vision_calibration::project::{
    DatasetSource, DeviceIdentity, LibraryScan, LibraryScanOptions, PROJECT_FILE,
    PROJECT_FILE_JSON, Project, ProjectError, ProjectRunOptions, ProjectSpec, RunInfo, RunNotes,
    RunSource, RunState, scan_library,
};

/// Writes a line to standard output. A failed write is ignored: a reader
/// that stops early (`vcal sample list | head -1`) is not an error.
macro_rules! say {
    ($($arg:tt)*) => {{
        let _ = writeln!(std::io::stdout(), $($arg)*);
    }};
}

/// Exit status of a `run` whose quality verdict is fail.
const VERDICT_FAIL: u8 = 3;
/// Exit status of a wrong command line, as clap's.
const USAGE: u8 = 2;
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
    /// share one size. For a project, also its configuration changes, its
    /// exclusions and its device.
    Check {
        /// The manifest (TOML or JSON), or a project (its folder or its
        /// project file).
        #[arg(value_name = "MANIFEST|PROJECT")]
        manifest: PathBuf,
    },
    /// Detect the target, calibrate, and write the export
    /// (`calibration.json`), the quality report (`quality.json`) and, with
    /// `--report`, a page to read (`report.html`). A project's run goes to
    /// its next `runs/run-NNN/` instead, as `export.json`, `quality.json`
    /// and `run.json`, the run's record.
    Run {
        /// The manifest (TOML or JSON), or a project (its folder or its
        /// project file).
        #[arg(value_name = "MANIFEST|PROJECT")]
        manifest: PathBuf,
        /// Directory for the output files [default: the manifest's
        /// directory]. Not for a project, whose runs are in its folder.
        #[arg(short, long)]
        out: Option<PathBuf>,
        /// Changes to the workflow's default configuration, a TOML or JSON
        /// file: what it names replaces the default, what it leaves out keeps
        /// it. For a fisheye camera:
        /// `camera.model = { projection = "equidistant", distortion =
        /// "kannala_brandt4", sensor = "identity" }`. For a project, the
        /// changes apply over the project's, for this run only, and its
        /// record keeps them.
        #[arg(short, long)]
        config: Option<PathBuf>,
        /// Detect every image again instead of reusing the detection cache
        /// (`.detections` next to the manifest, or in the project).
        #[arg(long)]
        redetect: bool,
        /// Also write `report.html`: the verdict, each camera's parameters
        /// with σ, and the residual and projection-σ figures, in one page
        /// that opens offline.
        #[arg(long)]
        report: bool,
        /// A label for the run, kept with it in the project. Only for a
        /// project.
        #[arg(long)]
        label: Option<String>,
        /// Start the calibration from a seed: a `DeviceSeed` file (JSON, or
        /// TOML by its extension), the lenses, mounts, hand-eye and laser
        /// planes of a previous calibration of the device or of one of its
        /// type. Its cameras must be the manifest's, for images of the same
        /// size, and a calibrated lens of the configured camera model. The
        /// seed is a starting point: the calibration still estimates
        /// everything. Only for a manifest.
        #[arg(long, value_name = "FILE", conflicts_with = "device_spec")]
        seed: Option<PathBuf>,
        /// Start the calibration from a device spec (`spec.json`): each
        /// camera's datasheet focal length and mount tilt, and the drawing's
        /// camera and hand-eye mounts. Only for a manifest.
        #[arg(long, value_name = "FILE")]
        device_spec: Option<PathBuf>,
    },
    /// Make and read calibration projects: a dataset with its configuration
    /// changes, exclusions and every run.
    Project {
        #[command(subcommand)]
        command: ProjectCommand,
    },
    /// List or download the sample datasets.
    Sample {
        #[command(subcommand)]
        command: SampleCommand,
    },
}

#[derive(Debug, Subcommand)]
enum ProjectCommand {
    /// Make a folder a calibration project (`calibration-project.toml`).
    ///
    /// With `--manifest`, the project references that manifest file (or,
    /// with `--embed`, holds a copy of it). Without it, a manifest is
    /// drafted from the file names in `--data` and written into the project
    /// file; fill in what it lists under `_unresolved`.
    Init {
        /// The project folder, made when missing.
        folder: PathBuf,
        /// The folder of images to draft a manifest from [default: the
        /// project folder]. A folder that already holds `dataset.toml` or
        /// `dataset.json` is refused: pass that file with `--manifest`.
        #[arg(long, conflicts_with = "manifest")]
        data: Option<PathBuf>,
        /// A manifest file for the project to reference.
        #[arg(long)]
        manifest: Option<PathBuf>,
        /// Copy the manifest into the project file instead of referencing
        /// it; its paths keep resolving against its folder.
        #[arg(long, requires = "manifest")]
        embed: bool,
        /// The project's name [default: the folder's name].
        #[arg(long)]
        name: Option<String>,
        /// The calibrated device's own identifier (a serial number): the key
        /// its runs are grouped by across projects.
        #[arg(long)]
        device_id: Option<String>,
        /// The device type or product line.
        #[arg(long, requires = "device_id")]
        device_product: Option<String>,
        /// A name for the device, for display.
        #[arg(long, requires = "device_id")]
        device_label: Option<String>,
    },
    /// List a project's runs: id, time (UTC), verdict, views, each camera's
    /// RMS reprojection error, star and label.
    Runs {
        /// The project (its folder or its project file).
        project: PathBuf,
        /// Print the runs as JSON (`RunInfo` records).
        #[arg(long)]
        json: bool,
    },
    /// Add an export calibrated elsewhere to a project as a run.
    Import {
        /// The project (its folder or its project file).
        project: PathBuf,
        /// The export: a JSON file, or a folder `vcal run -o` wrote.
        export: PathBuf,
        /// A label for the run.
        #[arg(long)]
        label: Option<String>,
    },
    /// Label, annotate or star a run. An empty text clears a field.
    Note {
        /// The project (its folder or its project file).
        project: PathBuf,
        /// The run's id (`run-003`).
        run: String,
        /// The run's label.
        #[arg(long)]
        label: Option<String>,
        /// Free text.
        #[arg(long)]
        notes: Option<String>,
        /// Mark the run as one to keep.
        #[arg(long, conflicts_with = "unstar")]
        star: bool,
        /// Remove the mark.
        #[arg(long)]
        unstar: bool,
    },
    /// List the projects under folders, each device's runs together.
    /// Folders and projects that cannot be read are listed on standard
    /// error.
    Scan {
        /// The folders to search.
        #[arg(required = true)]
        roots: Vec<PathBuf>,
        /// How many folders deep to search.
        #[arg(long, default_value_t = LibraryScanOptions::default().max_depth)]
        max_depth: usize,
        /// Print the scan as JSON: `projects`, `failures` and `devices`.
        #[arg(long)]
        json: bool,
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
        Command::Check { manifest } if is_project(&manifest) => {
            check_project(&manifest).map(|()| 0)
        }
        Command::Check { manifest } => check(&manifest).map(|()| 0),
        Command::Run {
            manifest,
            out,
            config,
            redetect,
            report,
            label,
            seed,
            device_spec,
        } => {
            if is_project(&manifest) {
                if out.is_some() {
                    Err(Usage(
                        "-o is for a manifest's run; a project's runs are written to its runs folder"
                            .into(),
                    )
                    .into())
                } else if seed.is_some() || device_spec.is_some() {
                    Err(Usage("--seed and --device-spec are for a manifest's run".into()).into())
                } else {
                    calibrate_project(
                        &manifest,
                        config.as_deref(),
                        redetect,
                        report,
                        label,
                        control,
                    )
                }
            } else if label.is_some() {
                Err(
                    Usage("--label is for a project's run; a manifest's run keeps no label".into())
                        .into(),
                )
            } else {
                let seed = match (seed, device_spec) {
                    (Some(path), _) => seed_file(&path).map(Some),
                    (None, Some(path)) => seed_from_spec(&path).map(Some),
                    (None, None) => Ok(None),
                };
                seed.and_then(|seed| {
                    calibrate(
                        &manifest,
                        out.as_deref(),
                        config.as_deref(),
                        seed.as_ref(),
                        redetect,
                        report,
                        control,
                    )
                })
            }
        }
        Command::Project { command } => project(command).map(|()| 0),
        Command::Sample { command } => sample(command).map(|()| 0),
    };
    outcome.unwrap_or_else(|err| {
        if matches!(
            err.downcast_ref::<CalibrateDatasetError>(),
            Some(CalibrateDatasetError::Cancelled)
        ) || matches!(
            err.downcast_ref::<ProjectError>(),
            Some(ProjectError::Cancelled)
        ) {
            eprintln!("interrupted");
            return INTERRUPTED;
        }
        if let Some(Usage(message)) = err.downcast_ref::<Usage>() {
            eprintln!("error: {message}");
            return USAGE;
        }
        eprintln!("error: {}", describe(&err));
        1
    })
}

/// A command line clap accepts but that asks for something the command
/// does not do: exit status [`USAGE`].
#[derive(Debug)]
struct Usage(String);

impl std::fmt::Display for Usage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Usage {}

/// Whether `path` names a project: a folder holding a project file, or a
/// project file. Anything else is read as a manifest.
fn is_project(path: &Path) -> bool {
    if path.is_dir() {
        path.join(PROJECT_FILE).is_file() || path.join(PROJECT_FILE_JSON).is_file()
    } else {
        path.file_name()
            .is_some_and(|n| n == PROJECT_FILE || n == PROJECT_FILE_JSON)
    }
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

/// The directory a manifest's relative paths resolve against (`.` for a
/// bare file name).
fn manifest_base_dir(path: &Path) -> PathBuf {
    path.parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
        .to_path_buf()
}

/// The manifest at `path` and the directory its relative paths resolve
/// against.
fn load(path: &Path) -> Result<(DatasetSpec, PathBuf)> {
    if path.is_dir() {
        bail!(
            "{} is a folder without a calibration project: pass its manifest file, or make \
             it a project with `vcal project init`",
            path.display()
        );
    }
    // The error names the manifest.
    let spec = DatasetSpec::from_path(path)?;
    let base = manifest_base_dir(path);
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

/// A seed and where it came from, for the run's output.
struct Seed {
    seed: DeviceSeed,
    source: String,
}

/// The [`DeviceSeed`] in the file at `path`, TOML or JSON by its extension.
fn seed_file(path: &Path) -> Result<Seed> {
    let context = || format!("reading the seed {}", path.display());
    let text = std::fs::read_to_string(path).with_context(context)?;
    let seed: DeviceSeed = if path
        .extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("toml"))
    {
        toml::from_str(&text).with_context(context)?
    } else {
        serde_json::from_str(&text).with_context(context)?
    };
    seed.validate().with_context(context)?;
    Ok(Seed {
        seed,
        source: format!("the seed {}", path.display()),
    })
}

/// The datasheet seed of the device spec at `path`.
fn seed_from_spec(path: &Path) -> Result<Seed> {
    let context = || format!("reading the device spec {}", path.display());
    let spec = DeviceSpec::from_path(path).with_context(context)?;
    let seed = DeviceSeed::from_spec(&spec).with_context(context)?;
    Ok(Seed {
        seed,
        source: format!("the device spec {}", path.display()),
    })
}

fn calibrate(
    manifest: &Path,
    out: Option<&Path>,
    config: Option<&Path>,
    seed: Option<&Seed>,
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
    let run = match seed {
        Some(seed) => {
            calibrate_dataset_with_seed(&spec, &base, config, &seed.seed, detection, None, &control)
        }
        None => calibrate_dataset(&spec, &base, config, detection, None, &control),
    };
    if let Some(line) = &status {
        line.clear();
    }
    let mut run = run?;
    // Interrupted after the last check: still stop before writing anything.
    if control.is_cancelled() {
        return Err(CalibrateDatasetError::Cancelled.into());
    }
    let out = out.map_or(base.clone(), Path::to_path_buf);
    // The export's image root is the manifest's folder; restate it from the
    // folder the export is written to: relative when that lies inside the
    // manifest's folder, absolute otherwise.
    if let Some(images) = run.export.diagnostics_mut().image_manifest.as_mut() {
        *images = images.rebased(&base, &out, &base);
    }

    say!(
        "{}: {} of {} views used",
        name(&spec.topology),
        run.usable_views,
        run.total_views
    );
    if let Some(seed) = seed {
        match &seed.seed.description {
            Some(description) => say!("started from {} ({description})", seed.source),
            None => say!("started from {}", seed.source),
        }
    }
    for rejected in &run.rejected {
        say!(
            "  left out {}: {}",
            rejected.path.display(),
            rejected.reason
        );
    }

    let export = &run.export;
    let ids: Vec<String> = spec.cameras.iter().map(|c| c.id.clone()).collect();
    print_cameras(export, &run.image_sizes, &ids);

    let report = assess(export, &run.image_sizes, &QualityThresholds::default())?;
    print_verdict(&report);

    std::fs::create_dir_all(&out).with_context(|| format!("creating {}", out.display()))?;
    // Every file is built before any is written, and written all or none: a
    // run that cannot write its report leaves no half-written result.
    let export_path = out.join("calibration.json");
    let report_path = out.join("quality.json");
    let mut files = vec![
        (export_path.clone(), json_text(export)?),
        (report_path.clone(), json_text(&report)?),
    ];
    if html {
        let mut options = ReportOptions::default();
        let what = spec
            .description
            .clone()
            .unwrap_or_else(|| manifest.display().to_string());
        options.title = format!("{}: {what}", name(&spec.topology));
        options.camera_names = spec.cameras.iter().map(|c| c.id.clone()).collect();
        let page = html_report(export, &run.image_sizes, &options)?;
        files.insert(0, (out.join("report.html"), page));
    }
    write_all(&files)?;
    say!(
        "wrote {} and {}",
        export_path.display(),
        report_path.display()
    );
    if html {
        say!("wrote {}", out.join("report.html").display());
    }
    Ok(if report.verdict == Verdict::Fail {
        VERDICT_FAIL
    } else {
        0
    })
}

/// Each camera's size and intrinsics, with σ when the export has the
/// covariance; `ids` names the cameras.
fn print_cameras(export: &dyn CalibratedCameras, image_sizes: &[[u32; 2]], ids: &[String]) {
    let cameras = export.camera_params();
    let uncertainty = export.parameter_uncertainty();
    for (i, (camera, [width, height])) in cameras.iter().zip(image_sizes).enumerate() {
        let id = ids.get(i).map_or("", String::as_str);
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
        say!(
            "no parameter covariance: {}",
            format_uncertainty_unavailable(reason)
        );
    }
}

/// The verdict and every check that did not pass.
fn print_verdict(report: &QualityReport) {
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
}

/// The project at `path` and its check: what `check` does for a manifest,
/// plus the configuration changes, the exclusions and the device.
fn check_project(path: &Path) -> Result<()> {
    let project = Project::open(path)?;
    project.check()?;
    let dataset = project.dataset()?;
    let sizes = camera_image_sizes(&dataset, &project.data_root()?)?;
    say!("{}: {}", project.file().display(), name(&dataset.topology));
    for (camera, [width, height]) in dataset.cameras.iter().zip(sizes) {
        say!("  camera {}: {width} × {height} px", camera.id);
    }
    say!("ok");
    Ok(())
}

/// `vcal run <project>`: the project's next run, printed as a manifest's
/// run is.
fn calibrate_project(
    path: &Path,
    config: Option<&Path>,
    redetect: bool,
    html: bool,
    label: Option<String>,
    control: &RunControl,
) -> Result<u8> {
    let project = Project::open(path)?;
    let mut options = ProjectRunOptions::new(format!("vcal {}", env!("CARGO_PKG_VERSION")));
    if let Some(path) = config {
        options.config_patch = config_patch(path)?;
    }
    options.redetect = redetect;
    options.report_html = html;
    options.notes.label = label.filter(|l| !l.is_empty());
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
    // The CLI has no laser-line extractor; a laser topology fails before any
    // detection with an error that says so.
    let run = project.calibrate(&options, None, &control);
    if let Some(line) = &status {
        line.clear();
    }
    let run = run?;
    let Some(record) = run.record() else {
        bail!("run {} has no record", run.id);
    };
    if let RunSource::Calibrated(calibrated) = &record.source {
        say!(
            "{}: {} of {} views used",
            name(&record.summary.kind),
            calibrated.usable_views,
            calibrated.total_views
        );
        for left_out in &calibrated.left_out {
            say!(
                "  left out {}: {}",
                left_out.image.display(),
                left_out.reason
            );
        }
    }
    let export_text = std::fs::read_to_string(run.export_path())
        .with_context(|| format!("reading {}", run.export_path().display()))?;
    let export = vision_calibration::common::calibrated_cameras(&export_text)?;
    let dataset = project.dataset()?;
    let image_sizes = camera_image_sizes(&dataset, &project.data_root()?)?;
    let ids: Vec<String> = dataset.cameras.iter().map(|c| c.id.clone()).collect();
    print_cameras(export.as_ref(), &image_sizes, &ids);
    let report: QualityReport = serde_json::from_slice(
        &std::fs::read(run.quality_path())
            .with_context(|| format!("reading {}", run.quality_path().display()))?,
    )?;
    print_verdict(&report);
    say!("wrote {} ({})", run.dir.display(), run.id);
    Ok(if report.verdict == Verdict::Fail {
        VERDICT_FAIL
    } else {
        0
    })
}

fn project(command: ProjectCommand) -> Result<()> {
    match command {
        ProjectCommand::Init {
            folder,
            data,
            manifest,
            embed,
            name,
            device_id,
            device_product,
            device_label,
        } => {
            let device = device_id.map(|id| {
                let mut device = DeviceIdentity::new(id);
                device.product = device_product;
                device.label = device_label;
                device
            });
            project_init(
                &folder,
                data.as_deref(),
                manifest.as_deref(),
                embed,
                name,
                device,
            )
        }
        ProjectCommand::Runs { project, json } => project_runs(&project, json),
        ProjectCommand::Import {
            project,
            export,
            label,
        } => {
            let project = Project::open(&project)?;
            let mut notes = RunNotes::new();
            notes.label = label.filter(|l| !l.is_empty());
            let run = project.import_export(&export, notes)?;
            say!("imported {} as {}", export.display(), run.id);
            Ok(())
        }
        ProjectCommand::Note {
            project,
            run,
            label,
            notes,
            star,
            unstar,
        } => {
            let project = Project::open(&project)?;
            let mut current = project.run(&run)?.notes;
            let text = |value: String| (!value.is_empty()).then_some(value);
            if let Some(label) = label {
                current.label = text(label);
            }
            if let Some(notes) = notes {
                current.notes = text(notes);
            }
            if star {
                current.starred = true;
            }
            if unstar {
                current.starred = false;
            }
            project.set_run_notes(&run, &current)?;
            say!("{run}: {}", notes_line(&current));
            Ok(())
        }
        ProjectCommand::Scan {
            roots,
            max_depth,
            json,
        } => {
            let mut options = LibraryScanOptions::new();
            options.max_depth = max_depth;
            project_scan(&scan_library(&roots, &options), json)
        }
    }
}

/// `vcal project init`.
fn project_init(
    folder: &Path,
    data: Option<&Path>,
    manifest: Option<&Path>,
    embed: bool,
    name: Option<String>,
    device: Option<DeviceIdentity>,
) -> Result<()> {
    let folder_abs = absolute(folder)?;
    let mut unresolved = Vec::new();
    let dataset = match manifest {
        Some(manifest) => {
            let manifest_abs = absolute(manifest)?;
            if embed {
                let spec = DatasetSpec::from_path(&manifest_abs)?;
                let root = manifest_abs
                    .parent()
                    .map_or_else(|| PathBuf::from("."), Path::to_path_buf);
                DatasetSource::inline(within(&folder_abs, &root), spec)
            } else {
                if !manifest_abs.is_file() {
                    bail!("{} is not a manifest file", manifest.display());
                }
                DatasetSource::file(within(&folder_abs, &manifest_abs))
            }
        }
        None => {
            let data = data.unwrap_or(folder);
            for existing in [MANIFEST_FILE, "dataset.json"] {
                if data.join(existing).is_file() {
                    bail!(
                        "{} holds {existing}: pass it with --manifest {} (add --embed to copy it \
                         into the project)",
                        data.display(),
                        data.join(existing).display()
                    );
                }
            }
            let spec = sniff_folder(data)
                .with_context(|| format!("drafting a manifest for {}", data.display()))?;
            unresolved.clone_from(&spec.unresolved);
            DatasetSource::inline(within(&folder_abs, &absolute(data)?), spec)
        }
    };
    let mut spec = ProjectSpec::new(dataset);
    spec.name = name;
    spec.device = device;
    let project = Project::create(folder, spec)?;
    say!("wrote {}", project.file().display());
    print_unresolved(&unresolved, &mut std::io::stdout());
    Ok(())
}

/// `vcal project runs`.
fn project_runs(path: &Path, json: bool) -> Result<()> {
    let project = Project::open(path)?;
    let runs = project.runs()?;
    if json {
        say!("{}", serde_json::to_string_pretty(&runs)?);
        return Ok(());
    }
    for run in &runs {
        say!("{}", run_line(run));
    }
    Ok(())
}

/// One run as `vcal project runs` lists it.
fn run_line(run: &RunInfo) -> String {
    let (time, views) = match &run.state {
        RunState::Recorded(record) => {
            let views = match &record.source {
                RunSource::Calibrated(calibrated) => format!(
                    "{}/{} views",
                    calibrated.usable_views, calibrated.total_views
                ),
                _ => "imported".to_string(),
            };
            (utc(record.timestamp), views)
        }
        RunState::Unrecorded(unrecorded) => {
            let views = match (unrecorded.usable_views, unrecorded.total_views) {
                (Some(used), Some(total)) => format!("{used}/{total} views"),
                _ => "-".to_string(),
            };
            ("-".to_string(), views)
        }
        RunState::Unreadable { message } => {
            return format!("{}  unreadable: {message}", run.id);
        }
        _ => ("-".to_string(), "-".to_string()),
    };
    let summary = run.summary();
    let verdict = summary
        .and_then(|s| s.verdict)
        .map_or_else(|| "-".to_string(), |v| name(&v));
    let cameras = summary.map_or_else(String::new, |s| {
        s.cameras
            .iter()
            .map(|c| match c.rms_px {
                Some(rms) => format!("{} {rms:.3} px", c.id),
                None => format!("{} -", c.id),
            })
            .collect::<Vec<_>>()
            .join(", ")
    });
    let mut line = format!("{}  {time}  {verdict}  {views}  {cameras}", run.id);
    let notes = notes_line(&run.notes);
    if !notes.is_empty() {
        line.push_str("  ");
        line.push_str(&notes);
    }
    line
}

/// A run's star and label.
fn notes_line(notes: &RunNotes) -> String {
    let star = if notes.starred { "*" } else { "" };
    match &notes.label {
        Some(label) if notes.starred => format!("{star} {label}"),
        Some(label) => label.clone(),
        None => star.to_string(),
    }
}

/// `vcal project scan`.
fn project_scan(scan: &LibraryScan, json: bool) -> Result<()> {
    for failure in &scan.failures {
        eprintln!("warning: {}: {}", failure.path.display(), failure.message);
    }
    let devices = scan.devices();
    if json {
        #[derive(Serialize)]
        struct Output<'a> {
            #[serde(flatten)]
            scan: &'a LibraryScan,
            devices: &'a [vision_calibration::project::DeviceHistory],
        }
        say!(
            "{}",
            serde_json::to_string_pretty(&Output {
                scan,
                devices: &devices
            })?
        );
        return Ok(());
    }
    for history in &devices {
        let device = &history.device;
        let mut heading = match &device.product {
            Some(product) => format!("{product} {}", device.id),
            None => device.id.clone(),
        };
        if let Some(label) = &device.label {
            heading.push_str(&format!(" ({label})"));
        }
        say!("{heading}:");
        for entry in &history.runs {
            say!("  {}  {}", entry.project.display(), run_line(&entry.run));
        }
    }
    let loose: Vec<_> = scan
        .projects
        .iter()
        .filter(|p| p.device.is_none())
        .collect();
    if !loose.is_empty() {
        say!("projects without a device:");
        for project in loose {
            say!("  {}  ({} runs)", project.dir.display(), project.runs.len());
        }
    }
    Ok(())
}

/// `path` made absolute against the current directory, its `.` and `..`
/// folded.
fn absolute(path: &Path) -> Result<PathBuf> {
    let path =
        std::path::absolute(path).with_context(|| format!("resolving {}", path.display()))?;
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                out.pop();
            }
            other => out.push(other),
        }
    }
    Ok(out)
}

/// `target` as the project file writes it: relative to the project folder
/// when it lies inside it (`.` for the folder itself), absolute otherwise.
fn within(folder: &Path, target: &Path) -> PathBuf {
    match target.strip_prefix(folder) {
        Ok(rel) if rel.as_os_str().is_empty() => PathBuf::from("."),
        Ok(rel) => PathBuf::from(rel.to_string_lossy().replace('\\', "/")),
        Err(_) => target.to_path_buf(),
    }
}

/// Seconds since the Unix epoch as the UTC date and minute, `2026-11-26 11:21`.
fn utc(seconds: u64) -> String {
    let days = i64::try_from(seconds / 86_400).unwrap_or(0);
    let rest = seconds % 86_400;
    // Days to the civil date, proleptic Gregorian (Howard Hinnant's
    // algorithm).
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    format!(
        "{year:04}-{month:02}-{day:02} {:02}:{:02}",
        rest / 3_600,
        (rest % 3_600) / 60
    )
}

/// The default configuration of `topology` with the changes the file at
/// `path` makes: a [`ConfigPatch`], TOML or JSON by the file's extension.
fn configuration(topology: ProblemKind, path: &Path) -> Result<CalibrationConfig> {
    let patch = config_patch(path)?;
    CalibrationConfig::patched(topology, &patch)
        .with_context(|| format!("reading the configuration {}", path.display()))
}

/// The [`ConfigPatch`] in the file at `path`, TOML or JSON by its extension.
fn config_patch(path: &Path) -> Result<ConfigPatch> {
    let context = || format!("reading the configuration {}", path.display());
    let text = std::fs::read_to_string(path).with_context(context)?;
    if path
        .extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("toml"))
    {
        ConfigPatch::from_toml_str(&text)
    } else {
        ConfigPatch::from_json_str(&text)
    }
    .with_context(context)
}

fn json_text(value: &impl Serialize) -> Result<String> {
    Ok(serde_json::to_string_pretty(value)? + "\n")
}

/// Writes every file or none: each goes to a temporary sibling first, and
/// the files take their names only once all of them are written, in the
/// order given (put the likeliest to fail first). A failure removes the
/// temporary files, so the directory is as it was unless a later rename
/// fails after an earlier one succeeded.
fn write_all(files: &[(PathBuf, String)]) -> Result<()> {
    let staged = |path: &Path| {
        let mut name = path.as_os_str().to_owned();
        name.push(".partial");
        PathBuf::from(name)
    };
    let cleanup = || {
        for (path, _) in files {
            let _ = std::fs::remove_file(staged(path));
        }
    };
    for (path, text) in files {
        if let Err(err) = std::fs::write(staged(path), text) {
            cleanup();
            return Err(err).with_context(|| format!("writing {}", path.display()));
        }
    }
    for (path, _) in files {
        if let Err(err) = std::fs::rename(staged(path), path) {
            cleanup();
            return Err(err).with_context(|| format!("writing {}", path.display()));
        }
    }
    Ok(())
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

/// Format an [`UncertaintyUnavailable`] reason into readable CLI prose.
fn format_uncertainty_unavailable(
    reason: &vision_calibration::core::UncertaintyUnavailable,
) -> String {
    use vision_calibration::core::UncertaintyUnavailable;
    match reason {
        UncertaintyUnavailable::RankDeficient { unobservable } => {
            if unobservable.is_empty() {
                "rank_deficient".to_string()
            } else {
                format!("rank_deficient (unobservable: {})", unobservable.join(", "))
            }
        }
        UncertaintyUnavailable::AtBound { labels } => {
            if labels.is_empty() {
                "at_bound".to_string()
            } else {
                format!("at_bound (labels: {})", labels.join(", "))
            }
        }
        _ => match serde_json::to_value(reason) {
            Ok(serde_json::Value::Object(map)) => map
                .get("kind")
                .and_then(|v| v.as_str())
                .map(str::to_string)
                .unwrap_or_else(|| name(reason)),
            _ => name(reason),
        },
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use vision_calibration::core::UncertaintyUnavailable;

    #[test]
    fn bare_manifest_filename_resolves_base_dir_to_dot() {
        assert_eq!(manifest_base_dir(Path::new("dataset.toml")), Path::new("."));
        assert_eq!(
            manifest_base_dir(Path::new("sub/dataset.toml")),
            Path::new("sub")
        );
    }

    #[test]
    fn utc_formats_the_date_and_minute() {
        assert_eq!(utc(0), "1970-01-01 00:00");
        assert_eq!(utc(951_782_400), "2000-02-29 00:00");
        assert_eq!(utc(1_795_692_082), "2026-11-26 11:21");
    }

    #[test]
    fn within_is_relative_inside_the_folder_and_absolute_outside() {
        let folder = Path::new("/work/project");
        assert_eq!(within(folder, folder), Path::new("."));
        assert_eq!(
            within(folder, Path::new("/work/project/a/b.toml")),
            Path::new("a/b.toml")
        );
        assert_eq!(within(folder, Path::new("/data/x")), Path::new("/data/x"));
    }

    #[test]
    fn uncertainty_unavailable_formats_struct_and_unit_variants_cleanly() {
        assert_eq!(
            format_uncertainty_unavailable(&UncertaintyUnavailable::NotComputed),
            "not_computed"
        );
        let rank = format_uncertainty_unavailable(&UncertaintyUnavailable::RankDeficient {
            unobservable: vec!["fx".into(), "k1".into()],
        });
        assert_eq!(rank, "rank_deficient (unobservable: fx, k1)");
        let bound = format_uncertainty_unavailable(&UncertaintyUnavailable::AtBound {
            labels: vec!["k2".into()],
        });
        assert_eq!(bound, "at_bound (labels: k2)");
    }
}
