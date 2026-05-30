use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use vision_calibration_bench::registry::load_registry;

/// Calibration benchmarking harness.
///
/// Run calibration pipelines against registered datasets, collect quality
/// metrics, and compare results against frozen golden fixtures.
#[derive(Parser)]
#[command(name = "calib-bench", about, version)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Run calibration on a registered dataset and emit a BenchRecord.
    Run(RunArgs),
    /// Print a summary report from a stored BenchRecord JSON file.
    Report(ReportArgs),
    /// Compare a BenchRecord against a frozen golden fixture.
    Compare(CompareArgs),
    /// Freeze the current BenchRecord as the new golden fixture.
    FreezeFixtures(FreezeFixturesArgs),
    /// List all datasets registered in the bench registry.
    List(ListArgs),
}

/// Arguments for the `run` subcommand.
#[derive(Parser)]
struct RunArgs {
    /// Dataset id to run (matches a registry entry `id`).
    #[arg(long)]
    dataset: String,
    /// Path to the bench registry JSON. Defaults to the crate's
    /// `registry/public.json`.
    #[arg(long)]
    registry: Option<PathBuf>,
}

/// Arguments for the `report` subcommand.
#[derive(Parser)]
struct ReportArgs {
    /// Path to a BenchRecord JSON file produced by `run`.
    record: Option<String>,
}

/// Arguments for the `compare` subcommand.
#[derive(Parser)]
struct CompareArgs {
    /// Path to the BenchRecord JSON to compare against its golden fixture.
    record: Option<String>,
}

/// Arguments for the `freeze-fixtures` subcommand.
#[derive(Parser)]
struct FreezeFixturesArgs {
    /// Path to a BenchRecord JSON to promote to the golden fixture store.
    record: Option<String>,
}

/// Arguments for the `list` subcommand.
#[derive(Parser)]
struct ListArgs {
    /// Show only datasets matching this tier (e.g. "a", "b").
    tier: Option<String>,
}

/// Default registry path: `<crate>/registry/public.json`.
fn default_registry_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("registry/public.json")
}

/// Current git SHA, or `"unknown"` if git is unavailable.
#[cfg(feature = "tier-b")]
fn git_sha() -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".to_string())
}

/// Unix epoch seconds as a string (no chrono dependency).
#[cfg(feature = "tier-b")]
fn unix_epoch_secs_string() -> String {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs().to_string())
        .unwrap_or_else(|_| "0".to_string())
}

/// Cargo features active for this build (best-effort).
#[cfg(feature = "tier-b")]
fn active_features() -> Vec<String> {
    let mut f = Vec::new();
    if cfg!(feature = "tier-a") {
        f.push("tier-a".to_string());
    }
    if cfg!(feature = "tier-b") {
        f.push("tier-b".to_string());
    }
    if cfg!(feature = "laser") {
        f.push("laser".to_string());
    }
    f
}

fn cmd_run(args: &RunArgs) -> Result<()> {
    let registry_path = args.registry.clone().unwrap_or_else(default_registry_path);
    let registry = load_registry(&registry_path)?;
    let entry = registry.find(&args.dataset).with_context(|| {
        format!(
            "dataset '{}' not found in {}",
            args.dataset,
            registry_path.display()
        )
    })?;

    run_dataset(entry)?;
    Ok(())
}

/// Route a dataset to the runner matching its [`ProblemKind`], inject
/// provenance, and print the resulting record.
///
/// `PlanarIntrinsics`, `RigExtrinsics`, and `SingleCamHandeye` are wired;
/// other kinds print a clear "not yet wired" message and return Ok so the CLI
/// stays usable while the remaining runners land.
#[cfg(feature = "tier-b")]
fn run_dataset(entry: &vision_calibration_bench::registry::BenchEntry) -> Result<()> {
    use vision_calibration_bench::registry::ProblemKind;
    use vision_calibration_bench::run::{
        run_planar_intrinsics, run_rig_extrinsics, run_single_cam_handeye,
    };

    // Resolve a relative data_root against the workspace root (derived from
    // `CARGO_MANIFEST_DIR` = `<root>/crates/<crate>`) so the harness works
    // regardless of the caller's CWD.
    let mut entry = entry.clone();
    if entry.data_root.is_relative() {
        let workspace_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        entry.data_root = workspace_root.join(&entry.data_root);
    }

    let mut record = match entry.problem {
        ProblemKind::PlanarIntrinsics => run_planar_intrinsics(&entry)?,
        ProblemKind::RigExtrinsics => run_rig_extrinsics(&entry)?,
        ProblemKind::SingleCamHandeye => run_single_cam_handeye(&entry)?,
        other => {
            anyhow::bail!("problem kind {other:?} is not yet wired into `calib-bench run`");
        }
    };

    // Inject provenance the record type keeps pure (never read inside the run).
    record.ident.git_sha = git_sha();
    record.ident.timestamp_rfc3339 = unix_epoch_secs_string();
    record.ident.config_hash = 0;
    record.ident.features = active_features();

    println!("{}", serde_json::to_string_pretty(&record)?);
    Ok(())
}

#[cfg(not(feature = "tier-b"))]
fn run_dataset(_entry: &vision_calibration_bench::registry::BenchEntry) -> Result<()> {
    println!("run requires --features tier-b");
    Ok(())
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Run(args) => cmd_run(&args)?,
        Command::Report(_args) => {
            println!("report: not implemented yet");
        }
        Command::Compare(_args) => {
            println!("compare: not implemented yet");
        }
        Command::FreezeFixtures(_args) => {
            println!("freeze-fixtures: not implemented yet");
        }
        Command::List(_args) => {
            println!("list: not implemented yet");
        }
    }
    Ok(())
}
