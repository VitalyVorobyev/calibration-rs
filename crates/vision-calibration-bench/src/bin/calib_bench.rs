use std::io::Read;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use vision_calibration_bench::record::BenchRecord;
use vision_calibration_bench::registry::{
    BenchEntry, BenchHandEyeMode, HandeyeBaOverride, ProblemKind, RigHandeyeOverride,
    SingleCamHandeyeOverride, load_registry,
};
use vision_calibration_pipeline::analysis::ReprojLevel;

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
    /// Run deterministic diagnostic sweeps.
    Diagnose(DiagnoseArgs),
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
    /// Optional path to write full per-feature residuals. The default record
    /// printed to stdout stays compact.
    #[arg(long)]
    residuals_out: Option<PathBuf>,
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

/// Arguments for diagnostic commands.
#[derive(Parser)]
struct DiagnoseArgs {
    #[command(subcommand)]
    command: DiagnoseCommand,
}

/// Diagnostic command variants.
#[derive(Subcommand)]
enum DiagnoseCommand {
    /// Run fixed hand-eye configuration sweeps.
    Handeye(DiagnoseHandeyeArgs),
    /// Profile detector and extractor stages without running calibration.
    Stages(DiagnoseStagesArgs),
}

/// Arguments for `diagnose handeye`.
#[derive(Parser)]
struct DiagnoseHandeyeArgs {
    /// Dataset id to diagnose.
    #[arg(long)]
    dataset: String,
    /// Path to the bench registry JSON.
    #[arg(long)]
    registry: Option<PathBuf>,
}

/// Arguments for `diagnose stages`.
#[derive(Parser)]
struct DiagnoseStagesArgs {
    /// Dataset id to diagnose.
    #[arg(long)]
    dataset: String,
    /// Path to the bench registry JSON.
    #[arg(long)]
    registry: Option<PathBuf>,
    /// Limit profiled images per camera.
    #[arg(long)]
    max_images: Option<usize>,
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
    let entry = load_entry(&args.dataset, args.registry.as_deref())?;
    let record = run_dataset_record(&entry)?;

    if let Some(path) = &args.residuals_out {
        let sidecar = record
            .residual_sidecar
            .as_ref()
            .context("run did not produce a residual sidecar")?;
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("create {}", parent.display()))?;
        }
        std::fs::write(path, serde_json::to_string_pretty(sidecar)?)
            .with_context(|| format!("write residual sidecar {}", path.display()))?;
    }

    println!("{}", serde_json::to_string_pretty(&record)?);
    Ok(())
}

fn load_entry(dataset: &str, registry_path: Option<&Path>) -> Result<BenchEntry> {
    let registry_path = registry_path
        .map(PathBuf::from)
        .unwrap_or_else(default_registry_path);
    let registry = load_registry(&registry_path)?;
    registry.find(dataset).cloned().with_context(|| {
        format!(
            "dataset '{}' not found in {}",
            dataset,
            registry_path.display()
        )
    })
}

fn resolve_entry_data_root(mut entry: BenchEntry) -> BenchEntry {
    if entry.data_root.is_relative() {
        let workspace_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        entry.data_root = workspace_root.join(&entry.data_root);
    }
    entry
}

/// Route a dataset to the runner matching its [`ProblemKind`], inject
/// provenance, and print the resulting record.
///
/// `PlanarIntrinsics`, `RigExtrinsics`, and `SingleCamHandeye` are wired;
/// other kinds print a clear "not yet wired" message and return Ok so the CLI
/// stays usable while the remaining runners land.
#[cfg(feature = "tier-b")]
fn run_dataset_record(entry: &BenchEntry) -> Result<BenchRecord> {
    use vision_calibration_bench::run::{
        run_planar_intrinsics, run_rig_extrinsics, run_rig_handeye, run_single_cam_handeye,
    };

    // Resolve a relative data_root against the workspace root (derived from
    // `CARGO_MANIFEST_DIR` = `<root>/crates/<crate>`) so the harness works
    // regardless of the caller's CWD.
    let entry = resolve_entry_data_root(entry.clone());

    let mut record = match entry.problem {
        ProblemKind::PlanarIntrinsics => run_planar_intrinsics(&entry)?,
        ProblemKind::RigExtrinsics => run_rig_extrinsics(&entry)?,
        ProblemKind::SingleCamHandeye => run_single_cam_handeye(&entry)?,
        ProblemKind::RigHandeye => run_rig_handeye(&entry)?,
        other => {
            anyhow::bail!("problem kind {other:?} is not yet wired into `calib-bench run`");
        }
    };

    // Inject provenance the record type keeps pure (never read inside the run).
    record.ident.git_sha = git_sha();
    record.ident.timestamp_rfc3339 = unix_epoch_secs_string();
    record.ident.config_hash = 0;
    record.ident.features = active_features();

    Ok(record)
}

#[cfg(not(feature = "tier-b"))]
fn run_dataset_record(_entry: &BenchEntry) -> Result<BenchRecord> {
    anyhow::bail!("run requires --features tier-b")
}

fn cmd_report(args: &ReportArgs) -> Result<()> {
    let record = read_record(args.record.as_deref())?;
    println!("{}", render_markdown_report(&record));
    Ok(())
}

fn read_record(path: Option<&str>) -> Result<BenchRecord> {
    let mut text = String::new();
    match path {
        Some(path) => {
            text = std::fs::read_to_string(path).with_context(|| format!("read {path}"))?;
        }
        None => {
            std::io::stdin()
                .read_to_string(&mut text)
                .context("read BenchRecord JSON from stdin")?;
        }
    }
    serde_json::from_str(&text).context("parse BenchRecord JSON")
}

fn render_markdown_report(record: &BenchRecord) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "# Benchmark Report: {}\n\n",
        record.ident.dataset_id
    ));
    out.push_str(&format!(
        "- problem: `{}`\n- converged: `{}`\n- reported mean reprojection: {:.5} px\n",
        record.ident.problem, record.convergence.converged, record.fit.reported_mean_reproj_px
    ));
    out.push_str(&format!(
        "- timing: init={} ms, optimize={} ms, detection={} ms, total={} ms\n",
        record.timing.init_ms,
        record.timing.optimize_ms,
        record.timing.detection_ms,
        record.timing.total_ms
    ));

    if let Some(report) = &record.reproj_report {
        out.push_str("\n## Reprojection Levels\n\n");
        out.push_str("| level | mean | rms | p95 | max | count |\n");
        out.push_str("|---|---:|---:|---:|---:|---:|\n");
        for level in &report.levels {
            out.push_str(&format!(
                "| {:?} | {:.5} | {:.5} | {:.5} | {:.5} | {} |\n",
                level.level,
                level.overall.mean,
                level.overall.rms,
                level.overall.p95,
                level.overall.max,
                level.overall.count
            ));
        }
        if !report.gaps.is_empty() {
            out.push_str("\n## Level Gaps\n\n");
            out.push_str("| from | to | delta px | ratio prev | ratio floor |\n");
            out.push_str("|---|---|---:|---:|---:|\n");
            for gap in &report.gaps {
                out.push_str(&format!(
                    "| {:?} | {:?} | {:.5} | {} | {} |\n",
                    gap.from,
                    gap.to,
                    gap.mean_delta_px,
                    fmt_opt(gap.ratio_to_previous),
                    fmt_opt(gap.ratio_to_intrinsic)
                ));
            }
        }
    }

    if let Some(detection) = &record.detection {
        out.push_str("\n## Detection\n\n");
        out.push_str("| camera | used / total | features | coverage |\n");
        out.push_str("|---|---:|---:|---:|\n");
        for stat in &detection.per_camera {
            out.push_str(&format!(
                "| {} | {} / {} | {} / {} | {:.2}% |\n",
                stat.camera_id,
                stat.images_used,
                stat.images_total,
                stat.features_detected,
                stat.features_expected,
                stat.coverage_pct
            ));
        }
    }

    if let Some(laser) = &record.laser {
        out.push_str("\n## Laser Extraction\n\n");
        out.push_str(&format!(
            "- total points: {}\n- images used: {}\n- extraction: {} ms\n\n",
            laser.total_points, laser.total_images_used, laser.extract_ms
        ));
        out.push_str("| camera | used / total | points | extract ms |\n");
        out.push_str("|---|---:|---:|---:|\n");
        for stat in &laser.per_camera {
            out.push_str(&format!(
                "| {} | {} / {} | {} | {} |\n",
                stat.camera_id,
                stat.images_used,
                stat.images_total,
                stat.points_extracted,
                stat.extract_ms
            ));
        }
    }

    if let Some(robot) = &record.robot_corrections {
        out.push_str("\n## Robot Pose Corrections\n\n");
        out.push_str("| count | mean rot | max rot | mean trans | max trans |\n");
        out.push_str("|---:|---:|---:|---:|---:|\n");
        out.push_str(&format!(
            "| {} | {:.3} deg | {:.3} deg | {:.3} mm | {:.3} mm |\n",
            robot.count,
            robot.mean_rot_deg,
            robot.max_rot_deg,
            robot.mean_trans_mm,
            robot.max_trans_mm
        ));
    }

    out
}

fn fmt_opt(v: Option<f64>) -> String {
    v.map(|v| format!("{v:.3}"))
        .unwrap_or_else(|| "-".to_string())
}

fn cmd_diagnose(args: &DiagnoseArgs) -> Result<()> {
    match &args.command {
        DiagnoseCommand::Handeye(args) => cmd_diagnose_handeye(args),
        DiagnoseCommand::Stages(args) => cmd_diagnose_stages(args),
    }
}

fn cmd_diagnose_stages(args: &DiagnoseStagesArgs) -> Result<()> {
    let entry = resolve_entry_data_root(load_entry(&args.dataset, args.registry.as_deref())?);
    let profile = profile_dataset_stages(&entry, args.max_images)?;
    println!("{}", serde_json::to_string_pretty(&profile)?);
    Ok(())
}

#[cfg(feature = "tier-b")]
fn profile_dataset_stages(
    entry: &BenchEntry,
    max_images: Option<usize>,
) -> Result<vision_calibration_bench::run::tier_b::DatasetStageProfile> {
    vision_calibration_bench::run::tier_b::profile_dataset_stages(entry, max_images)
}

#[cfg(not(feature = "tier-b"))]
fn profile_dataset_stages(
    _entry: &BenchEntry,
    _max_images: Option<usize>,
) -> Result<serde_json::Value> {
    anyhow::bail!("diagnose stages requires --features tier-b")
}

fn cmd_diagnose_handeye(args: &DiagnoseHandeyeArgs) -> Result<()> {
    let entry = load_entry(&args.dataset, args.registry.as_deref())?;
    anyhow::ensure!(
        matches!(
            entry.problem,
            ProblemKind::SingleCamHandeye | ProblemKind::RigHandeye
        ),
        "diagnose handeye supports single_cam_handeye and rig_handeye datasets"
    );

    let cases = handeye_cases(&entry);
    println!("# Hand-Eye Diagnostic: {}\n", entry.id);
    println!(
        "| case | status | intrinsic mean | hand-eye mean | ratio | robot trans mean/max | robot rot mean/max | note |"
    );
    println!("|---|---|---:|---:|---:|---:|---:|---|");
    for (name, case_entry) in cases {
        match run_dataset_record(&case_entry) {
            Ok(record) => {
                let (floor, constrained) = handeye_level_means(&record);
                let robot_trans = record
                    .robot_corrections
                    .as_ref()
                    .map(|r| format!("{:.3}/{:.3} mm", r.mean_trans_mm, r.max_trans_mm))
                    .unwrap_or_else(|| "-".to_string());
                let robot_rot = record
                    .robot_corrections
                    .as_ref()
                    .map(|r| format!("{:.3}/{:.3} deg", r.mean_rot_deg, r.max_rot_deg))
                    .unwrap_or_else(|| "-".to_string());
                println!(
                    "| {} | ok | {} | {} | {} | {} | {} |  |",
                    name,
                    fmt_opt(floor),
                    fmt_opt(constrained),
                    fmt_ratio(constrained, floor),
                    robot_trans,
                    robot_rot
                );
            }
            Err(err) => {
                println!(
                    "| {} | fail | - | - | - | - | - | {} |",
                    name,
                    escape_md(&err.to_string())
                );
            }
        }
    }
    Ok(())
}

fn handeye_cases(entry: &BenchEntry) -> Vec<(&'static str, BenchEntry)> {
    let mut cases = Vec::new();
    cases.push(("default", entry.clone()));

    let mut alternate_mode = entry.clone();
    if set_alternate_handeye_mode(&mut alternate_mode) {
        cases.push(("alternate_handeye_mode", alternate_mode));
    }

    let mut no_refine = entry.clone();
    set_robot_ba(&mut no_refine, Some(false), None, None);
    cases.push(("robot_refine_off", no_refine));

    let mut loose = entry.clone();
    set_robot_ba(
        &mut loose,
        Some(true),
        Some(5.0_f64.to_radians()),
        Some(0.010),
    );
    cases.push(("loose_robot_prior", loose));

    if entry.robot_poses.is_some() {
        let mut inverse = entry.clone();
        if let Some(src) = inverse.robot_poses.as_mut() {
            src.convention = "gripper_se3_base".to_string();
        }
        cases.push(("pose_convention_inverted", inverse));
    }

    cases
}

fn set_alternate_handeye_mode(entry: &mut BenchEntry) -> bool {
    match entry.problem {
        ProblemKind::SingleCamHandeye => {
            let overrides = entry
                .single_cam_handeye
                .get_or_insert_with(SingleCamHandeyeOverride::default);
            overrides.handeye_mode = Some(
                match overrides
                    .handeye_mode
                    .unwrap_or(BenchHandEyeMode::EyeInHand)
                {
                    BenchHandEyeMode::EyeInHand => BenchHandEyeMode::EyeToHand,
                    BenchHandEyeMode::EyeToHand => BenchHandEyeMode::EyeInHand,
                },
            );
            true
        }
        ProblemKind::RigHandeye => {
            let overrides = entry
                .rig_handeye
                .get_or_insert_with(RigHandeyeOverride::default);
            overrides.handeye_mode = Some(
                match overrides
                    .handeye_mode
                    .unwrap_or(BenchHandEyeMode::EyeInHand)
                {
                    BenchHandEyeMode::EyeInHand => BenchHandEyeMode::EyeToHand,
                    BenchHandEyeMode::EyeToHand => BenchHandEyeMode::EyeInHand,
                },
            );
            true
        }
        _ => false,
    }
}

fn set_robot_ba(
    entry: &mut BenchEntry,
    refine_robot_poses: Option<bool>,
    robot_rot_sigma: Option<f64>,
    robot_trans_sigma: Option<f64>,
) {
    let ba = HandeyeBaOverride {
        refine_robot_poses,
        robot_rot_sigma,
        robot_trans_sigma,
    };
    match entry.problem {
        ProblemKind::SingleCamHandeye => {
            let overrides = entry
                .single_cam_handeye
                .get_or_insert_with(SingleCamHandeyeOverride::default);
            overrides.handeye_ba = Some(ba);
        }
        ProblemKind::RigHandeye => {
            let overrides = entry
                .rig_handeye
                .get_or_insert_with(RigHandeyeOverride::default);
            overrides.handeye_ba = Some(ba);
        }
        _ => {}
    }
}

fn handeye_level_means(record: &BenchRecord) -> (Option<f64>, Option<f64>) {
    let Some(report) = &record.reproj_report else {
        return (None, None);
    };
    let floor = report
        .levels
        .iter()
        .find(|l| l.level == ReprojLevel::Intrinsic)
        .map(|l| l.overall.mean);
    let constrained = report
        .levels
        .iter()
        .rev()
        .find(|l| l.level == ReprojLevel::HandEye)
        .map(|l| l.overall.mean);
    (floor, constrained)
}

fn fmt_ratio(num: Option<f64>, den: Option<f64>) -> String {
    match (num, den) {
        (Some(num), Some(den)) if den > 0.0 => format!("{:.3}", num / den),
        _ => "-".to_string(),
    }
}

fn escape_md(s: &str) -> String {
    s.replace('|', "\\|").replace('\n', " ")
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Run(args) => cmd_run(&args)?,
        Command::Report(args) => cmd_report(&args)?,
        Command::Compare(_args) => {
            println!("compare: not implemented yet");
        }
        Command::FreezeFixtures(_args) => {
            println!("freeze-fixtures: not implemented yet");
        }
        Command::List(_args) => {
            println!("list: not implemented yet");
        }
        Command::Diagnose(args) => cmd_diagnose(&args)?,
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use vision_calibration_bench::record::{
        BENCH_SCHEMA_VERSION, CompactLevelReport, CompactReprojReport, Convergence, Detection,
        DetectionStat, Fit, Ident, LaserCamStat, LaserMetrics, ReprojLevelGap,
        RobotCorrectionSummary, Timing,
    };
    use vision_calibration_core::{FeatureResidualHistogram, ReprojectionStats};
    use vision_calibration_optim::SolveReport;
    use vision_calibration_pipeline::analysis::LevelStats;

    #[test]
    fn report_renderer_includes_quality_sections() {
        let stats = ReprojectionStats::from_errors(&[0.8, 1.0, 1.2]);
        let level_stats = LevelStats {
            mean: 1.0,
            median: 1.0,
            rms: 1.01,
            p95: 1.19,
            max: 1.2,
            count: 3,
        };
        let record = BenchRecord {
            ident: Ident {
                dataset_id: "ds".to_string(),
                problem: "rig_handeye".to_string(),
                tier: "b".to_string(),
                git_sha: "abc".to_string(),
                timestamp_rfc3339: "0".to_string(),
                config_hash: 0,
                bench_schema_version: BENCH_SCHEMA_VERSION,
                features: vec!["tier-b".to_string()],
            },
            convergence: Convergence {
                init_ok: true,
                converged: true,
                report: SolveReport {
                    final_cost: 1.0,
                    num_iters: 2,
                },
            },
            fit: Fit {
                overall: stats,
                per_camera: vec![stats],
                per_camera_hist: vec![FeatureResidualHistogram::default()],
                reported_mean_reproj_px: 1.2,
                reported_per_cam_px: vec![1.2],
            },
            generalization: None,
            stability: None,
            detection: Some(Detection {
                per_camera: vec![DetectionStat {
                    camera_id: "cam0".to_string(),
                    images_total: 2,
                    images_used: 1,
                    features_detected: 130,
                    features_expected: 260,
                    coverage_pct: 50.0,
                    detect_ms: 7,
                }],
                total_detected: 130,
                total_expected: 260,
            }),
            laser: Some(LaserMetrics {
                per_camera: vec![LaserCamStat {
                    camera_id: "cam0".to_string(),
                    images_total: 2,
                    images_used: 2,
                    points_extracted: 42,
                    extract_ms: 3,
                    plane_residual_m: None,
                    line_residual_px: None,
                    inlier_ratio: None,
                }],
                total_points: 42,
                total_images_used: 2,
                extract_ms: 3,
            }),
            robot_corrections: Some(RobotCorrectionSummary {
                count: 2,
                mean_rot_deg: 0.1,
                max_rot_deg: 0.2,
                mean_trans_mm: 0.4,
                max_trans_mm: 0.8,
            }),
            delta_to_prior: None,
            timing: Timing {
                init_ms: 1,
                optimize_ms: 2,
                total_ms: 10,
                detection_ms: 7,
            },
            reproj_report: Some(CompactReprojReport {
                headline_px: 1.2,
                levels: vec![
                    CompactLevelReport {
                        level: ReprojLevel::Intrinsic,
                        overall: level_stats,
                        per_camera: vec![level_stats],
                        per_view: vec![level_stats],
                        residual_count: 3,
                        top_outliers: Vec::new(),
                    },
                    CompactLevelReport {
                        level: ReprojLevel::HandEye,
                        overall: LevelStats {
                            mean: 1.2,
                            ..level_stats
                        },
                        per_camera: vec![level_stats],
                        per_view: vec![level_stats],
                        residual_count: 3,
                        top_outliers: Vec::new(),
                    },
                ],
                gaps: vec![ReprojLevelGap {
                    from: ReprojLevel::Intrinsic,
                    to: ReprojLevel::HandEye,
                    mean_delta_px: 0.2,
                    ratio_to_previous: Some(1.2),
                    ratio_to_intrinsic: Some(1.2),
                }],
            }),
            residual_sidecar: None,
        };

        let md = render_markdown_report(&record);
        assert!(md.contains("# Benchmark Report: ds"));
        assert!(md.contains("## Reprojection Levels"));
        assert!(md.contains("## Level Gaps"));
        assert!(md.contains("## Detection"));
        assert!(md.contains("## Laser Extraction"));
        assert!(md.contains("## Robot Pose Corrections"));
    }
}
