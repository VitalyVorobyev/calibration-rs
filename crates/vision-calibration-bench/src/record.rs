//! The metric record a benchmark run produces.
//!
//! [`BenchRecord`] is pure, serde-serializable data: it captures *what a run
//! produced*, never *how to judge it* (that lives in [`crate::compare`]). It
//! reuses existing workspace types verbatim -- [`ReprojectionStats`],
//! [`FeatureResidualHistogram`], and [`SolveReport`] -- rather than redefining
//! them.
//!
//! # No `PartialEq`
//!
//! Several embedded workspace types ([`ReprojectionStats`], [`SolveReport`]) do
//! not implement `PartialEq`, so neither does [`BenchRecord`]. The serde
//! roundtrip tests assert equality by comparing the re-serialized JSON, which is
//! a strictly stronger check of the wire format than structural equality.
//!
//! All optional sections ([`Generalization`], [`Stability`], [`Detection`],
//! [`LaserMetrics`], [`DeltaToPrior`]) are `None` for runs that did not exercise
//! that capability (e.g. a Tier-A planar intrinsics run has no detection or
//! laser metrics).

use serde::{Deserialize, Serialize};
use vision_calibration_core::{FeatureResidualHistogram, ReprojectionStats, TargetFeatureResidual};
use vision_calibration_optim::SolveReport;
use vision_calibration_pipeline::analysis::{LevelStats, ReprojLevel, ReprojReport};

/// Schema version for [`BenchRecord`]. Bump on any breaking layout change so
/// frozen records carry a version stamp ([`Ident::bench_schema_version`]).
///
/// v3 replaces the full in-record [`ReprojReport`] with a compact report and a
/// transient optional [`ResidualSidecar`]. Full residual vectors are written
/// only when the CLI is asked for a sidecar.
pub const BENCH_SCHEMA_VERSION: u32 = 3;

/// Maximum per-level outliers kept inside compact benchmark records.
pub const DEFAULT_TOP_OUTLIER_LIMIT: usize = 16;

/// One benchmark run's full metric record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchRecord {
    /// Identity / provenance of the run.
    pub ident: Ident,
    /// Convergence outcome of init + optimization.
    pub convergence: Convergence,
    /// Reprojection fit quality on the training data.
    pub fit: Fit,
    /// Held-out cross-validation generalization, if measured.
    pub generalization: Option<Generalization>,
    /// Parameter stability across resampled runs, if measured.
    pub stability: Option<Stability>,
    /// Detection coverage, if the detector ran (Tier-B).
    pub detection: Option<Detection>,
    /// Laser-plane metrics, if this is a laserline problem.
    pub laser: Option<LaserMetrics>,
    /// Per-view robot-pose correction magnitudes, if robot-pose refinement ran.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub robot_corrections: Option<RobotCorrectionSummary>,
    /// Calibration artifacts for dashboard inspection: camera matrices,
    /// distortion coefficients, sensor tilt, and named SE(3) transforms.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub artifacts: Option<CalibrationArtifacts>,
    /// Change versus a frozen prior calibration, if a prior was supplied.
    pub delta_to_prior: Option<DeltaToPrior>,
    /// Wall-clock timing breakdown.
    pub timing: Timing,
    /// Compact hierarchical multi-level reprojection report (intrinsic floor →
    /// rig-extrinsic → hand-eye), if computed. Full residual vectors live in
    /// [`Self::residual_sidecar`] during the process run and are not serialized
    /// into the default record.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reproj_report: Option<CompactReprojReport>,
    /// Optional full residual sidecar, carried in memory only so the CLI can
    /// write it when `--residuals-out` is provided.
    #[serde(skip)]
    pub residual_sidecar: Option<ResidualSidecar>,
}

/// Compact version of [`ReprojReport`] suitable for default JSON output.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompactReprojReport {
    /// The most-constrained level's mean error in pixels.
    pub headline_px: f64,
    /// Compact per-level summaries, ordered least-constrained first.
    pub levels: Vec<CompactLevelReport>,
    /// Adjacent level gaps, plus ratios against the previous level and floor.
    pub gaps: Vec<ReprojLevelGap>,
}

/// Compact summary of one reprojection constraint level.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompactLevelReport {
    /// Constraint level.
    pub level: ReprojLevel,
    /// Aggregate statistics over finite residuals.
    pub overall: LevelStats,
    /// Per-camera aggregates.
    pub per_camera: Vec<LevelStats>,
    /// Per-view aggregates.
    pub per_view: Vec<LevelStats>,
    /// Number of residual records before compacting.
    pub residual_count: usize,
    /// Bounded list of worst residuals for immediate diagnosis.
    pub top_outliers: Vec<TargetFeatureResidual>,
}

/// Gap between two adjacent reprojection levels.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ReprojLevelGap {
    /// Less-constrained source level.
    pub from: ReprojLevel,
    /// More-constrained target level.
    pub to: ReprojLevel,
    /// Difference in mean error, `to.mean - from.mean`.
    pub mean_delta_px: f64,
    /// Ratio `to.mean / from.mean`, or `None` when the denominator is zero.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ratio_to_previous: Option<f64>,
    /// Ratio `to.mean / intrinsic.mean`, or `None` when the intrinsic floor is
    /// absent or zero.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ratio_to_intrinsic: Option<f64>,
}

/// Full residual sidecar emitted only on request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResidualSidecar {
    /// Sidecar schema version. Kept in lockstep with [`BENCH_SCHEMA_VERSION`].
    pub bench_schema_version: u32,
    /// Dataset identifier.
    pub dataset_id: String,
    /// Headline reprojection error in pixels.
    pub headline_px: f64,
    /// Full residual vectors per level.
    pub levels: Vec<ResidualSidecarLevel>,
}

/// Full residuals for one reprojection level.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResidualSidecarLevel {
    /// Constraint level.
    pub level: ReprojLevel,
    /// Full per-feature residual records.
    pub residuals: Vec<TargetFeatureResidual>,
}

/// Summary of optimized per-view robot-pose corrections.
///
/// Rotation is reported as the norm of the se(3) rotation vector in degrees;
/// translation is reported as the norm of the se(3) translation vector in
/// millimetres. The source optimizer stores translation internally in metres.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RobotCorrectionSummary {
    /// Number of per-view corrections.
    pub count: usize,
    /// Average rotation correction magnitude in degrees.
    pub mean_rot_deg: f64,
    /// Maximum rotation correction magnitude in degrees.
    pub max_rot_deg: f64,
    /// Average translation correction magnitude in millimetres.
    pub mean_trans_mm: f64,
    /// Maximum translation correction magnitude in millimetres.
    pub max_trans_mm: f64,
    /// Rotation prior sigma in degrees, when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prior_rot_deg: Option<f64>,
    /// Translation prior sigma in millimetres, when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prior_trans_mm: Option<f64>,
    /// Maximum rotation correction divided by the prior sigma.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_rot_prior_ratio: Option<f64>,
    /// Maximum translation correction divided by the prior sigma.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_trans_prior_ratio: Option<f64>,
    /// Whether any maximum correction exceeds its configured prior sigma.
    #[serde(default)]
    pub exceeds_prior: bool,
}

/// Compact calibration artifacts intended for report viewers.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CalibrationArtifacts {
    /// Unit used for transform translations in this section.
    pub spatial_unit: String,
    /// Unit used for rotation-vector display values.
    pub angle_unit: String,
    /// Per-camera intrinsics and distortion.
    pub cameras: Vec<CameraArtifact>,
    /// Named SE(3) transforms. Each transform maps points from
    /// [`TransformArtifact::from_frame`] into [`TransformArtifact::to_frame`].
    pub transforms: Vec<TransformArtifact>,
}

/// One camera's calibrated model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CameraArtifact {
    /// Stable camera id from the dataset registry.
    pub camera_id: String,
    /// Pinhole camera matrix in pixels.
    pub camera_matrix_px: [[f64; 3]; 3],
    /// Scalar intrinsic parameters in pixels.
    pub intrinsics_px: IntrinsicsArtifact,
    /// Distortion model name.
    pub distortion_model: String,
    /// Brown-Conrady distortion coefficients.
    pub distortion: DistortionArtifact,
    /// Scheimpflug tilt parameters, when the run used a tilted sensor model.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scheimpflug: Option<ScheimpflugArtifact>,
}

/// Scalar pinhole intrinsic parameters.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct IntrinsicsArtifact {
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
    pub skew: f64,
}

/// Brown-Conrady 5-parameter distortion model.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DistortionArtifact {
    pub k1: f64,
    pub k2: f64,
    pub k3: f64,
    pub p1: f64,
    pub p2: f64,
}

/// Scheimpflug sensor tilt artifact.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ScheimpflugArtifact {
    pub tilt_x_rad: f64,
    pub tilt_y_rad: f64,
}

/// Named SE(3) transform artifact.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TransformArtifact {
    /// Field-like transform name, e.g. `cam0_se3_rig`.
    pub name: String,
    /// Destination frame. The transform maps `from_frame` points into this frame.
    pub to_frame: String,
    /// Source frame. The transform maps points from this frame into `to_frame`.
    pub from_frame: String,
    /// Translation vector in millimetres.
    pub translation_mm: [f64; 3],
    /// Unit quaternion `[x, y, z, w]`.
    pub rotation_quat_xyzw: [f64; 4],
    /// Rotation vector in degrees.
    pub rotation_rotvec_deg: [f64; 3],
}

impl RobotCorrectionSummary {
    /// Build a magnitude summary from se(3) deltas `[rx, ry, rz, tx, ty, tz]`,
    /// where rotation is in radians and translation is in metres.
    pub fn from_deltas(deltas: &[[f64; 6]]) -> Option<Self> {
        Self::from_deltas_with_priors(deltas, None, None)
    }

    /// Build a magnitude summary and compare maxima to prior sigmas when
    /// supplied. Rotation prior is radians; translation prior is metres.
    pub fn from_deltas_with_priors(
        deltas: &[[f64; 6]],
        robot_rot_sigma: Option<f64>,
        robot_trans_sigma: Option<f64>,
    ) -> Option<Self> {
        if deltas.is_empty() {
            return None;
        }

        let mut sum_rot = 0.0;
        let mut max_rot = 0.0_f64;
        let mut sum_trans = 0.0;
        let mut max_trans = 0.0_f64;
        for delta in deltas {
            let rot = (delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]).sqrt();
            let trans = (delta[3] * delta[3] + delta[4] * delta[4] + delta[5] * delta[5]).sqrt();
            sum_rot += rot;
            max_rot = max_rot.max(rot);
            sum_trans += trans;
            max_trans = max_trans.max(trans);
        }
        let n = deltas.len() as f64;
        let prior_rot_deg = robot_rot_sigma.map(f64::to_degrees);
        let prior_trans_mm = robot_trans_sigma.map(|v| v * 1000.0);
        let max_rot_deg = max_rot.to_degrees();
        let max_trans_mm = max_trans * 1000.0;
        let max_rot_prior_ratio = prior_rot_deg
            .filter(|v| *v > 0.0)
            .map(|prior| max_rot_deg / prior);
        let max_trans_prior_ratio = prior_trans_mm
            .filter(|v| *v > 0.0)
            .map(|prior| max_trans_mm / prior);
        let exceeds_prior = max_rot_prior_ratio.is_some_and(|ratio| ratio > 1.0)
            || max_trans_prior_ratio.is_some_and(|ratio| ratio > 1.0);
        Some(Self {
            count: deltas.len(),
            mean_rot_deg: (sum_rot / n).to_degrees(),
            max_rot_deg,
            mean_trans_mm: (sum_trans / n) * 1000.0,
            max_trans_mm,
            prior_rot_deg,
            prior_trans_mm,
            max_rot_prior_ratio,
            max_trans_prior_ratio,
            exceeds_prior,
        })
    }
}

/// Split a full pipeline [`ReprojReport`] into the compact in-record shape and
/// an optional sidecar that preserves every per-feature residual.
pub fn compact_reproj_report(
    dataset_id: &str,
    report: ReprojReport,
) -> (CompactReprojReport, ResidualSidecar) {
    compact_reproj_report_with_limit(dataset_id, report, DEFAULT_TOP_OUTLIER_LIMIT)
}

/// Same as [`compact_reproj_report`], with an explicit outlier limit for tests.
pub fn compact_reproj_report_with_limit(
    dataset_id: &str,
    report: ReprojReport,
    top_outlier_limit: usize,
) -> (CompactReprojReport, ResidualSidecar) {
    let headline_px = report.headline_px;
    let intrinsic_mean = report
        .levels
        .iter()
        .find(|level| level.level == ReprojLevel::Intrinsic)
        .map(|level| level.overall.mean)
        .filter(|v| *v > 0.0);

    let mut compact_levels = Vec::with_capacity(report.levels.len());
    let mut sidecar_levels = Vec::with_capacity(report.levels.len());
    for level in report.levels {
        let mut top_outliers = level.residuals.clone();
        top_outliers.sort_by(|a, b| {
            let ae = a.error_px.unwrap_or(f64::NEG_INFINITY);
            let be = b.error_px.unwrap_or(f64::NEG_INFINITY);
            be.total_cmp(&ae)
        });
        top_outliers.truncate(top_outlier_limit);

        let residual_count = level.residuals.len();
        sidecar_levels.push(ResidualSidecarLevel {
            level: level.level,
            residuals: level.residuals,
        });
        compact_levels.push(CompactLevelReport {
            level: level.level,
            overall: level.overall,
            per_camera: level.per_camera,
            per_view: level.per_view,
            residual_count,
            top_outliers,
        });
    }

    let mut gaps = Vec::new();
    for pair in compact_levels.windows(2) {
        let from = &pair[0];
        let to = &pair[1];
        gaps.push(ReprojLevelGap {
            from: from.level,
            to: to.level,
            mean_delta_px: to.overall.mean - from.overall.mean,
            ratio_to_previous: ratio(to.overall.mean, from.overall.mean),
            ratio_to_intrinsic: intrinsic_mean.and_then(|floor| ratio(to.overall.mean, floor)),
        });
    }

    (
        CompactReprojReport {
            headline_px,
            levels: compact_levels,
            gaps,
        },
        ResidualSidecar {
            bench_schema_version: BENCH_SCHEMA_VERSION,
            dataset_id: dataset_id.to_string(),
            headline_px,
            levels: sidecar_levels,
        },
    )
}

fn ratio(num: f64, den: f64) -> Option<f64> {
    if den > 0.0 && num.is_finite() && den.is_finite() {
        Some(num / den)
    } else {
        None
    }
}

/// Identity and provenance of a benchmark run.
///
/// `git_sha` and `timestamp_rfc3339` are injected by the harness from the
/// environment; they are plain `String` fields so the record itself stays pure
/// (no wall-clock or process state read inside the type).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Ident {
    /// Stable dataset identifier (matches a registry entry `id`).
    pub dataset_id: String,
    /// Problem kind the run exercised.
    pub problem: String,
    /// Tier the run executed under (`"a"` / `"b"`).
    pub tier: String,
    /// Git SHA the binary was built from (injected externally).
    pub git_sha: String,
    /// Run timestamp, RFC 3339 (injected externally).
    pub timestamp_rfc3339: String,
    /// Hash of the effective config, for grouping comparable runs.
    pub config_hash: u64,
    /// Schema version of this record ([`BENCH_SCHEMA_VERSION`]).
    pub bench_schema_version: u32,
    /// Cargo features active for the run.
    pub features: Vec<String>,
}

/// Convergence outcome of the linear init followed by nonlinear optimization.
///
/// Embeds [`SolveReport`] verbatim (final cost, iteration count).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Convergence {
    /// Whether the linear initialization succeeded.
    pub init_ok: bool,
    /// Whether the optimizer reported convergence.
    pub converged: bool,
    /// The underlying solver report (final cost, iterations).
    pub report: SolveReport,
}

/// Reprojection fit quality on the data the calibration was fit to.
///
/// `overall` / `per_camera` / `per_camera_hist` are computed by the bench from
/// per-feature residuals. `reported_*` mirror whatever the calibration export
/// itself reported, so divergence between bench-computed and self-reported
/// numbers is detectable.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fit {
    /// Aggregate reprojection statistics across all cameras.
    pub overall: ReprojectionStats,
    /// Per-camera reprojection statistics (length = camera count).
    pub per_camera: Vec<ReprojectionStats>,
    /// Per-camera reprojection-error histograms (length = camera count).
    pub per_camera_hist: Vec<FeatureResidualHistogram>,
    /// Mean reprojection error as reported by the calibration export (pixels).
    pub reported_mean_reproj_px: f64,
    /// Per-camera mean reprojection error as reported by the export (pixels).
    pub reported_per_cam_px: Vec<f64>,
}

/// Held-out cross-validation generalization metrics.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Generalization {
    /// Number of folds used.
    pub folds: usize,
    /// Aggregate held-out RMS reprojection error (pixels).
    pub heldout_rms_px: f64,
    /// Per-fold held-out RMS reprojection error (pixels).
    pub heldout_rms_px_per_fold: Vec<f64>,
    /// Training RMS reprojection error for reference (pixels).
    pub train_rms_px: f64,
}

/// Parameter stability across resampled / reseeded runs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Stability {
    /// Number of resampled runs performed.
    pub n_runs: usize,
    /// Fraction of the data used in each subset.
    pub subset_frac: f64,
    /// Seeds used for the resampled runs (length = `n_runs`).
    pub seeds: Vec<u64>,
    /// Per-parameter spread statistics.
    pub params: Vec<ParamSpread>,
    /// Names of parameters flagged as multimodal (unstable across runs).
    pub multimodal_flags: Vec<String>,
}

/// Spread of a single parameter across resampled runs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParamSpread {
    /// Parameter name (e.g. `"fx"`, `"k1"`).
    pub name: String,
    /// Mean value across runs.
    pub mean: f64,
    /// Standard deviation across runs.
    pub std: f64,
    /// Minimum value observed.
    pub min: f64,
    /// Maximum value observed.
    pub max: f64,
    /// Coefficient of variation (`std / mean`).
    pub cv: f64,
    /// Detected modes, if the distribution looked multimodal.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub modes: Option<Vec<f64>>,
}

/// Detection coverage across cameras (Tier-B only).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Detection {
    /// Per-camera detection statistics.
    pub per_camera: Vec<DetectionStat>,
    /// Total features detected across all cameras.
    pub total_detected: usize,
    /// Total features expected across all cameras.
    pub total_expected: usize,
}

/// Detection statistics for a single camera.
///
/// Has `f64` fields (`coverage_pct`), so it is `PartialEq` but not `Eq`; that in
/// turn forces [`Detection`] to drop `Eq` as well.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DetectionStat {
    /// Camera identifier.
    pub camera_id: String,
    /// Total images available for this camera.
    pub images_total: usize,
    /// Images actually used (e.g. board found).
    pub images_used: usize,
    /// Features detected across used images.
    pub features_detected: usize,
    /// Features expected across used images.
    pub features_expected: usize,
    /// Detection coverage as a percentage (0–100).
    pub coverage_pct: f64,
    /// Detection wall-clock time for this camera (milliseconds).
    pub detect_ms: u64,
}

/// Laser-plane and extraction metrics across cameras.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LaserMetrics {
    /// Per-camera laser statistics.
    pub per_camera: Vec<LaserCamStat>,
    /// Total extracted laser pixels across cameras.
    pub total_points: usize,
    /// Total images with non-empty laser extraction.
    pub total_images_used: usize,
    /// Total extraction time in milliseconds.
    pub extract_ms: u64,
}

/// Laser metrics for a single camera.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LaserCamStat {
    /// Camera identifier.
    pub camera_id: String,
    /// Laser images available for this camera.
    pub images_total: usize,
    /// Laser images with at least one extracted point.
    pub images_used: usize,
    /// Extracted laser pixels.
    pub points_extracted: usize,
    /// Extraction time in milliseconds.
    pub extract_ms: u64,
    /// Plane residuals in metres, once a plane fit is available.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub plane_residual_m: Option<ReprojectionStats>,
    /// Laser-line residuals in pixels, once a pixel-space fit is available.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub line_residual_px: Option<ReprojectionStats>,
    /// Inlier ratio of laser points to the fitted plane (0–1), once available.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub inlier_ratio: Option<f64>,
}

/// Change in calibration parameters versus a frozen prior.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeltaToPrior {
    /// Per-parameter deltas.
    pub params: Vec<ParamDelta>,
}

/// Delta of a single parameter versus its prior value.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParamDelta {
    /// Parameter name.
    pub name: String,
    /// Current value.
    pub current: f64,
    /// Prior value.
    pub prior: f64,
    /// Absolute difference (`current - prior`).
    pub abs_delta: f64,
    /// Relative difference (`abs_delta / prior`).
    pub rel_delta: f64,
}

/// Wall-clock timing breakdown (all milliseconds).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Timing {
    /// Time spent in linear initialization.
    pub init_ms: u64,
    /// Time spent in nonlinear optimization.
    pub optimize_ms: u64,
    /// Total run time.
    pub total_ms: u64,
    /// Time spent in detection (0 for Tier-A).
    pub detection_ms: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_record() -> BenchRecord {
        let stats = ReprojectionStats::from_errors(&[0.1, 0.2, 0.3]);
        let hist = FeatureResidualHistogram::default();
        BenchRecord {
            ident: Ident {
                dataset_id: "puzzle_130x130".into(),
                problem: "rig_handeye".into(),
                tier: "a".into(),
                git_sha: "deadbeef".into(),
                timestamp_rfc3339: "2026-05-30T12:00:00Z".into(),
                config_hash: 0x1234_5678_9abc_def0,
                bench_schema_version: BENCH_SCHEMA_VERSION,
                features: vec!["tier-a".into()],
            },
            convergence: Convergence {
                init_ok: true,
                converged: true,
                report: SolveReport {
                    final_cost: 1.5,
                    num_iters: 12,
                },
            },
            fit: Fit {
                overall: stats,
                per_camera: vec![stats, stats],
                per_camera_hist: vec![hist.clone(), hist],
                reported_mean_reproj_px: 0.2,
                reported_per_cam_px: vec![0.18, 0.22],
            },
            generalization: Some(Generalization {
                folds: 5,
                heldout_rms_px: 0.31,
                heldout_rms_px_per_fold: vec![0.30, 0.32, 0.31, 0.29, 0.33],
                train_rms_px: 0.21,
            }),
            stability: Some(Stability {
                n_runs: 16,
                subset_frac: 0.7,
                seeds: vec![0, 1, 2],
                params: vec![ParamSpread {
                    name: "fx".into(),
                    mean: 1000.0,
                    std: 2.0,
                    min: 996.0,
                    max: 1004.0,
                    cv: 0.002,
                    modes: Some(vec![999.0, 1001.0]),
                }],
                multimodal_flags: vec!["fx".into()],
            }),
            detection: Some(Detection {
                per_camera: vec![DetectionStat {
                    camera_id: "cam0".into(),
                    images_total: 40,
                    images_used: 38,
                    features_detected: 6080,
                    features_expected: 6400,
                    coverage_pct: 95.0,
                    detect_ms: 1200,
                }],
                total_detected: 6080,
                total_expected: 6400,
            }),
            laser: Some(LaserMetrics {
                per_camera: vec![LaserCamStat {
                    camera_id: "cam0".into(),
                    images_total: 20,
                    images_used: 20,
                    points_extracted: 14_400,
                    extract_ms: 90,
                    plane_residual_m: Some(stats),
                    line_residual_px: Some(stats),
                    inlier_ratio: Some(0.98),
                }],
                total_points: 14_400,
                total_images_used: 20,
                extract_ms: 90,
            }),
            robot_corrections: Some(RobotCorrectionSummary {
                count: 2,
                mean_rot_deg: 0.1,
                max_rot_deg: 0.2,
                mean_trans_mm: 0.5,
                max_trans_mm: 0.8,
                prior_rot_deg: Some(0.5),
                prior_trans_mm: Some(1.0),
                max_rot_prior_ratio: Some(0.4),
                max_trans_prior_ratio: Some(0.8),
                exceeds_prior: false,
            }),
            artifacts: Some(CalibrationArtifacts {
                spatial_unit: "mm".into(),
                angle_unit: "deg".into(),
                cameras: vec![CameraArtifact {
                    camera_id: "cam0".into(),
                    camera_matrix_px: [[1000.0, 0.0, 500.0], [0.0, 1000.0, 400.0], [0.0, 0.0, 1.0]],
                    intrinsics_px: IntrinsicsArtifact {
                        fx: 1000.0,
                        fy: 1000.0,
                        cx: 500.0,
                        cy: 400.0,
                        skew: 0.0,
                    },
                    distortion_model: "brown_conrady5".into(),
                    distortion: DistortionArtifact {
                        k1: 0.1,
                        k2: -0.01,
                        k3: 0.0,
                        p1: 0.0,
                        p2: 0.0,
                    },
                    scheimpflug: None,
                }],
                transforms: vec![TransformArtifact {
                    name: "camera0_se3_target_view0".into(),
                    to_frame: "camera0".into(),
                    from_frame: "target/view_0".into(),
                    translation_mm: [0.0, 0.0, 500.0],
                    rotation_quat_xyzw: [0.0, 0.0, 0.0, 1.0],
                    rotation_rotvec_deg: [0.0, 0.0, 0.0],
                }],
            }),
            delta_to_prior: Some(DeltaToPrior {
                params: vec![ParamDelta {
                    name: "fx".into(),
                    current: 1001.0,
                    prior: 1000.0,
                    abs_delta: 1.0,
                    rel_delta: 0.001,
                }],
            }),
            timing: Timing {
                init_ms: 5,
                optimize_ms: 120,
                total_ms: 130,
                detection_ms: 1200,
            },
            reproj_report: None,
            residual_sidecar: None,
        }
    }

    /// `BenchRecord` cannot derive `PartialEq` (`ReprojectionStats` /
    /// `SolveReport` lack it), so we assert the roundtrip by comparing
    /// re-serialized JSON -- a strictly stronger check of the wire format than
    /// structural equality.
    fn assert_json_roundtrip(record: &BenchRecord) {
        let json = serde_json::to_string(record).expect("serialize");
        let back: BenchRecord = serde_json::from_str(&json).expect("deserialize");
        let json2 = serde_json::to_string(&back).expect("re-serialize");
        assert_eq!(json, json2);
    }

    #[test]
    fn bench_record_roundtrips() {
        assert_json_roundtrip(&sample_record());
    }

    #[test]
    fn v5_laser_joint_record_shape_is_present() {
        let record = sample_record();
        let laser = record.laser.as_ref().expect("sample carries laser metrics");
        let cam = &laser.per_camera[0];
        assert!(cam.plane_residual_m.is_some());
        assert!(cam.line_residual_px.is_some());
        assert!(record.robot_corrections.is_some());
        assert_eq!(record.fit.reported_per_cam_px.len(), 2);
    }

    #[test]
    fn bench_record_minimal_options_none() {
        let mut record = sample_record();
        record.generalization = None;
        record.stability = None;
        record.detection = None;
        record.laser = None;
        record.robot_corrections = None;
        record.artifacts = None;
        record.delta_to_prior = None;
        assert_json_roundtrip(&record);
    }

    #[test]
    fn compact_report_keeps_outliers_and_sidecar() {
        let residuals = vec![
            residual(0, 0, 0.1),
            residual(0, 1, 2.0),
            residual(1, 0, 1.0),
        ];
        let level = vision_calibration_pipeline::analysis::LevelReport {
            level: ReprojLevel::Intrinsic,
            overall: LevelStats::from_residuals(&residuals),
            per_camera: vec![LevelStats::from_residuals(&residuals[..2])],
            per_view: vec![LevelStats::from_residuals(&residuals)],
            residuals,
        };
        let report = ReprojReport {
            headline_px: level.overall.mean,
            levels: vec![level],
        };

        let (compact, sidecar) = compact_reproj_report_with_limit("ds", report, 2);
        assert_eq!(compact.levels[0].residual_count, 3);
        assert_eq!(compact.levels[0].top_outliers.len(), 2);
        assert_eq!(compact.levels[0].top_outliers[0].error_px, Some(2.0));
        assert_eq!(sidecar.dataset_id, "ds");
        assert_eq!(sidecar.levels[0].residuals.len(), 3);
    }

    #[test]
    fn compact_report_headline_matches_most_constrained_level() {
        let intrinsic_residuals = vec![residual(0, 0, 0.1), residual(0, 1, 0.2)];
        let handeye_residuals = vec![residual(0, 0, 1.1), residual(0, 1, 1.3)];
        let intrinsic = vision_calibration_pipeline::analysis::LevelReport {
            level: ReprojLevel::Intrinsic,
            overall: LevelStats::from_residuals(&intrinsic_residuals),
            per_camera: vec![LevelStats::from_residuals(&intrinsic_residuals)],
            per_view: vec![LevelStats::from_residuals(&intrinsic_residuals)],
            residuals: intrinsic_residuals,
        };
        let handeye = vision_calibration_pipeline::analysis::LevelReport {
            level: ReprojLevel::HandEye,
            overall: LevelStats::from_residuals(&handeye_residuals),
            per_camera: vec![LevelStats::from_residuals(&handeye_residuals)],
            per_view: vec![LevelStats::from_residuals(&handeye_residuals)],
            residuals: handeye_residuals,
        };
        let expected_headline = handeye.overall.mean;
        let report = ReprojReport {
            headline_px: expected_headline,
            levels: vec![intrinsic, handeye],
        };

        let (compact, _sidecar) = compact_reproj_report_with_limit("handeye", report, 4);
        assert_eq!(compact.headline_px, expected_headline);
        assert_eq!(
            compact.headline_px,
            compact.levels.last().expect("level").overall.mean
        );
        assert_eq!(compact.gaps.len(), 1);
        assert!(compact.gaps[0].ratio_to_intrinsic.unwrap() > 1.0);
    }

    #[test]
    fn robot_correction_summary_uses_degrees_and_mm() {
        let deltas = [
            [0.0, 0.0, 1.0_f64.to_radians(), 0.001, 0.0, 0.0],
            [0.0, 2.0_f64.to_radians(), 0.0, 0.0, 0.003, 0.004],
        ];
        let summary = RobotCorrectionSummary::from_deltas_with_priors(
            &deltas,
            Some(1.0_f64.to_radians()),
            Some(0.004),
        )
        .expect("summary");
        assert_eq!(summary.count, 2);
        assert!((summary.mean_rot_deg - 1.5).abs() < 1.0e-12);
        assert!((summary.max_rot_deg - 2.0).abs() < 1.0e-12);
        assert!((summary.mean_trans_mm - 3.0).abs() < 1.0e-12);
        assert!((summary.max_trans_mm - 5.0).abs() < 1.0e-12);
        assert!(summary.exceeds_prior);
        assert!((summary.max_rot_prior_ratio.unwrap() - 2.0).abs() < 1.0e-12);
        assert!((summary.max_trans_prior_ratio.unwrap() - 1.25).abs() < 1.0e-12);
    }

    fn residual(camera: usize, pose: usize, error_px: f64) -> TargetFeatureResidual {
        let mut residual = TargetFeatureResidual::default();
        residual.camera = camera;
        residual.pose = pose;
        residual.feature = 0;
        residual.observed_px = [0.0, 0.0];
        residual.projected_px = Some([error_px, 0.0]);
        residual.error_px = Some(error_px);
        residual
    }
}
