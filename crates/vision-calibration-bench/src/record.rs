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
use vision_calibration_core::{FeatureResidualHistogram, ReprojectionStats};
use vision_calibration_optim::SolveReport;
use vision_calibration_pipeline::analysis::ReprojReport;

/// Schema version for [`BenchRecord`]. Bump on any breaking layout change so
/// frozen records carry a version stamp ([`Ident::bench_schema_version`]).
///
/// v2 adds [`BenchRecord::reproj_report`] (the hierarchical multi-level
/// reprojection report: intrinsic floor → rig-extrinsic → hand-eye).
pub const BENCH_SCHEMA_VERSION: u32 = 2;

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
    /// Change versus a frozen prior calibration, if a prior was supplied.
    pub delta_to_prior: Option<DeltaToPrior>,
    /// Wall-clock timing breakdown.
    pub timing: Timing,
    /// Hierarchical multi-level reprojection report (intrinsic floor →
    /// rig-extrinsic → hand-eye), if computed. The level deltas localize the
    /// error source (detection / camera model vs. rig vs. robot chain).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reproj_report: Option<ReprojReport>,
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

/// Laser-plane metrics across cameras.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LaserMetrics {
    /// Per-camera laser statistics.
    pub per_camera: Vec<LaserCamStat>,
}

/// Laser metrics for a single camera.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LaserCamStat {
    /// Plane residuals in metres (reusing the reprojection stats container).
    pub plane_residual_m: ReprojectionStats,
    /// Plane residuals in pixels.
    pub plane_residual_px: ReprojectionStats,
    /// Inlier ratio of laser points to the fitted plane (0–1).
    pub inlier_ratio: f64,
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
                    plane_residual_m: stats,
                    plane_residual_px: stats,
                    inlier_ratio: 0.98,
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
    fn bench_record_minimal_options_none() {
        let mut record = sample_record();
        record.generalization = None;
        record.stability = None;
        record.detection = None;
        record.laser = None;
        record.delta_to_prior = None;
        assert_json_roundtrip(&record);
    }
}
