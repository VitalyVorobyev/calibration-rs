//! Committed regression baselines for the acceptance suite.
//!
//! A baseline is a slim, committed snapshot of a dataset's accepted fit
//! (`baselines/<dataset_id>.json`). `calib-bench accept` compares every
//! fresh run against its baseline when one exists and fails on drift
//! beyond the entry's stated tolerance — so algorithm changes that move
//! any dataset's fit are caught at acceptance time, not discovered later.
//!
//! Baselines hold only reprojection statistics (no images, no residual
//! dumps), so private-dataset baselines are safe to commit; they carry the
//! same information a run summary prints.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use vision_calibration_core::ReprojectionStats;

use crate::record::Fit;

/// Slim committed snapshot of an accepted fit.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BaselineFit {
    /// Dataset id this baseline belongs to.
    pub dataset_id: String,
    /// Git SHA of the run that froze this baseline.
    pub git_sha: String,
    /// Unix epoch seconds when frozen (string, provenance only).
    pub frozen_at: String,
    /// Bench-recomputed overall reprojection statistics.
    pub overall: ReprojectionStats,
    /// Per-camera reprojection statistics.
    pub per_camera: Vec<ReprojectionStats>,
}

impl BaselineFit {
    /// Extract the baseline-relevant slice of an accepted fit.
    pub fn new(dataset_id: &str, git_sha: &str, frozen_at: &str, fit: &Fit) -> Self {
        Self {
            dataset_id: dataset_id.to_string(),
            git_sha: git_sha.to_string(),
            frozen_at: frozen_at.to_string(),
            overall: fit.overall,
            per_camera: fit.per_camera.clone(),
        }
    }
}

/// Path of a dataset's baseline file inside `dir`.
pub fn baseline_path(dir: &Path, dataset_id: &str) -> PathBuf {
    dir.join(format!("{dataset_id}.json"))
}

/// Load a baseline if one exists. `Ok(None)` when the file is absent.
pub fn load_baseline(dir: &Path, dataset_id: &str) -> Result<Option<BaselineFit>> {
    let path = baseline_path(dir, dataset_id);
    if !path.is_file() {
        return Ok(None);
    }
    let raw = std::fs::read_to_string(&path)
        .with_context(|| format!("read baseline {}", path.display()))?;
    let baseline: BaselineFit =
        serde_json::from_str(&raw).with_context(|| format!("parse baseline {}", path.display()))?;
    Ok(Some(baseline))
}

/// Write (or overwrite) a dataset's baseline.
pub fn save_baseline(dir: &Path, baseline: &BaselineFit) -> Result<PathBuf> {
    std::fs::create_dir_all(dir).with_context(|| format!("create {}", dir.display()))?;
    let path = baseline_path(dir, &baseline.dataset_id);
    std::fs::write(&path, serde_json::to_string_pretty(baseline)?)
        .with_context(|| format!("write baseline {}", path.display()))?;
    Ok(path)
}

/// Compare a fresh record against a committed baseline.
///
/// Returns the list of regressions (empty = within tolerance).
/// Improvements never fail — refreezing a better baseline is a manual,
/// reviewed act (`accept --freeze-baselines`). Drift is judged on the
/// overall mean + RMS and every per-camera mean, all with the same
/// relative tolerance; a per-camera count mismatch is structural drift
/// and always flagged.
pub fn compare_to_baseline(fit: &Fit, baseline: &BaselineFit, rel_tol: f64) -> Vec<String> {
    let mut regressions = Vec::new();
    let worse = |current: f64, frozen: f64| current > frozen * (1.0 + rel_tol) + 1e-12;

    if worse(fit.overall.mean, baseline.overall.mean) {
        regressions.push(format!(
            "overall mean {:.6} px vs baseline {:.6} px (tol {:.1}%)",
            fit.overall.mean,
            baseline.overall.mean,
            rel_tol * 100.0
        ));
    }
    if worse(fit.overall.rms, baseline.overall.rms) {
        regressions.push(format!(
            "overall rms {:.6} px vs baseline {:.6} px (tol {:.1}%)",
            fit.overall.rms,
            baseline.overall.rms,
            rel_tol * 100.0
        ));
    }
    // Coverage is structural: the pipeline is deterministic, so any change
    // in residual counts (views/corners gained or lost) means the detection
    // or filtering behavior changed — even if the surviving residuals look
    // *better*. That must be accepted explicitly via a refreeze, never
    // silently.
    if fit.overall.count != baseline.overall.count {
        regressions.push(format!(
            "overall residual count changed: {} vs baseline {}",
            fit.overall.count, baseline.overall.count
        ));
    }
    if fit.per_camera.len() != baseline.per_camera.len() {
        regressions.push(format!(
            "per-camera count changed: {} vs baseline {}",
            fit.per_camera.len(),
            baseline.per_camera.len()
        ));
        return regressions;
    }
    for (i, (current, frozen)) in fit.per_camera.iter().zip(&baseline.per_camera).enumerate() {
        if current.count != frozen.count {
            regressions.push(format!(
                "camera {i} residual count changed: {} vs baseline {}",
                current.count, frozen.count
            ));
        }
        if worse(current.mean, frozen.mean) {
            regressions.push(format!(
                "camera {i} mean {:.6} px vs baseline {:.6} px (tol {:.1}%)",
                current.mean,
                frozen.mean,
                rel_tol * 100.0
            ));
        }
    }
    regressions
}

#[cfg(test)]
mod tests {
    use super::*;
    use vision_calibration_core::ReprojectionStats;

    fn stats(mean: f64) -> ReprojectionStats {
        ReprojectionStats::from_summary(mean, mean * 1.1, mean * 3.0, 1000)
    }

    fn fit_with_mean(mean: f64) -> Fit {
        Fit {
            overall: stats(mean),
            per_camera: vec![stats(mean)],
            per_camera_hist: vec![],
            reported_mean_reproj_px: mean,
            reported_per_cam_px: vec![mean],
        }
    }

    #[test]
    fn roundtrip_and_compare() {
        let fit = fit_with_mean(0.25);
        let baseline = BaselineFit::new("ds", "abc", "0", &fit);
        let json = serde_json::to_string(&baseline).unwrap();
        let back: BaselineFit = serde_json::from_str(&json).unwrap();
        assert_eq!(back.dataset_id, "ds");

        // Same fit: no regressions. Improvement: no regressions.
        assert!(compare_to_baseline(&fit, &back, 0.05).is_empty());
        assert!(compare_to_baseline(&fit_with_mean(0.20), &back, 0.05).is_empty());

        // 10% worse trips a 5% tolerance on mean, rms, and camera 0.
        let regs = compare_to_baseline(&fit_with_mean(0.275), &back, 0.05);
        assert_eq!(regs.len(), 3, "{regs:?}");

        // Within tolerance: 4% worse passes at 5%.
        assert!(compare_to_baseline(&fit_with_mean(0.26), &back, 0.05).is_empty());
    }

    #[test]
    fn dropped_residuals_are_structural_drift_even_when_stats_improve() {
        // A "better" fit with fewer residuals (dropped hard views/corners)
        // must fail: coverage changed.
        let fit = fit_with_mean(0.25);
        let baseline = BaselineFit::new("ds", "abc", "0", &fit);
        let mut shrunk = fit_with_mean(0.20); // improved stats...
        shrunk.overall = ReprojectionStats::from_summary(0.20, 0.22, 0.6, 900); // ...fewer residuals
        shrunk.per_camera = vec![ReprojectionStats::from_summary(0.20, 0.22, 0.6, 900)];
        let regs = compare_to_baseline(&shrunk, &baseline, 0.05);
        assert_eq!(regs.len(), 2, "{regs:?}"); // overall + camera 0 counts
        assert!(regs.iter().all(|r| r.contains("residual count changed")));
    }

    #[test]
    fn per_camera_count_change_is_structural_drift() {
        let fit = fit_with_mean(0.25);
        let mut baseline = BaselineFit::new("ds", "abc", "0", &fit);
        baseline.per_camera.push(stats(0.25));
        let regs = compare_to_baseline(&fit, &baseline, 0.05);
        assert_eq!(regs.len(), 1);
        assert!(regs[0].contains("per-camera count changed"));
    }

    #[test]
    fn save_load_roundtrip() {
        let dir = std::env::temp_dir().join(format!("baseline-test-{}", std::process::id()));
        let baseline = BaselineFit::new("ds8", "abc", "0", &fit_with_mean(0.3));
        let path = save_baseline(&dir, &baseline).unwrap();
        assert!(path.ends_with("ds8.json"));
        let loaded = load_baseline(&dir, "ds8").unwrap().unwrap();
        assert_eq!(loaded.overall.count, 1000);
        assert!(load_baseline(&dir, "missing").unwrap().is_none());
        std::fs::remove_dir_all(&dir).ok();
    }
}
