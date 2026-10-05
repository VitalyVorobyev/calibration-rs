//! Solver benchmark: performance and quality of the non-linear solve across
//! all problem types on synthetic ground-truth scenes.
//!
//! - [`scenes`]: the deterministic scene matrix (problem × sensor × scale ×
//!   pixel noise × outliers) and its generators. Pixel noise is uniform with
//!   the stated per-axis standard deviation; contaminated scenes displace 5 %
//!   of the target observations by 10–30 px and solve under Huber or Cauchy
//!   (set through `solver.robust_loss`, or the problem's `calib_loss` for the
//!   laser-carrying problems; their laser loss keeps the problem default).
//!   The frozen-rig laser problem fits laser planes only, so it has no
//!   contaminated scenes.
//! - [`drivers`]: runs a scene through the public step functions, timing the
//!   init and optimize phases separately.
//! - [`metrics`]: solver-independent quality: the objective `½ Σ ρ(e²)`,
//!   reprojection RMS and ground-truth parameter errors, all computed from
//!   the pipeline export.
//! - [`record`]: the serializable report.
//!
//! Two solves of one scene agree to roughly 1e-12 relative, not bit for bit:
//! the backend orders its parameter blocks through a randomly seeded hash map.

pub mod drivers;
pub mod metrics;
pub mod record;
pub mod scenes;

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::panic::{AssertUnwindSafe, catch_unwind};

use metrics::{GtErrors, QualityMetrics};
use record::{RunStatus, SOLVER_SCHEMA_VERSION, SolverBenchReport, SolverRunRecord};
use scenes::{Preset, Problem, Scene, SceneSpec, scene_specs};

/// Default number of timed repeats per scene.
pub const DEFAULT_REPEATS: usize = 3;

/// Generate, solve and score one scene. Never fails: errors (including
/// panics inside a solver) become an `Error` record.
pub fn run_scene(spec: &SceneSpec, repeats: usize) -> SolverRunRecord {
    let attempt = catch_unwind(AssertUnwindSafe(|| -> anyhow::Result<SolverRunRecord> {
        let scene = Scene::build(spec)?;
        let measured = drivers::measure(&scene, repeats)?;
        let metrics = QualityMetrics::evaluate(
            &measured.residuals,
            &scene.outliers,
            spec.loss,
            &measured.params,
            &scene.truth,
        );
        Ok(SolverRunRecord {
            scene: spec.clone(),
            timing: Some(measured.timing),
            solve_report: measured.report,
            metrics: Some(metrics),
            status: RunStatus::Ok,
        })
    }));
    match attempt {
        Ok(Ok(record)) => record,
        Ok(Err(e)) => SolverRunRecord::failed(spec.clone(), format!("{e:#}")),
        Err(_) => SolverRunRecord::failed(spec.clone(), "solver panicked"),
    }
}

/// Run a preset. `filter` restricts the run to the listed problems (empty =
/// all). Scenes run sequentially in preset order.
pub fn run(preset: Preset, repeats: usize, filter: &[Problem]) -> SolverBenchReport {
    let records = scene_specs(preset)
        .into_iter()
        .filter(|s| filter.is_empty() || filter.contains(&s.problem))
        .map(|spec| run_scene(&spec, repeats))
        .collect();
    SolverBenchReport {
        schema_version: SOLVER_SCHEMA_VERSION,
        git_sha: crate::record::git_sha(),
        timestamp_unix_secs: crate::record::unix_epoch_secs_string(),
        preset: preset.to_string(),
        repeats,
        records,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Markdown rendering
// ─────────────────────────────────────────────────────────────────────────────

/// The scene id without its leading `problem/` component.
fn scene_label(spec: &SceneSpec) -> String {
    let id = spec.id();
    id.split_once('/')
        .map_or(id.clone(), |(_, rest)| rest.to_string())
}

fn gt_summary(gt: &GtErrors) -> String {
    let mut parts = Vec::new();
    if let Some(v) = gt.focal_rel {
        parts.push(format!("f {:.3}%", v * 100.0));
    }
    if let Some(v) = gt.principal_point_px {
        parts.push(format!("pp {v:.3}px"));
    }
    if let Some(v) = gt.distortion_abs {
        parts.push(format!("dist {v:.1e}"));
    }
    if let Some(v) = gt.sensor_tilt_deg {
        parts.push(format!("tilt {v:.3}°"));
    }
    if let (Some(r), Some(t)) = (gt.rig_rot_deg, gt.rig_trans_mm) {
        parts.push(format!("rig {r:.3}°/{t:.2}mm"));
    }
    if let (Some(r), Some(t)) = (gt.handeye_rot_deg, gt.handeye_trans_mm) {
        parts.push(format!("he {r:.3}°/{t:.2}mm"));
    }
    if let (Some(n), Some(d)) = (gt.laser_normal_deg, gt.laser_distance_mm) {
        parts.push(format!("laser {n:.3}°/{d:.2}mm"));
    }
    parts.join(", ")
}

fn opt(v: Option<f64>, f: impl Fn(f64) -> String) -> String {
    v.map_or_else(|| "-".to_string(), f)
}

/// Render a report as Markdown: one table per problem.
pub fn render_markdown(report: &SolverBenchReport) -> String {
    let mut out = String::new();
    let _ = writeln!(
        out,
        "# Solver benchmark ({} preset, {} repeats, {})\n",
        report.preset,
        report.repeats,
        &report.git_sha[..report.git_sha.len().min(10)]
    );
    for problem in Problem::ALL {
        let rows: Vec<_> = report
            .records
            .iter()
            .filter(|r| r.scene.problem == problem)
            .collect();
        if rows.is_empty() {
            continue;
        }
        let _ = writeln!(out, "## {problem}\n");
        let _ = writeln!(
            out,
            "| scene | init ms | optimize ms | iters | objective | inlier RMS px | GT errors |"
        );
        let _ = writeln!(out, "|---|---:|---:|---:|---:|---:|---|");
        for r in rows {
            let label = scene_label(&r.scene);
            match (&r.status, &r.timing, &r.metrics) {
                (RunStatus::Ok, Some(t), Some(m)) => {
                    let _ = writeln!(
                        out,
                        "| {label} | {} | {:.1} | {} | {:.4e} | {:.4} | {} |",
                        opt(t.init_ms, |v| format!("{v:.1}")),
                        t.optimize_ms,
                        r.solve_report
                            .as_ref()
                            .map_or("-".to_string(), |s| s.num_iters.to_string()),
                        m.objective,
                        m.inlier_rms_px,
                        gt_summary(&m.gt),
                    );
                }
                (status, _, _) => {
                    let msg = match status {
                        RunStatus::Error(e) => e.replace('|', "/").replace('\n', " "),
                        RunStatus::Ok => "missing data".to_string(),
                    };
                    let _ = writeln!(out, "| {label} | | | | | | ERROR: {msg} |");
                }
            }
        }
        out.push('\n');
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// Comparison
// ─────────────────────────────────────────────────────────────────────────────

/// Optimize-time ratio (`b / a`) above which a scene is flagged slower.
pub const TIME_RATIO_REGRESSION: f64 = 1.25;
/// Optimize times below this (ms) are too noisy to flag.
pub const TIME_FLOOR_MS: f64 = 2.0;
/// Relative objective increase that is flagged.
pub const OBJECTIVE_REL_REGRESSION: f64 = 1e-3;
/// Inlier-RMS increase (pixels) that is flagged.
pub const INLIER_RMS_REGRESSION_PX: f64 = 5e-3;
/// A ground-truth error grows by more than this fraction (plus
/// [`GT_ABS_FLOOR`]) to be flagged.
pub const GT_REL_REGRESSION: f64 = 0.10;
/// Absolute slack for ground-truth error growth.
pub const GT_ABS_FLOOR: f64 = 1e-5;

/// Comparison of one scene present in both reports.
#[derive(Debug, Clone)]
pub struct SceneComparison {
    /// Scene id.
    pub id: String,
    /// `b.optimize_ms / a.optimize_ms`.
    pub optimize_ratio: Option<f64>,
    /// `(b.objective - a.objective) / |a.objective|`.
    pub objective_rel_delta: Option<f64>,
    /// `b.inlier_rms - a.inlier_rms`, pixels.
    pub inlier_rms_delta: Option<f64>,
    /// `(name, a, b)` for every ground-truth error present in both.
    pub gt: Vec<(&'static str, f64, f64)>,
    /// Human-readable regression flags (empty when none).
    pub regressions: Vec<String>,
}

/// Result of comparing two reports.
#[derive(Debug, Clone, Default)]
pub struct Comparison {
    /// Scenes present in both, in `a` order.
    pub matched: Vec<SceneComparison>,
    /// Scene ids only in `a`.
    pub only_in_a: Vec<String>,
    /// Scene ids only in `b`.
    pub only_in_b: Vec<String>,
}

impl Comparison {
    /// Number of flagged regressions across all scenes.
    pub fn regression_count(&self) -> usize {
        self.matched.iter().map(|s| s.regressions.len()).sum()
    }
}

/// Join two reports by scene id and compare `b` against the baseline `a`.
pub fn compare(a: &SolverBenchReport, b: &SolverBenchReport) -> Comparison {
    let index_b: BTreeMap<String, &SolverRunRecord> =
        b.records.iter().map(|r| (r.scene.id(), r)).collect();
    let ids_a: std::collections::BTreeSet<String> =
        a.records.iter().map(|r| r.scene.id()).collect();
    let mut out = Comparison::default();
    for ra in &a.records {
        let id = ra.scene.id();
        match index_b.get(&id) {
            Some(rb) => out.matched.push(compare_scene(id, ra, rb)),
            None => out.only_in_a.push(id),
        }
    }
    out.only_in_b = index_b
        .keys()
        .filter(|id| !ids_a.contains(*id))
        .cloned()
        .collect();
    out
}

fn compare_scene(id: String, a: &SolverRunRecord, b: &SolverRunRecord) -> SceneComparison {
    let mut c = SceneComparison {
        id,
        optimize_ratio: None,
        objective_rel_delta: None,
        inlier_rms_delta: None,
        gt: Vec::new(),
        regressions: Vec::new(),
    };
    match (&a.status, &b.status) {
        (RunStatus::Ok, RunStatus::Error(e)) => {
            c.regressions.push(format!("now fails: {e}"));
            return c;
        }
        (RunStatus::Error(_), RunStatus::Ok) => return c,
        (RunStatus::Error(_), RunStatus::Error(_)) => return c,
        (RunStatus::Ok, RunStatus::Ok) => {}
    }
    if let (Some(ta), Some(tb)) = (&a.timing, &b.timing) {
        let ratio = tb.optimize_ms / ta.optimize_ms;
        c.optimize_ratio = Some(ratio);
        if ratio > TIME_RATIO_REGRESSION && tb.optimize_ms > TIME_FLOOR_MS {
            c.regressions.push(format!("optimize {ratio:.2}x slower"));
        }
    }
    if let (Some(ma), Some(mb)) = (&a.metrics, &b.metrics) {
        let rel = (mb.objective - ma.objective) / ma.objective.abs().max(f64::MIN_POSITIVE);
        c.objective_rel_delta = Some(rel);
        if rel > OBJECTIVE_REL_REGRESSION {
            c.regressions
                .push(format!("objective +{:.2}%", rel * 100.0));
        }
        let d = mb.inlier_rms_px - ma.inlier_rms_px;
        c.inlier_rms_delta = Some(d);
        if d > INLIER_RMS_REGRESSION_PX {
            c.regressions.push(format!("inlier RMS +{d:.4}px"));
        }
        let gt_b: BTreeMap<_, _> = mb.gt.entries().into_iter().collect();
        for (name, va) in ma.gt.entries() {
            if let Some(&vb) = gt_b.get(name) {
                c.gt.push((name, va, vb));
                if vb - va > GT_REL_REGRESSION * va + GT_ABS_FLOOR {
                    c.regressions.push(format!("{name} {va:.3e} -> {vb:.3e}"));
                }
            }
        }
    }
    c
}

/// Render a comparison as Markdown.
pub fn render_comparison(cmp: &Comparison) -> String {
    let mut out = String::from("# Solver benchmark comparison (b vs a)\n\n");
    let _ = writeln!(
        out,
        "| scene | optimize x | Δ objective | Δ inlier RMS px | GT errors a -> b | flags |"
    );
    let _ = writeln!(out, "|---|---:|---:|---:|---|---|");
    for s in &cmp.matched {
        let gt =
            s.gt.iter()
                .map(|(n, a, b)| format!("{n} {a:.2e}->{b:.2e}"))
                .collect::<Vec<_>>()
                .join(", ");
        let _ = writeln!(
            out,
            "| {} | {} | {} | {} | {} | {} |",
            s.id,
            opt(s.optimize_ratio, |v| format!("{v:.2}")),
            opt(s.objective_rel_delta, |v| format!("{:+.3e}", v)),
            opt(s.inlier_rms_delta, |v| format!("{v:+.4}")),
            gt,
            if s.regressions.is_empty() {
                "ok".to_string()
            } else {
                format!("REGRESSION: {}", s.regressions.join("; "))
            },
        );
    }
    for id in &cmp.only_in_a {
        let _ = writeln!(out, "\nonly in a: {id}");
    }
    for id in &cmp.only_in_b {
        let _ = writeln!(out, "\nonly in b: {id}");
    }
    let _ = writeln!(
        out,
        "\n{} scene(s) compared, {} regression flag(s).",
        cmp.matched.len(),
        cmp.regression_count()
    );
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use metrics::QualityMetrics;
    use record::{RunStatus, SolverTiming, TimingSample};
    use scenes::{OutlierSpec, Scale, SensorKind};
    use vision_calibration::optim::RobustLoss;

    fn record(optimize_ms: f64, objective: f64, focal_rel: f64) -> SolverRunRecord {
        let gt = GtErrors {
            focal_rel: Some(focal_rel),
            ..GtErrors::default()
        };
        SolverRunRecord {
            scene: SceneSpec {
                problem: Problem::PlanarIntrinsics,
                sensor: SensorKind::Pinhole,
                scale: Scale::Small,
                noise_px: 0.1,
                outliers: OutlierSpec::None,
                loss: RobustLoss::None,
            },
            timing: SolverTiming::from_samples(vec![TimingSample {
                init_ms: Some(1.0),
                optimize_ms,
            }]),
            solve_report: None,
            metrics: Some(QualityMetrics {
                objective,
                inlier_rms_px: 0.1,
                all_rms_px: 0.1,
                laser_rms_px: None,
                num_features: 10,
                num_outliers: 0,
                gt,
            }),
            status: RunStatus::Ok,
        }
    }

    fn report(records: Vec<SolverRunRecord>) -> SolverBenchReport {
        SolverBenchReport {
            schema_version: SOLVER_SCHEMA_VERSION,
            git_sha: "x".into(),
            timestamp_unix_secs: "0".into(),
            preset: "quick".into(),
            repeats: 1,
            records,
        }
    }

    #[test]
    fn report_round_trips_through_json() {
        let r = report(vec![record(10.0, 5.0, 1e-3)]);
        let json = serde_json::to_string(&r).unwrap();
        let back: SolverBenchReport = serde_json::from_str(&json).unwrap();
        assert_eq!(back.records.len(), 1);
        assert_eq!(back.records[0].scene, r.records[0].scene);
        assert_eq!(back.records[0].metrics, r.records[0].metrics);
        let failed = SolverRunRecord::failed(r.records[0].scene.clone(), "boom");
        let json = serde_json::to_string(&failed).unwrap();
        let back: SolverRunRecord = serde_json::from_str(&json).unwrap();
        assert_eq!(back.status, RunStatus::Error("boom".into()));
    }

    #[test]
    fn comparing_a_report_with_itself_flags_nothing() {
        let r = report(vec![record(10.0, 5.0, 1e-3)]);
        let cmp = compare(&r, &r);
        assert_eq!(cmp.matched.len(), 1);
        assert_eq!(cmp.regression_count(), 0);
        assert_eq!(cmp.matched[0].optimize_ratio, Some(1.0));
    }

    #[test]
    fn comparison_flags_slowdowns_and_quality_loss() {
        let a = report(vec![record(10.0, 5.0, 1e-3)]);
        let b = report(vec![record(20.0, 5.1, 2e-3)]);
        let cmp = compare(&a, &b);
        let flags = cmp.matched[0].regressions.join(" | ");
        assert!(flags.contains("slower"), "{flags}");
        assert!(flags.contains("objective"), "{flags}");
        assert!(flags.contains("focal_rel"), "{flags}");
        // An improvement flags nothing.
        assert_eq!(compare(&b, &a).regression_count(), 0);
        // A scene that stops solving is a regression; unmatched scenes are listed.
        let broken = report(vec![SolverRunRecord::failed(
            a.records[0].scene.clone(),
            "x",
        )]);
        assert_eq!(compare(&a, &broken).regression_count(), 1);
        assert_eq!(compare(&a, &report(vec![])).only_in_a.len(), 1);
        assert!(render_comparison(&cmp).contains("REGRESSION"));
    }

    #[test]
    fn markdown_has_one_table_per_problem() {
        let md = render_markdown(&report(vec![record(10.0, 5.0, 1e-3)]));
        assert!(md.contains("## planar_intrinsics"));
        assert!(md.contains("| pinhole/small/n0.1/clean |"));
        assert!(!md.contains("## rig_extrinsics"));
    }
}
