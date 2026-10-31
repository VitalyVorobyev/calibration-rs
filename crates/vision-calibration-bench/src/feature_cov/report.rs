//! The study's files: one JSON per camera and per downstream dataset, the
//! summary JSON and a Markdown report, under `target/feature-cov/`.
//!
//! A private dataset's files hold its aggregates only; per-view and
//! per-split values are written for public datasets alone.

use std::fmt::Write as _;
use std::path::Path;

use anyhow::{Context, Result};
use serde::Serialize;
use vision_calibration_core::NoiseLevel;

use super::arms::Arm;
use super::downstream::Downstream;
use super::planar::{UnitRaw, UnitSummary};
use super::rule::{Decision, Pooled, PooledHonesty};
use super::stats::{Class, Interval};
use super::timing::Timing;
use super::{Options, Study};

fn write_json(path: &Path, value: &impl Serialize) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    }
    std::fs::write(path, serde_json::to_string_pretty(value)?)
        .with_context(|| format!("write {}", path.display()))
}

fn file_name(label: &str) -> String {
    label.replace('/', "__")
}

#[derive(Serialize)]
struct UnitFile<'a> {
    summary: &'a UnitSummary,
    /// Per-view and per-split values; public datasets only.
    #[serde(skip_serializing_if = "Option::is_none")]
    raw: Option<&'a UnitRaw>,
}

/// Write one camera's file.
pub fn write_unit(out: &Path, summary: &UnitSummary, raw: &UnitRaw) -> Result<()> {
    let file = UnitFile {
        summary,
        raw: (!summary.private).then_some(raw),
    };
    write_json(
        &out.join("units")
            .join(format!("{}.json", file_name(&summary.unit))),
        &file,
    )
}

/// Write one dataset's downstream file.
pub fn write_downstream(out: &Path, result: &Downstream) -> Result<()> {
    write_json(
        &out.join("downstream")
            .join(format!("{}.json", result.dataset)),
        result,
    )
}

#[derive(Serialize)]
struct Protocol {
    random_splits: usize,
    resamples: usize,
    timing_images: Option<usize>,
    is_protocol: bool,
    git_sha: String,
}

#[derive(Serialize)]
struct Summary<'a> {
    protocol: Protocol,
    unavailable: &'a [String],
    units: Vec<&'a UnitSummary>,
    downstream: &'a [Downstream],
    timing: Option<&'a Timing>,
    decision: &'a Decision,
}

/// Write `summary.json` and `report.md`.
pub fn write_summary(
    out: &Path,
    study: &Study,
    timing: Option<&Timing>,
    decision: &Decision,
    options: &Options,
) -> Result<()> {
    let summary = Summary {
        protocol: Protocol {
            random_splits: options.random_splits,
            resamples: options.resamples,
            timing_images: options.timing_images,
            is_protocol: options.is_protocol(),
            git_sha: crate::record::git_sha(),
        },
        unavailable: &study.unavailable,
        units: study.units.iter().map(|(s, _)| s).collect(),
        downstream: &study.downstream,
        timing,
        decision,
    };
    write_json(&out.join("summary.json"), &summary)?;
    let md = markdown(study, timing, decision, options);
    std::fs::write(out.join("report.md"), &md).context("write report.md")?;
    Ok(())
}

fn px(v: f64) -> String {
    if v.is_finite() {
        format!("{v:.4}")
    } else {
        "–".into()
    }
}

fn plain(i: Option<Interval>) -> String {
    i.map_or("–".into(), |i| {
        format!("{:.3} [{:.3}, {:.3}]", i.point, i.lo, i.hi)
    })
}

fn pct(v: Option<f64>) -> String {
    v.map_or("–".into(), |v| format!("{:.1} %", 100.0 * v))
}

fn noise(level: Option<NoiseLevel>) -> String {
    match level {
        Some(NoiseLevel::Sigma(s)) => format!("σ̂ {s:.4} px"),
        Some(NoiseLevel::Sigma0(s)) => format!("σ̂₀ {s:.4}"),
        _ => "–".into(),
    }
}

fn pair(v: Option<[f64; 2]>) -> String {
    v.map_or("–".into(), |[d, mm]| format!("{d:.4}° / {mm:.3} mm"))
}

fn markdown(
    study: &Study,
    timing: Option<&Timing>,
    decision: &Decision,
    options: &Options,
) -> String {
    let mut s = String::new();
    let w = &mut s;
    let _ = writeln!(w, "# Feature covariance study — harness report\n");
    let _ = writeln!(
        w,
        "Splits: even/odd + {} random. Bootstrap: {} resamples. {}",
        options.random_splits,
        options.resamples,
        if options.is_protocol() {
            "Protocol settings."
        } else {
            "**Not the protocol settings (smoke run).**"
        }
    );
    let _ = writeln!(w, "{METHODS}");
    if !study.unavailable.is_empty() {
        let _ = writeln!(w, "\nUnavailable: {}.", study.unavailable.join(", "));
    }

    let _ = writeln!(w, "\n## Cameras\n");
    for (u, _) in &study.units {
        let _ = writeln!(
            w,
            "### {}{}\n\n{} views, {} corners, {}×{} px. Power check: {}.{}\n",
            u.unit,
            if u.private {
                " (private, aggregates)"
            } else {
                ""
            },
            u.views,
            u.corners,
            u.image_size[0],
            u.image_size[1],
            if u.decisive {
                "decisive"
            } else {
                "**not decisive**"
            },
            declined(&u.declined)
        );
        let _ = writeln!(
            w,
            "Decisive grid: {} of {} nodes inside the hull of the corners.\n",
            u.hull_nodes[0], u.hull_nodes[1]
        );
        let _ = writeln!(
            w,
            "| arm | held-out RMS | E1 held-out | E1 homography | E2 split RMS | E2 effect | E3 r | ≤2.30 | ≤6.18 | full grid: split RMS, r, ≤2.30, ≤6.18 | full fit | failures |"
        );
        let _ = writeln!(w, "|---|---|---|---|---|---|---|---|---|---|---|---|");
        for (arm, a) in &u.arms {
            let h = a.honesty;
            let _ = writeln!(
                w,
                "| {arm} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |",
                px(a.heldout_rms_px),
                effect(a.heldout_effect.map(|e| (e.interval, e.margin, e.class))),
                effect(a.homography_effect.map(|e| (e.interval, e.margin, e.class))),
                px(a.split_rms_px),
                effect(a.split_effect.map(|e| (e.interval, e.margin, e.class))),
                plain(h.map(|h| h.ratio)),
                pct_interval(h.map(|h| h.coverage_68)),
                pct_interval(h.map(|h| h.coverage_95)),
                full_grid(&a.full_grid),
                noise(a.noise),
                a.failures
            );
        }
        let _ = writeln!(w);
    }

    let _ = writeln!(w, "## Pooled over decisive cameras\n");
    let _ = writeln!(
        w,
        "Decisive: {}.\n",
        if decision.decisive.is_empty() {
            "none".into()
        } else {
            decision.decisive.join(", ")
        }
    );
    let _ = writeln!(
        w,
        "| arm | cameras | E1 held-out | E1 homography | E2 | E3 r | \\|r − 1\\| | ≤2.30 | ≤6.18 | honest | uniform: r, \\|r − 1\\|, ≤2.30, ≤6.18, honest |"
    );
    let _ = writeln!(w, "|---|---|---|---|---|---|---|---|---|---|---|");
    for (arm, v) in &decision.arms {
        if let Some(p) = &v.pooled {
            pooled_row(w, *arm, p);
        }
    }

    let _ = writeln!(w, "\n## Downstream (E4, medians over splits)\n");
    for d in &study.downstream {
        let _ = writeln!(
            w,
            "### {} ({} views){}\n",
            d.dataset,
            d.views,
            declined(&d.declined)
        );
        let _ = writeln!(
            w,
            "| arm | cam_se3_rig per camera | hand-eye | stated σ (full) | predicted (halves) | splits | failures |"
        );
        let _ = writeln!(w, "|---|---|---|---|---|---|---|");
        for (arm, a) in &d.arms {
            let cams = a
                .cam_se3_rig
                .iter()
                .map(|(c, v)| format!("cam{c}: {}", pair(Some(*v))))
                .collect::<Vec<_>>()
                .join("; ");
            let _ = writeln!(
                w,
                "| {arm} | {} | {} | {} | {} | {} | {} |",
                if cams.is_empty() { "–".into() } else { cams },
                pair(a.handeye),
                pair(a.handeye_sigma),
                pair(a.handeye_predicted),
                a.splits_used,
                a.failures
            );
        }
        let _ = writeln!(w);
    }

    let _ = writeln!(w, "## Detection cost (E5)\n");
    match timing {
        Some(t) => {
            let _ = writeln!(
                w,
                "Single-threaded: {}. Release build: {}. Images: {}.\n",
                t.single_threaded, t.release_build, t.pooled.images
            );
            let _ = writeln!(
                w,
                "| arm | median ms | added (ratio of medians) | added (median of per-image ratios) |"
            );
            let _ = writeln!(w, "|---|---|---|---|");
            for (arm, ms) in &t.pooled.median_ms {
                let _ = writeln!(
                    w,
                    "| {arm} | {ms:.2} | {} | {} |",
                    t.pooled
                        .added
                        .get(arm)
                        .map_or("–".into(), |v| format!("{:+.1} %", 100.0 * v)),
                    t.pooled
                        .added_paired
                        .get(arm)
                        .map_or("–".into(), |v| format!("{:+.1} %", 100.0 * v)),
                );
            }
        }
        None => {
            let _ = writeln!(w, "Not timed.");
        }
    }

    let _ = writeln!(w, "\n## Decision rule\n");
    for (arm, v) in &decision.arms {
        let _ = writeln!(
            w,
            "- **{arm}**: worse on {}; not neutral on {}; absent on {}; pooled E1 better: {}; \
             a pooled effect worse: {}; materially more honest: {}; materially less honest: {}; \
             added time: {}; all neutral: {}; flips: {}.",
            list(&v.worse_on),
            list(&v.not_neutral_on),
            list(&v.absent_on),
            v.pooled_e1_better,
            v.pooled_worse,
            v.more_honest,
            v.less_honest,
            v.added_time
                .map_or("–".into(), |t| format!("{:+.1} %", 100.0 * t)),
            v.neutral,
            v.flips
        );
    }
    let _ = writeln!(w, "\nOutcome: `{:?}`.", decision.outcome);
    s
}

fn declined(d: &std::collections::BTreeMap<Arm, super::arms::Declined>) -> String {
    d.iter()
        .map(|(arm, fault)| format!(" `{arm}` declined: {fault:?}."))
        .collect()
}

fn list(v: &[String]) -> String {
    if v.is_empty() {
        "none".into()
    } else {
        v.join(", ")
    }
}

fn pooled_row(w: &mut String, arm: Arm, p: &Pooled) {
    let honesty = |h: Option<PooledHonesty>| match h {
        Some(h) => [
            plain(Some(h.ratio)),
            plain(Some(h.gap)),
            pct_interval(Some(h.coverage_68)),
            pct_interval(Some(h.coverage_95)),
            h.honest.to_string(),
        ],
        None => std::array::from_fn(|_| "–".to_string()),
    };
    let [r, gap, c68, c95, honest] = honesty(p.honesty);
    let uniform = honesty(p.uniform_honesty).join(", ");
    let e = |x: super::rule::PooledEffect| effect(Some((x.interval, x.margin, x.class)));
    let _ = writeln!(
        w,
        "| {arm} | {} | {} | {} | {} | {r} | {gap} | {c68} | {c95} | {honest} | {uniform} |",
        p.units.len(),
        e(p.heldout),
        e(p.homography),
        e(p.split),
    );
}

fn full_grid(f: &super::planar::FullGrid) -> String {
    format!(
        "{}, {}, {}, {}",
        px(f.split_rms_px),
        f.ratio.map_or("–".into(), |r| format!("{r:.3}")),
        pct(f.coverage_68),
        pct(f.coverage_95)
    )
}

/// An effect, its interval, its margin and its class.
fn effect(e: Option<(Interval, f64, Class)>) -> String {
    match e {
        Some((i, margin, class)) => format!(
            "{:+.4} [{:+.4}, {:+.4}] ±{margin:.4} **{class:?}**",
            i.point, i.lo, i.hi
        ),
        None => "–".into(),
    }
}

fn pct_interval(i: Option<Interval>) -> String {
    i.map_or("–".into(), |i| {
        format!(
            "{:.1} % [{:.1}, {:.1}]",
            100.0 * i.point,
            100.0 * i.lo,
            100.0 * i.hi
        )
    })
}

/// How the harness measures and decides: the protocol with Amendment 1.
const METHODS: &str = "\
Amendment 1 applies throughout.

- **Effects** are arm − `uniform` in raw pixels, with 95 % percentile-bootstrap intervals. \
E1 intervals resample views (the pooled-RMS difference over paired views); E2 and E3 intervals \
resample splits.
- **Classes.** An effect is **Better** or **Worse** only if its interval excludes 0 and its size \
reaches the practical margin (shown as ±margin); otherwise **Neutral**. E1 (both measures): \
`max(0.005 px, 1 %` of uniform's held-out RMS`)` on that camera, and for pooled effects the same of \
the median over cameras of uniform's held-out RMS. E2: 2 % of uniform's split RMS (decisive grid), \
pooled likewise.
- **E1.** View `i` is in fold `i mod 5`. Each held-out view's pose is refitted unweighted with the \
fold's camera fixed; residuals are raw pixels for every arm. The homography measure undistorts the \
held-out corners with the fold's camera and fits a normalized-DLT plus Gauss–Newton homography.
- **E2/E3 grid.** The decisive grid is the 16 × 12 nodes inside the convex hull of all the camera's \
detected corners (pixel coordinates, all views). Full-grid figures are reported only.
- **E2** is the median over splits (even/odd plus seeded random halves) of the hull-grid RMS \
of `d`, the pixel difference after casting each node's ray with half A and projecting it with half B.
- **E3** is the pooled ratio `r = Σ_splits Σ_nodes z² / (2 N)` over all `(split, node)` pairs, with \
`z² = dᵀ (Σ_A + Σ_B)⁻¹ d` (the halves are disjoint, hence independent), and the pooled fractions of \
pairs below `z² = 2.30` and `6.18` (nominal 68.3 % and 95.4 %). Its interval resamples splits and \
recomputes the pooled figures. **Honest** means `r` in 0.85–1.15 and both coverages within ±5 points.
- **Materially more honest:** pooled `|r − 1|` at most 0.8 × uniform's, and each pooled coverage at \
least 5 points closer to nominal than uniform's. *Materially less honest* is the same with the arms \
exchanged; either makes E3 non-neutral.
- **Pooling** is over the decisive cameras (the camera is the unit; a rig contributes each camera): \
the median of the per-camera statistics, with a hierarchical bootstrap that resamples views or \
splits within each camera. A camera is decisive when `uniform_noise` (seeded N(0, 0.05²) px per \
axis on training and held-out pixels alike) raises its held-out RMS with an interval above 0.
- **Arms.** A covariance arm the detector cannot supply on a camera is absent there, recorded by \
fault kind. Rig cameras enter E1–E3 as planar calibrations with the rig route's camera, \
intrinsics-init and solver settings (and a Scheimpflug tile's seed); E4 runs the joint solves.
- **Rule.** Flip to an arm when it is present on every decisive camera, no effect is Worse (per \
decisive camera or pooled), both pooled E1 measures are Better or E3 is materially more honest, and \
E5 adds at most 15 %. Otherwise recommend against weighting if any weighting arm is Worse on a \
decisive camera; keep it opt-in if every effect of every arm is Neutral; else the default stays \
`uniform` with no clause met.
- **E4** reports medians over the same kind of splits, no intervals. **E5** times the detector \
alone on decoded, cropped images, single-threaded, rotating the arm order per image; the added \
cost is the ratio of pooled medians.";
