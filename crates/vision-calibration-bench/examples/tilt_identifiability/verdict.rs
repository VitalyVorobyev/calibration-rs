//! The determination sweep's tables: the decision statistic against its
//! labels (A), what fixing the tilt costs (B), and the camera frame each
//! model ends in (D). See [`crate::determine`].

use std::io::Write as _;

use anyhow::Result;

use crate::determine::{
    Cell, CellResult, Design, M0, M0P, M1, M2, MODELS, par_map, run_cell, run_stat_draw,
};
use crate::downstream::{Bench, Metrics};
use crate::study::{Size, deg, header, num, row};

/// Thresholds on the tilt's σ, degrees.
pub const THETAS: [f64; 4] = [0.1, 0.25, 0.5, 1.0];
const HFOVS: [f64; 5] = [20.0, 35.0, 50.0, 70.0, 90.0];
const DISTS: [f64; 4] = [-0.005, -0.02, -0.05, -0.15];
const TILTS: [f64; 5] = [0.0, 2.0, 5.0, 10.0, 20.0];
/// Noise levels `(σ px, per-view error px)` of the sweep.
const LEVELS: [(f64, f64); 2] = [(0.2, 0.0), (0.5, 0.0)];
/// A label is borderline within this factor of the threshold.
const BORDER: f64 = 1.25;

/// The 99th percentile of χ² with 2 dof: the Wald and likelihood-ratio
/// tests of "tilt = 0" reject above it.
pub const CHI2_2_99: f64 = 9.21;

fn cells() -> Vec<Cell> {
    let mut v = Vec::new();
    for design in [Design::Rich, Design::Poor] {
        for &hfov in &HFOVS {
            for &dist in &DISTS {
                for &tilt in &TILTS {
                    v.push(Cell {
                        hfov,
                        dist,
                        tilt: (tilt, 0.0),
                        design,
                    });
                }
            }
        }
    }
    v
}

/// The noise floor of each metric for a camera, design and noise level: the
/// RMS of the correct model's (M1's) error over every tilt and draw.
fn floor(results: &[CellResult], hfov: f64, dist: f64, design: Design, level: usize) -> [f64; 4] {
    let mut ss = [0.0; 4];
    let mut n = 0usize;
    for r in results {
        if r.cell.hfov == hfov && r.cell.dist == dist && r.cell.design == design {
            for d in &r.draws[level].1 {
                for (k, v) in d.metrics[M1].values().iter().enumerate() {
                    ss[k] += v * v;
                }
                n += 1;
            }
        }
    }
    ss.map(|s| (s / n.max(1) as f64).sqrt())
}

/// The largest bias-to-floor ratio over the four metrics.
fn worst_ratio(bias: &Metrics, floor: &[f64; 4]) -> f64 {
    bias.values()
        .iter()
        .zip(floor)
        .map(|(b, f)| b / f)
        .fold(0.0, f64::max)
}

fn find(results: &[CellResult], hfov: f64, dist: f64, tilt: f64, design: Design) -> &CellResult {
    results
        .iter()
        .find(|r| {
            r.cell.hfov == hfov
                && r.cell.dist == dist
                && r.cell.tilt.0 == tilt
                && r.cell.design == design
        })
        .expect("a swept cell")
}

fn median(mut v: Vec<f64>) -> f64 {
    v.retain(|x| x.is_finite());
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

/// The determination sweep: A, B and D.
pub fn sweep(size: &Size, dump: Option<&str>) -> Result<()> {
    let cells = cells();
    let indexed: Vec<(usize, Cell)> = cells.iter().copied().enumerate().collect();
    let results = par_map(&indexed, |(i, c)| {
        run_cell(c, &LEVELS, size.det_draws, 1000 + 101 * *i as u64)
    });
    let results: Vec<CellResult> = results.into_iter().collect::<Result<_>>()?;
    if let Some(path) = dump {
        dump_draws(&results, path)?;
    }
    println!(
        "## Is the tilt determined? A sweep of {} cameras × 2 designs × 2 noise levels, {} draws \
         each\n",
        HFOVS.len() * DISTS.len() * TILTS.len(),
        size.det_draws
    );
    statistic_table(&results);
    rates_table(&results);
    view_error_table(size)?;
    cost_map(&results);
    bound_table(&results);
    detail_table(&results);
    frame_table(&results);
    rule_table(&results);
    Ok(())
}

/// A1: the stated σ against the Fisher σ and the Monte-Carlo error.
fn statistic_table(results: &[CellResult]) {
    println!(
        "### A1 The tilt's stated σ (M1, view-scatter scaled; larger semi-axis, degrees) per \
         camera\n"
    );
    println!(
        "Cell: stated σ̂ (median over tilts and draws) · stated / Fisher at the truth (median) · \
         RMS of the tilt error over the stated σ̂ (larger axis, pooled over tilts and draws). σ(M2) is the Fisher σ with \
         `p1, p2` free.\n"
    );
    for design in [Design::Rich, Design::Poor] {
        for (li, &(noise, _)) in LEVELS.iter().enumerate() {
            println!("\n{}, noise {noise} px\n", design.name());
            let mut cols = vec!["corner distortion".to_string()];
            cols.extend(HFOVS.iter().map(|h| format!("HFOV {h}°")));
            cols.extend(HFOVS.iter().map(|h| format!("σ(M2) {h}°")));
            header(&cols.iter().map(String::as_str).collect::<Vec<_>>());
            for &dist in &DISTS {
                let mut cells = vec![num(dist, 2)];
                let mut m2 = Vec::new();
                for &hfov in &HFOVS {
                    let (mut stated, mut ratio) = (Vec::new(), Vec::new());
                    let (mut ss, mut n) = (0.0, 0usize);
                    for &tilt in &TILTS {
                        let r = find(results, hfov, dist, tilt, design);
                        m2.push(r.fisher_m2 * noise);
                        for d in &r.draws[li].1 {
                            if let Some(s) = d.stat_m1 {
                                stated.push(s.sigma);
                                ratio.push(s.sigma / (r.fisher_m1 * noise));
                                ss += (d.tilt_err[0] / s.sigma)
                                    .powi(2)
                                    .max((d.tilt_err[1] / s.sigma).powi(2));
                                n += 1;
                            }
                        }
                    }
                    let rms = (ss / n.max(1) as f64).sqrt();
                    cells.push(format!(
                        "{} · {} · {}",
                        deg(median(stated)),
                        num(median(ratio), 3),
                        num(rms, 2)
                    ));
                }
                for (i, _) in HFOVS.iter().enumerate() {
                    cells.push(deg(m2[i * TILTS.len()]));
                }
                row(&cells);
            }
        }
    }
}

/// A2: decision error rates of the rule "determined iff σ̂ ≤ θ".
fn rates_table(results: &[CellResult]) {
    println!(
        "\n### A2 Decision \"determined iff stated σ ≤ θ\" against the Fisher label \
         (independent noise)\n"
    );
    println!(
        "Label: determined when the Fisher σ at the truth is at most θ/{BORDER}, undetermined \
         when at least {BORDER}·θ, borderline between. Every draw of every cell, both designs \
         and noise levels.\n"
    );
    header(&[
        "θ",
        "draws labelled determined",
        "false undetermined",
        "draws labelled undetermined",
        "false determined",
        "borderline draws",
        "borderline decided determined",
    ]);
    for &theta in &THETAS {
        let t = theta.to_radians();
        let (mut nd, mut fu, mut nu, mut fd, mut nb, mut bd) = (0, 0, 0, 0, 0, 0);
        for r in results {
            for (li, &(noise, _)) in LEVELS.iter().enumerate() {
                let fisher = r.fisher_m1 * noise;
                for d in &r.draws[li].1 {
                    let decided = d.stat_m1.is_some_and(|s| s.sigma <= t);
                    if fisher * BORDER <= t {
                        nd += 1;
                        fu += usize::from(!decided);
                    } else if fisher >= BORDER * t {
                        nu += 1;
                        fd += usize::from(decided);
                    } else {
                        nb += 1;
                        bd += usize::from(decided);
                    }
                }
            }
        }
        row(&[
            format!("{theta}°"),
            nd.to_string(),
            fu.to_string(),
            nu.to_string(),
            fd.to_string(),
            nb.to_string(),
            bd.to_string(),
        ]);
    }
}

/// A3: with errors shared by a view's corners, the Fisher σ is no label;
/// the Monte-Carlo scatter is, and the view-scatter scaling is what keeps the
/// statistic honest.
fn view_error_table(size: &Size) -> Result<()> {
    let (noise, view_error) = (0.2, 0.3);
    println!(
        "\n### A3 Errors shared by a view's corners (noise {noise} px + a per-view quadratic \
         field of {view_error} px), 20 views, tilt (5°, 0); {} draws\n",
        size.label_draws
    );
    println!(
        "Label: the Monte-Carlo σ (larger semi-axis of the tilt errors' second moment). k: the \
         view-scatter factor the library scaled the covariance by (1 when it did not).\n"
    );
    header(&[
        "HFOV",
        "corner distortion",
        "Fisher σ (independent noise)",
        "Monte-Carlo σ",
        "stated σ, scaled (median)",
        "stated σ, unscaled",
        "k (median)",
        "draws scaled",
        "decided determined at θ = 0.25°: scaled / unscaled",
        "label at 0.25°",
    ]);
    let mut cells = Vec::new();
    for &hfov in &HFOVS {
        for &dist in &DISTS {
            cells.push((hfov, dist));
        }
    }
    let rows = par_map(&cells, |&(hfov, dist)| -> Result<Vec<String>> {
        let cell = Cell {
            hfov,
            dist,
            tilt: (5.0, 0.0),
            design: Design::Rich,
        };
        let bench = Bench::new(cell.spec(), &cell.design.view_design())?;
        let fisher = crate::determine::fisher_sigma(&bench, crate::model::FreeSet::Default, noise);
        let mut errs = Vec::new();
        let (mut scaled, mut unscaled, mut ks) = (Vec::new(), Vec::new(), Vec::new());
        for d in 0..size.label_draws {
            let (stat, err) = run_stat_draw(&bench, noise, view_error, 500 + 31 * d as u64)?;
            errs.push(err);
            if let Some(s) = stat {
                scaled.push(s.sigma);
                unscaled.push(s.sigma_unscaled);
                ks.push((s.sigma / s.sigma_unscaled).powi(2));
            }
        }
        let n = errs.len() as f64;
        let (xx, xy, yy) = errs.iter().fold((0.0, 0.0, 0.0), |a, e| {
            (
                a.0 + e[0] * e[0] / n,
                a.1 + e[0] * e[1] / n,
                a.2 + e[1] * e[1] / n,
            )
        });
        let mc = nalgebra::Matrix2::new(xx, xy, xy, yy)
            .symmetric_eigen()
            .eigenvalues
            .max()
            .sqrt();
        let t = 0.25_f64.to_radians();
        let n_scaled = ks.iter().filter(|&&k| k > 1.0 + 1e-9).count();
        Ok(vec![
            num(hfov, 2),
            num(dist, 2),
            deg(fisher),
            deg(mc),
            deg(median(scaled.clone())),
            deg(median(unscaled.clone())),
            num(median(ks), 3),
            format!("{n_scaled}/{}", scaled.len()),
            format!(
                "{} / {}",
                scaled.iter().filter(|&&s| s <= t).count(),
                unscaled.iter().filter(|&&s| s <= t).count()
            ),
            if mc * BORDER <= t {
                "determined".into()
            } else if mc >= BORDER * t {
                "undetermined".into()
            } else {
                "borderline".into()
            },
        ])
    });
    for r in rows {
        row(&r?);
    }
    Ok(())
}

/// B1: the cost of fixing the tilt at zero, with `p` held (M0) and free
/// (M0p), as the largest bias-to-floor ratio over the four metrics.
fn cost_map(results: &[CellResult]) {
    println!(
        "\n### B1 What fixing the tilt at zero costs: largest bias / noise floor over held-out \
         reprojection, rig transfer, rig triangulation and hand-eye prediction\n"
    );
    println!(
        "Cell: M0 (τ = 0, `p` held) / M0p (τ = 0, `p` free). Bias: the noise-free fit's error; \
         floor: the correct model's (M1's) RMS error under noise. `U` marks a camera whose \
         stated σ(tilt) exceeds 0.5° (undetermined at θ = 0.5°).\n"
    );
    for design in [Design::Rich, Design::Poor] {
        for (li, &(noise, _)) in LEVELS.iter().enumerate() {
            println!("\n{}, noise {noise} px\n", design.name());
            let mut cols = vec![
                "HFOV".to_string(),
                "corner distortion".into(),
                "σ(τ) M1 / M2".into(),
            ];
            cols.extend(TILTS[1..].iter().map(|t| format!("tilt {t}°")));
            header(&cols.iter().map(String::as_str).collect::<Vec<_>>());
            for &hfov in &HFOVS {
                for &dist in &DISTS {
                    let fl = floor(results, hfov, dist, design, li);
                    let r0 = find(results, hfov, dist, 5.0, design);
                    let s1 = r0.fisher_m1 * noise;
                    let mut cells = vec![
                        num(hfov, 2),
                        num(dist, 2),
                        format!(
                            "{}° / {}°{}",
                            deg(s1),
                            deg(r0.fisher_m2 * noise),
                            if s1 > 0.5_f64.to_radians() { " U" } else { "" }
                        ),
                    ];
                    for &tilt in &TILTS[1..] {
                        let r = find(results, hfov, dist, tilt, design);
                        cells.push(format!(
                            "{} / {}",
                            num(worst_ratio(&r.bias[M0], &fl), 2),
                            num(worst_ratio(&r.bias[M0P], &fl), 2)
                        ));
                    }
                    row(&cells);
                }
            }
        }
    }
}

/// B2: the likelihood ratio a fixed tilt costs against `(τ/σ)²`.
fn bound_table(results: &[CellResult]) {
    println!("\n### B2 The cost in χ² against `(τ/σ)²` (20 views, noise 0.2 px)\n");
    println!(
        "`Δχ²` = 2·cost / σ² of the noise-free fit; to first order it is `(τ/σ(M1))²` for M0 \
         and `(τ/σ(M2))²` for M0p (fixing a coordinate costs its squared z-score). Cell: \
         Δχ² / predicted.\n"
    );
    let noise = 0.2;
    let mut cols = vec!["HFOV".to_string(), "corner distortion".into()];
    for t in &TILTS[1..] {
        cols.push(format!("M0, tilt {t}°"));
    }
    for t in &TILTS[1..] {
        cols.push(format!("M0p, tilt {t}°"));
    }
    header(&cols.iter().map(String::as_str).collect::<Vec<_>>());
    for &hfov in &HFOVS {
        for &dist in &DISTS {
            let mut cells = vec![num(hfov, 2), num(dist, 2)];
            for (model, sigma_of) in [(M0, 0), (M0P, 1)] {
                for &tilt in &TILTS[1..] {
                    let r = find(results, hfov, dist, tilt, Design::Rich);
                    let s = if sigma_of == 0 {
                        r.fisher_m1
                    } else {
                        r.fisher_m2
                    } * noise;
                    let pred = (tilt.to_radians() / s).powi(2);
                    let got = r.cost2[model] / (noise * noise);
                    cells.push(format!("{} / {}", num(got, 2), num(pred, 2)));
                }
            }
            row(&cells);
        }
    }
}

const DETAIL: [(f64, f64); 5] = [
    (50.0, -0.05),
    (35.0, -0.02),
    (50.0, -0.005),
    (90.0, -0.005),
    (90.0, -0.15),
];

/// B3: per metric, and the radial terms, for a few cameras.
fn detail_table(results: &[CellResult]) {
    println!("\n### B3 Per metric (20 views, noise 0.2 px): bias of M0 / M0p, and the floor\n");
    println!(
        "Metrics: held-out reprojection px, rig transfer px, rig triangulation ‰ of the \
         distance, hand-eye prediction px; `k1`, `k2`: bias in units of M1's RMS error.\n"
    );
    let mut cols = vec!["camera", "tilt", "model"];
    cols.extend(Metrics::NAMES);
    cols.extend(["k1 bias / σ", "k2 bias / σ"]);
    header(&cols);
    for &(hfov, dist) in &DETAIL {
        let fl = floor(results, hfov, dist, Design::Rich, 0);
        let mut ks = [0.0; 2];
        let mut n = 0usize;
        for &tilt in &TILTS {
            for d in &find(results, hfov, dist, tilt, Design::Rich).draws[0].1 {
                ks[0] += d.k_err[M1][0].powi(2);
                ks[1] += d.k_err[M1][1].powi(2);
                n += 1;
            }
        }
        let ks = ks.map(|s| (s / n as f64).sqrt());
        row(&[
            format!("{hfov}°, {dist}"),
            "any".into(),
            "floor (M1 under noise)".into(),
            num(fl[0], 2),
            num(fl[1], 2),
            num(fl[2], 2),
            num(fl[3], 2),
            format!("σ(k1) {}", num(ks[0], 2)),
            format!("σ(k2) {}", num(ks[1], 2)),
        ]);
        for &tilt in &[5.0, 20.0] {
            let r = find(results, hfov, dist, tilt, Design::Rich);
            for m in [M0, M0P] {
                let v = r.bias[m].values();
                row(&[
                    format!("{hfov}°, {dist}"),
                    format!("{tilt}°"),
                    MODELS[m].into(),
                    num(v[0], 2),
                    num(v[1], 2),
                    num(v[2], 2),
                    num(v[3], 2),
                    num(r.bias_k[m][0] / ks[0], 2),
                    num(r.bias_k[m][1] / ks[1], 2),
                ]);
            }
        }
    }
}

/// D: the frame each model's camera ends in.
fn frame_table(results: &[CellResult]) {
    println!(
        "\n### D The fitted camera frame: its rotation from the lens-axis frame / from the \
         sensor-normal frame, degrees (noise-free fits)\n"
    );
    let mut cols = vec!["camera".to_string(), "tilt".into()];
    cols.extend(MODELS.iter().map(|m| m.to_string()));
    header(&cols.iter().map(String::as_str).collect::<Vec<_>>());
    for &(hfov, dist) in &DETAIL {
        for &tilt in &[5.0, 20.0] {
            let r = find(results, hfov, dist, tilt, Design::Rich);
            let mut cells = vec![format!("{hfov}°, {dist}"), format!("{tilt}°")];
            for b in &r.bias {
                cells.push(format!(
                    "{} / {}",
                    num(b.frame_lens_deg, 2),
                    num(b.frame_sensor_deg, 2)
                ));
            }
            row(&cells);
        }
    }
}

/// The candidate procedures, applied draw by draw: which model each picks.
#[derive(Clone, Copy)]
enum Rule {
    AlwaysM1,
    AlwaysM0p,
    /// σ̂ ≤ θ → M1, else M0.
    SigmaZero(f64),
    /// σ̂ ≤ θ → M1, else M0p.
    SigmaPlain(f64),
    /// σ̂ ≤ θ → M1, else M0p unless the data reject it against M2 (then M1).
    SigmaPlainGuarded(f64),
    /// σ̂ of M2 (tilt told apart from decentring) ≤ θ → M2, else as
    /// `SigmaPlainGuarded` without the first test.
    SeparableGuarded(f64),
    /// M1 when the data reject the plain camera against M2, else M0p.
    LikelihoodRatio,
}

impl Rule {
    fn name(self) -> String {
        match self {
            Rule::AlwaysM1 => "always estimate (M1)".into(),
            Rule::AlwaysM0p => "always τ = 0, p free (M0p)".into(),
            Rule::SigmaZero(t) => format!("σ̂ ≤ {t}° → M1, else M0"),
            Rule::SigmaPlain(t) => format!("σ̂ ≤ {t}° → M1, else M0p"),
            Rule::SigmaPlainGuarded(t) => {
                format!("σ̂ ≤ {t}° → M1, else M0p unless LR(M0p:M2) > {CHI2_2_99} → M1")
            }
            Rule::SeparableGuarded(t) => {
                format!("σ̂(M2) ≤ {t}° → M2, else M0p unless LR(M0p:M2) > {CHI2_2_99} → M1")
            }
            Rule::LikelihoodRatio => format!("LR(M0p:M2) > {CHI2_2_99} → M1, else M0p"),
        }
    }

    fn pick(self, d: &crate::determine::Draw) -> usize {
        let det = |t: f64| d.stat_m1.is_some_and(|s| s.sigma <= t.to_radians());
        match self {
            Rule::AlwaysM1 => M1,
            Rule::AlwaysM0p => M0P,
            Rule::SigmaZero(t) => {
                if det(t) {
                    M1
                } else {
                    M0
                }
            }
            Rule::SigmaPlain(t) => {
                if det(t) {
                    M1
                } else {
                    M0P
                }
            }
            Rule::SigmaPlainGuarded(t) => {
                if det(t) || d.lr_m0p > CHI2_2_99 {
                    M1
                } else {
                    M0P
                }
            }
            Rule::SeparableGuarded(t) => {
                if d.stat_m2.is_some_and(|s| s.sigma <= t.to_radians()) {
                    M2
                } else if d.lr_m0p > CHI2_2_99 {
                    M1
                } else {
                    M0P
                }
            }
            Rule::LikelihoodRatio => {
                if d.lr_m0p > CHI2_2_99 {
                    M1
                } else {
                    M0P
                }
            }
        }
    }
}

/// F (offline): each procedure's outcome, draw by draw, against the
/// correct model's in the same draw.
fn rule_table(results: &[CellResult]) {
    println!(
        "\n### F1 Candidate procedures over every draw: outcome error against the correct \
         model's (M1) in the same draw\n"
    );
    println!(
        "Ratio: largest over the four metrics of (picked model's error / M1's error). Columns: \
         share of draws whose ratio exceeds 1.5, the 99th-percentile and largest ratio, and the \
         share of draws that estimated the tilt. Both designs and noise levels, every \
         camera.\n"
    );
    let rules = [
        Rule::AlwaysM1,
        Rule::AlwaysM0p,
        Rule::SigmaZero(0.5),
        Rule::SigmaPlain(0.25),
        Rule::SigmaPlain(0.5),
        Rule::SigmaPlain(1.0),
        Rule::SigmaPlainGuarded(0.25),
        Rule::SigmaPlainGuarded(0.5),
        Rule::SigmaPlainGuarded(1.0),
        Rule::SeparableGuarded(0.5),
        Rule::LikelihoodRatio,
    ];
    header(&[
        "procedure",
        "tilt",
        "ratio > 1.5",
        "99th pct",
        "max",
        "estimated",
    ]);
    for rule in rules {
        for &tilt in &TILTS {
            let mut ratios = Vec::new();
            let mut estimated = 0usize;
            for r in results.iter().filter(|r| r.cell.tilt.0 == tilt) {
                for (_, draws) in &r.draws {
                    for d in draws {
                        let m = rule.pick(d);
                        estimated += usize::from(m == M1 || m == M2);
                        let a = d.metrics[m].values();
                        let b = d.metrics[M1].values();
                        ratios.push(a.iter().zip(&b).map(|(x, y)| x / y).fold(0.0, f64::max));
                    }
                }
            }
            let n = ratios.len();
            ratios.sort_by(|a, b| a.total_cmp(b));
            let over = ratios.iter().filter(|&&x| x > 1.5).count();
            row(&[
                rule.name(),
                format!("{tilt}°"),
                format!("{:.1} %", 100.0 * over as f64 / n as f64),
                num(ratios[(n * 99 / 100).min(n - 1)], 3),
                num(ratios[n - 1], 3),
                format!("{:.0} %", 100.0 * estimated as f64 / n as f64),
            ]);
        }
    }
}

/// Every draw as a tab-separated row, for exploration.
fn dump_draws(results: &[CellResult], path: &str) -> Result<()> {
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    writeln!(
        f,
        "design\tnoise\tview_error\thfov\tdist\ttilt\tfisher1\tfisher2\tsigma\tsigma_unscaled\twald\tsigma2\tlr0\tlr0p\terr_x\terr_y\t{}\t{}",
        (0..4)
            .flat_map(|m| Metrics::NAMES
                .iter()
                .map(move |n| format!("{}:{n}", ["M0", "M0p", "M1", "M2"][m])))
            .collect::<Vec<_>>()
            .join("\t"),
        (0..4)
            .flat_map(|m| (0..4).map(move |k| format!(
                "bias{}:{}",
                ["M0", "M0p", "M1", "M2"][m],
                k
            )))
            .collect::<Vec<_>>()
            .join("\t"),
    )?;
    for r in results {
        for ((noise, ve), draws) in &r.draws {
            for d in draws {
                let s1 = d.stat_m1;
                let s2 = d.stat_m2;
                let mut cols = vec![
                    format!("{:?}", r.cell.design),
                    noise.to_string(),
                    ve.to_string(),
                    r.cell.hfov.to_string(),
                    r.cell.dist.to_string(),
                    r.cell.tilt.0.to_string(),
                    (r.fisher_m1 * noise).to_degrees().to_string(),
                    (r.fisher_m2 * noise).to_degrees().to_string(),
                    s1.map_or(f64::NAN, |s| s.sigma.to_degrees()).to_string(),
                    s1.map_or(f64::NAN, |s| s.sigma_unscaled.to_degrees())
                        .to_string(),
                    s1.map_or(f64::NAN, |s| s.wald).to_string(),
                    s2.map_or(f64::NAN, |s| s.sigma.to_degrees()).to_string(),
                    d.lr_m0.to_string(),
                    d.lr_m0p.to_string(),
                    d.tilt_err[0].to_degrees().to_string(),
                    d.tilt_err[1].to_degrees().to_string(),
                ];
                for m in d.metrics.iter().chain(&r.bias) {
                    cols.extend(m.values().iter().map(f64::to_string));
                }
                writeln!(f, "{}", cols.join("\t"))?;
            }
        }
    }
    Ok(())
}
