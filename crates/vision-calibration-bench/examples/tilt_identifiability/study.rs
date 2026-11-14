//! The study's parts, each printing Markdown tables.

use anyhow::Result;
use vision_calibration_optim::SolverBackend;

use crate::model::{Cam, FreeSet, Param, TX, TY, column_norms, factorization_error, fisher};
use crate::scene::{CamSpec, Coverage, Rng, Scene, ViewDesign, build_scene};
use crate::solve::{backend_opts, route_cold, route_seeded, solve_direct, solve_param};

/// Pixel noise of the Fisher tables (σ scales linearly with it).
const NOISE: f64 = 0.2;

/// Sample sizes.
pub struct Size {
    pub mc_draws: usize,
    pub starts: usize,
    pub cold_draws: usize,
}

impl Size {
    pub fn full() -> Self {
        Self {
            mc_draws: 100,
            starts: 16,
            cold_draws: 8,
        }
    }
    pub fn quick() -> Self {
        Self {
            mc_draws: 12,
            starts: 4,
            cold_draws: 2,
        }
    }
}

fn spec_with(f: impl FnOnce(&mut CamSpec)) -> CamSpec {
    let mut s = CamSpec::baseline();
    f(&mut s);
    s
}

fn design_with(f: impl FnOnce(&mut ViewDesign)) -> ViewDesign {
    let mut d = ViewDesign::baseline();
    f(&mut d);
    d
}

/// A number for a table: `∞`, `—` for NaN, else `digits` significant.
fn num(x: f64, digits: usize) -> String {
    if x.is_nan() {
        "—".into()
    } else if x.is_infinite() {
        "∞".into()
    } else if x == 0.0 {
        "0".into()
    } else {
        let mag = x.abs().log10().floor() as i32;
        let dec = (digits as i32 - 1 - mag).max(0) as usize;
        if !(-4..6).contains(&mag) {
            format!("{x:.*e}", digits.saturating_sub(1))
        } else {
            format!("{x:.dec$}")
        }
    }
}

fn deg(x: f64) -> String {
    num(x.to_degrees(), 2)
}

fn header(cols: &[&str]) {
    println!("| {} |", cols.join(" | "));
    println!(
        "|{}|",
        cols.iter().map(|_| "---").collect::<Vec<_>>().join("|")
    );
}

fn row(cells: &[String]) {
    println!("| {} |", cells.join(" | "));
}

fn fisher_raw(scene: &Scene, free: FreeSet) -> crate::model::Fisher {
    fisher(scene, &scene.cam, &scene.poses, Param::Raw, &free.indices())
}

// ─── analytic ──────────────────────────────────────────────────────────────

/// The factorization and the decentering expansion, checked numerically.
pub fn analytic() -> Result<()> {
    println!("## Analytic checks\n");
    println!("### Exact factorization K·H(τ) = K'(τ)·R(τ)\n");
    header(&["tilt (x, y) °", "max |Δpixel| over a 21×21 grid"]);
    for &t in &[
        (1.0, 0.0),
        (5.0, 0.0),
        (5.0, 5.0),
        (20.0, 0.0),
        (20.0, 20.0),
    ] {
        let cam = spec_with(|s| s.tilt_deg = t).camera();
        row(&[format!("{t:?}"), num(factorization_error(&cam), 2)]);
    }

    println!("\n### Tilt as decentred distortion\n");
    println!(
        "Noise-free tilted data fitted with the tilt held at zero: with `p1, p2` \
         held (the tilt's signal left), with `p1, p2` free (prediction `p1 = −k1·tan τx`, \
         `p2 = k1·tan τy`), with `s1..s4` free. RMS in pixels.\n"
    );
    header(&[
        "HFOV °",
        "corner distortion",
        "tilt (x, y) °",
        "RMS, p held",
        "RMS, p free",
        "RMS, s free",
        "p1 fitted / predicted",
        "p2 fitted / predicted",
    ]);
    for &hfov in &[30.0, 60.0, 90.0] {
        for &dist in &[-0.02, -0.10] {
            for &t in &[(1.0, 0.0), (5.0, 0.0), (5.0, 5.0), (20.0, 0.0)] {
                let spec = spec_with(|s| {
                    s.hfov_deg = hfov;
                    s.corner_distortion = dist;
                    s.tilt_deg = t;
                });
                let scene = build_scene(spec, &ViewDesign::baseline())?;
                let data = scene.dataset(0.0, 1);
                // Start in the effective pinhole with the tilt at zero.
                let cam = scene.cam;
                let kp = Param::Effective.intrinsics_of_raw(&cam.k, &cam.tilt);
                let start = Cam {
                    k: kp,
                    d: cam.d,
                    tilt: [0.0, 0.0],
                };
                let poses: Vec<_> = scene
                    .poses
                    .iter()
                    .map(|p| Param::Gauge.pose_to(&cam, p))
                    .collect();
                let fit = |free| {
                    solve_direct(
                        &data,
                        &start,
                        &poses,
                        free,
                        None,
                        backend_opts(SolverBackend::TinySolver, 400, true),
                    )
                };
                let rms = |o: &crate::solve::Outcome| {
                    (2.0 * o.cost / scene.num_points() as f64 / 2.0).sqrt()
                };
                let held = fit(FreeSet::NoTilt)?;
                let tang = fit(FreeSet::TangentialNoTilt)?;
                let prism = fit(FreeSet::PrismNoTilt)?;
                let k1 = cam.d[0];
                let pred = [-k1 * cam.tilt[0].tan(), k1 * cam.tilt[1].tan()];
                let ratio = |fit: f64, pred: f64| {
                    if pred.abs() < 1e-12 {
                        format!("{} / 0", num(fit, 2))
                    } else {
                        num(fit / pred, 3)
                    }
                };
                row(&[
                    num(hfov, 2),
                    num(dist, 2),
                    format!("{t:?}"),
                    num(rms(&held), 2),
                    num(rms(&tang), 2),
                    num(rms(&prism), 2),
                    ratio(tang.cam.d[3], pred[0]),
                    ratio(tang.cam.d[4], pred[1]),
                ]);
            }
        }
    }
    Ok(())
}

// ─── fisher ────────────────────────────────────────────────────────────────

pub fn fisher_tables() -> Result<()> {
    println!("## Fisher information at the truth (σ at {NOISE} px)\n");
    let baseline = build_scene(CamSpec::baseline(), &ViewDesign::baseline())?;
    println!(
        "Baseline: 1280×960, HFOV 50° (f = {:.0} px), corner distortion −0.05 \
         (k1 = {:.3}), tilt (5°, 0), 20 views, board tilt ±30°, depth 1.5:1, \
         full coverage, roll ±15°, 12×9 corners ({} points).\n",
        baseline.cam.k[0],
        baseline.cam.d[0],
        baseline.num_points()
    );

    // F1: field of view × distortion.
    let hfovs = [10.0, 20.0, 35.0, 50.0, 70.0, 90.0];
    let dists = [0.0, -0.005, -0.02, -0.05, -0.15];
    for (title, free, what) in [
        ("σ(tilt_x) °, default free set", FreeSet::Default, 0),
        ("corr(cy, tilt_x), default free set", FreeSet::Default, 1),
        ("σ(tilt_x) °, + p1, p2", FreeSet::Tangential, 0),
        ("corr(p1, tilt_x), + p1, p2", FreeSet::Tangential, 2),
        ("σ(tilt_x) °, + s1..s4", FreeSet::Prism, 0),
    ] {
        println!(
            "\n### F1 {title}: field of view × corner distortion (tilt 5° on x; \
             unobs.: σ above 5°)\n"
        );
        let mut cols = vec!["corner distortion".to_string()];
        cols.extend(hfovs.iter().map(|h| format!("HFOV {h}°")));
        header(&cols.iter().map(String::as_str).collect::<Vec<_>>());
        for &d in &dists {
            let mut cells = vec![num(d, 2)];
            for &h in &hfovs {
                let scene = build_scene(
                    spec_with(|s| {
                        s.hfov_deg = h;
                        s.corner_distortion = d;
                    }),
                    &ViewDesign::baseline(),
                )?;
                let f = fisher_raw(&scene, free);
                cells.push(match what {
                    // Without distortion the information is singular; the
                    // central-difference Jacobian's rounding can leave a
                    // finite but meaningless σ there.
                    0 if f.sigma(TX, NOISE).to_degrees() > 5.0 => "unobs.".into(),
                    0 => deg(f.sigma(TX, NOISE)),
                    1 => num(f.corr(3, TX), 3),
                    _ => num(f.corr(7, TX), 3),
                });
            }
            row(&cells);
        }
    }

    // F2: tilt magnitude and axes.
    println!("\n### F2 Tilt magnitude (baseline camera and views)\n");
    header(&[
        "tilt (x, y) °",
        "σ(τx) °",
        "σ(τy) °",
        "σ(fx) px",
        "σ(cy) px",
        "corr(cy,τx)",
        "corr(cx,τy)",
        "corr(fx,τx)",
        "corr(fy,τx)",
        "corr(k1,τx)",
        "max |corr| of τx (with)",
        "σ(τx) inflation",
        "cond cam P0",
        "cond cam P3",
    ]);
    for &t in &[0.0, 1.0, 3.0, 5.0, 10.0, 20.0] {
        for both in [false, true] {
            if t == 0.0 && both {
                continue;
            }
            let tilt = if both { (t, t) } else { (t, 0.0) };
            let scene = build_scene(spec_with(|s| s.tilt_deg = tilt), &ViewDesign::baseline())?;
            let f = fisher_raw(&scene, FreeSet::Default);
            let g = fisher(
                &scene,
                &scene.cam,
                &scene.poses,
                Param::Gauge,
                &FreeSet::Default.indices(),
            );
            let (mc, with) = f.max_corr(TX);
            row(&[
                format!("{tilt:?}"),
                deg(f.sigma(TX, NOISE)),
                deg(f.sigma(TY, NOISE)),
                num(f.sigma(0, NOISE), 2),
                num(f.sigma(3, NOISE), 2),
                num(f.corr(3, TX), 3),
                num(f.corr(2, TY), 3),
                num(f.corr(0, TX), 3),
                num(f.corr(1, TX), 3),
                num(f.corr(4, TX), 3),
                format!("{} ({})", num(mc, 3), crate::model::RAW_NAMES[with]),
                num(f.inflation(TX), 3),
                num(f.cond_cam, 2),
                num(g.cond_cam, 2),
            ]);
        }
    }

    // F3: view design, one factor at a time.
    println!("\n### F3 View design, one factor at a time (baseline camera)\n");
    header(&[
        "factor",
        "level",
        "points",
        "σ(τx) °",
        "σ(τy) °",
        "σ(fx) px",
        "σ(cx) px",
        "σ(cy) px",
        "corr(cy,τx)",
        "corr(cx,τy)",
        "σ(τx) ° + p",
        "σ(τx) ° weak dist (−0.01)",
    ]);
    let mut designs: Vec<(String, String, ViewDesign)> =
        vec![("baseline".into(), "—".into(), ViewDesign::baseline())];
    for n in [5, 10, 40] {
        designs.push(("views".into(), n.to_string(), design_with(|d| d.views = n)));
    }
    for t in [3.0, 10.0, 20.0, 45.0] {
        designs.push((
            "board tilt ±°".into(),
            num(t, 2),
            design_with(|d| d.board_tilt_deg = t),
        ));
    }
    for r in [1.0, 1.25, 2.0, 3.0] {
        designs.push((
            "depth ratio".into(),
            num(r, 3),
            design_with(|d| d.depth_ratio = r),
        ));
    }
    for c in [Coverage::Central, Coverage::OneSided] {
        designs.push((
            "coverage".into(),
            c.name().into(),
            design_with(|d| d.coverage = c),
        ));
    }
    for r in [0.0, 90.0, 180.0] {
        designs.push(("roll ±°".into(), num(r, 3), design_with(|d| d.roll_deg = r)));
    }
    for (factor, level, design) in &designs {
        let scene = build_scene(CamSpec::baseline(), design)?;
        let f = fisher_raw(&scene, FreeSet::Default);
        let t = fisher_raw(&scene, FreeSet::Tangential);
        let weak = build_scene(spec_with(|s| s.corner_distortion = -0.01), design)?;
        let w = fisher_raw(&weak, FreeSet::Default);
        row(&[
            factor.clone(),
            level.clone(),
            scene.num_points().to_string(),
            deg(f.sigma(TX, NOISE)),
            deg(f.sigma(TY, NOISE)),
            num(f.sigma(0, NOISE), 2),
            num(f.sigma(2, NOISE), 2),
            num(f.sigma(3, NOISE), 2),
            num(f.corr(3, TX), 3),
            num(f.corr(2, TY), 3),
            deg(t.sigma(TX, NOISE)),
            deg(w.sigma(TX, NOISE)),
        ]);
    }

    // F4: image aspect.
    println!("\n### F4 Image size and aspect (HFOV 50°, corner distortion −0.05)\n");
    header(&[
        "image",
        "f px",
        "σ(τx) °",
        "σ(τy) °",
        "σ(cy) px",
        "corr(cy,τx)",
        "corr(cx,τy)",
    ]);
    for size in [
        (720.0, 540.0),
        (1280.0, 960.0),
        (1920.0, 1080.0),
        (1024.0, 1024.0),
    ] {
        let scene = build_scene(spec_with(|s| s.size = size), &ViewDesign::baseline())?;
        let f = fisher_raw(&scene, FreeSet::Default);
        row(&[
            format!("{}×{}", size.0, size.1),
            num(scene.cam.k[0], 4),
            deg(f.sigma(TX, NOISE)),
            deg(f.sigma(TY, NOISE)),
            num(f.sigma(3, NOISE), 2),
            num(f.corr(3, TX), 3),
            num(f.corr(2, TY), 3),
        ]);
    }

    // F5: parameterizations.
    println!("\n### F5 Parameterizations (default free set)\n");
    for p in Param::ALL {
        println!("- {}: {}", p.short(), p.name());
    }
    println!(
        "\nThe marginal σ of the tilt is the same in all four (a reparameterization of \
         the other coordinates cannot change it). `σ(τx) inflation` = 1/√(1 − R²) of τx on \
         the other camera coordinates.\n"
    );
    header(&[
        "cell",
        "param",
        "cond full",
        "cond cam",
        "σ(τx) °",
        "σ(c_y-like) px",
        "σ(f_x-like) px",
        "corr(c_y-like, τx)",
        "corr(f_x-like, τx)",
        "max |corr| of τx (with)",
        "σ(τx) inflation",
    ]);
    let cells: Vec<(String, CamSpec, ViewDesign)> = vec![
        (
            "baseline".into(),
            CamSpec::baseline(),
            ViewDesign::baseline(),
        ),
        (
            "weak distortion −0.005".into(),
            spec_with(|s| s.corner_distortion = -0.005),
            ViewDesign::baseline(),
        ),
        (
            "tilt 20°".into(),
            spec_with(|s| s.tilt_deg = (20.0, 0.0)),
            ViewDesign::baseline(),
        ),
        (
            "tilt (10°, 10°), weak −0.01".into(),
            spec_with(|s| {
                s.tilt_deg = (10.0, 10.0);
                s.corner_distortion = -0.01;
            }),
            ViewDesign::baseline(),
        ),
        (
            "HFOV 20°".into(),
            spec_with(|s| s.hfov_deg = 20.0),
            ViewDesign::baseline(),
        ),
        (
            "real-rig-like: 720×540, HFOV 35°, −0.07, tilt −5°".into(),
            spec_with(|s| {
                s.size = (720.0, 540.0);
                s.hfov_deg = 35.0;
                s.corner_distortion = -0.07;
                s.tilt_deg = (-5.0, 0.3);
            }),
            design_with(|d| d.views = 22),
        ),
    ];
    for (name, spec, design) in &cells {
        let scene = build_scene(*spec, design)?;
        for p in Param::ALL {
            let f = fisher(
                &scene,
                &scene.cam,
                &scene.poses,
                p,
                &FreeSet::Default.indices(),
            );
            let (mc, with) = f.max_corr(TX);
            row(&[
                name.clone(),
                p.short().into(),
                num(f.cond_full, 2),
                num(f.cond_cam, 2),
                deg(f.sigma(TX, NOISE)),
                num(f.sigma(3, NOISE), 2),
                num(f.sigma(0, NOISE), 2),
                num(f.corr(3, TX), 3),
                num(f.corr(0, TX), 3),
                format!("{} ({})", num(mc, 3), p.coord_name(with)),
                num(f.inflation(TX), 3),
            ]);
        }
    }

    // F6: free sets.
    println!("\n### F6 Free sets (σ in the raw parameterization)\n");
    header(&[
        "cell",
        "free set",
        "σ(τx) °",
        "σ(τy) °",
        "σ(cx) px",
        "σ(cy) px",
        "σ(p1)",
        "corr(τx,p1)",
        "corr(τy,p2)",
        "corr(p1,s3)",
        "σ(τx) inflation",
        "cond cam",
    ]);
    let free_cells: Vec<(String, CamSpec)> = vec![
        ("baseline".into(), CamSpec::baseline()),
        (
            "HFOV 90°, −0.15".into(),
            spec_with(|s| {
                s.hfov_deg = 90.0;
                s.corner_distortion = -0.15;
            }),
        ),
        (
            "HFOV 20°, −0.02".into(),
            spec_with(|s| {
                s.hfov_deg = 20.0;
                s.corner_distortion = -0.02;
            }),
        ),
    ];
    for (name, spec) in &free_cells {
        let scene = build_scene(*spec, &ViewDesign::baseline())?;
        for free in [
            FreeSet::Default,
            FreeSet::K3,
            FreeSet::Tangential,
            FreeSet::Prism,
            FreeSet::TangentialPrism,
            FreeSet::NoTilt,
            FreeSet::TangentialNoTilt,
        ] {
            let f = fisher_raw(&scene, free);
            row(&[
                name.clone(),
                free.name().into(),
                deg(f.sigma(TX, NOISE)),
                deg(f.sigma(TY, NOISE)),
                num(f.sigma(2, NOISE), 2),
                num(f.sigma(3, NOISE), 2),
                num(f.sigma(7, NOISE), 2),
                num(f.corr(TX, 7), 3),
                num(f.corr(TY, 8), 3),
                num(f.corr(7, 11), 3),
                num(f.inflation(TX), 3),
                num(f.cond_cam, 2),
            ]);
        }
    }
    Ok(())
}

// ─── Monte-Carlo ───────────────────────────────────────────────────────────

/// A Monte-Carlo cell.
struct McCell {
    name: &'static str,
    spec: CamSpec,
    design: ViewDesign,
    noise: f64,
    free: FreeSet,
}

fn mc_cells() -> Vec<McCell> {
    let base = |name, spec, design| McCell {
        name,
        spec,
        design,
        noise: NOISE,
        free: FreeSet::Default,
    };
    vec![
        base("baseline", CamSpec::baseline(), ViewDesign::baseline()),
        base(
            "weak distortion −0.005",
            spec_with(|s| s.corner_distortion = -0.005),
            ViewDesign::baseline(),
        ),
        base(
            "HFOV 20°",
            spec_with(|s| s.hfov_deg = 20.0),
            ViewDesign::baseline(),
        ),
        base(
            "tilt (20°, 0)",
            spec_with(|s| s.tilt_deg = (20.0, 0.0)),
            ViewDesign::baseline(),
        ),
        base("5 views", CamSpec::baseline(), design_with(|d| d.views = 5)),
        base(
            "board tilt ±3°",
            CamSpec::baseline(),
            design_with(|d| d.board_tilt_deg = 3.0),
        ),
        base(
            "left half only",
            CamSpec::baseline(),
            design_with(|d| d.coverage = Coverage::OneSided),
        ),
        McCell {
            noise: 0.5,
            ..base("noise 0.5 px", CamSpec::baseline(), ViewDesign::baseline())
        },
        McCell {
            free: FreeSet::Tangential,
            ..base("+ p1, p2 free", CamSpec::baseline(), ViewDesign::baseline())
        },
        McCell {
            free: FreeSet::Tangential,
            ..base(
                "+ p1, p2 free, HFOV 90°, −0.15",
                spec_with(|s| {
                    s.hfov_deg = 90.0;
                    s.corner_distortion = -0.15;
                }),
                ViewDesign::baseline(),
            )
        },
        base(
            "true p1 = 5e-4, p held at 0",
            spec_with(|s| s.p = (5e-4, 0.0)),
            ViewDesign::baseline(),
        ),
        McCell {
            free: FreeSet::Tangential,
            ..base(
                "true p1 = 5e-4, p free",
                spec_with(|s| s.p = (5e-4, 0.0)),
                ViewDesign::baseline(),
            )
        },
    ]
}

/// The raw coordinates the Monte-Carlo reports.
const MC_COORDS: [(usize, &str); 7] = [
    (0, "fx"),
    (2, "cx"),
    (3, "cy"),
    (4, "k1"),
    (7, "p1"),
    (TX, "tilt_x"),
    (TY, "tilt_y"),
];

/// Mean and sample standard deviation.
fn mean_std(v: &[f64]) -> (f64, f64) {
    let n = v.len() as f64;
    let m = v.iter().sum::<f64>() / n;
    let s = (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n - 1.0).max(1.0)).sqrt();
    (m, s)
}

pub fn monte_carlo(size: &Size) -> Result<()> {
    println!(
        "## Monte-Carlo through `optimize_planar_intrinsics` ({} draws per cell, \
         started at the truth, plain least squares)\n",
        size.mc_draws
    );
    println!(
        "Per coordinate: bias (mean error) / scatter (std of the estimates) / mean stated σ; \
         `ratio` = scatter / stated σ. Angles in degrees, `k1` and `p1` unitless, the rest in \
         pixels. Bias is significant when it exceeds about 2.6·scatter/√draws. `tight`: the \
         same solve run to a relative decrease of 1e-14. The decentering expansion predicts \
         that a true `p1` held at zero moves the tilt by `Δτx = −p1/k1` and `cy` by about \
         `fy·Δτx`: {:.3}° and {:.1} px in the `p1 = 5e-4` cells.\n",
        {
            let c = spec_with(|s| s.p = (5e-4, 0.0)).camera();
            (-5e-4 / c.d[0]).to_degrees()
        },
        {
            let c = spec_with(|s| s.p = (5e-4, 0.0)).camera();
            c.k[1] * (-5e-4 / c.d[0])
        }
    );
    header(&[
        "cell",
        "backend",
        "coordinate",
        "bias",
        "scatter",
        "stated σ",
        "ratio",
        "Fisher σ",
        "no covariance",
    ]);
    for cell in mc_cells() {
        let scene = build_scene(cell.spec, &cell.design)?;
        let fisher_at = fisher_raw(&scene, cell.free);
        let truth = scene.cam.raw();
        let coords: Vec<(usize, &str)> = MC_COORDS
            .iter()
            .copied()
            .filter(|(i, _)| cell.free.indices().contains(i))
            .collect();
        for (backend, tight) in [
            (SolverBackend::TinySolver, false),
            (SolverBackend::Factrs, false),
            (SolverBackend::TinySolver, true),
        ] {
            let mut errs: Vec<Vec<f64>> = vec![Vec::new(); coords.len()];
            let mut stated: Vec<Vec<f64>> = vec![Vec::new(); coords.len()];
            let mut missing = 0usize;
            for draw in 0..size.mc_draws {
                let data = scene.dataset(cell.noise, 1000 + draw as u64);
                // Start at the truth with the tangential terms the solve
                // holds at zero, as the library's default would.
                let mut start = scene.cam;
                if !cell.free.indices().contains(&7) {
                    start.d[3] = 0.0;
                    start.d[4] = 0.0;
                }
                let out = solve_direct(
                    &data,
                    &start,
                    &scene.poses,
                    cell.free,
                    None,
                    backend_opts(backend, if tight { 500 } else { 200 }, tight),
                )?;
                let est = out.cam.raw();
                if out.cov.is_none() {
                    missing += 1;
                }
                for (j, &(i, name)) in coords.iter().enumerate() {
                    errs[j].push(est[i] - truth[i]);
                    if let Some(s) = out.sigma_of(name) {
                        stated[j].push(s);
                    }
                }
            }
            for (j, &(i, name)) in coords.iter().enumerate() {
                let (bias, scatter) = mean_std(&errs[j]);
                let sigma = if stated[j].is_empty() {
                    f64::NAN
                } else {
                    stated[j].iter().sum::<f64>() / stated[j].len() as f64
                };
                let unit = |x: f64| if i >= TX { x.to_degrees() } else { x };
                row(&[
                    cell.name.into(),
                    format!("{backend:?}{}", if tight { " tight" } else { "" }),
                    name.into(),
                    num(unit(bias), 2),
                    num(unit(scatter), 2),
                    num(unit(sigma), 2),
                    num(scatter / sigma, 3),
                    num(unit(fisher_at.sigma(i, cell.noise)), 2),
                    missing.to_string(),
                ]);
            }
        }
    }
    Ok(())
}

// ─── basin ─────────────────────────────────────────────────────────────────

/// A seed quality: the largest relative focal error, tilt error (degrees)
/// and principal-point error (pixels).
#[derive(Clone, Copy)]
struct SeedLevel {
    name: &'static str,
    focal: f64,
    tilt_deg: f64,
    pp: f64,
}

const SEED_LEVELS: [SeedLevel; 4] = [
    SeedLevel {
        name: "good (5 %, 2°, 20 px)",
        focal: 0.05,
        tilt_deg: 2.0,
        pp: 20.0,
    },
    SeedLevel {
        name: "coarse (15 %, 5°, 50 px)",
        focal: 0.15,
        tilt_deg: 5.0,
        pp: 50.0,
    },
    SeedLevel {
        name: "poor (30 %, 10°, 100 px)",
        focal: 0.30,
        tilt_deg: 10.0,
        pp: 100.0,
    },
    SeedLevel {
        name: "bad (50 %, 15°, 200 px)",
        focal: 0.50,
        tilt_deg: 15.0,
        pp: 200.0,
    },
];

fn basin_cells() -> Vec<(&'static str, CamSpec, ViewDesign)> {
    vec![
        ("baseline", CamSpec::baseline(), ViewDesign::baseline()),
        (
            "weak distortion −0.005",
            spec_with(|s| s.corner_distortion = -0.005),
            ViewDesign::baseline(),
        ),
        (
            "HFOV 20°",
            spec_with(|s| s.hfov_deg = 20.0),
            ViewDesign::baseline(),
        ),
        (
            "tilt (20°, 0)",
            spec_with(|s| s.tilt_deg = (20.0, 0.0)),
            ViewDesign::baseline(),
        ),
        (
            "tilt (10°, 10°), weak −0.01",
            spec_with(|s| {
                s.tilt_deg = (10.0, 10.0);
                s.corner_distortion = -0.01;
            }),
            ViewDesign::baseline(),
        ),
        ("5 views", CamSpec::baseline(), design_with(|d| d.views = 5)),
        (
            "real-rig-like",
            spec_with(|s| {
                s.size = (720.0, 540.0);
                s.hfov_deg = 35.0;
                s.corner_distortion = -0.07;
                s.tilt_deg = (-5.0, 0.3);
            }),
            design_with(|d| d.views = 22),
        ),
    ]
}

/// How a result compares with the reference minimum.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Verdict {
    /// Every reported coordinate within 0.5 σ of the reference.
    Reached,
    /// The reference cost (to 1e-3) but a coordinate ≥ 0.5 σ off: stopped
    /// in the valley.
    Valley,
    /// A higher cost with the tilt's sign wrong.
    WrongSign,
    /// A higher cost on a bound.
    AtBound,
    /// A higher cost on the iteration budget.
    Budget,
    /// Another higher-cost minimum.
    LocalMin,
    /// The solve returned an error.
    Error,
}

struct Reference {
    cam: Cam,
    cost: f64,
    sigma: [f64; NCOORD],
}

const COORDS: [usize; 7] = [0, 1, 2, 3, 4, TX, TY];
const NCOORD: usize = COORDS.len();

fn reference(data: &vision_calibration_core::PlanarDataset, scene: &Scene) -> Result<Reference> {
    let mut start = scene.cam;
    start.d[3] = 0.0;
    start.d[4] = 0.0;
    let out = solve_direct(
        data,
        &start,
        &scene.poses,
        FreeSet::Default,
        None,
        backend_opts(SolverBackend::TinySolver, 500, true),
    )?;
    let names = ["fx", "fy", "cx", "cy", "k1", "tilt_x", "tilt_y"];
    let mut sigma = [f64::NAN; NCOORD];
    for (s, n) in sigma.iter_mut().zip(names) {
        *s = out.sigma_of(n).unwrap_or(f64::NAN);
    }
    Ok(Reference {
        cam: out.cam,
        cost: out.cost,
        sigma,
    })
}

/// Largest |Δ|/σ over the reported coordinates.
fn distance(cam: &Cam, r: &Reference) -> f64 {
    let (a, b) = (cam.raw(), r.cam.raw());
    COORDS
        .iter()
        .zip(r.sigma)
        .map(|(&i, s)| (a[i] - b[i]).abs() / s)
        .fold(0.0, f64::max)
}

fn verdict(out: &Result<crate::solve::Outcome>, r: &Reference, max_iters: usize) -> Verdict {
    let Ok(o) = out else {
        return Verdict::Error;
    };
    let excess = (o.cost - r.cost) / r.cost;
    if excess < 1e-3 {
        if distance(&o.cam, r) < 0.5 {
            Verdict::Reached
        } else {
            Verdict::Valley
        }
    } else if o
        .unavailable
        .as_deref()
        .is_some_and(|s| s.contains("AtBound"))
    {
        Verdict::AtBound
    } else if o.cam.tilt[0].signum() != r.cam.tilt[0].signum()
        && r.cam.tilt[0].abs() > 3.0 * r.sigma[5]
    {
        Verdict::WrongSign
    } else if o.iters >= max_iters {
        Verdict::Budget
    } else {
        Verdict::LocalMin
    }
}

/// Whether the result itself flags a failure: no covariance, or an RMS
/// more than 10 % above the reference's.
fn flagged(out: &Result<crate::solve::Outcome>, r: &Reference) -> bool {
    match out {
        Ok(o) => o.cov.is_none() || o.cost > 1.21 * r.cost,
        Err(_) => true,
    }
}

#[derive(Default)]
struct Tally {
    counts: std::collections::BTreeMap<String, usize>,
    iters: Vec<usize>,
    failures: usize,
    flagged: usize,
    total: usize,
}

impl Tally {
    fn add(&mut self, v: Verdict, iters: Option<usize>, flag: bool) {
        self.total += 1;
        *self.counts.entry(format!("{v:?}")).or_default() += 1;
        if v == Verdict::Reached {
            if let Some(i) = iters {
                self.iters.push(i);
            }
        } else {
            self.failures += 1;
            if flag {
                self.flagged += 1;
            }
        }
    }

    fn reached(&self) -> usize {
        self.counts.get("Reached").copied().unwrap_or(0)
    }

    fn median_iters(&self) -> String {
        if self.iters.is_empty() {
            return "—".into();
        }
        let mut v = self.iters.clone();
        v.sort_unstable();
        format!("{} ({}–{})", v[v.len() / 2], v[0], v[v.len() - 1])
    }

    fn modes(&self) -> String {
        let s: Vec<String> = self
            .counts
            .iter()
            .filter(|(k, _)| k.as_str() != "Reached")
            .map(|(k, n)| format!("{k} {n}"))
            .collect();
        if s.is_empty() {
            "—".into()
        } else {
            s.join(", ")
        }
    }
}

/// The perturbed seed of start `i`: each error uniform in `[0.5, 1]×` the
/// level's bound, of random sign.
fn perturbed_seed(cam: &Cam, level: &SeedLevel, rng: &mut Rng) -> ([f64; 4], [f64; 2]) {
    let mut signed = |m: f64| {
        let s = if rng.unit() < 0.5 { -1.0 } else { 1.0 };
        s * m * rng.range(0.5, 1.0)
    };
    let fscale = 1.0 + signed(level.focal);
    let k = [
        cam.k[0] * fscale,
        cam.k[0] * fscale,
        cam.k[2] + signed(level.pp),
        cam.k[3] + signed(level.pp),
    ];
    let t = [
        cam.tilt[0] + signed(level.tilt_deg).to_radians(),
        cam.tilt[1] + signed(level.tilt_deg).to_radians(),
    ];
    (k, t)
}

pub fn basin(size: &Size) -> Result<()> {
    println!(
        "## Convergence from perturbed seeds ({} starts per cell and level, σ = {NOISE} px)\n",
        size.starts
    );
    println!(
        "`Reached`: every one of fx, fy, cx, cy, k1, τx, τy within 0.5σ of the reference \
         minimum (a tight solve from the truth on the same data). Failure modes: `Valley` \
         (the reference cost to 1e-3, a coordinate ≥ 0.5σ off), `WrongSign`, `AtBound`, \
         `Budget`, `LocalMin`, `Error`. `flagged`: of the failures, how many the result \
         itself marks (no covariance, or RMS > 1.1× the reference's). Iterations: median \
         (range) over reached starts; for the route, the final solve's.\n"
    );
    header(&[
        "cell",
        "seed",
        "solve",
        "reached",
        "failure modes",
        "flagged / failures",
        "iterations",
    ]);
    let solves = [
        "route TinySolver",
        "route Factrs",
        "direct P0 (library factor)",
        "P0 user residual",
        "P1 c̃ = c − f·tan τ",
        "P2 K'",
        "P3 K' + sensor-frame poses",
    ];
    for (name, spec, design) in basin_cells() {
        let scene = build_scene(spec, &design)?;
        for level in &SEED_LEVELS {
            let mut tallies: Vec<Tally> = solves.iter().map(|_| Tally::default()).collect();
            for start in 0..size.starts {
                let data = scene.dataset(NOISE, 5000 + start as u64);
                let r = reference(&data, &scene)?;
                let mut rng = Rng::new(9000 + start as u64);
                let (seed_k, seed_t) = perturbed_seed(&scene.cam, level, &mut rng);
                let max_route = 120;
                let max_direct = 200;
                let mut init: Option<(Cam, Vec<vision_calibration_core::Iso3>)> = None;
                for (j, backend) in [SolverBackend::TinySolver, SolverBackend::Factrs]
                    .into_iter()
                    .enumerate()
                {
                    let out = route_seeded(&data, seed_k, seed_t, backend);
                    if let Ok((_, c, p)) = &out
                        && init.is_none()
                    {
                        init = Some((*c, p.clone()));
                    }
                    let out = out.map(|(o, _, _)| o);
                    let v = verdict(&out, &r, max_route);
                    let iters = out.as_ref().ok().map(|o| o.iters);
                    tallies[j].add(v, iters, flagged(&out, &r));
                }
                let Some((init_cam, init_poses)) = init else {
                    for t in tallies.iter_mut().skip(2) {
                        t.add(Verdict::Error, None, true);
                    }
                    continue;
                };
                let direct = solve_direct(
                    &data,
                    &init_cam,
                    &init_poses,
                    FreeSet::Default,
                    None,
                    backend_opts(SolverBackend::TinySolver, max_direct, false),
                );
                let v = verdict(&direct, &r, max_direct);
                tallies[2].add(
                    v,
                    direct.as_ref().ok().map(|o| o.iters),
                    flagged(&direct, &r),
                );
                for (j, p) in Param::ALL.into_iter().enumerate() {
                    let out = solve_param(
                        &data,
                        &init_cam,
                        &init_poses,
                        p,
                        FreeSet::Default,
                        max_direct,
                    );
                    let v = verdict(&out, &r, max_direct);
                    tallies[3 + j].add(v, out.as_ref().ok().map(|o| o.iters), flagged(&out, &r));
                }
            }
            for (solve, t) in solves.iter().zip(&tallies) {
                row(&[
                    name.into(),
                    level.name.into(),
                    (*solve).into(),
                    format!("{}/{}", t.reached(), t.total),
                    t.modes(),
                    format!("{}/{}", t.flagged, t.failures),
                    t.median_iters(),
                ]);
            }
        }
    }

    println!("\n### The cold route (no seed, `step_init` + `step_optimize`)\n");
    header(&[
        "cell",
        "backend",
        "reached",
        "failure modes",
        "flagged / failures",
    ]);
    for (name, spec, design) in basin_cells() {
        let scene = build_scene(spec, &design)?;
        for backend in [SolverBackend::TinySolver, SolverBackend::Factrs] {
            let mut t = Tally::default();
            for draw in 0..size.cold_draws {
                let data = scene.dataset(NOISE, 7000 + draw as u64);
                let r = reference(&data, &scene)?;
                let out = route_cold(&data, backend);
                let v = verdict(&out, &r, 120);
                if let (Ok(o), true) = (&out, v != Verdict::Reached) {
                    eprintln!(
                        "cold {name} {backend:?} draw {draw}: {v:?}, tilt ({:.2}°, {:.2}°), \
                         fx {:.0} (truth {:.0}), cy {:.0} (truth {:.0}), {}",
                        o.cam.tilt[0].to_degrees(),
                        o.cam.tilt[1].to_degrees(),
                        o.cam.k[0],
                        scene.cam.k[0],
                        o.cam.k[3],
                        scene.cam.k[3],
                        o.unavailable.as_deref().unwrap_or("covariance available")
                    );
                }
                t.add(v, None, flagged(&out, &r));
            }
            row(&[
                name.into(),
                format!("{backend:?}"),
                format!("{}/{}", t.reached(), t.total),
                t.modes(),
                format!("{}/{}", t.flagged, t.failures),
            ]);
        }
    }
    Ok(())
}

// ─── solver ────────────────────────────────────────────────────────────────

pub fn solver(size: &Size) -> Result<()> {
    println!("## Solver: stopping rule and Jacobi scaling\n");
    println!(
        "Direct joint solves (tiny-solver) from the coarse seed's linear init: the default \
         stopping rule against a run to a relative decrease of 1e-14 (500 iterations). \
         `max |Δ|/σ`: the largest move of fx, fy, cx, cy, k1, τx, τy between the two in units \
         of the reference σ. Scaling drift: the tilt and principal-point columns' norms at the \
         init over those at the truth (the solver freezes the Jacobi scaling at the first \
         iterate).\n"
    );
    header(&[
        "cell",
        "iterations default / tight",
        "max |Δ|/σ default→tight (median, max)",
        "default reached / tight reached",
        "scaling drift τx, τy, cx, cy, fx (max over starts)",
    ]);
    let level = SEED_LEVELS[1];
    for (name, spec, design) in basin_cells() {
        let scene = build_scene(spec, &design)?;
        let (mut it_d, mut it_t, mut moves) = (Vec::new(), Vec::new(), Vec::new());
        let (mut reached_d, mut reached_t) = (0, 0);
        let mut drift = [0.0f64; 5];
        let truth_norms = column_norms(&scene, &scene.cam, &scene.poses, Param::Raw);
        for start in 0..size.starts {
            let data = scene.dataset(NOISE, 5000 + start as u64);
            let r = reference(&data, &scene)?;
            let mut rng = Rng::new(9000 + start as u64);
            let (seed_k, seed_t) = perturbed_seed(&scene.cam, &level, &mut rng);
            let Ok((_, init_cam, init_poses)) =
                route_seeded(&data, seed_k, seed_t, SolverBackend::TinySolver)
            else {
                continue;
            };
            let d = solve_direct(
                &data,
                &init_cam,
                &init_poses,
                FreeSet::Default,
                None,
                backend_opts(SolverBackend::TinySolver, 200, false),
            );
            let t = solve_direct(
                &data,
                &init_cam,
                &init_poses,
                FreeSet::Default,
                None,
                backend_opts(SolverBackend::TinySolver, 500, true),
            );
            if let (Ok(d), Ok(t)) = (&d, &t) {
                it_d.push(d.iters);
                it_t.push(t.iters);
                let (a, b) = (d.cam.raw(), t.cam.raw());
                moves.push(
                    COORDS
                        .iter()
                        .zip(r.sigma)
                        .map(|(&i, s)| (a[i] - b[i]).abs() / s)
                        .fold(0.0, f64::max),
                );
            }
            reached_d += usize::from(verdict(&d, &r, 200) == Verdict::Reached);
            reached_t += usize::from(verdict(&t, &r, 500) == Verdict::Reached);
            let init_norms = column_norms(&scene, &init_cam, &init_poses, Param::Raw);
            for (k, &c) in [TX, TY, 2, 3, 0].iter().enumerate() {
                let ratio = init_norms[c] / truth_norms[c];
                drift[k] = drift[k].max(ratio.max(1.0 / ratio));
            }
        }
        moves.sort_by(f64::total_cmp);
        let med = |v: &[usize]| {
            let mut v = v.to_vec();
            v.sort_unstable();
            v.get(v.len() / 2).copied().unwrap_or(0)
        };
        row(&[
            name.into(),
            format!("{} / {}", med(&it_d), med(&it_t)),
            format!(
                "{}, {}",
                num(moves.get(moves.len() / 2).copied().unwrap_or(f64::NAN), 2),
                num(moves.last().copied().unwrap_or(f64::NAN), 2)
            ),
            format!("{reached_d} / {reached_t}"),
            drift
                .iter()
                .map(|d| num(*d, 3))
                .collect::<Vec<_>>()
                .join(", "),
        ]);
    }
    Ok(())
}
