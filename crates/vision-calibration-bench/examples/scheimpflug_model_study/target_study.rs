//! `--parts target`: which target terms (in-plane offsets, shape, cover
//! layer, a view-angle shift) and which camera term (image field) remove the
//! residual, fitted jointly over every given camera (`target.rs`) and scored
//! on held-out captures.
//!
//! Folds are by capture (the image file), so a held-out capture is held out
//! of every camera at once. Each fold refits the route per camera and the
//! joint model on the training captures; the held-out views then get their
//! poses fitted with cameras and target held.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, Result, bail};
use serde::Serialize;
use vision_calibration_bench::floor::{self, FloorCamera, FloorPoint};
use vision_calibration_core::{DistortionParams, NoMeta, SensorParams, View};
use vision_calibration_optim::PlanarIntrinsicsEstimate;

use crate::target::{
    self, CamModel, Free, JCam, JFit, Offsets, Scope, TCfg, TPoint, TargetModel, bow_extent,
    lines_rms, offsets_rms,
};
use crate::{CameraData, route, subset};

const BOW_DEG: usize = 4;

const MODELS: &[TCfg] = &[
    TCfg::base("baseline"),
    TCfg {
        offsets: Offsets::PerPoint,
        ..TCfg::base("+offsets")
    },
    TCfg {
        offsets: Offsets::Separable,
        ..TCfg::base("+offsets separable")
    },
    TCfg {
        bow_deg: 2,
        bow: Scope::Shared,
        ..TCfg::base("+shape2 (H1)")
    },
    TCfg {
        bow_deg: BOW_DEG,
        bow: Scope::Shared,
        ..TCfg::base("+shape4 (H1)")
    },
    TCfg {
        bow_deg: BOW_DEG,
        bow: Scope::PerCamera,
        ..TCfg::base("+shape4 per camera")
    },
    TCfg {
        cover: Scope::Shared,
        ..TCfg::base("+cover (H2)")
    },
    TCfg {
        cover: Scope::PerCamera,
        ..TCfg::base("+cover per camera")
    },
    TCfg {
        angle: Scope::Shared,
        ..TCfg::base("+angle a3")
    },
    TCfg {
        field_deg: 4,
        ..TCfg::base("+field4 (H3)")
    },
    TCfg {
        field_deg: 4,
        field_shared: true,
        ..TCfg::base("+field4 shared")
    },
    TCfg {
        field_deg: 4,
        field_at_obs: true,
        ..TCfg::base("+field4 @obs")
    },
    TCfg {
        offsets: Offsets::PerPoint,
        field_deg: 4,
        field_at_obs: true,
        ..TCfg::base("+offsets+field4 @obs")
    },
    TCfg {
        field_deg: 4,
        hold_camera: true,
        ..TCfg::base("+field4 camera-held")
    },
    TCfg {
        offsets: Offsets::PerPoint,
        field_deg: 4,
        hold_camera: true,
        ..TCfg::base("+offsets+field4 camera-held")
    },
    TCfg {
        offsets: Offsets::PerPoint,
        field_deg: 6,
        hold_camera: true,
        ..TCfg::base("+offsets+field6 camera-held")
    },
    TCfg {
        bow_deg: BOW_DEG,
        bow: Scope::Shared,
        cover: Scope::Shared,
        ..TCfg::base("+shape4+cover")
    },
    TCfg {
        offsets: Offsets::PerPoint,
        bow_deg: BOW_DEG,
        bow: Scope::Shared,
        cover: Scope::Shared,
        ..TCfg::base("+offsets+shape4+cover")
    },
    TCfg {
        offsets: Offsets::PerPoint,
        field_deg: 3,
        ..TCfg::base("+offsets+field3")
    },
    TCfg {
        offsets: Offsets::PerPoint,
        field_deg: 4,
        ..TCfg::base("+offsets+field4")
    },
    TCfg {
        offsets: Offsets::PerPoint,
        field_deg: 6,
        ..TCfg::base("+offsets+field6")
    },
    TCfg {
        offsets: Offsets::PerPoint,
        field_deg: 4,
        field_shared: true,
        ..TCfg::base("+offsets+field4 shared")
    },
    TCfg {
        offsets: Offsets::Separable,
        field_deg: 4,
        ..TCfg::base("+offsets separable+field4")
    },
    TCfg {
        offsets: Offsets::PerPoint,
        field_deg: 4,
        cover: Scope::Shared,
        ..TCfg::base("+offsets+field4+cover")
    },
    TCfg {
        offsets: Offsets::PerPoint,
        field_deg: 4,
        cover: Scope::PerCamera,
        ..TCfg::base("+offsets+field4+cover per camera")
    },
    TCfg {
        offsets: Offsets::PerPoint,
        field_deg: 4,
        angle: Scope::Shared,
        ..TCfg::base("+offsets+field4+angle")
    },
    TCfg {
        offsets: Offsets::PerPoint,
        field_deg: 4,
        bow_deg: BOW_DEG,
        bow: Scope::Shared,
        ..TCfg::base("+offsets+field4+shape4")
    },
    TCfg {
        bow_deg: BOW_DEG,
        bow: Scope::Shared,
        cover: Scope::Shared,
        field_deg: 4,
        ..TCfg::base("+shape4+cover+field4")
    },
    TCfg {
        offsets: Offsets::PerPoint,
        bow_deg: BOW_DEG,
        bow: Scope::Shared,
        cover: Scope::Shared,
        field_deg: 4,
        ..TCfg::base("+offsets+shape4+cover+field4")
    },
];

#[derive(Debug, Clone, Serialize)]
struct FitSummary {
    iters: usize,
    cost: f64,
    covariance: String,
    sigma: BTreeMap<String, f64>,
    cover_mm: Vec<f64>,
    angle_mm: Vec<f64>,
    /// Shape peak-to-valley and RMS after removing a plane, µm, per group.
    shape_pv_um: Vec<f64>,
    shape_rms_um: Vec<f64>,
    shape_coef_mm: Vec<Vec<f64>>,
    /// Offsets RMS after removing the gauge, µm.
    offsets_rms_um: f64,
    cols_rms_um: f64,
    rows_rms_um: f64,
    fx: Vec<f64>,
    k1: Vec<f64>,
    tilt_x: Vec<f64>,
    /// Per-point offsets `[x, y, dx, dy]`, mm (gauge as fitted).
    points: Vec<[f64; 4]>,
    cols: Vec<[f64; 2]>,
    rows: Vec<[f64; 2]>,
}

#[derive(Debug, Clone, Serialize)]
struct Stats {
    rms: f64,
    mean: f64,
    median: f64,
    points: usize,
}

#[derive(Debug, Clone, Serialize)]
struct Row {
    model: String,
    train: Stats,
    held: Stats,
    held_per_camera: Vec<Stats>,
    train_per_camera: Vec<Stats>,
    held_gap_px: f64,
    held_smooth_px: f64,
    /// Repeatable per-point part of the held-out board-frame residual, µm.
    held_point_repeat_um: f64,
    held_point_fraction: f64,
    full: FitSummary,
    folds: Vec<FitSummary>,
}

fn stats(pts: &[&TPoint]) -> Stats {
    let mut m: Vec<f64> = pts
        .iter()
        .map(|p| p.residual_px[0].hypot(p.residual_px[1]))
        .collect();
    let n = m.len().max(1) as f64;
    let rms = (m.iter().map(|v| v * v).sum::<f64>() / n).sqrt();
    let mean = m.iter().sum::<f64>() / n;
    m.sort_by(f64::total_cmp);
    let median = m.get(m.len() / 2).copied().unwrap_or(f64::NAN);
    Stats {
        rms,
        mean,
        median,
        points: m.len(),
    }
}

fn cam_model(est: &PlanarIntrinsicsEstimate) -> Result<CamModel> {
    let c = &est.params.camera;
    let DistortionParams::BrownConrady5 { params: d } = &c.distortion else {
        bail!("the route returns Brown-Conrady");
    };
    let SensorParams::Scheimpflug { params: s } = c.sensor else {
        bail!("the route returns a Scheimpflug sensor");
    };
    Ok(CamModel {
        intr: [
            c.intrinsics.fx,
            c.intrinsics.fy,
            c.intrinsics.cx,
            c.intrinsics.cy,
        ],
        dist: [d.k1, d.k2, d.k3, d.p1, d.p2],
        tilt: [s.tilt_x, s.tilt_y],
        field: Vec::new(),
    })
}

fn summarize(cfg: &TCfg, f: &JFit, frame: target::Frame, board: &[[f64; 2]]) -> FitSummary {
    let (mut pv, mut rms) = (Vec::new(), Vec::new());
    for b in &f.target.bow {
        let (a, r) = bow_extent(b, cfg.bow_deg, frame, board);
        pv.push(a);
        rms.push(r);
    }
    FitSummary {
        iters: f.iters,
        cost: f.cost,
        covariance: f.covariance.clone(),
        sigma: f.sigma.clone(),
        cover_mm: f.target.cover.clone(),
        angle_mm: f.target.angle.clone(),
        shape_pv_um: pv,
        shape_rms_um: rms,
        shape_coef_mm: f.target.bow.clone(),
        offsets_rms_um: offsets_rms(&f.target.points),
        cols_rms_um: lines_rms(&f.target.cols),
        rows_rms_um: lines_rms(&f.target.rows),
        fx: f.cams.iter().map(|c| c.intr[0]).collect(),
        k1: f.cams.iter().map(|c| c.dist[0]).collect(),
        tilt_x: f.cams.iter().map(|c| c.tilt[0]).collect(),
        points: f
            .target
            .points
            .iter()
            .map(|(k, o)| [k.0 as f64 / 100.0, k.1 as f64 / 100.0, o[0], o[1]])
            .collect(),
        cols: f.target.cols.iter().map(|(k, v)| [*k as f64, *v]).collect(),
        rows: f.target.rows.iter().map(|(k, v)| [*k as f64, *v]).collect(),
    }
}

/// Repeatable per-point part of board-frame residuals pooled over cameras.
fn point_repeat(pts: &[TPoint]) -> floor::VarianceComponents {
    let mut g: BTreeMap<target::Key, Vec<[f64; 2]>> = BTreeMap::new();
    for p in pts {
        g.entry(target::key_of(p.board_mm))
            .or_default()
            .push(p.board_res_mm);
    }
    floor::variance_components(g.values(), floor::MIN_GROUP_OBS)
}

pub fn run(cams: &[CameraData], k: usize, out: &Path, only: &[String]) -> Result<()> {
    // Folds by capture.
    let mut captures: Vec<&str> = cams
        .iter()
        .flat_map(|c| c.captures.iter().map(String::as_str))
        .collect();
    captures.sort_unstable();
    captures.dedup();
    let fold_of: BTreeMap<&str, usize> = captures
        .iter()
        .enumerate()
        .map(|(i, c)| (*c, i % k))
        .collect();
    eprintln!("target study: {} captures, {k} folds", captures.len());
    let capture_index: BTreeMap<&str, usize> =
        captures.iter().enumerate().map(|(i, c)| (*c, i)).collect();

    // Board normalization and the board points.
    let mut board: Vec<[f64; 2]> = Vec::new();
    for c in cams {
        for v in &c.dataset.views {
            for p in &v.obs.points_3d {
                board.push([p.x * 1e3, p.y * 1e3]);
            }
        }
    }
    board.sort_by(|a, b| a[0].total_cmp(&b[0]).then(a[1].total_cmp(&b[1])));
    board.dedup_by(|a, b| target::key_of(*a) == target::key_of(*b));
    let (xs, ys): (Vec<f64>, Vec<f64>) = board.iter().map(|b| (b[0], b[1])).unzip();
    let ext = |v: &[f64]| {
        (
            v.iter().copied().fold(f64::MAX, f64::min),
            v.iter().copied().fold(f64::MIN, f64::max),
        )
    };
    let ((x0, x1), (y0, y1)) = (ext(&xs), ext(&ys));
    let frame = target::Frame {
        centre: [(x0 + x1) / 2e3, (y0 + y1) / 2e3],
        half: ((x1 - x0).max(y1 - y0) / 2e3).max(1e-3),
    };
    eprintln!(
        "board points {} over x {x0:.1}..{x1:.1} y {y0:.1}..{y1:.1} mm",
        board.len()
    );

    // Routes: full and per fold.
    let mut full_routes = Vec::new();
    let mut fold_routes: Vec<Vec<PlanarIntrinsicsEstimate>> = Vec::new();
    let mut fold_ids: Vec<Vec<(Vec<usize>, Vec<usize>)>> = Vec::new();
    for c in cams {
        full_routes.push(route(&c.dataset, &c.seed)?);
        let mut fr = Vec::new();
        let mut ids = Vec::new();
        for f in 0..k {
            let n = c.dataset.num_views();
            let (test, train): (Vec<usize>, Vec<usize>) =
                (0..n).partition(|&v| fold_of[c.captures[v].as_str()] == f);
            fr.push(route(&subset(&c.dataset, &train)?, &c.seed)?);
            ids.push((train, test));
        }
        fold_routes.push(fr);
        fold_ids.push(ids);
    }

    let jcam = |ci: usize, ids: &[usize], model: CamModel, poses: Vec<_>| JCam {
        size: cams[ci].size,
        ids: ids.to_vec(),
        views: ids
            .iter()
            .map(|&i| &cams[ci].dataset.views[i])
            .collect::<Vec<&View<NoMeta>>>(),
        model,
        poses,
    };

    let mut rows = Vec::new();
    let mut diag_held: Vec<(String, Vec<TPoint>)> = Vec::new();
    let mut dump: BTreeMap<String, Vec<[f64; 13]>> = BTreeMap::new();
    for cfg in MODELS {
        if !only.is_empty() && !only.iter().any(|o| o == cfg.name) {
            continue;
        }
        eprintln!("== {}", cfg.name);
        let all_cams: Vec<JCam<'_>> = (0..cams.len())
            .map(|ci| {
                let n = cams[ci].dataset.num_views();
                let all: Vec<usize> = (0..n).collect();
                Ok(jcam(
                    ci,
                    &all,
                    cam_model(&full_routes[ci])?,
                    full_routes[ci].params.cam_se3_target.clone(),
                ))
            })
            .collect::<Result<_>>()?;
        let full = target::fit(
            cfg,
            &all_cams,
            &TargetModel::default(),
            frame,
            Free::All,
            300,
        )?;
        eprintln!(
            "  full: {} iters, cost {:.3}, cov {}",
            full.iters, full.cost, full.covariance
        );
        let mut held: Vec<TPoint> = Vec::new();
        let mut folds = Vec::new();
        for f in 0..k {
            let train_cams: Vec<JCam<'_>> = (0..cams.len())
                .map(|ci| {
                    let (train, _) = &fold_ids[ci][f];
                    let r = &fold_routes[ci][f];
                    Ok(jcam(
                        ci,
                        train,
                        cam_model(r)?,
                        r.params.cam_se3_target.clone(),
                    ))
                })
                .collect::<Result<_>>()?;
            let tr = target::fit(
                cfg,
                &train_cams,
                &TargetModel::default(),
                frame,
                Free::All,
                300,
            )?;
            let test_cams: Vec<JCam<'_>> = (0..cams.len())
                .map(|ci| {
                    let (_, test) = &fold_ids[ci][f];
                    let poses = test
                        .iter()
                        .map(|&i| full_routes[ci].params.cam_se3_target[i])
                        .collect();
                    jcam(ci, test, tr.cams[ci].clone(), poses)
                })
                .collect();
            let te = target::fit(cfg, &test_cams, &tr.target, frame, Free::PosesOnly, 100)?;
            eprintln!(
                "  fold {f}: train {} iters, held-out {} pts",
                tr.iters,
                te.points.len()
            );
            folds.push(summarize(cfg, &tr, frame, &board));
            held.extend(te.points);
        }
        let per_cam = |pts: &[TPoint]| -> Vec<Stats> {
            (0..cams.len())
                .map(|ci| stats(&pts.iter().filter(|p| p.cam == ci).collect::<Vec<_>>()))
                .collect()
        };
        let floor_cams: Vec<FloorCamera> = (0..cams.len())
            .map(|ci| FloorCamera {
                id: cams[ci].id.clone(),
                image_size: Some([f64::from(cams[ci].size[0]), f64::from(cams[ci].size[1])]),
                points: held
                    .iter()
                    .filter(|p| p.cam == ci)
                    .map(|p| FloorPoint {
                        view: p.view,
                        board_mm: p.board_mm,
                        observed_px: p.observed_px,
                        residual_px: Some(p.residual_px),
                    })
                    .collect(),
            })
            .collect();
        let fl = floor::analyze(&floor_cams)?;
        let rep = point_repeat(&held);
        let row = Row {
            model: cfg.name.to_string(),
            train: stats(&full.points.iter().collect::<Vec<_>>()),
            held: stats(&held.iter().collect::<Vec<_>>()),
            held_per_camera: per_cam(&held),
            train_per_camera: per_cam(&full.points),
            held_gap_px: fl.pooled.calibrated.smooth_gap_rms_px,
            held_smooth_px: fl.pooled.quartic.rms_px,
            held_point_repeat_um: 1e3 * rep.between_rms,
            held_point_fraction: rep.fraction,
            full: summarize(cfg, &full, frame, &board),
            folds,
        };
        print_row(&row);
        rows.push(row);
        dump.insert(
            cfg.name.to_string(),
            held.iter()
                .map(|p| {
                    [
                        p.cam as f64,
                        p.view as f64,
                        p.board_mm[0],
                        p.board_mm[1],
                        p.observed_px[0],
                        p.observed_px[1],
                        p.residual_px[0],
                        p.residual_px[1],
                        p.board_res_mm[0],
                        p.board_res_mm[1],
                        p.theta,
                        capture_index[cams[p.cam].captures[p.view].as_str()] as f64,
                        p.depth_mm,
                    ]
                })
                .collect(),
        );
        if matches!(cfg.name, "baseline" | "+offsets+field4 camera-held") {
            diag_held.push((cfg.name.to_string(), held));
        }
    }

    print_table(&rows, cams);
    for (name, h) in &diag_held {
        diagnostics(name, h, cams);
    }
    std::fs::write(
        out.join("target_points.json"),
        serde_json::to_string(&dump)?,
    )
    .context("write target_points.json")?;
    std::fs::write(
        out.join("target_study.json"),
        serde_json::to_string_pretty(&rows)?,
    )
    .context("write target_study.json")?;
    Ok(())
}

fn print_row(r: &Row) {
    eprintln!(
        "  train {:.4}/{:.4}  held {:.4}/{:.4}/{:.4}  gap {:.4}  point-repeat {:.1} um ({:.1}%)",
        r.train.rms,
        r.train.mean,
        r.held.rms,
        r.held.mean,
        r.held.median,
        r.held_gap_px,
        r.held_point_repeat_um,
        100.0 * r.held_point_fraction
    );
    let f = &r.full;
    eprintln!(
        "  full: cover {:?} angle {:?} shape pv {:?} um, offsets {:.1} um, cols {:.1} rows {:.1} um, sigma {:?}",
        f.cover_mm,
        f.angle_mm,
        f.shape_pv_um,
        f.offsets_rms_um,
        f.cols_rms_um,
        f.rows_rms_um,
        f.sigma
    );
    for (i, g) in r.folds.iter().enumerate() {
        eprintln!(
            "  fold {i}: cover {:?} angle {:?} shape pv {:?} um, offsets {:.1} um",
            g.cover_mm, g.angle_mm, g.shape_pv_um, g.offsets_rms_um
        );
    }
}

fn print_table(rows: &[Row], cams: &[CameraData]) {
    println!("\n## Target study: held-out captures (rms / mean / median px)");
    let names: Vec<String> = cams
        .iter()
        .map(|c| c.id.rsplit('_').next().unwrap_or("").to_string())
        .collect();
    println!(
        "| model | train rms / mean | held rms / mean / median | gap | point repeat µm | {} |",
        names.join(" | ")
    );
    println!("|---|---|---|---:|---:|{}", "---:|".repeat(names.len()));
    for r in rows {
        let per: Vec<String> = r
            .held_per_camera
            .iter()
            .map(|s| format!("{:.3}", s.mean))
            .collect();
        println!(
            "| {} | {:.3} / {:.3} | {:.3} / {:.3} / {:.3} | {:.3} | {:.1} | {} |",
            r.model,
            r.train.rms,
            r.train.mean,
            r.held.rms,
            r.held.mean,
            r.held.median,
            r.held_gap_px,
            r.held_point_repeat_um,
            per.join(" | ")
        );
    }
    if let Some(r) = rows.first() {
        println!(
            "smooth-fit floor (held-out point set): {:.3}",
            r.held_smooth_px
        );
    }
}

fn pearson(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    let ma = a.iter().sum::<f64>() / n;
    let mb = b.iter().sum::<f64>() / n;
    let (mut ab, mut aa, mut bb) = (0.0, 0.0, 0.0);
    for (x, y) in a.iter().zip(b) {
        ab += (x - ma) * (y - mb);
        aa += (x - ma).powi(2);
        bb += (y - mb).powi(2);
    }
    ab / (aa * bb).sqrt()
}

/// View geometry, the between-camera agreement of per-point means in the
/// board frame and in the image, and pixel locking, on the baseline's
/// held-out residuals.
/// A residual in the board frame (mm) and in the image (px).
type ResidualPair<'a> = (&'a [f64; 2], &'a [f64; 2]);

fn diagnostics(name: &str, pts: &[TPoint], cams: &[CameraData]) {
    println!("\n## Held-out diagnostics: `{name}`");
    println!(
        "view angle per camera (deg): p10 / median / p90; in-plane view azimuth (deg): median"
    );
    for (ci, c) in cams.iter().enumerate() {
        let mut th: Vec<f64> = pts
            .iter()
            .filter(|p| p.cam == ci)
            .map(|p| p.theta.to_degrees())
            .collect();
        th.sort_by(f64::total_cmp);
        let mut az: Vec<f64> = pts
            .iter()
            .filter(|p| p.cam == ci)
            .map(|p| p.dir[1].atan2(p.dir[0]).to_degrees())
            .collect();
        az.sort_by(f64::total_cmp);
        let q = |v: &[f64], f: f64| v[((v.len() - 1) as f64 * f) as usize];
        println!(
            "  {}: {:.1} / {:.1} / {:.1}; azimuth {:.0} (p10 {:.0}, p90 {:.0})",
            c.id,
            q(&th, 0.1),
            q(&th, 0.5),
            q(&th, 0.9),
            q(&az, 0.5),
            q(&az, 0.1),
            q(&az, 0.9)
        );
    }

    // Per (camera, point) means, board frame and image frame.
    let mut by: BTreeMap<(usize, target::Key), Vec<ResidualPair<'_>>> = BTreeMap::new();
    for p in pts {
        by.entry((p.cam, target::key_of(p.board_mm)))
            .or_default()
            .push((&p.board_res_mm, &p.residual_px));
    }
    let means: BTreeMap<(usize, target::Key), ([f64; 2], [f64; 2])> = by
        .into_iter()
        .filter(|(_, v)| v.len() >= 5)
        .map(|(key, v)| {
            let n = v.len() as f64;
            let mut b = [0.0; 2];
            let mut i = [0.0; 2];
            for (x, y) in &v {
                b[0] += x[0] / n;
                b[1] += x[1] / n;
                i[0] += y[0] / n;
                i[1] += y[1] / n;
            }
            (key, (b, i))
        })
        .collect();
    println!("between-camera correlation of per-point means: board frame | image frame (points)");
    let (mut sb, mut si, mut np) = (0.0, 0.0, 0);
    for a in 0..cams.len() {
        for b in a + 1..cams.len() {
            let (mut xb, mut yb, mut xi, mut yi) = (vec![], vec![], vec![], vec![]);
            for ((c, key), (mb, mi)) in &means {
                if *c != a {
                    continue;
                }
                if let Some((nb, ni)) = means.get(&(b, *key)) {
                    xb.extend_from_slice(mb);
                    yb.extend_from_slice(nb);
                    xi.extend_from_slice(mi);
                    yi.extend_from_slice(ni);
                }
            }
            if xb.len() >= 6 {
                let (cb, cim) = (pearson(&xb, &yb), pearson(&xi, &yi));
                sb += cb;
                si += cim;
                np += 1;
                println!(
                    "  cam{a}-cam{b}: {:+.3} | {:+.3} ({})",
                    cb,
                    cim,
                    xb.len() / 2
                );
            }
        }
    }
    println!(
        "  mean over {np} pairs: board {:+.3} | image {:+.3}",
        sb / np.max(1) as f64,
        si / np.max(1) as f64
    );

    // Pixel locking: residual against the sub-pixel phase of the detection.
    let tau = std::f64::consts::TAU;
    for (axis, name) in [(0usize, "u"), (1, "v")] {
        let (mut ss, mut sc, mut s2, mut c2, mut rr) = (0.0, 0.0, 0.0, 0.0, 0.0);
        for p in pts {
            let ph = tau * p.observed_px[axis].rem_euclid(1.0);
            let r = p.residual_px[axis];
            ss += r * ph.sin();
            sc += r * ph.cos();
            s2 += ph.sin().powi(2);
            c2 += ph.cos().powi(2);
            rr += r * r;
        }
        let n = pts.len() as f64;
        let (a, b) = (ss / s2, sc / c2);
        // Each coefficient's standard error is about σ·sqrt(2/n).
        println!(
            "pixel locking {name}: sin {a:+.4} cos {b:+.4} px (amplitude {:.4}, se {:.4}) over {n} points",
            a.hypot(b),
            (rr / n).sqrt() * (2.0 / n).sqrt()
        );
    }
}
