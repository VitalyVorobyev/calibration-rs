//! `--parts mimic`: what shape the camera-fixed field has, from a noise-free
//! copy of the best mapping.
//!
//! The route's camera with a degree-4 image field added (evaluated at the
//! model's own projection, the camera held at the route), fitted to a
//! camera's views, is taken as the truth: its field is the camera-fixed part
//! of the route's residual, and its projections at its own poses replace the
//! detections. On that noise-free data
//!
//! - every candidate model is fitted from the route's solution: its residual
//!   is pure model mismatch, so a structured term that captures the field
//!   reaches zero here before it is worth a held-out run;
//! - the route's own residual (`Δ`, the part of the mapping the route
//!   cannot represent, after its parameters and poses have taken what they
//!   can) is decomposed into complex monomials `z^p·z̄^q` about the
//!   principal point. A term `z^p·z̄^q` turns by `m = p − q − 1` under a
//!   rotation of the image about the centre: `m = 0` is radial (real
//!   coefficient) or a radial twist (imaginary), `|m| = 1` decentring,
//!   prism and tilt, `|m| = 2` astigmatic (elliptic), and so on.

use std::collections::BTreeMap;

use anyhow::Result;
use nalgebra::{Complex, DMatrix, DVector, Point2};
use vision_calibration_core::{CorrespondenceView, NoMeta, View};

use crate::model::{self, Free, GenCfg, GenFit};
use crate::{CamCtx, CameraData, GEN_MODELS, gen_init, rms_median};

/// Candidates fitted to the noise-free mapping.
fn candidates() -> Vec<GenCfg> {
    let by = |n: &str| GEN_MODELS.iter().find(|g| g.name == n).cloned();
    let mut v: Vec<GenCfg> = [
        "gen base",
        "gen +field4",
        "gen +field2",
        "gen +field3 only",
        "gen +field3",
        "gen +dist centre",
        "gen +k3 +dc",
        "gen p+s, tilt held",
        "gen sensor h3 @foot",
        "gen sensor h3 @free",
        "gen +elliptic",
        "gen +depth k1",
    ]
    .iter()
    .filter_map(|n| by(n))
    .collect();
    v.push(GenCfg {
        name: "gen all 9 distortion terms",
        free_dist: &model::DIST_NAMES,
        ..GenCfg::base("")
    });
    v
}

/// Views whose detections are `fit`'s projections.
fn clean_views(fit: &GenFit, views: &[View<NoMeta>]) -> Result<Vec<View<NoMeta>>> {
    let mut by_view: BTreeMap<usize, Vec<Point2<f64>>> = BTreeMap::new();
    for p in &fit.residuals {
        by_view.entry(p.view).or_default().push(Point2::new(
            p.observed_px[0] - p.residual_px[0],
            p.observed_px[1] - p.residual_px[1],
        ));
    }
    views
        .iter()
        .enumerate()
        .map(|(i, v)| {
            Ok(View::new(
                CorrespondenceView::new(v.obs.points_3d.clone(), by_view[&i].clone())?,
                NoMeta,
            ))
        })
        .collect()
}

/// Complex-monomial decomposition of a residual field about `centre`
/// (pixels, scaled by `scale`), terms `z^p·z̄^q` of degree 0..=`deg`: the
/// share of the field's sum of squares each rotation order `m = p − q − 1`
/// explains when fitted alone (with the radial and twist halves of `m = 0`
/// apart), and each degree alone. The basis is not orthogonal over the
/// observed region, so the shares do not add up.
pub fn orders(pts: &[([f64; 2], [f64; 2])], centre: [f64; 2], scale: f64, deg: usize) -> String {
    let terms: Vec<(i32, i32)> = (0..=deg as i32)
        .flat_map(|d| (0..=d).map(move |p| (p, d - p)))
        .collect();
    // Real columns: (re, im) of each complex coefficient.
    let mut a = DMatrix::<f64>::zeros(2 * pts.len(), 2 * terms.len());
    let mut b = DVector::<f64>::zeros(2 * pts.len());
    for (i, (px, r)) in pts.iter().enumerate() {
        let z = Complex::new((px[0] - centre[0]) / scale, (px[1] - centre[1]) / scale);
        for (j, &(p, q)) in terms.iter().enumerate() {
            let m = z.powi(p) * z.conj().powi(q);
            // (c_re + i c_im)·m = (c_re m_re − c_im m_im) + i(c_re m_im + c_im m_re)
            a[(2 * i, 2 * j)] = m.re;
            a[(2 * i, 2 * j + 1)] = -m.im;
            a[(2 * i + 1, 2 * j)] = m.im;
            a[(2 * i + 1, 2 * j + 1)] = m.re;
        }
        b[2 * i] = r[0];
        b[2 * i + 1] = r[1];
    }
    let tss = b.norm_squared();
    let share = |cols: &[usize]| -> f64 {
        if cols.is_empty() {
            return 0.0;
        }
        let sub = DMatrix::from_fn(a.nrows(), cols.len(), |i, j| a[(i, cols[j])]);
        let ata = sub.transpose() * &sub;
        let atb = sub.transpose() * &b;
        match ata.cholesky() {
            Some(ch) => {
                let c = ch.solve(&atb);
                1.0 - (&b - &sub * c).norm_squared() / tss
            }
            None => f64::NAN,
        }
    };
    let all: Vec<usize> = (0..2 * terms.len()).collect();
    let total = (tss / pts.len() as f64).sqrt();
    let mut out = format!(
        "rms {total:.4}, deg<={deg} explains {:.0}% |",
        100.0 * share(&all)
    );
    let orders: Vec<i32> = {
        let mut m: Vec<i32> = terms.iter().map(|&(p, q)| p - q - 1).collect();
        m.sort_unstable();
        m.dedup();
        m
    };
    for m in orders {
        let js: Vec<usize> = terms
            .iter()
            .enumerate()
            .filter(|(_, t)| t.0 - t.1 - 1 == m)
            .map(|(j, _)| j)
            .collect();
        if m == 0 {
            let re: Vec<usize> = js.iter().map(|j| 2 * j).collect();
            let im: Vec<usize> = js.iter().map(|j| 2 * j + 1).collect();
            out.push_str(&format!(
                " m0 radial {:.0}% twist {:.0}%",
                100.0 * share(&re),
                100.0 * share(&im)
            ));
        } else {
            let cols: Vec<usize> = js.iter().flat_map(|j| [2 * j, 2 * j + 1]).collect();
            out.push_str(&format!(" m{m:+} {:.0}%", 100.0 * share(&cols)));
        }
    }
    out.push_str(" | by degree:");
    for d in 0..=deg as i32 {
        let cols: Vec<usize> = terms
            .iter()
            .enumerate()
            .filter(|(_, t)| t.0 + t.1 == d)
            .flat_map(|(j, _)| [2 * j, 2 * j + 1])
            .collect();
        out.push_str(&format!(" {d}:{:.0}%", 100.0 * share(&cols)));
    }
    out
}

/// A coarse map of a residual field: mean vector per cell, 8 × 6 cells.
fn cell_map(pts: &[([f64; 2], [f64; 2])], size: [u32; 2]) -> Vec<Option<[f64; 2]>> {
    let (w, h) = (f64::from(size[0]), f64::from(size[1]));
    let mut s = vec![[0.0f64; 3]; 48];
    for (px, r) in pts {
        let c = ((px[0] / w * 8.0) as usize).min(7);
        let rr = ((px[1] / h * 6.0) as usize).min(5);
        let e = &mut s[rr * 8 + c];
        e[0] += r[0];
        e[1] += r[1];
        e[2] += 1.0;
    }
    s.iter()
        .map(|e| (e[2] >= 10.0).then(|| [e[0] / e[2], e[1] / e[2]]))
        .collect()
}

pub struct MimicOut {
    pub lines: Vec<String>,
    pub map: Vec<Option<[f64; 2]>>,
}

pub fn run(cam: &CameraData, ctx: &CamCtx) -> Result<MimicOut> {
    let n = cam.dataset.num_views();
    let all: Vec<usize> = (0..n).collect();
    let views = cam.dataset.views.clone();
    let init = gen_init(&ctx.base, ctx.centre)?;
    let poses = &ctx.base.params.cam_se3_target;
    let truth_cfg = GEN_MODELS
        .iter()
        .find(|g| g.name == "gen +field4 camera held")
        .expect("the reference field");
    let truth = model::fit(
        truth_cfg,
        &all,
        &views,
        &init,
        poses,
        cam.size,
        Free::All,
        500,
    )?;
    let clean = clean_views(&truth, &views)?;
    let mut lines = Vec::new();
    let mut map = Vec::new();
    for cfg in candidates() {
        let f = model::fit(&cfg, &all, &clean, &init, poses, cam.size, Free::All, 800)?;
        let (rms, _) = rms_median(&f.residuals);
        let p = &f.params;
        lines.push(format!(
            "{}|{}|{:.4}|fx {:.1} fy {:.1} cx {:.1} cy {:.1} tilt {:+.4} {:+.4} k1 {:+.3} k2 {:+.3}|iters {}",
            cam.id,
            cfg.name,
            rms,
            p.intr[0],
            p.intr[1],
            p.intr[2],
            p.intr[3],
            p.tilt[0],
            p.tilt[1],
            p.dist[0],
            p.dist[1],
            f.iters
        ));
        if cfg.name == "gen base" {
            let pts: Vec<([f64; 2], [f64; 2])> = f
                .residuals
                .iter()
                .map(|q| (q.observed_px, q.residual_px))
                .collect();
            let pp = [p.intr[2], p.intr[3]];
            let half = f64::from(cam.size[0]) / 2.0;
            lines.push(format!(
                "{}|Δ orders about pp, deg 5|{}",
                cam.id,
                orders(&pts, pp, half, 5)
            ));
            // The same about the foot of the sensor normal.
            let (sx, sy) = model::foot(p.tilt[0], p.tilt[1]);
            let footpx = [p.intr[0] * sx + p.intr[2], p.intr[1] * sy + p.intr[3]];
            lines.push(format!(
                "{}|Δ orders about foot ({:.0}, {:.0}), deg 5|{}",
                cam.id,
                footpx[0],
                footpx[1],
                orders(&pts, footpx, half, 5)
            ));
            map = cell_map(&pts, cam.size);
        }
    }
    Ok(MimicOut { lines, map })
}

/// Correlation of two cameras' cell maps over the cells both have.
pub fn map_corr(a: &[Option<[f64; 2]>], b: &[Option<[f64; 2]>]) -> (f64, usize) {
    let (mut x, mut y) = (Vec::new(), Vec::new());
    for (p, q) in a.iter().zip(b) {
        if let (Some(p), Some(q)) = (p, q) {
            x.extend(p);
            y.extend(q);
        }
    }
    (crate::pearson(&x, &y), x.len() / 2)
}

pub fn print_map(id: &str, map: &[Option<[f64; 2]>]) {
    println!("{id}: route Δ (noise-free), mean px per 8x6 cell, (du,dv)");
    for r in 0..6 {
        let row: Vec<String> = (0..8)
            .map(|c| match map[r * 8 + c] {
                Some(v) => format!("{:+.2},{:+.2}", v[0], v[1]),
                None => "     .     ".into(),
            })
            .collect();
        println!("  {}", row.join(" "));
    }
}
