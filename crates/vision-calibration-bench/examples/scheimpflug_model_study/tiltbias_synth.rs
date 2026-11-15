//! `--parts tiltsweep`: how much does an unmodelled defocus-coupled
//! displacement bias a plain (distortion-derived) Scheimpflug calibration,
//! and where?
//!
//! A synthetic camera (the library's Brown-Conrady `k1, k2` under a tilted
//! sensor) whose plane of sharp focus is that of a thin lens with the same
//! tilt (`focus::Lens`), a seeded board, and a displacement added to every
//! detection: the `g + |g| × deg 0–3` law of `tiltbias.rs` (signed and
//! unsigned defocus about the plane of sharp focus, times complex
//! monomials of the image position), with the coefficients measured on the
//! ring-grid capture or drawn at random. The displacement is scaled so the
//! part the plain route leaves in its residual has the measured size
//! ([`VISIBLE_PX`]).
//!
//! The route's bias is the camera part of the least-squares fit of the
//! displacement with every pose free: linear in the displacement, so it is
//! computed noise-free by one Gauss-Newton step at the truth (`linfield`),
//! averaged over view geometries. The stated σ is the same fit's, at
//! [`NOISE_PX`] per axis. A Monte-Carlo through the library route on both
//! backends checks a few cells, with the null (no displacement); the joint
//! fit with the true law is the positive control.

use anyhow::{Context, Result};
use nalgebra::{DVector, Point2, Point3, Vector3};
use vision_calibration::common::IntrinsicsManualInit;
use vision_calibration_core::{
    BrownConrady5, CorrespondenceView, DistortionParams, FxFyCxCy, Iso3, NoMeta, PlanarDataset,
    ScheimpflugParams, View,
};
use vision_calibration_optim::SolverBackend;

use crate::focus::{Lens, tilt_rotation};
use crate::linfield::{self, Basis, LinPoint, NCAM, Spec};
use crate::model::{GenCfg, GenParams, project_point};
use crate::synth::{self, Mix, SIZE};
use crate::tiltbias::{self, BlurMap, Consts, Data, Law, LawBasis, PointCov};

const DEG: f64 = 180.0 / std::f64::consts::PI;
/// The displacement the route leaves in its residual, px RMS (norm): the
/// ring-grid capture's held-out route residual 0.1426 px against 0.0993
/// with the `(|g| − med)` law refitted jointly, `√(0.1426² − 0.0993²)`.
pub const VISIBLE_PX: f64 = 0.10;
/// Per-axis detection noise of the stated σ, px.
pub const NOISE_PX: f64 = 0.07;
/// Geometry draws averaged per cell.
const GEOMETRIES: u64 = 4;
/// Random forms per geometry.
const RANDOM_FORMS: usize = 48;
/// The law of the sweep: the unsigned-blur law, identifiable next to the
/// tilt in the real-data joint fits (a signed one is not).
const LAW: Law = Law::Abs;

/// The measured form: the rig mean of the six ring-grid cameras'
/// coefficients of [`LAW`] (`--parts tiltbias`, full data), in the order of
/// [`LawBasis`] (per monomial, coefficient of `1` then of `i`).
pub const MEASURED: &[f64] = &[
    1.6402e-3, -4.2438e-2, -3.0944e-1, 1.5979e-2, 2.3624e-1, -2.3432e-2, -6.8276e-2, -8.9210e-2,
    3.5123e-2, 3.6988e-2, 3.1480e-2, 7.9409e-2, 6.1105e-1, -8.7440e-2, -4.5068e-1, 8.9841e-2,
    -2.0467e-1, 2.0664e-2, 5.2331e-2, -2.3082e-2,
];

/// One cell of the sweep.
#[derive(Debug, Clone)]
pub struct Cell {
    pub label: String,
    /// Sensor tilt `(τx, τy)`, degrees.
    pub tilt: [f64; 2],
    /// Horizontal field of view, degrees.
    pub hfov: f64,
    /// Radial distortion at the image corner, `k1·r_c²`.
    pub corner: f64,
    /// Board-centre distance range, m.
    pub depth: [f64; 2],
    pub views: usize,
}

/// The studied rig: 720×540, `f ≈ 1140 px` (35°), `k1 ≈ −0.41`,
/// `k2 ≈ 0.24`, tilt −5.2°, boards at 0.17–0.33 m, 20 views.
pub fn baseline() -> Cell {
    Cell {
        label: "baseline".into(),
        tilt: [-5.2, 0.0],
        hfov: 35.0,
        corner: -0.065,
        depth: [0.17, 0.33],
        views: 20,
    }
}

/// A synthetic camera: the truth and the focus geometry of a thin lens
/// with the same tilt.
pub struct Cam {
    pub truth: GenParams,
    /// The plane of sharp focus in the `K'` frame, 1/m.
    pub pi: Vector3<f64>,
    /// `D·f'/4`, px·m.
    pub kappa: f64,
    /// Board pitch, m (scaled with the field of view).
    pub step: f64,
}

/// Board: 8 × 7 points (about the markers per view of the real capture).
const BOARD: [usize; 2] = [8, 7];

pub fn camera(c: &Cell) -> Cam {
    let tau = [c.tilt[0] / DEG, c.tilt[1] / DEG];
    let r = tilt_rotation(tau);
    let fx = 360.0 / (c.hfov / 2.0 / DEG).tan();
    let r_c = 450.0 / fx;
    let k1 = c.corner / (r_c * r_c);
    let k2 = -0.09 * c.corner / r_c.powi(4);
    // The harmonic middle of the depth range is in focus on axis.
    let z_focus = 2.0 / (1.0 / c.depth[0] + 1.0 / c.depth[1]);
    let centre = [364.0, 271.0];
    let mut lens = Lens::new(0.006, 1.0, z_focus, tau, 7e-6, 1.22, centre);
    let (k, _) = lens.pinhole();
    lens.pitch *= k[(0, 0)] / (fx * r[(2, 2)]);
    let (k, _) = lens.pinhole();
    let pi = r * lens.pof_lens();
    let mut truth = synth::typical();
    truth.intr = [
        k[(0, 0)] / r[(2, 2)],
        k[(1, 1)] / r[(2, 2)],
        centre[0],
        centre[1],
    ];
    truth.dist = [k1, k2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    truth.tilt = tau;
    Cam {
        truth,
        pi,
        kappa: lens.pupil() * k[(0, 0)] / 4.0,
        step: 0.011 * (c.hfov / 2.0 / DEG).tan() / (17.45 / DEG).tan(),
    }
}

impl Cam {
    /// Signed defocus of the lens-frame point `x`, px.
    pub fn g(&self, x: &Vector3<f64>, tau: [f64; 2]) -> f64 {
        let xk = tilt_rotation(tau) * x;
        self.kappa * (1.0 / xk.z - self.pi.dot(&Vector3::new(xk.x / xk.z, xk.y / xk.z, 1.0)))
    }
}

/// The scene's noise-free views and their true poses.
pub fn views(cam: &Cam, c: &Cell, seed: u64) -> (Vec<View<NoMeta>>, Vec<Iso3>) {
    let plain = GenCfg::base("plain");
    let fx = cam.truth.intr[0];
    let half = [
        (BOARD[0] - 1) as f64 * cam.step / 2.0,
        (BOARD[1] - 1) as f64 * cam.step / 2.0,
    ];
    let poses = synth::poses_range(c.views, seed, [360.0 / fx, 270.0 / fx], 0.6, c.depth, half);
    let mut views = Vec::new();
    let mut kept = Vec::new();
    for pose in poses {
        let (mut p3, mut p2) = (Vec::new(), Vec::new());
        for j in 0..BOARD[1] {
            for i in 0..BOARD[0] {
                let b = [i as f64 * cam.step, j as f64 * cam.step];
                if (pose * Point3::new(b[0], b[1], 0.0)).z <= 0.01 {
                    continue;
                }
                let u = project_point(&plain, &cam.truth, SIZE, &pose, b);
                if u[0] < 2.0 || u[1] < 2.0 || u[0] > 718.0 || u[1] > 538.0 {
                    continue;
                }
                p3.push(Point3::new(b[0], b[1], 0.0));
                p2.push(Point2::new(u[0], u[1]));
            }
        }
        if p3.len() >= 12
            && let Ok(obs) = CorrespondenceView::new(p3, p2)
        {
            views.push(View::new(obs, NoMeta));
            kept.push(pose);
        }
    }
    (views, kept)
}

/// The true covariates of a linearized point.
fn point_cov(cam: &Cam, p: &LinPoint) -> PointCov {
    let x = (p.pose * Point3::new(p.board[0], p.board[1], 0.0)).coords;
    PointCov {
        gblur: 0.0,
        depth: p.depth,
        g: cam.g(&x, cam.truth.tilt),
        g_fix: 0.0,
        a_fix: 0.0,
    }
}

fn consts_of(cam: &Cam, pts: &[LinPoint]) -> Consts {
    let mut a: Vec<f64> = pts.iter().map(|p| point_cov(cam, p).g.abs()).collect();
    a.sort_by(f64::total_cmp);
    Consts {
        gblur_med: 0.0,
        g_abs_med: a[a.len() / 2],
        zbar: pts.iter().map(|p| p.depth).sum::<f64>() / pts.len() as f64,
    }
}

/// The route's response to a displacement form on one geometry: the
/// camera change per unit visible RMS, and the camera's σ at
/// [`NOISE_PX`].
#[derive(Debug, Clone)]
pub struct Response {
    /// Camera bias with the displacement scaled to [`VISIBLE_PX`] visible.
    pub bias: [f64; NCAM],
    pub sigma: [f64; NCAM],
    /// RMS of the displacement over its visible part.
    pub total_over_visible: f64,
}

/// The linearized route on a geometry: the points at the truth, and the
/// route's response to each form.
pub struct Linear {
    pts: Vec<LinPoint>,
    basis: LawBasis,
    consts: Consts,
    cov: Vec<PointCov>,
}

impl Linear {
    pub fn new(cam: &Cam, c: &Cell, seed: u64) -> Self {
        let (views, poses) = views(cam, c, seed);
        let ids: Vec<usize> = (0..views.len()).collect();
        let pts = linfield::linearize_plain(&cam.truth, SIZE, &ids, &views, &poses);
        let consts = consts_of(cam, &pts);
        let mut cov: Vec<PointCov> = pts.iter().map(|p| point_cov(cam, p)).collect();
        let us: Vec<[f64; 2]> = pts.iter().map(|p| p.u).collect();
        let g: Vec<f64> = cov.iter().map(|v| v.g).collect();
        if let Some((gf, af)) = tiltbias::image_fixed_parts(&us, &g, SIZE) {
            for ((v, a), b) in cov.iter_mut().zip(gf).zip(af) {
                v.g_fix = a;
                v.a_fix = b;
            }
        }
        Self {
            basis: LawBasis::new(LAW, SIZE, consts),
            consts,
            pts,
            cov,
        }
    }

    pub fn consts(&self) -> Consts {
        self.consts
    }

    /// The displacement of `form` at each point (unscaled).
    pub fn displacement(&self, form: &DVector<f64>) -> Vec<[f64; 2]> {
        self.displacement_of(&self.basis, form)
    }

    fn displacement_of(&self, basis: &LawBasis, form: &DVector<f64>) -> Vec<[f64; 2]> {
        self.pts
            .iter()
            .zip(&self.cov)
            .map(|(p, v)| basis.displacement(p.u, v, form))
            .collect()
    }

    /// A basis of another law over the same points.
    pub fn basis_of(&self, law: Law) -> LawBasis {
        LawBasis::new(law, SIZE, self.consts)
    }

    /// The route fitted to the displacement `d` alone: the camera change,
    /// the unit covariance's diagonal, and the RMS left in the residual.
    fn route_fit(&self, d: &[[f64; 2]]) -> ([f64; NCAM], [f64; NCAM], f64) {
        let mut pts = self.pts.clone();
        for (p, di) in pts.iter_mut().zip(d) {
            p.r = *di;
            p.w = 1.0;
        }
        let spec = Spec {
            camera_free: true,
            local: None,
            penalty: None,
            ortho: false,
            shared: None,
            held: None,
        };
        let sol = linfield::fit(&[&pts], &spec);
        let cs = &sol.cams[0];
        let res = linfield::predict(&pts, &cs.dtheta, None, None, None);
        (
            cs.dtheta,
            std::array::from_fn(|k| cs.cov[(k, k)]),
            linfield::stats(&res)[0],
        )
    }

    /// The factor that gives `form` the visible size [`VISIBLE_PX`].
    pub fn scale(&self, form: &DVector<f64>) -> f64 {
        VISIBLE_PX / self.route_fit(&self.displacement(form)).2
    }

    pub fn respond(&self, form: &DVector<f64>) -> Response {
        self.respond_of(&self.basis, form)
    }

    /// [`Linear::respond`] for a form of another law's basis.
    pub fn respond_of(&self, basis: &LawBasis, form: &DVector<f64>) -> Response {
        let d = self.displacement_of(basis, form);
        let (dtheta, var, visible) = self.route_fit(&d);
        let s = VISIBLE_PX / visible;
        Response {
            bias: std::array::from_fn(|k| s * dtheta[k]),
            sigma: std::array::from_fn(|k| NOISE_PX * var[k].sqrt()),
            total_over_visible: linfield::stats(&d)[0] / visible,
        }
    }
}

/// A random form of the law: each coefficient of a monomial of degree `d`
/// normal with σ `1/(1 + d)`.
pub fn random_form(basis: &LawBasis, rng: &mut Mix) -> DVector<f64> {
    DVector::from_iterator(
        basis.len(),
        basis
            .col_degrees()
            .into_iter()
            .map(|d| rng.normal() / (1.0 + d as f64)),
    )
}

/// The measured form, when compiled in and of the law's length.
fn measured() -> Option<DVector<f64>> {
    let n = LawBasis::new(LAW, SIZE, Consts::default()).len();
    (MEASURED.len() == n).then(|| DVector::from_row_slice(MEASURED))
}

fn quantile(mut v: Vec<f64>, q: f64) -> f64 {
    v.sort_by(f64::total_cmp);
    v[((v.len() - 1) as f64 * q).round() as usize]
}

/// Indices of the reported parameters: `tilt_x tilt_y cx cy k1`.
const REP: [usize; 5] = [6, 7, 2, 3, 4];

fn cell_line(c: &Cell) -> (String, String) {
    let cam = camera(c);
    let measured = measured();
    let mut meas: Vec<Response> = Vec::new();
    let mut rnd: Vec<Response> = Vec::new();
    let mut fam: [Vec<Response>; 2] = [Vec::new(), Vec::new()];
    for gs in 0..GEOMETRIES {
        let lin = Linear::new(&cam, c, 100 + gs);
        if let Some(m) = &measured {
            meas.push(lin.respond(m));
        }
        let mut rng = Mix(7_000 + gs);
        for _ in 0..RANDOM_FORMS {
            rnd.push(lin.respond(&random_form(&lin.basis, &mut rng)));
        }
        for (fi, law) in [Law::Image, Law::Var].into_iter().enumerate() {
            let b = lin.basis_of(law);
            let mut rng = Mix(8_000 + 100 * fi as u64 + gs);
            for _ in 0..RANDOM_FORMS {
                fam[fi].push(lin.respond_of(&b, &random_form(&b, &mut rng)));
            }
        }
    }
    let mean = |rs: &[Response], f: &dyn Fn(&Response) -> f64| {
        rs.iter().map(f).sum::<f64>() / rs.len().max(1) as f64
    };
    let sig = |k: usize| mean(&rnd, &|r| r.sigma[k]);
    let unit = |k: usize| if k >= 6 { DEG } else { 1.0 };
    let meas_txt: Vec<String> = REP
        .iter()
        .map(|&k| {
            if meas.is_empty() {
                return "—".into();
            }
            let b = mean(&meas, &|r| r.bias[k]);
            let sd = (mean(&meas, &|r| (r.bias[k] - b).powi(2))).sqrt();
            format!(
                "{:+.3} ± {:.3} ({:+.1}σ)",
                b * unit(k),
                sd * unit(k),
                b / sig(k)
            )
        })
        .collect();
    let rnd_txt: Vec<String> = REP
        .iter()
        .map(|&k| {
            let z: Vec<f64> = rnd.iter().map(|r| (r.bias[k] / r.sigma[k]).abs()).collect();
            format!("{:.1} / {:.1}", quantile(z.clone(), 0.5), quantile(z, 0.9))
        })
        .collect();
    let fam_txt: Vec<String> = fam
        .iter()
        .flat_map(|f| {
            [6usize, 7].map(|k| {
                let z: Vec<f64> = f.iter().map(|r| (r.bias[k] / r.sigma[k]).abs()).collect();
                format!("{:.1} / {:.1}", quantile(z.clone(), 0.5), quantile(z, 0.9))
            })
        })
        .collect();
    (
        format!(
            "{}|{:.4}°|{:.2}|{:.2}|{}|{}",
            c.label,
            sig(6) * DEG,
            sig(3),
            mean(&rnd, &|r| r.total_over_visible),
            meas_txt.join("|"),
            rnd_txt.join("|")
        ),
        format!(
            "{}|{}|{:.2}|{:.2}",
            c.label,
            fam_txt.join("|"),
            mean(&fam[0], &|r| r.total_over_visible),
            mean(&fam[1], &|r| r.total_over_visible)
        ),
    )
}

pub fn cells() -> Vec<Cell> {
    let mut cells = vec![baseline()];
    let with = |label: &str, f: &dyn Fn(&mut Cell)| {
        let mut c = baseline();
        c.label = label.to_string();
        f(&mut c);
        c
    };
    for (tx, ty) in [
        (0.0, 0.0),
        (-2.0, 0.0),
        (-10.0, 0.0),
        (-15.0, 0.0),
        (0.0, 5.0),
        (-10.0, 10.0),
    ] {
        cells.push(with(&format!("tilt ({tx}°, {ty}°)"), &|c| {
            c.tilt = [tx, ty]
        }));
    }
    for d in [-0.005, -0.02, -0.15] {
        cells.push(with(&format!("corner distortion {d}"), &|c| c.corner = d));
    }
    for f in [20.0, 60.0, 90.0] {
        cells.push(with(&format!("HFOV {f}°"), &|c| c.hfov = f));
    }
    for (lo, hi) in [(0.23, 0.27), (0.12, 0.5)] {
        cells.push(with(&format!("depth {lo}–{hi} m"), &|c| {
            c.depth = [lo, hi]
        }));
    }
    for v in [5, 10, 40] {
        cells.push(with(&format!("{v} views"), &|c| c.views = v));
    }
    cells
}

// ───────────────────────── Monte-Carlo ─────────────────────────

/// A scene with the displacement `form`, detection noise `noise_px` per
/// axis (seed `noise`), and a detector blur from the true defocus (relative
/// noise `noise_px / 0.07 × 4 %`) for the joint fit's focus model.
fn noisy(
    cam: &Cam,
    c: &Cell,
    geom: u64,
    noise: u64,
    noise_px: f64,
    form: Option<(&LawBasis, &DVector<f64>)>,
) -> Result<(PlanarDataset, BlurMap, Vec<Iso3>)> {
    let (views, poses) = views(cam, c, geom);
    let ids: Vec<usize> = (0..views.len()).collect();
    let pts = linfield::linearize_plain(&cam.truth, SIZE, &ids, &views, &poses);
    let mut rng = Mix(noise);
    let mut out = Vec::new();
    let mut blur = BlurMap::new();
    let mut k = 0;
    for v in &views {
        let mut p2 = Vec::new();
        for _ in &v.obs.points_2d {
            let p = &pts[k];
            let pc = point_cov(cam, p);
            let d = form.map_or([0.0; 2], |(b, f)| b.displacement(p.u, &pc, f));
            let u = [
                p.u[0] + d[0] + noise_px * rng.normal(),
                p.u[1] + d[1] + noise_px * rng.normal(),
            ];
            let rel = 0.04 * noise_px / NOISE_PX;
            let s = (0.72f64.powi(2) + pc.g * pc.g).sqrt() * (1.0 + rel * rng.normal());
            blur.insert(crate::blur::key(u), s.max(0.2));
            p2.push(Point2::new(u[0], u[1]));
            k += 1;
        }
        out.push(View::new(
            CorrespondenceView::new(v.obs.points_3d.clone(), p2)?,
            NoMeta,
        ));
    }
    Ok((PlanarDataset::new(out)?, blur, poses))
}

fn seed_of(t: &GenParams) -> IntrinsicsManualInit {
    let mut s = IntrinsicsManualInit::default();
    s.intrinsics = Some(FxFyCxCy {
        fx: t.intr[0],
        fy: t.intr[1],
        cx: t.intr[2],
        cy: t.intr[3],
    });
    s.distortion = Some(DistortionParams::BrownConrady5 {
        params: BrownConrady5 {
            k1: t.dist[0],
            k2: t.dist[1],
            k3: 0.0,
            p1: 0.0,
            p2: 0.0,
            iters: 8,
        },
    });
    s.sensor = Some(ScheimpflugParams {
        tilt_x: t.tilt[0],
        tilt_y: t.tilt[1],
    });
    s
}

/// Per draw: estimate − truth of `tilt_x tilt_y cx cy k1` and the stated σ.
#[derive(Default)]
struct Mc {
    err: Vec<[f64; 5]>,
    sig: Vec<[f64; 5]>,
}

impl Mc {
    fn line(&self, name: &str, pred: Option<&[f64; NCAM]>) -> String {
        let n = self.err.len() as f64;
        let txt: Vec<String> = (0..5)
            .map(|i| {
                let k = REP[i];
                let u = if k >= 6 { DEG } else { 1.0 };
                let m = self.err.iter().map(|e| e[i]).sum::<f64>() / n;
                let sd =
                    (self.err.iter().map(|e| (e[i] - m).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
                let st = self.sig.iter().map(|s| s[i]).sum::<f64>() / n;
                let p = pred.map_or(String::new(), |b| format!(", lin {:+.3}", b[k] * u));
                format!(
                    "{:+.3} ± {:.3}{p} (scatter/stated {:.2}, bias {:+.1}σ)",
                    m * u,
                    sd / n.sqrt() * u,
                    sd / st,
                    m / st
                )
            })
            .collect();
        format!("{name}|{}", txt.join("|"))
    }
}

fn lib_err(est: &crate::ParamSummary, truth: &GenParams) -> ([f64; 5], [f64; 5]) {
    let s = |n: &str| {
        est.sigma
            .iter()
            .find(|(k, _)| k.ends_with(n))
            .map_or(f64::NAN, |(_, v)| *v)
    };
    (
        [
            est.tilt_x - truth.tilt[0],
            est.tilt_y - truth.tilt[1],
            est.cx - truth.intr[2],
            est.cy - truth.intr[3],
            est.k1 - truth.dist[0],
        ],
        [s("tilt_x"), s("tilt_y"), s("cx"), s("cy"), s("k1")],
    )
}

fn mc_cell(c: &Cell, draws: usize) -> Result<Vec<String>> {
    let cam = camera(c);
    let measured = measured().context("no measured form compiled in")?;
    let mut lines = Vec::new();
    for with_d in [false, true] {
        let (mut tiny, mut factrs, mut joint) = (Mc::default(), Mc::default(), Mc::default());
        let mut pred = [0.0; NCAM];
        for d in 0..draws {
            let geom = 500 + d as u64;
            let lin = Linear::new(&cam, c, geom);
            let consts = lin.consts();
            let basis = LawBasis::new(LAW, SIZE, consts);
            let scale = lin.scale(&measured);
            let form = &measured * scale;
            if with_d {
                let r = lin.respond(&measured);
                for k in 0..NCAM {
                    pred[k] += r.bias[k] / draws as f64;
                }
            }
            let (ds, blur, poses) = noisy(
                &cam,
                c,
                geom,
                9_000 + d as u64,
                NOISE_PX,
                with_d.then_some((&basis, &form)),
            )?;
            let seed = seed_of(&cam.truth);
            let route = crate::route(&ds, &seed)?;
            for (backend, acc) in [
                (SolverBackend::TinySolver, &mut tiny),
                (SolverBackend::Factrs, &mut factrs),
            ] {
                let est = crate::refine(&ds, &route.params, &crate::VARIANTS[0], backend, 300)?;
                let (e, s) = lib_err(&crate::summarize(&est), &cam.truth);
                acc.err.push(e);
                acc.sig.push(s);
            }
            // Positive control: the joint fit with the true law.
            let data = Data {
                views: &ds.views,
                blur: &blur,
                size: SIZE,
            };
            let ids: Vec<usize> = (0..ds.num_views()).collect();
            let init = crate::gen_init(&route, [0.0; 2])?;
            let j = tiltbias::joint_fit(
                data,
                &ids,
                &init,
                &route.params.cam_se3_target,
                Some(&basis),
                &[],
            )?;
            let _ = poses;
            joint.err.push([
                j.params.tilt[0] - cam.truth.tilt[0],
                j.params.tilt[1] - cam.truth.tilt[1],
                j.params.intr[2] - cam.truth.intr[2],
                j.params.intr[3] - cam.truth.intr[3],
                j.params.dist[0] - cam.truth.dist[0],
            ]);
            joint
                .sig
                .push([j.sigma[6], j.sigma[7], j.sigma[2], j.sigma[3], j.sigma[4]]);
        }
        let tag = if with_d { "measured form" } else { "null" };
        let p = with_d.then_some(&pred);
        lines.push(tiny.line(&format!("{} | {tag} | route, tiny-solver", c.label), p));
        lines.push(factrs.line(&format!("{} | {tag} | route, factrs", c.label), p));
        lines.push(joint.line(
            &format!("{} | {tag} | joint fit with the law", c.label),
            None,
        ));
    }
    Ok(lines)
}

pub fn sweep(draws: usize) -> Result<()> {
    println!(
        "\n## Tilt bias sweep: plain route on a {}-law displacement ({VISIBLE_PX} px visible), σ at {NOISE_PX} px per axis, {GEOMETRIES} geometries × {RANDOM_FORMS} random forms per cell",
        LAW.name()
    );
    println!(
        "cell|σ(τx)|σ(cy) px|total/visible|measured: τx °|τy °|cx px|cy px|k1|random forms |bias|/σ median / 90 %: τx|τy|cx|cy|k1"
    );
    let mut controls = Vec::new();
    for c in cells() {
        let (a, b) = cell_line(&c);
        println!("{a}");
        controls.push(b);
    }
    println!(
        "\n## Controls: random forms of other families, same visible size; |bias|/σ median / 90 %"
    );
    println!(
        "cell|image field deg 0-4: τx|τy|depth-varying g, |g| only: τx|τy|total/visible: image|depth-varying"
    );
    for l in controls {
        println!("{l}");
    }
    if measured().is_none() {
        println!("(no measured form compiled in; the Monte-Carlo needs it)");
        return Ok(());
    }
    println!(
        "\n## Monte-Carlo ({draws} draws): estimate − truth, mean ± its σ, per parameter tilt_x ° | tilt_y ° | cx px | cy px | k1"
    );
    let base = baseline();
    let mc_cells = [
        base.clone(),
        Cell {
            label: "tilt (0°, 0°)".into(),
            tilt: [0.0, 0.0],
            ..base.clone()
        },
        Cell {
            label: "corner distortion -0.02".into(),
            corner: -0.02,
            ..base.clone()
        },
        Cell {
            label: "HFOV 60°".into(),
            hfov: 60.0,
            ..base
        },
    ];
    for c in &mc_cells {
        for l in mc_cell(c, draws)? {
            println!("{l}");
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The synthetic camera is the thin lens's: its plane of sharp focus
    /// passes through the on-axis focus point, and a point on it has zero
    /// defocus.
    #[test]
    fn camera_focus_geometry_is_consistent() {
        let c = baseline();
        let cam = camera(&c);
        let z_focus = 2.0 / (1.0 / c.depth[0] + 1.0 / c.depth[1]);
        // On the lens axis at the focus distance (lens frame).
        let x = Vector3::new(0.0, 0.0, z_focus);
        let g = cam.g(&x, cam.truth.tilt);
        assert!(g.abs() < 1e-6 * cam.kappa / z_focus, "g {g}");
        // Defocus grows on either side along the axis, with opposite signs.
        let near = cam.g(&Vector3::new(0.0, 0.0, 0.8 * z_focus), cam.truth.tilt);
        let far = cam.g(&Vector3::new(0.0, 0.0, 1.25 * z_focus), cam.truth.tilt);
        assert!(near > 0.0 && far < 0.0, "{near} {far}");
    }

    /// The route's response is linear in the displacement and is what the
    /// iterated (nonlinear) plain fit of a noise-free displaced scene finds.
    #[test]
    fn linear_response_matches_the_iterated_route() {
        let c = baseline();
        let cam = camera(&c);
        let lin = Linear::new(&cam, &c, 101);
        let mut rng = Mix(5);
        let form = random_form(&lin.basis, &mut rng);
        let r1 = lin.respond(&form);
        let r2 = lin.respond(&(&form * 3.0));
        for k in 0..NCAM {
            assert!((r1.bias[k] - r2.bias[k]).abs() <= 1e-6 * r1.bias[k].abs().max(1e-12));
        }
        let f = &form * lin.scale(&form);
        let (ds, blur, poses) = noisy(&cam, &c, 101, 1, 0.0, Some((&lin.basis, &f))).unwrap();
        let data = Data {
            views: &ds.views,
            blur: &blur,
            size: SIZE,
        };
        let ids: Vec<usize> = (0..ds.num_views()).collect();
        let j = tiltbias::joint_fit(data, &ids, &cam.truth, &poses, None, &[]).unwrap();
        assert!(j.converged);
        for k in [2, 3, 6, 7] {
            let e = linfield::cam_get(&j.params, k) - linfield::cam_get(&cam.truth, k);
            assert!(
                // Second order: the bias is tens of σ here.
                (e - r1.bias[k]).abs() < 0.05 * r1.bias[k].abs() + 0.3 * r1.sigma[k],
                "parameter {k}: iterated {e} vs linear {}",
                r1.bias[k]
            );
        }
    }

    /// Positive control: the plain fit of a scene displaced by the measured
    /// form is biased by many σ; the joint fit with the true law recovers
    /// the camera.
    #[test]
    fn the_joint_fit_with_the_law_removes_the_bias() {
        let c = baseline();
        let cam = camera(&c);
        let lin = Linear::new(&cam, &c, 102);
        let form = measured().expect("the measured form is compiled in");
        let r = lin.respond(&form);
        assert!(
            r.bias[6].abs() > 5.0 * r.sigma[6],
            "{} vs σ {}",
            r.bias[6],
            r.sigma[6]
        );
        let f = &form * lin.scale(&form);
        let (ds, blur, poses) = noisy(&cam, &c, 102, 1, 0.0, Some((&lin.basis, &f))).unwrap();
        let data = Data {
            views: &ds.views,
            blur: &blur,
            size: SIZE,
        };
        let ids: Vec<usize> = (0..ds.num_views()).collect();
        let j = tiltbias::joint_fit(data, &ids, &cam.truth, &poses, Some(&lin.basis), &[]).unwrap();
        assert!(j.converged);
        for k in [2, 3, 4, 6, 7] {
            let e = linfield::cam_get(&j.params, k) - linfield::cam_get(&cam.truth, k);
            assert!(
                e.abs() < 0.05 * r.sigma[k] + 0.02 * r.bias[k].abs(),
                "parameter {k}: error {e}, plain-route bias {}",
                r.bias[k]
            );
        }
    }
}
