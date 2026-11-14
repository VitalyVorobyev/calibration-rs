//! `--parts focus` and `focussweep`: can per-marker blur determine the
//! Scheimpflug tilt? (`focus.rs` holds the model; findings in
//! `docs/notes/tilt-from-focus.md`.)
//!
//! - `focus`: per registry camera, the plane of sharp focus fitted to the
//!   ring-grid detector's blur in the frame of the plain pinhole `K'` (the
//!   route's tilt rotation applied to its poses), its fit quality against
//!   the free quadratic, residual structure, fold stability, the tilt it
//!   implies for a range of pixel pitches, the pitch at which it matches
//!   the distortion-derived tilt, and the direction agreement (which needs
//!   no pitch).
//! - `focussweep`: synthetic blur from the exact paraxial thick lens
//!   (`focus::Lens`), Monte-Carlo over noise and view geometry, over tilt,
//!   magnification, pupil magnification, f-number, depth range, views,
//!   noise, and two wrong models (field curvature, a non-quadrature
//!   detector).

use anyhow::Result;
use nalgebra::{Point3, Translation3, UnitQuaternion, Vector3};
use vision_calibration_core::{Iso3, SensorParams};

use crate::CameraData;
use crate::focus::{
    BlurObs, Lens, PofFit, axis_of_tilt, fit_pof, focal_from_pof, implied_inv_v0, quad_fit_robust,
    quad_predict, tilt_cov, tilt_from_pof, tilt_rotation,
};
use crate::synth::Mix;

const DEG: f64 = 180.0 / std::f64::consts::PI;

/// Explained variance of `obs` by `pred`.
fn explained(obs: &[f64], pred: &[f64]) -> f64 {
    let n = obs.len() as f64;
    let m = obs.iter().sum::<f64>() / n;
    let tss: f64 = obs.iter().map(|o| (o - m).powi(2)).sum();
    let rss: f64 = obs.iter().zip(pred).map(|(o, p)| (o - p).powi(2)).sum();
    1.0 - rss / tss
}

fn mean_sd(v: &[f64]) -> (f64, f64) {
    let n = v.len() as f64;
    if v.is_empty() {
        return (f64::NAN, f64::NAN);
    }
    let m = v.iter().sum::<f64>() / n;
    let s = (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n - 1.0).max(1.0)).sqrt();
    (m, s)
}

/// Delete-a-group jackknife standard deviation of fold estimates.
fn jack_sd(v: &[f64]) -> f64 {
    let k = v.len() as f64;
    let m = v.iter().sum::<f64>() / k;
    ((k - 1.0) / k * v.iter().map(|x| (x - m).powi(2)).sum::<f64>()).sqrt()
}

fn wrap(a: f64) -> f64 {
    let t = std::f64::consts::TAU;
    (a + std::f64::consts::PI).rem_euclid(t) - std::f64::consts::PI
}

// ───────────────────────── real data ─────────────────────────

struct RealObs {
    view: usize,
    o: BlurObs,
    px: [f64; 2],
}

/// The pitch (µm) at which the thin-lens inversion of `pi` reproduces the
/// tilt `tau`, and the azimuth of `π_xy` minus that of the tilt's axis
/// (deg).
fn implied(pi: &Vector3<f64>, tau: [f64; 2], fpx: f64) -> (f64, f64) {
    let w = implied_inv_v0(pi, tau);
    let a = axis_of_tilt(tau);
    (
        1e6 / (w * fpx),
        wrap(pi.y.atan2(pi.x) - a.y.atan2(a.x)) * DEG,
    )
}

/// The blur observations of `views` (dataset indices) with their poses
/// (`poses[k]` for `views[k]`), in the `K'` frame of `tau`.
fn real_obs(cam: &CameraData, views: &[usize], poses: &[Iso3], tau: [f64; 2]) -> Vec<RealObs> {
    let r = tilt_rotation(tau);
    let mut out = Vec::new();
    for (&v, pose) in views.iter().zip(poses) {
        let view = &cam.dataset.views[v];
        for (p3, p2) in view.obs.points_3d.iter().zip(&view.obs.points_2d) {
            let Some(&blur) = cam.blur.get(&crate::blur::key([p2.x, p2.y])) else {
                continue;
            };
            let xc: Point3<f64> = pose * p3;
            out.push(RealObs {
                view: v,
                o: BlurObs {
                    x: r * xc.coords,
                    blur,
                },
                px: [p2.x, p2.y],
            });
        }
    }
    out
}

fn tilt_of(est: &vision_calibration_optim::PlanarIntrinsicsEstimate) -> Result<[f64; 2]> {
    let SensorParams::Scheimpflug { params: s } = est.params.camera.sensor else {
        anyhow::bail!("the route returns a Scheimpflug sensor");
    };
    Ok([s.tilt_x, s.tilt_y])
}

/// Per camera: what the cross-camera summary needs.
struct CamResult {
    id: String,
    tau: [f64; 2],
    /// Jackknife σ of the routed tilt over the view folds.
    tau_jk: [f64; 2],
    fpx: f64,
    pi: Vector3<f64>,
    pitch: f64,
    pitch_jk: f64,
}

pub fn real(cams: &[CameraData], folds: usize) -> Result<()> {
    println!("\n## Focus: plane of sharp focus from the detector's blur");
    let pitches = [2.2, 3.45, 4.5, 5.5, 6.9];
    let mut summary: Vec<String> = Vec::new();
    let mut pitch_rows: Vec<String> = Vec::new();
    let mut results: Vec<CamResult> = Vec::new();
    for cam in cams {
        let n = cam.dataset.num_views();
        let base = crate::route(&cam.dataset, &cam.seed)?;
        let tau = tilt_of(&base)?;
        let sig = crate::summarize(&base).sigma;
        let stated_tau = |name: &str| {
            sig.iter()
                .find(|(k, _)| k.ends_with(name))
                .map_or(f64::NAN, |(_, v)| *v)
        };
        let fpx = base.params.camera.intrinsics.fx * tilt_rotation(tau)[(2, 2)];
        let all_views: Vec<usize> = (0..n).collect();
        let obs = real_obs(cam, &all_views, &base.params.cam_se3_target, tau);
        let all: Vec<BlurObs> = obs.iter().map(|o| o.o).collect();
        let Some(fit) = fit_pof(&all, None) else {
            println!("{}: the focus fit failed ({} points)", cam.id, all.len());
            continue;
        };
        let a = axis_of_tilt(tau);
        let axis_c = [a.x / a.z, a.y / a.z];
        let fit_c0 = fit_pof(&all, Some([0.0, 0.0]));
        let fit_ca = fit_pof(&all, Some(axis_c));
        let quad = quad_fit_robust(&all);
        let blurs: Vec<f64> = all.iter().map(|o| o.blur).collect();
        let pred = |f: &PofFit| all.iter().map(|o| f.predict(&o.x)).collect::<Vec<_>>();
        let ev_phys = explained(&blurs, &pred(&fit));
        let ev_quad = quad.map(|q| {
            explained(
                &blurs,
                &all.iter()
                    .map(|o| quad_predict(&q, &o.x))
                    .collect::<Vec<_>>(),
            )
        });
        let ev_curv = fit_c0.as_ref().map(|f| explained(&blurs, &pred(f)));
        // Held out by view folds (blur only; the full route's geometry).
        let (mut ho, mut hp, mut hq) = (Vec::new(), Vec::new(), Vec::new());
        // Jointly: each fold's own route (tilt, poses) and focus fit.
        let (mut fold_pitch, mut fold_daz, mut fold_tau) = (Vec::new(), Vec::new(), Vec::new());
        let mut fold_pi: Vec<Vector3<f64>> = Vec::new();
        let mut fold_tf: Vec<f64> = Vec::new();
        for f in 0..folds {
            let train: Vec<BlurObs> = obs
                .iter()
                .filter(|o| o.view % folds != f)
                .map(|o| o.o)
                .collect();
            let test: Vec<&RealObs> = obs.iter().filter(|o| o.view % folds == f).collect();
            if let (Some(ft), Some(qt)) = (fit_pof(&train, None), quad_fit_robust(&train)) {
                for o in &test {
                    ho.push(o.o.blur);
                    hp.push(ft.predict(&o.o.x));
                    hq.push(quad_predict(&qt, &o.o.x));
                }
                fold_tf.push(tilt_from_pof(&ft.pi, fpx * 5.5e-6)[0]);
            }
            let tv: Vec<usize> = (0..n).filter(|v| v % folds != f).collect();
            let fr = crate::route(&crate::subset(&cam.dataset, &tv)?, &cam.seed)?;
            let ftau = tilt_of(&fr)?;
            let fo = real_obs(cam, &tv, &fr.params.cam_se3_target, ftau);
            let fobs: Vec<BlurObs> = fo.iter().map(|o| o.o).collect();
            let Some(ff) = fit_pof(&fobs, None) else {
                continue;
            };
            let ffpx = fr.params.camera.intrinsics.fx * tilt_rotation(ftau)[(2, 2)];
            let (p, d) = implied(&ff.pi, ftau, ffpx);
            fold_pitch.push(p);
            fold_daz.push(d);
            fold_tau.push(ftau);
            fold_pi.push(ff.pi);
        }
        let (pitch, daz) = implied(&fit.pi, tau, fpx);
        let jp = jack_sd(&fold_pitch);
        let jaz = jack_sd(&fold_daz);
        let jtau = [
            jack_sd(&fold_tau.iter().map(|t| t[0]).collect::<Vec<_>>()),
            jack_sd(&fold_tau.iter().map(|t| t[1]).collect::<Vec<_>>()),
        ];
        let jpi: Vec<f64> = (0..3)
            .map(|k| jack_sd(&fold_pi.iter().map(|p| p[k]).collect::<Vec<_>>()))
            .collect();
        // Stated σ of the pitch (independent markers, the tilt held) by the
        // delta method.
        let w = implied_inv_v0(&fit.pi, tau);
        let t = [a.x / a.z, a.y / a.z];
        let tt = t[0] * t[0] + t[1] * t[1];
        let dw = Vector3::new(t[0] / tt, t[1] / tt, -1.0);
        let sw = (dw.transpose() * fit.cov_pi * dw)[(0, 0)].sqrt();
        let sp_stated = 1e6 / (w * w * fpx) * sw;
        // Residual structure: image thirds, depth terciles, defocus sign.
        let res: Vec<f64> = all.iter().map(|o| o.blur - fit.predict(&o.x)).collect();
        let mut cells = [[(0.0, 0usize); 3]; 3];
        for (o, e) in obs.iter().zip(&res) {
            let i = ((o.px[0] / f64::from(cam.size[0]) * 3.0) as usize).min(2);
            let j = ((o.px[1] / f64::from(cam.size[1]) * 3.0) as usize).min(2);
            cells[j][i].0 += e;
            cells[j][i].1 += 1;
        }
        let cell_txt: Vec<String> = cells
            .iter()
            .map(|row| {
                row.iter()
                    .map(|c| format!("{:+.2}", c.0 / c.1.max(1) as f64))
                    .collect::<Vec<_>>()
                    .join(" ")
            })
            .collect();
        let mut zs: Vec<f64> = all.iter().map(|o| o.x.z).collect();
        zs.sort_by(f64::total_cmp);
        let (z1, z2) = (zs[zs.len() / 3], zs[2 * zs.len() / 3]);
        let mut terc = [(0.0, 0usize); 3];
        for (o, e) in all.iter().zip(&res) {
            let k = if o.x.z < z1 {
                0
            } else if o.x.z < z2 {
                1
            } else {
                2
            };
            terc[k].0 += e;
            terc[k].1 += 1;
        }
        // Sign asymmetry: residuals of points before against beyond the PoF
        // (|g| above the in-focus blur).
        let (mut near, mut far) = (Vec::new(), Vec::new());
        for (o, e) in all.iter().zip(&res) {
            let g = fit.g(&o.x);
            if g.abs() > fit.s0 {
                if g > 0.0 {
                    near.push(*e);
                } else {
                    far.push(*e);
                }
            }
        }
        let (mn, sn) = mean_sd(&near);
        let (mf, sf) = mean_sd(&far);
        let asym_z = (mn - mf) / (sn * sn / near.len() as f64 + sf * sf / far.len() as f64).sqrt();
        let theta = (fit.pi.x.hypot(fit.pi.y) / fit.pi.z).atan() * DEG;
        let pl = tilt_rotation(tau).transpose() * fit.pi;
        println!(
            "{}: {} markers in {} views, blur {:.2}–{:.2} px, depth {:.3}–{:.3} m (2–98 %), routed tilt ({:+.4}, {:+.4}) rad (stated ± {:.4}, {:.4}; fold jackknife ± {:.4}, {:.4}), f' {:.1} px",
            cam.id,
            all.len(),
            n,
            blurs.iter().copied().fold(f64::INFINITY, f64::min),
            blurs.iter().copied().fold(0.0, f64::max),
            zs[zs.len() / 50],
            zs[zs.len() - 1 - zs.len() / 50],
            tau[0],
            tau[1],
            stated_tau("tilt_x"),
            stated_tau("tilt_y"),
            jtau[0],
            jtau[1],
            fpx
        );
        println!(
            "  PoF K' frame π = ({:+.3}, {:+.3}, {:+.3}) /m (fold jackknife ± {:.3}, {:.3}, {:.3}; stated ± {:.3}, {:.3}, {:.3}); routed frame ({:+.3}, {:+.3}, {:+.3}); axial focus {:.3} m, PoF angle to the sensor plane {:.1}°",
            fit.pi.x,
            fit.pi.y,
            fit.pi.z,
            jpi[0],
            jpi[1],
            jpi[2],
            fit.cov_pi[(0, 0)].sqrt(),
            fit.cov_pi[(1, 1)].sqrt(),
            fit.cov_pi[(2, 2)].sqrt(),
            pl.x,
            pl.y,
            pl.z,
            1.0 / fit.pi.z,
            theta
        );
        println!(
            "  κ {:.3} px·m, σ0 {:.3} px, robust residual {:.3} px; explained variance of σ: focus model (5) {:.3}, + curvature (6) {:.3}, free quadratic (10) {:.3}; held out {:.3} / free {:.3}",
            fit.kappa,
            fit.s0,
            fit.scale,
            ev_phys,
            ev_curv.unwrap_or(f64::NAN),
            ev_quad.unwrap_or(f64::NAN),
            explained(&ho, &hp),
            explained(&ho, &hq)
        );
        println!(
            "  residual by image thirds (rows top→bottom) [{}] px; depth terciles {:+.3} {:+.3} {:+.3} px; before − beyond the PoF {:+.3} px (z {:+.1})",
            cell_txt.join(" | "),
            terc[0].0 / terc[0].1 as f64,
            terc[1].0 / terc[1].1 as f64,
            terc[2].0 / terc[2].1 as f64,
            mn - mf,
            asym_z,
        );
        for (label, fc) in [
            ("curvature about K' axis", &fit_c0),
            ("curvature about routed lens axis", &fit_ca),
        ] {
            if let Some(fc) = fc {
                let (c, sc) = fc.curvature().unwrap_or((f64::NAN, f64::NAN));
                let (p, d) = implied(&fc.pi, tau, fpx);
                println!(
                    "  {label}: c {:+.2} ± {:.2} /m, π ({:+.3}, {:+.3}, {:+.3}), implied pitch {:.2} µm, Δazimuth {:+.1}°",
                    c, sc, fc.pi.x, fc.pi.y, fc.pi.z, p, d
                );
            }
        }
        println!(
            "  implied pitch (thin lens, = pitch / m_p) {:.2} µm (fold jackknife with fold routes ± {:.2}; stated, tilt held ± {:.2}); direction: π_xy azimuth − tilt azimuth {:+.1}° (± {:.1}); folds {}; focus τx at 5.5 µm jackknife ± {:.4}",
            pitch,
            jp,
            sp_stated,
            daz,
            jaz,
            fold_pitch
                .iter()
                .map(|p| format!("{p:.2}"))
                .collect::<Vec<_>>()
                .join(" "),
            jack_sd(&fold_tf)
        );
        let mut row = cam.id.rsplit('_').next().unwrap_or("").to_string();
        for p in pitches {
            let v0 = fpx * p * 1e-6;
            let tf = tilt_from_pof(&fit.pi, v0);
            let c = tilt_cov(&fit.pi, &fit.cov_pi, v0);
            let ff = focal_from_pof(&fit.pi, v0);
            let nn = ff * fpx / (4.0 * fit.kappa);
            row.push_str(&format!(
                "|{:+.4} ± {:.4}, {:+.4} (F {:.2} mm, N {:.1})",
                tf[0],
                c[0][0].sqrt(),
                tf[1],
                ff * 1e3,
                nn
            ));
        }
        row.push_str(&format!("|{:+.4}, {:+.4}", tau[0], tau[1]));
        pitch_rows.push(row);
        summary.push(format!(
            "{}|{:+.2}|{:+.2}|{:+.2}|{:.0}|{:.3}|{:.3}|{:.3}|{:.3}|{:.3}|{:.2} ± {:.2}|{:+.1} ± {:.1}",
            cam.id.rsplit('_').next().unwrap_or(""),
            fit.pi.x,
            fit.pi.y,
            fit.pi.z,
            theta,
            fit.kappa,
            fit.s0,
            ev_phys,
            explained(&ho, &hp),
            explained(&ho, &hq),
            pitch,
            jp,
            daz,
            jaz
        ));
        results.push(CamResult {
            id: cam.id.clone(),
            tau,
            tau_jk: jtau,
            fpx,
            pi: fit.pi,
            pitch,
            pitch_jk: jp,
        });
    }
    println!(
        "\ncamera|πx /m|πy /m|πz /m|PoF ∠ °|κ px·m|σ0 px|EV|EV held|EV held free|pitch* µm|Δazimuth °"
    );
    for s in &summary {
        println!("{s}");
    }
    println!(
        "\ncamera|{}|routed tilt",
        pitches
            .iter()
            .map(|p| format!("τx, τy at {p} µm"))
            .collect::<Vec<_>>()
            .join("|")
    );
    for s in &pitch_rows {
        println!("{s}");
    }
    if results.len() < 2 {
        return Ok(());
    }
    let ps: Vec<f64> = results.iter().map(|x| x.pitch).collect();
    let (m, sd) = mean_sd(&ps);
    let wsum: f64 = results
        .iter()
        .map(|x| 1.0 / (x.pitch_jk * x.pitch_jk))
        .sum();
    let wm: f64 = results
        .iter()
        .map(|x| x.pitch / (x.pitch_jk * x.pitch_jk))
        .sum::<f64>()
        / wsum;
    let chi2: f64 = results
        .iter()
        .map(|x| ((x.pitch - wm) / x.pitch_jk).powi(2))
        .sum();
    println!(
        "\nimplied pitch across cameras: mean {m:.2} µm, sd {sd:.2} ({:.0} %), weighted mean {wm:.2}, χ² {chi2:.1} on {} dof (fold jackknife σ)",
        100.0 * sd / m,
        results.len() - 1
    );
    // At the common pitch: the focus tilt of every camera against its routed
    // tilt.
    println!(
        "at the mean implied pitch {m:.2} µm: camera|focus τx|routed τx ± jackknife|difference|focus τy|routed τy ± jackknife"
    );
    let (mut tf, mut td) = (Vec::new(), Vec::new());
    for r in &results {
        let t = tilt_from_pof(&r.pi, r.fpx * m * 1e-6);
        tf.push(t[0]);
        td.push(r.tau[0]);
        println!(
            "{}|{:+.4}|{:+.4} ± {:.4}|{:+.4}|{:+.4}|{:+.4} ± {:.4}",
            r.id.rsplit('_').next().unwrap_or(""),
            t[0],
            r.tau[0],
            r.tau_jk[0],
            t[0] - r.tau[0],
            t[1],
            r.tau[1],
            r.tau_jk[1]
        );
    }
    let (mf, sf) = mean_sd(&tf);
    let (md, sdd) = mean_sd(&td);
    let corr = {
        let c: f64 = tf
            .iter()
            .zip(&td)
            .map(|(a, b)| (a - mf) * (b - md))
            .sum::<f64>()
            / (tf.len() as f64 - 1.0);
        c / (sf * sdd)
    };
    println!(
        "across cameras: focus τx {mf:+.4} sd {sf:.4}, routed τx {md:+.4} sd {sdd:.4}, correlation {corr:+.2}"
    );
    Ok(())
}

// ───────────────────────── synthetic ─────────────────────────

/// A synthetic configuration.
#[derive(Debug, Clone)]
struct Cfg {
    label: String,
    tau: [f64; 2],
    f: f64,
    mp: f64,
    fnum: f64,
    z_focus: f64,
    /// Board-centre distance range, m.
    depth: [f64; 2],
    /// Board tilt range, rad.
    board_tilt: f64,
    views: usize,
    /// Base (in-focus) blur, px.
    s_base: f64,
    /// Relative and absolute blur noise, and a per-view relative jitter.
    noise: [f64; 3],
    fc: f64,
    /// Exponent of the base/defocus combination (2: quadrature).
    q: f64,
}

/// The baseline: the studied rig's camera as its blur fit describes it
/// (`f' ≈ 1145 px`; the PoF's slope `π_y ≈ −15 /m` at `τ ≈ −0.09` is a
/// thin lens of `F ≈ 6 mm`; axial focus 0.23 m in the lens frame; `κ ≈
/// 1.4 px·m` is an entrance pupil of 4.9 mm, f/1.2; base blur 0.72 px;
/// robust blur residual ≈ 0.11 px).
fn baseline() -> Cfg {
    Cfg {
        label: "baseline".into(),
        tau: [-0.09, 0.0],
        f: 0.006,
        mp: 1.0,
        fnum: 1.22,
        z_focus: 0.23,
        depth: [0.17, 0.33],
        board_tilt: 0.45,
        views: 20,
        s_base: 0.72,
        noise: [0.04, 0.06, 0.0],
        fc: 0.0,
        q: 2.0,
    }
}

/// One synthetic camera: the lens and the noise-free blur of every marker
/// in the `K'` frame of the distortion-derived tilt.
struct Scene {
    tau_d: [f64; 2],
    v0: f64,
    fpx: f64,
    /// (view, K'-frame point, noise-free blur).
    pts: Vec<(usize, Vector3<f64>, f64)>,
}

fn scene(c: &Cfg, seed: u64) -> Scene {
    let pitch = 7e-6;
    let mut lens = Lens::new(c.f, c.mp, c.z_focus, c.tau, pitch, c.fnum, [364.0, 271.0]);
    lens.fc = c.fc;
    // Keep f' at the studied camera's value: the pitch follows the focal
    // length (so the magnification sweep changes the magnification, not
    // the pixel scale).
    let (k, _) = lens.pinhole();
    lens.pitch *= k[(0, 0)] / 1145.0;
    let (k, _) = lens.pinhole();
    let tau_d = lens.tilt_d();
    let r = tilt_rotation(tau_d);
    let mut rng = Mix(seed);
    let mut pts = Vec::new();
    let board: Vec<[f64; 2]> = (0..15)
        .flat_map(|j| {
            (0..14).map(move |i| {
                [
                    0.008 * (f64::from(i) + 0.5 * f64::from(j % 2)),
                    0.008 * 0.866_025_4 * f64::from(j),
                ]
            })
        })
        .collect();
    let half = [360.0 / 1145.0, 270.0 / 1145.0];
    for v in 0..c.views {
        let rot = UnitQuaternion::from_euler_angles(
            rng.range(-c.board_tilt, c.board_tilt),
            rng.range(-c.board_tilt, c.board_tilt),
            rng.range(-0.6, 0.6),
        );
        let z = rng.range(c.depth[0], c.depth[1]);
        let centre = Vector3::new(
            rng.range(-0.6, 0.6) * half[0] * z,
            rng.range(-0.6, 0.6) * half[1] * z,
            z,
        );
        let t = centre - rot * Vector3::new(0.056, 0.0485, 0.0);
        let pose = Iso3::from_parts(Translation3::from(t), rot);
        let jitter = 1.0 + c.noise[2] * rng.normal();
        for b in &board {
            let x = pose * Point3::new(b[0], b[1], 0.0);
            if x.z <= 0.01 {
                continue;
            }
            let u = lens.project(&x.coords);
            if !(0.0..720.0).contains(&u[0]) || !(0.0..540.0).contains(&u[1]) {
                continue;
            }
            let spot = lens.spot_sigma(&x.coords, 64);
            let s = (c.s_base.powf(c.q) + spot.powf(c.q)).powf(1.0 / c.q) * jitter;
            pts.push((v, r * x.coords, s));
        }
    }
    Scene {
        tau_d,
        v0: k[(0, 0)] * lens.pitch,
        fpx: k[(0, 0)],
        pts,
    }
}

#[derive(Default)]
struct CellStats {
    ex: Vec<f64>,
    ey: Vec<f64>,
    stated: Vec<f64>,
    ratio: Vec<f64>,
    daz: Vec<f64>,
    ev: Vec<f64>,
    curv_z: Vec<f64>,
    fails: usize,
    tau_d: [f64; 2],
    mu: f64,
    npts: f64,
}

fn run_cell(c: &Cfg, draws: usize) -> CellStats {
    let mut st = CellStats::default();
    let lens = Lens::new(c.f, c.mp, c.z_focus, c.tau, 7e-6, c.fnum, [364.0, 271.0]);
    st.mu = lens.magnification();
    for d in 0..draws {
        let sc = scene(c, 1000 + d as u64);
        st.tau_d = sc.tau_d;
        st.npts += sc.pts.len() as f64 / draws as f64;
        let mut rng = Mix(77 + 31 * d as u64);
        let obs: Vec<BlurObs> = sc
            .pts
            .iter()
            .map(|(_, x, s)| BlurObs {
                x: *x,
                blur: (s * (1.0 + c.noise[0] * rng.normal()) + c.noise[1] * rng.normal()).max(0.2),
            })
            .collect();
        let Some(fit) = fit_pof(&obs, None) else {
            st.fails += 1;
            continue;
        };
        let tf = tilt_from_pof(&fit.pi, sc.v0);
        let e = [(tf[0] - sc.tau_d[0]) * DEG, (tf[1] - sc.tau_d[1]) * DEG];
        if e[0].hypot(e[1]) > 5.0 {
            st.fails += 1;
            continue;
        }
        st.ex.push(e[0]);
        st.ey.push(e[1]);
        let cv = tilt_cov(&fit.pi, &fit.cov_pi, sc.v0);
        st.stated.push(cv[0][0].sqrt() * DEG);
        let td = sc.tau_d;
        if td[0].hypot(td[1]) > 1e-3 {
            let w = implied_inv_v0(&fit.pi, td);
            st.ratio.push(1.0 / (w * sc.v0));
            let a = axis_of_tilt(td);
            st.daz
                .push(wrap(fit.pi.y.atan2(fit.pi.x) - a.y.atan2(a.x)) * DEG);
        }
        let b: Vec<f64> = obs.iter().map(|o| o.blur).collect();
        let p: Vec<f64> = obs.iter().map(|o| fit.predict(&o.x)).collect();
        st.ev.push(explained(&b, &p));
        if let Some(fc) = fit_pof(&obs, Some([0.0, 0.0])) {
            st.curv_z.push(fc.beta[4] / fc.cov[4][4].sqrt());
        }
        let _ = sc.fpx;
    }
    st
}

fn cell_line(c: &Cfg, st: &CellStats) -> String {
    let (bx, sx) = mean_sd(&st.ex);
    let (by, sy) = mean_sd(&st.ey);
    let (ms, _) = mean_sd(&st.stated);
    let (rm, rs) = mean_sd(&st.ratio);
    let (am, asd) = mean_sd(&st.daz);
    let (ev, _) = mean_sd(&st.ev);
    let (cz, _) = mean_sd(&st.curv_z.iter().map(|v| v.abs()).collect::<Vec<_>>());
    format!(
        "{}|{:+.2}, {:+.2}|{:.3}|{:.0}|{:+.3} ± {:.3}|{:+.3} ± {:.3}|{:.3}|{:.2}|{:.3} ± {:.3}|{:+.2} ± {:.2}|{:.3}|{:.1}|{}",
        c.label,
        st.tau_d[0] * DEG,
        st.tau_d[1] * DEG,
        st.mu,
        st.npts,
        bx,
        sx,
        by,
        sy,
        ms,
        sx / ms,
        rm,
        rs,
        am,
        asd,
        ev,
        cz,
        st.fails
    )
}

/// The synthetic sweep.
pub fn sweep(draws: usize) -> Result<()> {
    let b = baseline();
    let mut cells: Vec<Cfg> = vec![b.clone()];
    let with = |label: &str, f: &dyn Fn(&mut Cfg)| {
        let mut c = baseline();
        c.label = label.to_string();
        f(&mut c);
        c
    };
    for (tx, ty) in [
        (0.0, 0.0),
        (-2.0, 0.0),
        (-5.0, 0.0),
        (-10.0, 0.0),
        (-15.0, 0.0),
        (0.0, 5.0),
        (0.0, 15.0),
        (-5.0, 5.0),
        (-10.0, 10.0),
    ] {
        cells.push(with(&format!("tilt ({tx}°, {ty}°)"), &|c| {
            c.tau = [tx / DEG, ty / DEG];
        }));
    }
    for f in [0.004, 0.012, 0.025] {
        // Same entrance pupil diameter: the blur scale stays.
        cells.push(with(&format!("F {} mm", f * 1e3), &|c| {
            c.fnum *= f / c.f;
            c.f = f;
        }));
    }
    for mp in [0.7, 0.85, 1.2, 1.4] {
        cells.push(with(&format!("m_p {mp}"), &|c| c.mp = mp));
    }
    for n in [2.0, 2.8, 4.0, 8.0, 16.0] {
        cells.push(with(&format!("f/{n}"), &|c| c.fnum = n));
    }
    for (lo, hi, bt, l) in [
        (0.23, 0.27, 0.45, "depth 0.23–0.27 m"),
        (0.245, 0.255, 0.45, "depth 0.245–0.255 m"),
        (0.17, 0.33, 0.1, "board tilt ±0.1 rad"),
        (0.245, 0.255, 0.05, "one distance, frontal"),
        (0.45, 0.6, 0.45, "depth 0.45–0.6 m"),
    ] {
        cells.push(with(l, &|c| {
            c.depth = [lo, hi];
            c.board_tilt = bt;
        }));
    }
    for v in [3, 5, 10, 40] {
        cells.push(with(&format!("{v} views"), &|c| c.views = v));
    }
    for (rel, abs, view, l) in [
        (0.02, 0.01, 0.0, "noise 2 % + 0.01 px"),
        (0.10, 0.05, 0.0, "noise 10 % + 0.05 px"),
        (0.20, 0.10, 0.0, "noise 20 % + 0.1 px"),
        (0.05, 0.03, 0.05, "+ per-view 5 %"),
    ] {
        cells.push(with(l, &|c| c.noise = [rel, abs, view]));
    }
    for s in [0.3, 1.2] {
        cells.push(with(&format!("base blur {s} px"), &|c| c.s_base = s));
    }
    for fc in [1.0, 3.0, 6.0] {
        cells.push(with(&format!("field curvature {fc} /m"), &|c| c.fc = fc));
    }
    for q in [1.5, 3.0] {
        cells.push(with(&format!("combination exponent {q}"), &|c| c.q = q));
    }
    cells.push(with("m_p 0.7, tilt (−10°, 5°)", &|c| {
        c.mp = 0.7;
        c.tau = [-10.0 / DEG, 5.0 / DEG];
    }));
    println!("\n## Focus sweep: {draws} draws per cell (noise and view geometry)");
    println!(
        "cell|τ_d °|μ|markers|τx error °|τy error °|stated σ(τx) °|scatter/stated|pitch*/pitch|Δazimuth °|EV|curvature |z||fails"
    );
    for c in &cells {
        let st = run_cell(c, draws);
        println!("{}", cell_line(c, &st));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The baseline synthetic camera recovers the tilt from its blur with
    /// no bias beyond its scatter.
    #[test]
    fn baseline_recovers_the_tilt() {
        let st = run_cell(&baseline(), 6);
        assert_eq!(st.fails, 0);
        let (bx, sx) = mean_sd(&st.ex);
        assert!(bx.abs() < 0.1 + 2.0 * sx, "bias {bx}° scatter {sx}°");
        let (r, _) = mean_sd(&st.ratio);
        assert!((r - 1.0).abs() < 0.05, "pitch ratio {r}");
    }
}
