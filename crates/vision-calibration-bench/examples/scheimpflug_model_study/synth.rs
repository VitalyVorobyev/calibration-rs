//! `--parts synth`: synthetic ground truth for the camera terms of this
//! study, scored exactly as the datasets are (4-fold held-out views).
//!
//! A scene is a Scheimpflug camera like the studied ones (720×540,
//! `f ≈ 1145 px`, `k1 ≈ −0.41`, `tilt_x ≈ −0.096 rad`) looking at a 5 mm
//! grid from 170–330 mm, with seeded poses and Gaussian pixel noise. The
//! truth is a [`GenCfg`] with its [`GenParams`], so a scene can be drawn from
//! the plain model, from a structured term, or from a free field outside
//! every parametric term.

use std::collections::BTreeMap;

use anyhow::Result;
use nalgebra::{Point2, Point3, Rotation3, Translation3, UnitQuaternion, Vector3};
use vision_calibration::common::IntrinsicsManualInit;
use vision_calibration_core::{
    BrownConrady5, CorrespondenceView, DistortionParams, FxFyCxCy, Iso3, NoMeta, PlanarDataset,
    ScheimpflugParams, View,
};

use crate::CameraData;
use crate::model::{GenCfg, GenParams, project_point};

pub const SIZE: [u32; 2] = [720, 540];

/// The studied cameras' typical values (route solution of the ring-grid
/// rig).
pub fn typical() -> GenParams {
    GenParams {
        intr: [1140.0, 1149.0, 364.0, 271.0],
        dist: [-0.415, 0.24, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        tilt: [-0.096, 0.004],
        dc: [0.0; 2],
        depth: 0.0,
        pupil: 0.0,
        board: Vec::new(),
        field: Vec::new(),
        sensor: Vec::new(),
        misc: [0.0; 4],
        spline: Vec::new(),
        board_centre: [0.0; 2],
    }
}

/// SplitMix64: a fixed, platform-independent stream.
struct Mix(u64);

impl Mix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform in `[0, 1)`.
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn range(&mut self, a: f64, b: f64) -> f64 {
        a + (b - a) * self.unit()
    }
    /// Standard normal (Box–Muller).
    fn normal(&mut self) -> f64 {
        let u1 = self.unit().max(1e-300);
        let u2 = self.unit();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

/// Board poses: tilts up to 0.45 rad, roll up to 0.6 rad, the board centre
/// at 170–330 mm and spread over the image.
pub fn poses(n: usize, seed: u64) -> Vec<Iso3> {
    let mut rng = Mix(seed);
    (0..n)
        .map(|_| {
            let r = Rotation3::from_euler_angles(
                rng.range(-0.45, 0.45),
                rng.range(-0.45, 0.45),
                rng.range(-0.6, 0.6),
            );
            let z = rng.range(0.17, 0.33);
            let c = Vector3::new(rng.range(-0.1, 0.1) * z, rng.range(-0.07, 0.07) * z, z);
            // Board centre (57.5, 42.5) mm placed at `c`.
            let t = c - r * Vector3::new(0.0575, 0.0425, 0.0);
            Iso3::from_parts(
                Translation3::from(t),
                UnitQuaternion::from_rotation_matrix(&r),
            )
        })
        .collect()
}

/// A camera's synthetic views: a 24 × 18 grid at 5 mm, the points inside
/// the image, noise `sigma_px` per axis; `seeds` are the geometry's and the
/// noise's. `extra` adds a displacement (px) at
/// the projected position: a camera-fixed field outside the model.
pub fn scene(
    id: &str,
    cfg: &GenCfg,
    truth: &GenParams,
    n_views: usize,
    seeds: (u64, u64),
    sigma_px: f64,
    extra: &dyn Fn([f64; 2]) -> [f64; 2],
) -> Result<CameraData> {
    Ok(scene_with_poses(id, cfg, truth, n_views, seeds, sigma_px, extra)?.0)
}

/// [`scene`] with the true pose of each kept view.
pub fn scene_with_poses(
    id: &str,
    cfg: &GenCfg,
    truth: &GenParams,
    n_views: usize,
    seeds: (u64, u64),
    sigma_px: f64,
    extra: &dyn Fn([f64; 2]) -> [f64; 2],
) -> Result<(CameraData, Vec<Iso3>)> {
    let (geom_seed, noise_seed) = seeds;
    let mut kept = Vec::new();
    let mut rng = Mix(noise_seed ^ 0x5EED);
    let mut views = Vec::new();
    for pose in poses(n_views, geom_seed) {
        let (mut p3, mut p2) = (Vec::new(), Vec::new());
        for j in 0..18 {
            for i in 0..24 {
                let b = [i as f64 * 0.005, j as f64 * 0.005];
                if (pose * Point3::new(b[0], b[1], 0.0)).z <= 0.0 {
                    continue;
                }
                let uv = project_point(cfg, truth, SIZE, &pose, b);
                let e = extra(uv);
                let uv = [
                    uv[0] + e[0] + sigma_px * rng.normal(),
                    uv[1] + e[1] + sigma_px * rng.normal(),
                ];
                if uv[0] < 2.0
                    || uv[1] < 2.0
                    || uv[0] > f64::from(SIZE[0]) - 2.0
                    || uv[1] > f64::from(SIZE[1]) - 2.0
                {
                    continue;
                }
                p3.push(Point3::new(b[0], b[1], 0.0));
                p2.push(Point2::new(uv[0], uv[1]));
            }
        }
        if p3.len() >= 30 {
            views.push(View {
                obs: CorrespondenceView::new(p3, p2)?,
                meta: NoMeta,
            });
            kept.push(pose);
        }
    }
    // The route is seeded with the plain model's part of the truth.
    let mut seed_init = IntrinsicsManualInit::default();
    seed_init.intrinsics = Some(FxFyCxCy {
        fx: truth.intr[0],
        fy: truth.intr[1],
        cx: truth.intr[2],
        cy: truth.intr[3],
    });
    seed_init.distortion = Some(DistortionParams::BrownConrady5 {
        params: BrownConrady5 {
            k1: truth.dist[0],
            k2: truth.dist[1],
            k3: 0.0,
            p1: 0.0,
            p2: 0.0,
            iters: 8,
        },
    });
    seed_init.sensor = Some(ScheimpflugParams {
        tilt_x: truth.tilt[0],
        tilt_y: truth.tilt[1],
    });
    Ok((
        CameraData {
            id: id.to_string(),
            dataset: PlanarDataset::new(views)?,
            seed: seed_init,
            size: SIZE,
            blur: BTreeMap::new(),
            captures: Vec::new(),
        },
        kept,
    ))
}

/// A smooth camera-fixed field outside every parametric term of the study
/// (and outside the spline's span): quartic and cubic image monomials, up to
/// about 0.6 px in the corners.
pub fn outside_field(uv: [f64; 2]) -> [f64; 2] {
    let x = (uv[0] - 360.0) / 360.0;
    let y = (uv[1] - 270.0) / 270.0;
    [
        0.30 * x * x * y * y - 0.18 * x * x * x * y + 0.15 * y * y * y,
        0.24 * x * x * x * x - 0.21 * x * y * y * y + 0.12 * x * x * y,
    ]
}

/// The spline of the proposed term in the synthetic truth.
pub fn spline_cfg() -> GenCfg {
    GenCfg {
        spline: [6, 5],
        spline_sigma: 10.0,
        spline_ortho: true,
        ..GenCfg::base("gen spline6x5 ortho s10")
    }
}

/// A scene drawn from the proposed model: the typical camera plus a smooth
/// spline field (≈ 0.15 px RMS) with no component along the camera's own
/// parameters and a common rigid motion on this scene's points. Returns the
/// scene, the truth's model and its parameters.
pub fn spline_scene(
    id: &str,
    geom_seed: u64,
    noise_seed: u64,
) -> Result<(CameraData, GenCfg, GenParams)> {
    let cfg = spline_cfg();
    let mut truth = typical();
    // The plain scene's geometry: where the constraint is evaluated.
    let plain = GenCfg::base("plain");
    let (geo, kept) = scene_with_poses(
        id,
        &plain,
        &truth,
        20,
        (geom_seed, noise_seed),
        0.0,
        &|_| [0.0; 2],
    )?;
    let n = crate::model::spline_len(cfg.spline);
    let nx = cfg.spline[0] + 3;
    let mut c = vec![0.0; 2 * n];
    for k in 0..n {
        let (i, j) = ((k % nx) as f64, (k / nx) as f64);
        c[k] = 0.6 * (0.9 * i + 0.4 * j).cos();
        c[n + k] = 0.6 * (0.5 * i - 0.7 * j).sin();
    }
    let rows = crate::model::ortho_rows(
        &cfg,
        &[],
        &geo.dataset.views,
        &truth,
        &kept,
        [f64::from(SIZE[0]), f64::from(SIZE[1])],
    );
    truth.spline = project_out(&c, &rows);
    let cam = scene(id, &cfg, &truth, 20, (geom_seed, noise_seed), 0.07, &|_| {
        [0.0; 2]
    })?;
    Ok((cam, cfg, truth))
}

/// `c` with its least-squares component along the span of `rows` removed.
pub fn project_out(c: &[f64], rows: &[Vec<f64>]) -> Vec<f64> {
    let m = rows.len();
    let mut g = nalgebra::DMatrix::<f64>::zeros(m, m);
    let mut b = nalgebra::DVector::<f64>::zeros(m);
    for i in 0..m {
        b[i] = rows[i].iter().zip(c).map(|(a, x)| a * x).sum();
        for j in 0..m {
            g[(i, j)] = rows[i].iter().zip(&rows[j]).map(|(a, x)| a * x).sum();
        }
        // A small ridge: some rows are nearly parallel (cx and a rotation
        // about y move the image alike).
        g[(i, i)] += 1e-9;
    }
    let y = g
        .cholesky()
        .map(|ch| ch.solve(&b))
        .unwrap_or_else(|| b.clone() * 0.0);
    let mut out = c.to_vec();
    for i in 0..m {
        for (o, a) in out.iter_mut().zip(&rows[i]) {
            *o -= y[i] * a;
        }
    }
    out
}

/// One estimated quantity of a fit, read from its parameters.
type Probe = Box<dyn Fn(&GenParams) -> f64>;

/// Monte-Carlo of the proposed term's estimates against their stated σ:
/// `n` noise draws on one geometry drawn from [`spline_scene`]'s truth, each
/// fitted from the truth. Returns report lines: per parameter the truth,
/// the mean estimate, the Monte-Carlo standard deviation and the mean
/// stated σ.
pub fn monte_carlo(n: usize, geom_seed: u64) -> Result<Vec<String>> {
    let (_, cfg, truth) = spline_scene("mc", geom_seed, 0)?;
    let plain = GenCfg::base("gen base");
    let ns = crate::model::spline_len(cfg.spline);
    let nx = cfg.spline[0] + 3;
    let centre = (cfg.spline[1] + 3) / 2 * nx + nx / 2;
    let probes: Vec<(String, Probe)> = vec![
        ("intr.fx".into(), Box::new(|p: &GenParams| p.intr[0])),
        ("intr.fy".into(), Box::new(|p: &GenParams| p.intr[1])),
        ("intr.cx".into(), Box::new(|p: &GenParams| p.intr[2])),
        ("intr.cy".into(), Box::new(|p: &GenParams| p.intr[3])),
        ("dist.k1".into(), Box::new(|p: &GenParams| p.dist[0])),
        ("dist.k2".into(), Box::new(|p: &GenParams| p.dist[1])),
        ("tilt.tilt_x".into(), Box::new(|p: &GenParams| p.tilt[0])),
        ("tilt.tilt_y".into(), Box::new(|p: &GenParams| p.tilt[1])),
        (
            format!("spline.{centre}"),
            Box::new(move |p: &GenParams| p.spline[centre]),
        ),
        (
            format!("spline.{}", centre + 1),
            Box::new(move |p: &GenParams| p.spline[centre + 1]),
        ),
        (
            format!("spline.{}", ns + centre),
            Box::new(move |p: &GenParams| p.spline[ns + centre]),
        ),
        (
            format!("spline.{}", ns + centre - nx),
            Box::new(move |p: &GenParams| p.spline[ns + centre - nx]),
        ),
    ];
    let mut lines = Vec::new();
    let penalty = GenCfg {
        spline_penalty: true,
        ..cfg.clone()
    };
    for (label, model, truth_used) in [
        ("spline truth, proposed term", &cfg, &truth),
        (
            "spline truth, proposed term, prior as a penalty",
            &penalty,
            &truth,
        ),
        ("plain truth, plain model", &plain, &typical()),
    ] {
        let mut vals: Vec<Vec<f64>> = vec![Vec::new(); probes.len()];
        let mut sig: Vec<Vec<f64>> = vec![Vec::new(); probes.len()];
        let mut rms = Vec::new();
        for k in 1..=n as u64 {
            let truth_cfg = if model.spline == [0, 0] { &plain } else { &cfg };
            let (cam, poses) = scene_with_poses(
                "mc",
                truth_cfg,
                truth_used,
                20,
                (geom_seed, k),
                0.07,
                &|_| [0.0; 2],
            )?;
            let ids: Vec<usize> = (0..cam.dataset.num_views()).collect();
            let fit = crate::model::fit(
                model,
                &ids,
                &cam.dataset.views,
                truth_used,
                &poses,
                SIZE,
                crate::model::Free::All,
                200,
            )?;
            rms.push(crate::rms_median(&fit.residuals).0);
            for (i, (name, f)) in probes.iter().enumerate() {
                if name.starts_with("spline") && model.spline == [0, 0] {
                    continue;
                }
                vals[i].push(f(&fit.params));
                if let Some(s) = fit.sigma.get(name) {
                    sig[i].push(*s);
                }
            }
        }
        lines.push(format!(
            "## Monte-Carlo, {label}: {n} noise draws, geometry {geom_seed}, rms {:.4}",
            rms.iter().sum::<f64>() / rms.len() as f64
        ));
        lines.push("parameter | truth | mean | MC sd | mean stated σ | MC sd / σ".into());
        for (i, (name, f)) in probes.iter().enumerate() {
            if vals[i].is_empty() {
                continue;
            }
            let (m, sd) = crate::mean_std(&vals[i]);
            let s = sig[i].iter().sum::<f64>() / sig[i].len().max(1) as f64;
            lines.push(format!(
                "{name} | {:.5} | {m:.5} | {sd:.5} | {s:.5} | {:.2}",
                f(truth_used),
                sd / s
            ));
        }
    }
    Ok(lines)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{self, Free};

    fn fit_all(cfg: &GenCfg, cam: &CameraData, init: &GenParams, poses: &[Iso3]) -> model::GenFit {
        let ids: Vec<usize> = (0..cam.dataset.num_views()).collect();
        model::fit(
            cfg,
            &ids,
            &cam.dataset.views,
            init,
            poses,
            SIZE,
            Free::All,
            100,
        )
        .expect("a synthetic fit")
    }

    /// A field evaluated at the observed pixel is not a camera model: on data
    /// from the plain model it pulls `k1` far from the truth and returns a
    /// residual below the noise it was given. Evaluated at the model's own
    /// projection, the same field gains nothing and leaves `k1` alone.
    #[test]
    fn a_field_at_the_observation_contracts_the_noise() {
        let plain = GenCfg::base("plain");
        let truth = typical();
        let (cam, poses) =
            scene_with_poses("t", &plain, &truth, 10, (3, 3), 0.07, &|_| [0.0; 2]).unwrap();
        let base = fit_all(&plain, &cam, &truth, &poses);
        let field = GenCfg {
            field_deg: 4,
            ..GenCfg::base("field4")
        };
        let at_obs = GenCfg {
            field_at_obs: true,
            ..field.clone()
        };
        let proper = fit_all(&field, &cam, &truth, &poses);
        let eiv = fit_all(&at_obs, &cam, &truth, &poses);
        let rms = |f: &model::GenFit| crate::rms_median(&f.residuals).0;
        // 48 field terms over ~2·1500 residuals: a fair field may lower the
        // RMS by about 1 %.
        assert!(
            rms(&proper) > 0.98 * rms(&base),
            "{} vs {}",
            rms(&proper),
            rms(&base)
        );
        assert!((proper.params.dist[0] - truth.dist[0]).abs() < 0.05);
        assert!(
            rms(&eiv) < 0.93 * rms(&base),
            "{} vs {}",
            rms(&eiv),
            rms(&base)
        );
        assert!((eiv.params.dist[0] - truth.dist[0]).abs() > 1.0);
    }

    /// Data drawn from the proposed term (camera + an orthogonal spline):
    /// the camera and the field are recovered, and a field outside the
    /// spline's span is approximated to well below the noise.
    #[test]
    fn the_spline_term_recovers_its_truth() {
        let (cam, cfg, truth) = spline_scene("t", 5, 5).unwrap();
        let (_, poses) =
            scene_with_poses("t", &cfg, &truth, 20, (5, 5), 0.07, &|_| [0.0; 2]).unwrap();
        let mut start = truth.clone();
        start.spline = Vec::new();
        let fit = fit_all(&cfg, &cam, &start, &poses);
        let p = &fit.params;
        assert!((p.intr[0] - truth.intr[0]).abs() < 1.0, "fx {}", p.intr[0]);
        assert!((p.intr[3] - truth.intr[3]).abs() < 1.0, "cy {}", p.intr[3]);
        assert!(
            (p.tilt[0] - truth.tilt[0]).abs() < 1e-3,
            "tilt_x {}",
            p.tilt[0]
        );
        assert!((p.dist[0] - truth.dist[0]).abs() < 5e-3, "k1 {}", p.dist[0]);
        // The field at the observed points, against the truth's.
        let n = model::spline_len(cfg.spline);
        let mut err = 0.0;
        let mut mag = 0.0;
        let mut cnt = 0.0;
        for v in &cam.dataset.views {
            for q in &v.obs.points_2d {
                let size = [f64::from(SIZE[0]), f64::from(SIZE[1])];
                let a = model::spline_eval(cfg.spline, size, &p.spline, q.x, q.y);
                let b = model::spline_eval(cfg.spline, size, &truth.spline, q.x, q.y);
                err += (a.0 - b.0).powi(2) + (a.1 - b.1).powi(2);
                mag += b.0.powi(2) + b.1.powi(2);
                cnt += 1.0;
            }
        }
        let (err, mag) = ((err / cnt).sqrt(), (mag / cnt).sqrt());
        assert!(mag > 0.1, "the truth's field is {mag} px");
        assert!(err < 0.15 * mag, "field error {err} px against {mag} px");
        assert_eq!(p.spline.len(), 2 * n);
    }

    /// On data from the plain model the spline term gains no more than its
    /// degrees of freedom allow and leaves the camera where the plain model
    /// puts it.
    #[test]
    fn the_spline_term_gains_nothing_on_the_plain_model() {
        let plain = GenCfg::base("plain");
        let truth = typical();
        let (cam, poses) =
            scene_with_poses("t", &plain, &truth, 12, (9, 9), 0.07, &|_| [0.0; 2]).unwrap();
        let base = fit_all(&plain, &cam, &truth, &poses);
        let spl = fit_all(&spline_cfg(), &cam, &truth, &poses);
        let rms = |f: &model::GenFit| crate::rms_median(&f.residuals).0;
        assert!(
            rms(&spl) > 0.97 * rms(&base),
            "{} vs {}",
            rms(&spl),
            rms(&base)
        );
        assert!((spl.params.intr[0] - base.params.intr[0]).abs() < 0.5);
        assert!((spl.params.tilt[0] - base.params.tilt[0]).abs() < 5e-4);
        assert!((spl.params.dist[0] - base.params.dist[0]).abs() < 2e-3);
    }
}
