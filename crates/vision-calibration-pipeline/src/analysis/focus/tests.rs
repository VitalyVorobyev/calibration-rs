//! Tests of the focus fit: recovery from the exact thick-lens blur,
//! the local image scale under distortion, the derived quantities, the
//! typed reasons, determinism and the Monte-Carlo σ.

use nalgebra::{Matrix3, Point2, Point3, Translation3, UnitQuaternion, Vector3};

use vision_calibration_core::{
    CameraParams, CorrespondenceView, DistortionKind, DistortionParams, FxFyCxCy, Iso3,
    PlanarDataset, PointBlur, ProjectionParams, SensorKind, SensorParams, View,
};

use super::*;

// ─── The exact paraxial thick lens ──────────────────────────────────────────

/// `R = Ry(τy)·Rx(τx)`, the library's tilt rotation: a point of the lens
/// frame `X` is `R·X` in the sensor frame.
fn tilt_rotation(tau: [f64; 2]) -> Matrix3<f64> {
    let (sx, cx) = tau[0].sin_cos();
    let (sy, cy) = tau[1].sin_cos();
    let rx = Matrix3::new(1.0, 0.0, 0.0, 0.0, cx, sx, 0.0, -sx, cx);
    let ry = Matrix3::new(cy, 0.0, -sy, 0.0, 1.0, 0.0, sy, 0.0, cy);
    ry * rx
}

/// RQ decomposition of a 3×3 matrix with `det > 0`: `m = k·r`, `k` upper
/// triangular with a positive diagonal, `r` a rotation.
fn rq3(m: &Matrix3<f64>) -> (Matrix3<f64>, Matrix3<f64>) {
    let m3: Vector3<f64> = m.row(2).transpose();
    let m2: Vector3<f64> = m.row(1).transpose();
    let m1: Vector3<f64> = m.row(0).transpose();
    let k33 = m3.norm();
    let r3 = m3 / k33;
    let k23 = m2.dot(&r3);
    let u2 = m2 - k23 * r3;
    let k22 = u2.norm();
    let r2 = u2 / k22;
    let k13 = m1.dot(&r3);
    let k12 = m1.dot(&r2);
    let u1 = m1 - k13 * r3 - k12 * r2;
    let k11 = u1.norm();
    let r1 = u1 / k11;
    let k = Matrix3::new(k11, k12, k13, 0.0, k22, k23, 0.0, 0.0, k33);
    let r = Matrix3::from_rows(&[r1.transpose(), r2.transpose(), r3.transpose()]);
    (k, r)
}

/// A paraxial thick lens with a tilted sensor, in the lens frame (object
/// space, origin at the entrance pupil, `z` along the axis): focal length
/// `f`, pupil magnification `mp`, the sensor at axial image distance `s_img`
/// from the rear principal plane, tilted by `tau_s`. The spot of a point is
/// the entrance pupil's shadow cast through it onto the plane of sharp
/// focus (the sensor's conjugate), imaged by the chief-ray map.
struct Lens {
    f: f64,
    mp: f64,
    s_img: f64,
    tau_s: [f64; 2],
    pitch: f64,
    fnum: f64,
    centre: [f64; 2],
}

impl Lens {
    fn new(
        f: f64,
        mp: f64,
        z_focus: f64,
        tau: [f64; 2],
        pitch: f64,
        fnum: f64,
        centre: [f64; 2],
    ) -> Self {
        let zh = f * (1.0 - mp) / mp;
        let s0 = z_focus - zh;
        Self {
            f,
            mp,
            s_img: 1.0 / (1.0 / f - 1.0 / s0),
            tau_s: tau,
            pitch,
            fnum,
            centre,
        }
    }

    fn pupil(&self) -> f64 {
        self.f / self.fnum
    }

    fn slope(&self) -> [f64; 2] {
        let r3 = tilt_rotation(self.tau_s).row(2).transpose();
        [r3.x / r3.z, r3.y / r3.z]
    }

    /// The object point conjugate to the image-space point `(x, y, w)`.
    fn conj(&self, y: [f64; 3]) -> Vector3<f64> {
        let s = 1.0 / (1.0 / self.f - 1.0 / y[2]);
        let m = -y[2] / s;
        let zh = self.f * (1.0 - self.mp) / self.mp;
        Vector3::new(y[0] / m, y[1] / m, s + zh)
    }

    /// The plane of sharp focus in the lens frame, `π·X = 1`.
    fn pof(&self) -> Vector3<f64> {
        let h = 1e-3;
        let g = self.slope();
        let p = |x: f64, y: f64| self.conj([x, y, self.s_img + g[0] * x + g[1] * y]);
        let (a, b, c) = (p(0.0, 0.0), p(h, 0.0), p(0.0, h));
        let n = (b - a).cross(&(c - a));
        n / n.dot(&a)
    }

    /// The chief-ray map: lens-frame point to homogeneous pixel.
    fn chief(&self) -> Matrix3<f64> {
        let we = self.f * (1.0 - self.mp);
        let q0 = self.s_img - we;
        let [gx, gy] = self.slope();
        let r = tilt_rotation(self.tau_s);
        let e1 = Vector3::new(r[(0, 0)], r[(0, 1)], -r[(0, 2)]);
        let e2 = Vector3::new(r[(1, 0)], r[(1, 1)], -r[(1, 2)]);
        let l = |e: &Vector3<f64>| [e.x + gx * e.z, e.y + gy * e.z];
        let (l1, l2) = (l(&e1), l(&e2));
        let s = q0 / self.pitch;
        let [cx, cy] = self.centre;
        Matrix3::new(
            s * l1[0] + cx * gx,
            s * l1[1] + cx * gy,
            cx * self.mp,
            s * l2[0] + cy * gx,
            s * l2[1] + cy * gy,
            cy * self.mp,
            gx,
            gy,
            self.mp,
        )
    }

    /// `K'` and `R` with `chief = K'·R`.
    fn pinhole(&self) -> (Matrix3<f64>, Matrix3<f64>) {
        let m = self.chief();
        let m = if m.determinant() < 0.0 { -m } else { m };
        let (k, r) = rq3(&m);
        (k / k[(2, 2)], r)
    }

    /// The tilt a calibration reports (the third row of the RQ rotation).
    fn tilt_d(&self) -> [f64; 2] {
        let (_, r) = self.pinhole();
        let r3 = r.row(2);
        [(-r3[1]).atan2(r3[2]), r3[0].clamp(-1.0, 1.0).asin()]
    }

    /// Per-axis σ (px) of the geometric spot of the lens-frame point `x`
    /// (`n` pupil samples).
    fn spot_sigma(&self, x: &Vector3<f64>, n: usize) -> f64 {
        let m = self.chief();
        let pi = self.pof();
        let r = self.pupil() / 2.0;
        let golden = std::f64::consts::PI * (3.0 - 5f64.sqrt());
        let (mut su, mut sv, mut suu, mut svv) = (0.0, 0.0, 0.0, 0.0);
        for k in 0..n {
            let rr = r * ((k as f64 + 0.5) / n as f64).sqrt();
            let (s, c) = (k as f64 * golden).sin_cos();
            let e = Vector3::new(rr * c, rr * s, 0.0);
            let d = e - x;
            let t = (1.0 - pi.dot(x)) / pi.dot(&d);
            let p = m * (x + t * d);
            let (u, v) = (p.x / p.z, p.y / p.z);
            su += u;
            sv += v;
            suu += u * u;
            svv += v * v;
        }
        let nn = n as f64;
        let var = (suu / nn - (su / nn).powi(2)) + (svv / nn - (sv / nn).powi(2));
        (var.max(0.0) / 2.0).sqrt()
    }

    /// The library camera of this lens (a single-axis tilt, so the chief-ray
    /// map is exactly `K·H(τ)`), and its distortion-free scale `f'`.
    fn camera(&self) -> (CameraParams, f64) {
        let (kp, _) = self.pinhole();
        let tau = self.tilt_d();
        let r = tilt_rotation(tau);
        let fx = kp[(0, 0)] / r[(2, 2)];
        let fy = kp[(1, 1)] / r[(2, 2)];
        let intrinsics = FxFyCxCy {
            fx,
            fy,
            cx: kp[(0, 2)] + fx * r[(0, 2)],
            cy: kp[(1, 2)] + fy * r[(1, 2)],
        };
        let camera = CameraParams::new(
            ProjectionParams::Pinhole,
            DistortionParams::None,
            SensorParams::from_values(SensorKind::Scheimpflug, &tau).unwrap(),
            intrinsics,
        );
        let fref = ((kp[(0, 0)].powi(2) + kp[(1, 1)].powi(2) + kp[(0, 1)].powi(2)) / 2.0).sqrt();
        (camera, fref)
    }
}

// ─── A synthetic capture ────────────────────────────────────────────────────

/// SplitMix64 with Box-Muller normals: a fixed sequence per seed.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    }

    fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next()
    }

    fn normal(&mut self) -> f64 {
        let u = self.next().max(1e-300);
        let v = self.next();
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
    }
}

/// The target: a 12 × 10 grid, 6 mm pitch, centred on its origin.
fn target() -> Vec<Point3<f64>> {
    let mut pts = Vec::new();
    for j in 0..10 {
        for i in 0..12 {
            pts.push(Point3::new(
                (i as f64 - 5.5) * 0.006,
                (j as f64 - 4.5) * 0.006,
                0.0,
            ));
        }
    }
    pts
}

/// `views` target poses at 0.17–0.33 m, tilted up to ±0.45 rad about both
/// target axes, the target centre near the optical axis.
fn poses(views: usize, seed: u64) -> Vec<Iso3> {
    let mut rng = Rng(seed);
    (0..views)
        .map(|v| {
            let z = 0.17 + 0.16 * (v as f64 + 0.5) / views as f64;
            let rot = UnitQuaternion::from_euler_angles(
                rng.uniform(-0.45, 0.45),
                rng.uniform(-0.45, 0.45),
                rng.uniform(-0.3, 0.3),
            );
            let t = Translation3::new(rng.uniform(-0.01, 0.01), rng.uniform(-0.01, 0.01), z);
            Iso3::from_parts(t, rot)
        })
        .collect()
}

/// One synthetic capture: noise-free `σ` per point (the caller adds noise
/// and offsets), the camera and the true values.
struct Scene {
    camera: CameraParams,
    poses: Vec<Iso3>,
    points: Vec<Point3<f64>>,
    /// `σ` (px, pixel integration included) per view and point.
    sigma: Vec<Vec<f64>>,
    pof: Vector3<f64>,
    kappa: f64,
    s0sq: f64,
}

/// The thick lens's blur of every point, with in-focus optical blur
/// `s0` (px) in quadrature.
fn lens_scene(lens: &Lens, s0: f64, views: usize, seed: u64) -> Scene {
    let (camera, fref) = lens.camera();
    let poses = poses(views, seed);
    let points = target();
    let sigma = poses
        .iter()
        .map(|pose| {
            points
                .iter()
                .map(|p| {
                    let x = (pose * p).coords;
                    let spot = lens.spot_sigma(&x, 256);
                    (s0 * s0 + spot * spot + PIXEL_VARIANCE).sqrt()
                })
                .collect()
        })
        .collect();
    Scene {
        camera,
        poses,
        points,
        sigma,
        pof: lens.pof(),
        kappa: lens.pupil() * fref / 4.0,
        s0sq: s0 * s0,
    }
}

/// The blur the model itself predicts (no offset), for a camera whose local
/// scale `scale(x)` is given in closed form.
// The model's parameters, the camera and the capture, each named at the
// call site.
#[allow(clippy::too_many_arguments)]
fn model_scene(
    camera: CameraParams,
    pof: Vector3<f64>,
    kappa: f64,
    s0sq: f64,
    normal: Vector3<f64>,
    scale: &dyn Fn(&Vector3<f64>) -> f64,
    views: usize,
    seed: u64,
) -> Scene {
    let poses = poses(views, seed);
    let points = target();
    let sigma = poses
        .iter()
        .map(|pose| {
            points
                .iter()
                .map(|p| {
                    let x = (pose * p).coords;
                    let d = (1.0 - pof.dot(&x)) / normal.dot(&x);
                    let g = kappa * scale(&x) * d;
                    (s0sq + PIXEL_VARIANCE + g * g).sqrt()
                })
                .collect()
        })
        .collect();
    Scene {
        camera,
        poses,
        points,
        sigma,
        pof,
        kappa,
        s0sq,
    }
}

/// An untilted pinhole camera, 720 × 540, `f = 1190` px.
fn plain_camera(distortion: DistortionParams) -> CameraParams {
    CameraParams::new(
        ProjectionParams::Pinhole,
        distortion,
        SensorParams::Identity,
        FxFyCxCy {
            fx: 1190.0,
            fy: 1190.0,
            cx: 364.0,
            cy: 271.0,
        },
    )
}

/// The dataset of `scene` with per-view offsets `offsets` and noise of
/// standard deviation `sd(σ)` drawn from `rng` (none when `rng` is `None`);
/// every point states the variance `sd(σ)²` of its `σ`.
fn dataset(
    scene: &Scene,
    offsets: &[f64],
    sd: &dyn Fn(f64) -> f64,
    mut rng: Option<&mut Rng>,
) -> PlanarDataset {
    let model = scene.camera.build().unwrap();
    let views = scene
        .poses
        .iter()
        .zip(&scene.sigma)
        .enumerate()
        .map(|(v, (pose, sigmas))| {
            let p2: Vec<Point2<f64>> = scene
                .points
                .iter()
                .map(|p| model.project_camera_point(&(pose * p).coords).unwrap())
                .collect();
            let blur = sigmas
                .iter()
                .map(|&s| {
                    let noise = rng.as_deref_mut().map_or(0.0, Rng::normal);
                    let obs =
                        (s + offsets.get(v).copied().unwrap_or(0.0) + sd(s) * noise).max(0.05);
                    Some(isotropic_blur(obs, sd(s)))
                })
                .collect();
            let obs = CorrespondenceView::new(scene.points.clone(), p2)
                .unwrap()
                .with_blur(blur)
                .unwrap();
            View::without_meta(obs)
        })
        .collect();
    PlanarDataset::new(views).unwrap()
}

/// An isotropic blur of width `sigma` whose `σ` has standard deviation
/// `sd`: `Σ = σ²·I`, and a covariance of `(xx, xy, yy)` with
/// `var((xx + yy)/2) = 4σ²·sd²`.
fn isotropic_blur(sigma: f64, sd: f64) -> PointBlur {
    let a = 4.0 * sigma * sigma * sd * sd;
    PointBlur::new(
        [sigma * sigma, 0.0, sigma * sigma],
        [[a, 0.0, a], [0.0, a, 0.0], [a, 0.0, a]],
    )
    .unwrap()
}

fn fitted(diagnostic: FocusDiagnostic) -> FocusFit {
    match diagnostic {
        FocusDiagnostic::Fitted(fit) => fit,
        other => panic!("not fitted: {other:?}"),
    }
}

fn studied_lens(mp: f64) -> Lens {
    Lens::new(0.008, mp, 0.25, [-0.09, 0.0], 7e-6, 2.0, [364.0, 271.0])
}

/// The blur scale `s` the fit itself computes for `camera`, to generate
/// data from the model (the thick lens checks the model; this checks the
/// fit and its σ).
fn own_scale(camera: &CameraParams) -> impl Fn(&Vector3<f64>) -> f64 {
    let model = camera.build().unwrap();
    let geometry = SensorGeometry::new(camera, &model).unwrap();
    move |x: &Vector3<f64>| geometry.point(&model, x).unwrap().scale
}

/// The fit of a fitted or a not-crossed diagnostic.
fn any_fit(diagnostic: FocusDiagnostic) -> FocusFit {
    match diagnostic {
        FocusDiagnostic::Fitted(fit) | FocusDiagnostic::NotCrossed(fit) => fit,
        other => panic!("not fitted: {other:?}"),
    }
}

/// The studied lens's camera with blur from the model itself, the plane of
/// sharp focus `pof_factor · π_lens` (1: in focus at 0.25 m on the axis,
/// crossed by the views at 0.17–0.33 m; 0.5: at 0.5 m, beyond them all).
fn tilted_model_scene(pof_factor: f64, views: usize) -> (Scene, Vector3<f64>) {
    let lens = studied_lens(1.0);
    let (camera, fref) = lens.camera();
    let normal = tilt_rotation(lens.tilt_d()).row(2).transpose();
    let kappa = lens.pupil() * fref / 4.0;
    let scale = own_scale(&camera);
    let scene = model_scene(
        camera.clone(),
        lens.pof() * pof_factor,
        kappa,
        0.5,
        normal,
        &scale,
        views,
        7,
    );
    (scene, normal)
}

// ─── Recovery ───────────────────────────────────────────────────────────────

/// The library camera of the lens reproduces its chief-ray map.
#[test]
fn lens_camera_is_the_chief_ray_map() {
    for (mp, tau) in [(1.0, [-0.09, 0.0]), (0.8, [-0.09, 0.0]), (1.0, [-0.2, 0.1])] {
        let lens = Lens::new(0.008, mp, 0.25, tau, 7e-6, 2.0, [364.0, 271.0]);
        let (camera, _) = lens.camera();
        let model = camera.build().unwrap();
        for x in [
            Vector3::new(0.02, -0.03, 0.2),
            Vector3::new(-0.04, 0.03, 0.3),
        ] {
            let p = lens.chief() * x;
            let q = model.project_camera_point(&x).unwrap();
            assert!((q.x - p.x / p.z).abs() < 1e-8 && (q.y - p.y / p.z).abs() < 1e-8);
        }
    }
}

/// Fit noise-free thick-lens blur with zero-mean per-view offsets and check
/// it against the lens: `|Δπ|/|π|`, `κ` relative, residual RMS (px).
fn check_thick_lens(lens: &Lens, rel_pi: f64, rel_kappa: f64, rms: f64) -> (Scene, FocusFit) {
    let scene = lens_scene(lens, 0.7, 20, 7);
    let offsets: Vec<f64> = (0..20).map(|v| 0.03 * ((v % 5) as f64 - 2.0)).collect();
    let data = dataset(&scene, &offsets, &|s| 0.02 + 0.03 * s, None);
    let fit =
        fitted(fit_focus(&data, &scene.camera, &scene.poses, &FocusOptions::default()).unwrap());
    let pi = Vector3::from(fit.plane_inv_m);
    let rel = (pi - scene.pof).norm() / scene.pof.norm();
    let k_rel = fit.blur_scale_px_m / scene.kappa - 1.0;
    let worst_offset = fit
        .views
        .iter()
        .zip(&offsets)
        .map(|(v, o)| (v.offset_px - o).abs())
        .fold(0.0, f64::max);
    let tilt = lens.tilt_d().map(f64::to_degrees);
    println!(
        "tilt {tilt:.1?}° mp {}: |Δπ|/|π| {rel:.2e}, κ {k_rel:+.2e}, σ0² {:+.1e} px², worst offset \
         {worst_offset:.1e} px, residual RMS {:.1e} px",
        lens.mp,
        fit.in_focus_variance_px2 - scene.s0sq,
        fit.residual_rms_px
    );
    assert!(rel < rel_pi, "{tilt:?} mp {}: |Δπ|/|π| {rel:.2e}", lens.mp);
    assert!(
        k_rel.abs() < rel_kappa,
        "{tilt:?} mp {}: κ {k_rel:+.2e}",
        lens.mp
    );
    assert!(
        fit.residual_rms_px < rms,
        "residual RMS {}",
        fit.residual_rms_px
    );
    assert!((fit.in_focus_variance_px2 - scene.s0sq).abs() < 1e-3);
    for (v, view) in fit.views.iter().enumerate() {
        assert!(view.used);
        assert!(
            (view.offset_px - offsets[v]).abs() < 5e-4,
            "view {v}: offset {} vs {}",
            view.offset_px,
            offsets[v]
        );
    }
    assert!(fit.views.iter().filter(|v| v.crosses_focus).count() >= 3);
    assert_eq!(fit.points, 20 * 120);
    assert_eq!(fit.unprojected_points, 0);
    assert!(fit.stated_weights);
    // The sensor normal is the tilted sensor's.
    let n = Vector3::from(fit.sensor_normal);
    let want = tilt_rotation(lens.tilt_d()).row(2).transpose();
    assert!((n - want).norm() < 1e-12);
    (scene, fit)
}

/// Noise-free blur from the exact thick lens, with zero-mean per-view
/// offsets: the plane, the blur scale, the in-focus blur and the offsets
/// come back. The entrance pupil lies in the lens plane, so the model is
/// exact to first order in the pupil's size: the plane within 2e-4, the
/// blur scale within 0.1 %, the residual RMS under a millipixel.
#[test]
fn recovers_the_plane_from_the_thick_lens() {
    for mp in [1.0, 0.8] {
        check_thick_lens(&studied_lens(mp), 2e-4, 1e-3, 1e-3);
    }
}

/// At an 11.5° × 5.7° sensor tilt the pupil's obliquity changes the blur by
/// ±5 % across the image: the plane still comes back, and its bias is a
/// small fraction of its σ at a realistic noise.
#[test]
fn recovers_the_plane_at_a_large_tilt() {
    let lens = Lens::new(0.008, 1.0, 0.25, [-0.2, 0.1], 7e-6, 2.0, [364.0, 271.0]);
    let (scene, fit) = check_thick_lens(&lens, 2e-4, 3e-3, 1e-3);
    let mut rng = Rng(5);
    let noisy = dataset(&scene, &[], &|s| 0.02 + 0.03 * s, Some(&mut rng));
    let stated = fitted(
        fit_focus(
            &noisy,
            &scene.camera,
            &scene.poses,
            &FocusOptions::default(),
        )
        .unwrap(),
    );
    for i in 0..3 {
        let bias = (fit.plane_inv_m[i] - scene.pof[i]) / stated.covariance[i][i].sqrt();
        assert!(bias.abs() < 0.1, "π[{i}] bias {bias:.2} σ");
    }
}

/// Blur generated by the model itself on a distorted camera, its local
/// scale in closed form (radial `k1`: tangential `1 + k1·r²`, radial
/// `1 + 3·k1·r²`): recovered to rounding, so the scale enters as derived.
#[test]
fn distortion_scales_the_blur_locally() {
    let k1 = -0.25;
    let camera = plain_camera(
        DistortionParams::from_values(DistortionKind::BrownConrady5, &[k1, 0.0, 0.0, 0.0, 0.0])
            .unwrap(),
    );
    let scale = move |x: &Vector3<f64>| {
        let r2 = (x.x * x.x + x.y * x.y) / (x.z * x.z);
        (((1.0 + k1 * r2).powi(2) + (1.0 + 3.0 * k1 * r2).powi(2)) / 2.0).sqrt()
    };
    let pof = Vector3::new(1.0, -6.0, 4.0);
    let scene = model_scene(camera, pof, 1.2, 0.4, Vector3::z(), &scale, 12, 3);
    let data = dataset(&scene, &[], &|s| 0.02 + 0.03 * s, None);
    let fit =
        fitted(fit_focus(&data, &scene.camera, &scene.poses, &FocusOptions::default()).unwrap());
    let pi = Vector3::from(fit.plane_inv_m);
    assert!((pi - pof).norm() < 1e-6, "{pi:?}");
    assert!((fit.blur_scale_px_m - 1.2).abs() < 1e-7);
    assert!((fit.in_focus_variance_px2 - 0.4).abs() < 1e-6);
    assert!(fit.residual_rms_px < 1e-7);
}

/// A blur level common to every image is not a parameter. With views that
/// cross the plane it moves the blur scale and the in-focus blur and
/// hardly the plane; without a crossing it moves the plane's position
/// along the sensor normal, and not its direction across it.
#[test]
fn a_common_blur_level_moves_what_the_docs_say() {
    let sd = |s: f64| 0.02 + 0.03 * s;
    let common = vec![0.02; 20];
    let opts = FocusOptions::default();

    let (scene, normal) = tilted_model_scene(1.0, 20);
    let base = fitted(
        fit_focus(
            &dataset(&scene, &[], &sd, None),
            &scene.camera,
            &scene.poses,
            &opts,
        )
        .unwrap(),
    );
    let moved = fitted(
        fit_focus(
            &dataset(&scene, &common, &sd, None),
            &scene.camera,
            &scene.poses,
            &opts,
        )
        .unwrap(),
    );
    // Shifts in σ of a capture with noise `sd`.
    let mut rng = Rng(5);
    let noisy = any_fit(
        fit_focus(
            &dataset(&scene, &[], &sd, Some(&mut rng)),
            &scene.camera,
            &scene.poses,
            &opts,
        )
        .unwrap(),
    );
    let value = |f: &FocusFit, k: usize| match k {
        0..=2 => f.plane_inv_m[k],
        3 => f.blur_scale_px_m,
        _ => f.in_focus_variance_px2,
    };
    let crossed: Vec<f64> = (0..5)
        .map(|k| (value(&moved, k) - value(&base, k)) / noisy.covariance[k][k].sqrt())
        .collect();
    println!("crossed, +0.02 px everywhere: shift in σ {crossed:.2?}");
    assert!(crossed[..3].iter().all(|s| s.abs() < 0.5));
    assert!(crossed[3] > 1.5 && crossed[4] > 5.0);

    // Focused beyond every view: nothing crosses.
    let (scene, _) = tilted_model_scene(0.5, 20);
    let diagnose = |offsets: &[f64]| match fit_focus(
        &dataset(&scene, offsets, &sd, None),
        &scene.camera,
        &scene.poses,
        &opts,
    )
    .unwrap()
    {
        FocusDiagnostic::NotCrossed(fit) => fit,
        other => panic!("{other:?}"),
    };
    let (base, moved) = (diagnose(&[]), diagnose(&[0.05; 20]));
    let mut rng = Rng(5);
    let noisy = any_fit(
        fit_focus(
            &dataset(&scene, &[], &sd, Some(&mut rng)),
            &scene.camera,
            &scene.poses,
            &opts,
        )
        .unwrap(),
    );
    let across = Matrix3::identity() - normal * normal.transpose();
    let d_across = across * (Vector3::from(moved.plane_inv_m) - Vector3::from(base.plane_inv_m));
    let d_along = normal.dot(&(Vector3::from(moved.plane_inv_m) - Vector3::from(base.plane_inv_m)));
    let c = Matrix3::from_fn(|i, j| noisy.covariance[i][j]);
    let along_sigma = (normal.transpose() * c * normal)[(0, 0)].sqrt();
    let across_sigma = (across * c * across.transpose()).map_diagonal(f64::sqrt);
    let across_z = d_across.component_div(&across_sigma);
    let theta_sigma = noisy.image_line.unwrap().covariance[0][0].sqrt();
    let (l0, l1) = (base.image_line.unwrap(), moved.image_line.unwrap());
    let theta_z = (l1.angle_rad - l0.angle_rad) / theta_sigma;
    println!(
        "not crossed, +0.05 px: shift in σ across {across_z:.3?}, along {:.2}, θ {theta_z:.3}",
        d_along / along_sigma,
    );
    assert!(d_along.abs() > 0.5 * along_sigma);
    assert!(across_z.amax() < 0.1 && theta_z.abs() < 0.1);
}

// ─── Derived quantities ─────────────────────────────────────────────────────

/// The image line is where the plane's in-focus distance diverges, with the
/// in-focus side where the docs say; the depth-of-field planes are where
/// the predicted blur at the distortion-free scale (`s = 1`) reaches the
/// limit.
#[test]
fn derived_quantities_mean_what_they_say() {
    let lens = studied_lens(1.0);
    let scene = lens_scene(&lens, 0.7, 20, 7);
    let data = dataset(&scene, &[], &|s| 0.02 + 0.03 * s, None);
    let limit = 1.5;
    let fit = fitted(
        fit_focus(
            &data,
            &scene.camera,
            &scene.poses,
            &FocusOptions::new(Some(limit), false),
        )
        .unwrap(),
    );
    let pi = Vector3::from(fit.plane_inv_m);

    // A pixel on the line sees the plane at infinity.
    let line = fit.image_line.expect("a tilted plane has an image line");
    let k = &scene.camera.intrinsics;
    let model = scene.camera.build().unwrap();
    let kmat = Matrix3::new(k.fx, 0.0, k.cx, 0.0, k.fy, k.cy, 0.0, 0.0, 1.0);
    let h = match &model.sensor {
        vision_calibration_core::AnySensor::Homography(s) => s.h,
        _ => unreachable!(),
    };
    let m_inv = (kmat * h).try_inverse().unwrap();
    let (c, s) = (line.angle_rad.cos(), line.angle_rad.sin());
    let on = Vector3::new(
        line.offset_px * c - 100.0 * s,
        line.offset_px * s + 100.0 * c,
        1.0,
    );
    let dir = m_inv * on;
    assert!(pi.dot(&dir).abs() < 1e-9 * pi.norm() * dir.norm());
    // Inside (towards smaller u·cos θ + v·sin θ) the plane is in front.
    let inside = on - Vector3::new(50.0 * c, 50.0 * s, 0.0);
    let dir = m_inv * inside;
    assert!(pi.dot(&(dir / dir.z)) > 0.0);

    // On the near and far planes the predicted blur is the limit.
    let dof = fit
        .depth_of_field
        .expect("the limit is above the in-focus blur");
    let n = Vector3::from(fit.sensor_normal);
    for plane in [dof.near_plane_inv_m, dof.far_plane_inv_m] {
        let p = Vector3::from(plane);
        // A point on the plane along a ray r: X = r / (p·r).
        let r = Vector3::new(0.05, -0.02, 1.0);
        let x = r / p.dot(&r);
        let d = (1.0 - pi.dot(&x)) / n.dot(&x);
        let sigma =
            (fit.in_focus_variance_px2 + PIXEL_VARIANCE + (fit.blur_scale_px_m * d).powi(2)).sqrt();
        assert!((sigma - limit).abs() < 1e-9, "{sigma}");
    }
    // No depth of field below the in-focus blur.
    let fit = fitted(
        fit_focus(
            &data,
            &scene.camera,
            &scene.poses,
            &FocusOptions::new(Some(0.3), false),
        )
        .unwrap(),
    );
    assert!(fit.depth_of_field.is_none());
}

// ─── Typed reasons and errors ───────────────────────────────────────────────

#[test]
fn reasons_it_is_unavailable() {
    let lens = studied_lens(1.0);
    let scene = lens_scene(&lens, 0.7, 6, 7);
    let sd = |s: f64| 0.02 + 0.03 * s;

    // No blur at all.
    let mut bare = dataset(&scene, &[], &sd, None);
    for view in &mut bare.views {
        view.obs = view.obs.clone().with_blur(Vec::new()).unwrap();
    }
    let got = fit_focus(&bare, &scene.camera, &scene.poses, &FocusOptions::default()).unwrap();
    assert!(matches!(got, FocusDiagnostic::NoBlur), "{got:?}");

    // Blur on two points per view (too few for an offset) and four points
    // in one view.
    let mut sparse = dataset(&scene, &[], &sd, None);
    for (v, view) in sparse.views.iter_mut().enumerate() {
        let keep = if v == 0 { 4 } else { 2 };
        let blur = (0..view.obs.len())
            .map(|i| if i < keep { view.obs.blur(i) } else { None })
            .collect();
        view.obs = view.obs.clone().with_blur(blur).unwrap();
    }
    let got = fit_focus(
        &sparse,
        &scene.camera,
        &scene.poses,
        &FocusOptions::default(),
    )
    .unwrap();
    assert!(
        matches!(got, FocusDiagnostic::TooFewPoints { need: 6, got: 4 }),
        "{got:?}"
    );

    // Every view beyond the plane of sharp focus: nothing crosses it, and
    // the plane's direction across the sensor normal comes back.
    let pof = Vector3::new(0.0, -3.0, 8.0); // in focus at 0.125 m on the axis
    let far = model_scene(
        plain_camera(DistortionParams::None),
        pof,
        1.2,
        0.4,
        Vector3::z(),
        &|_| 1.0,
        12,
        3,
    );
    let data = dataset(&far, &[], &sd, None);
    let got = fit_focus(&data, &far.camera, &far.poses, &FocusOptions::default()).unwrap();
    let FocusDiagnostic::NotCrossed(fit) = got else {
        panic!("{got:?}")
    };
    assert!(fit.views.iter().all(|v| !v.crosses_focus));
    assert!((fit.plane_inv_m[0] - pof.x).abs() < 1e-6);
    assert!((fit.plane_inv_m[1] - pof.y).abs() < 1e-6);
    // The in-focus distance `1/(8 − 3·y/z)` grows down the image (+v):
    // θ = 90°.
    let line = fit.image_line.unwrap();
    assert!((line.angle_rad - std::f64::consts::FRAC_PI_2).abs() < 1e-6);

    // Every view at one depth facing the camera: κ and the plane's tilt
    // trade.
    let mut frontal = model_scene(
        plain_camera(DistortionParams::None),
        Vector3::new(0.5, -6.0, 5.0),
        1.2,
        0.4,
        Vector3::z(),
        &|_| 1.0,
        8,
        3,
    );
    frontal.poses = (0..8)
        .map(|v| {
            Iso3::from_parts(
                Translation3::new(0.004 * (v as f64 - 3.5), 0.0, 0.25),
                UnitQuaternion::identity(),
            )
        })
        .collect();
    let normal = Vector3::z();
    frontal.sigma = frontal
        .poses
        .iter()
        .map(|pose| {
            frontal
                .points
                .iter()
                .map(|p| {
                    let x = (pose * p).coords;
                    let d = (1.0 - frontal.pof.dot(&x)) / normal.dot(&x);
                    (0.4 + PIXEL_VARIANCE + (1.2 * d).powi(2)).sqrt()
                })
                .collect()
        })
        .collect();
    let data = dataset(&frontal, &[], &sd, None);
    let got = fit_focus(
        &data,
        &frontal.camera,
        &frontal.poses,
        &FocusOptions::default(),
    )
    .unwrap();
    assert!(matches!(got, FocusDiagnostic::Undetermined), "{got:?}");
}

/// The blur of point `i` of view `v` with covariance `cov`.
fn set_blur_cov(data: &mut PlanarDataset, v: usize, i: usize, cov: [[f64; 3]; 3]) {
    let obs = &data.views[v].obs;
    let mut blur: Vec<Option<PointBlur>> = (0..obs.len()).map(|k| obs.blur(k)).collect();
    blur[i] = Some(PointBlur::new(blur[i].unwrap().psf_px(), cov).unwrap());
    data.views[v].obs = obs.clone().with_blur(blur).unwrap();
}

/// Every point weighted by its stated variance, or every point equally
/// when none states one; a mix is an error, not a switch to equal weights.
#[test]
fn weights_are_all_stated_or_none() {
    let lens = studied_lens(1.0);
    let scene = lens_scene(&lens, 0.7, 6, 7);
    let opts = FocusOptions::default();
    let mut data = dataset(&scene, &[], &|s| 0.02 + 0.03 * s, None);
    set_blur_cov(&mut data, 3, 17, [[0.0; 3]; 3]);
    let err = fit_focus(&data, &scene.camera, &scene.poses, &opts).unwrap_err();
    assert!(
        matches!(
            err,
            Error::MixedBlurCovariance {
                with: 719,
                without: 1
            }
        ),
        "{err}"
    );

    let none = dataset(&scene, &[], &|_| 0.0, None);
    let fit = fitted(fit_focus(&none, &scene.camera, &scene.poses, &opts).unwrap());
    assert!(!fit.stated_weights);
}

#[test]
fn rejects_invalid_input() {
    let lens = studied_lens(1.0);
    let scene = lens_scene(&lens, 0.7, 4, 7);
    let data = dataset(&scene, &[], &|s| 0.02 + 0.03 * s, None);
    let opts = FocusOptions::default();
    let invalid = |r: Result<FocusDiagnostic, Error>| matches!(r, Err(Error::InvalidInput { .. }));
    assert!(invalid(fit_focus(
        &data,
        &scene.camera,
        &scene.poses[..3],
        &opts
    )));
    for limit in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        let bad = FocusOptions::new(Some(limit), false);
        assert!(invalid(fit_focus(&data, &scene.camera, &scene.poses, &bad)));
    }
    let fisheye = scene
        .camera
        .clone()
        .with_projection(ProjectionParams::Equidistant);
    assert!(invalid(fit_focus(&data, &fisheye, &scene.poses, &opts)));
    // A camera or a pose that is not finite.
    let mut nan_camera = scene.camera.clone();
    nan_camera.intrinsics.cx = f64::NAN;
    assert!(invalid(fit_focus(&data, &nan_camera, &scene.poses, &opts)));
    let nan_tilt = scene
        .camera
        .clone()
        .with_sensor(SensorParams::from_values(SensorKind::Scheimpflug, &[f64::NAN, 0.0]).unwrap());
    assert!(invalid(fit_focus(&data, &nan_tilt, &scene.poses, &opts)));
    let mut nan_poses = scene.poses.clone();
    nan_poses[2].translation.vector.x = f64::NAN;
    assert!(invalid(fit_focus(&data, &scene.camera, &nan_poses, &opts)));
}

/// A blur point that does not project at its pose (here behind the camera)
/// is left out and counted, not an error.
#[test]
fn skips_points_that_do_not_project() {
    let lens = studied_lens(1.0);
    let scene = lens_scene(&lens, 0.7, 6, 7);
    let mut data = dataset(&scene, &[], &|s| 0.02 + 0.03 * s, None);
    let view = &data.views[2].obs;
    let mut p3 = view.points_3d.clone();
    // 1 m behind the target: about 0.75 m behind the camera.
    p3[5] = scene.poses[2].inverse() * Point3::new(0.0, 0.0, -0.75);
    let blur = (0..view.len()).map(|k| view.blur(k)).collect();
    data.views[2].obs = CorrespondenceView::new(p3, view.points_2d.clone())
        .unwrap()
        .with_blur(blur)
        .unwrap();
    let fit =
        fitted(fit_focus(&data, &scene.camera, &scene.poses, &FocusOptions::default()).unwrap());
    assert_eq!(fit.unprojected_points, 1);
    assert_eq!(fit.points, 6 * 120 - 1);
    assert_eq!(fit.views[2].points, 119);
}

// ─── Wire form and determinism ──────────────────────────────────────────────

#[test]
fn json_round_trip_and_determinism() {
    let lens = studied_lens(1.0);
    let scene = lens_scene(&lens, 0.7, 10, 7);
    let mut rng = Rng(11);
    let data = dataset(&scene, &[], &|s| 0.02 + 0.03 * s, Some(&mut rng));
    let opts = FocusOptions::new(Some(1.5), true);
    let a = fit_focus(&data, &scene.camera, &scene.poses, &opts).unwrap();
    let b = fit_focus(&data, &scene.camera, &scene.poses, &opts).unwrap();
    let ja = serde_json::to_string(&a).unwrap();
    assert_eq!(ja, serde_json::to_string(&b).unwrap());
    let back: FocusDiagnostic = serde_json::from_str(&ja).unwrap();
    assert_eq!(serde_json::to_string(&back).unwrap(), ja);
    assert!(ja.starts_with(r#"{"status":"fitted""#), "{}", &ja[..40]);
    let fit = fitted(a);
    let Some(FocusJackknife::Estimated(jackknife)) = &fit.view_jackknife else {
        panic!("{:?}", fit.view_jackknife)
    };
    assert_eq!(jackknife.views, 10);
    assert!(jackknife.image_line.is_some() && jackknife.depth_of_field.is_some());

    for reason in [
        FocusDiagnostic::NoBlur,
        FocusDiagnostic::TooFewPoints { need: 7, got: 4 },
        FocusDiagnostic::Undetermined,
    ] {
        let j = serde_json::to_string(&reason).unwrap();
        let back: FocusDiagnostic = serde_json::from_str(&j).unwrap();
        assert_eq!(serde_json::to_string(&back).unwrap(), j);
    }
    for reason in [
        FocusJackknife::TooFewViews { views: 2 },
        FocusJackknife::Undetermined { view: 4 },
    ] {
        let j = serde_json::to_string(&reason).unwrap();
        let back: FocusJackknife = serde_json::from_str(&j).unwrap();
        assert_eq!(serde_json::to_string(&back).unwrap(), j);
    }
    let j = serde_json::to_string(&opts).unwrap();
    assert_eq!(j, r#"{"blur_limit_px":1.5,"view_jackknife":true}"#);
    assert_eq!(serde_json::from_str::<FocusOptions>(&j).unwrap(), opts);
    assert_eq!(
        serde_json::from_str::<FocusOptions>("{}").unwrap(),
        FocusOptions::default()
    );
}

/// The jackknife needs three fitted views, and is computed only on request.
#[test]
fn jackknife_needs_three_views() {
    // A two-view fit (its parameters: the model's five and one offset).
    let lens = studied_lens(1.0);
    let (camera, _) = lens.camera();
    let model = camera.build().unwrap();
    let geometry = SensorGeometry::new(&camera, &model).unwrap();
    let solution = Solution {
        theta: vec![1.0, 0.1, -0.2, -4.0, 0.5, 0.01],
        cov: DMatrix::zeros(6, 6),
        residual_dof: 1,
        noise_scale: 1.0,
    };
    let jackknife = view_jackknife(
        &geometry,
        &[],
        &[],
        &solution,
        &[Some(0), None, Some(1)],
        &FocusOptions::new(None, true),
    );
    assert!(matches!(
        jackknife,
        FocusJackknife::TooFewViews { views: 2 }
    ));

    let scene = lens_scene(&lens, 0.7, 6, 7);
    let data = dataset(&scene, &[], &|s| 0.02 + 0.03 * s, None);
    let fit =
        fitted(fit_focus(&data, &scene.camera, &scene.poses, &FocusOptions::default()).unwrap());
    assert!(fit.view_jackknife.is_none());
}

// ─── Monte-Carlo ────────────────────────────────────────────────────────────

/// Scatter over draws against the RMS of a stated σ.
fn ratio(values: &[f64], sigmas: &[f64]) -> f64 {
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let scatter = (values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
    scatter / (sigmas.iter().map(|s| s * s).sum::<f64>() / n).sqrt()
}

/// The stated σ of every reported quantity against the scatter over noise
/// draws (blur from the model on a tilted camera, noise `0.02 + 0.03·σ` px
/// with its stated variance, fresh per-view offsets of 0.03 px sd each
/// draw, centred like the fit's). With the offsets fitted, the stated
/// covariance holds; no offset is injected in the null half, where the
/// offsets must scatter as their σ says and the noise scale stay at 1.
#[test]
#[ignore = "Monte-Carlo, run in release: cargo test --release -p vision-calibration-pipeline focus -- --ignored"]
fn monte_carlo_sigma_agrees() {
    let (scene, _) = tilted_model_scene(1.0, 20);
    let sd = |s: f64| 0.02 + 0.03 * s;
    let draws = 1000;
    let opts = FocusOptions::new(Some(1.5), false);

    for inject in [true, false] {
        let mut rng = Rng(if inject { 101 } else { 202 });
        // Quantities: π (3), κ, σ0², the offsets of the first and the last
        // view, line θ, ρ, ε, near-plane z.
        let mut values: Vec<Vec<f64>> = vec![Vec::new(); 11];
        let mut sigmas: Vec<Vec<f64>> = vec![Vec::new(); 11];
        let mut z_offsets = Vec::new();
        let mut noise_scales = Vec::new();
        for _ in 0..draws {
            let mut offsets: Vec<f64> = (0..20)
                .map(|_| if inject { 0.03 * rng.normal() } else { 0.0 })
                .collect();
            // The fit's offsets are relative to their mean.
            let mean_offset = offsets.iter().sum::<f64>() / offsets.len() as f64;
            offsets.iter_mut().for_each(|o| *o -= mean_offset);
            let data = dataset(&scene, &offsets, &sd, Some(&mut rng));
            let fit = fitted(fit_focus(&data, &scene.camera, &scene.poses, &opts).unwrap());
            let line = fit.image_line.unwrap();
            let dof = fit.depth_of_field.unwrap();
            let c = fit.covariance;
            let v = fit.views[0];
            // The last view's offset is minus the sum of the others.
            let last = fit.views[19];
            let row = [
                (fit.plane_inv_m[0], c[0][0]),
                (fit.plane_inv_m[1], c[1][1]),
                (fit.plane_inv_m[2], c[2][2]),
                (fit.blur_scale_px_m, c[3][3]),
                (fit.in_focus_variance_px2, c[4][4]),
                (v.offset_px - offsets[0], v.offset_sigma_px.powi(2)),
                (last.offset_px - offsets[19], last.offset_sigma_px.powi(2)),
                (line.angle_rad, line.covariance[0][0]),
                (line.offset_px, line.covariance[1][1]),
                (
                    dof.defocus_limit_inv_m,
                    dof.defocus_limit_sigma_inv_m.powi(2),
                ),
                (dof.near_plane_inv_m[2], dof.near_plane_covariance[2][2]),
            ];
            for (k, (value, var)) in row.into_iter().enumerate() {
                values[k].push(value);
                sigmas[k].push(var.sqrt());
            }
            if !inject {
                z_offsets.extend(fit.views.iter().map(|v| v.offset_px / v.offset_sigma_px));
            }
            noise_scales.push(fit.noise_scale);
        }
        let names = [
            "π_x", "π_y", "π_z", "κ", "σ0²", "δ_0", "δ_19", "θ", "ρ", "ε", "near_z",
        ];
        for k in 0..names.len() {
            let ratio = ratio(&values[k], &sigmas[k]);
            println!("inject {inject}: {} scatter/stated {ratio:.3}", names[k]);
            assert!((0.9..1.1).contains(&ratio), "{}: ratio {ratio}", names[k]);
        }
        let mean_scale = noise_scales.iter().sum::<f64>() / noise_scales.len() as f64;
        println!("inject {inject}: mean noise scale {mean_scale:.4}");
        assert!((mean_scale - 1.0).abs() < 0.03, "noise scale {mean_scale}");
        if !inject {
            let rms =
                (z_offsets.iter().map(|z| z * z).sum::<f64>() / z_offsets.len() as f64).sqrt();
            println!("null: offset z-score RMS {rms:.3}");
            assert!((rms - 1.0).abs() < 0.05, "offset z RMS {rms}");
        }
    }
}

/// Blur errors correlated within an image: each view gets a random blur
/// gradient across the image (0.06 px sd at the edges) and a random offset
/// (0.03 px sd) whose mean over the views is not removed. The stated σ is
/// short (the model treats the errors as independent beyond the offsets,
/// and the common level as zero); the view jackknife covers the scatter.
#[test]
#[ignore = "Monte-Carlo, run in release: cargo test --release -p vision-calibration-pipeline focus -- --ignored"]
fn monte_carlo_view_jackknife_covers_correlated_blur() {
    let views = 20;
    let (scene, _) = tilted_model_scene(1.0, views);
    let model = scene.camera.build().unwrap();
    let pixels: Vec<Vec<Point2<f64>>> = scene
        .poses
        .iter()
        .map(|pose| {
            scene
                .points
                .iter()
                .map(|p| model.project_camera_point(&(pose * p).coords).unwrap())
                .collect()
        })
        .collect();
    let sd = |s: f64| 0.02 + 0.03 * s;
    let draws = 300;
    let opts = FocusOptions::new(Some(1.5), true);
    let mut rng = Rng(303);
    let names = ["π_x", "π_y", "π_z", "κ", "σ0²", "θ", "ρ", "ε"];
    let mut values: Vec<Vec<f64>> = vec![Vec::new(); 8];
    let mut stated: Vec<Vec<f64>> = vec![Vec::new(); 8];
    let mut jackknifed: Vec<Vec<f64>> = vec![Vec::new(); 8];
    for _ in 0..draws {
        let offsets: Vec<f64> = (0..views).map(|_| 0.03 * rng.normal()).collect();
        let mut data = dataset(&scene, &offsets, &sd, Some(&mut rng));
        for (view, px) in data.views.iter_mut().zip(&pixels) {
            let (a, b) = (0.06 * rng.normal(), 0.06 * rng.normal());
            let blur = (0..view.obs.len())
                .map(|i| {
                    let blur = view.obs.blur(i).unwrap();
                    let s = blur.sigma_px()
                        + a * (px[i].x - 364.0) / 364.0
                        + b * (px[i].y - 271.0) / 271.0;
                    Some(PointBlur::new([s * s, 0.0, s * s], blur.cov()).unwrap())
                })
                .collect();
            view.obs = view.obs.clone().with_blur(blur).unwrap();
        }
        let fit = fitted(fit_focus(&data, &scene.camera, &scene.poses, &opts).unwrap());
        let Some(FocusJackknife::Estimated(j)) = fit.view_jackknife.clone() else {
            panic!("{:?}", fit.view_jackknife)
        };
        let (c, jc) = (fit.covariance, j.covariance);
        let (line, j_line) = (fit.image_line.unwrap(), j.image_line.unwrap());
        let (dof, j_dof) = (fit.depth_of_field.unwrap(), j.depth_of_field.unwrap());
        let row = [
            (fit.plane_inv_m[0], c[0][0], jc[0][0]),
            (fit.plane_inv_m[1], c[1][1], jc[1][1]),
            (fit.plane_inv_m[2], c[2][2], jc[2][2]),
            (fit.blur_scale_px_m, c[3][3], jc[3][3]),
            (fit.in_focus_variance_px2, c[4][4], jc[4][4]),
            (
                line.angle_rad,
                line.covariance[0][0],
                j_line.covariance[0][0],
            ),
            (
                line.offset_px,
                line.covariance[1][1],
                j_line.covariance[1][1],
            ),
            (
                dof.defocus_limit_inv_m,
                dof.defocus_limit_sigma_inv_m.powi(2),
                j_dof.defocus_limit_sigma_inv_m.powi(2),
            ),
        ];
        for (k, (value, var, j_var)) in row.into_iter().enumerate() {
            values[k].push(value);
            stated[k].push(var.sqrt());
            jackknifed[k].push(j_var.sqrt());
        }
    }
    for k in 0..names.len() {
        let (r_stated, r_jack) = (
            ratio(&values[k], &stated[k]),
            ratio(&values[k], &jackknifed[k]),
        );
        println!(
            "{}: scatter/stated {r_stated:.3}, scatter/jackknife {r_jack:.3}",
            names[k]
        );
        assert!(
            (0.8..1.2).contains(&r_jack),
            "{}: jackknife {r_jack}",
            names[k]
        );
        assert!(r_stated > 1.5, "{}: stated {r_stated}", names[k]);
    }
}
