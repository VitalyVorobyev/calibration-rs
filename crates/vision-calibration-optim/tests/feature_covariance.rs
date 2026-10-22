//! Per-point pixel covariance: the whitened solve against Monte-Carlo
//! scatter, and against a solve that weighs every point alike.
//!
//! Every point states a covariance whose scale (σ 0.1–1 px along its major
//! axis), anisotropy (up to 4:1) and orientation vary from point to point,
//! and its noise is drawn from that covariance. Per scene the study reports:
//! (a) the mean and spread of `σ̂₀`, which is about 1 when the stated
//! covariances hold; (b) the predicted σ of each parameter over its
//! Monte-Carlo σ; (c) the RMS of the normalized errors `err/σ` and the share
//! within 2σ; (d) the RMSE of the whitened solve over that of the uniformly
//! weighted one on the same data.
//!
//! The full studies are ignored in the default (debug) run:
//! `cargo test --release -p vision-calibration-optim --test feature_covariance -- --ignored --nocapture`.
//! `STUDY_RUNS` overrides the run count (default 500: at 200 the Monte-Carlo σ alone scatters by 5 %, and the ±10 % gate on 30 parameters sees 3σ draws).

#![allow(missing_docs)]

use nalgebra::{DVector, Matrix2, Translation3, UnitQuaternion, Vector2};
use vision_calibration_core::{
    BrownConrady5, CameraParams, CorrespondenceView, DistortionParams, FxFyCxCy, Iso3, NoMeta,
    NoiseLevel, ParameterCovariance, PinholeCamera, PlanarDataset, PointCovariance, Pt2, Pt3,
    ResidualKind, RigDataset, RigView, RigViewObs, Uncertainty, View, make_pinhole_camera,
    pinhole_camera_params,
};
use vision_calibration_optim::{
    BackendSolveOptions, PlanarIntrinsicsParams, PlanarIntrinsicsSolveOptions, RigExtrinsicsParams,
    RigExtrinsicsSolveOptions, optimize_planar_intrinsics, optimize_rig_extrinsics,
};

/// SplitMix64 of a key: a deterministic uniform stream with no shared state.
fn mix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// A uniform in `(0, 1)` keyed by `parts`.
fn uniform(parts: &[u64]) -> f64 {
    let key = parts.iter().fold(0x5EED_u64, |h, &p| mix(h ^ mix(p)));
    ((mix(key) >> 11) as f64 + 0.5) / (1u64 << 53) as f64
}

/// The stated covariance of point `point` of camera `camera` in view
/// `view`, fixed across noise realizations: major-axis σ uniform in
/// 0.1–1 px, minor axis up to four times smaller, any orientation. Also
/// returns its square root `R diag(σ₁, σ₂)`, which colours unit noise.
fn point_covariance(camera: usize, view: usize, point: usize) -> (PointCovariance, Matrix2<f64>) {
    let key = |k: u64| uniform(&[1, camera as u64, view as u64, point as u64, k]);
    let major = 0.1 + 0.9 * key(0);
    let minor = major / (1.0 + 3.0 * key(1));
    let theta = std::f64::consts::PI * key(2);
    let (s, c) = theta.sin_cos();
    let root = Matrix2::new(c, -s, s, c) * Matrix2::new(major, 0.0, 0.0, minor);
    let cov = root * root.transpose();
    (
        PointCovariance::new(cov[(0, 0)], cov[(0, 1)], cov[(1, 1)]).unwrap(),
        root,
    )
}

/// A standard normal pair keyed by the realization and the point.
fn gauss2(seed: u64, camera: usize, view: usize, point: usize) -> Vector2<f64> {
    let key = |k: u64| uniform(&[2, seed, camera as u64, view as u64, point as u64, k]);
    let (u, v) = (key(0), key(1));
    let r = (-2.0 * u.ln()).sqrt();
    let a = std::f64::consts::TAU * v;
    Vector2::new(r * a.cos(), r * a.sin())
}

/// A 9×9 board of 20 mm cells, centred on the target origin.
fn board() -> Vec<Pt3> {
    (0..81)
        .map(|i| {
            Pt3::new(
                (i % 9) as f64 * 0.02 - 0.08,
                (i / 9) as f64 * 0.02 - 0.08,
                0.0,
            )
        })
        .collect()
}

/// Ten board poses in front of a camera, varied in tilt.
fn board_poses() -> Vec<Iso3> {
    [
        (0.0, 0.0, 0.0, -0.01, -0.01, 0.45),
        (0.20, 0.0, 0.0, -0.02, 0.0, 0.50),
        (-0.20, 0.0, 0.0, 0.01, -0.02, 0.48),
        (0.0, 0.20, 0.0, -0.01, 0.01, 0.52),
        (0.0, -0.20, 0.0, 0.02, -0.01, 0.47),
        (0.15, 0.15, 0.0, -0.02, -0.02, 0.55),
        (-0.15, 0.15, 0.05, 0.01, 0.0, 0.49),
        (0.15, -0.15, -0.05, -0.01, 0.02, 0.53),
        (-0.15, -0.15, 0.0, 0.0, -0.01, 0.46),
        (0.10, -0.10, 0.10, -0.02, 0.01, 0.51),
    ]
    .map(|(rx, ry, rz, tx, ty, tz)| {
        Iso3::from_parts(
            Translation3::new(tx, ty, tz),
            UnitQuaternion::from_euler_angles(rx, ry, rz),
        )
    })
    .to_vec()
}

fn pinhole_truth(fx: f64, cx: f64) -> PinholeCamera {
    make_pinhole_camera(
        FxFyCxCy {
            fx,
            fy: fx + 5.0,
            cx,
            cy: 258.0,
        },
        BrownConrady5 {
            k1: -0.08,
            k2: 0.04,
            k3: 0.0,
            p1: 0.0005,
            p2: -0.0003,
            iters: 8,
        },
    )
}

/// The board seen by `camera` at `camera_se3_target` in realization `seed`:
/// with its stated covariances and with none, on the same noisy pixels.
fn observe(
    camera: &PinholeCamera,
    camera_se3_target: &Iso3,
    seed: u64,
    cam: usize,
    view: usize,
) -> (CorrespondenceView, CorrespondenceView) {
    let mut p3 = Vec::new();
    let mut p2 = Vec::new();
    let mut cov = Vec::new();
    for (i, p) in board().iter().enumerate() {
        let Some(uv) = camera.project_point(&camera_se3_target.transform_point(p)) else {
            continue;
        };
        let (c, root) = point_covariance(cam, view, i);
        let e = root * gauss2(seed, cam, view, i);
        p3.push(*p);
        p2.push(Pt2::new(uv.x + e.x, uv.y + e.y));
        cov.push(c);
    }
    (
        CorrespondenceView::new_with_covariances(p3.clone(), p2.clone(), cov).unwrap(),
        CorrespondenceView::new(p3, p2).unwrap(),
    )
}

/// The error of a labelled pinhole-camera parameter.
fn camera_error(truth: &PinholeCamera, estimate: &CameraParams, label: &str) -> f64 {
    let DistortionParams::BrownConrady5 { params: d } = &estimate.distortion else {
        panic!("the estimate lost its Brown-Conrady model");
    };
    let (k, kt, dt) = (&estimate.intrinsics, &truth.k, &truth.dist);
    match label {
        "fx" => k.fx - kt.fx,
        "fy" => k.fy - kt.fy,
        "cx" => k.cx - kt.cx,
        "cy" => k.cy - kt.cy,
        "k1" => d.k1 - dt.k1,
        "k2" => d.k2 - dt.k2,
        "p1" => d.p1 - dt.p1,
        "p2" => d.p2 - dt.p2,
        other => panic!("unexpected camera label {other}"),
    }
}

/// The right-perturbation error `[ω, v]` coordinate of a pose.
fn pose_coordinate(truth: &Iso3, estimate: &Iso3, coordinate: &str) -> f64 {
    let rel = truth.inverse() * estimate;
    let (w, v) = (rel.rotation.scaled_axis(), rel.translation.vector);
    let index = ["rx", "ry", "rz", "tx", "ty", "tz"]
        .iter()
        .position(|c| *c == coordinate)
        .unwrap_or_else(|| panic!("unexpected pose coordinate {coordinate}"));
    [w.x, w.y, w.z, v.x, v.y, v.z][index]
}

/// The reprojection noise level of a result.
fn reprojection_level(cov: &ParameterCovariance) -> NoiseLevel {
    cov.noise
        .iter()
        .find(|n| n.kind == ResidualKind::Reprojection)
        .expect("a reprojection noise level")
        .level
}

fn available(uncertainty: &Uncertainty) -> &ParameterCovariance {
    match uncertainty {
        Uncertainty::Available(cov) => cov,
        other => panic!("no covariance: {other:?}"),
    }
}

/// One run's errors (whitened and uniform solves) and the whitened
/// solve's predicted variances and `σ̂₀`.
#[derive(Default)]
struct Study {
    labels: Vec<String>,
    whitened: Vec<DVector<f64>>,
    uniform: Vec<DVector<f64>>,
    predicted_var: Vec<DVector<f64>>,
    sigma0: Vec<f64>,
}

/// The study's figures.
struct Summary {
    sigma0_mean: f64,
    sigma0_sd: f64,
    /// Per label: predicted σ / Monte-Carlo σ.
    sigma_ratio: Vec<f64>,
    /// Per label: RMS of `err/σ`.
    normalized_rms: Vec<f64>,
    /// Over every label and run: share of `|err| ≤ 2σ`.
    within_2sigma: f64,
    /// Per label: RMSE of the whitened solve over the uniform one's.
    efficiency: Vec<f64>,
}

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

impl Study {
    fn push(
        &mut self,
        whitened: &ParameterCovariance,
        error_whitened: impl Fn(&str) -> f64,
        error_uniform: impl Fn(&str) -> f64,
    ) {
        if self.labels.is_empty() {
            self.labels.clone_from(&whitened.labels);
        }
        assert_eq!(
            self.labels, whitened.labels,
            "labels are stable across runs"
        );
        let NoiseLevel::Sigma0(s0) = reprojection_level(whitened) else {
            panic!("the stated covariances whiten the reprojection residuals");
        };
        let n = self.labels.len();
        let at =
            |f: &dyn Fn(&str) -> f64| DVector::from_iterator(n, self.labels.iter().map(|l| f(l)));
        self.whitened.push(at(&error_whitened));
        self.uniform.push(at(&error_uniform));
        self.predicted_var.push(DVector::from_iterator(
            n,
            (0..n).map(|i| whitened.covariance[i][i]),
        ));
        self.sigma0.push(s0);
    }

    fn summary(&self) -> Summary {
        let runs = self.whitened.len() as f64;
        let n = self.labels.len();
        let column =
            |data: &[DVector<f64>], i: usize| data.iter().map(|e| e[i]).collect::<Vec<_>>();
        let rms = |v: &[f64]| (v.iter().map(|x| x * x).sum::<f64>() / v.len() as f64).sqrt();
        let mut sigma_ratio = Vec::new();
        let mut normalized_rms = Vec::new();
        let mut efficiency = Vec::new();
        let mut within = 0usize;
        for i in 0..n {
            let errors = column(&self.whitened, i);
            let m = mean(&errors);
            let mc_sd = (errors.iter().map(|e| (e - m).powi(2)).sum::<f64>() / (runs - 1.0)).sqrt();
            let predicted = mean(&column(&self.predicted_var, i)).sqrt();
            sigma_ratio.push(predicted / mc_sd);
            let normalized: Vec<f64> = errors
                .iter()
                .zip(column(&self.predicted_var, i))
                .map(|(e, var)| e / var.sqrt())
                .collect();
            within += normalized.iter().filter(|z| z.abs() <= 2.0).count();
            normalized_rms.push(rms(&normalized));
            efficiency.push(rms(&errors) / rms(&column(&self.uniform, i)));
        }
        let sigma0_mean = mean(&self.sigma0);
        let sigma0_sd = (self
            .sigma0
            .iter()
            .map(|s| (s - sigma0_mean).powi(2))
            .sum::<f64>()
            / (runs - 1.0))
            .sqrt();
        Summary {
            sigma0_mean,
            sigma0_sd,
            sigma_ratio,
            normalized_rms,
            within_2sigma: within as f64 / (runs * n as f64),
            efficiency,
        }
    }

    fn report(&self, scene: &str) -> Summary {
        let s = self.summary();
        eprintln!(
            "{scene}: {} runs, σ̂₀ = {:.3} ± {:.3}, {:.1} % of normalized errors within 2σ",
            self.whitened.len(),
            s.sigma0_mean,
            s.sigma0_sd,
            100.0 * s.within_2sigma
        );
        eprintln!(
            "  {:<22} {:>9} {:>9} {:>11}",
            "label", "σ ratio", "norm RMS", "RMSE w/u"
        );
        for (i, label) in self.labels.iter().enumerate() {
            eprintln!(
                "  {label:<22} {:>9.3} {:>9.3} {:>11.3}",
                s.sigma_ratio[i], s.normalized_rms[i], s.efficiency[i]
            );
        }
        s
    }
}

fn runs() -> u64 {
    std::env::var("STUDY_RUNS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(500)
}

/// Planar intrinsics of a Brown-Conrady camera, ten views of a 9×9 board.
fn planar_study(runs: u64) -> Study {
    let camera = pinhole_truth(1150.0, 372.0);
    let poses = board_poses();
    let initial =
        PlanarIntrinsicsParams::new(pinhole_camera_params(&camera), poses.clone()).unwrap();
    let mut study = Study::default();
    for seed in 0..runs {
        let (with, without): (Vec<_>, Vec<_>) = poses
            .iter()
            .enumerate()
            .map(|(v, pose)| {
                let (a, b) = observe(&camera, pose, seed, 0, v);
                (View::without_meta(a), View::without_meta(b))
            })
            .unzip();
        let solve = |views: Vec<View<NoMeta>>| {
            optimize_planar_intrinsics(
                &PlanarDataset::new(views).unwrap(),
                &initial,
                PlanarIntrinsicsSolveOptions::default(),
                BackendSolveOptions::default(),
            )
            .unwrap()
        };
        let whitened = solve(with);
        let uniform = solve(without);
        assert!(matches!(
            reprojection_level(available(&uniform.uncertainty)),
            NoiseLevel::Sigma(_)
        ));
        study.push(
            available(&whitened.uncertainty),
            |l| camera_error(&camera, &whitened.params.camera, l),
            |l| camera_error(&camera, &uniform.params.camera, l),
        );
    }
    study
}

/// Two cameras (intrinsics free, camera 0 the rig frame), ten rig poses.
fn rig_study(runs: u64) -> Study {
    let cameras = [pinhole_truth(1150.0, 372.0), pinhole_truth(1120.0, 380.0)];
    let rig_se3_cam = [
        Iso3::identity(),
        Iso3::from_parts(
            Translation3::new(0.06, 0.002, -0.004),
            UnitQuaternion::from_euler_angles(0.01, -0.08, 0.005),
        ),
    ];
    let rig_se3_target = board_poses();
    let initial = RigExtrinsicsParams {
        cameras: cameras.iter().map(pinhole_camera_params).collect(),
        cam_to_rig: rig_se3_cam.to_vec(),
        rig_from_target: rig_se3_target.clone(),
    };
    let truth_cam1 = rig_se3_cam[1].inverse();
    let mut study = Study::default();
    for seed in 0..runs {
        let mut with = Vec::new();
        let mut without = Vec::new();
        for (v, pose) in rig_se3_target.iter().enumerate() {
            let (a, b): (Vec<_>, Vec<_>) = (0..2)
                .map(|c| {
                    let (a, b) =
                        observe(&cameras[c], &(rig_se3_cam[c].inverse() * pose), seed, c, v);
                    (Some(a), Some(b))
                })
                .unzip();
            with.push(RigView {
                meta: NoMeta,
                obs: RigViewObs { cameras: a },
            });
            without.push(RigView {
                meta: NoMeta,
                obs: RigViewObs { cameras: b },
            });
        }
        let solve = |views| {
            let opts = RigExtrinsicsSolveOptions {
                fix_extrinsics: vec![true, false],
                ..Default::default()
            };
            optimize_rig_extrinsics(
                RigDataset::new(views, 2).unwrap(),
                initial.clone(),
                opts,
                BackendSolveOptions::default(),
            )
            .unwrap()
        };
        let whitened = solve(with);
        let uniform = solve(without);
        let error = |params: &RigExtrinsicsParams, label: &str| {
            let (cam, rest) = label.split_once('.').unwrap();
            let c: usize = cam.strip_prefix("cam").unwrap().parse().unwrap();
            match rest.strip_prefix("cam_se3_rig.") {
                Some(coordinate) => {
                    pose_coordinate(&truth_cam1, &params.cam_to_rig[1].inverse(), coordinate)
                }
                None => camera_error(&cameras[c], &params.cameras[c], rest),
            }
        };
        study.push(
            available(&whitened.uncertainty),
            |l| error(&whitened.params, l),
            |l| error(&uniform.params, l),
        );
    }
    study
}

/// The whitened planar solve: `σ̂₀` near 1, predicted σ near the scatter,
/// honest normalized errors, and a lower error than weighing every point
/// alike.
#[test]
#[ignore = "Monte-Carlo study; run in release with --ignored"]
fn planar_whitened_covariance_matches_monte_carlo() {
    let s = planar_study(runs()).report("planar intrinsics");
    assert!((s.sigma0_mean - 1.0).abs() < 0.05, "σ̂₀ = {}", s.sigma0_mean);
    for (i, r) in s.sigma_ratio.iter().enumerate() {
        assert!((r - 1.0).abs() < 0.10, "parameter {i}: σ ratio {r:.3}");
    }
    for z in &s.normalized_rms {
        assert!((z - 1.0).abs() < 0.10, "normalized RMS {z:.3}");
    }
    assert!(
        (s.within_2sigma - 0.954).abs() < 0.02,
        "{}",
        s.within_2sigma
    );
    for e in &s.efficiency {
        assert!(*e < 1.0, "the whitened solve is less accurate: {e:.3}");
    }
}

/// The same for a two-camera rig, extrinsics and both cameras' intrinsics.
#[test]
#[ignore = "Monte-Carlo study; run in release with --ignored"]
fn rig_whitened_covariance_matches_monte_carlo() {
    let s = rig_study(runs()).report("rig extrinsics");
    assert!((s.sigma0_mean - 1.0).abs() < 0.05, "σ̂₀ = {}", s.sigma0_mean);
    for (i, r) in s.sigma_ratio.iter().enumerate() {
        assert!((r - 1.0).abs() < 0.10, "parameter {i}: σ ratio {r:.3}");
    }
    for z in &s.normalized_rms {
        assert!((z - 1.0).abs() < 0.10, "normalized RMS {z:.3}");
    }
    assert!(
        (s.within_2sigma - 0.954).abs() < 0.02,
        "{}",
        s.within_2sigma
    );
    for e in &s.efficiency {
        assert!(*e < 1.0, "the whitened solve is less accurate: {e:.3}");
    }
}

/// A short version of the planar study for the default run: loose bounds,
/// the whitened noise level is the variance factor and the whitened solve
/// is the more accurate on average.
#[test]
fn planar_whitened_solve_is_consistent() {
    let s = planar_study(12).summary();
    assert!((s.sigma0_mean - 1.0).abs() < 0.15, "σ̂₀ = {}", s.sigma0_mean);
    for r in &s.sigma_ratio {
        assert!((0.5..2.0).contains(r), "σ ratio {r:.3}");
    }
    assert!(mean(&s.efficiency) < 1.0, "{:?}", s.efficiency);
}
