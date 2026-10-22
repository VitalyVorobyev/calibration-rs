//! The reported parameter uncertainty against Monte-Carlo scatter, through
//! the public solve functions.
//!
//! Each gate solves a few hundred noise realizations, so they are ignored in
//! the default (debug) test run; CI runs them in release:
//! `cargo test --release -p vision-calibration-optim --test uncertainty -- --ignored`.

#![allow(missing_docs)]

use nalgebra::{DMatrix, DVector, Matrix2, Translation3, UnitQuaternion, Vector2, Vector3};
use vision_calibration_core::synthetic::noise::UniformPixelNoise;
use vision_calibration_core::{
    BrownConrady5, CameraParams, CameraProject, CorrespondenceView, DistortionParams, FxFyCxCy,
    Iso3, NoMeta, ParameterCovariance, PinholeCamera, PlanarDataset, ProjectionParams, Pt2, Pt3,
    RigDataset, RigView, RigViewObs, ScheimpflugParams, SensorParams, Uncertainty, Vec2, View,
    make_pinhole_camera, pinhole_camera_params,
};
use vision_calibration_optim::{
    BackendSolveOptions, HandeyeDataset, HandeyeMode, HandeyeParams, HandeyeSolveOptions,
    LaserPlane, LaserlineMeta, LaserlineParams, LaserlineSolveOptions, LaserlineView,
    PlanarIntrinsicsParams, PlanarIntrinsicsSolveOptions, RigExtrinsicsParams,
    RigExtrinsicsSolveOptions, RobotPoseMeta, RobustLoss, apply_robot_delta, optimize_handeye,
    optimize_laserline, optimize_planar_intrinsics, optimize_rig_extrinsics,
    projection_uncertainty,
};

const RUNS: u64 = 200;
/// The empirical σ of `RUNS` realizations has a relative standard error of
/// about `1/√(2·RUNS)`, 5 % at 200; a 20-parameter scene whose σ are not
/// dominated by a few directions needs more runs for a ±10 % gate.
const LARGE_SCENE_RUNS: u64 = 400;

/// Predicted against empirical scatter over seeded noise realizations.
#[derive(Default)]
struct MonteCarlo {
    labels: Vec<String>,
    errors: Vec<DVector<f64>>,
    predicted_var: Option<DVector<f64>>,
    nees: f64,
    /// The covariance's rank, the χ² degrees of freedom of the NEES: below
    /// the label count when a unit vector is reported by its components.
    rank: usize,
}

impl MonteCarlo {
    /// Records one run: the estimate's error for each shared label, in the
    /// order of `cov.labels`.
    fn push(&mut self, cov: &ParameterCovariance, error: impl Fn(&str) -> f64) {
        if self.labels.is_empty() {
            self.labels.clone_from(&cov.labels);
        }
        assert_eq!(self.labels, cov.labels, "labels are stable across runs");
        let n = self.labels.len();
        let error = DVector::from_iterator(n, self.labels.iter().map(|l| error(l)));
        let matrix = DMatrix::from_fn(n, n, |i, j| cov.covariance[i][j]);
        // NEES on the correlation matrix, whose rank does not depend on the
        // parameters' units.
        let scale = matrix.diagonal().map(|v| 1.0 / v.sqrt());
        let correlation = DMatrix::from_fn(n, n, |i, j| matrix[(i, j)] * scale[i] * scale[j]);
        let svd = correlation.svd(true, true);
        self.rank = svd.rank(1e-9);
        let inverse = svd.pseudo_inverse(1e-9).unwrap();
        let normalized = error.component_mul(&scale);
        self.nees += (normalized.transpose() * inverse * &normalized)[(0, 0)];
        let var = matrix.diagonal();
        self.predicted_var = Some(match self.predicted_var.take() {
            Some(acc) => acc + var,
            None => var,
        });
        self.errors.push(error);
    }

    /// Each predicted σ within `tolerance` of the empirical one, and the
    /// mean NEES near χ²'s mean (see the band below).
    fn assert_calibrated(&self, tolerance: f64) {
        let runs = self.errors.len() as f64;
        let n = self.labels.len();
        let predicted = (self.predicted_var.clone().unwrap() / runs).map(f64::sqrt);
        let mean = self.errors.iter().fold(DVector::zeros(n), |a, e| a + e) / runs;
        let empirical = (self
            .errors
            .iter()
            .fold(DVector::zeros(n), |a, e| a + (e - &mean).map(|v| v * v))
            / (runs - 1.0))
            .map(f64::sqrt);
        for (i, label) in self.labels.iter().enumerate() {
            let ratio = predicted[i] / empirical[i];
            assert!(
                (ratio - 1.0).abs() < tolerance,
                "{label}: predicted σ {:.3e}, empirical {:.3e} (ratio {ratio:.3})",
                predicted[i],
                empirical[i]
            );
        }
        // The plug-in covariance, evaluated at each run's own estimate, puts
        // the NEES a few percent above χ²'s mean; that offset does not
        // shrink with the run count, so the band is the wider of three
        // standard errors and the σ tolerance.
        let k = self.rank as f64;
        let nees = self.nees / runs;
        let band = (3.0 * (2.0 * k / runs).sqrt()).max(tolerance * k);
        let ratios = predicted.component_div(&empirical);
        eprintln!(
            "{n} parameters: σ ratio {:.2}–{:.2}, NEES {nees:.2}",
            ratios.min(),
            ratios.max()
        );
        assert!(
            (nees - k).abs() < band,
            "NEES mean {nees:.3}, expected {k} ± {band:.3}"
        );
    }
}

/// The right-perturbation coordinates `[ω, v]` of `estimate = truth·Exp(δ)`,
/// to first order.
fn pose_error(truth: &Iso3, estimate: &Iso3) -> [f64; 6] {
    let rel = truth.inverse() * estimate;
    let w = rel.rotation.scaled_axis();
    let v = rel.translation.vector;
    [w.x, w.y, w.z, v.x, v.y, v.z]
}

/// The error of a labelled pose coordinate (`<name>.rx` … `<name>.tz`).
fn pose_coordinate(truth: &Iso3, estimate: &Iso3, coordinate: &str) -> f64 {
    let index = ["rx", "ry", "rz", "tx", "ty", "tz"]
        .iter()
        .position(|c| *c == coordinate)
        .unwrap_or_else(|| panic!("unexpected pose coordinate {coordinate}"));
    pose_error(truth, estimate)[index]
}

/// The error of a labelled pinhole-camera parameter.
fn camera_error(truth: &PinholeCamera, estimate: &CameraParams, label: &str) -> f64 {
    let DistortionParams::BrownConrady5 { params: d } = &estimate.distortion else {
        panic!("the estimate lost its Brown-Conrady model");
    };
    let (k, kt) = (&estimate.intrinsics, &truth.k);
    let dt = &truth.dist;
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

/// The board's projections in `camera` at `camera_se3_target`, with noise
/// from stream `stream`.
fn observe(
    camera: &PinholeCamera,
    camera_se3_target: &Iso3,
    noise: &UniformPixelNoise,
    stream: usize,
) -> CorrespondenceView {
    let (p3, p2): (Vec<Pt3>, Vec<Pt2>) = board()
        .iter()
        .enumerate()
        .filter_map(|(i, p)| {
            let uv = camera.project_point(&camera_se3_target.transform_point(p))?;
            let uv = noise.apply(stream, i, Vec2::new(uv.x, uv.y));
            Some((*p, Pt2::new(uv.x, uv.y)))
        })
        .unzip();
    CorrespondenceView::new(p3, p2).unwrap()
}

fn truth() -> PlanarIntrinsicsParams {
    let poses = [
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
    });
    PlanarIntrinsicsParams::new(
        CameraParams {
            projection: ProjectionParams::Pinhole,
            distortion: DistortionParams::BrownConrady5 {
                params: BrownConrady5 {
                    k1: -0.08,
                    k2: 0.04,
                    k3: 0.0,
                    p1: 0.0,
                    p2: 0.0,
                    iters: 8,
                },
            },
            sensor: SensorParams::Scheimpflug {
                params: ScheimpflugParams {
                    tilt_x: 0.05,
                    tilt_y: -0.02,
                },
            },
            intrinsics: FxFyCxCy {
                fx: 1150.0,
                fy: 1155.0,
                cx: 372.0,
                cy: 258.0,
            },
        },
        poses.to_vec(),
    )
    .unwrap()
}

/// The camera model of a Scheimpflug parameter pack.
fn scheimpflug_camera(params: &PlanarIntrinsicsParams) -> CameraParams {
    params.camera.clone()
}

/// Views of a 9×9 board of 20 mm cells, with seeded pixel noise.
fn dataset(params: &PlanarIntrinsicsParams, noise: UniformPixelNoise) -> PlanarDataset {
    let camera = scheimpflug_camera(params).build().unwrap();
    let board: Vec<Pt3> = (0..81)
        .map(|i| {
            Pt3::new(
                (i % 9) as f64 * 0.02 - 0.08,
                (i / 9) as f64 * 0.02 - 0.08,
                0.0,
            )
        })
        .collect();
    let views = params
        .camera_se3_target
        .iter()
        .enumerate()
        .map(|(v, pose)| {
            let (p3, p2): (Vec<Pt3>, Vec<Pt2>) = board
                .iter()
                .enumerate()
                .filter_map(|(i, p)| {
                    let uv = camera.project_camera_point(&pose.transform_point(p).coords)?;
                    let uv = noise.apply(v, i, Vec2::new(uv.x, uv.y));
                    Some((*p, Pt2::new(uv.x, uv.y)))
                })
                .unzip();
            View::without_meta(CorrespondenceView::new(p3, p2).unwrap())
        })
        .collect();
    PlanarDataset::new(views).unwrap()
}

/// The labelled value of a Scheimpflug parameter pack.
fn value(params: &PlanarIntrinsicsParams, label: &str) -> f64 {
    let k = params.camera.intrinsics;
    let DistortionParams::BrownConrady5 { params: d } = &params.camera.distortion else {
        unreachable!("the scene uses Brown-Conrady");
    };
    let SensorParams::Scheimpflug { params: tilt } = &params.camera.sensor else {
        unreachable!("the scene has a Scheimpflug sensor");
    };
    match label {
        "fx" => k.fx,
        "fy" => k.fy,
        "cx" => k.cx,
        "cy" => k.cy,
        "k1" => d.k1,
        "k2" => d.k2,
        "tilt_x" => tilt.tilt_x,
        "tilt_y" => tilt.tilt_y,
        other => panic!("unexpected label {other}"),
    }
}

/// The Scheimpflug camera's predicted σ (intrinsics, radial distortion,
/// tilt) match their scatter over seeded noise within 10 %, and the NEES
/// matches χ².
#[test]
#[ignore = "Monte-Carlo gate; run in release with --ignored"]
fn scheimpflug_covariance_matches_monte_carlo() {
    let gt = truth();
    let mut mc = MonteCarlo::default();
    for seed in 0..RUNS {
        let data = dataset(
            &gt,
            UniformPixelNoise {
                seed: 500 + seed,
                max_abs_px: 0.5,
            },
        );
        // The Scheimpflug model's default fix list holds k3, p1, p2.
        let opts = PlanarIntrinsicsSolveOptions::default();
        let est =
            optimize_planar_intrinsics(&data, &gt, opts, BackendSolveOptions::default()).unwrap();
        let Uncertainty::Available(cov) = &est.uncertainty else {
            panic!("no covariance: {:?}", est.uncertainty);
        };
        assert!(cov.fixed.contains(&"k3".to_string()), "{:?}", cov.fixed);
        assert!(cov.fixed.contains(&"p1".to_string()), "{:?}", cov.fixed);
        assert_eq!(cov.views.len(), gt.camera_se3_target.len());
        mc.push(cov, |l| value(&est.params, l) - value(&gt, l));
    }
    assert_eq!(
        mc.labels,
        ["fx", "fy", "cx", "cy", "k1", "k2", "tilt_x", "tilt_y"]
    );
    mc.assert_calibrated(0.10);
}

/// The projection σ of the Scheimpflug camera matches the scatter of where
/// fixed camera-frame rays land, at the image centre, two opposite corners
/// (outside the board's coverage, so extrapolated through the distortion)
/// and an interior node.
#[test]
#[ignore = "Monte-Carlo gate; run in release with --ignored"]
fn projection_uncertainty_matches_monte_carlo() {
    const IMAGE: [u32; 2] = [744, 516];
    let gt = truth();
    let truth_camera = scheimpflug_camera(&gt).build().unwrap();
    let map = |est: &PlanarIntrinsicsParams, cov: &ParameterCovariance| {
        projection_uncertainty(&scheimpflug_camera(est), cov, 0, IMAGE).unwrap()
    };
    // Probe pixels and the σ each map reports there.
    let first = map(&gt, &solve_scheimpflug(&gt, 0).1);
    let node = |c: usize, r: usize| {
        let x = c as f64 * f64::from(IMAGE[0] - 1) / (first.columns - 1) as f64;
        let y = r as f64 * f64::from(IMAGE[1] - 1) / (first.rows - 1) as f64;
        (Pt2::new(x, y), r * first.columns + c)
    };
    let (last_c, last_r) = (first.columns - 1, first.rows - 1);
    let probes = [node(0, 0), node(last_c, last_r), node(3, 2)];
    let center = Pt2::new(f64::from(IMAGE[0] - 1) / 2.0, f64::from(IMAGE[1] - 1) / 2.0);
    let rays: Vec<Vector3<f64>> = std::iter::once(center)
        .chain(probes.iter().map(|(p, _)| *p))
        .map(|p| {
            truth_camera
                .backproject_pixel(&p)
                .point_at_unit_depth()
                .unwrap()
        })
        .collect();

    let mut pixels: Vec<Vec<Vector2<f64>>> = vec![Vec::new(); rays.len()];
    let mut predicted = vec![0.0; rays.len()];
    for seed in 0..RUNS {
        let (est, cov) = solve_scheimpflug(&gt, seed);
        let map = map(&est, &cov);
        let sigma =
            std::iter::once(map.center).chain(probes.iter().map(|(_, i)| map.sigma[*i].unwrap()));
        for (acc, s) in predicted.iter_mut().zip(sigma) {
            *acc += s * s;
        }
        let camera = scheimpflug_camera(&est).build().unwrap();
        for (ray, out) in rays.iter().zip(&mut pixels) {
            let q = camera.project_point_c(ray).unwrap();
            out.push(Vector2::new(q.x, q.y));
        }
    }
    let runs = RUNS as f64;
    for (k, samples) in pixels.iter().enumerate() {
        let mean = samples.iter().fold(Vector2::zeros(), |a, q| a + q) / runs;
        let scatter = samples.iter().fold(Matrix2::zeros(), |a, q| {
            a + (q - mean) * (q - mean).transpose()
        }) / (runs - 1.0);
        let empirical = scatter.symmetric_eigenvalues().max().sqrt();
        let predicted = (predicted[k] / runs).sqrt();
        let ratio = predicted / empirical;
        eprintln!("probe {k}: predicted σ {predicted:.4} px, empirical {empirical:.4} px");
        assert!(
            (ratio - 1.0).abs() < 0.10,
            "probe {k}: predicted σ {predicted:.4e}, empirical {empirical:.4e} (ratio {ratio:.3})"
        );
    }
    assert!(
        first.corner.unwrap() > first.center,
        "extrapolation is less certain"
    );
}

/// One noise realization of the Scheimpflug scene, solved.
fn solve_scheimpflug(
    gt: &PlanarIntrinsicsParams,
    seed: u64,
) -> (PlanarIntrinsicsParams, ParameterCovariance) {
    let data = dataset(
        gt,
        UniformPixelNoise {
            seed: 500 + seed,
            max_abs_px: 0.5,
        },
    );
    let opts = PlanarIntrinsicsSolveOptions::default();
    let est = optimize_planar_intrinsics(&data, gt, opts, BackendSolveOptions::default()).unwrap();
    let Uncertainty::Available(cov) = est.uncertainty else {
        panic!("no covariance: {:?}", est.uncertainty);
    };
    (est.params, cov)
}

/// A stereo rig whose cameras are refined with their extrinsics: the σ of
/// both cameras' intrinsics and of camera 1's `cam_se3_rig` (reported as
/// the inverse of the stored camera-to-rig pose) match their scatter.
#[test]
#[ignore = "Monte-Carlo gate; run in release with --ignored"]
fn rig_extrinsics_covariance_matches_monte_carlo() {
    let cameras = [pinhole_truth(1150.0, 372.0), pinhole_truth(1120.0, 380.0)];
    // rig_se3_cam: camera 1 sits 6 cm to the right, toed in.
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

    let mut mc = MonteCarlo::default();
    for seed in 0..RUNS {
        let noise = UniformPixelNoise {
            seed: 900 + seed,
            max_abs_px: 0.5,
        };
        let views = rig_se3_target
            .iter()
            .enumerate()
            .map(|(v, pose)| RigView {
                meta: NoMeta,
                obs: RigViewObs {
                    cameras: (0..2)
                        .map(|c| {
                            let cam_se3_target = rig_se3_cam[c].inverse() * pose;
                            Some(observe(&cameras[c], &cam_se3_target, &noise, 2 * v + c))
                        })
                        .collect(),
                },
            })
            .collect();
        let data = RigDataset::new(views, 2).unwrap();
        let opts = RigExtrinsicsSolveOptions {
            fix_extrinsics: vec![true, false],
            ..Default::default()
        };
        let est =
            optimize_rig_extrinsics(data, initial.clone(), opts, BackendSolveOptions::default())
                .unwrap();
        let Uncertainty::Available(cov) = &est.uncertainty else {
            panic!("no covariance: {:?}", est.uncertainty);
        };
        assert!(cov.fixed.contains(&"cam0.cam_se3_rig".to_string()));
        assert_eq!(cov.views[0].labels[0], "rig_se3_target.rx");
        let cam_se3_rig_1 = (rig_se3_cam[1].inverse(), est.params.cam_to_rig[1].inverse());
        mc.push(cov, |label| {
            let (cam, rest) = label.split_once('.').unwrap();
            let c: usize = cam.strip_prefix("cam").unwrap().parse().unwrap();
            match rest.strip_prefix("cam_se3_rig.") {
                Some(coordinate) => {
                    assert_eq!(c, 1);
                    pose_coordinate(&cam_se3_rig_1.0, &cam_se3_rig_1.1, coordinate)
                }
                None => camera_error(&cameras[c], &est.params.cameras[c], rest),
            }
        });
    }
    assert!(mc.labels.contains(&"cam1.cam_se3_rig.tz".to_string()));
    mc.assert_calibrated(0.10);
}

/// An eye-in-hand scene: the camera, its hand-eye pose and the board in the
/// base frame, and the robot poses that put the camera at each board pose
/// (`camera_se3_target = X⁻¹ · B⁻¹ · Y`, so `B = Y · camera_se3_target⁻¹ · X⁻¹`).
struct EyeInHand {
    camera: PinholeCamera,
    gripper_se3_camera: Iso3,
    base_se3_target: Iso3,
    camera_se3_target: Vec<Iso3>,
    base_se3_gripper: Vec<Iso3>,
}

impl EyeInHand {
    fn new() -> Self {
        let gripper_se3_camera = Iso3::from_parts(
            Translation3::new(0.03, -0.02, 0.08),
            UnitQuaternion::from_euler_angles(0.1, -0.05, 0.3),
        );
        let base_se3_target = Iso3::from_parts(
            Translation3::new(0.6, -0.1, 0.02),
            UnitQuaternion::from_euler_angles(0.0, 0.05, 1.2),
        );
        let camera_se3_target = board_poses();
        let base_se3_gripper = camera_se3_target
            .iter()
            .map(|c| base_se3_target * c.inverse() * gripper_se3_camera.inverse())
            .collect();
        Self {
            camera: pinhole_truth(1150.0, 372.0),
            gripper_se3_camera,
            base_se3_target,
            camera_se3_target,
            base_se3_gripper,
        }
    }

    fn initial(&self) -> HandeyeParams {
        HandeyeParams {
            cameras: vec![pinhole_camera_params(&self.camera)],
            cam_to_rig: vec![Iso3::identity()],
            handeye: self.gripper_se3_camera,
            target_poses: vec![self.base_se3_target],
        }
    }

    /// The dataset with pixel noise and the given (observed) robot poses.
    fn dataset(&self, noise: &UniformPixelNoise, robot: &[Iso3]) -> HandeyeDataset {
        let views = self
            .camera_se3_target
            .iter()
            .zip(robot)
            .enumerate()
            .map(|(v, (c, b))| RigView {
                meta: RobotPoseMeta {
                    base_se3_gripper: *b,
                },
                obs: RigViewObs {
                    cameras: vec![Some(observe(&self.camera, c, noise, v))],
                },
            })
            .collect();
        HandeyeDataset::new(views, 1, HandeyeMode::EyeInHand).unwrap()
    }

    /// The error of a shared label of a hand-eye estimate.
    fn error(&self, est: &vision_calibration_optim::HandeyeEstimate, label: &str) -> f64 {
        match label.split_once('.') {
            Some(("gripper_se3_camera", c)) => {
                pose_coordinate(&self.gripper_se3_camera, &est.params.handeye, c)
            }
            Some(("base_se3_target", c)) => {
                pose_coordinate(&self.base_se3_target, &est.params.target_poses[0], c)
            }
            _ => camera_error(&self.camera, &est.params.cameras[0], label),
        }
    }
}

/// A single eye-in-hand camera with exact robot poses: the σ of the
/// intrinsics, the hand-eye pose and the target pose match their scatter.
#[test]
#[ignore = "Monte-Carlo gate; run in release with --ignored"]
fn handeye_covariance_matches_monte_carlo() {
    let scene = EyeInHand::new();
    let mut mc = MonteCarlo::default();
    for seed in 0..RUNS {
        let noise = UniformPixelNoise {
            seed: 1300 + seed,
            max_abs_px: 0.5,
        };
        let data = scene.dataset(&noise, &scene.base_se3_gripper);
        let opts = HandeyeSolveOptions {
            fix_extrinsics: vec![true],
            ..Default::default()
        };
        let est =
            optimize_handeye(data, scene.initial(), opts, BackendSolveOptions::default()).unwrap();
        let Uncertainty::Available(cov) = &est.uncertainty else {
            panic!("no covariance: {:?}", est.uncertainty);
        };
        assert_eq!(
            cov.fixed,
            ["k3"],
            "the identity camera-to-rig pose is not listed"
        );
        mc.push(cov, |label| scene.error(&est, label));
    }
    assert!(mc.labels.contains(&"gripper_se3_camera.tz".to_string()));
    mc.assert_calibrated(0.10);
}

/// The same scene with every robot pose perturbed as the robot-pose priors
/// assume: refining them, with the priors weighed against the estimated
/// pixel noise and counted as known information, the σ still match their
/// scatter. The default priors (0.5°, 1 mm) at ±0.5 px.
#[test]
#[ignore = "Monte-Carlo gate; run in release with --ignored"]
fn handeye_with_robot_pose_priors_matches_monte_carlo() {
    let defaults = HandeyeSolveOptions::default();
    robot_pose_priors_monte_carlo(0.5, defaults.robot_rot_sigma, defaults.robot_trans_sigma);
}

/// As above with sub-pixel corners (±0.17 px, σ 0.1 px) and an accurate
/// robot (0.05°, 0.2 mm), where weighing the priors against 1 px rather than
/// the pixel noise made them too strong for the data.
#[test]
#[ignore = "Monte-Carlo gate; run in release with --ignored"]
fn handeye_with_sub_pixel_corners_and_robot_pose_priors_matches_monte_carlo() {
    robot_pose_priors_monte_carlo(0.17, 0.05f64.to_radians(), 2.0e-4);
}

/// The robot-pose prior gate: pixel noise uniform in `±max_abs_px`, robot
/// poses perturbed with the priors' standard deviations.
fn robot_pose_priors_monte_carlo(max_abs_px: f64, rot_sigma: f64, trans_sigma: f64) {
    let scene = EyeInHand::new();
    let opts = HandeyeSolveOptions {
        fix_extrinsics: vec![true],
        refine_robot_poses: true,
        robot_rot_sigma: rot_sigma,
        robot_trans_sigma: trans_sigma,
        ..Default::default()
    };
    // Uniform noise of the priors' standard deviation: half-width σ·√3.
    let half_width = [
        opts.robot_rot_sigma * 3f64.sqrt(),
        opts.robot_trans_sigma * 3f64.sqrt(),
    ];
    let mut mc = MonteCarlo::default();
    for seed in 0..LARGE_SCENE_RUNS {
        let mut state = 0x5DEE_CE66_D1CE_4E5B ^ seed;
        let mut uniform = move || {
            // SplitMix64 to a value in [-1, 1).
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            (z >> 11) as f64 / (1u64 << 52) as f64 - 1.0
        };
        let robot: Vec<Iso3> = scene
            .base_se3_gripper
            .iter()
            .map(|b| {
                let eps: [f64; 6] = std::array::from_fn(|i| -half_width[i / 3] * uniform());
                apply_robot_delta(b, &eps)
            })
            .collect();
        let noise = UniformPixelNoise {
            seed: 1700 + seed,
            max_abs_px,
        };
        let data = scene.dataset(&noise, &robot);
        let est = optimize_handeye(
            data,
            scene.initial(),
            opts.clone(),
            BackendSolveOptions::default(),
        )
        .unwrap();
        let Uncertainty::Available(cov) = &est.uncertainty else {
            panic!("no covariance: {:?}", est.uncertainty);
        };
        assert_eq!(cov.views.len(), scene.base_se3_gripper.len());
        assert_eq!(cov.views[0].labels[0], "robot_delta.rx");
        assert_eq!(cov.fixed, ["k3"]);
        mc.push(cov, |label| scene.error(&est, label));
    }
    mc.assert_calibrated(0.10);
}

/// A laser sheet beside the camera, crossing the optical axis at 0.5 m.
fn laser_plane_truth() -> LaserPlane {
    let normal = Vector3::new(0.5, 0.04, 0.1).normalize();
    LaserPlane::new(normal, -normal.dot(&Vector3::new(0.1, 0.0, 0.0)))
}

/// The laser line on the board at `camera_se3_target`: points of the board's
/// plane on the laser plane, within the board, projected with seeded noise.
fn laser_pixels(
    camera: &PinholeCamera,
    camera_se3_target: &Iso3,
    plane: &LaserPlane,
    noise: &UniformPixelNoise,
    stream: usize,
) -> Vec<Pt2> {
    // The laser plane in target coordinates, `n_t · p + d_t = 0`.
    let n_t = camera_se3_target.rotation.inverse() * plane.normal.into_inner();
    let d_t = plane.distance + plane.normal.dot(&camera_se3_target.translation.vector);
    (0..60)
        .filter_map(|i| {
            let y = -0.08 + 0.16 * i as f64 / 59.0;
            let x = -(n_t.y * y + d_t) / n_t.x;
            if x.abs() > 0.08 {
                return None;
            }
            let p = camera_se3_target.transform_point(&Pt3::new(x, y, 0.0));
            let uv = camera.project_point(&p)?;
            let uv = noise.apply(stream, i, Vec2::new(uv.x, uv.y));
            Some(Pt2::new(uv.x, uv.y))
        })
        .collect()
}

/// The labelled value of a laserline parameter pack.
fn laserline_value(params: &LaserlineParams, label: &str) -> f64 {
    let n = params.plane.normal;
    match label {
        "plane.nx" => n.x,
        "plane.ny" => n.y,
        "plane.nz" => n.z,
        "plane.d" => params.plane.distance,
        camera_label => camera_error(&pinhole_truth(0.0, 0.0), &params.camera, camera_label),
    }
}

/// The laserline device's predicted σ (intrinsics, distortion and the laser
/// plane, from corner and laser-line residuals together) match their
/// scatter over seeded noise within 10 %, and the NEES matches χ².
#[test]
#[ignore = "Monte-Carlo gate; run in release with --ignored"]
fn laserline_covariance_matches_monte_carlo() {
    let camera = pinhole_truth(1150.0, 372.0);
    let plane = laser_plane_truth();
    let poses = board_poses();
    let gt =
        LaserlineParams::new(pinhole_camera_params(&camera), poses.clone(), plane.clone()).unwrap();
    let mut mc = MonteCarlo::default();
    for seed in 0..LARGE_SCENE_RUNS {
        let corners = UniformPixelNoise {
            seed: 900 + seed,
            max_abs_px: 0.5,
        };
        let laser = UniformPixelNoise {
            seed: 7_000 + seed,
            max_abs_px: 0.5,
        };
        let data: Vec<LaserlineView> = poses
            .iter()
            .enumerate()
            .map(|(v, pose)| {
                View::new(
                    observe(&camera, pose, &corners, v),
                    LaserlineMeta {
                        laser_pixels: laser_pixels(&camera, pose, &plane, &laser, v),
                        laser_weights: Vec::new(),
                    },
                )
            })
            .collect();
        let est = optimize_laserline(
            &data,
            &gt,
            &LaserlineSolveOptions {
                // A Huber scale in the line distance's unit, pixels.
                laser_loss: RobustLoss::Huber { scale: 1.0 },
                ..LaserlineSolveOptions::default()
            },
            &BackendSolveOptions::default(),
        )
        .unwrap();
        let Uncertainty::Available(cov) = &est.uncertainty else {
            panic!("no covariance: {:?}", est.uncertainty);
        };
        assert_eq!(cov.noise.len(), 2, "corner and laser-line noise");
        assert_eq!(cov.views.len(), poses.len(), "no pose is held fixed");
        mc.push(cov, |l| {
            laserline_value(&est.params, l) - laserline_value(&gt, l)
        });
    }
    assert_eq!(
        mc.labels,
        [
            "fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "plane.nx", "plane.ny", "plane.nz",
            "plane.d"
        ]
    );
    mc.assert_calibrated(0.10);
}
