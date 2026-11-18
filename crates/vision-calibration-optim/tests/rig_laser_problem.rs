//! A robotless joint rig + laser-plane solve through the public problem
//! builder: two cameras (one pinhole, one tilted), shared per-capture board
//! poses, a laser plane per camera, sparse camera/capture coverage and noisy
//! seeds; no robot poses.
//!
//! The Monte-Carlo gate at the end is ignored in the default (debug) run;
//! CI runs it in release:
//! `cargo test --release -p vision-calibration-optim --test rig_laser_problem -- --ignored`.

#![allow(missing_docs)]

use nalgebra::{DMatrix, Translation3, UnitQuaternion, Vector3};
use vision_calibration_core::synthetic::laser::{BoardExtent, laser_stripe_pixels};
use vision_calibration_core::synthetic::noise::UniformPixelNoise;
use vision_calibration_core::{
    BrownConrady5, CalibratedRig, CameraFix, CameraParams, DistortionParams, FxFyCxCy, Iso3,
    LaserPlane, ProjectionParams, Pt2, Pt3, ResidualKind, ScheimpflugParams, SensorParams,
    Uncertainty, UncertaintyUnavailable, Vec2,
};
use vision_calibration_optim::problem::{
    BlockId, CameraId, LaserNoise, PixelNoise, PlaneId, PoseChain, Problem, Solution, SolveOptions,
    TargetPoint, solve,
};
use vision_calibration_optim::{
    LaserlineResidualType, RobustLoss, SolverBackend, camera_parameters,
};

const CAPTURES: usize = 10;
const BOARD: (usize, usize, f64) = (13, 10, 0.02);
const WIDTH: f64 = 1280.0;
const HEIGHT: f64 = 960.0;
/// Corner and laser-pixel noise, uniform in `±` this many pixels.
const CORNER_NOISE: f64 = 0.2;
const LASER_NOISE: f64 = 0.3;

/// The cameras that see the board's corners in capture `k`.
fn sees_corners(cam: usize, k: usize) -> bool {
    match cam {
        0 => k <= 6,
        _ => k >= 3,
    }
}

/// The cameras that see the laser line in capture `k`: camera 1 sees it in
/// capture 2, where only camera 0 sees the corners.
fn sees_laser(cam: usize, k: usize) -> bool {
    match cam {
        0 => k <= 6 && k != 2,
        _ => k >= 2 && k != 8,
    }
}

fn iso(rotation: [f64; 3], translation: [f64; 3]) -> Iso3 {
    Iso3::from_parts(
        Translation3::new(translation[0], translation[1], translation[2]),
        UnitQuaternion::from_euler_angles(rotation[0], rotation[1], rotation[2]),
    )
}

#[derive(Clone)]
struct Scene {
    cameras: [CameraParams; 2],
    /// `cam_se3_rig`; camera 0 is the rig frame.
    cam_se3_rig: [Iso3; 2],
    rig_se3_target: Vec<Iso3>,
    /// Each camera's laser plane, in its own frame.
    planes: [LaserPlane; 2],
}

fn truth() -> Scene {
    let bc = |k1, k2, p1, p2| DistortionParams::BrownConrady5 {
        params: BrownConrady5 {
            k1,
            k2,
            k3: 0.0,
            p1,
            p2,
            iters: 8,
        },
    };
    let cam0 = CameraParams::new(
        ProjectionParams::Pinhole,
        bc(-0.08, 0.05, 0.0005, -0.0003),
        SensorParams::Identity,
        FxFyCxCy {
            fx: 1000.0,
            fy: 996.0,
            cx: 642.0,
            cy: 478.0,
        },
    );
    let cam1 = CameraParams::new(
        ProjectionParams::Pinhole,
        bc(-0.05, 0.02, 0.0, 0.0),
        SensorParams::Scheimpflug {
            params: ScheimpflugParams {
                tilt_x: 0.035,
                tilt_y: -0.0175,
            },
        },
        FxFyCxCy {
            fx: 1100.0,
            fy: 1098.0,
            cx: 655.0,
            cy: 470.0,
        },
    );
    // Camera 1 sits 15 cm along x, turned toward the working volume.
    let rig_se3_cam1 = iso([0.0, -0.149, 0.0], [0.15, 0.0, 0.0]);
    let cam_se3_rig = [Iso3::identity(), rig_se3_cam1.inverse()];
    let (nx, ny, s) = BOARD;
    let half = Translation3::new(-0.5 * (nx - 1) as f64 * s, -0.5 * (ny - 1) as f64 * s, 0.0);
    let rx = [0.25, -0.2, 0.1, -0.3, 0.15, 0.0, -0.1, 0.3, -0.25, 0.2];
    let ry = [-0.2, 0.3, 0.25, 0.0, -0.3, 0.2, -0.15, 0.1, 0.3, -0.25];
    let rz = [0.1, -0.05, 0.2, 0.0, -0.15, 0.05, 0.1, -0.2, 0.0, 0.15];
    let dx = [-0.02, 0.02, 0.0, 0.03, -0.03, 0.01, -0.01, 0.02, 0.0, -0.02];
    let dy = [
        0.01, -0.015, 0.015, 0.0, -0.01, 0.005, -0.01, 0.0, 0.01, -0.005,
    ];
    let dz = [
        0.0, 0.05, -0.05, 0.03, -0.03, 0.08, -0.08, 0.02, -0.02, 0.06,
    ];
    let rig_se3_target = (0..CAPTURES)
        .map(|k| iso([rx[k], ry[k], rz[k]], [0.075 + dx[k], dy[k], 0.5 + dz[k]]) * half)
        .collect();
    // One laser sheet, mostly across the boards' long side, in the rig frame.
    let normal = Vector3::new(0.2, 1.0, 0.1).normalize();
    let through = Vector3::new(0.075, 0.0, 0.5);
    let laser_rig = LaserPlane::new(normal, -normal.dot(&through)).unwrap();
    let planes = [
        laser_rig.transform_by(&cam_se3_rig[0]),
        laser_rig.transform_by(&cam_se3_rig[1]),
    ];
    Scene {
        cameras: [cam0, cam1],
        cam_se3_rig,
        rig_se3_target,
        planes,
    }
}

fn board() -> Vec<Pt3> {
    let (nx, ny, s) = BOARD;
    (0..ny)
        .flat_map(|j| (0..nx).map(move |i| Pt3::new(i as f64 * s, j as f64 * s, 0.0)))
        .collect()
}

/// One camera's board corners in one capture: target point and pixel.
type Corners = Vec<(Pt3, Pt2)>;

/// Observations keyed by `(capture, camera)`: corners and laser pixels.
struct Observations {
    corners: Vec<[Option<Corners>; 2]>,
    laser: Vec<[Option<Vec<Pt2>>; 2]>,
}

fn observe(scene: &Scene, seed: u64) -> Observations {
    let corner_noise = UniformPixelNoise {
        seed,
        max_abs_px: CORNER_NOISE,
    };
    let laser_noise = UniformPixelNoise {
        seed: seed ^ 0x5eed_1a5e,
        max_abs_px: LASER_NOISE,
    };
    let models = scene.cameras.clone().map(|c| c.build().unwrap());
    let extent = BoardExtent::from_grid(BOARD.0, BOARD.1, BOARD.2);
    let in_image = |p: &Pt2| p.x >= 0.0 && p.y >= 0.0 && p.x < WIDTH && p.y < HEIGHT;
    let mut corners = Vec::new();
    let mut laser = Vec::new();
    for k in 0..CAPTURES {
        let mut c = [None, None];
        let mut l = [None, None];
        for cam in 0..2 {
            let cam_se3_target = scene.cam_se3_rig[cam] * scene.rig_se3_target[k];
            let key = 2 * k + cam;
            if sees_corners(cam, k) {
                let obs: Corners = board()
                    .into_iter()
                    .enumerate()
                    .map(|(i, p)| {
                        let uv = models[cam]
                            .project_point(&cam_se3_target.transform_point(&p))
                            .expect("the board is in front of the camera");
                        let uv = corner_noise.apply(key, i, Vec2::new(uv.x, uv.y));
                        (p, Pt2::new(uv.x, uv.y))
                    })
                    .collect();
                assert!(
                    obs.iter().all(|(_, uv)| in_image(uv)),
                    "capture {k} cam {cam}"
                );
                c[cam] = Some(obs);
            }
            if sees_laser(cam, k) {
                let pixels = laser_stripe_pixels(
                    &models[cam],
                    &cam_se3_target,
                    &scene.planes[cam].normal().into_inner(),
                    scene.planes[cam].distance(),
                    &extent,
                    key,
                    &laser_noise,
                );
                assert!(pixels.len() > 20, "a stripe in capture {k} cam {cam}");
                assert!(pixels.iter().all(in_image));
                l[cam] = Some(pixels);
            }
        }
        corners.push(c);
        laser.push(l);
    }
    Observations { corners, laser }
}

/// `pose · Exp([angle·axis, trans·dir])` with a direction varying with `k`.
fn perturb(pose: &Iso3, angle_deg: f64, trans_m: f64, k: usize) -> Iso3 {
    let a = k as f64;
    let axis = Vector3::new(1.0, (a * 1.3).sin(), (a * 0.7).cos()).normalize();
    let dir = Vector3::new((a * 0.9).cos(), 1.0, (a * 1.7).sin()).normalize();
    pose * Iso3::from_parts(
        Translation3::from(dir * trans_m),
        UnitQuaternion::from_scaled_axis(axis * angle_deg.to_radians()),
    )
}

/// The seeds: intrinsics 1 % / 3 px off, distortion zero, `tilt_x` 0.5°
/// off, camera 1's extrinsic and every board pose 1° / 5 mm off, each plane
/// 2° / 3 mm off.
fn seeds(truth: &Scene) -> Scene {
    let camera = |c: &CameraParams, tilt: bool| {
        let k = c.intrinsics;
        let DistortionParams::BrownConrady5 { params: d } = &c.distortion else {
            unreachable!()
        };
        let sensor = match (&c.sensor, tilt) {
            (SensorParams::Scheimpflug { params }, true) => SensorParams::Scheimpflug {
                params: ScheimpflugParams {
                    tilt_x: params.tilt_x + 0.5_f64.to_radians(),
                    tilt_y: params.tilt_y,
                },
            },
            (s, _) => s.clone(),
        };
        CameraParams::new(
            ProjectionParams::Pinhole,
            DistortionParams::BrownConrady5 {
                params: BrownConrady5 {
                    k1: 0.0,
                    k2: 0.0,
                    k3: 0.0,
                    p1: 0.0,
                    p2: 0.0,
                    iters: d.iters,
                },
            },
            sensor,
            FxFyCxCy {
                fx: k.fx * 1.01,
                fy: k.fy * 1.01,
                cx: k.cx + 3.0,
                cy: k.cy - 3.0,
            },
        )
    };
    let plane = |p: &LaserPlane, k: usize| {
        let axis = Vector3::new(1.0, 0.3 * k as f64, 0.5).normalize();
        let n =
            UnitQuaternion::from_scaled_axis(axis * 2.0_f64.to_radians()) * p.normal().into_inner();
        LaserPlane::new(n, p.distance() + 0.003).unwrap()
    };
    Scene {
        cameras: [
            camera(&truth.cameras[0], false),
            camera(&truth.cameras[1], true),
        ],
        cam_se3_rig: [
            truth.cam_se3_rig[0],
            perturb(&truth.cam_se3_rig[1], 1.0, 0.005, 11),
        ],
        rig_se3_target: truth
            .rig_se3_target
            .iter()
            .enumerate()
            .map(|(k, p)| perturb(p, 1.0, 0.005, k))
            .collect(),
        planes: [plane(&truth.planes[0], 0), plane(&truth.planes[1], 1)],
    }
}

/// What to hold fixed.
#[derive(Clone)]
struct Fixes {
    cameras: [CameraFix; 2],
    cam1_extrinsic: bool,
}

/// Run (a): everything free but the gauge, the default fixes (`k3`; `p1`,
/// `p2`, `k3` under the tilted sensor) and camera 1's `tilt_y`.
fn fixes_a() -> Fixes {
    let mut tilt = CameraFix::default();
    tilt.sensor = Some(vec!["tilt_y".into()]);
    Fixes {
        cameras: [CameraFix::default(), tilt],
        cam1_extrinsic: false,
    }
}

struct Built {
    problem: Problem,
    cameras: [CameraId; 2],
    cam1_extrinsic: BlockId,
    poses: Vec<BlockId>,
    planes: [PlaneId; 2],
}

/// The problem: camera 0 is the rig frame (a board pose reaches it
/// directly), camera 1 through its `cam_se3_rig`; labels as the workflows'
/// so the covariance builds a [`CalibratedRig`].
fn build(
    seed: &Scene,
    obs: &Observations,
    fixes: &Fixes,
    residual: LaserlineResidualType,
) -> Built {
    let mut problem = Problem::new();
    let cameras = [0, 1].map(|i| {
        problem
            .add_camera(&seed.cameras[i], &fixes.cameras[i])
            .unwrap()
    });
    let cam1_extrinsic = problem
        .add_pose("cam1.cam_se3_rig", &seed.cam_se3_rig[1])
        .unwrap();
    if fixes.cam1_extrinsic {
        problem.fix(cam1_extrinsic).unwrap();
    }
    let planes = [0, 1].map(|i| {
        problem
            .add_plane(format!("cam{i}.plane"), &seed.planes[i])
            .unwrap()
    });
    let mut poses = Vec::new();
    for k in 0..CAPTURES {
        let pose = problem
            .add_pose(
                format!("capture{k}.rig_se3_target"),
                &seed.rig_se3_target[k],
            )
            .unwrap();
        poses.push(pose);
        for cam in 0..2 {
            let chain = if cam == 0 {
                PoseChain::Pose(pose)
            } else {
                PoseChain::Rig {
                    cam_se3_rig: cam1_extrinsic,
                    rig_se3_target: pose,
                }
            };
            for (point, pixel) in obs.corners[k][cam].iter().flatten() {
                problem
                    .add_reprojection(
                        cameras[cam],
                        chain,
                        TargetPoint::Known(*point),
                        *pixel,
                        PixelNoise::Estimated,
                        RobustLoss::None,
                    )
                    .unwrap();
            }
            for pixel in obs.laser[k][cam].iter().flatten() {
                problem
                    .add_laser(
                        cameras[cam],
                        chain,
                        planes[cam],
                        *pixel,
                        residual,
                        LaserNoise::Estimated,
                        RobustLoss::None,
                    )
                    .unwrap();
            }
        }
    }
    Built {
        problem,
        cameras,
        cam1_extrinsic,
        poses,
        planes,
    }
}

/// A solved scene read back from `solution`.
fn solved(built: &Built, solution: &Solution) -> Scene {
    Scene {
        cameras: built.cameras.map(|c| solution.camera(c).unwrap()),
        cam_se3_rig: [
            Iso3::identity(),
            solution.pose(built.cam1_extrinsic).unwrap(),
        ],
        rig_se3_target: built
            .poses
            .iter()
            .map(|&p| solution.pose(p).unwrap())
            .collect(),
        planes: built.planes.map(|p| solution.plane(p).unwrap()),
    }
}

/// Rotation (degrees) and translation (mm) between two poses.
fn pose_error(a: &Iso3, b: &Iso3) -> (f64, f64) {
    let d = a.inverse() * b;
    (
        d.rotation.angle().to_degrees(),
        d.translation.vector.norm() * 1e3,
    )
}

/// Plane normal angle (degrees) and distance difference (mm).
fn plane_error(a: &LaserPlane, b: &LaserPlane) -> (f64, f64) {
    let angle = a.normal().dot(&b.normal()).clamp(-1.0, 1.0).acos();
    (
        angle.to_degrees(),
        (a.distance() - b.distance()).abs() * 1e3,
    )
}

/// The worst errors of `scene` against `truth`.
#[derive(Debug)]
struct Errors {
    pose_deg: f64,
    pose_mm: f64,
    extrinsic_deg: f64,
    extrinsic_mm: f64,
    plane_deg: f64,
    plane_mm: f64,
    /// Relative focal-length error.
    focal: f64,
    tilt_x_deg: f64,
}

fn errors(truth: &Scene, scene: &Scene) -> Errors {
    let poses: Vec<(f64, f64)> = truth
        .rig_se3_target
        .iter()
        .zip(&scene.rig_se3_target)
        .map(|(a, b)| pose_error(a, b))
        .collect();
    let planes: Vec<(f64, f64)> = (0..2)
        .map(|i| plane_error(&truth.planes[i], &scene.planes[i]))
        .collect();
    let (extrinsic_deg, extrinsic_mm) = pose_error(&truth.cam_se3_rig[1], &scene.cam_se3_rig[1]);
    let focal = (0..2)
        .map(|i| {
            let (t, s) = (truth.cameras[i].intrinsics, scene.cameras[i].intrinsics);
            ((t.fx - s.fx) / t.fx)
                .abs()
                .max(((t.fy - s.fy) / t.fy).abs())
        })
        .fold(0.0, f64::max);
    let tilt = |c: &CameraParams| match &c.sensor {
        SensorParams::Scheimpflug { params } => params.tilt_x,
        _ => unreachable!(),
    };
    let max = |v: &[(f64, f64)], f: fn(&(f64, f64)) -> f64| v.iter().map(f).fold(0.0, f64::max);
    Errors {
        pose_deg: max(&poses, |e| e.0),
        pose_mm: max(&poses, |e| e.1),
        extrinsic_deg,
        extrinsic_mm,
        plane_deg: max(&planes, |e| e.0),
        plane_mm: max(&planes, |e| e.1),
        focal,
        tilt_x_deg: (tilt(&truth.cameras[1]) - tilt(&scene.cameras[1]))
            .abs()
            .to_degrees(),
    }
}

/// The error of `scene` against `truth` at a covariance label: a camera
/// parameter, a pose's right-perturbation coordinate (`<pose>.rx` …
/// `.tz`) or a plane coordinate (`<plane>.nx` … `.d`).
fn labelled_error(truth: &Scene, scene: &Scene, label: &str) -> f64 {
    let pose = |t: &Iso3, e: &Iso3, c: &str| {
        let rel = t.inverse() * e;
        let w = rel.rotation.scaled_axis();
        let v = rel.translation.vector;
        let i = ["rx", "ry", "rz", "tx", "ty", "tz"]
            .iter()
            .position(|x| *x == c)
            .unwrap();
        [w.x, w.y, w.z, v.x, v.y, v.z][i]
    };
    let (head, coordinate) = label.rsplit_once('.').unwrap();
    if let Some(i) = ["cam0.plane", "cam1.plane"].iter().position(|p| *p == head) {
        let (t, e) = (&truth.planes[i], &scene.planes[i]);
        return match coordinate {
            "d" => e.distance() - t.distance(),
            "nx" => e.normal().x - t.normal().x,
            "ny" => e.normal().y - t.normal().y,
            "nz" => e.normal().z - t.normal().z,
            _ => unreachable!("{label}"),
        };
    }
    if head == "cam1.cam_se3_rig" {
        return pose(&truth.cam_se3_rig[1], &scene.cam_se3_rig[1], coordinate);
    }
    if let Some(k) = head
        .strip_prefix("capture")
        .and_then(|h| h.strip_suffix(".rig_se3_target"))
    {
        let k: usize = k.parse().unwrap();
        return pose(
            &truth.rig_se3_target[k],
            &scene.rig_se3_target[k],
            coordinate,
        );
    }
    let i = ["cam0", "cam1"].iter().position(|c| *c == head).unwrap();
    let at = |c: &CameraParams| {
        camera_parameters(c)
            .unwrap()
            .into_iter()
            .find(|(l, _)| *l == coordinate)
            .unwrap()
            .1
    };
    at(&scene.cameras[i]) - at(&truth.cameras[i])
}

/// The largest `|error| / σ` over every estimated parameter, with its label.
fn worst_z(truth: &Scene, scene: &Scene, solution: &Solution) -> (String, f64) {
    let covariance = solution.uncertainty.covariance().unwrap();
    covariance
        .labels
        .iter()
        .enumerate()
        .filter(|(_, l)| !l.ends_with(".nx") && !l.ends_with(".nz"))
        .map(|(i, l)| {
            let z = labelled_error(truth, scene, l).abs() / covariance.covariance[i][i].sqrt();
            (l.clone(), z)
        })
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .unwrap()
}

/// `σ̂ = 1.4826 · median |r|`, the estimator the solve reports.
fn mad_sigma(mut values: Vec<f64>) -> f64 {
    values.iter_mut().for_each(|v| *v = v.abs());
    values.sort_by(f64::total_cmp);
    let n = values.len();
    let median = if n.is_multiple_of(2) {
        0.5 * (values[n / 2 - 1] + values[n / 2])
    } else {
        values[n / 2]
    };
    1.482_602_218_505_602 * median
}

/// The scalar reprojection residuals (pixels) and point-to-plane laser
/// distances (metres) of `scene`, recomputed through the core camera API
/// alone: projection for the corners; back-projection, the ray's meeting
/// with the board and the plane equation for the laser pixels.
fn independent_residuals(scene: &Scene, obs: &Observations) -> (Vec<f64>, Vec<f64>) {
    let models = scene.cameras.clone().map(|c| c.build().unwrap());
    let (mut reprojection, mut laser) = (Vec::new(), Vec::new());
    for k in 0..CAPTURES {
        for cam in 0..2 {
            let cam_se3_target = scene.cam_se3_rig[cam] * scene.rig_se3_target[k];
            for (point, pixel) in obs.corners[k][cam].iter().flatten() {
                let uv = models[cam]
                    .project_point(&cam_se3_target.transform_point(point))
                    .unwrap();
                reprojection.extend([pixel.x - uv.x, pixel.y - uv.y]);
            }
            for pixel in obs.laser[k][cam].iter().flatten() {
                let ray = models[cam].backproject_pixel(pixel);
                let origin = cam_se3_target.inverse_transform_point(&ray.origin.into());
                let direction = cam_se3_target.inverse_transform_vector(&ray.direction);
                let on_board = origin + direction * (-origin.z / direction.z);
                let p = cam_se3_target.transform_point(&on_board);
                laser.push(scene.planes[cam].point_distance(&p));
            }
        }
    }
    (reprojection, laser)
}

fn noise_of(solution: &Solution, kind: ResidualKind) -> f64 {
    solution
        .noise
        .iter()
        .find(|n| n.kind == kind)
        .unwrap()
        .level
        .value()
}

fn options(backend: SolverBackend) -> SolveOptions {
    let mut options = SolveOptions::default();
    options.backend = backend;
    options.max_iters = 200;
    options
}

/// Every free block's error improves on its seed.
fn assert_improved(before: &Errors, after: &Errors, extrinsic: bool) {
    assert!(after.pose_deg < before.pose_deg && after.pose_mm < before.pose_mm);
    if extrinsic {
        assert!(after.extrinsic_deg < before.extrinsic_deg);
        assert!(after.extrinsic_mm < before.extrinsic_mm);
    }
    assert!(after.plane_deg < before.plane_deg && after.plane_mm < before.plane_mm);
    assert!(after.focal < before.focal && after.tilt_x_deg < before.tilt_x_deg);
}

/// The largest difference, in σ, between two solutions of one problem.
fn worst_difference(a: &Scene, b: &Scene, solution: &Solution) -> (String, f64) {
    worst_z(a, b, solution)
}

/// The solve from the noisy seeds reaches the solve from the truth (the
/// minimum the data define) to a small fraction of σ, the error against the
/// truth is within the stated σ, and every free block improves on its seed.
fn assert_converged(
    truth: &Scene,
    seed: &Scene,
    obs: &Observations,
    fixes: &Fixes,
    residual: LaserlineResidualType,
    backend: SolverBackend,
) -> (Built, Solution) {
    let built = build(seed, obs, fixes, residual);
    let solution = solve(&built.problem, &options(backend)).unwrap();
    assert!(solution.report.num_iters < 200, "{backend:?} converged");
    let scene = solved(&built, &solution);
    let reference = {
        let mut start = truth.clone();
        // The held parameters at their seeds, as in the solve from the seeds.
        start.cameras[1] = held_at(&truth.cameras[1], &seed.cameras[1]);
        if fixes.cam1_extrinsic {
            start.cam_se3_rig = seed.cam_se3_rig;
        }
        let built = build(&start, obs, fixes, residual);
        solved(&built, &solve(&built.problem, &options(backend)).unwrap())
    };
    let before = errors(truth, seed);
    let after = errors(truth, &scene);
    let (label, difference) = worst_difference(&reference, &scene, &solution);
    let (worst, z) = worst_z(truth, &scene, &solution);
    eprintln!("{residual:?} {backend:?}: before {before:?}");
    eprintln!("{residual:?} {backend:?}: after  {after:?}");
    eprintln!(
        "{residual:?} {backend:?}: from the truth within {difference:.3} σ ({label}); worst error \
         {z:.2} σ ({worst})"
    );
    assert!(
        difference < 0.2,
        "{label}: {difference:.3} σ from the minimum"
    );
    assert!(z < 4.5, "{worst}: {z:.2} σ from the truth");
    assert_improved(&before, &after, !fixes.cam1_extrinsic);
    (built, solution)
}

/// `truth` with camera 1's `tilt_y` at `seed`'s, which run (a) holds.
fn held_at(truth: &CameraParams, seed: &CameraParams) -> CameraParams {
    let mut camera = truth.clone();
    if let (SensorParams::Scheimpflug { params: t }, SensorParams::Scheimpflug { params: s }) =
        (&mut camera.sensor, &seed.sensor)
    {
        t.tilt_y = s.tilt_y;
    }
    camera
}

#[test]
fn a_robotless_rig_and_its_laser_planes_converge_from_noisy_seeds() {
    let truth = truth();
    let obs = observe(&truth, 1);
    let seed = seeds(&truth);
    let mut results = Vec::new();
    for backend in [SolverBackend::TinySolver, SolverBackend::Factrs] {
        let (built, solution) = assert_converged(
            &truth,
            &seed,
            &obs,
            &fixes_a(),
            LaserlineResidualType::PointToPlane,
            backend,
        );
        let scene = solved(&built, &solution);

        // Held parameters stay at their seeds, bit for bit: camera 1's
        // tilt_y, and the default fixes (k3; p1, p2 under the tilted
        // sensor).
        let held = |c: &CameraParams| {
            let DistortionParams::BrownConrady5 { params } = &c.distortion else {
                unreachable!()
            };
            let tilt_y = match &c.sensor {
                SensorParams::Scheimpflug { params } => params.tilt_y,
                _ => 0.0,
            };
            let tilted = matches!(c.sensor, SensorParams::Scheimpflug { .. });
            let p = if tilted {
                [params.p1, params.p2]
            } else {
                [0.0, 0.0]
            };
            [params.k3, p[0], p[1], tilt_y]
        };
        for i in 0..2 {
            assert_eq!(held(&scene.cameras[i]), held(&seed.cameras[i]));
        }

        // The residuals, recomputed independently, give the noise levels
        // the solve reports, and those match the injected noise.
        let (reprojection, laser) = independent_residuals(&scene, &obs);
        let reprojection_sigma = mad_sigma(reprojection.clone());
        let laser_sigma = mad_sigma(laser.clone());
        let rms = |v: &[f64]| (v.iter().map(|x| x * x).sum::<f64>() / v.len() as f64).sqrt();
        eprintln!(
            "{backend:?}: reprojection σ̂ {reprojection_sigma:.4} px (rms {:.4}, {} values), \
             laser σ̂ {:.4} mm (rms {:.4} mm, {} values)",
            rms(&reprojection),
            reprojection.len(),
            laser_sigma * 1e3,
            rms(&laser) * 1e3,
            laser.len()
        );
        let reported = noise_of(&solution, ResidualKind::Reprojection);
        assert!(
            (reported - reprojection_sigma).abs() < 1e-9 * reported,
            "{reported} vs {reprojection_sigma}"
        );
        let reported = noise_of(&solution, ResidualKind::LaserPlane);
        assert!(
            (reported - laser_sigma).abs() < 1e-9 * reported,
            "{reported} vs {laser_sigma}"
        );
        // Uniform noise in ±a has median |r| = a/2.
        let injected = 1.482_602_218_505_602 * CORNER_NOISE / 2.0;
        assert!((reprojection_sigma / injected - 1.0).abs() < 0.1);

        // The covariance labels the planes and the extrinsic as the
        // workflows do, so it builds a calibrated rig with laser planes.
        let covariance = solution.uncertainty.covariance().unwrap();
        for label in [
            "cam1.plane.nx",
            "cam1.plane.d",
            "cam1.cam_se3_rig.rx",
            "cam0.fx",
            "cam1.tilt_x",
        ] {
            assert!(covariance.index(label).is_some(), "{label}");
        }
        assert!(covariance.index("cam1.tilt_y").is_none());
        CalibratedRig::new(
            scene.cameras.to_vec(),
            scene.cam_se3_rig.to_vec(),
            solution.uncertainty.clone(),
        )
        .unwrap()
        .with_laser_planes(scene.planes.iter().cloned().map(Some).collect())
        .unwrap();
        results.push((scene, solution));
    }
    // Both backends find the same solution.
    let (tiny, solution) = &results[0];
    let factrs = &results[1].0;
    let poses = tiny
        .rig_se3_target
        .iter()
        .zip(&factrs.rig_se3_target)
        .chain([(&tiny.cam_se3_rig[1], &factrs.cam_se3_rig[1])])
        .map(|(a, b)| {
            let (deg, mm) = pose_error(a, b);
            deg.to_radians().max(mm * 1e-3)
        })
        .fold(0.0, f64::max);
    let planes = (0..2)
        .map(|i| {
            let (a, b) = (&tiny.planes[i], &factrs.planes[i]);
            (a.normal().into_inner() - b.normal().into_inner())
                .norm()
                .max((a.distance() - b.distance()).abs())
        })
        .fold(0.0, f64::max);
    let cameras = (0..2)
        .map(|i| {
            let a = camera_parameters(&tiny.cameras[i]).unwrap();
            let b = camera_parameters(&factrs.cameras[i]).unwrap();
            a.iter()
                .zip(&b)
                .map(|((_, x), (_, y))| (x - y).abs() / x.abs().max(1.0))
                .fold(0.0, f64::max)
        })
        .fold(0.0, f64::max);
    let (label, sigma) = worst_difference(tiny, factrs, solution);
    eprintln!(
        "tiny vs factrs: poses {poses:.2e} (rad / m), planes {planes:.2e}, cameras {cameras:.2e} \
         (relative), worst {sigma:.2e} σ ({label})"
    );
    assert!(sigma < 0.01, "{label}: the backends differ by {sigma} σ");
}

#[test]
fn fixed_extrinsics_and_intrinsics_stay_fixed() {
    let truth = truth();
    let obs = observe(&truth, 2);
    let mut seed = seeds(&truth);
    // Known extrinsics and a known camera 0.
    seed.cam_se3_rig = truth.cam_se3_rig;
    seed.cameras[0] = truth.cameras[0].clone();
    let mut fixes = fixes_a();
    fixes.cameras[0] = CameraFix::all(seed.cameras[0].spec().unwrap());
    fixes.cam1_extrinsic = true;
    for backend in [SolverBackend::TinySolver, SolverBackend::Factrs] {
        let (built, solution) = assert_converged(
            &truth,
            &seed,
            &obs,
            &fixes,
            LaserlineResidualType::PointToPlane,
            backend,
        );
        let scene = solved(&built, &solution);
        // The extrinsic block holds its seed's values, bit for bit.
        let q = seed.cam_se3_rig[1].rotation.into_inner();
        let t = seed.cam_se3_rig[1].translation.vector;
        assert_eq!(
            solution.value(built.cam1_extrinsic).unwrap(),
            [q.i, q.j, q.k, q.w, t.x, t.y, t.z]
        );
        assert_eq!(
            camera_parameters(&scene.cameras[0]).unwrap(),
            camera_parameters(&seed.cameras[0]).unwrap()
        );
        let covariance = solution.uncertainty.covariance().unwrap();
        assert!(covariance.index("cam1.cam_se3_rig.rx").is_none());
        assert!(covariance.index("cam0.fx").is_none());
    }
}

#[test]
fn the_line_distance_residual_converges_too() {
    let truth = truth();
    let obs = observe(&truth, 3);
    let (_, solution) = assert_converged(
        &truth,
        &seeds(&truth),
        &obs,
        &fixes_a(),
        LaserlineResidualType::LineDistNormalized,
        SolverBackend::TinySolver,
    );
    assert!(
        solution
            .noise
            .iter()
            .any(|n| n.kind == ResidualKind::LaserLine)
    );
}

#[test]
fn a_camera_that_shares_no_capture_is_rank_deficient() {
    let truth = truth();
    let mut obs = observe(&truth, 4);
    // Camera 1 keeps captures 7–9 only, which camera 0 never sees.
    for k in 0..7 {
        obs.corners[k][1] = None;
        obs.laser[k][1] = None;
    }
    let built = build(
        &seeds(&truth),
        &obs,
        &fixes_a(),
        LaserlineResidualType::PointToPlane,
    );
    let solution = solve(&built.problem, &options(SolverBackend::TinySolver)).unwrap();
    assert!(
        matches!(
            solution.uncertainty,
            Uncertainty::Unavailable {
                reason: UncertaintyUnavailable::RankDeficient { .. }
            }
        ),
        "{:?}",
        solution.uncertainty
    );
}

/// The σ the solve states against the scatter of the estimates over seeded
/// noise realizations, for every estimated parameter but the planes'
/// `nx`/`nz` (a unit normal has two degrees of freedom; `ny` stands for
/// its direction): `(label, stated σ / empirical σ)` sorted by label, with
/// the mean NEES over the labels and its χ² dimension.
fn monte_carlo(residual: LaserlineResidualType, runs: u64) -> (Vec<(String, f64)>, f64, usize) {
    let truth = truth();
    let mut labels: Vec<String> = Vec::new();
    let mut errors = Vec::new();
    let mut predicted: Option<DMatrix<f64>> = None;
    let mut nees = 0.0;
    for run in 0..runs {
        let obs = observe(&truth, 1_000 + run);
        let built = build(&truth, &obs, &fixes_a(), residual);
        let solution = solve(&built.problem, &options(SolverBackend::TinySolver)).unwrap();
        let scene = solved(&built, &solution);
        let covariance = solution.uncertainty.covariance().unwrap();
        if labels.is_empty() {
            labels = covariance
                .labels
                .iter()
                .filter(|l| !l.ends_with(".nx") && !l.ends_with(".nz"))
                .cloned()
                .collect();
        }
        let n = labels.len();
        let at: Vec<usize> = labels
            .iter()
            .map(|l| covariance.index(l).unwrap())
            .collect();
        let sigma = DMatrix::from_fn(n, n, |i, j| covariance.covariance[at[i]][at[j]]);
        let e = nalgebra::DVector::from_iterator(
            n,
            labels.iter().map(|l| labelled_error(&truth, &scene, l)),
        );
        // On the correlation matrix, whose conditioning does not depend on
        // the parameters' units.
        let scale = sigma.diagonal().map(|v| 1.0 / v.sqrt());
        let correlation = DMatrix::from_fn(n, n, |i, j| sigma[(i, j)] * scale[i] * scale[j]);
        let normalized = e.component_mul(&scale);
        nees += (normalized.transpose() * correlation.try_inverse().unwrap() * &normalized)[(0, 0)];
        predicted = Some(match predicted.take() {
            Some(acc) => acc + sigma,
            None => sigma,
        });
        errors.push(e);
    }
    let runs = runs as f64;
    let n = labels.len();
    let predicted = predicted.unwrap();
    let mean = errors
        .iter()
        .fold(nalgebra::DVector::zeros(n), |a, e| a + e)
        / runs;
    let ratios = labels
        .iter()
        .enumerate()
        .map(|(i, l)| {
            let empirical = (errors.iter().map(|e| (e[i] - mean[i]).powi(2)).sum::<f64>()
                / (runs - 1.0))
                .sqrt();
            (l.clone(), (predicted[(i, i)] / runs).sqrt() / empirical)
        })
        .collect();
    (ratios, nees / runs, n)
}

/// The stated σ of every parameter (cameras, the extrinsic, every board
/// pose, the planes) matches its Monte-Carlo scatter, for the metric
/// point-to-plane residual and the line distance in pixels, and the mean
/// NEES matches χ².
#[test]
#[ignore = "Monte-Carlo gate; run in release with --ignored"]
fn rig_laser_covariance_matches_monte_carlo() {
    const RUNS: u64 = 400;
    for residual in [
        LaserlineResidualType::PointToPlane,
        LaserlineResidualType::LineDistNormalized,
    ] {
        let (ratios, nees, k) = monte_carlo(residual, RUNS);
        let (min, max) = ratios
            .iter()
            .fold((f64::MAX, f64::MIN), |(lo, hi), (_, r)| {
                (lo.min(*r), hi.max(*r))
            });
        eprintln!("{residual:?}: {k} parameters, σ ratio {min:.3}–{max:.3}, NEES {nees:.2}");
        for (label, ratio) in ratios
            .iter()
            .filter(|(l, _)| l.contains("plane") || l.starts_with("cam1.c") || l.contains("tilt"))
        {
            eprintln!("  {label}: {ratio:.3}");
        }
        for (label, ratio) in &ratios {
            assert!(
                (ratio - 1.0).abs() < 0.12,
                "{residual:?} {label}: σ ratio {ratio:.3}"
            );
        }
        let band = (3.0 * (2.0 * k as f64 / RUNS as f64).sqrt()).max(0.12 * k as f64);
        assert!(
            (nees - k as f64).abs() < band,
            "{residual:?}: NEES {nees:.2}, χ² {k} ± {band:.2}"
        );
    }
}
