//! A camera with a residual field, held fixed, through every builder on both
//! solver backends.
//!
//! The data are noise-free projections of a tilted, distorted camera with a
//! smooth field of a few pixels. Each solve starts from perturbed
//! intrinsics, poses or planes with the true field held: it must reach the
//! noise floor, recover what it calibrates, return the field unchanged and
//! list it among the fixed parameters. The same data with the field
//! stripped from the start cannot reach the floor, so the field is honoured
//! rather than ignored. A camera with a zero field solves bit for bit like
//! the plain camera.

use nalgebra::{Isometry3, Rotation3, Translation3, Vector3};
use vision_calibration_core::{
    BSplineField, BrownConrady5, CameraFix, CameraModel, CameraParams, CorrespondenceView,
    DistortionParams, FieldParams, FxFyCxCy, Iso3, LaserPlane, PlanarDataset, Pt2, Pt3, Real,
    RigView, RigViewObs, ScheimpflugParams, SensorParams, Uncertainty, View,
};
use vision_calibration_optim::problem::{
    PixelNoise, PoseChain, Problem, SolveOptions, TargetPoint,
};
use vision_calibration_optim::{
    BackendSolveOptions, HandeyeDataset, HandeyeMode, HandeyeParams, HandeyeSolveOptions,
    LaserlineMeta, LaserlineParams, LaserlineResidualType, LaserlineSolveOptions,
    PlanarIntrinsicsParams, PlanarIntrinsicsSolveOptions, RigHandeyeLaserlineDataset,
    RigHandeyeLaserlineParams, RigHandeyeLaserlineSolveOptions, RigLaserlineDataset,
    RigLaserlineSolveOptions, RigLaserlineUpstream, RigLaserlineView, RobotPoseMeta, RobustLoss,
    SolverBackend, optimize_handeye, optimize_laserline, optimize_planar_intrinsics,
    optimize_rig_handeye_laserline, optimize_rig_laserline,
};

const BACKENDS: [SolverBackend; 2] = [SolverBackend::TinySolver, SolverBackend::Factrs];
const IMAGE: [u32; 2] = [1280, 720];

fn truth_k() -> FxFyCxCy<Real> {
    FxFyCxCy {
        fx: 800.0,
        fy: 790.0,
        cx: 640.0,
        cy: 360.0,
    }
}

fn perturbed_k() -> FxFyCxCy<Real> {
    FxFyCxCy {
        fx: 808.0,
        fy: 784.0,
        cx: 644.0,
        cy: 357.0,
    }
}

/// A smooth field of up to 3 px over the image, 6 × 4 cells.
fn field() -> FieldParams {
    let cells = [6, 4];
    let mut f = BSplineField::zeros(cells, IMAGE).unwrap();
    let nx = cells[0] as usize + 3;
    for (k, (du, dv)) in f.du.iter_mut().zip(f.dv.iter_mut()).enumerate() {
        let (i, j) = ((k % nx) as Real, (k / nx) as Real);
        *du = 3.0 * (0.6 * i + 0.4 * j).sin();
        *dv = 2.0 * (0.5 * i - 0.7 * j).cos();
    }
    FieldParams::BSpline { params: f }
}

/// A tilted Brown-Conrady camera with intrinsics `k` and field `field`.
fn camera(k: FxFyCxCy<Real>, field: FieldParams) -> CameraParams {
    CameraParams::new(
        vision_calibration_core::ProjectionParams::Pinhole,
        DistortionParams::BrownConrady5 {
            params: BrownConrady5 {
                k1: -0.12,
                k2: 0.03,
                k3: 0.0,
                p1: 0.0,
                p2: 0.0,
                iters: 20,
            },
        },
        SensorParams::Scheimpflug {
            params: ScheimpflugParams {
                tilt_x: 0.04,
                tilt_y: -0.03,
            },
        },
        k,
    )
    .with_field(field)
}

fn board_poses() -> Vec<Iso3> {
    [
        (0.0, 0.0),
        (0.25, 0.0),
        (-0.25, 0.0),
        (0.0, 0.25),
        (0.0, -0.25),
        (0.18, 0.18),
        (-0.18, 0.18),
        (0.18, -0.18),
    ]
    .iter()
    .enumerate()
    .map(|(i, &(rx, ry))| {
        Isometry3::from_parts(
            Translation3::new(-0.12, -0.09, 0.6 + 0.03 * (i % 3) as Real),
            Rotation3::from_euler_angles(rx, ry, 0.05 * i as Real).into(),
        )
    })
    .collect()
}

fn board_points() -> Vec<Pt3> {
    (0..7)
        .flat_map(|y| (0..9).map(move |x| Pt3::new(x as Real * 0.03, y as Real * 0.03, 0.0)))
        .collect()
}

fn observe(model: &CameraModel, cam_se3_target: &Iso3) -> CorrespondenceView {
    let (mut p3, mut p2) = (Vec::new(), Vec::new());
    for pw in board_points() {
        if let Some(px) = model.project_point(&cam_se3_target.transform_point(&pw)) {
            p3.push(pw);
            p2.push(px);
        }
    }
    CorrespondenceView::new(p3, p2).unwrap()
}

fn laser_plane() -> LaserPlane {
    let normal = Vector3::new(0.1, 0.9, 0.3).normalize();
    let centre = board_poses()[0].transform_point(&Pt3::new(0.12, 0.09, 0.0));
    LaserPlane::new(normal, -normal.dot(&centre.coords)).unwrap()
}

fn perturbed_plane(plane: &LaserPlane) -> LaserPlane {
    LaserPlane::new(
        (plane.normal().into_inner() + Vector3::new(0.02, -0.01, 0.01)).normalize(),
        plane.distance() + 0.005,
    )
    .unwrap()
}

fn laser_pixels(model: &CameraModel, cam_se3_target: &Iso3, plane: &LaserPlane) -> Vec<Pt2> {
    let n_t = cam_se3_target.rotation.inverse() * plane.normal().into_inner();
    let d_t = plane.distance() + plane.normal().dot(&cam_se3_target.translation.vector);
    (0..41)
        .filter_map(|i| {
            let x = i as Real * 0.006;
            let y = -(n_t.x * x + d_t) / n_t.y;
            if !(0.0..=0.18).contains(&y) {
                return None;
            }
            model.project_point(&cam_se3_target.transform_point(&Pt3::new(x, y, 0.0)))
        })
        .collect()
}

fn tight(backend: SolverBackend) -> BackendSolveOptions {
    BackendSolveOptions {
        backend,
        max_iters: 200,
        min_abs_decrease: Some(1e-16),
        min_rel_decrease: Some(1e-16),
        min_error: Some(1e-20),
        ..Default::default()
    }
}

fn plane_errors(a: &LaserPlane, b: &LaserPlane) -> (Real, Real) {
    let angle = a.normal().dot(&b.normal()).abs().min(1.0).acos();
    (angle, (a.distance() - b.distance()).abs())
}

fn planar_dataset(truth: &CameraParams) -> PlanarDataset {
    let model = truth.build().unwrap();
    PlanarDataset::new(
        board_poses()
            .iter()
            .map(|pose| View::without_meta(observe(&model, pose)))
            .collect(),
    )
    .unwrap()
}

fn assert_intrinsics(got: &FxFyCxCy<Real>, label: &str) {
    let k = truth_k();
    for (name, a, b) in [
        ("fx", got.fx, k.fx),
        ("fy", got.fy, k.fy),
        ("cx", got.cx, k.cx),
        ("cy", got.cy, k.cy),
    ] {
        // A tilt couples to the principal point, so the noise-free optimum
        // is reached to about 1e-5 px at the default tolerances.
        assert!((a - b).abs() < 1e-3, "{label}: {name} {a} vs {b}");
    }
}

fn fixed_lists_the_field(uncertainty: &Uncertainty, label: &str) {
    let Uncertainty::Available(cov) = uncertainty else {
        panic!("{label}: no covariance");
    };
    assert!(
        cov.fixed.iter().any(|l| l == "field"),
        "{label}: fixed {:?}",
        cov.fixed
    );
    assert!(
        !cov.labels.iter().any(|l| l.contains("du_")),
        "{label}: the held field is in the covariance"
    );
}

#[test]
fn planar_intrinsics_holds_the_field_on_both_backends() {
    let truth = camera(truth_k(), field());
    let dataset = planar_dataset(&truth);
    for backend in BACKENDS {
        let label = format!("{backend:?}");
        let initial =
            PlanarIntrinsicsParams::new(camera(perturbed_k(), field()), board_poses()).unwrap();
        let result = optimize_planar_intrinsics(
            &dataset,
            &initial,
            PlanarIntrinsicsSolveOptions::default(),
            tight(backend),
        )
        .unwrap_or_else(|e| panic!("{label}: {e}"));
        assert!(
            result.mean_reproj_error < 1e-6,
            "{label}: reprojection {} px",
            result.mean_reproj_error
        );
        assert_intrinsics(&result.params.camera.intrinsics, &label);
        assert_eq!(result.params.camera.field, truth.field, "{label}");
        fixed_lists_the_field(&result.uncertainty, &label);

        // Without the field the data cannot be fitted.
        let stripped =
            PlanarIntrinsicsParams::new(camera(perturbed_k(), FieldParams::None), board_poses())
                .unwrap();
        let plain = optimize_planar_intrinsics(
            &dataset,
            &stripped,
            PlanarIntrinsicsSolveOptions::default(),
            tight(backend),
        )
        .unwrap();
        assert!(
            plain.mean_reproj_error > 0.05,
            "{label}: the field's data fit a plain camera to {} px",
            plain.mean_reproj_error
        );
    }
}

#[test]
fn a_zero_field_solves_bit_for_bit_like_the_plain_camera() {
    let truth = camera(truth_k(), field());
    let dataset = planar_dataset(&truth);
    let zero = FieldParams::BSpline {
        params: BSplineField::zeros([6, 4], IMAGE).unwrap(),
    };
    for backend in BACKENDS {
        let solve = |field: FieldParams| {
            let initial =
                PlanarIntrinsicsParams::new(camera(perturbed_k(), field), board_poses()).unwrap();
            optimize_planar_intrinsics(
                &dataset,
                &initial,
                PlanarIntrinsicsSolveOptions::default(),
                tight(backend),
            )
            .unwrap()
        };
        let (a, b) = (solve(FieldParams::None), solve(zero.clone()));
        let bits = |p: &PlanarIntrinsicsParams| -> Vec<u64> {
            let blocks = p.camera.blocks();
            blocks[..4]
                .iter()
                .flatten()
                .chain(p.cam_se3_target.iter().flat_map(|pose| {
                    pose.translation
                        .vector
                        .iter()
                        .chain(pose.rotation.coords.iter())
                }))
                .map(|v| v.to_bits())
                .collect()
        };
        assert_eq!(bits(&a.params), bits(&b.params), "{backend:?}");
        assert_eq!(
            a.report.final_cost.to_bits(),
            b.report.final_cost.to_bits(),
            "{backend:?}"
        );
    }
}

#[test]
fn handeye_holds_the_field_on_both_backends() {
    let gripper_se3_cam = Isometry3::from_parts(
        Translation3::new(0.05, -0.03, 0.12),
        Rotation3::from_euler_angles(0.1, -0.05, 0.2).into(),
    );
    let base_se3_target = Isometry3::from_parts(
        Translation3::new(0.4, 0.1, -0.2),
        Rotation3::from_euler_angles(0.0, 0.0, 0.3).into(),
    );
    let truth = camera(truth_k(), field());
    let model = truth.build().unwrap();
    let views: Vec<RigView<RobotPoseMeta>> = board_poses()
        .iter()
        .map(|cam_se3_target| RigView {
            obs: RigViewObs {
                cameras: vec![Some(observe(&model, cam_se3_target))],
            },
            meta: RobotPoseMeta {
                base_se3_gripper: base_se3_target
                    * cam_se3_target.inverse()
                    * gripper_se3_cam.inverse(),
            },
        })
        .collect();
    for backend in BACKENDS {
        let dataset = HandeyeDataset::new(views.clone(), 1, HandeyeMode::EyeInHand).unwrap();
        let initial = HandeyeParams::new(
            vec![camera(perturbed_k(), field())],
            vec![Iso3::identity()],
            Isometry3::from_parts(
                Translation3::new(0.052, -0.028, 0.118),
                Rotation3::from_euler_angles(0.104, -0.048, 0.197).into(),
            ),
            vec![base_se3_target],
        );
        let opts = HandeyeSolveOptions {
            fix_extrinsics: vec![true],
            ..Default::default()
        };
        let result = optimize_handeye(dataset, initial, opts, tight(backend))
            .unwrap_or_else(|e| panic!("{backend:?}: {e}"));
        assert!(
            result.mean_reproj_error < 1e-6,
            "{backend:?}: reprojection {} px",
            result.mean_reproj_error
        );
        assert_eq!(result.params.cameras[0].field, truth.field);
        let dt =
            (result.params.handeye.translation.vector - gripper_se3_cam.translation.vector).norm();
        assert!(dt < 1e-6, "{backend:?}: hand-eye translation {dt}");
    }
}

#[test]
fn laserline_holds_the_field_on_both_backends() {
    let plane = laser_plane();
    let truth = camera(truth_k(), field());
    let model = truth.build().unwrap();
    let poses = board_poses();
    let dataset: Vec<_> = poses
        .iter()
        .map(|pose| {
            View::new(
                observe(&model, pose),
                LaserlineMeta {
                    laser_pixels: laser_pixels(&model, pose, &plane),
                    laser_weights: Vec::new(),
                },
            )
        })
        .collect();
    for backend in BACKENDS {
        for residual in [
            LaserlineResidualType::PointToPlane,
            LaserlineResidualType::LineDistNormalized,
        ] {
            let label = format!("{backend:?}/{residual:?}");
            let initial = LaserlineParams::new(
                camera(perturbed_k(), field()),
                poses.clone(),
                perturbed_plane(&plane),
            )
            .unwrap();
            let opts = LaserlineSolveOptions {
                laser_residual_type: residual,
                ..Default::default()
            };
            let result = optimize_laserline(&dataset, &initial, &opts, &tight(backend))
                .unwrap_or_else(|e| panic!("{label}: {e}"));
            assert!(
                result.report.final_cost < 1e-10,
                "{label}: final cost {}",
                result.report.final_cost
            );
            assert_eq!(result.params.camera.field, truth.field, "{label}");
            let (angle, dd) = plane_errors(&result.params.laser_plane, &plane);
            assert!(
                angle < 1e-6 && dd < 1e-7,
                "{label}: plane {angle} rad, {dd} m"
            );
        }
    }
}

#[test]
fn rig_laserline_holds_upstream_fields_on_both_backends() {
    let cameras = [
        camera(truth_k(), field()),
        camera(truth_k(), FieldParams::None),
    ];
    let models = [cameras[0].build().unwrap(), cameras[1].build().unwrap()];
    let rig_se3_cam = [
        Iso3::identity(),
        Isometry3::from_parts(
            Translation3::new(0.08, 0.0, 0.0),
            Rotation3::from_euler_angles(0.0, -0.04, 0.0).into(),
        ),
    ];
    let rig_se3_target = board_poses();
    let plane_rig = laser_plane();
    let planes_cam = [
        plane_rig.transform_by(&rig_se3_cam[0].inverse()),
        plane_rig.transform_by(&rig_se3_cam[1].inverse()),
    ];
    let views: Vec<RigLaserlineView> = rig_se3_target
        .iter()
        .map(|rig_pose| {
            let per_cam: Vec<_> = (0..2)
                .map(|c| {
                    let cam_se3_target = rig_se3_cam[c].inverse() * rig_pose;
                    (
                        Some(observe(&models[c], &cam_se3_target)),
                        Some(laser_pixels(&models[c], &cam_se3_target, &planes_cam[c])),
                    )
                })
                .collect();
            RigLaserlineView {
                cameras: per_cam.iter().map(|(obs, _)| obs.clone()).collect(),
                laser_pixels: per_cam.into_iter().map(|(_, px)| px).collect(),
            }
        })
        .collect();
    for backend in BACKENDS {
        let dataset = RigLaserlineDataset::new(views.clone(), 2).unwrap();
        let upstream = RigLaserlineUpstream {
            cameras: cameras.to_vec(),
            rig_se3_cam: rig_se3_cam.to_vec(),
            rig_se3_target: rig_se3_target.clone(),
            uncertainty: Default::default(),
        };
        let initial: Vec<_> = planes_cam.iter().map(perturbed_plane).collect();
        let result = optimize_rig_laserline(
            &dataset,
            &upstream,
            &initial,
            &RigLaserlineSolveOptions::default(),
            &tight(backend),
        )
        .unwrap_or_else(|e| panic!("{backend:?}: {e}"));
        for (c, (got, gt)) in result.laser_planes_cam.iter().zip(&planes_cam).enumerate() {
            let (angle, dd) = plane_errors(got, gt);
            assert!(
                angle < 1e-5 && dd < 1e-5,
                "{backend:?}: camera {c} plane {angle} rad, {dd} m"
            );
        }
    }
}

/// The joint rig hand-eye laser solve refines a rig of cameras that carry
/// fields, holds the fields, and lists each among the fixed parameters by
/// its camera.
#[test]
fn joint_rig_laserline_holds_the_fields_on_both_backends() {
    let cameras = [
        camera(truth_k(), field()),
        camera(truth_k(), FieldParams::None),
    ];
    let models = [cameras[0].build().unwrap(), cameras[1].build().unwrap()];
    let rig_se3_cam = [
        Iso3::identity(),
        Isometry3::from_parts(
            Translation3::new(0.08, 0.0, 0.0),
            Rotation3::from_euler_angles(0.0, -0.04, 0.0).into(),
        ),
    ];
    let rig_se3_target = board_poses();
    let plane_rig = laser_plane();
    let planes_cam = [
        plane_rig.transform_by(&rig_se3_cam[0].inverse()),
        plane_rig.transform_by(&rig_se3_cam[1].inverse()),
    ];
    let views: Vec<RigLaserlineView> = rig_se3_target
        .iter()
        .map(|rig_pose| {
            let per_cam: Vec<_> = (0..2)
                .map(|c| {
                    let cam_se3_target = rig_se3_cam[c].inverse() * rig_pose;
                    (
                        Some(observe(&models[c], &cam_se3_target)),
                        Some(laser_pixels(&models[c], &cam_se3_target, &planes_cam[c])),
                    )
                })
                .collect();
            RigLaserlineView {
                cameras: per_cam.iter().map(|(obs, _)| obs.clone()).collect(),
                laser_pixels: per_cam.into_iter().map(|(_, px)| px).collect(),
            }
        })
        .collect();
    let gripper_se3_rig = Isometry3::from_parts(
        Translation3::new(0.05, -0.03, 0.12),
        Rotation3::from_euler_angles(0.1, -0.05, 0.2).into(),
    );
    let base_se3_target = Isometry3::from_parts(
        Translation3::new(0.4, 0.1, -0.2),
        Rotation3::from_euler_angles(0.0, 0.0, 0.3).into(),
    );
    let robot_poses: Vec<Iso3> = rig_se3_target
        .iter()
        .map(|t| base_se3_target * t.inverse() * gripper_se3_rig.inverse())
        .collect();
    for backend in BACKENDS {
        let dataset = RigHandeyeLaserlineDataset::from_rig_dataset(
            RigLaserlineDataset::new(views.clone(), 2).unwrap(),
            robot_poses.clone(),
            HandeyeMode::EyeInHand,
        )
        .unwrap();
        let initial = RigHandeyeLaserlineParams::new(
            vec![camera(perturbed_k(), field()), cameras[1].clone()],
            rig_se3_cam.to_vec(),
            Isometry3::from_parts(
                Translation3::new(0.052, -0.028, 0.118),
                Rotation3::from_euler_angles(0.104, -0.048, 0.197).into(),
            ),
            base_se3_target,
            planes_cam.iter().map(perturbed_plane).collect(),
        );
        let opts = RigHandeyeLaserlineSolveOptions {
            fix_extrinsics: vec![true, true],
            fix_planes: vec![false, false],
            ..Default::default()
        };
        let result = optimize_rig_handeye_laserline(dataset, initial, opts, tight(backend))
            .unwrap_or_else(|e| panic!("{backend:?}: {e}"));
        assert!(
            result.mean_reproj_error < 1e-6,
            "{backend:?}: reprojection {} px",
            result.mean_reproj_error
        );
        assert_eq!(result.params.cameras[0].field, cameras[0].field);
        assert!(result.params.cameras[1].field.is_none());
        let (angle, dd) = plane_errors(&result.params.laser_planes_cam[0], &planes_cam[0]);
        assert!(
            angle < 1e-5 && dd < 1e-5,
            "{backend:?}: plane {angle} rad, {dd} m"
        );
        let Uncertainty::Available(cov) = &result.uncertainty else {
            panic!("{backend:?}: no covariance");
        };
        assert!(
            cov.fixed.iter().any(|l| l == "cam0.field"),
            "{backend:?}: fixed {:?}",
            cov.fixed
        );
    }
}

/// The public problem builder holds a camera's field too: a pose-only
/// solve through it recovers the true pose from the field camera's pixels.
#[test]
fn the_problem_builder_holds_the_field_on_both_backends() {
    let truth = camera(truth_k(), field());
    let model = truth.build().unwrap();
    let pose = board_poses()[3];
    let obs = observe(&model, &pose);
    for backend in BACKENDS {
        let mut problem = Problem::new();
        let cam = problem
            .add_camera(&truth, &CameraFix::all(truth.spec().unwrap()))
            .unwrap();
        let start = pose
            * Isometry3::from_parts(
                Translation3::new(0.004, -0.003, 0.006),
                Rotation3::from_euler_angles(0.01, -0.008, 0.012).into(),
            );
        let block = problem.add_pose("pose", &start).unwrap();
        for (pw, px) in obs.points_3d.iter().zip(&obs.points_2d) {
            problem
                .add_reprojection(
                    cam,
                    PoseChain::Pose(block),
                    TargetPoint::Known(*pw),
                    *px,
                    PixelNoise::Estimated,
                    RobustLoss::None,
                )
                .unwrap();
        }
        let mut options = SolveOptions::default();
        options.backend = backend;
        let solution = vision_calibration_optim::problem::solve(&problem, &options)
            .unwrap_or_else(|e| panic!("{backend:?}: {e}"));
        let got = solution.pose(block).unwrap();
        let dt = (got.translation.vector - pose.translation.vector).norm();
        assert!(dt < 1e-5, "{backend:?}: pose translation {dt}");
        assert_eq!(solution.camera(cam).unwrap().field, truth.field);
    }
}
