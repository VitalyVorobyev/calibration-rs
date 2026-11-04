//! The upstream calibration's share in a rig laser fit's covariance.
//!
//! The planes are fitted with the upstream calibration held; with its
//! covariance `Σ_u` stated, the planes' covariance gains `A Σ_u Aᵀ` (and the
//! cross-covariance `A Σ_u` with the cameras and extrinsics), with `A`
//! how the fitted planes move with the upstream. A noise-free scene of
//! diverse board poses checks `A` against the planes' finite-difference
//! motion as each kind of upstream parameter moves and the planes are
//! refitted, and the hand-eye chain's map into the upstream's coordinates
//! against the chain's own finite differences.

use nalgebra::{DMatrix, Translation3, UnitQuaternion, Vector3};
use vision_calibration_core::synthetic::laser::{BoardExtent, laser_stripe_pixels};
use vision_calibration_core::synthetic::noise::UniformPixelNoise;
use vision_calibration_core::synthetic::planar::grid_points;
use vision_calibration_core::synthetic::poses::{
    BoardPoseSpec, centered_board_poses, make_iso, rig_layout,
};
use vision_calibration_core::{
    BrownConrady5, CameraParams, CorrespondenceView, FxFyCxCy, Iso3, LaserPlane,
    ParameterCovariance, Pt2, Uncertainty, make_pinhole_camera, pinhole_camera_params,
};
use vision_calibration_optim::{
    BackendSolveOptions, HandeyeMode, RigLaserlineDataset, RigLaserlineEstimate,
    RigLaserlineSolveOptions, RigLaserlineUpstream, RigLaserlineView, SolverBackend,
    handeye_upstream_uncertainty, optimize_rig_laserline,
};

const CAMERAS: usize = 2;

fn camera() -> CameraParams {
    let k = FxFyCxCy {
        fx: 900.0,
        fy: 895.0,
        cx: 640.0,
        cy: 360.0,
    };
    let distortion = BrownConrady5 {
        k1: -0.08,
        k2: 0.02,
        k3: 0.0,
        p1: 0.0,
        p2: 0.0,
        iters: 8,
    };
    pinhole_camera_params(&make_pinhole_camera(k, distortion))
}

/// The laser plane in the rig frame: 16.7° about y, 0.5 m ahead.
fn plane_rig() -> LaserPlane {
    let n = Vector3::new(0.30, 0.0, 1.0).normalize();
    LaserPlane::new(n, -n.z * 0.5).unwrap()
}

fn board_poses() -> Vec<Iso3> {
    let spec = BoardPoseSpec::new;
    let specs = [
        spec(0.00, 0.00, 0.00, 0.000),
        spec(0.12, -0.05, 0.35, -0.012),
        spec(-0.10, 0.09, -0.40, 0.010),
        spec(0.06, 0.14, 0.60, 0.014),
        spec(-0.13, -0.07, -0.25, -0.014),
        spec(0.09, -0.15, 0.45, 0.008),
    ];
    centered_board_poses(&specs, (0.12, 0.09), 0.5)
}

/// A noise-free rig laser scene and the upstream it was made with.
fn scene() -> (RigLaserlineDataset, RigLaserlineUpstream, Vec<LaserPlane>) {
    let cameras = vec![camera(); CAMERAS];
    let cam_se3_rig = rig_layout(CAMERAS, 0.10, 0.05);
    let rig_se3_target = board_poses();
    let board = grid_points(9, 7, 0.03);
    let extent = BoardExtent::from_grid(9, 7, 0.03);
    let exact = UniformPixelNoise::default();
    let planes: Vec<LaserPlane> = cam_se3_rig
        .iter()
        .map(|e| plane_rig().transform_by(e))
        .collect();
    let views = rig_se3_target
        .iter()
        .enumerate()
        .map(|(v, target)| {
            let mut view = RigLaserlineView {
                cameras: Vec::new(),
                laser_pixels: Vec::new(),
            };
            for c in 0..CAMERAS {
                let model = cameras[c].build().unwrap();
                let pose = cam_se3_rig[c] * target;
                let pixels: Vec<Pt2> = board
                    .iter()
                    .map(|p| model.project_point(&pose.transform_point(p)).unwrap())
                    .collect();
                view.cameras.push(Some(
                    CorrespondenceView::new(board.clone(), pixels).unwrap(),
                ));
                let camera = vision_calibration_core::make_pinhole_camera(
                    cameras[c].intrinsics,
                    match cameras[c].distortion {
                        vision_calibration_core::DistortionParams::BrownConrady5 { params } => {
                            params
                        }
                        _ => unreachable!("a Brown-Conrady camera"),
                    },
                );
                let normal = planes[c].normal().into_inner();
                let stripe = laser_stripe_pixels(
                    &camera,
                    &pose,
                    &normal,
                    planes[c].distance(),
                    &extent,
                    v,
                    &exact,
                );
                assert!(
                    stripe.len() > 10,
                    "view {v} camera {c}: a stripe on the board"
                );
                view.laser_pixels.push(Some(stripe));
            }
            view
        })
        .collect();
    let upstream = RigLaserlineUpstream {
        cameras,
        cam_to_rig: cam_se3_rig.iter().map(|e| e.inverse()).collect(),
        rig_se3_target,
        uncertainty: Uncertainty::default(),
    };
    (
        RigLaserlineDataset::new(views, CAMERAS).unwrap(),
        upstream,
        planes,
    )
}

fn solve(
    dataset: &RigLaserlineDataset,
    upstream: &RigLaserlineUpstream,
    initial: &[LaserPlane],
    backend: SolverBackend,
) -> RigLaserlineEstimate {
    let options = BackendSolveOptions {
        backend,
        max_iters: 200,
        min_abs_decrease: Some(1e-30),
        min_rel_decrease: Some(1e-15),
        min_error: Some(1e-30),
        ..Default::default()
    };
    // Start off the truth, so the solve iterates to the minimum.
    let start: Vec<LaserPlane> = initial
        .iter()
        .map(|p| {
            let n = p.normal().into_inner() + Vector3::new(0.01, -0.005, 0.0);
            LaserPlane::new(n.normalize(), p.distance() + 0.003).unwrap()
        })
        .collect();
    optimize_rig_laserline(
        dataset,
        upstream,
        &start,
        &RigLaserlineSolveOptions::default(),
        &options,
    )
    .unwrap()
}

/// `pose · Exp(δ)`, to first order the right perturbation a pose's
/// covariance is over.
fn perturbed(pose: &Iso3, delta: &[f64; 6]) -> Iso3 {
    pose * Iso3::from_parts(
        Translation3::new(delta[3], delta[4], delta[5]),
        UnitQuaternion::from_scaled_axis(Vector3::new(delta[0], delta[1], delta[2])),
    )
}

fn pose_labels(name: &str) -> Vec<String> {
    ["rx", "ry", "rz", "tx", "ty", "tz"]
        .iter()
        .map(|c| format!("{name}.{c}"))
        .collect()
}

/// One upstream coordinate.
#[derive(Debug, Clone, Copy)]
enum Coordinate {
    Fx(usize),
    K1(usize),
    Extrinsic(usize, usize),
    Target(usize, usize),
}

impl Coordinate {
    /// The labels of the parameter it belongs to, and its position there.
    fn labels(self) -> (Vec<String>, usize) {
        match self {
            Self::Fx(c) => (vec![format!("cam{c}.fx")], 0),
            Self::K1(c) => (vec![format!("cam{c}.k1")], 0),
            Self::Extrinsic(c, i) => (pose_labels(&format!("cam{c}.cam_se3_rig")), i),
            Self::Target(v, i) => (pose_labels(&format!("view{v}.rig_se3_target")), i),
        }
    }

    fn step(self) -> f64 {
        match self {
            Self::Fx(_) => 1e-1,
            Self::K1(_) => 1e-3,
            Self::Extrinsic(_, i) | Self::Target(_, i) if i < 3 => 1e-4,
            _ => 1e-5,
        }
    }

    fn moved(self, upstream: &RigLaserlineUpstream, h: f64) -> RigLaserlineUpstream {
        let mut out = upstream.clone();
        let mut delta = [0.0; 6];
        match self {
            Self::Fx(c) => out.cameras[c].intrinsics.fx += h,
            Self::K1(c) => match &mut out.cameras[c].distortion {
                vision_calibration_core::DistortionParams::BrownConrady5 { params } => {
                    params.k1 += h
                }
                other => panic!("unexpected distortion {other:?}"),
            },
            Self::Extrinsic(c, i) => {
                delta[i] = h;
                // `cam_to_rig` is `cam_se3_rig⁻¹`.
                let cam_se3_rig = out.cam_to_rig[c].inverse();
                out.cam_to_rig[c] = perturbed(&cam_se3_rig, &delta).inverse();
            }
            Self::Target(v, i) => {
                delta[i] = h;
                out.rig_se3_target[v] = perturbed(&out.rig_se3_target[v], &delta);
            }
        }
        out
    }

    /// An upstream covariance over this coordinate's parameter alone:
    /// variance `sigma²` on the coordinate, none on the others.
    fn alone(self, sigma: f64) -> Uncertainty {
        let (labels, at) = self.labels();
        let n = labels.len();
        Uncertainty::Available(ParameterCovariance {
            labels,
            covariance: (0..n)
                .map(|i| {
                    (0..n)
                        .map(|j| {
                            if i == at && j == at {
                                sigma * sigma
                            } else {
                                0.0
                            }
                        })
                        .collect()
                })
                .collect(),
            views: Vec::new(),
            fixed: Vec::new(),
            noise: Vec::new(),
            num_residuals: 0,
            dof: 0,
            view_scatter: None,
        })
    }
}

/// The planes `[nx, ny, nz, d]` of every camera, in camera frames, signed
/// as `reference`'s.
fn planes(estimate: &RigLaserlineEstimate, reference: &[LaserPlane]) -> Vec<f64> {
    estimate
        .laser_planes_cam
        .iter()
        .zip(reference)
        .flat_map(|(p, r)| {
            let s = p.normal().dot(&r.normal()).signum();
            let n = p.normal().into_inner() * s;
            [n.x, n.y, n.z, s * p.distance()]
        })
        .collect()
}

fn plane_labels() -> Vec<String> {
    (0..CAMERAS)
        .flat_map(|c| ["nx", "ny", "nz", "d"].map(|p| format!("cam{c}.plane.{p}")))
        .collect()
}

/// For one upstream coordinate of variance `σ²`, the planes' covariance
/// gains `σ² a aᵀ` and their cross-covariance with a listed camera
/// parameter or extrinsic is `σ² a`, with `a` the planes' finite-difference
/// motion as the coordinate moves and the planes are refitted.
#[test]
fn the_upstream_share_is_the_planes_motion() {
    let (dataset, upstream, truth) = scene();
    let held = solve(&dataset, &upstream, &truth, SolverBackend::TinySolver);
    let held = held
        .uncertainty
        .covariance()
        .expect("the planes' own")
        .clone();
    let sigma = 0.1;
    for coordinate in [
        Coordinate::Fx(0),
        Coordinate::K1(1),
        Coordinate::Extrinsic(1, 1),
        Coordinate::Extrinsic(1, 5),
        Coordinate::Target(2, 0),
        Coordinate::Target(4, 5),
    ] {
        let h = coordinate.step();
        let solve_at = |h: f64| {
            let moved = coordinate.moved(&upstream, h);
            planes(
                &solve(&dataset, &moved, &truth, SolverBackend::TinySolver),
                &truth,
            )
        };
        let (up, down) = (solve_at(h), solve_at(-h));
        let a: Vec<f64> = up
            .iter()
            .zip(&down)
            .map(|(u, d)| (u - d) / (2.0 * h))
            .collect();
        let largest = a.iter().map(|v| v.abs()).fold(0.0, f64::max);
        assert!(largest > 1e-6, "{coordinate:?}: the planes move: {a:?}");

        let mut with = upstream.clone();
        with.uncertainty = coordinate.alone(sigma);
        let estimate = solve(&dataset, &with, &truth, SolverBackend::TinySolver);
        let cov = estimate.uncertainty.covariance().expect("covariance");
        let labels = plane_labels();
        let at = |c: &ParameterCovariance, l: &str| c.index(l).expect(l);
        let scale = largest * largest * sigma * sigma;
        let mut worst: f64 = 0.0;
        for (i, li) in labels.iter().enumerate() {
            for (j, lj) in labels.iter().enumerate() {
                let share = cov.covariance[at(cov, li)][at(cov, lj)]
                    - held.covariance[at(&held, li)][at(&held, lj)];
                worst = worst.max((share - sigma * sigma * a[i] * a[j]).abs() / scale);
            }
        }
        let (names, index) = coordinate.labels();
        let label = &names[index];
        let mut cross_worst: f64 = 0.0;
        match coordinate {
            Coordinate::Target(..) => assert!(cov.index(label).is_none(), "{label} reported"),
            _ => {
                let k = at(cov, label);
                assert_eq!(cov.covariance[k][k], sigma * sigma);
                for (i, li) in labels.iter().enumerate() {
                    let cross = cov.covariance[at(cov, li)][k];
                    cross_worst = cross_worst
                        .max((cross - sigma * sigma * a[i]).abs() / (largest * sigma * sigma));
                }
            }
        }
        eprintln!(
            "{label}: |a| {largest:.3e}, share error {worst:.1e}, cross error {cross_worst:.1e}"
        );
        // The synthetic stripe sits within 3e-4 px of the model, so the
        // Gauss-Newton sensitivity is exact to about that order.
        assert!(
            worst < 5e-3,
            "{label}: share off by {worst:.1e} of its scale"
        );
        assert!(
            cross_worst < 5e-3,
            "{label}: cross off by {cross_worst:.1e}"
        );
    }
}

/// The joint covariance is the same on both backends.
#[test]
fn both_backends_give_the_upstream_share() {
    let (dataset, mut upstream, truth) = scene();
    let mut labels: Vec<String> = vec!["cam0.fx".into(), "cam1.k1".into()];
    labels.extend(pose_labels("cam1.cam_se3_rig"));
    labels.extend(pose_labels("view3.rig_se3_target"));
    let n = labels.len();
    let variance = |l: &str| match l.rsplit('.').next().unwrap() {
        "fx" => 0.25,
        "k1" => 1e-6,
        "rx" | "ry" | "rz" => 1e-7,
        _ => 1e-8,
    };
    upstream.uncertainty = Uncertainty::Available(ParameterCovariance {
        covariance: (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| if i == j { variance(&labels[i]) } else { 0.0 })
                    .collect()
            })
            .collect(),
        labels,
        views: Vec::new(),
        fixed: Vec::new(),
        noise: Vec::new(),
        num_residuals: 0,
        dof: 0,
        view_scatter: None,
    });
    let tiny = solve(&dataset, &upstream, &truth, SolverBackend::TinySolver);
    let factrs = solve(&dataset, &upstream, &truth, SolverBackend::Factrs);
    let (a, b) = (
        tiny.uncertainty.covariance().unwrap(),
        factrs.uncertainty.covariance().unwrap(),
    );
    assert_eq!(a.labels, b.labels);
    let m = DMatrix::from_fn(a.labels.len(), a.labels.len(), |i, j| a.covariance[i][j]);
    for (ra, rb) in a.covariance.iter().zip(&b.covariance) {
        for (x, y) in ra.iter().zip(rb) {
            assert!((x - y).abs() <= 1e-6 * m.amax(), "{x:e} vs {y:e}");
        }
    }
}

/// The hand-eye chain's map into the upstream's target poses matches the
/// chain's finite differences, in both modes: a unit variance on one
/// coordinate of the hand-eye or the target's pose gives each view's
/// `rig_se3_target` the covariance `j jᵀ` of its motion `j`.
#[test]
fn the_handeye_chain_maps_into_the_target_poses() {
    let handeye = make_iso((0.1, -0.05, 0.15), (0.05, 0.03, 0.15));
    let target = make_iso((0.0, 0.2, 0.0), (0.4, -0.1, 0.2));
    let stations = [
        make_iso((0.3, 0.0, 0.1), (0.1, 0.0, 0.5)),
        make_iso((-0.2, 0.25, -0.3), (-0.05, 0.1, 0.45)),
    ];
    for mode in [HandeyeMode::EyeInHand, HandeyeMode::EyeToHand] {
        let chain = |handeye: &Iso3, target: &Iso3| -> Vec<Iso3> {
            stations
                .iter()
                .map(|b| match mode {
                    HandeyeMode::EyeInHand => handeye.inverse() * b.inverse() * target,
                    HandeyeMode::EyeToHand => handeye * b * target,
                })
                .collect()
        };
        let poses = chain(&handeye, &target);
        let (handeye_name, target_name) = match mode {
            HandeyeMode::EyeInHand => ("gripper_se3_rig", "base_se3_target"),
            HandeyeMode::EyeToHand => ("rig_se3_base", "gripper_se3_target"),
        };
        let mut labels = vec!["cam0.fx".to_string()];
        labels.extend(pose_labels(handeye_name));
        labels.extend(pose_labels(target_name));
        for k in 1..13 {
            let n = labels.len();
            let covariance = (0..n)
                .map(|i| {
                    (0..n)
                        .map(|j| if i == k && j == k { 1.0 } else { 0.0 })
                        .collect()
                })
                .collect();
            let calibration = Uncertainty::Available(ParameterCovariance {
                labels: labels.clone(),
                covariance,
                views: Vec::new(),
                fixed: vec!["cam0.cam_se3_rig".into()],
                noise: Vec::new(),
                num_residuals: 10,
                dof: 3,
                view_scatter: None,
            });
            let mapped =
                handeye_upstream_uncertainty(&calibration, mode, &handeye, 2, &poses).unwrap();
            let mapped = mapped.covariance().unwrap();
            assert_eq!(mapped.labels[0], "cam0.fx");
            assert_eq!(mapped.labels.len(), 1 + 6 * stations.len());
            assert_eq!(mapped.fixed, ["cam0.cam_se3_rig"]);

            // The chain's motion along the coordinate.
            let h = 1e-6;
            let mut delta = [0.0; 6];
            delta[(k - 1) % 6] = h;
            let moved = if k <= 6 {
                chain(&perturbed(&handeye, &delta), &target)
            } else {
                chain(&handeye, &perturbed(&target, &delta))
            };
            for (v, (pose, moved)) in poses.iter().zip(&moved).enumerate() {
                let step = pose.inverse() * moved;
                let w = step.rotation.scaled_axis() / h;
                let t = step.translation.vector / h;
                let j = [w.x, w.y, w.z, t.x, t.y, t.z];
                let at = |c: usize| {
                    mapped
                        .index(&format!(
                            "view{v}.rig_se3_target.{}",
                            ["rx", "ry", "rz", "tx", "ty", "tz"][c]
                        ))
                        .unwrap()
                };
                for a in 0..6 {
                    for b in 0..6 {
                        let got = mapped.covariance[at(a)][at(b)];
                        assert!(
                            (got - j[a] * j[b]).abs() < 1e-5,
                            "{mode:?} coordinate {k} view {v} ({a}, {b}): {got} vs {}",
                            j[a] * j[b]
                        );
                    }
                }
            }
        }
    }
}
