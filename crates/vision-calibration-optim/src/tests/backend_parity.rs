//! Both backends on one solve: the same minimum, fixed components left
//! exactly in place, bounds enforced.

use crate::backend::{self, BackendSolveOptions, SolverBackend};
use crate::ir::{
    Bound, CameraModelDesc, FactorKind, FixedMask, ManifoldKind, ProblemIR, ReprojChain,
    ResidualBlock, RobustLoss,
};
use crate::params::intrinsics::{INTRINSICS_DIM, pack_intrinsics};
use crate::params::pose_se3::iso3_to_se3_dvec;
use nalgebra::{DVector, Isometry3, Rotation3, Translation3};
use std::collections::HashMap;
use vision_calibration_core::{
    BrownConrady5, Camera, FxFyCxCySkew, IdentitySensor, Pinhole, Pt3, Real,
};

const BACKENDS: [SolverBackend; 2] = [SolverBackend::TinySolver, SolverBackend::Factrs];

/// How a [`planar_problem`]'s observations deviate from the truth.
#[derive(Clone, Copy)]
enum Data {
    /// Exact projections.
    Exact,
    /// Deterministic sub-pixel noise, and with `outliers` every 17th corner
    /// displaced by about 30 px.
    Noisy { outliers: bool },
}

/// A Brown-Conrady camera seen in five board poses; `k3` (at its true
/// value) and the first pose are fixed, and with `bounded` `fx` is boxed in
/// below its true value.
fn planar_problem(
    loss: RobustLoss,
    data: Data,
    bounded: bool,
) -> (ProblemIR, HashMap<String, DVector<f64>>, FxFyCxCySkew<Real>) {
    let k_gt = FxFyCxCySkew {
        fx: 800.0,
        fy: 780.0,
        cx: 640.0,
        cy: 360.0,
        skew: 0.0,
    };
    let dist_gt = BrownConrady5 {
        k1: -0.2,
        k2: 0.05,
        k3: 0.0,
        p1: 0.001,
        p2: -0.0008,
        iters: 8,
    };
    let camera = Camera::new(Pinhole, dist_gt, IdentitySensor, k_gt);
    let poses: Vec<Isometry3<Real>> = [
        ([0.1, -0.05, 0.2], [0.1, -0.05, 1.0]),
        ([-0.3, 0.06, -0.15], [-0.08, 0.03, 1.1]),
        ([0.25, -0.3, 0.12], [0.05, 0.08, 0.95]),
        ([-0.2, -0.25, 0.3], [0.02, -0.06, 1.05]),
        ([0.3, 0.2, -0.1], [-0.04, 0.02, 0.9]),
    ]
    .iter()
    .map(|(r, t)| {
        Isometry3::from_parts(
            Translation3::new(t[0], t[1], t[2]),
            Rotation3::from_euler_angles(r[0], r[1], r[2]).into(),
        )
    })
    .collect();

    let mut ir = ProblemIR::new();
    let mut initial = HashMap::new();
    let cam = ir.add_param_block(
        "cam",
        INTRINSICS_DIM,
        ManifoldKind::Euclidean,
        FixedMask::all_free(),
        bounded.then(|| {
            vec![Bound {
                idx: 0,
                lower: 790.0,
                upper: 799.0,
            }]
        }),
    );
    initial.insert(
        "cam".to_string(),
        pack_intrinsics(&FxFyCxCySkew {
            fx: 795.0,
            fy: 790.0,
            cx: 645.0,
            cy: 365.0,
            skew: 0.0,
        })
        .unwrap(),
    );
    let dist = ir.add_param_block(
        "dist",
        5,
        ManifoldKind::Euclidean,
        FixedMask::fix_indices(&[2]),
        None,
    );
    initial.insert(
        "dist".to_string(),
        DVector::from_row_slice(&[-0.15, 0.02, 0.0, 0.0, 0.0]),
    );

    for (v, pose) in poses.iter().enumerate() {
        let name = format!("pose/{v}");
        let fixed = if v == 0 {
            FixedMask::fix_indices(&[0, 1, 2, 3, 4, 5, 6])
        } else {
            FixedMask::all_free()
        };
        let id = ir.add_param_block(&name, 7, ManifoldKind::SE3, fixed, None);
        // The fixed pose starts at the truth; the others are perturbed.
        let start = if v == 0 {
            *pose
        } else {
            Isometry3::from_parts(
                Translation3::from(
                    pose.translation.vector + nalgebra::Vector3::new(0.004, -0.003, 0.01),
                ),
                pose.rotation * nalgebra::UnitQuaternion::from_euler_angles(0.01, -0.008, 0.005),
            )
        };
        initial.insert(name, iso3_to_se3_dvec(&start));
        let mut n = 0usize;
        for y in -4..=4 {
            for x in -5..=5 {
                let pw = Pt3::new(x as Real * 0.03, y as Real * 0.03, 0.0);
                let Some(mut px) = camera.project_point(&pose.transform_point(&pw)) else {
                    continue;
                };
                n += 1;
                if let Data::Noisy { outliers } = data {
                    let phase = n as Real * 0.7 + v as Real;
                    px.x += 0.3 * phase.sin();
                    px.y += 0.3 * (1.3 * phase).cos();
                    if outliers && n.is_multiple_of(17) {
                        px.x += 25.0;
                        px.y -= 18.0;
                    }
                }
                ir.add_residual_block(ResidualBlock {
                    params: vec![cam, dist, id],
                    loss,
                    factor: FactorKind::ReprojPoint {
                        model: CameraModelDesc::PINHOLE4_DIST5,
                        chain: ReprojChain::SinglePose,
                        pw: [pw.x, pw.y, pw.z],
                        uv: [px.x, px.y],
                        w: 1.0,
                    },
                    residual_dim: 2,
                });
            }
        }
    }
    (ir, initial, k_gt)
}

fn opts(backend: SolverBackend) -> BackendSolveOptions {
    BackendSolveOptions {
        backend,
        max_iters: 200,
        min_abs_decrease: Some(1e-14),
        min_rel_decrease: Some(1e-14),
        min_error: Some(1e-20),
        ..BackendSolveOptions::default()
    }
}

#[test]
fn fixed_components_stay_exactly_in_place_and_bounds_hold() {
    let (ir, initial, _) = planar_problem(RobustLoss::None, Data::Exact, true);
    for backend in BACKENDS {
        let solution = backend::solve(&ir, &initial, &opts(backend)).expect("solve");
        let dist = &solution.params["dist"];
        assert_eq!(dist[2], initial["dist"][2], "{backend:?}: k3 moved");
        assert_eq!(
            solution.params["pose/0"], initial["pose/0"],
            "{backend:?}: the fixed pose moved"
        );
        let fx = solution.params["cam"][0];
        assert!(
            (790.0..=799.0).contains(&fx),
            "{backend:?}: fx {fx} left its bounds"
        );
        assert_ne!(dist[0], initial["dist"][0], "{backend:?}: k1 did not move");
    }
}

/// On noisy data, with and without outliers, both backends stop at the same
/// minimum of the same robust objective.
#[test]
fn backends_reach_the_same_minimum() {
    for (loss, outliers) in [
        (RobustLoss::None, false),
        (RobustLoss::Huber { scale: 1.0 }, true),
        (RobustLoss::Cauchy { scale: 1.0 }, true),
        (RobustLoss::Arctan { scale: 4.0 }, true),
    ] {
        let (ir, initial, _) = planar_problem(loss, Data::Noisy { outliers }, false);
        let tiny = backend::solve(&ir, &initial, &opts(SolverBackend::TinySolver)).unwrap();
        let factrs = backend::solve(&ir, &initial, &opts(SolverBackend::Factrs)).unwrap();
        for (name, a) in &tiny.params {
            let b = &factrs.params[name];
            let scale = a.amax().max(1.0);
            assert!(
                (a - b).amax() <= 1e-6 * scale,
                "{loss:?}: {name} differs: {a} vs {b}"
            );
        }
        let (ct, cf) = (tiny.solve_report.final_cost, factrs.solve_report.final_cost);
        assert!(
            (ct - cf).abs() <= 1e-9 * ct,
            "{loss:?}: final cost {ct} vs {cf}"
        );
    }
}

/// Noise-free and unbounded, both backends recover the truth.
#[test]
fn backends_recover_ground_truth() {
    let (ir, initial, k_gt) = planar_problem(RobustLoss::None, Data::Exact, false);
    for backend in BACKENDS {
        let solution = backend::solve(&ir, &initial, &opts(backend)).unwrap();
        let cam = &solution.params["cam"];
        for (got, want) in cam.iter().zip([k_gt.fx, k_gt.fy, k_gt.cx, k_gt.cy]) {
            assert!(
                (got - want).abs() < 1e-6 * want,
                "{backend:?}: intrinsics {cam} vs truth"
            );
        }
        let cost = solution.solve_report.final_cost;
        assert!(cost < 1e-9, "{backend:?}: final cost {cost:e}");
    }
}
