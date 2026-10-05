//! Cross-backend parity at a point: for every factor kind × camera model ×
//! chain, the tiny-solver and factrs engines must produce the same residual,
//! Jacobian and robust cost.
//!
//! Both engines retract SE3 as `x·exp(δ)` with a rotation-first tangent and
//! share the S² map, so their Jacobians agree column for column at the
//! linearization point; a one-residual IR lays out the columns in slot order
//! in both.

use std::collections::HashMap;

use nalgebra::{DVector, UnitQuaternion, Vector3};

use crate::backend::{factrs_backend, tiny_solver_backend};
use crate::ir::{
    CameraModelDesc, FactorKind, FixedMask, HandEyeMode, LaserChain, ProblemIR, ReprojChain,
    ResidualBlock, RobustLoss,
};

const MODELS: [CameraModelDesc; 10] = [
    CameraModelDesc::PINHOLE4,
    CameraModelDesc::PINHOLE4_DIST5,
    CameraModelDesc::PINHOLE4_RATIONAL8,
    CameraModelDesc::PINHOLE4_THINPRISM9,
    CameraModelDesc::PINHOLE4_DIVISION1,
    CameraModelDesc {
        sensor: crate::ir::SensorKind::Scheimpflug2,
        ..CameraModelDesc::PINHOLE4
    },
    CameraModelDesc::PINHOLE4_DIST5_SCHEIMPFLUG2,
    CameraModelDesc::PINHOLE4_RATIONAL8_SCHEIMPFLUG2,
    CameraModelDesc::PINHOLE4_THINPRISM9_SCHEIMPFLUG2,
    CameraModelDesc::PINHOLE4_DIVISION1_SCHEIMPFLUG2,
];

/// An SE3 block `[qx, qy, qz, qw, tx, ty, tz]` from a rotation vector and a
/// translation (a unit quaternion, so no backend renormalizes it).
fn se3(rotvec: [f64; 3], t: [f64; 3]) -> DVector<f64> {
    let q = UnitQuaternion::from_scaled_axis(Vector3::from(rotvec));
    DVector::from_row_slice(&[q.i, q.j, q.k, q.w, t[0], t[1], t[2]])
}

/// Fixture values for a slot, chosen so every chain puts the target point in
/// front of the camera.
fn fixture(role: &str, dim: usize) -> DVector<f64> {
    match (role, dim) {
        ("intrinsics", 4) => DVector::from_row_slice(&[812.3, 798.7, 645.2, 357.9]),
        ("distortion", 1) => DVector::from_row_slice(&[-0.21]),
        ("distortion", 5) => DVector::from_row_slice(&[-0.11, 0.07, 0.012, 0.0015, -0.0023]),
        ("distortion", 8) => {
            DVector::from_row_slice(&[-0.11, 0.07, 0.012, 0.004, -0.002, 0.001, 0.0015, -0.0023])
        }
        ("distortion", 9) => DVector::from_row_slice(&[
            -0.11, 0.07, 0.012, 0.0015, -0.0023, 0.0007, -0.0004, 0.0003, 0.0002,
        ]),
        ("sensor", 2) => DVector::from_row_slice(&[0.021, -0.013]),
        ("camera_se3_target" | "pose", 7) => se3([0.10, -0.04, 0.08], [0.04, 0.02, 0.92]),
        ("extrinsics" | "cam_se3_rig", 7) => se3([0.04, 0.07, -0.02], [0.08, -0.03, 0.05]),
        ("handeye", 7) => se3([-0.06, 0.04, 0.02], [0.03, -0.02, 0.10]),
        ("target" | "target_ref", 7) => se3([0.08, 0.10, -0.04], [0.10, 0.05, 0.85]),
        ("robot_delta", 6) => {
            DVector::from_row_slice(&[0.0012, -0.0021, 0.0033, 0.0006, -0.0011, 0.0024])
        }
        ("plane_normal", 3) => {
            let n = Vector3::new(0.09, 0.17, 1.0).normalize();
            DVector::from_row_slice(&[n.x, n.y, n.z])
        }
        ("plane_distance", 1) => DVector::from_row_slice(&[-0.33]),
        other => panic!("no fixture for {other:?}"),
    }
}

/// A one-residual IR for `factor` plus its initial values.
fn one_residual_ir(
    factor: FactorKind,
    loss: RobustLoss,
) -> (ProblemIR, HashMap<String, DVector<f64>>) {
    let mut ir = ProblemIR::new();
    let mut initial = HashMap::new();
    let params = factor
        .param_layout()
        .iter()
        .enumerate()
        .map(|(i, slot)| {
            let name = format!("{}_{i}", slot.role);
            initial.insert(name.clone(), fixture(slot.role, slot.dim));
            ir.add_param_block(name, slot.dim, slot.manifold, FixedMask::all_free(), None)
        })
        .collect();
    ir.add_residual_block(ResidualBlock {
        params,
        loss,
        residual_dim: factor.residual_dim(),
        factor,
    });
    (ir, initial)
}

/// Every factor kind × camera model × chain.
fn all_factors() -> Vec<FactorKind> {
    let robot = [0.024, 0.011, 0.032, 0.999_15, 0.08, -0.04, 0.06];
    let norm = (robot[..4].iter().map(|v| v * v).sum::<f64>()).sqrt();
    let robot = [
        robot[0] / norm,
        robot[1] / norm,
        robot[2] / norm,
        robot[3] / norm,
        robot[4],
        robot[5],
        robot[6],
    ];
    let mut out = vec![FactorKind::Se3TangentPrior {
        sqrt_info: [1.0, 2.0, 0.5, 3.0, 1.5, 0.7],
    }];
    for model in MODELS {
        for chain in [
            ReprojChain::SinglePose,
            ReprojChain::TwoSe3,
            ReprojChain::HandEye {
                base_se3_gripper: robot,
                mode: HandEyeMode::EyeToHand,
            },
            ReprojChain::HandEyeRobotDelta {
                base_se3_gripper: robot,
                mode: HandEyeMode::EyeInHand,
            },
        ] {
            out.push(FactorKind::ReprojPoint {
                model,
                chain,
                pw: [0.113, -0.072, 0.004],
                uv: [684.2, 341.7],
                w: 1.7,
            });
        }
        for chain in [
            LaserChain::SinglePose,
            LaserChain::RigHandEye {
                base_se3_gripper: robot,
                mode: HandEyeMode::EyeInHand,
            },
            LaserChain::RigHandEyeRobotDelta {
                base_se3_gripper: robot,
                mode: HandEyeMode::EyeToHand,
            },
        ] {
            let (laser_pixel, w) = ([702.0, 391.0], 1.3);
            out.push(FactorKind::LaserPointToPlane {
                model,
                chain,
                laser_pixel,
                w,
            });
            out.push(FactorKind::LaserLineDistance {
                model,
                chain,
                laser_pixel,
                w,
            });
        }
    }
    out
}

#[test]
fn every_factor_linearizes_identically_in_both_backends() {
    let factors = all_factors();
    assert_eq!(factors.len(), 1 + MODELS.len() * (4 + 3 * 2));
    for factor in factors {
        let (ir, init) = one_residual_ir(factor.clone(), RobustLoss::None);
        let (rt, jt, ct) = tiny_solver_backend::linearize_at(&ir, &init);
        let (rf, jf, cf) = factrs_backend::linearize_at(&ir, &init);

        assert_eq!(rt.len(), rf.len(), "{factor:?}: residual length");
        assert_eq!(jt.shape(), jf.shape(), "{factor:?}: Jacobian shape");
        let r_scale = rt.amax().max(1.0);
        assert!(
            (&rt - &rf).amax() <= 1e-12 * r_scale,
            "{factor:?}: residual {rt} vs {rf}"
        );
        assert!(rt.iter().all(|v| v.is_finite()), "{factor:?}: residual");
        let j_scale = jt.amax().max(1.0);
        let dj = (&jt - &jf).amax();
        assert!(
            dj <= 1e-9 * j_scale,
            "{factor:?}: Jacobians differ by {dj:e} (scale {j_scale:e})"
        );
        assert!(
            (ct - cf).abs() <= 1e-12 * ct.abs().max(1.0),
            "{factor:?}: cost {ct} vs {cf}"
        );
    }
}

#[test]
fn robust_costs_agree_across_backends() {
    let factor = all_factors()[2].clone();
    for loss in [
        RobustLoss::Huber { scale: 3.0 },
        RobustLoss::Cauchy { scale: 3.0 },
        RobustLoss::Arctan { scale: 3.0 },
    ] {
        let (ir, init) = one_residual_ir(factor.clone(), loss);
        let (_, _, ct) = tiny_solver_backend::linearize_at(&ir, &init);
        let (_, _, cf) = factrs_backend::linearize_at(&ir, &init);
        let (ir_l2, init_l2) = one_residual_ir(factor.clone(), RobustLoss::None);
        let (_, _, s) = tiny_solver_backend::linearize_at(&ir_l2, &init_l2);
        assert!(
            (ct - cf).abs() <= 1e-12 * ct.abs(),
            "{loss:?}: cost {ct} vs {cf}"
        );
        assert!(
            (ct - loss.rho(s)).abs() <= 1e-12 * ct.abs(),
            "{loss:?}: cost {ct} is not rho(|r|^2) = {}",
            loss.rho(s)
        );
    }
}
