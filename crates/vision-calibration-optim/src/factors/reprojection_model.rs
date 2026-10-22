//! Backend-independent reprojection residual models.

use crate::factors::camera_blocks::CameraBlocks;
use crate::ir::{CameraModelSpec, ReprojChain};
use nalgebra::{
    DVector, DVectorView, Matrix3, Quaternion, RealField, SVector, UnitQuaternion, Vector3,
};

fn skew_matrix<T: RealField>(w: &Vector3<T>) -> Matrix3<T> {
    Matrix3::new(
        T::zero(),
        -w.z.clone(),
        w.y.clone(),
        w.z.clone(),
        T::zero(),
        -w.x.clone(),
        -w.y.clone(),
        w.x.clone(),
        T::zero(),
    )
}

/// `exp` of an se(3) tangent `[ω, v]`: the rotation `Exp(ω)` and the
/// translation `V(ω) v`.
///
/// Below [`SE3_EXP_SERIES_ANGLE`] the rotation and the `V` coefficients are
/// their Taylor series, which are exact to working precision there and
/// differentiable at zero. The closed forms are not: `UnitQuaternion::
/// from_scaled_axis` returns the identity for a zero vector, so its
/// derivative there is zero, and an optimizer starting from `xi = 0` (every
/// robot-pose correction does) would see no rotation Jacobian.
pub(crate) fn se3_exp<T: RealField>(xi: DVectorView<'_, T>) -> (UnitQuaternion<T>, Vector3<T>) {
    debug_assert!(xi.len() == 6, "se3 tangent must have 6 params");
    let w = Vector3::new(xi[0].clone(), xi[1].clone(), xi[2].clone());
    let v = Vector3::new(xi[3].clone(), xi[4].clone(), xi[5].clone());

    let theta2 = w.norm_squared();
    let series =
        theta2.clone() <= T::from_f64(SE3_EXP_SERIES_ANGLE * SE3_EXP_SERIES_ANGLE).unwrap();

    let w_hat = skew_matrix(&w);
    let w_hat2 = w_hat.clone() * w_hat.clone();

    let (b, c, rot) = if series {
        // b = (1 − cos θ)/θ², c = (θ − sin θ)/θ³, q = (cos θ/2, w sin(θ/2)/θ),
        // each to O(θ²); the next terms are below 1e-19 for θ ≤ 1e-4.
        let f = |a: f64, k: f64| T::from_f64(a).unwrap() - theta2.clone() * T::from_f64(k).unwrap();
        let b = f(0.5, 1.0 / 24.0);
        let c = f(1.0 / 6.0, 1.0 / 120.0);
        let q = Quaternion::from_parts(f(1.0, 1.0 / 8.0), w.clone() * f(0.5, 1.0 / 48.0));
        (b, c, UnitQuaternion::new_normalize(q))
    } else {
        let theta = theta2.clone().sqrt();
        let theta3 = theta2.clone() * theta.clone();
        let b = (T::one() - theta.clone().cos()) / theta2;
        let c = (theta.clone() - theta.sin()) / theta3;
        (b, c, UnitQuaternion::from_scaled_axis(w))
    };

    let v_mat = Matrix3::identity() + w_hat * b + w_hat2 * c;
    let t = v_mat * v;
    (rot, t)
}

/// Rotation angle (radians) below which [`se3_exp`] uses its series.
const SE3_EXP_SERIES_ANGLE: f64 = 1e-4;

/// Extract `(rotation, translation)` from an SE3 parameter block
/// `[qx, qy, qz, qw, tx, ty, tz]`.
pub(crate) fn se3_from_block<T: RealField>(v: &DVector<T>) -> (UnitQuaternion<T>, Vector3<T>) {
    debug_assert!(v.len() == 7, "SE3 block must have 7 params");
    let q = UnitQuaternion::from_quaternion(Quaternion::new(
        v[3].clone(),
        v[0].clone(),
        v[1].clone(),
        v[2].clone(),
    ));
    let t = Vector3::new(v[4].clone(), v[5].clone(), v[6].clone());
    (q, t)
}

/// Lift a measured SE3 `[qx, qy, qz, qw, tx, ty, tz]` into the scalar type.
pub(crate) fn se3_from_f64_array<T: RealField>(a: &[f64; 7]) -> (UnitQuaternion<T>, Vector3<T>) {
    let q = UnitQuaternion::from_quaternion(Quaternion::new(
        T::from_f64(a[3]).unwrap(),
        T::from_f64(a[0]).unwrap(),
        T::from_f64(a[1]).unwrap(),
        T::from_f64(a[2]).unwrap(),
    ));
    let t = Vector3::new(
        T::from_f64(a[4]).unwrap(),
        T::from_f64(a[5]).unwrap(),
        T::from_f64(a[6]).unwrap(),
    );
    (q, t)
}

/// Transform a target-frame point into the camera frame through a
/// [`ReprojChain`].
///
/// `blocks` are the chain's parameter blocks in IR order (camera blocks
/// already stripped). The point is pushed through the chain step by step,
/// matching the per-chain operation order of the enumerated kernels exactly.
pub(crate) fn reproj_chain_transform<T: RealField>(
    chain: &ReprojChain,
    blocks: &[DVector<T>],
    pw: [f64; 3],
) -> Vector3<T> {
    let pw_t = Vector3::new(
        T::from_f64(pw[0]).unwrap(),
        T::from_f64(pw[1]).unwrap(),
        T::from_f64(pw[2]).unwrap(),
    );
    match chain {
        ReprojChain::SinglePose => {
            debug_assert!(blocks.len() == 1, "SinglePose chain expects 1 block");
            let (rot, t) = se3_from_block(&blocks[0]);
            rot.transform_vector(&pw_t) + t
        }
        ReprojChain::TwoSe3 => {
            debug_assert!(blocks.len() == 2, "TwoSe3 chain expects 2 blocks");
            let (extr_q, extr_t) = se3_from_block(&blocks[0]);
            let (pose_q, pose_t) = se3_from_block(&blocks[1]);
            let p_rig = pose_q.transform_vector(&pw_t) + pose_t;
            extr_q.inverse_transform_vector(&(p_rig - extr_t))
        }
        ReprojChain::Handeye {
            base_se3_gripper,
            mode,
        } => {
            debug_assert!(blocks.len() == 3, "Handeye chain expects 3 blocks");
            let (robot_q, robot_t) = se3_from_f64_array::<T>(base_se3_gripper);
            handeye_chain_transform(
                &blocks[0], &blocks[1], &blocks[2], robot_q, robot_t, *mode, &pw_t,
            )
        }
        ReprojChain::HandeyeRobotDelta {
            base_se3_gripper,
            mode,
        } => {
            debug_assert!(
                blocks.len() == 4,
                "HandeyeRobotDelta chain expects 4 blocks"
            );
            let (robot_q, robot_t) = se3_from_f64_array::<T>(base_se3_gripper);
            let (delta_q, delta_t) = se3_exp(blocks[3].as_view());
            let robot_q = delta_q.clone() * robot_q;
            let robot_t = delta_q.transform_vector(&robot_t) + delta_t;
            handeye_chain_transform(
                &blocks[0], &blocks[1], &blocks[2], robot_q, robot_t, *mode, &pw_t,
            )
        }
    }
}

/// Shared hand-eye chain: target -> (robot, hand-eye per mode) -> rig -> camera.
fn handeye_chain_transform<T: RealField>(
    extr: &DVector<T>,
    handeye: &DVector<T>,
    target: &DVector<T>,
    robot_q: UnitQuaternion<T>,
    robot_t: Vector3<T>,
    mode: crate::ir::HandeyeMode,
    pw_t: &Vector3<T>,
) -> Vector3<T> {
    let (extr_q, extr_t) = se3_from_block(extr);
    let (handeye_q, handeye_t) = se3_from_block(handeye);
    let (target_q, target_t) = se3_from_block(target);
    match mode {
        crate::ir::HandeyeMode::EyeInHand => {
            // target -> robot_base -> gripper -> rig -> camera
            let p_base = target_q.transform_vector(pw_t) + target_t.clone();
            let p_gripper = robot_q.inverse_transform_vector(&(p_base - robot_t.clone()));
            let p_rig = handeye_q.inverse_transform_vector(&(p_gripper - handeye_t.clone()));
            extr_q.inverse_transform_vector(&(p_rig - extr_t.clone()))
        }
        crate::ir::HandeyeMode::EyeToHand => {
            // target -> gripper -> robot_base -> rig -> camera
            let p_gripper = target_q.transform_vector(pw_t) + target_t.clone();
            let p_base = robot_q.transform_vector(&p_gripper) + robot_t.clone();
            let p_rig = handeye_q.transform_vector(&p_base) + handeye_t.clone();
            extr_q.inverse_transform_vector(&(p_rig - extr_t.clone()))
        }
    }
}

/// Reprojection residual of a camera model and pose chain.
///
/// `params` is the full IR-ordered block list: the camera blocks of `model`
/// (`[intrinsics, projection?, distortion?, sensor?]`) followed by the
/// chain blocks.
pub(crate) fn reproj_residual<T: RealField>(
    model: CameraModelSpec,
    chain: &ReprojChain,
    params: &[DVector<T>],
    pw: [f64; 3],
    uv: [f64; 2],
    w: f64,
) -> SVector<T, 2> {
    let (camera, chain_blocks) = CameraBlocks::split(model, params);
    let p_camera = reproj_chain_transform(chain, chain_blocks, pw);
    let (u_proj, v_proj) = camera.project(&p_camera);

    let sqrt_w = T::from_f64(w.sqrt()).unwrap();
    let u_meas = T::from_f64(uv[0]).unwrap();
    let v_meas = T::from_f64(uv[1]).unwrap();
    SVector::<T, 2>::new(
        (u_meas - u_proj) * sqrt_w.clone(),
        (v_meas - v_proj) * sqrt_w,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::HandeyeMode;
    use crate::ir::{DistortionKind, SensorKind};
    use nalgebra::DVector;

    /// At `xi = 0` (where every robot-pose correction starts) the rotation
    /// has its exact Jacobian, `∂q_xyz/∂ω = I/2`, and across the series
    /// threshold the two forms agree to working precision.
    #[test]
    fn se3_exp_is_differentiable_at_zero_and_continuous() {
        use num_dual::Dual64;
        for axis in 0..3 {
            let mut xi = DVector::from_element(6, Dual64::from(0.0));
            xi[axis] = Dual64::new(0.0, 1.0);
            let (q, t) = se3_exp(xi.as_view());
            let imag = q.as_ref().imag();
            for k in 0..3 {
                let expected = if k == axis { 0.5 } else { 0.0 };
                assert!((imag[k].eps - expected).abs() < 1e-15, "∂q{k}/∂ω{axis}");
                assert_eq!(t[k].eps, 0.0);
            }
        }
        let axis = nalgebra::Vector3::new(0.3, -0.5, 0.8).normalize();
        let at = |angle: f64| {
            let w = axis * angle;
            let xi = DVector::from_row_slice(&[w.x, w.y, w.z, 0.01, -0.02, 0.03]);
            se3_exp(xi.as_view())
        };
        let (below, above) = (
            at(SE3_EXP_SERIES_ANGLE * (1.0 - 1e-9)),
            at(SE3_EXP_SERIES_ANGLE),
        );
        assert!(below.0.angle_to(&above.0) < 1e-12);
        assert!((below.1 - above.1).norm() < 1e-15);
    }

    fn fixture_blocks() -> (DVector<f64>, DVector<f64>, Vec<DVector<f64>>) {
        let intr = DVector::from_row_slice(&[812.3, 798.7, 645.2, 357.9]);
        let dist = DVector::from_row_slice(&[-0.11, 0.07, 0.012, 0.0015, -0.0023]);
        let poses = vec![
            DVector::from_row_slice(&[0.021, 0.034, -0.012, 0.999_03, 0.12, -0.05, 0.83]),
            DVector::from_row_slice(&[0.051, -0.022, 0.041, 0.997_55, 0.41, 0.21, 0.92]),
            DVector::from_row_slice(&[-0.031, 0.018, 0.009, 0.999_24, 0.08, -0.04, 1.12]),
        ];
        (intr, dist, poses)
    }

    const PW: [f64; 3] = [0.113, -0.072, 0.004];
    const UV: [f64; 2] = [684.2, 341.7];
    const W: f64 = 1.7;

    #[test]
    fn distortion_changes_projection() {
        let intr = DVector::from_row_slice(&[800.0_f64, 800.0, 640.0, 360.0]);
        let dist_zero = DVector::from_row_slice(&[0.0, 0.0, 0.0, 0.0, 0.0]);
        let dist_barrel = DVector::from_row_slice(&[-0.3, 0.1, 0.0, 0.0, 0.0]);
        let pose = DVector::from_row_slice(&[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0]);
        let pw = [0.5, 0.5, 1.0];
        let uv = [1000.0, 700.0];

        let r1 = reproj_residual(
            CameraModelSpec::pinhole(DistortionKind::BrownConrady5, SensorKind::Identity),
            &ReprojChain::SinglePose,
            &[intr.clone(), dist_zero, pose.clone()],
            pw,
            uv,
            1.0,
        );
        let r2 = reproj_residual(
            CameraModelSpec::pinhole(DistortionKind::BrownConrady5, SensorKind::Identity),
            &ReprojChain::SinglePose,
            &[intr, dist_barrel, pose],
            pw,
            uv,
            1.0,
        );

        let diff = (r1[0] - r2[0]).abs();
        assert!(
            diff > 1.0,
            "Expected residuals to differ by >1.0, got diff={diff}"
        );
    }

    #[test]
    fn zero_distortion_matches_no_distortion() {
        let (intr, _, poses) = fixture_blocks();
        let dist_zero = DVector::from_row_slice(&[0.0; 5]);
        let pose = &poses[0];

        let r_nodist = reproj_residual(
            CameraModelSpec::pinhole(DistortionKind::None, SensorKind::Identity),
            &ReprojChain::SinglePose,
            &[intr.clone(), pose.clone()],
            PW,
            UV,
            W,
        );
        let r_zerodist = reproj_residual(
            CameraModelSpec::pinhole(DistortionKind::BrownConrady5, SensorKind::Identity),
            &ReprojChain::SinglePose,
            &[intr, dist_zero, pose.clone()],
            PW,
            UV,
            W,
        );
        assert_eq!(
            r_nodist, r_zerodist,
            "zero Brown-Conrady coefficients must be an exact identity"
        );
    }

    /// Zero rational polynomial coefficients must produce the same result as no distortion.
    #[test]
    fn zero_rational_matches_no_distortion() {
        let (intr, _, poses) = fixture_blocks();
        let dist_zero = DVector::from_row_slice(&[0.0; 8]);
        let pose = &poses[0];

        let r_nodist = reproj_residual(
            CameraModelSpec::pinhole(DistortionKind::None, SensorKind::Identity),
            &ReprojChain::SinglePose,
            &[intr.clone(), pose.clone()],
            PW,
            UV,
            W,
        );
        let r_zerodist = reproj_residual(
            CameraModelSpec::pinhole(DistortionKind::Rational8, SensorKind::Identity),
            &ReprojChain::SinglePose,
            &[intr, dist_zero, pose.clone()],
            PW,
            UV,
            W,
        );
        assert_eq!(
            r_nodist, r_zerodist,
            "zero rational coefficients must be an exact identity"
        );
    }

    /// Zero thin-prism coefficients must produce the same result as no distortion.
    #[test]
    fn zero_thin_prism_matches_no_distortion() {
        let (intr, _, poses) = fixture_blocks();
        let dist_zero = DVector::from_row_slice(&[0.0; 9]);
        let pose = &poses[0];

        let r_nodist = reproj_residual(
            CameraModelSpec::pinhole(DistortionKind::None, SensorKind::Identity),
            &ReprojChain::SinglePose,
            &[intr.clone(), pose.clone()],
            PW,
            UV,
            W,
        );
        let r_zerodist = reproj_residual(
            CameraModelSpec::pinhole(DistortionKind::ThinPrism9, SensorKind::Identity),
            &ReprojChain::SinglePose,
            &[intr, dist_zero, pose.clone()],
            PW,
            UV,
            W,
        );
        assert_eq!(
            r_nodist, r_zerodist,
            "zero thin-prism coefficients must be an exact identity"
        );
    }

    /// Zero division lambda must produce the same result as no distortion.
    #[test]
    fn zero_division_matches_no_distortion() {
        let (intr, _, poses) = fixture_blocks();
        let dist_zero = DVector::from_row_slice(&[0.0_f64]);
        let pose = &poses[0];

        let r_nodist = reproj_residual(
            CameraModelSpec::pinhole(DistortionKind::None, SensorKind::Identity),
            &ReprojChain::SinglePose,
            &[intr.clone(), pose.clone()],
            PW,
            UV,
            W,
        );
        let r_zerodist = reproj_residual(
            CameraModelSpec::pinhole(DistortionKind::Division1, SensorKind::Identity),
            &ReprojChain::SinglePose,
            &[intr, dist_zero, pose.clone()],
            PW,
            UV,
            W,
        );
        assert_eq!(
            r_nodist, r_zerodist,
            "zero division lambda must be an exact identity"
        );
    }

    /// Zero Scheimpflug tilt is an exact identity, so the Scheimpflug2 kernel
    /// with a zero sensor block must reproduce the identity-sensor kernel for
    /// every chain.
    #[test]
    fn zero_tilt_scheimpflug_matches_identity_sensor_for_all_chains() {
        let (intr, dist, poses) = fixture_blocks();
        let sensor_zero = DVector::from_row_slice(&[0.0, 0.0]);
        let robot_se3 = [0.024, 0.011, 0.032, 0.999_15, 0.51, -0.22, 0.78];
        let delta = DVector::from_row_slice(&[0.0012, -0.0021, 0.0033, 0.0006, -0.0011, 0.0024]);

        let chains: Vec<(ReprojChain, Vec<DVector<f64>>)> = vec![
            (ReprojChain::SinglePose, vec![poses[0].clone()]),
            (
                ReprojChain::TwoSe3,
                vec![poses[0].clone(), poses[1].clone()],
            ),
            (
                ReprojChain::Handeye {
                    base_se3_gripper: robot_se3,
                    mode: HandeyeMode::EyeInHand,
                },
                vec![poses[0].clone(), poses[1].clone(), poses[2].clone()],
            ),
            (
                ReprojChain::HandeyeRobotDelta {
                    base_se3_gripper: robot_se3,
                    mode: HandeyeMode::EyeToHand,
                },
                vec![
                    poses[0].clone(),
                    poses[1].clone(),
                    poses[2].clone(),
                    delta.clone(),
                ],
            ),
        ];

        for (chain, chain_blocks) in chains {
            let mut params_pin = vec![intr.clone(), dist.clone()];
            params_pin.extend(chain_blocks.iter().cloned());
            let mut params_sch = vec![intr.clone(), dist.clone(), sensor_zero.clone()];
            params_sch.extend(chain_blocks.iter().cloned());

            let r_pin = reproj_residual(
                CameraModelSpec::pinhole(DistortionKind::BrownConrady5, SensorKind::Identity),
                &chain,
                &params_pin,
                PW,
                UV,
                W,
            );
            let r_sch = reproj_residual(
                CameraModelSpec::pinhole(DistortionKind::BrownConrady5, SensorKind::Scheimpflug),
                &chain,
                &params_sch,
                PW,
                UV,
                W,
            );
            assert!(
                (r_pin[0] - r_sch[0]).abs() < 1e-10 && (r_pin[1] - r_sch[1]).abs() < 1e-10,
                "zero-tilt mismatch for {chain:?}: pin={r_pin:?} sch={r_sch:?}"
            );
        }
    }

    /// The robot-delta chain with a zero delta must reproduce the plain
    /// hand-eye chain exactly (se3_exp(0) is the identity).
    #[test]
    fn zero_robot_delta_matches_handeye_chain() {
        let (intr, dist, poses) = fixture_blocks();
        let robot_se3 = [0.024, 0.011, 0.032, 0.999_15, 0.51, -0.22, 0.78];
        let zero_delta = DVector::from_element(6, 0.0);

        for mode in [HandeyeMode::EyeInHand, HandeyeMode::EyeToHand] {
            let r_he = reproj_residual(
                CameraModelSpec::pinhole(DistortionKind::BrownConrady5, SensorKind::Identity),
                &ReprojChain::Handeye {
                    base_se3_gripper: robot_se3,
                    mode,
                },
                &[
                    intr.clone(),
                    dist.clone(),
                    poses[0].clone(),
                    poses[1].clone(),
                    poses[2].clone(),
                ],
                PW,
                UV,
                W,
            );
            let r_delta = reproj_residual(
                CameraModelSpec::pinhole(DistortionKind::BrownConrady5, SensorKind::Identity),
                &ReprojChain::HandeyeRobotDelta {
                    base_se3_gripper: robot_se3,
                    mode,
                },
                &[
                    intr.clone(),
                    dist.clone(),
                    poses[0].clone(),
                    poses[1].clone(),
                    poses[2].clone(),
                    zero_delta.clone(),
                ],
                PW,
                UV,
                W,
            );
            assert!(
                (r_he[0] - r_delta[0]).abs() < 1e-12 && (r_he[1] - r_delta[1]).abs() < 1e-12,
                "zero-delta mismatch {mode:?}: he={r_he:?} delta={r_delta:?}"
            );
        }
    }
}
