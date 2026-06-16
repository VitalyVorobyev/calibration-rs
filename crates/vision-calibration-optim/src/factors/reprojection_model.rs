//! Backend-independent reprojection residual models.

use crate::factors::camera_kernels::{DistortionKernel, ProjectionKernel, SensorKernel};
use crate::ir::ReprojChain;
use nalgebra::{
    DVector, DVectorView, Matrix3, Quaternion, RealField, SVector, UnitQuaternion, Vector3,
};

/// Apply Brown-Conrady distortion to normalized coordinates (generic for autodiff).
///
/// This implements the Brown-Conrady distortion model with radial (k1, k2, k3)
/// and tangential (p1, p2) coefficients. The function is generic over `RealField`
/// to support automatic differentiation.
pub(crate) fn distort_brown_conrady_generic<T: RealField>(
    x: T,
    y: T,
    k1: T,
    k2: T,
    k3: T,
    p1: T,
    p2: T,
) -> (T, T) {
    let r2 = x.clone() * x.clone() + y.clone() * y.clone();
    let r4 = r2.clone() * r2.clone();
    let r6 = r4.clone() * r2.clone();

    let radial = T::one() + k1 * r2.clone() + k2 * r4 + k3 * r6;

    let two = T::one() + T::one();
    let x2 = x.clone() * x.clone();
    let y2 = y.clone() * y.clone();
    let xy = x.clone() * y.clone();

    let x_tan =
        two.clone() * p1.clone() * xy.clone() + p2.clone() * (r2.clone() + two.clone() * x2);
    let y_tan = p1 * (r2 + two.clone() * y2) + two * p2 * xy;

    (x.clone() * radial.clone() + x_tan, y * radial + y_tan)
}

/// Apply rational polynomial distortion to normalized coordinates (generic for autodiff).
///
/// Implements the OpenCV rational model with numerator coefficients `(k1,k2,k3)`,
/// denominator coefficients `(k4,k5,k6)`, and tangential coefficients `(p1,p2)`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn distort_rational_generic<T: RealField>(
    x: T,
    y: T,
    k1: T,
    k2: T,
    k3: T,
    k4: T,
    k5: T,
    k6: T,
    p1: T,
    p2: T,
) -> (T, T) {
    let r2 = x.clone() * x.clone() + y.clone() * y.clone();
    let r4 = r2.clone() * r2.clone();
    let r6 = r4.clone() * r2.clone();

    let num = T::one() + k1 * r2.clone() + k2 * r4.clone() + k3 * r6.clone();
    let den = T::one() + k4 * r2.clone() + k5 * r4 + k6 * r6;
    let radial = num / den;

    let two = T::one() + T::one();
    let x2 = x.clone() * x.clone();
    let y2 = y.clone() * y.clone();
    let xy = x.clone() * y.clone();

    let x_tan =
        two.clone() * p1.clone() * xy.clone() + p2.clone() * (r2.clone() + two.clone() * x2);
    let y_tan = p1 * (r2 + two.clone() * y2) + two * p2 * xy;

    (x.clone() * radial.clone() + x_tan, y * radial + y_tan)
}

/// Apply thin-prism distortion to normalized coordinates (generic for autodiff).
///
/// Extends Brown-Conrady with four thin-prism coefficients `(s1,s2,s3,s4)`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn distort_thin_prism_generic<T: RealField>(
    x: T,
    y: T,
    k1: T,
    k2: T,
    k3: T,
    p1: T,
    p2: T,
    s1: T,
    s2: T,
    s3: T,
    s4: T,
) -> (T, T) {
    let r2 = x.clone() * x.clone() + y.clone() * y.clone();
    let r4 = r2.clone() * r2.clone();
    let r6 = r4.clone() * r2.clone();

    let radial = T::one() + k1 * r2.clone() + k2 * r4.clone() + k3 * r6;

    let two = T::one() + T::one();
    let x2 = x.clone() * x.clone();
    let y2 = y.clone() * y.clone();
    let xy = x.clone() * y.clone();

    let x_tan =
        two.clone() * p1.clone() * xy.clone() + p2.clone() * (r2.clone() + two.clone() * x2);
    let y_tan = p1 * (r2.clone() + two.clone() * y2) + two * p2 * xy;

    let prism_x = s1 * r2.clone() + s2 * r4.clone();
    let prism_y = s3 * r2 + s4 * r4;

    (
        x.clone() * radial.clone() + x_tan + prism_x,
        y * radial + y_tan + prism_y,
    )
}

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

pub(crate) fn se3_exp<T: RealField>(xi: DVectorView<'_, T>) -> (UnitQuaternion<T>, Vector3<T>) {
    debug_assert!(xi.len() == 6, "se3 tangent must have 6 params");
    let w = Vector3::new(xi[0].clone(), xi[1].clone(), xi[2].clone());
    let v = Vector3::new(xi[3].clone(), xi[4].clone(), xi[5].clone());

    let theta = w.norm();
    let eps = T::from_f64(1e-9).unwrap();

    let w_hat = skew_matrix(&w);
    let w_hat2 = w_hat.clone() * w_hat.clone();

    let (b, c) = if theta.clone() <= eps {
        let half = T::from_f64(0.5).unwrap();
        let sixth = T::from_f64(1.0 / 6.0).unwrap();
        (half, sixth)
    } else {
        let theta2 = theta.clone() * theta.clone();
        let theta3 = theta2.clone() * theta.clone();
        let sin_theta = theta.clone().sin();
        let cos_theta = theta.clone().cos();
        let b = (T::one() - cos_theta) / theta2;
        let c = (theta - sin_theta) / theta3;
        (b, c)
    };

    let v_mat = Matrix3::identity() + w_hat * b + w_hat2 * c;
    let t = v_mat * v;
    let rot = UnitQuaternion::from_scaled_axis(w);
    (rot, t)
}

/// Compute tilt projection matrix for Scheimpflug sensor (generic for autodiff).
///
/// This implements the OpenCV-compatible tilted sensor model using rotations
/// around X and Y axes followed by z-normalization projection.
pub(crate) fn tilt_projection_matrix_generic<T: RealField>(
    tau_x: T,
    tau_y: T,
) -> nalgebra::Matrix3<T> {
    let s_tx = tau_x.clone().sin();
    let c_tx = tau_x.cos();
    let s_ty = tau_y.clone().sin();
    let c_ty = tau_y.cos();

    let zero = T::zero();
    let one = T::one();

    let rot_x = nalgebra::Matrix3::new(
        one.clone(),
        zero.clone(),
        zero.clone(),
        zero.clone(),
        c_tx.clone(),
        s_tx.clone(),
        zero.clone(),
        -s_tx.clone(),
        c_tx,
    );
    let rot_y = nalgebra::Matrix3::new(
        c_ty.clone(),
        zero.clone(),
        -s_ty.clone(),
        zero.clone(),
        one.clone(),
        zero.clone(),
        s_ty.clone(),
        zero.clone(),
        c_ty,
    );
    let rot_xy = rot_y * rot_x;

    let r22 = rot_xy[(2, 2)].clone();
    let r02 = rot_xy[(0, 2)].clone();
    let r12 = rot_xy[(1, 2)].clone();

    nalgebra::Matrix3::new(
        r22.clone(),
        zero.clone(),
        -r02,
        zero.clone(),
        r22.clone(),
        -r12,
        zero.clone(),
        zero.clone(),
        one,
    ) * rot_xy
}

/// Apply Scheimpflug sensor homography to normalized coordinates (generic for autodiff).
pub(crate) fn apply_scheimpflug_generic<T: RealField>(
    x_norm: T,
    y_norm: T,
    tau_x: T,
    tau_y: T,
) -> (T, T) {
    let h = tilt_projection_matrix_generic(tau_x, tau_y);
    let p = nalgebra::Vector3::new(x_norm, y_norm, T::one());
    let p_tilted = h * p;
    let eps = T::from_f64(1e-12).unwrap();
    let z_safe = if p_tilted.z.clone() > eps.clone() {
        p_tilted.z.clone()
    } else {
        eps
    };
    (
        p_tilted.x.clone() / z_safe.clone(),
        p_tilted.y.clone() / z_safe,
    )
}

/// Apply inverse Scheimpflug sensor homography to sensor coordinates (generic for autodiff).
///
/// Maps sensor-plane normalized coordinates back to the normalized camera plane.
pub(crate) fn apply_scheimpflug_inverse_generic<T: RealField>(
    x_sensor: T,
    y_sensor: T,
    tau_x: T,
    tau_y: T,
) -> (T, T) {
    let h = tilt_projection_matrix_generic(tau_x, tau_y);
    let h_inv = match h.try_inverse() {
        Some(inv) => inv,
        None => return (x_sensor, y_sensor),
    };
    let p = h_inv * Vector3::new(x_sensor, y_sensor, T::one());
    let eps = T::from_f64(1e-12).unwrap();
    let z_safe = if p.z.clone().abs() > eps.clone() {
        p.z.clone()
    } else {
        eps
    };
    (p.x.clone() / z_safe.clone(), p.y.clone() / z_safe)
}

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
        ReprojChain::HandEye {
            base_se3_gripper,
            mode,
        } => {
            debug_assert!(blocks.len() == 3, "HandEye chain expects 3 blocks");
            let (robot_q, robot_t) = se3_from_f64_array::<T>(base_se3_gripper);
            handeye_chain_transform(
                &blocks[0], &blocks[1], &blocks[2], robot_q, robot_t, *mode, &pw_t,
            )
        }
        ReprojChain::HandEyeRobotDelta {
            base_se3_gripper,
            mode,
        } => {
            debug_assert!(
                blocks.len() == 4,
                "HandEyeRobotDelta chain expects 4 blocks"
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
    mode: crate::ir::HandEyeMode,
    pw_t: &Vector3<T>,
) -> Vector3<T> {
    let (extr_q, extr_t) = se3_from_block(extr);
    let (handeye_q, handeye_t) = se3_from_block(handeye);
    let (target_q, target_t) = se3_from_block(target);
    match mode {
        crate::ir::HandEyeMode::EyeInHand => {
            // target -> robot_base -> gripper -> rig -> camera
            let p_base = target_q.transform_vector(pw_t) + target_t.clone();
            let p_gripper = robot_q.inverse_transform_vector(&(p_base - robot_t.clone()));
            let p_rig = handeye_q.inverse_transform_vector(&(p_gripper - handeye_t.clone()));
            extr_q.inverse_transform_vector(&(p_rig - extr_t.clone()))
        }
        crate::ir::HandEyeMode::EyeToHand => {
            // target -> gripper -> robot_base -> rig -> camera
            let p_gripper = target_q.transform_vector(pw_t) + target_t.clone();
            let p_base = robot_q.transform_vector(&p_gripper) + robot_t.clone();
            let p_rig = handeye_q.transform_vector(&p_base) + handeye_t.clone();
            extr_q.inverse_transform_vector(&(p_rig - extr_t.clone()))
        }
    }
}

/// Reprojection residual generic over the camera-model kernels and pose chain.
///
/// `params` is the full IR-ordered block list: `[intrinsics, distortion?,
/// sensor?, <chain blocks>]`, with the optional blocks present exactly when
/// the corresponding kernel dimension is non-zero.
pub(crate) fn reproj_residual_model_generic<P, D, S, T>(
    chain: &ReprojChain,
    params: &[DVector<T>],
    pw: [f64; 3],
    uv: [f64; 2],
    w: f64,
) -> SVector<T, 2>
where
    P: ProjectionKernel,
    D: DistortionKernel,
    S: SensorKernel,
    T: RealField,
{
    let mut idx = 1;
    let dist = (D::DIM > 0).then(|| {
        let v = params[idx].as_view();
        idx += 1;
        v
    });
    let sensor = (S::DIM > 0).then(|| {
        let v = params[idx].as_view();
        idx += 1;
        v
    });

    let p_camera = reproj_chain_transform(chain, &params[idx..], pw);
    let (x_norm, y_norm) = P::normalize(&p_camera);
    let (x_dist, y_dist) = D::distort(dist, x_norm, y_norm);
    let (x_sensor, y_sensor) = S::to_sensor(sensor, x_dist, y_dist);

    let intr = &params[0];
    debug_assert!(intr.len() >= 4, "intrinsics must have 4 params");
    let fx = intr[0].clone();
    let fy = intr[1].clone();
    let cx = intr[2].clone();
    let cy = intr[3].clone();
    let u_proj = fx * x_sensor + cx;
    let v_proj = fy * y_sensor + cy;

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
    use crate::factors::camera_kernels::{
        BrownConrady5Kernel, DivisionKernel, IdentitySensorKernel, NoDistortionKernel,
        PinholeKernel, RationalKernel, Scheimpflug2Kernel, ThinPrismKernel,
    };
    use crate::ir::HandEyeMode;
    use nalgebra::DVector;

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

    /// The optimizer's autodiff tilt matrix (this module) must equal core's
    /// f64 Scheimpflug homography bit-for-bit, so the Jacobian path and the
    /// forward model share one OpenCV-compatible convention.
    #[test]
    fn tilt_generic_matches_core_f64() {
        for &(tx, ty) in &[(0.05, -0.03), (0.1, 0.1), (-0.12, 0.07), (0.2, -0.25)] {
            let generic = tilt_projection_matrix_generic::<f64>(tx, ty);
            let core = vision_calibration_core::ScheimpflugParams {
                tilt_x: tx,
                tilt_y: ty,
            }
            .compile()
            .h;
            let diff = (generic - core).abs().max();
            assert!(diff < 1e-12, "(τx={tx}, τy={ty}) max abs diff {diff:e}");
        }
    }

    #[test]
    fn distortion_changes_projection() {
        let intr = DVector::from_row_slice(&[800.0, 800.0, 640.0, 360.0]);
        let dist_zero = DVector::from_row_slice(&[0.0, 0.0, 0.0, 0.0, 0.0]);
        let dist_barrel = DVector::from_row_slice(&[-0.3, 0.1, 0.0, 0.0, 0.0]);
        let pose = DVector::from_row_slice(&[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0]);
        let pw = [0.5, 0.5, 1.0];
        let uv = [1000.0, 700.0];

        let r1 = reproj_residual_model_generic::<
            PinholeKernel,
            BrownConrady5Kernel,
            IdentitySensorKernel,
            f64,
        >(
            &ReprojChain::SinglePose,
            &[intr.clone(), dist_zero, pose.clone()],
            pw,
            uv,
            1.0,
        );
        let r2 = reproj_residual_model_generic::<
            PinholeKernel,
            BrownConrady5Kernel,
            IdentitySensorKernel,
            f64,
        >(
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

        let r_nodist = reproj_residual_model_generic::<
            PinholeKernel,
            NoDistortionKernel,
            IdentitySensorKernel,
            f64,
        >(
            &ReprojChain::SinglePose,
            &[intr.clone(), pose.clone()],
            PW,
            UV,
            W,
        );
        let r_zerodist = reproj_residual_model_generic::<
            PinholeKernel,
            BrownConrady5Kernel,
            IdentitySensorKernel,
            f64,
        >(
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

        let r_nodist = reproj_residual_model_generic::<
            PinholeKernel,
            NoDistortionKernel,
            IdentitySensorKernel,
            f64,
        >(
            &ReprojChain::SinglePose,
            &[intr.clone(), pose.clone()],
            PW,
            UV,
            W,
        );
        let r_zerodist = reproj_residual_model_generic::<
            PinholeKernel,
            RationalKernel,
            IdentitySensorKernel,
            f64,
        >(
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

        let r_nodist = reproj_residual_model_generic::<
            PinholeKernel,
            NoDistortionKernel,
            IdentitySensorKernel,
            f64,
        >(
            &ReprojChain::SinglePose,
            &[intr.clone(), pose.clone()],
            PW,
            UV,
            W,
        );
        let r_zerodist = reproj_residual_model_generic::<
            PinholeKernel,
            ThinPrismKernel,
            IdentitySensorKernel,
            f64,
        >(
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

        let r_nodist = reproj_residual_model_generic::<
            PinholeKernel,
            NoDistortionKernel,
            IdentitySensorKernel,
            f64,
        >(
            &ReprojChain::SinglePose,
            &[intr.clone(), pose.clone()],
            PW,
            UV,
            W,
        );
        let r_zerodist = reproj_residual_model_generic::<
            PinholeKernel,
            DivisionKernel,
            IdentitySensorKernel,
            f64,
        >(
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
                ReprojChain::HandEye {
                    base_se3_gripper: robot_se3,
                    mode: HandEyeMode::EyeInHand,
                },
                vec![poses[0].clone(), poses[1].clone(), poses[2].clone()],
            ),
            (
                ReprojChain::HandEyeRobotDelta {
                    base_se3_gripper: robot_se3,
                    mode: HandEyeMode::EyeToHand,
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

            let r_pin = reproj_residual_model_generic::<
                PinholeKernel,
                BrownConrady5Kernel,
                IdentitySensorKernel,
                f64,
            >(&chain, &params_pin, PW, UV, W);
            let r_sch = reproj_residual_model_generic::<
                PinholeKernel,
                BrownConrady5Kernel,
                Scheimpflug2Kernel,
                f64,
            >(&chain, &params_sch, PW, UV, W);
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

        for mode in [HandEyeMode::EyeInHand, HandEyeMode::EyeToHand] {
            let r_he = reproj_residual_model_generic::<
                PinholeKernel,
                BrownConrady5Kernel,
                IdentitySensorKernel,
                f64,
            >(
                &ReprojChain::HandEye {
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
            let r_delta = reproj_residual_model_generic::<
                PinholeKernel,
                BrownConrady5Kernel,
                IdentitySensorKernel,
                f64,
            >(
                &ReprojChain::HandEyeRobotDelta {
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
