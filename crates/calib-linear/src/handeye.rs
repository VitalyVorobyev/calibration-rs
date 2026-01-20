//! Hand-eye calibration (AX = XB) using Tsai–Lenz.
//!
//! Provides a linear initialization from paired pose streams, returning the
//! rigid transform between the gripper and camera frames.

use anyhow::Result;
use calib_core::{Iso3, Real};
use log::debug;
use nalgebra::{
    DMatrix, DVector, Isometry3, Matrix3, Quaternion, Translation3, Unit, UnitQuaternion, Vector3,
};

/// Motion pair for Tsai–Lenz AX = XB:
/// A: relative motion in robot/hand chain (base->gripper)
/// B: relative motion in camera/target chain (camera->target)
#[derive(Debug, Clone, Copy)]
pub struct MotionPair {
    pub rot_a: Matrix3<Real>,
    pub rot_b: Matrix3<Real>,
    pub tra_a: Vector3<Real>,
    pub tra_b: Vector3<Real>,
}

/// Linear hand–eye initialisation using the Tsai–Lenz formulation.
#[derive(Debug, Clone, Copy)]
pub struct HandEyeInit;

/// Build a single motion pair from two pose samples.
///
/// base_se3_gripper_*: ^B T_G
/// cam_se3_target_*:   ^T T_C
///
/// A = (^B T_G,a)^(-1) (^B T_G,b)
/// B = (^T T_C,a)^(-1) (^T T_C,b)
fn make_motion_pair(
    base_se3_gripper_a: &Iso3,
    cam_se3_target_a: &Iso3,
    base_se3_gripper_b: &Iso3,
    cam_se3_target_b: &Iso3,
) -> Result<MotionPair> {
    let affine_a = base_se3_gripper_a.inverse() * base_se3_gripper_b;
    let affine_b = cam_se3_target_a.inverse() * cam_se3_target_b;

    let rot_a = project_to_so3(*affine_a.rotation.to_rotation_matrix().matrix())?;
    let rot_b = project_to_so3(*affine_b.rotation.to_rotation_matrix().matrix())?;
    let tra_a = affine_a.translation.vector;
    let tra_b = affine_b.translation.vector;

    Ok(MotionPair {
        rot_a,
        rot_b,
        tra_a,
        tra_b,
    })
}

/// Check if a motion pair is usable:
/// - has sufficient rotation in both chains
/// - optionally rejects near-parallel rotation axes (ill-conditioned)
fn is_good_pair(
    pair: &MotionPair,
    min_angle: Real,
    reject_axis_parallel: bool,
    axis_parallel_eps: Real,
) -> bool {
    let alpha = log_so3(&pair.rot_a);
    let beta = log_so3(&pair.rot_b);
    let norm_a = alpha.norm();
    let norm_b = beta.norm();
    let min_rot = norm_a.min(norm_b);

    if min_rot < min_angle {
        debug!(
            "motion pair rejected: small rotation {:.3} deg",
            min_rot * 180.0 / std::f64::consts::PI
        );
        return false;
    }

    if reject_axis_parallel {
        let aa = if norm_a < 1e-9 { 0.0 } else { 1.0 };
        let bb = if norm_b < 1e-9 { 0.0 } else { 1.0 };
        if aa * bb > 0.0 {
            let sin_axis = (alpha.normalize().cross(&beta.normalize())).norm();
            if sin_axis < axis_parallel_eps {
                debug!("motion pair rejected: near-parallel axes");
                return false;
            }
        }
    }

    true
}

/// Build all valid motion pairs from pose streams.
///
/// `base_se3_gripper` are gripper poses in the base frame, and
/// `cam_se3_target` are camera poses in the target frame.
///
/// Pairs with too-small rotations or near-parallel rotation axes can be
/// rejected to improve conditioning.
pub fn build_all_pairs(
    base_se3_gripper: &[Iso3],
    cam_se3_target: &[Iso3],
    min_angle_deg: Real,        // discard too-small motions
    reject_axis_parallel: bool, // guard against ill-conditioning
    axis_parallel_eps: Real,
) -> Result<Vec<MotionPair>> {
    if base_se3_gripper.len() != cam_se3_target.len() {
        anyhow::bail!(
            "inconsistent hand-eye input sizes: base {} vs cam {}",
            base_se3_gripper.len(),
            cam_se3_target.len()
        );
    }
    if base_se3_gripper.len() < 2 {
        anyhow::bail!("need at least 2 poses, got {}", base_se3_gripper.len());
    }

    let num_poses = base_se3_gripper.len();
    let min_angle = min_angle_deg * std::f64::consts::PI / 180.0;

    let mut pairs = Vec::with_capacity(num_poses * (num_poses - 1) / 2);

    for i in 0..(num_poses - 1) {
        for j in (i + 1)..num_poses {
            let pair = make_motion_pair(
                &base_se3_gripper[i],
                &cam_se3_target[i],
                &base_se3_gripper[j],
                &cam_se3_target[j],
            )?;

            if is_good_pair(&pair, min_angle, reject_axis_parallel, axis_parallel_eps) {
                pairs.push(pair);
            } else {
                debug!("skipping pair ({},{})", i, j);
            }
        }
    }

    if pairs.is_empty() {
        anyhow::bail!("no valid motion pairs after filtering");
    }

    Ok(pairs)
}

// ---------- weighted Tsai–Lenz rotation over all pairs ----------

fn estimate_rotation_allpairs_weighted(pairs: &[MotionPair]) -> Result<Matrix3<Real>> {
    fn quat_left(q: &UnitQuaternion<Real>) -> nalgebra::Matrix4<Real> {
        let w = q.w;
        let (x, y, z) = (q.i, q.j, q.k);
        nalgebra::Matrix4::new(w, -x, -y, -z, x, w, -z, y, y, z, w, -x, z, -y, x, w)
    }

    fn quat_right(q: &UnitQuaternion<Real>) -> nalgebra::Matrix4<Real> {
        let w = q.w;
        let (x, y, z) = (q.i, q.j, q.k);
        nalgebra::Matrix4::new(w, -x, -y, -z, x, w, z, -y, y, -z, w, x, z, y, -x, w)
    }

    let num_pairs = pairs.len();
    let mut m = DMatrix::<Real>::zeros(4 * num_pairs, 4);

    for (idx, p) in pairs.iter().enumerate() {
        let qa = UnitQuaternion::from_rotation_matrix(&nalgebra::Rotation3::from_matrix_unchecked(
            p.rot_a,
        ));
        let qb = UnitQuaternion::from_rotation_matrix(&nalgebra::Rotation3::from_matrix_unchecked(
            p.rot_b,
        ));

        let row_start = 4 * idx;
        m.view_mut((row_start, 0), (4, 4))
            .copy_from(&(quat_left(&qa) - quat_right(&qb)));
    }

    let svd = m.svd(true, true);
    let v_t = svd
        .v_t
        .ok_or_else(|| anyhow::anyhow!("svd failed during hand-eye estimation"))?;
    let q_vec = v_t.row(v_t.nrows() - 1);

    let q = Quaternion::new(q_vec[0], q_vec[1], q_vec[2], q_vec[3]).normalize();
    Ok(UnitQuaternion::from_quaternion(q)
        .to_rotation_matrix()
        .into_inner())
}

// ---------- weighted Tsai–Lenz translation over all pairs ----------

fn estimate_translation_allpairs_weighted(
    pairs: &[MotionPair],
    rot_x: &Matrix3<Real>,
) -> Result<Vector3<Real>> {
    let num_pairs = pairs.len() as i32;
    let mut mat_c = DMatrix::<Real>::zeros(3 * num_pairs as usize, 3);
    let mut vec_w = DVector::<Real>::zeros(3 * num_pairs as usize);

    for idx in 0..num_pairs {
        let p = &pairs[idx as usize];

        let rot_a = &p.rot_a;
        let tran_a = &p.tra_a;
        let tran_b = &p.tra_b;
        let weight = 1.0;

        mat_c
            .view_mut((3 * idx as usize, 0), (3, 3))
            .copy_from(&(weight * (rot_a - Matrix3::identity())));

        vec_w
            .rows_mut(3 * idx as usize, 3)
            .copy_from(&(weight * (rot_x * tran_b - tran_a)));
    }

    let ridge = 1e-12;
    ridge_llsq(&mat_c, &vec_w, ridge)
}

/// Main linear hand–eye init: Tsai–Lenz with all pairs.
///
/// `base_se3_gripper`: gripper poses in base frame.
/// `camera_se3_target`: camera poses in target frame.
///
/// Returns `X = ^G T_C` (gripper -> camera).
pub fn estimate_handeye_dlt(
    base_se3_gripper: &[Iso3],
    camera_se3_target: &[Iso3],
    min_angle_deg: Real,
) -> Result<Iso3> {
    HandEyeInit::tsai_lenz(base_se3_gripper, camera_se3_target, min_angle_deg)
}

impl HandEyeInit {
    /// Tsai–Lenz hand–eye initialisation over all motion pairs.
    ///
    /// `min_angle_deg` controls the minimum motion magnitude used to build
    /// pairs; this helps reject ill-conditioned data.
    pub fn tsai_lenz(
        base_se3_gripper: &[Iso3],
        camera_se3_target: &[Iso3],
        min_angle_deg: Real,
    ) -> Result<Iso3> {
        let pairs = build_all_pairs(
            base_se3_gripper,
            camera_se3_target,
            min_angle_deg,
            true, // reject_axis_parallel
            1e-3, // axis_parallel_eps
        )?;

        let rot_x = estimate_rotation_allpairs_weighted(&pairs)?;
        let g_tra_c = estimate_translation_allpairs_weighted(&pairs, &rot_x)?;

        let rot = UnitQuaternion::from_rotation_matrix(
            &nalgebra::Rotation3::from_matrix_unchecked(rot_x),
        );
        let trans = Translation3::from(g_tra_c);
        Ok(Isometry3::from_parts(trans, rot))
    }
}

/// Project a general 3x3 matrix to the closest rotation matrix (SO(3))
/// using SVD.
fn project_to_so3(m: Matrix3<Real>) -> Result<Matrix3<Real>> {
    let svd = m.svd(true, true);
    let u = svd
        .u
        .ok_or_else(|| anyhow::anyhow!("svd failed during hand-eye estimation"))?;
    let v_t = svd
        .v_t
        .ok_or_else(|| anyhow::anyhow!("svd failed during hand-eye estimation"))?;
    let mut r = u * v_t;

    // Ensure det(R) > 0
    if r.determinant() < 0.0 {
        let mut u_flipped = u;
        u_flipped.column_mut(2).neg_mut();
        r = u_flipped * v_t;
    }
    Ok(r)
}

/// log: SO(3) -> so(3) as a 3-vector (axis * angle)
fn log_so3(r: &Matrix3<Real>) -> Vector3<Real> {
    let rot = UnitQuaternion::from_rotation_matrix(&nalgebra::Rotation3::from_matrix_unchecked(*r));
    let angle = rot.angle();
    if angle < 1e-12 {
        return Vector3::zeros();
    }
    let axis: Unit<Vector3<Real>> = rot
        .axis()
        .unwrap_or_else(|| Unit::new_unchecked(Vector3::x_axis().into_inner()));
    axis.into_inner() * angle
}

/// Ridge-regularized least squares:
/// min ||A x - b||^2 + λ ||x||^2
fn ridge_llsq(a: &DMatrix<Real>, b: &DVector<Real>, lambda: Real) -> Result<Vector3<Real>> {
    let m = a.nrows();
    let n = a.ncols(); // should be 3

    // Build augmented system [A; sqrt(λ) I] x ≈ [b; 0]
    if n != 3 {
        anyhow::bail!("linear solve failed during hand-eye estimation");
    }

    let mut a_aug = DMatrix::<Real>::zeros(m + n, n);
    a_aug.view_mut((0, 0), (m, n)).copy_from(a);

    let sqrt_lambda = lambda.sqrt();
    for i in 0..n {
        a_aug[(m + i, i)] = sqrt_lambda;
    }

    let mut b_aug = DVector::<Real>::zeros(m + n);
    b_aug.rows_mut(0, m).copy_from(b);

    let svd = a_aug.svd(true, true);
    let x = svd
        .solve(&b_aug, 1e-12)
        .map_err(|_| anyhow::anyhow!("linear solve failed during hand-eye estimation"))?;

    Ok(Vector3::new(x[0], x[1], x[2]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Rotation3;

    fn make_iso(angles: (Real, Real, Real), t: (Real, Real, Real)) -> Iso3 {
        let rot = Rotation3::from_euler_angles(angles.0, angles.1, angles.2);
        let tr = Translation3::new(t.0, t.1, t.2);
        Isometry3::from_parts(tr, rot.into())
    }

    /// Compare two SE(3) poses via translation norm + rotation angle
    fn pose_error(a: &Iso3, b: &Iso3) -> (Real, Real) {
        let dt = (a.translation.vector - b.translation.vector).norm();

        let r_a = a.rotation.to_rotation_matrix();
        let r_b = b.rotation.to_rotation_matrix();
        let r_diff = r_a.transpose() * r_b;
        let trace = r_diff.matrix().trace();
        let cos_theta = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0);
        let angle = cos_theta.acos();

        (dt, angle)
    }

    #[test]
    fn handeye_dlt_recovers_ground_truth() {
        // Ground-truth hand-eye (gripper -> camera) and base->target
        let x_gt = make_iso((0.2, -0.1, 0.05), (0.1, -0.05, 0.2)); // ^G T_C
        let y_gt = make_iso((-0.1, 0.05, 0.2), (-0.2, 0.1, 1.0)); // ^B T_T

        // Generate synthetic poses
        let num_poses = 6;
        let mut base_se3_gripper = Vec::with_capacity(num_poses);
        let mut camera_se3_target = Vec::with_capacity(num_poses);

        for k in 0..num_poses {
            let kf = k as Real;
            // Some mildly varying robot poses (base->gripper)
            let bg = make_iso(
                (0.05 * kf, -0.03 * kf, 0.02 * kf),
                (0.1 * kf, -0.05 * kf, 0.8 + 0.05 * kf),
            );
            base_se3_gripper.push(bg);

            // From ^B T_G * X = Y ^C T_T  =>  ^C T_T = Y^{-1} ^B T_G X
            let ct = y_gt.inverse() * bg * x_gt;
            camera_se3_target.push(ct);
        }

        let x_est = estimate_handeye_dlt(&base_se3_gripper, &camera_se3_target, 1.0).unwrap();

        let (dt, ang) = pose_error(&x_est, &x_gt);
        println!("handeye dt = {}, ang = {} rad", dt, ang);

        // Tolerances can be relaxed if you later add noise.
        assert!(dt < 1e-6, "translation error too large: {}", dt);
        assert!(ang < 1e-6, "rotation error too large: {}", ang);
    }
}
