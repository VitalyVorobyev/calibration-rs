//! Closed-form hand-eye calibration from robot poses and camera poses.
//!
//! Every view `i` pairs a measured robot pose `T_B_G_i` (`base_se3_gripper`)
//! with the pose of a planar target seen by a camera, `T_C_T_i`
//! (`cam_se3_target`). Two rigid transforms are unknown and constant over
//! the views:
//!
//! - **eye-in-hand** (camera on the gripper): `gripper_se3_cam` and
//!   `base_se3_target`, with `T_B_G_i · T_G_C = T_B_T · T_T_C_i`;
//! - **eye-to-hand** (target on the gripper, camera fixed):
//!   `gripper_se3_target` and `cam_se3_base`, with
//!   `T_B_G_i · T_G_T = T_B_C · T_C_T_i`.
//!
//! Both are the robot-world equation `A_i X = Z B_i` with `A_i = T_B_G_i`.
//! [`estimate_eye_in_hand`] and [`estimate_eye_to_hand`] solve it with one of
//! four closed-form methods ([`HandeyeMethod`]). Three eliminate `Z` by
//! forming relative motions between view pairs, `A_ij X = X B_ij`, and close
//! `Z` through the first view; Shah's method solves for `X` and `Z` together
//! from the absolute poses.

use crate::Error;
use log::debug;
use nalgebra::{
    DMatrix, DVector, Isometry3, Matrix2, Matrix3, Matrix4, Quaternion, SMatrix, SVector,
    Translation3, Unit, UnitQuaternion, Vector3, Vector4,
};
use serde::{Deserialize, Serialize};
use vision_calibration_core::linalg::iso_from_rotation;
use vision_calibration_core::{Iso3, Real};

/// The closed-form solver of a hand-eye initialization.
///
/// All four need robot rotations about at least two non-parallel axes; they
/// differ in how they weigh the views and in what they estimate jointly.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum HandeyeMethod {
    /// Tsai and Lenz (1989): over every pair of views, the rotation from the
    /// quaternion form of `R_A R_X = R_X R_B`, then the translation by least
    /// squares. The default.
    #[default]
    TsaiLenz,
    /// Park and Martin (1994): the rotation that best aligns the rotation
    /// vectors (axis · angle) of the camera motions with the robot's, then
    /// the translation by least squares. Weights a pair by its angle.
    ParkMartin,
    /// Daniilidis (1999): rotation and translation together, from the dual
    /// quaternions of the motion pairs.
    Daniilidis,
    /// Shah (2013): the hand-eye transform and the fixed transform at the
    /// other end of the chain together, from the absolute poses (`AX = ZB`)
    /// instead of motion pairs, rotation first (Kronecker form) and then
    /// translation.
    Shah,
}

/// The eye-in-hand initialization: a camera on the gripper, a fixed target.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct EyeInHandEstimate {
    /// The camera in the gripper frame (`T_G_C`).
    pub gripper_se3_cam: Iso3,
    /// The target in the robot base frame (`T_B_T`).
    pub base_se3_target: Iso3,
}

/// The eye-to-hand initialization: a target on the gripper, a fixed camera.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct EyeToHandEstimate {
    /// The target in the gripper frame (`T_G_T`).
    pub gripper_se3_target: Iso3,
    /// The robot base in the camera frame (`T_C_B`).
    pub cam_se3_base: Iso3,
}

/// Eye-in-hand initialization from robot poses and target poses.
///
/// `base_se3_gripper[i]` and `cam_se3_target[i]` are the robot pose and
/// the observed target pose of view `i`. View pairs whose rotation, in either
/// the robot or the camera, is below `min_motion_angle_deg` carry too little
/// rotation to use: the motion-pair methods drop them, and every method
/// requires the remaining ones to rotate the robot about two non-parallel
/// axes. The motion-pair methods derive `base_se3_target` from the first
/// view; Shah's method estimates it over all views.
///
/// # Errors
///
/// [`Error::InvalidInput`] if the pose counts differ or a pose is not
/// finite, [`Error::InsufficientData`] for fewer than 2 views,
/// [`Error::NoValidMotionPairs`] if the motions do not determine the
/// transforms, [`Error::Singular`] if a linear solve fails.
pub fn estimate_eye_in_hand(
    method: HandeyeMethod,
    base_se3_gripper: &[Iso3],
    cam_se3_target: &[Iso3],
    min_motion_angle_deg: Real,
) -> Result<EyeInHandEstimate, Error> {
    check_poses(base_se3_gripper, cam_se3_target)?;
    // T_B_G_i · T_G_C = T_B_T · T_T_C_i: the stream B_i is T_T_C_i.
    let target_se3_cam: Vec<Iso3> = cam_se3_target.iter().map(|t| t.inverse()).collect();
    let (x, z) = solve_ax_zb(
        method,
        base_se3_gripper,
        &target_se3_cam,
        min_motion_angle_deg,
    )?;
    let base_se3_target = z.unwrap_or_else(|| base_se3_gripper[0] * x * cam_se3_target[0]);
    Ok(EyeInHandEstimate {
        gripper_se3_cam: x,
        base_se3_target,
    })
}

/// Eye-to-hand initialization from robot poses and target poses.
///
/// The arguments and the motion filter are those of
/// [`estimate_eye_in_hand`]. The motion-pair methods derive
/// `cam_se3_base` from the first view; Shah's method estimates it over
/// all views.
///
/// # Errors
///
/// As [`estimate_eye_in_hand`].
pub fn estimate_eye_to_hand(
    method: HandeyeMethod,
    base_se3_gripper: &[Iso3],
    cam_se3_target: &[Iso3],
    min_motion_angle_deg: Real,
) -> Result<EyeToHandEstimate, Error> {
    check_poses(base_se3_gripper, cam_se3_target)?;
    // T_B_G_i · T_G_T = T_B_C · T_C_T_i: the stream B_i is T_C_T_i.
    let (x, z) = solve_ax_zb(
        method,
        base_se3_gripper,
        cam_se3_target,
        min_motion_angle_deg,
    )?;
    let cam_se3_base = match z {
        Some(base_se3_cam) => base_se3_cam.inverse(),
        None => cam_se3_target[0] * (base_se3_gripper[0] * x).inverse(),
    };
    Ok(EyeToHandEstimate {
        gripper_se3_target: x,
        cam_se3_base,
    })
}

fn check_poses(base_se3_gripper: &[Iso3], stream: &[Iso3]) -> Result<(), Error> {
    if base_se3_gripper.len() != stream.len() {
        return Err(Error::invalid_input(format!(
            "inconsistent hand-eye input sizes: {} robot poses vs {} target poses",
            base_se3_gripper.len(),
            stream.len()
        )));
    }
    if base_se3_gripper.len() < 2 {
        return Err(Error::InsufficientData {
            need: 2,
            got: base_se3_gripper.len(),
        });
    }
    let finite = |p: &Iso3| {
        p.translation.vector.iter().all(|v| v.is_finite())
            && p.rotation.coords.iter().all(|v| v.is_finite())
    };
    if !base_se3_gripper.iter().chain(stream).all(finite) {
        return Err(Error::invalid_input("a hand-eye input pose is not finite"));
    }
    Ok(())
}

/// Solve `A_i X = Z B_i` for `X`, and for `Z` when the method estimates it.
fn solve_ax_zb(
    method: HandeyeMethod,
    base_se3_gripper: &[Iso3],
    stream: &[Iso3],
    min_angle_deg: Real,
) -> Result<(Iso3, Option<Iso3>), Error> {
    // What the solve needs is a well-posed *set* of pairs; the same set gates
    // Shah's method, whose rotation is undetermined under the same condition.
    let pairs = build_all_pairs(base_se3_gripper, stream, min_angle_deg)?;
    if !robot_axes_span_two_directions(&pairs, 1e-3) {
        return Err(Error::NoValidMotionPairs);
    }
    match method {
        HandeyeMethod::TsaiLenz => {
            let rot_x = rotation_tsai_lenz(&pairs)?;
            Ok((with_pair_translation(&pairs, rot_x)?, None))
        }
        HandeyeMethod::ParkMartin => {
            let rot_x = rotation_park_martin(&pairs)?;
            Ok((with_pair_translation(&pairs, rot_x)?, None))
        }
        HandeyeMethod::Daniilidis => Ok((daniilidis(&pairs)?, None)),
        HandeyeMethod::Shah => {
            let (x, z) = shah(base_se3_gripper, stream)?;
            Ok((x, Some(z)))
        }
    }
}

// ---------- motion pairs ----------

/// Motion pair of `A X = X B`: `A = A_i⁻¹ A_j` (robot), `B = B_i⁻¹ B_j`
/// (observation stream).
#[derive(Debug, Clone, Copy)]
struct MotionPair {
    rot_a: Matrix3<Real>,
    rot_b: Matrix3<Real>,
    tra_a: Vector3<Real>,
    tra_b: Vector3<Real>,
}

/// Every view pair whose rotation in both streams reaches `min_angle_deg`.
///
/// There is no per-pair A/B axis test: A and B share their rotation axis
/// whenever the hand-eye rotation is the identity (a camera bolted square to
/// the flange) or rotates about that axis, and such pairs are perfectly
/// informative.
fn build_all_pairs(
    base_se3_gripper: &[Iso3],
    stream: &[Iso3],
    min_angle_deg: Real,
) -> Result<Vec<MotionPair>, Error> {
    let num_poses = base_se3_gripper.len();
    let min_angle = min_angle_deg * std::f64::consts::PI / 180.0;
    let mut pairs = Vec::with_capacity(num_poses * (num_poses - 1) / 2);
    for i in 0..(num_poses - 1) {
        for j in (i + 1)..num_poses {
            let affine_a = base_se3_gripper[i].inverse() * base_se3_gripper[j];
            let affine_b = stream[i].inverse() * stream[j];
            let pair = MotionPair {
                rot_a: vision_calibration_core::linalg::project_to_so3(
                    affine_a.rotation.to_rotation_matrix().matrix(),
                )?,
                rot_b: vision_calibration_core::linalg::project_to_so3(
                    affine_b.rotation.to_rotation_matrix().matrix(),
                )?,
                tra_a: affine_a.translation.vector,
                tra_b: affine_b.translation.vector,
            };
            let min_rot = log_so3(&pair.rot_a).norm().min(log_so3(&pair.rot_b).norm());
            if min_rot >= min_angle {
                pairs.push(pair);
            } else {
                debug!(
                    "skipping pair ({i},{j}): rotation {:.3} deg",
                    min_rot.to_degrees()
                );
            }
        }
    }
    if pairs.is_empty() {
        return Err(Error::NoValidMotionPairs);
    }
    Ok(pairs)
}

/// Whether the robot motions of `pairs` rotate about at least two axes that
/// are not parallel (sine of the angle between them ≥ `eps`).
///
/// This is the condition for a unique solution: with every rotation axis
/// parallel, the rotation of X about that axis and the translation along it
/// are unobservable, in `AX = XB` and in `AX = ZB` alike.
fn robot_axes_span_two_directions(pairs: &[MotionPair], eps: Real) -> bool {
    let axes: Vec<Vector3<Real>> = pairs
        .iter()
        .map(|p| log_so3(&p.rot_a))
        .filter(|w| w.norm() > 1e-9)
        .map(|w| w.normalize())
        .collect();
    let Some(first) = axes.first() else {
        return false;
    };
    axes.iter().any(|a| first.cross(a).norm() >= eps)
}

/// The unit quaternion of a rotation, with a non-negative scalar part.
///
/// `q` and `-q` are the same rotation, but `q_A q_X = q_X q_B` holds only
/// when `q_A` and `q_B` carry the same sign of their (equal) scalar parts.
/// nalgebra's conversion leaves the sign to its branch choice, which for
/// rotations past 120° depends on the axis, so the two of a pair can
/// disagree.
fn quat_of(r: &Matrix3<Real>) -> UnitQuaternion<Real> {
    let q = UnitQuaternion::from_rotation_matrix(&nalgebra::Rotation3::from_matrix_unchecked(*r));
    if q.w < 0.0 {
        UnitQuaternion::new_unchecked(-q.into_inner())
    } else {
        q
    }
}

/// The quaternions of a pair, `q_B` signed so that `q_A = q_X q_B q_X⁻¹`
/// holds for a rotation `q_x` near the solution.
///
/// Away from 180° this is the sign [`quat_of`] gives, for any `q_x` within
/// 90° of the truth. Within noise of 180° both scalar parts are at the level
/// of the noise, and their signs say nothing about the pair.
fn sign_matched_quats(
    p: &MotionPair,
    q_x: &UnitQuaternion<Real>,
) -> (UnitQuaternion<Real>, UnitQuaternion<Real>) {
    let (qa, qb) = (quat_of(&p.rot_a), quat_of(&p.rot_b));
    if qa.coords.dot(&(q_x * qb * q_x.inverse()).coords) < 0.0 {
        (qa, UnitQuaternion::new_unchecked(-qb.into_inner()))
    } else {
        (qa, qb)
    }
}

// ---------- rotation, then translation (Tsai–Lenz, Park–Martin) ----------

/// Rotation of `R_A R_X = R_X R_B` from the stacked quaternion constraints
/// `[L(q_A) − R(q_B)] q_X = 0`.
fn rotation_tsai_lenz(pairs: &[MotionPair]) -> Result<Matrix3<Real>, Error> {
    Ok(quat_tsai_lenz(pairs)?.to_rotation_matrix().into_inner())
}

/// [`rotation_tsai_lenz`] as a quaternion.
///
/// A pair within noise of 180° can enter with `q_B` of the wrong sign, as
/// `q_A q_X = −q_X q_B`. At 180° exactly the true `q_X` is still an
/// eigenvector of that pair's block, so the pair raises the smallest
/// eigenvalue without moving its eigenvector, and the solution holds while
/// the consistent pairs outnumber such pairs.
fn quat_tsai_lenz(pairs: &[MotionPair]) -> Result<UnitQuaternion<Real>, Error> {
    fn quat_left(q: &UnitQuaternion<Real>) -> Matrix4<Real> {
        let (w, x, y, z) = (q.w, q.i, q.j, q.k);
        Matrix4::new(w, -x, -y, -z, x, w, -z, y, y, z, w, -x, z, -y, x, w)
    }
    fn quat_right(q: &UnitQuaternion<Real>) -> Matrix4<Real> {
        let (w, x, y, z) = (q.w, q.i, q.j, q.k);
        Matrix4::new(w, -x, -y, -z, x, w, z, -y, y, -z, w, x, z, y, -x, w)
    }

    let mut m = DMatrix::<Real>::zeros(4 * pairs.len(), 4);
    for (idx, p) in pairs.iter().enumerate() {
        let (qa, qb) = (quat_of(&p.rot_a), quat_of(&p.rot_b));
        m.view_mut((4 * idx, 0), (4, 4))
            .copy_from(&(quat_left(&qa) - quat_right(&qb)));
    }
    // Smallest right-singular vector of the `4N×4` system via `AᵀA` symmetric
    // eigen — avoids nalgebra's hang-prone dense SVD.
    let q_vec = vision_calibration_core::linalg::null_space(&m)?.vector;
    let q = Quaternion::new(q_vec[0], q_vec[1], q_vec[2], q_vec[3]).normalize();
    Ok(UnitQuaternion::from_quaternion(q))
}

/// Rotation of `R_A = R_X R_B R_Xᵀ` from the rotation vectors: `α = R_X β`
/// for `α = log R_A`, `β = log R_B`, so `R_X` is the rotation nearest to
/// `Σ α βᵀ` (orthogonal Procrustes).
fn rotation_park_martin(pairs: &[MotionPair]) -> Result<Matrix3<Real>, Error> {
    let mut h = Matrix3::<Real>::zeros();
    for p in pairs {
        h += log_so3(&p.rot_a) * log_so3(&p.rot_b).transpose();
    }
    Ok(vision_calibration_core::linalg::project_to_so3(&h)?)
}

/// `X` from its rotation and the stacked translation constraints
/// `(R_A − I) t_X = R_X t_B − t_A`.
fn with_pair_translation(pairs: &[MotionPair], rot_x: Matrix3<Real>) -> Result<Iso3, Error> {
    let mut mat_c = DMatrix::<Real>::zeros(3 * pairs.len(), 3);
    let mut vec_w = DVector::<Real>::zeros(3 * pairs.len());
    for (idx, p) in pairs.iter().enumerate() {
        mat_c
            .view_mut((3 * idx, 0), (3, 3))
            .copy_from(&(p.rot_a - Matrix3::identity()));
        vec_w
            .rows_mut(3 * idx, 3)
            .copy_from(&(rot_x * p.tra_b - p.tra_a));
    }
    let t = crate::math::ridge_lstsq(&mat_c, &vec_w, 1e-12)?;
    Ok(iso_from_rotation(&rot_x, &Vector3::new(t[0], t[1], t[2])))
}

// ---------- Daniilidis: dual quaternions ----------

/// `X` from the dual-quaternion form of `A X = X B`.
///
/// With `q̌ = q + ε q'`, `q' = ½ t q`, each pair gives six linear equations
/// in the eight entries of `(q_X, q'_X)`:
///
/// ```text
/// [ a − b    [a + b]×    0       0      ] (q_X )
/// [ a' − b'  [a' + b']×  a − b   [a + b]×] (q'_X) = 0
/// ```
///
/// (`a`, `b`, `a'`, `b'` the vector parts of the pair's real and dual
/// quaternions, sign-matched so `q_A = q_X q_B q_X⁻¹`). The solution is
/// the two-dimensional null space `λ₁ v₁ + λ₂ v₂` of the stack, fixed by the
/// unit-norm and `q · q' = 0` constraints. Translations are divided by their
/// RMS first so the dual rows weigh like the real ones; the solution is
/// rescaled after.
///
/// The signs are matched through the Tsai–Lenz rotation
/// ([`sign_matched_quats`]), not by the scalar parts alone: within noise of
/// 180° those vanish, and a pair entering with mismatched signs moves this
/// solution (by 1° and 5 mm from two such pairs among 91, noise-free) where
/// it leaves Tsai–Lenz's in place.
fn daniilidis(pairs: &[MotionPair]) -> Result<Iso3, Error> {
    let q_ref = quat_tsai_lenz(pairs)?;
    let rms = (pairs
        .iter()
        .map(|p| p.tra_a.norm_squared() + p.tra_b.norm_squared())
        .sum::<Real>()
        / (2 * pairs.len()) as Real)
        .sqrt();
    let scale = if rms > 0.0 && rms.is_finite() {
        rms
    } else {
        1.0
    };

    let mut normal = SMatrix::<Real, 8, 8>::zeros();
    for p in pairs {
        let (qa, qb) = sign_matched_quats(p, &q_ref);
        let dual = |t: &Vector3<Real>, q: &UnitQuaternion<Real>| {
            Quaternion::from_imag(0.5 * t / scale) * q.into_inner()
        };
        let (qa_d, qb_d) = (dual(&p.tra_a, &qa), dual(&p.tra_b, &qb));
        let (a, b) = (qa.imag(), qb.imag());
        let (ad, bd) = (qa_d.imag(), qb_d.imag());

        let mut s = SMatrix::<Real, 6, 8>::zeros();
        s.fixed_view_mut::<3, 1>(0, 0).copy_from(&(a - b));
        s.fixed_view_mut::<3, 3>(0, 1)
            .copy_from(&(a + b).cross_matrix());
        s.fixed_view_mut::<3, 1>(3, 0).copy_from(&(ad - bd));
        s.fixed_view_mut::<3, 3>(3, 1)
            .copy_from(&(ad + bd).cross_matrix());
        s.fixed_view_mut::<3, 1>(3, 4).copy_from(&(a - b));
        s.fixed_view_mut::<3, 3>(3, 5)
            .copy_from(&(a + b).cross_matrix());
        normal += s.transpose() * s;
    }
    if normal.iter().any(|v| !v.is_finite()) {
        return Err(Error::Singular);
    }
    let eigen = normal.symmetric_eigen();
    let mut order: [usize; 8] = [0, 1, 2, 3, 4, 5, 6, 7];
    order.sort_by(|&i, &j| eigen.eigenvalues[i].total_cmp(&eigen.eigenvalues[j]));
    let v1: SVector<Real, 8> = eigen.eigenvectors.column(order[0]).into_owned();
    let v2: SVector<Real, 8> = eigen.eigenvectors.column(order[1]).into_owned();
    let (u1, w1) = (v1.fixed_rows::<4>(0), v1.fixed_rows::<4>(4));
    let (u2, w2) = (v2.fixed_rows::<4>(0), v2.fixed_rows::<4>(4));

    // q · q' = 0 for q = λ₁u₁ + λ₂u₂, q' = λ₁w₁ + λ₂w₂ is the quadratic form
    // λᵀ C λ = 0. Its zero directions are √μ₂ e₁ ± √(−μ₁) e₂ for the
    // eigenpairs (μ₁ ≤ μ₂, e₁, e₂) of C; noise can make C definite, and then
    // the direction where |λᵀ C λ| is smallest is taken and q' orthogonalized.
    let c12 = 0.5 * (u1.dot(&w2) + u2.dot(&w1));
    let c = Matrix2::new(u1.dot(&w1), c12, c12, u2.dot(&w2));
    let ce = c.symmetric_eigen();
    let (lo, hi) = if ce.eigenvalues[0] <= ce.eigenvalues[1] {
        (0, 1)
    } else {
        (1, 0)
    };
    let (mu_lo, mu_hi) = (ce.eigenvalues[lo], ce.eigenvalues[hi]);
    let (e_lo, e_hi) = (ce.eigenvectors.column(lo), ce.eigenvectors.column(hi));
    let candidates: Vec<nalgebra::Vector2<Real>> = if mu_lo <= 0.0 && mu_hi >= 0.0 {
        let (p, n) = (mu_hi.sqrt(), (-mu_lo).sqrt());
        vec![p * e_lo + n * e_hi, p * e_lo - n * e_hi]
    } else if mu_lo.abs() <= mu_hi.abs() {
        vec![e_lo.into_owned()]
    } else {
        vec![e_hi.into_owned()]
    };
    let q_of = |l: &nalgebra::Vector2<Real>| -> Vector4<Real> { l[0] * u1 + l[1] * u2 };
    let best = candidates
        .iter()
        .max_by(|a, b| q_of(a).norm_squared().total_cmp(&q_of(b).norm_squared()))
        .ok_or(Error::Singular)?;
    let q_vec = q_of(best);
    let q_norm = q_vec.norm();
    if !q_norm.is_finite() || q_norm <= 1e-12 {
        return Err(Error::Singular);
    }
    let q_vec = q_vec / q_norm;
    let qd_vec: Vector4<Real> = (best[0] * w1 + best[1] * w2) / q_norm;
    let qd_vec = qd_vec - q_vec * q_vec.dot(&qd_vec);

    // nalgebra stores quaternions as (i, j, k, w); the solve's order is (w, i, j, k).
    let q =
        UnitQuaternion::from_quaternion(Quaternion::new(q_vec[0], q_vec[1], q_vec[2], q_vec[3]));
    let qd = Quaternion::new(qd_vec[0], qd_vec[1], qd_vec[2], qd_vec[3]);
    // q' = ½ t q  ⇒  t = 2 q' q*.
    let t = (qd * q.conjugate().into_inner()).imag() * (2.0 * scale);
    if t.iter().any(|v| !v.is_finite()) {
        return Err(Error::Singular);
    }
    Ok(Isometry3::from_parts(Translation3::from(t), q))
}

// ---------- Shah: AX = ZB from the absolute poses ----------

/// `X` and `Z` of `A_i X = Z B_i`.
///
/// Rotation: `R_A R_X = R_Z R_B` is linear in `(vec R_X, vec R_Z)`,
/// `[I ⊗ R_A, −R_Bᵀ ⊗ I] (vec R_X; vec R_Z) = 0` (column-major `vec`); the
/// null vector is scaled to `det R_X = 1` and both blocks are projected onto
/// SO(3). Translation: `[R_A, −I] (t_X; t_Z) = R_Z t_B − t_A`, by least
/// squares.
fn shah(base_se3_gripper: &[Iso3], stream: &[Iso3]) -> Result<(Iso3, Iso3), Error> {
    let n = base_se3_gripper.len();
    let rotations = |p: &Iso3| p.rotation.to_rotation_matrix().into_inner();

    let mut k = DMatrix::<Real>::zeros(9 * n, 18);
    for (i, (a, b)) in base_se3_gripper.iter().zip(stream).enumerate() {
        let (ra, rb) = (rotations(a), rotations(b));
        // (I ⊗ R_A): block (c, c) = R_A. (R_Bᵀ ⊗ I): block (r, c) = R_B[(c, r)] I.
        for blk in 0..3 {
            k.view_mut((9 * i + 3 * blk, 3 * blk), (3, 3))
                .copy_from(&ra);
        }
        for r in 0..3 {
            for c in 0..3 {
                let v = -rb[(c, r)];
                for d in 0..3 {
                    k[(9 * i + 3 * r + d, 9 + 3 * c + d)] = v;
                }
            }
        }
    }
    let v = vision_calibration_core::linalg::null_space(&k)?.vector;
    let unvec = |off: usize| Matrix3::from_fn(|r, c| v[off + 3 * c + r]);
    let (mx, mz) = (unvec(0), unvec(9));
    let det = mx.determinant();
    if !det.is_finite() || det.abs() <= 1e-12 {
        return Err(Error::Singular);
    }
    let alpha = det.signum() / det.abs().cbrt();
    let rot_x = vision_calibration_core::linalg::project_to_so3(&(alpha * mx))?;
    let rot_z = vision_calibration_core::linalg::project_to_so3(&(alpha * mz))?;

    let mut c = DMatrix::<Real>::zeros(3 * n, 6);
    let mut d = DVector::<Real>::zeros(3 * n);
    for (i, (a, b)) in base_se3_gripper.iter().zip(stream).enumerate() {
        c.view_mut((3 * i, 0), (3, 3)).copy_from(&rotations(a));
        c.view_mut((3 * i, 3), (3, 3))
            .copy_from(&(-Matrix3::<Real>::identity()));
        d.rows_mut(3 * i, 3)
            .copy_from(&(rot_z * b.translation.vector - a.translation.vector));
    }
    let t = crate::math::ridge_lstsq(&c, &d, 1e-12)?;
    Ok((
        iso_from_rotation(&rot_x, &Vector3::new(t[0], t[1], t[2])),
        iso_from_rotation(&rot_z, &Vector3::new(t[3], t[4], t[5])),
    ))
}

/// log: SO(3) → so(3) as a 3-vector (axis · angle).
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

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Rotation3;

    const METHODS: [HandeyeMethod; 4] = [
        HandeyeMethod::TsaiLenz,
        HandeyeMethod::ParkMartin,
        HandeyeMethod::Daniilidis,
        HandeyeMethod::Shah,
    ];

    fn make_iso(angles: (Real, Real, Real), t: (Real, Real, Real)) -> Iso3 {
        let rot = Rotation3::from_euler_angles(angles.0, angles.1, angles.2);
        Isometry3::from_parts(Translation3::new(t.0, t.1, t.2), rot.into())
    }

    /// (Δtranslation, Δrotation angle) between two poses.
    fn pose_error(a: &Iso3, b: &Iso3) -> (Real, Real) {
        let dt = (a.translation.vector - b.translation.vector).norm();
        (dt, a.rotation.angle_to(&b.rotation))
    }

    /// Eye-in-hand views: `T_C_T_i = (T_B_G_i · X)⁻¹ · Z`.
    fn eye_in_hand_views(x: &Iso3, z: &Iso3, robot: &[Iso3]) -> Vec<Iso3> {
        robot.iter().map(|bg| (bg * x).inverse() * z).collect()
    }

    fn varied_robot_poses() -> Vec<Iso3> {
        (0..8)
            .map(|k| {
                let kf = k as Real;
                make_iso(
                    (
                        0.25 * (kf * 0.9).sin(),
                        0.2 * (kf * 1.3).cos(),
                        0.3 * (kf * 0.7).sin(),
                    ),
                    (0.1 * kf.sin(), -0.05 * kf, 0.8 + 0.03 * kf),
                )
            })
            .collect()
    }

    fn assert_recovers(method: HandeyeMethod, x: &Iso3, z: &Iso3, robot: &[Iso3], tol: Real) {
        let cams = eye_in_hand_views(x, z, robot);
        let est = estimate_eye_in_hand(method, robot, &cams, 1.0).unwrap();
        let (dt, dr) = pose_error(&est.gripper_se3_cam, x);
        assert!(
            dt < tol && dr < tol,
            "{method:?}: X off by {dt} m, {dr} rad"
        );
        let (dt, dr) = pose_error(&est.base_se3_target, z);
        assert!(
            dt < tol && dr < tol,
            "{method:?}: Z off by {dt} m, {dr} rad"
        );
    }

    #[test]
    fn every_method_recovers_exact_eye_in_hand() {
        let x = make_iso((0.2, -0.1, 0.05), (0.1, -0.05, 0.2));
        let z = make_iso((-0.1, 0.05, 0.2), (-0.2, 0.1, 1.0));
        for method in METHODS {
            assert_recovers(method, &x, &z, &varied_robot_poses(), 1e-9);
        }
    }

    #[test]
    fn every_method_recovers_exact_eye_to_hand() {
        // T_C_T_i = T_C_B · T_B_G_i · T_G_T.
        let gripper_se3_target = make_iso((0.3, 0.1, -0.2), (0.0, 0.05, 0.12));
        let cam_se3_base = make_iso((2.6, -0.1, 0.3), (0.2, -0.1, 1.4));
        let robot = varied_robot_poses();
        let cams: Vec<Iso3> = robot
            .iter()
            .map(|bg| cam_se3_base * bg * gripper_se3_target)
            .collect();
        for method in METHODS {
            let est = estimate_eye_to_hand(method, &robot, &cams, 1.0).unwrap();
            let (dt, dr) = pose_error(&est.gripper_se3_target, &gripper_se3_target);
            assert!(
                dt < 1e-9 && dr < 1e-9,
                "{method:?}: X off by {dt} m, {dr} rad"
            );
            let (dt, dr) = pose_error(&est.cam_se3_base, &cam_se3_base);
            assert!(
                dt < 1e-9 && dr < 1e-9,
                "{method:?}: Z off by {dt} m, {dr} rad"
            );
        }
    }

    /// A camera mounted square to the flange (identity rotation, pure offset)
    /// makes every pair's A and B axes parallel, which is not degenerate: a
    /// per-pair parallel-axis filter would reject them all.
    #[test]
    fn every_method_recovers_an_identity_rotation_mount() {
        let x = make_iso((0.0, 0.0, 0.0), (0.04, 0.0, 0.03));
        let z = make_iso((0.0, 0.0, 0.3), (0.45, 0.15, 0.0));
        for method in METHODS {
            assert_recovers(method, &x, &z, &varied_robot_poses(), 1e-9);
        }
    }

    /// Also when X rotates about an axis some robot motions share.
    #[test]
    fn every_method_recovers_a_mount_rotated_about_a_motion_axis() {
        let x = make_iso((0.0, 0.0, 0.7), (0.02, -0.01, 0.05));
        let z = make_iso((0.1, -0.05, 0.2), (-0.2, 0.1, 1.0));
        let mut robot = varied_robot_poses();
        // Two poses differing by a pure rotation about z: their pair's A and B
        // axes coincide.
        robot.push(robot[0] * make_iso((0.0, 0.0, 0.5), (0.0, 0.0, 0.0)));
        for method in METHODS {
            assert_recovers(method, &x, &z, &robot, 1e-9);
        }
    }

    /// Relative rotations past 120°, where nalgebra may hand the two
    /// quaternions of a pair opposite signs. Before the signs were matched,
    /// such a pair entered the quaternion solves as `q_A q_X = −q_X q_B`:
    /// Tsai–Lenz returned a rotation 180° off from three views, and
    /// Daniilidis was off from five.
    #[test]
    fn every_method_recovers_motions_past_120_degrees() {
        let x = make_iso((0.4, -0.3, 0.9), (0.05, 0.02, 0.1));
        let z = make_iso((0.2, 0.1, -0.4), (0.3, -0.2, 0.9));
        let robot = [
            make_iso((0.0, 0.0, 0.0), (0.0, 0.0, 0.8)),
            make_iso((2.3, 0.4, -0.3), (0.1, 0.0, 0.7)),
            make_iso((-0.5, 2.4, 0.6), (-0.1, 0.1, 0.9)),
            make_iso((0.7, -0.6, 2.5), (0.0, -0.1, 0.8)),
            make_iso((-2.2, -0.8, 1.9), (0.05, 0.05, 0.75)),
        ];
        for views in [3, 5] {
            for method in METHODS {
                assert_recovers(method, &x, &z, &robot[..views], 1e-9);
            }
        }
    }

    /// Every robot motion about one axis leaves X underdetermined: rejected
    /// by every method.
    #[test]
    fn every_method_rejects_motions_about_a_single_axis() {
        let x = make_iso((0.2, -0.1, 0.05), (0.1, -0.05, 0.2));
        let z = make_iso((-0.1, 0.05, 0.2), (-0.2, 0.1, 1.0));
        let robot: Vec<Iso3> = (0..6)
            .map(|k| make_iso((0.0, 0.0, 0.3 * k as Real), (0.1, 0.0, 0.8)))
            .collect();
        let cams = eye_in_hand_views(&x, &z, &robot);
        for method in METHODS {
            assert!(
                matches!(
                    estimate_eye_in_hand(method, &robot, &cams, 1.0),
                    Err(Error::NoValidMotionPairs)
                ),
                "{method:?}"
            );
        }
    }

    /// Pure translations carry no rotation: every pair is below the angle
    /// filter, a typed error rather than a NaN.
    #[test]
    fn every_method_rejects_translation_only_motion() {
        let x = make_iso((0.2, -0.1, 0.05), (0.1, -0.05, 0.2));
        let z = make_iso((-0.1, 0.05, 0.2), (-0.2, 0.1, 1.0));
        let robot: Vec<Iso3> = (0..6)
            .map(|k| make_iso((0.1, 0.2, 0.3), (0.1 * k as Real, 0.02, 0.8)))
            .collect();
        let cams = eye_in_hand_views(&x, &z, &robot);
        for method in METHODS {
            assert!(
                matches!(
                    estimate_eye_in_hand(method, &robot, &cams, 1.0),
                    Err(Error::NoValidMotionPairs)
                ),
                "{method:?}"
            );
        }
    }

    #[test]
    fn invalid_input_is_a_typed_error() {
        let robot = varied_robot_poses();
        let cams = eye_in_hand_views(&Iso3::identity(), &Iso3::identity(), &robot);
        for method in METHODS {
            assert!(matches!(
                estimate_eye_in_hand(method, &robot[..3], &cams, 1.0),
                Err(Error::InvalidInput { .. })
            ));
            assert!(matches!(
                estimate_eye_to_hand(method, &robot[..1], &cams[..1], 1.0),
                Err(Error::InsufficientData { need: 2, got: 1 })
            ));
            let mut bad = cams.clone();
            bad[2].translation.vector.x = Real::NAN;
            assert!(matches!(
                estimate_eye_in_hand(method, &robot, &bad, 1.0),
                Err(Error::InvalidInput { .. })
            ));
        }
    }

    /// Small rotations with noise: every method returns a finite estimate
    /// (worse, not NaN) once the motions clear the angle filter.
    #[test]
    fn rotation_poor_noisy_motion_stays_finite() {
        let x = make_iso((0.2, -0.1, 0.05), (0.1, -0.05, 0.2));
        let z = make_iso((-0.1, 0.05, 0.2), (-0.2, 0.1, 1.0));
        let robot: Vec<Iso3> = (0..10)
            .map(|k| {
                let kf = k as Real;
                make_iso(
                    (
                        0.03 * (kf * 1.7).sin(),
                        0.03 * (kf * 2.3).cos(),
                        0.03 * kf.sin(),
                    ),
                    (0.05 * kf.cos(), 0.04 * kf.sin(), 0.8),
                )
            })
            .collect();
        let cams: Vec<Iso3> = eye_in_hand_views(&x, &z, &robot)
            .iter()
            .enumerate()
            .map(|(k, c)| {
                let kf = k as Real;
                make_iso(
                    (1e-3 * kf.sin(), -1e-3 * kf.cos(), 5e-4),
                    (1e-3 * kf.cos(), 0.0, -1e-3 * kf.sin()),
                ) * c
            })
            .collect();
        for method in METHODS {
            let est = estimate_eye_in_hand(method, &robot, &cams, 0.5).unwrap();
            let p = est.gripper_se3_cam;
            assert!(
                p.translation.vector.iter().all(|v| v.is_finite())
                    && p.rotation.coords.iter().all(|v| v.is_finite()),
                "{method:?}"
            );
        }
    }

    #[test]
    fn method_serde_is_snake_case() {
        assert_eq!(
            serde_json::to_string(&HandeyeMethod::ParkMartin).unwrap(),
            "\"park_martin\""
        );
        let m: HandeyeMethod = serde_json::from_str("\"tsai_lenz\"").unwrap();
        assert_eq!(m, HandeyeMethod::default());
    }

    /// Deterministic standard normal samples: SplitMix64 and Box–Muller.
    struct Gauss(u64);

    impl Gauss {
        fn uniform(&mut self) -> Real {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            ((z ^ (z >> 31)) >> 11) as Real / (1u64 << 53) as Real
        }

        fn sample(&mut self) -> Real {
            let u1 = self.uniform().max(Real::MIN_POSITIVE);
            let u2 = self.uniform();
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        }

        fn vector(&mut self, sigma: Real) -> Vector3<Real> {
            Vector3::new(self.sample(), self.sample(), self.sample()) * sigma
        }

        /// `p` moved by a random rotation and translation of per-axis
        /// standard deviations `rot_sigma` (rad) and `t_sigma`.
        fn perturb(&mut self, p: &Iso3, rot_sigma: Real, t_sigma: Real) -> Iso3 {
            let rot = UnitQuaternion::from_scaled_axis(self.vector(rot_sigma));
            Isometry3::from_parts(Translation3::from(self.vector(t_sigma)), rot) * p
        }
    }

    /// Eight stations rotated by up to 35°, then one partner per entry of
    /// `turns_deg`: station `k` turned by `turns_deg[k]` degrees about an
    /// axis of the gripper (the flange axis first), as a wrist flip does.
    fn stations_with_turns(turns_deg: &[Real]) -> Vec<Iso3> {
        let mut robot: Vec<Iso3> = (0..8)
            .map(|k| {
                let kf = k as Real;
                make_iso(
                    (
                        0.5 * (kf * 0.9).sin(),
                        0.4 * (kf * 1.3).cos(),
                        0.6 * (kf * 0.7).sin(),
                    ),
                    (0.1 * kf.sin(), -0.05 * kf, 0.8 + 0.03 * kf),
                )
            })
            .collect();
        let axes = [
            Vector3::z(),
            Vector3::x(),
            Vector3::new(0.3, 0.2, 1.0),
            Vector3::y(),
            Vector3::new(1.0, -1.0, 0.4),
            Vector3::new(-0.2, 0.1, 1.0),
        ];
        for (k, (&deg, axis)) in turns_deg.iter().zip(axes).enumerate() {
            let rot = UnitQuaternion::from_axis_angle(&Unit::new_normalize(axis), deg.to_radians());
            let turn = Isometry3::from_parts(Translation3::new(0.02, -0.01, 0.03), rot);
            robot.push(robot[k] * turn);
        }
        robot
    }

    /// Relative motions of exactly 180°: both quaternions of such a pair
    /// have a zero scalar part, so `w ≥ 0` cannot match their signs and
    /// nalgebra's branch choice decides. Two of this scene's 91 pairs come
    /// out mismatched; Tsai–Lenz and Park–Martin absorb them (the true
    /// rotation stays their solution), Daniilidis's dual-quaternion rows did
    /// not (1.03° and 5.0 mm off before its signs were matched).
    #[test]
    fn every_method_recovers_half_turn_motions() {
        let x = make_iso((0.4, -0.3, 0.9), (0.05, 0.02, 0.1));
        let z = make_iso((0.2, 0.1, -0.4), (0.3, -0.2, 0.9));
        let robot = stations_with_turns(&[180.0; 6]);
        for method in METHODS {
            assert_recovers(method, &x, &z, &robot, 1e-9);
            let cams: Vec<Iso3> = robot.iter().map(|bg| z * bg * x).collect();
            let est = estimate_eye_to_hand(method, &robot, &cams, 1.0).unwrap();
            let (dt, dr) = pose_error(&est.gripper_se3_target, &x);
            assert!(
                dt < 1e-9 && dr < 1e-9,
                "{method:?} eye-to-hand: X off by {dt} m, {dr} rad"
            );
        }
    }

    /// Wrist flips within noise of 180° (170°–180°) must not make any method
    /// worse than the same scene with quarter turns (85°–90°) in their
    /// place. Over 40 noisy draws (camera 0.2° and 1 mm, robot 0.04° and
    /// 0.2 mm per axis) the largest rotation and translation errors of `X`
    /// stay within 1.5× those of the quarter-turn scene, eye-in-hand and
    /// eye-to-hand.
    #[test]
    fn near_half_turn_motions_keep_every_method_accurate() {
        let x = make_iso((0.4, -0.3, 0.9), (0.05, 0.02, 0.1));
        let z = make_iso((0.2, 0.1, -0.4), (0.3, -0.2, 0.9));
        let half_turns = [170.0, 178.0, 179.0, 179.5, 179.9, 180.0];
        let quarter_turns = half_turns.map(|d| 0.5 * d);
        // The largest (rotation deg, translation mm) error of X over the draws.
        let worst = |method: HandeyeMethod, turns: &[Real], eye_to_hand: bool| {
            let robot = stations_with_turns(turns);
            let (mut rot, mut tra): (Real, Real) = (0.0, 0.0);
            for seed in 0..40 {
                let mut g = Gauss(0xC0FF_EE00 + seed);
                let cams: Vec<Iso3> = robot
                    .iter()
                    .map(|bg| {
                        let c = if eye_to_hand {
                            z * bg * x
                        } else {
                            (bg * x).inverse() * z
                        };
                        g.perturb(&c, 0.2_f64.to_radians(), 1e-3)
                    })
                    .collect();
                let measured: Vec<Iso3> = robot
                    .iter()
                    .map(|bg| g.perturb(bg, 0.04_f64.to_radians(), 2e-4))
                    .collect();
                let est = if eye_to_hand {
                    estimate_eye_to_hand(method, &measured, &cams, 1.0)
                        .map(|e| e.gripper_se3_target)
                } else {
                    estimate_eye_in_hand(method, &measured, &cams, 1.0).map(|e| e.gripper_se3_cam)
                }
                .unwrap();
                let (dt, dr) = pose_error(&est, &x);
                rot = rot.max(dr.to_degrees());
                tra = tra.max(dt * 1e3);
            }
            (rot, tra)
        };
        for eye_to_hand in [false, true] {
            for method in METHODS {
                let (rot_q, tra_q) = worst(method, &quarter_turns, eye_to_hand);
                let (rot_h, tra_h) = worst(method, &half_turns, eye_to_hand);
                println!(
                    "{method:?} eye_to_hand={eye_to_hand}: quarter turns {rot_q:.3}° {tra_q:.2} mm, \
                     half turns {rot_h:.3}° {tra_h:.2} mm"
                );
                assert!(
                    rot_h <= 1.5 * rot_q && tra_h <= 1.5 * tra_q,
                    "{method:?} eye_to_hand={eye_to_hand}: half turns {rot_h:.3}° {tra_h:.2} mm \
                     vs quarter turns {rot_q:.3}° {tra_q:.2} mm"
                );
            }
        }
    }
}
