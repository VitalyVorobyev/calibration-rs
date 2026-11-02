//! Planar pose estimation from a homography or from bearing vectors.
//!
//! Recovers the pose `T_C_B` of a board lying on `Z = 0` in its own
//! coordinates: from a plane-induced homography and pinhole intrinsics `K`,
//! or from the bearings of any central camera.

use crate::Error;
use nalgebra::{Matrix3, Rotation3, SMatrix, SVector, Translation3, UnitQuaternion, Vector3};
use vision_calibration_core::linalg::normalize_points_2d;
use vision_calibration_core::{Iso3, Mat3, Pt2, Real};

/// Linear pose initialisation from a homography and intrinsics.
///
/// This implements the classic decomposition of a plane-induced homography
/// `H` into a rotation and translation, assuming the target lies on the plane
/// `Z = 0` in its own coordinates.
#[derive(Debug, Clone, Copy)]
pub struct PlanarPoseSolver;

/// Estimate pose of a planar board (Z = 0) relative to camera, given intrinsics
/// `K` and homography `H` (plane -> image).
///
/// Returns an `Iso3` that maps board coordinates into camera coordinates.
///
/// # Errors
///
/// Returns [`Error::Singular`] if `K` is not invertible, the homography is
/// degenerate, or SVD fails during SO(3) projection.
pub fn estimate_planar_pose_from_h(kmtx: &Mat3, hmtx: &Mat3) -> Result<Iso3, Error> {
    PlanarPoseSolver::from_homography(kmtx, hmtx)
}

impl PlanarPoseSolver {
    /// Decompose a homography into a pose `T_C_B` given intrinsics `K`.
    ///
    /// The resulting rotation is projected onto SO(3); the translation is
    /// scaled so that the first two rotation columns have unit norm.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Singular`] if `K` is not invertible, the homography is
    /// degenerate, or SVD fails during SO(3) projection.
    pub fn from_homography(kmtx: &Mat3, hmtx: &Mat3) -> Result<Iso3, Error> {
        // K^{-1}
        let k_inv = kmtx.try_inverse().ok_or(Error::Singular)?;

        // Columns of H
        let h1 = hmtx.column(0);
        let h2 = hmtx.column(1);
        let h3 = hmtx.column(2).into_owned();

        let k_inv_h1 = k_inv * h1;
        let k_inv_h2 = k_inv * h2;

        // Scale factor λ: normalize first two columns (average for robustness)
        let norm1 = k_inv_h1.norm();
        let norm2 = k_inv_h2.norm();
        if norm1 <= 1e-12 || norm2 <= 1e-12 {
            return Err(Error::Singular);
        }
        let denom = (norm1 + norm2) * 0.5;
        if denom <= 1e-12 {
            return Err(Error::Singular);
        }
        let lambda = 1.0 / denom;

        let mut r1 = (lambda * k_inv_h1).into_owned();
        let mut r2 = (lambda * k_inv_h2).into_owned();
        let mut t_vec: Vector3<Real> = (lambda * (k_inv * h3)).into_owned();
        if t_vec.z < 0.0 {
            r1 = -r1;
            r2 = -r2;
            t_vec = -t_vec;
        }
        let r3 = r1.cross(&r2);
        if r3.norm() <= 1e-12 {
            return Err(Error::Singular);
        }

        let mut r_mat = Matrix3::<Real>::zeros();
        r_mat.set_column(0, &r1);
        r_mat.set_column(1, &r2);
        r_mat.set_column(2, &r3);

        // Project onto SO(3) (polar decomposition; see `math::project_to_so3`).
        let mut r_orth = crate::math::project_to_so3(&r_mat)?;

        if t_vec.z < 0.0 {
            r_orth.column_mut(0).neg_mut();
            r_orth.column_mut(1).neg_mut();
            t_vec = -t_vec;
        }

        Ok(build_iso(r_orth, t_vec))
    }
}

/// Estimate the pose of a planar board (`Z = 0`) from the camera's bearing
/// vectors to its points: a direct linear transform on `b × H p = 0`, then
/// the decomposition `H ∝ [r1 r2 t]`.
///
/// `board` holds each point's board coordinates, `bearings` the direction
/// from the camera centre to it (any length, any direction — points beside
/// or behind the camera, as a fisheye lens sees them, are fine). Any
/// central camera gives bearings by back-projecting its pixels, so this is
/// the pose seed of every central projection; for a pinhole camera it
/// equals [`estimate_planar_pose_from_h`] on undistorted pixels.
///
/// Returns `T_C_B`, mapping board coordinates into camera coordinates.
///
/// # Errors
///
/// - [`Error::InvalidInput`] if the slices differ in length or a bearing is
///   zero or not finite.
/// - [`Error::InsufficientData`] for fewer than four points.
/// - [`Error::Singular`] when the points are collinear or the system is
///   otherwise degenerate.
pub fn estimate_planar_pose_from_bearings(
    board: &[Pt2],
    bearings: &[Vector3<Real>],
) -> Result<Iso3, Error> {
    if board.len() != bearings.len() {
        return Err(Error::invalid_input(format!(
            "{} board points but {} bearings",
            board.len(),
            bearings.len()
        )));
    }
    if board.len() < 4 {
        return Err(Error::InsufficientData {
            need: 4,
            got: board.len(),
        });
    }
    let unit: Vec<Vector3<Real>> = bearings
        .iter()
        .map(|b| {
            let norm = b.norm();
            if norm > 0.0 && norm.is_finite() {
                Ok(b / norm)
            } else {
                Err(Error::invalid_input("a bearing is zero or not finite"))
            }
        })
        .collect::<Result<_, _>>()?;
    let (normalized, t_board) = normalize_points_2d(board).ok_or(Error::Singular)?;

    // Normal equations of the stacked rows of `[b]× (H p̃)`, with `h` the
    // row-major entries of H: the null vector is the eigenvector of the
    // smallest eigenvalue.
    let mut ata = SMatrix::<Real, 9, 9>::zeros();
    for (p, b) in normalized.iter().zip(&unit) {
        let p = Vector3::new(p.x, p.y, 1.0);
        let mut rows = [[0.0; 9]; 3];
        for k in 0..3 {
            // Row `i` of `[b]× H p̃` is `Σ_j [b]×(i, j) · (h_j · p̃)`.
            let skew = [[0.0, -b.z, b.y], [b.z, 0.0, -b.x], [-b.y, b.x, 0.0]];
            for (j, coef) in skew[k].iter().enumerate() {
                for m in 0..3 {
                    rows[k][3 * j + m] = coef * p[m];
                }
            }
        }
        for row in &rows {
            let r = SVector::<Real, 9>::from_row_slice(row);
            ata += r * r.transpose();
        }
    }
    let eigen = ata.symmetric_eigen();
    let (min_idx, _) = eigen
        .eigenvalues
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.total_cmp(b.1))
        .ok_or(Error::Singular)?;
    let h = eigen.eigenvectors.column(min_idx);
    let h_norm = Matrix3::new(h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], h[8]);
    let hmtx = h_norm * t_board;

    let h1 = hmtx.column(0).into_owned();
    let h2 = hmtx.column(1).into_owned();
    let scale = 0.5 * (h1.norm() + h2.norm());
    if scale.is_nan() || scale <= 1e-12 {
        return Err(Error::Singular);
    }
    // The sign that puts the board along its bearings, not opposite them.
    let alignment: Real = board
        .iter()
        .zip(&unit)
        .map(|(p, b)| b.dot(&(hmtx * Vector3::new(p.x, p.y, 1.0))))
        .sum();
    let lambda = alignment.signum() / scale;
    let r1 = h1 * lambda;
    let r2 = h2 * lambda;
    let t_vec = hmtx.column(2) * lambda;
    let r3 = r1.cross(&r2);
    if r3.norm() <= 1e-12 {
        return Err(Error::Singular);
    }
    let r_orth = crate::math::project_to_so3(&Matrix3::from_columns(&[r1, r2, r3]))?;
    Ok(build_iso(r_orth, t_vec))
}

fn build_iso(r_orth: Matrix3<Real>, t_vec: Vector3<Real>) -> Iso3 {
    let rot = UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(r_orth));
    let trans = Translation3::from(t_vec);

    Iso3::from_parts(trans, rot)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Isometry3, Matrix3, Rotation3, Vector3};
    use vision_calibration_core::FxFyCxCy;

    fn make_kmtx() -> Mat3 {
        let k = FxFyCxCy {
            fx: 800.0,
            fy: 780.0,
            cx: 640.0,
            cy: 360.0,
        };
        Matrix3::new(k.fx, 0.0, k.cx, 0.0, k.fy, k.cy, 0.0, 0.0, 1.0)
    }

    #[test]
    fn planar_pose_from_h_recovers_pose() {
        let kmtx = make_kmtx();

        // Synthetic pose: small rotation & translation
        let rot = Rotation3::from_euler_angles(0.1, -0.05, 0.2);
        let t = Vector3::new(0.1, -0.05, 1.0);
        let iso_gt = Isometry3::from_parts(Translation3::from(t), rot.into());

        // For a plane Z=0, homography is H = K [r1 r2 t]
        let r_mat_binding = iso_gt.rotation.to_rotation_matrix();
        let r_mat = r_mat_binding.matrix();
        let r1 = r_mat.column(0);
        let r2 = r_mat.column(1);
        let t = iso_gt.translation.vector;

        let mut hmtx = Mat3::zeros();
        hmtx.set_column(0, &(kmtx * r1));
        hmtx.set_column(1, &(kmtx * r2));
        hmtx.set_column(2, &(kmtx * t));

        let iso_est = estimate_planar_pose_from_h(&kmtx, &hmtx).unwrap();

        let t_est = iso_est.translation.vector;
        let r_est_binding = iso_est.rotation.to_rotation_matrix();
        let r_est = r_est_binding.matrix();

        // Compare R via angle between axes and translations
        assert!((t_est - iso_gt.translation.vector).norm() < 1e-3);

        let r_diff = r_est.transpose() * r_mat;
        let angle = ((r_diff.trace() - 1.0) * 0.5).clamp(-1.0, 1.0).acos();
        assert!(angle < 1e-3, "rotation error too large: {}", angle);
    }

    #[test]
    fn planar_pose_from_bearings_recovers_a_board_beside_and_behind_the_camera() {
        // A board lying across the camera's side: its points range from
        // in front (z > 0) to behind (z < 0) the image plane.
        let rot = Rotation3::from_euler_angles(0.2, 1.4, -0.3);
        let t = Vector3::new(0.25, -0.1, 0.02);
        let pose = Isometry3::from_parts(Translation3::from(t), rot.into());
        let board: Vec<Pt2> = (0..6)
            .flat_map(|i| (0..5).map(move |j| Pt2::new(i as Real * 0.04, j as Real * 0.04)))
            .collect();
        let bearings: Vec<Vector3<Real>> = board
            .iter()
            .map(|p| {
                let p_c = pose.transform_point(&nalgebra::Point3::new(p.x, p.y, 0.0));
                // Bearings of arbitrary length, as back-projection gives them.
                p_c.coords * (1.0 + p.x)
            })
            .collect();
        assert!(bearings.iter().any(|b| b.z < 0.0) && bearings.iter().any(|b| b.z > 0.0));

        let est = estimate_planar_pose_from_bearings(&board, &bearings).unwrap();
        assert!((est.translation.vector - t).norm() < 1e-9);
        assert!(est.rotation.angle_to(&pose.rotation) < 1e-9);
    }

    #[test]
    fn planar_pose_from_bearings_rejects_collinear_points() {
        let board: Vec<Pt2> = (0..6).map(|i| Pt2::new(i as Real * 0.1, 0.0)).collect();
        let bearings: Vec<Vector3<Real>> =
            board.iter().map(|p| Vector3::new(p.x, 0.0, 1.0)).collect();
        assert!(estimate_planar_pose_from_bearings(&board, &bearings).is_err());
    }
}
