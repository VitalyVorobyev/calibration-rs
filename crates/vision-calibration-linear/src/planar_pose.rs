//! Planar pose estimation from a homography.
//!
//! Decomposes a plane-induced homography into a pose `T_C_B` given intrinsics
//! `K`, assuming the board lies on `Z = 0` in its own coordinates.

use anyhow::Result;
use nalgebra::{Matrix3, Rotation3, Translation3, UnitQuaternion, Vector3};
use vision_calibration_core::{Iso3, Mat3, Real};

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
pub fn estimate_planar_pose_from_h(kmtx: &Mat3, hmtx: &Mat3) -> Result<Iso3> {
    PlanarPoseSolver::from_homography(kmtx, hmtx)
}

impl PlanarPoseSolver {
    /// Decompose a homography into a pose `T_C_B` given intrinsics `K`.
    ///
    /// The resulting rotation is projected onto SO(3); the translation is
    /// scaled so that the first two rotation columns have unit norm.
    pub fn from_homography(kmtx: &Mat3, hmtx: &Mat3) -> Result<Iso3> {
        // K^{-1}
        let k_inv = kmtx
            .try_inverse()
            .ok_or_else(|| anyhow::anyhow!("intrinsics matrix is not invertible"))?;

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
            anyhow::bail!("degenerate homography for planar pose extraction");
        }
        let denom = (norm1 + norm2) * 0.5;
        if denom <= 1e-12 {
            anyhow::bail!("degenerate homography for planar pose extraction");
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
            anyhow::bail!("degenerate homography for planar pose extraction");
        }

        let mut r_mat = Matrix3::<Real>::zeros();
        r_mat.set_column(0, &r1);
        r_mat.set_column(1, &r2);
        r_mat.set_column(2, &r3);

        // Project onto SO(3) (polar decomposition via SVD)
        let svd = r_mat.svd(true, true);
        let u = svd
            .u
            .ok_or_else(|| anyhow::anyhow!("svd failed during planar pose extraction"))?;
        let v_t = svd
            .v_t
            .ok_or_else(|| anyhow::anyhow!("svd failed during planar pose extraction"))?;
        let r_orth = u * v_t;

        // Ensure det(R) > 0
        let mut r_orth = if r_orth.determinant() < 0.0 {
            let mut u_flipped = u;
            u_flipped.column_mut(2).neg_mut();
            u_flipped * v_t
        } else {
            r_orth
        };

        if t_vec.z < 0.0 {
            r_orth.column_mut(0).neg_mut();
            r_orth.column_mut(1).neg_mut();
            t_vec = -t_vec;
        }

        Ok(build_iso(r_orth, t_vec))
    }
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
    use vision_calibration_core::FxFyCxCySkew;

    fn make_kmtx() -> Mat3 {
        let k = FxFyCxCySkew {
            fx: 800.0,
            fy: 780.0,
            cx: 640.0,
            cy: 360.0,
            skew: 0.0,
        };
        Matrix3::new(k.fx, k.skew, k.cx, 0.0, k.fy, k.cy, 0.0, 0.0, 1.0)
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
}
