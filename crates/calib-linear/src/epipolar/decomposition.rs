//! Essential matrix decomposition into rotation and translation.
//!
//! Recovers candidate camera poses from an essential matrix using SVD
//! decomposition. Returns four possible (R, t) pairs that must be
//! disambiguated via cheirality checks.

use anyhow::Result;
use calib_core::{Mat3, Real, Vec3};
use nalgebra::SMatrix;

/// Enforce essential matrix constraints via SVD projection.
///
/// Projects a 3x3 matrix onto the essential matrix manifold by forcing
/// the singular values to be (σ, σ, 0) where σ is the mean of the first
/// two singular values.
pub(super) fn enforce_essential_constraints(e: &Mat3) -> Result<Mat3> {
    let svd = e.svd(true, true);
    let u = svd.u.ok_or(anyhow::anyhow!("SVD failed"))?;
    let v_t = svd.v_t.ok_or(anyhow::anyhow!("SVD failed"))?;

    let s1 = svd.singular_values[0];
    let s2 = svd.singular_values[1];
    let s = 0.5 * (s1 + s2);

    let s_mat = SMatrix::<Real, 3, 3>::from_diagonal(&nalgebra::Vector3::new(s, s, 0.0));
    Ok(u * s_mat * v_t)
}

/// Decompose an essential matrix into candidate rotation and translation pairs.
///
/// Returns four possible `(R, t)` pairs; the correct one can be selected by
/// cheirality checks on triangulated points. The translation is unit-length
/// (direction only).
pub fn decompose_essential(e: &Mat3) -> Result<Vec<(Mat3, Vec3)>> {
    let e = enforce_essential_constraints(e)?;
    let svd = e.svd(true, true);
    let mut u = svd.u.ok_or(anyhow::anyhow!("SVD failed"))?;
    let mut v_t = svd.v_t.ok_or(anyhow::anyhow!("SVD failed"))?;

    if u.determinant() < 0.0 {
        u.column_mut(2).neg_mut();
    }
    if v_t.determinant() < 0.0 {
        v_t.row_mut(2).neg_mut();
    }

    let w = Mat3::new(0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);

    let r1 = u * w * v_t;
    let r2 = u * w.transpose() * v_t;

    let t = u.column(2).normalize();

    let mut solutions = vec![
        (r1, t.into_owned()),
        (r1, (-t).into_owned()),
        (r2, t.into_owned()),
        (r2, (-t).into_owned()),
    ];

    for (r, t) in solutions.iter_mut() {
        if r.determinant() < 0.0 {
            *r = -*r;
            *t = -*t;
        }
    }

    Ok(solutions)
}

#[cfg(test)]
mod tests {
    use super::*;
    use calib_core::Vec3;
    use nalgebra::Rotation3;

    fn skew(v: &Vec3) -> Mat3 {
        Mat3::new(0.0, -v.z, v.y, v.z, 0.0, -v.x, -v.y, v.x, 0.0)
    }

    #[test]
    fn essential_decomposition_recovers_pose() {
        let rot = Rotation3::from_euler_angles(0.1, -0.05, 0.2);
        let t = Vec3::new(0.1, 0.02, -0.03);

        let e = skew(&t) * rot.matrix();
        let solutions = decompose_essential(&e).unwrap();

        let mut found = false;
        for (r_est, t_est) in solutions {
            let r_diff = r_est.transpose() * rot.matrix();
            let trace = r_diff.trace();
            let cos_theta = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0);
            let ang = cos_theta.acos();

            let t_dir = t_est.normalize();
            let cos_t = t_dir.dot(&t.normalize()).abs();

            if ang < 1e-6 && (1.0 - cos_t) < 1e-6 {
                found = true;
                break;
            }
        }

        assert!(found, "essential decomposition did not recover pose");
    }
}
