//! Essential matrix decomposition into rotation and translation.
//!
//! Recovers candidate camera poses from an essential matrix using SVD
//! decomposition. Returns four possible (R, t) pairs that must be
//! disambiguated via cheirality checks.

use anyhow::Result;
use nalgebra::SMatrix;
use vision_calibration_core::{Mat3, Real, Vec3};

/// Enforce essential matrix constraints via SVD projection.
///
/// Projects a 3×3 matrix onto the essential matrix manifold by forcing
/// the singular values to be (σ, σ, 0) where σ is the mean of the first
/// two singular values.
pub(crate) fn enforce_essential_constraints(e: &Mat3) -> Result<Mat3> {
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
///
/// Returns `Err` if `E` is degenerate before the essential-constraint projection.
/// The check is performed on the **raw** input matrix to catch rank-1 inputs such
/// as `diag(1, 0, 0)` that would otherwise slip through to `(σ/2, σ/2, 0)` after
/// projection and produce an arbitrary translation direction:
///
/// - `σ₀ < 1e-10` — near-zero matrix (pure rotation with `t = 0`, or all-zeros).
/// - `σ₁ / σ₀ < 0.1` — rank-1 input; the second singular value has collapsed and
///   the translation direction is unobservable.
///
/// A valid estimated `E` (even from a noisy 8-point solve) has raw `σ₀ ≈ σ₁ > 0`,
/// so the threshold of 0.1 is conservative.
pub fn decompose_essential(e: &Mat3) -> Result<Vec<(Mat3, Vec3)>> {
    // Degeneracy check on the RAW input — before projection.
    // This catches rank-1 inputs like diag(1,0,0) that survive projection
    // to (σ/2, σ/2, 0) and would yield an arbitrary translation direction.
    {
        let sv = e.svd(false, false).singular_values;
        let s0 = sv[0];
        let s1 = sv[1];
        if s0 < 1e-10 {
            anyhow::bail!(
                "degenerate essential matrix: translation is unobservable \
                 (rank-deficient or zero E)"
            );
        }
        if s1 / s0 < 0.1 {
            anyhow::bail!(
                "degenerate essential matrix: translation is unobservable \
                 (rank-deficient or zero E)"
            );
        }
    }

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
    use nalgebra::Rotation3;

    fn skew(v: &Vec3) -> Mat3 {
        Mat3::new(0.0, -v.z, v.y, v.z, 0.0, -v.x, -v.y, v.x, 0.0)
    }

    /// Regression: zero E must return Err, not fabricate four poses.
    #[test]
    fn decompose_essential_rejects_zero_matrix() {
        let e = Mat3::zeros();
        assert!(
            decompose_essential(&e).is_err(),
            "zero essential matrix must return Err"
        );
    }

    /// Regression: a pure-rotation-derived E = [t]×R with t=0 is all zeros
    /// and must also be rejected.
    #[test]
    fn decompose_essential_rejects_pure_rotation_e() {
        use nalgebra::Rotation3;
        let rot = Rotation3::from_euler_angles(0.1, -0.05, 0.2);
        // t = 0 → E = [0]× R = 0
        let t_zero = Vec3::zeros();
        let e = skew(&t_zero) * rot.matrix();
        assert!(
            decompose_essential(&e).is_err(),
            "pure-rotation E (t=0 ⟹ E=0) must return Err"
        );
    }

    /// Regression: a rank-1 raw E such as `diag(1,0,0)` must return Err.
    ///
    /// The OLD guard checked singular values AFTER `enforce_essential_constraints`.
    /// `diag(1,0,0)` has σ = (1,0,0); after projection it becomes `(0.5, 0.5, 0)`,
    /// which passes the post-projection ratio check — a fabricated translation
    /// direction is returned.  The new pre-projection check catches it.
    #[test]
    fn decompose_essential_rejects_rank1_diag_input() {
        use nalgebra::Matrix3;
        // diag(1,0,0) — rank 1, σ = (1, 0, 0).
        let e = Matrix3::from_diagonal(&nalgebra::Vector3::new(1.0, 0.0, 0.0));
        assert!(
            decompose_essential(&e).is_err(),
            "rank-1 diag(1,0,0) essential matrix must return Err"
        );
    }

    /// Sanity: an existing valid-E decomposition test must still pass after the guard.
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
