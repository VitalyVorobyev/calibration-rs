//! Triangulation of 3D points from multiple views.
//!
//! [`triangulate_point_linear`] solves the homogeneous DLT system directly (a
//! fast algebraic estimate), and [`triangulate_point`] additionally refines that
//! estimate with Gauss-Newton iterations that minimize the geometric
//! reprojection error across all views (the maximum-likelihood point under
//! isotropic Gaussian image noise).

use anyhow::Result;
use nalgebra::{DMatrix, Matrix3, Vector3, Vector4};
use vision_calibration_core::{Pt2, Pt3, Real};

use crate::camera_matrix::Mat34;
use crate::math::{dlt_rank_ok, null_space};

/// Linear (DLT) triangulation from N ≥ 2 views.
///
/// `cameras` are 3×4 projection matrices `P_i`, and `points` are their
/// corresponding image coordinates. The returned 3D point is in the same
/// world frame as the camera matrices.
///
/// The homogeneous system is solved via [`null_space`] (the smallest-singular
/// direction of the `2N×4` design matrix), computed from the `AᵀA` symmetric
/// eigendecomposition rather than a dense SVD — see [`null_space`] for why the
/// SVD path is deliberately avoided.
///
/// # Errors
///
/// Returns an error if fewer than 2 views are given, the camera/point counts
/// disagree, the system is rank-deficient (identical cameras or coincident
/// rays / zero parallax), or the recovered point is at infinity.
pub fn triangulate_point_linear(cameras: &[Mat34], points: &[Pt2]) -> Result<Pt3> {
    if cameras.len() < 2 {
        anyhow::bail!("need at least 2 views, got {}", cameras.len());
    }
    if cameras.len() != points.len() {
        anyhow::bail!(
            "mismatched number of cameras ({}) and points ({})",
            cameras.len(),
            points.len()
        );
    }

    let mut a = DMatrix::<Real>::zeros(2 * cameras.len(), 4);
    for (i, (p, cam)) in points.iter().zip(cameras.iter()).enumerate() {
        let u = p.x;
        let v = p.y;

        let r0 = 2 * i;
        let r1 = 2 * i + 1;

        let row0 = cam.row(0);
        let row1 = cam.row(1);
        let row2 = cam.row(2);

        a.row_mut(r0).copy_from(&(u * row2 - row0));
        a.row_mut(r1).copy_from(&(v * row2 - row1));
    }

    let ns = null_space(&a)?;

    // The 4-column DLT triangulation matrix is well-posed at rank 3 (1-D null
    // space). Identical cameras or coincident rays collapse the boundary
    // singular value toward zero.
    if !dlt_rank_ok(&ns.singular_values, 4, 1, 1e-7) {
        anyhow::bail!(
            "zero-parallax / rank-deficient triangulation system \
             (identical cameras or coincident rays)"
        );
    }

    let x_h = &ns.vector;
    let w = x_h[3];
    if w.abs() <= Real::EPSILON {
        anyhow::bail!("triangulation produced an invalid point");
    }

    Ok(Pt3::new(x_h[0] / w, x_h[1] / w, x_h[2] / w))
}

/// Maximum iterations for the Gauss-Newton reprojection refinement.
const REFINE_MAX_ITERS: usize = 10;
/// Convergence threshold on the parameter step norm.
const REFINE_STEP_EPS: Real = 1e-12;

/// Refine a triangulated point by minimizing total squared reprojection error
/// across all views via Gauss-Newton.
///
/// This is a self-contained 3-parameter least-squares solve (no external
/// optimizer): each view contributes a 2-vector reprojection residual whose
/// `2×3` Jacobian w.r.t. the point is formed in closed form. Starting from a
/// good algebraic estimate (e.g. [`triangulate_point_linear`]) it converges in
/// a handful of iterations. Views whose depth collapses (`z ≈ 0`) are skipped
/// for that iteration; a singular normal matrix stops the refinement early and
/// returns the best estimate so far. The result is never worse-conditioned than
/// the input — on degenerate input it simply returns `init` unchanged.
pub fn refine_point(init: &Pt3, cameras: &[Mat34], points: &[Pt2]) -> Pt3 {
    let mut x = Vector3::new(init.x, init.y, init.z);

    for _ in 0..REFINE_MAX_ITERS {
        let mut jtj = Matrix3::<Real>::zeros();
        let mut jtr = Vector3::<Real>::zeros();

        for (cam, pt) in cameras.iter().zip(points.iter()) {
            let xh = Vector4::new(x[0], x[1], x[2], 1.0);
            let h = cam * xh;
            if h.z.abs() <= Real::EPSILON {
                continue;
            }
            let inv_z = 1.0 / h.z;
            let proj_u = h.x * inv_z;
            let proj_v = h.y * inv_z;

            // ∂proj/∂X from the left 3×3 block M of the projection matrix:
            //   ∂(h0/h2)/∂X = (M_row0 - proj_u · M_row2) / h2.
            let m = cam.fixed_view::<3, 3>(0, 0);
            let row0 = m.row(0).transpose();
            let row1 = m.row(1).transpose();
            let row2 = m.row(2).transpose();
            let ju = (row0 - proj_u * row2) * inv_z;
            let jv = (row1 - proj_v * row2) * inv_z;

            let r_u = proj_u - pt.x;
            let r_v = proj_v - pt.y;

            jtj += ju * ju.transpose() + jv * jv.transpose();
            jtr += ju * r_u + jv * r_v;
        }

        // Gauss-Newton step: solve (JᵀJ) Δ = -(Jᵀr).
        let Some(inv) = jtj.try_inverse() else {
            break;
        };
        let dx = -inv * jtr;
        x += dx;
        if dx.norm() < REFINE_STEP_EPS {
            break;
        }
    }

    Pt3::new(x[0], x[1], x[2])
}

/// Triangulate a point from N ≥ 2 views with nonlinear (reprojection) refinement.
///
/// Computes the [`triangulate_point_linear`] DLT estimate, then polishes it with
/// [`refine_point`] (Gauss-Newton minimization of the geometric reprojection
/// error). Prefer this over the bare linear solve when accuracy matters and the
/// image observations carry noise.
///
/// # Errors
///
/// Propagates the errors of [`triangulate_point_linear`] (too few views, count
/// mismatch, rank-deficiency, point at infinity).
pub fn triangulate_point(cameras: &[Mat34], points: &[Pt2]) -> Result<Pt3> {
    let init = triangulate_point_linear(cameras, points)?;
    Ok(refine_point(&init, cameras, points))
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Rotation3, Vector3 as NaVector3};

    fn project(cam: &Mat34, p: &Pt3) -> Pt2 {
        let x = cam * Vector4::new(p.x, p.y, p.z, 1.0);
        Pt2::new(x.x / x.z, x.y / x.z)
    }

    /// Build a projection matrix `P = [R | t]` from a rotation and translation.
    fn camera(rx: Real, ry: Real, rz: Real, t: NaVector3<Real>) -> Mat34 {
        let r = *Rotation3::from_euler_angles(rx, ry, rz).matrix();
        let mut p = Mat34::zeros();
        p.fixed_view_mut::<3, 3>(0, 0).copy_from(&r);
        p.set_column(3, &t);
        p
    }

    #[test]
    fn triangulation_two_views_recovers_point() {
        let cam1 = Mat34::new(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
        let cam2 = Mat34::new(1.0, 0.0, 0.0, -0.2, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);

        let pw = Pt3::new(0.1, -0.05, 2.0);
        let p1 = project(&cam1, &pw);
        let p2 = project(&cam2, &pw);

        let est = triangulate_point_linear(&[cam1, cam2], &[p1, p2]).unwrap();

        let err = (est - pw).norm();
        assert!(err < 1e-6, "triangulation error too large: {}", err);
    }

    /// Regression: two identical cameras (zero baseline) must return Err.
    #[test]
    fn triangulation_rejects_identical_cameras() {
        let cam = Mat34::new(
            800.0, 0.0, 320.0, 0.0, 0.0, 800.0, 240.0, 0.0, 0.0, 0.0, 1.0, 0.0,
        );
        let pw = Pt3::new(0.05, -0.02, 1.5);
        let img = project(&cam, &pw);
        assert!(
            triangulate_point_linear(&[cam, cam], &[img, img]).is_err(),
            "identical cameras (zero baseline) must return Err"
        );
    }

    /// Four spread-out views recover a noiseless point essentially exactly.
    #[test]
    fn triangulation_nview_recovers_point() {
        let cams = [
            camera(0.0, 0.0, 0.0, NaVector3::new(0.0, 0.0, 0.0)),
            camera(0.02, -0.03, 0.01, NaVector3::new(-0.4, 0.05, 0.02)),
            camera(-0.05, 0.04, 0.0, NaVector3::new(0.3, -0.2, 0.1)),
            camera(0.01, 0.06, -0.02, NaVector3::new(0.1, 0.35, -0.05)),
        ];
        let pw = Pt3::new(0.15, -0.1, 3.0);
        let pts: Vec<Pt2> = cams.iter().map(|c| project(c, &pw)).collect();

        let est = triangulate_point(&cams, &pts).unwrap();
        assert!(
            (est - pw).norm() < 1e-9,
            "n-view error: {}",
            (est - pw).norm()
        );
    }

    /// With per-view pixel noise, the Gauss-Newton refinement reduces the 3D
    /// error of the linear estimate (it minimizes geometric, not algebraic,
    /// error).
    #[test]
    fn refinement_improves_noisy_estimate() {
        let cams = [
            camera(0.0, 0.0, 0.0, NaVector3::new(0.0, 0.0, 0.0)),
            camera(0.03, -0.02, 0.01, NaVector3::new(-0.5, 0.04, 0.03)),
            camera(-0.04, 0.05, -0.01, NaVector3::new(0.4, -0.25, 0.08)),
            camera(0.02, 0.05, 0.0, NaVector3::new(0.15, 0.4, -0.06)),
            camera(-0.01, -0.04, 0.02, NaVector3::new(-0.2, -0.3, 0.12)),
        ];
        let pw = Pt3::new(-0.2, 0.12, 2.5);

        // Deterministic, view-dependent perturbation of the image points.
        let pts: Vec<Pt2> = cams
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let clean = project(c, &pw);
                let n = 0.002 * ((i as Real) - 2.0); // ±a few mrad in normalized coords
                Pt2::new(clean.x + n, clean.y - n)
            })
            .collect();

        let lin = triangulate_point_linear(&cams, &pts).unwrap();
        let refined = refine_point(&lin, &cams, &pts);

        let err_lin = (lin - pw).norm();
        let err_ref = (refined - pw).norm();
        assert!(
            err_ref <= err_lin + 1e-12,
            "refinement worsened the estimate: lin={err_lin}, refined={err_ref}"
        );
    }

    /// On degenerate input (zero parallax) the linear solve errors and the
    /// refinement, given the linear estimate, must not panic.
    #[test]
    fn refine_point_is_safe_on_degenerate_camera() {
        let cam = Mat34::new(
            800.0, 0.0, 320.0, 0.0, 0.0, 800.0, 240.0, 0.0, 0.0, 0.0, 1.0, 0.0,
        );
        let pw = Pt3::new(0.05, -0.02, 1.5);
        let img = project(&cam, &pw);
        // Refining from a reasonable guess with identical cameras returns a
        // finite point without panicking.
        let out = refine_point(&pw, &[cam, cam], &[img, img]);
        assert!(out.coords.iter().all(|v| v.is_finite()));
    }
}
