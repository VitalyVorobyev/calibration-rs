//! Nonlinear refinement of geometric estimates.
//!
//! Feature-gated behind `refine`. Uses tiny-solver for Levenberg-Marquardt
//! optimization with analytic residuals.
//!
//! - [`refine_homography`]: minimize symmetric transfer error

use crate::{MvgError, Result};
use nalgebra::DVector;
use tiny_solver::LevenbergMarquardtOptimizer;
use tiny_solver::factors::Factor;
use tiny_solver::optimizer::{Optimizer, OptimizerOptions};
use tiny_solver::problem::Problem;
use vision_calibration_core::{Mat3, Pt2, Real};

// ---------------------------------------------------------------------------
// Homography refinement
// ---------------------------------------------------------------------------

/// Factor for symmetric transfer error of a single correspondence under H.
struct HomographyTransferFactor {
    pt1: Pt2,
    pt2: Pt2,
}

impl<T: nalgebra::RealField> Factor<T> for HomographyTransferFactor {
    fn residual_func(&self, params: &[DVector<T>]) -> DVector<T> {
        debug_assert_eq!(params.len(), 1);
        let h = &params[0]; // 9 elements, row-major

        let x1 = T::from_f64(self.pt1.x).unwrap();
        let y1 = T::from_f64(self.pt1.y).unwrap();
        let x2 = T::from_f64(self.pt2.x).unwrap();
        let y2 = T::from_f64(self.pt2.y).unwrap();

        // Forward: H * [x1, y1, 1]
        let fx = h[0].clone() * x1.clone() + h[1].clone() * y1.clone() + h[2].clone();
        let fy = h[3].clone() * x1.clone() + h[4].clone() * y1.clone() + h[5].clone();
        let fw = h[6].clone() * x1.clone() + h[7].clone() * y1.clone() + h[8].clone();
        let dx_fwd = fx / fw.clone() - x2.clone();
        let dy_fwd = fy / fw - y2.clone();

        // Inverse: H^-1 * [x2, y2, 1] via adjugate (avoids explicit inverse).
        // For a 3×3 matrix, adjugate columns are cross products of pairs of rows.
        // H^-1 = adj(H) / det(H). Since we only care about the projected point,
        // we can use adj(H) directly (det cancels in homogeneous division).
        let a00 = h[4].clone() * h[8].clone() - h[5].clone() * h[7].clone();
        let a01 = h[2].clone() * h[7].clone() - h[1].clone() * h[8].clone();
        let a02 = h[1].clone() * h[5].clone() - h[2].clone() * h[4].clone();
        let a10 = h[5].clone() * h[6].clone() - h[3].clone() * h[8].clone();
        let a11 = h[0].clone() * h[8].clone() - h[2].clone() * h[6].clone();
        let a12 = h[2].clone() * h[3].clone() - h[0].clone() * h[5].clone();
        let a20 = h[3].clone() * h[7].clone() - h[4].clone() * h[6].clone();
        let a21 = h[1].clone() * h[6].clone() - h[0].clone() * h[7].clone();
        let a22 = h[0].clone() * h[4].clone() - h[1].clone() * h[3].clone();

        let bx = a00 * x2.clone() + a01 * y2.clone() + a02;
        let by = a10 * x2.clone() + a11 * y2.clone() + a12;
        let bw = a20 * x2 + a21 * y2 + a22;

        let dx_inv = bx / bw.clone() - x1;
        let dy_inv = by / bw - y1;

        DVector::from_vec(vec![dx_fwd, dy_fwd, dx_inv, dy_inv])
    }
}

/// Refine a homography by minimizing symmetric transfer error.
///
/// Takes an initial estimate `h_init` and correspondences `(pts1, pts2)`,
/// returns the refined homography.
pub fn refine_homography(h_init: &Mat3, pts1: &[Pt2], pts2: &[Pt2]) -> Result<Mat3> {
    if pts1.len() < 4 {
        return Err(MvgError::InsufficientData {
            need: 4,
            got: pts1.len(),
        });
    }
    if pts1.len() != pts2.len() {
        return Err(MvgError::CountMismatch {
            expected: pts1.len(),
            got: pts2.len(),
        });
    }

    let mut problem = Problem::new();

    for (p1, p2) in pts1.iter().zip(pts2.iter()) {
        let factor = HomographyTransferFactor { pt1: *p1, pt2: *p2 };
        problem.add_residual_block(4, &["h"], Box::new(factor), None);
    }

    let mut initial = std::collections::HashMap::new();
    let h_vec: Vec<Real> = (0..3)
        .flat_map(|r| (0..3).map(move |c| (r, c)))
        .map(|(r, c)| h_init[(r, c)])
        .collect();
    initial.insert("h".to_string(), DVector::from_vec(h_vec));

    let opts = OptimizerOptions {
        max_iteration: 50,
        ..Default::default()
    };

    let optimizer = LevenbergMarquardtOptimizer::default();
    let result = optimizer
        .optimize(&problem, &initial, Some(opts))
        .ok_or_else(|| MvgError::not_converged("homography refinement failed to converge"))?;

    let h_opt = &result["h"];
    let mut h = Mat3::zeros();
    for r in 0..3 {
        for c in 0..3 {
            h[(r, c)] = h_opt[r * 3 + c];
        }
    }

    // Normalize so h[2,2] = 1.
    let s = h[(2, 2)];
    if s.abs() > f64::EPSILON {
        h /= s;
    }

    Ok(h)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Rotation3;
    use vision_calibration_core::{Pt3, Vec3};

    #[test]
    fn refine_homography_improves_estimate() {
        let rot = Rotation3::from_euler_angles(0.05, -0.03, 0.02);
        let r = *rot.matrix();
        let t = Vec3::new(0.3, 0.01, 0.005);
        let n = Vec3::new(0.0, 0.0, 1.0);
        let d = 3.0;
        let h_gt = crate::homography::homography_from_pose_and_plane(&r, &t, &n, d);

        let plane_pts: Vec<Pt3> = vec![
            Pt3::new(0.5, 0.3, d),
            Pt3::new(-0.4, 0.2, d),
            Pt3::new(0.6, -0.3, d),
            Pt3::new(-0.3, -0.4, d),
            Pt3::new(0.1, 0.6, d),
            Pt3::new(0.4, -0.5, d),
        ];

        let pts1: Vec<_> = plane_pts
            .iter()
            .map(|p| Pt2::new(p.x / p.z, p.y / p.z))
            .collect();
        let pts2: Vec<_> = pts1
            .iter()
            .map(|p| crate::homography::homography_transfer(&h_gt, p))
            .collect();

        // Perturb the homography.
        let mut h_init = h_gt;
        h_init[(0, 0)] += 0.01;
        h_init[(1, 1)] += 0.01;

        let h_refined = refine_homography(&h_init, &pts1, &pts2).unwrap();

        // Refined should be closer to GT.
        let err_init = (h_init - h_gt).norm();
        let err_refined = (h_refined - h_gt).norm();
        assert!(
            err_refined < err_init,
            "refinement did not improve: init={:.6}, refined={:.6}",
            err_init,
            err_refined
        );
        assert!(
            err_refined < 1e-3,
            "refined error too large: {}",
            err_refined
        );
    }

    #[test]
    fn refine_homography_perfect_init_does_not_diverge() {
        let rot = Rotation3::from_euler_angles(0.05, -0.03, 0.02);
        let r = *rot.matrix();
        let t = Vec3::new(0.3, 0.01, 0.005);
        let n = Vec3::new(0.0, 0.0, 1.0);
        let d = 3.0;
        let h_gt = crate::homography::homography_from_pose_and_plane(&r, &t, &n, d);

        let plane_pts: Vec<Pt3> = vec![
            Pt3::new(0.5, 0.3, d),
            Pt3::new(-0.4, 0.2, d),
            Pt3::new(0.6, -0.3, d),
            Pt3::new(-0.3, -0.4, d),
        ];

        let pts1: Vec<_> = plane_pts
            .iter()
            .map(|p| Pt2::new(p.x / p.z, p.y / p.z))
            .collect();
        let pts2: Vec<_> = pts1
            .iter()
            .map(|p| crate::homography::homography_transfer(&h_gt, p))
            .collect();

        let h_refined = refine_homography(&h_gt, &pts1, &pts2).unwrap();
        let err = (h_refined - h_gt).norm();
        assert!(err < 1e-3, "perfect init diverged: error = {}", err);
    }
}
