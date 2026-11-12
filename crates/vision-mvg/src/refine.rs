//! Nonlinear refinement of geometric estimates, through the problem builder
//! of `vision-calibration-optim`.
//!
//! - [`refine_homography`]: minimize symmetric transfer error

use crate::{Error, Result};
use nalgebra::{DVector, RealField};
use vision_calibration_core::{Mat3, Pt2, Real};
use vision_calibration_optim::RobustLoss;
use vision_calibration_optim::problem::{Manifold, Noise, Problem, Residual, SolveOptions, solve};

/// The symmetric transfer error of one correspondence under H: the four
/// pixel residuals of `H·p1 − p2` and `H⁻¹·p2 − p1`.
struct HomographyTransfer {
    pt1: Pt2,
    pt2: Pt2,
}

impl Residual for HomographyTransfer {
    fn dim(&self) -> usize {
        4
    }

    fn block_dims(&self) -> Vec<usize> {
        vec![9]
    }

    fn eval<T: RealField>(&self, blocks: &[DVector<T>]) -> DVector<T> {
        let h = &blocks[0]; // 9 elements, row-major
        let c = |v: Real| T::from_f64(v).unwrap();
        let (x1, y1, x2, y2) = (c(self.pt1.x), c(self.pt1.y), c(self.pt2.x), c(self.pt2.y));

        // Forward: H * [x1, y1, 1]
        let fx = h[0].clone() * x1.clone() + h[1].clone() * y1.clone() + h[2].clone();
        let fy = h[3].clone() * x1.clone() + h[4].clone() * y1.clone() + h[5].clone();
        let fw = h[6].clone() * x1.clone() + h[7].clone() * y1.clone() + h[8].clone();
        let dx_fwd = fx / fw.clone() - x2.clone();
        let dy_fwd = fy / fw - y2.clone();

        // Inverse: H^-1 * [x2, y2, 1] via the adjugate (the determinant
        // cancels in the homogeneous division).
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
/// returns the refined homography, scaled so `h[2,2] = 1` where that entry is
/// not zero. A homography is defined up to scale: the solve fixes the entry
/// of `h_init` that is largest in magnitude, so the refinement has no scale
/// freedom. The residuals are in the units of the points and weigh equally.
///
/// # Errors
///
/// [`Error::InsufficientData`] for fewer than 4 correspondences,
/// [`Error::CountMismatch`] for different numbers of points,
/// [`Error::InvalidInput`] for a non-finite or all-zero `h_init`, and
/// [`Error::Numerical`] when the solve fails to run.
pub fn refine_homography(h_init: &Mat3, pts1: &[Pt2], pts2: &[Pt2]) -> Result<Mat3> {
    if pts1.len() < 4 {
        return Err(Error::InsufficientData {
            need: 4,
            got: pts1.len(),
        });
    }
    if pts1.len() != pts2.len() {
        return Err(Error::CountMismatch {
            expected: pts1.len(),
            got: pts2.len(),
        });
    }
    // Row-major, scaled so the largest-magnitude entry is +-1 (kept fixed).
    let mut h_vec: Vec<Real> = (0..3)
        .flat_map(|r| (0..3).map(move |c| (r, c)))
        .map(|(r, c)| h_init[(r, c)])
        .collect();
    if h_vec.iter().any(|v| !v.is_finite()) {
        return Err(Error::invalid_input("the initial homography is not finite"));
    }
    let (anchor, largest) = h_vec.iter().enumerate().fold((0, 0.0_f64), |best, (i, v)| {
        if v.abs() > best.1 { (i, v.abs()) } else { best }
    });
    if largest == 0.0 {
        return Err(Error::invalid_input("the initial homography is all zero"));
    }
    for v in &mut h_vec {
        *v /= largest;
    }

    let numerical = |e: vision_calibration_optim::Error| {
        Error::numerical(format!("homography refinement: {e}"))
    };
    let mut problem = Problem::new();
    let block = problem
        .add_block("h", &h_vec, Manifold::Euclidean)
        .map_err(numerical)?;
    problem
        .fix_components(block, &[anchor])
        .map_err(numerical)?;
    for (p1, p2) in pts1.iter().zip(pts2) {
        problem
            .add_residual(
                HomographyTransfer { pt1: *p1, pt2: *p2 },
                &[block],
                RobustLoss::None,
                Noise::Stated(1.0),
            )
            .map_err(numerical)?;
    }
    let mut options = SolveOptions::default();
    options.max_iters = 50;
    let solution = solve(&problem, &options).map_err(numerical)?;
    let values = solution
        .value(block)
        .ok_or_else(|| Error::numerical("homography refinement: no solution block"))?;
    let mut h = Mat3::from_row_slice(values);

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
