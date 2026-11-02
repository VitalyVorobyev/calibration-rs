//! Epipolar and homography residual functions.
//!
//! These residuals measure how well a fundamental/essential matrix or
//! homography explains a pair of corresponding points.

use crate::types::Correspondence2D;
use vision_calibration_core::{Mat3, Real, Vec3, from_homogeneous, to_homogeneous};

/// Algebraic epipolar residual: `x₂ᵀ F x₁`.
///
/// This is the simplest residual — it equals zero when the correspondence
/// exactly satisfies the epipolar constraint. Not geometrically meaningful
/// on its own, but fast and useful for algebraic minimization.
pub fn algebraic_residual(f: &Mat3, c: &Correspondence2D) -> Real {
    let x1 = nalgebra::Vector3::new(c.pt1.x, c.pt1.y, 1.0);
    let x2 = nalgebra::Vector3::new(c.pt2.x, c.pt2.y, 1.0);
    (x2.transpose() * f * x1)[0]
}

/// Sampson distance (first-order geometric approximation).
///
/// This is a good approximation to the true geometric distance and is
/// much cheaper to compute. It equals the reprojection error to first
/// order in the noise.
///
/// Returns the **squared** Sampson distance.
pub fn sampson_distance_squared(f: &Mat3, c: &Correspondence2D) -> Real {
    let x1 = nalgebra::Vector3::new(c.pt1.x, c.pt1.y, 1.0);
    let x2 = nalgebra::Vector3::new(c.pt2.x, c.pt2.y, 1.0);

    let fx1 = f * x1;
    let ftx2 = f.transpose() * x2;

    let xtfx = (x2.transpose() * f * x1)[0];

    let denom = fx1.x * fx1.x + fx1.y * fx1.y + ftx2.x * ftx2.x + ftx2.y * ftx2.y;
    if denom.abs() < Real::EPSILON {
        return Real::INFINITY;
    }

    (xtfx * xtfx) / denom
}

/// Sampson distance (first-order geometric approximation).
///
/// Returns the Sampson distance (not squared).
pub fn sampson_distance(f: &Mat3, c: &Correspondence2D) -> Real {
    sampson_distance_squared(f, c).abs().sqrt()
}

/// Symmetric epipolar distance.
///
/// Geometric distance from each point to its corresponding epipolar line,
/// averaged over both images:
///
/// `d² = (x₂ᵀFx₁)² * (1/(Fx₁)₁² + (Fx₁)₂²) + 1/((Fᵀx₂)₁² + (Fᵀx₂)₂²))`
///
/// Returns the **squared** symmetric epipolar distance.
pub fn symmetric_epipolar_distance_squared(f: &Mat3, c: &Correspondence2D) -> Real {
    let x1 = nalgebra::Vector3::new(c.pt1.x, c.pt1.y, 1.0);
    let x2 = nalgebra::Vector3::new(c.pt2.x, c.pt2.y, 1.0);

    let fx1 = f * x1;
    let ftx2 = f.transpose() * x2;

    let xtfx = (x2.transpose() * f * x1)[0];
    let xtfx2 = xtfx * xtfx;

    let d1_sq = fx1.x * fx1.x + fx1.y * fx1.y;
    let d2_sq = ftx2.x * ftx2.x + ftx2.y * ftx2.y;

    if d1_sq.abs() < Real::EPSILON || d2_sq.abs() < Real::EPSILON {
        return Real::INFINITY;
    }

    xtfx2 / d1_sq + xtfx2 / d2_sq
}

/// Symmetric epipolar distance (not squared).
pub fn symmetric_epipolar_distance(f: &Mat3, c: &Correspondence2D) -> Real {
    symmetric_epipolar_distance_squared(f, c).abs().sqrt()
}

/// Point-to-epipolar-line distance in each image.
///
/// Returns `(d1, d2)` where:
/// - `d1` = distance from `pt2` to the epipolar line `F x₁` in image 2
/// - `d2` = distance from `pt1` to the epipolar line `Fᵀ x₂` in image 1
pub fn epipolar_line_distance(f: &Mat3, c: &Correspondence2D) -> (Real, Real) {
    let x1 = nalgebra::Vector3::new(c.pt1.x, c.pt1.y, 1.0);
    let x2 = nalgebra::Vector3::new(c.pt2.x, c.pt2.y, 1.0);

    let l2 = f * x1; // epipolar line in image 2
    let l1 = f.transpose() * x2; // epipolar line in image 1

    let d1 = if l2.x * l2.x + l2.y * l2.y > Real::EPSILON {
        (l2.x * c.pt2.x + l2.y * c.pt2.y + l2.z).abs() / (l2.x * l2.x + l2.y * l2.y).sqrt()
    } else {
        Real::INFINITY
    };

    let d2 = if l1.x * l1.x + l1.y * l1.y > Real::EPSILON {
        (l1.x * c.pt1.x + l1.y * c.pt1.y + l1.z).abs() / (l1.x * l1.x + l1.y * l1.y).sqrt()
    } else {
        Real::INFINITY
    };

    (d1, d2)
}

/// First-order angular epipolar error of a bearing pair, in radians.
///
/// The Sampson distance on the sphere: with unit bearings `b1`, `b2` and an
/// essential matrix `E` (`b2ᵀ E b1 = 0`),
///
/// `e² = (b2ᵀ E b1)² / (‖(I − b1 b1ᵀ) Eᵀ b2‖² + ‖(I − b2 b2ᵀ) E b1‖²)`.
///
/// It is the angle by which the two rays would have to move to meet the
/// epipolar constraint, to first order, and does not change with the scale of
/// `E`. Returns `INFINITY` when the denominator vanishes (relative to `‖E‖²`),
/// i.e. a bearing at an epipole, where the constraint carries no information.
pub(crate) fn angular_epipolar_error(e: &Mat3, b1: &Vec3, b2: &Vec3) -> Real {
    let e_b1 = e * b1;
    let et_b2 = e.transpose() * b2;
    let num = b2.dot(&e_b1);
    let g1 = et_b2 - b1 * b1.dot(&et_b2);
    let g2 = e_b1 - b2 * b2.dot(&e_b1);
    let denom = g1.norm_squared() + g2.norm_squared();
    if !denom.is_finite() || denom <= Real::EPSILON * e.norm_squared() {
        return Real::INFINITY;
    }
    (num * num / denom).sqrt()
}

/// Symmetric transfer error for a homography.
///
/// Measures how well `H` maps `pt1 → pt2` and `H⁻¹` maps `pt2 → pt1`:
///
/// `error² = |H x₁ - x₂|² + |H⁻¹ x₂ - x₁|²`
///
/// Returns the **squared** symmetric transfer error, or `INFINITY` if `H`
/// is not invertible.
pub fn symmetric_transfer_error_squared(h: &Mat3, c: &Correspondence2D) -> Real {
    let h_inv = match h.try_inverse() {
        Some(inv) => inv,
        None => return Real::INFINITY,
    };

    let x1h = to_homogeneous(&c.pt1);
    let x2h = to_homogeneous(&c.pt2);

    let fwd = from_homogeneous(&(h * x1h));
    let bwd = from_homogeneous(&(h_inv * x2h));

    let d_fwd = (fwd.x - c.pt2.x).powi(2) + (fwd.y - c.pt2.y).powi(2);
    let d_bwd = (bwd.x - c.pt1.x).powi(2) + (bwd.y - c.pt1.y).powi(2);

    d_fwd + d_bwd
}

/// Symmetric transfer error for a homography (not squared).
pub fn symmetric_transfer_error(h: &Mat3, c: &Correspondence2D) -> Real {
    symmetric_transfer_error_squared(h, c).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Rotation3;
    use vision_calibration_core::{Pt2, Pt3, Vec3};

    fn skew(v: &Vec3) -> Mat3 {
        Mat3::new(0.0, -v.z, v.y, v.z, 0.0, -v.x, -v.y, v.x, 0.0)
    }

    fn make_essential() -> (Mat3, Vec<Correspondence2D>) {
        let rot = Rotation3::from_euler_angles(0.1, -0.05, 0.02);
        let t = Vec3::new(0.3, 0.05, 0.01);
        let e = skew(&t) * rot.matrix();

        let world = [
            Pt3::new(0.1, 0.2, 2.0),
            Pt3::new(-0.2, 0.1, 2.5),
            Pt3::new(0.3, -0.1, 3.0),
            Pt3::new(-0.15, -0.2, 2.2),
            Pt3::new(0.05, 0.3, 2.8),
        ];

        let corrs: Vec<_> = world
            .iter()
            .map(|pw| {
                let pc1 = pw.coords;
                let pc2 = rot * pw + t;
                Correspondence2D::new(
                    Pt2::new(pc1.x / pc1.z, pc1.y / pc1.z),
                    Pt2::new(pc2.x / pc2.z, pc2.y / pc2.z),
                )
            })
            .collect();

        (e, corrs)
    }

    #[test]
    fn algebraic_residual_zero_for_exact_correspondences() {
        let (e, corrs) = make_essential();
        for c in &corrs {
            let r = algebraic_residual(&e, c);
            assert!(r.abs() < 1e-10, "algebraic residual too large: {}", r);
        }
    }

    #[test]
    fn sampson_distance_zero_for_exact_correspondences() {
        let (e, corrs) = make_essential();
        for c in &corrs {
            let d = sampson_distance(&e, c);
            assert!(d < 1e-10, "Sampson distance too large: {}", d);
        }
    }

    #[test]
    fn symmetric_epipolar_distance_zero_for_exact() {
        let (e, corrs) = make_essential();
        for c in &corrs {
            let d = symmetric_epipolar_distance(&e, c);
            assert!(d < 1e-10, "symmetric distance too large: {}", d);
        }
    }

    #[test]
    fn epipolar_line_distance_zero_for_exact() {
        let (e, corrs) = make_essential();
        for c in &corrs {
            let (d1, d2) = epipolar_line_distance(&e, c);
            assert!(d1 < 1e-10, "d1 too large: {}", d1);
            assert!(d2 < 1e-10, "d2 too large: {}", d2);
        }
    }

    #[test]
    fn symmetric_transfer_error_zero_for_exact_homography() {
        // Scale homography: H = diag(2, 2, 1)
        let h = Mat3::new(2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0);

        let c = Correspondence2D::new(Pt2::new(1.0, 1.0), Pt2::new(2.0, 2.0));
        let err = symmetric_transfer_error(&h, &c);
        assert!(err < 1e-10, "transfer error too large: {}", err);
    }

    #[test]
    fn sampson_nonzero_for_wrong_correspondence() {
        let (e, _corrs) = make_essential();
        let bad = Correspondence2D::new(Pt2::new(0.5, 0.5), Pt2::new(-0.3, 0.8));
        let d = sampson_distance(&e, &bad);
        assert!(d > 1e-3, "expected nonzero Sampson for bad correspondence");
    }

    #[test]
    fn angular_epipolar_error_is_zero_on_exact_bearings_and_scale_invariant() {
        let rot = Rotation3::from_euler_angles(0.1, -0.05, 0.02);
        let t = Vec3::new(0.3, 0.05, 0.01);
        let e = skew(&t) * rot.matrix();
        // One point behind the first camera.
        let world = [Pt3::new(0.1, 0.2, 2.0), Pt3::new(-1.5, 0.3, -0.8)];
        for pw in &world {
            let b1 = pw.coords.normalize();
            let b2 = (rot * pw + t).coords.normalize();
            assert!(angular_epipolar_error(&e, &b1, &b2) < 1e-12);

            // A bearing moved by an angle δ is penalized by at most δ (the
            // first-order distance is the smallest move that fixes the pair).
            let delta = 1e-3;
            let axis = b2.cross(&Vec3::new(0.3, -0.5, 0.8)).normalize();
            let moved = Rotation3::from_scaled_axis(axis * delta) * b2;
            let err = angular_epipolar_error(&e, &b1, &moved);
            assert!(err > 0.0 && err <= delta * (1.0 + 1e-6), "err {err}");
            let err_scaled = angular_epipolar_error(&(e * 40.0), &b1, &moved);
            assert!((err - err_scaled).abs() < 1e-12 * err.max(1.0));
        }
    }
}
