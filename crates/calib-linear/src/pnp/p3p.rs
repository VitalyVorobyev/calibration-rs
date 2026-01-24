//! P3P (Perspective-3-Point) minimal solver for camera pose estimation.
//!
//! Solves for camera pose from exactly three non-collinear 3D-2D point
//! correspondences using Kneip's algebraic solution. Returns up to four
//! candidate poses that must be disambiguated.

use super::pose_utils::pose_from_points;
use crate::math::solve_quartic_real;
use anyhow::Result;
use calib_core::{FxFyCxCySkew, Iso3, Mat3, Pt2, Pt3, Real};
use nalgebra::Vector3;

/// Multiply two degree-4 polynomials (truncate to degree 4).
fn poly_mul_1d(a: &[Real; 5], b: &[Real; 5]) -> [Real; 5] {
    let mut out = [0.0; 5];
    for i in 0..5 {
        for j in 0..5 {
            if i + j > 4 {
                continue;
            }
            out[i + j] += a[i] * b[j];
        }
    }
    out
}

/// P3P minimal solver: returns up to four pose candidates.
///
/// Requires exactly three non-collinear points and intrinsics `k` to
/// convert pixels into rays. The resulting poses are in `T_C_W` form.
pub fn p3p(world: &[Pt3], image: &[Pt2], k: &FxFyCxCySkew<Real>) -> Result<Vec<Iso3>> {
    if world.len() != image.len() {
        anyhow::bail!(
            "invalid number of correspondences: expected 3, got {}",
            world.len().max(image.len())
        );
    }
    if world.len() != 3 {
        anyhow::bail!(
            "invalid number of correspondences: expected 3, got {}",
            world.len()
        );
    }

    let kmtx: Mat3 = k.k_matrix();
    let k_inv = kmtx
        .try_inverse()
        .ok_or_else(|| anyhow::anyhow!("intrinsics matrix is not invertible"))?;

    let mut bearings = Vec::with_capacity(3);
    for pi in image {
        let v = k_inv * Vector3::new(pi.x, pi.y, 1.0);
        bearings.push(v.normalize());
    }

    let a = (world[1] - world[2]).norm(); // BC
    let b = (world[0] - world[2]).norm(); // AC
    let c = (world[0] - world[1]).norm(); // AB

    if a <= Real::EPSILON || b <= Real::EPSILON || c <= Real::EPSILON {
        anyhow::bail!("degenerate 3d point configuration for normalization");
    }

    let cos_alpha = bearings[1].dot(&bearings[2]);
    let cos_beta = bearings[0].dot(&bearings[2]);
    let cos_gamma = bearings[0].dot(&bearings[1]);

    let a2 = a * a;
    let b2 = b * b;
    let c2 = c * c;

    let d = (b2 - a2) / c2;
    let e = b2 / c2;

    let n0 = 1.0 - d;
    let n1 = 2.0 * d * cos_gamma;
    let n2 = -(1.0 + d);

    let d0 = 2.0 * cos_beta;
    let d1 = -2.0 * cos_alpha;

    let e0 = 1.0 - e;
    let e1 = 2.0 * e * cos_gamma;
    let e2 = -e;

    let n_poly = [n0, n1, n2, 0.0, 0.0];
    let d_poly = [d0, d1, 0.0, 0.0, 0.0];
    let e_poly = [e0, e1, e2, 0.0, 0.0];

    let n2_poly = poly_mul_1d(&n_poly, &n_poly);
    let nd_poly = poly_mul_1d(&n_poly, &d_poly);
    let d2_poly = poly_mul_1d(&d_poly, &d_poly);
    let ed2_poly = poly_mul_1d(&e_poly, &d2_poly);

    let mut coeffs = [0.0; 5];
    for i in 0..5 {
        coeffs[i] = n2_poly[i] - 2.0 * cos_beta * nd_poly[i] + ed2_poly[i];
    }

    let roots = solve_quartic_real(coeffs[4], coeffs[3], coeffs[2], coeffs[1], coeffs[0]);
    if roots.is_empty() {
        anyhow::bail!("failed to solve the P3P polynomial system");
    }

    let mut solutions = Vec::new();
    for u in roots {
        let den = 2.0 * (cos_beta - u * cos_alpha);
        if den.abs() < 1e-12 {
            continue;
        }

        let k_val = 1.0 + u * u - 2.0 * u * cos_gamma;
        if k_val.abs() < 1e-12 {
            continue;
        }

        let n_val = n0 + n1 * u + n2 * u * u;
        let v = n_val / den;

        let x2 = c2 / k_val;
        if x2 <= 0.0 {
            continue;
        }
        let x = x2.sqrt();
        let y = u * x;
        let z = v * x;

        let pc1 = bearings[0] * x;
        let pc2 = bearings[1] * y;
        let pc3 = bearings[2] * z;

        if let Ok(pose) = pose_from_points(world, &[pc1, pc2, pc3]) {
            solutions.push((x, pose));
        }
    }

    if solutions.is_empty() {
        anyhow::bail!("failed to solve the P3P polynomial system");
    }

    solutions.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    Ok(solutions.into_iter().map(|(_, pose)| pose).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use calib_core::{Camera, IdentitySensor, NoDistortion, Pinhole};
    use nalgebra::{Isometry3, Rotation3, Translation3};

    #[test]
    fn p3p_recovers_pose_from_minimal_set() {
        let k = FxFyCxCySkew {
            fx: 800.0,
            fy: 780.0,
            cx: 640.0,
            cy: 360.0,
            skew: 0.0,
        };
        let cam = Camera::new(Pinhole, NoDistortion, IdentitySensor, k);

        let rot = Rotation3::from_euler_angles(0.1, -0.05, 0.2);
        let t = Translation3::new(0.1, -0.05, 1.0);
        let iso_gt = Isometry3::from_parts(t, rot.into());

        let world = vec![
            Pt3::new(0.2, -0.1, 0.8),
            Pt3::new(-0.1, 0.2, 1.1),
            Pt3::new(0.15, 0.1, 0.9),
        ];

        let mut image = Vec::new();
        for pw in &world {
            let pc = iso_gt.transform_point(pw);
            let uv = cam.project_point(&pc).unwrap();
            image.push(uv);
        }

        let sols = p3p(&world, &image, &k).unwrap();
        assert!(!sols.is_empty());

        let mut best_dt = f64::INFINITY;
        let mut best_ang = f64::INFINITY;
        for est in sols {
            let dt = (est.translation.vector - iso_gt.translation.vector).norm();
            let r_est = est.rotation.to_rotation_matrix();
            let r_gt = iso_gt.rotation.to_rotation_matrix();
            let r_diff = r_est.transpose() * r_gt;
            let trace = r_diff.matrix().trace();
            let cos_theta = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0);
            let ang = cos_theta.acos();
            best_dt = best_dt.min(dt);
            best_ang = best_ang.min(ang);
        }

        assert!(best_dt < 1e-4, "translation error too large: {}", best_dt);
        assert!(best_ang < 1e-4, "rotation error too large: {}", best_ang);
    }
}
