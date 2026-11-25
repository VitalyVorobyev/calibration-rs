//! P3P (Perspective-3-Point) minimal solver for camera pose estimation.
//!
//! Solves for camera pose from exactly three non-collinear 3D-2D point
//! correspondences using Grunert's / Fischler-Bolles' law-of-cosines quartic
//! formulation and Kabsch alignment. Returns up to four positive-depth
//! candidate poses that must be disambiguated.

use super::pose_utils::pose_from_points;
use crate::GeometryError;
use crate::math::ensure_finite;
use nalgebra::Vector3;
use vision_calibration_core::linalg::{solve_quadratic_real, solve_quartic_real};
use vision_calibration_core::{FxFyCxCy, Iso3, Mat3, Pt2, Pt3, Real};

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
///
/// # Errors
///
/// Returns [`GeometryError::CountMismatch`] if `world.len() != image.len()`,
/// [`GeometryError::InsufficientData`] if fewer than 3 correspondences are
/// given, [`GeometryError::InvalidInput`] if more than 3 correspondences are
/// given or any input is non-finite, [`GeometryError::Degenerate`] if the 3D
/// points are coincident or collinear, or [`GeometryError::Singular`] if `k` is
/// non-invertible.
pub fn p3p(world: &[Pt3], image: &[Pt2], k: &FxFyCxCy<Real>) -> Result<Vec<Iso3>, GeometryError> {
    ensure_finite("world point", world.iter().flat_map(|p| p.coords.iter()))?;
    ensure_finite("image point", image.iter().flat_map(|p| p.coords.iter()))?;
    ensure_finite("intrinsics", [&k.fx, &k.fy, &k.cx, &k.cy])?;
    if world.len() != image.len() {
        return Err(GeometryError::CountMismatch {
            expected: world.len(),
            got: image.len(),
        });
    }
    if world.len() < 3 {
        return Err(GeometryError::InsufficientData {
            need: 3,
            got: world.len(),
        });
    }
    if world.len() > 3 {
        return Err(GeometryError::invalid_input(format!(
            "invalid number of correspondences: expected 3, got {}",
            world.len()
        )));
    }

    let kmtx: Mat3 = k.k_matrix();
    let k_inv = kmtx.try_inverse().ok_or(GeometryError::Singular)?;

    let mut bearings = Vec::with_capacity(3);
    for pi in image {
        let v = k_inv * Vector3::new(pi.x, pi.y, 1.0);
        bearings.push(v.normalize());
    }

    let a = (world[1] - world[2]).norm(); // BC
    let b = (world[0] - world[2]).norm(); // AC
    let c = (world[0] - world[1]).norm(); // AB

    let max_side = a.max(b).max(c);
    if max_side <= Real::EPSILON
        || a <= 1e-12 * max_side
        || b <= 1e-12 * max_side
        || c <= 1e-12 * max_side
    {
        return Err(GeometryError::degenerate(
            "coincident 3D points in P3P input",
        ));
    }
    let ab = world[1] - world[0];
    let ac = world[2] - world[0];
    if ab.cross(&ac).norm() <= 1e-9 * max_side * max_side {
        return Err(GeometryError::degenerate(
            "collinear 3D points in P3P input",
        ));
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

    let mut roots = solve_quartic_real(coeffs[4], coeffs[3], coeffs[2], coeffs[1], coeffs[0]);
    // When D(u) = 2(cos_beta - u cos_alpha) and N(u) share a root u_sing = cos_beta / cos_alpha
    // (symmetric triangle: |BC| == |AC| and cos_alpha == cos_beta), u_sing is a double root
    // of the quartic, so companion eigenvalues can drift into small complex pairs or ~1e-8 offsets.
    if cos_alpha.abs() > 1e-12 {
        let u_sing = cos_beta / cos_alpha;
        let n_sing = n0 + n1 * u_sing + n2 * u_sing * u_sing;
        if u_sing > 0.0 && n_sing.abs() <= 1e-9 * (1.0 + u_sing * u_sing) {
            roots.retain(|&u| (u - u_sing).abs() > 1e-4);
            roots.push(u_sing);
        }
    }
    if roots.is_empty() {
        return Err(GeometryError::numerical(
            "failed to solve the P3P polynomial system",
        ));
    }

    let mut solutions = Vec::new();
    let a2_over_c2 = a2 / c2;
    for u in roots {
        if !u.is_finite() || u <= 0.0 {
            continue;
        }
        let k_val = 1.0 + u * u - 2.0 * u * cos_gamma;
        if k_val <= 1e-12 {
            continue;
        }

        let x2 = c2 / k_val;
        if !x2.is_finite() || x2 <= 0.0 {
            continue;
        }
        let x = x2.sqrt();
        let y = u * x;

        let den = 2.0 * (cos_beta - u * cos_alpha);
        let mut v_candidates = Vec::new();
        if den.abs() >= 1e-5 {
            let n_val = n0 + n1 * u + n2 * u * u;
            v_candidates.push(n_val / den);
        } else {
            // Near the common root of N(u) and D(u), recover v directly from the
            // triangle AC quadratic v^2 - 2 cos_beta v + (1 - e * k_val) = 0.
            let q_roots = solve_quadratic_real(1.0, -2.0 * cos_beta, 1.0 - e * k_val);
            for v in q_roots {
                let err_bc = (u * u + v * v - 2.0 * u * v * cos_alpha - a2_over_c2 * k_val).abs();
                if err_bc <= 1e-6 * (1.0 + u * u + v * v + a2_over_c2 * k_val) {
                    v_candidates.push(v);
                }
            }
        }

        for v in v_candidates {
            if !v.is_finite() || v <= 0.0 {
                continue;
            }
            let z = v * x;

            let pc1 = bearings[0] * x;
            let pc2 = bearings[1] * y;
            let pc3 = bearings[2] * z;

            if let Ok(pose) = pose_from_points(world, &[pc1, pc2, pc3]) {
                solutions.push((x, pose));
            }
        }
    }

    if solutions.is_empty() {
        return Err(GeometryError::numerical(
            "failed to solve the P3P polynomial system",
        ));
    }

    solutions.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    Ok(solutions.into_iter().map(|(_, pose)| pose).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Isometry3, Rotation3, Translation3};
    use vision_calibration_core::{Camera, IdentitySensor, NoDistortion, Pinhole};

    #[test]
    fn p3p_recovers_pose_from_minimal_set() {
        let k = FxFyCxCy {
            fx: 800.0,
            fy: 780.0,
            cx: 640.0,
            cy: 360.0,
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

    #[test]
    fn p3p_handles_symmetric_triangle_and_filters_negative_depths() {
        let k = FxFyCxCy {
            fx: 800.0,
            fy: 800.0,
            cx: 400.0,
            cy: 300.0,
        };
        let cam = Camera::new(Pinhole, NoDistortion, IdentitySensor, k);
        // Centered fronto-parallel isosceles triangle: a == b and cos_alpha == cos_beta.
        let iso_gt = Isometry3::from_parts(
            Translation3::new(0.0, 0.0, 1.5),
            Rotation3::identity().into(),
        );
        let world = vec![
            Pt3::new(-0.2, -0.1, 0.0),
            Pt3::new(0.2, -0.1, 0.0),
            Pt3::new(0.0, 0.25, 0.0),
        ];
        let image: Vec<Pt2> = world
            .iter()
            .map(|p| cam.project_point(&iso_gt.transform_point(p)).unwrap())
            .collect();
        let sols = p3p(&world, &image, &k).unwrap();
        assert!(!sols.is_empty());
        for est in &sols {
            for p in &world {
                assert!(
                    est.transform_point(p).z > 0.0,
                    "candidate placed point behind camera"
                );
            }
        }
        let best_dt = sols
            .iter()
            .map(|e| (e.translation.vector - iso_gt.translation.vector).norm())
            .fold(f64::INFINITY, f64::min);
        assert!(best_dt < 1e-6, "symmetric triangle missed GT: dt={best_dt}");
    }

    #[test]
    fn p3p_rejects_collinear_points() {
        let k = FxFyCxCy {
            fx: 800.0,
            fy: 800.0,
            cx: 400.0,
            cy: 300.0,
        };
        let world = vec![
            Pt3::new(0.0, 0.0, 1.0),
            Pt3::new(0.1, 0.2, 1.0),
            Pt3::new(0.2, 0.4, 1.0),
        ];
        let image = vec![
            Pt2::new(400.0, 300.0),
            Pt2::new(480.0, 460.0),
            Pt2::new(560.0, 620.0),
        ];
        assert!(matches!(
            p3p(&world, &image, &k),
            Err(GeometryError::Degenerate { .. })
        ));
    }
}
