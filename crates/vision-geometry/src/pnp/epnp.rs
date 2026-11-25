//! EPnP (Efficient Perspective-n-Point) solver for camera pose estimation.
//!
//! Implements Lepetit's control-point formulation for 4+ points. Uses
//! eigen decomposition of the world point covariance to define a minimal
//! basis, then solves for control point positions in the camera frame.

use super::planar::planar_pose;
use super::pose_utils::pose_from_points;
use crate::GeometryError;
use crate::math::ensure_finite;
use nalgebra::{SMatrix, SVector, Vector3, linalg::SymmetricEigen};
use vision_calibration_core::{FxFyCxCy, Iso3, Mat3, Pt2, Pt3, Real, Vec3};

/// A spread along a principal axis below this fraction (as a variance ratio,
/// `1e-8` is `1e-4` in length) of the largest makes the points flat (or, for
/// two axes, collinear).
const FLAT_RATIO: Real = 1e-8;

/// EPnP pose estimation for 4+ points.
///
/// Uses a control-point formulation derived from the covariance of the
/// 3D points. Returns a single pose estimate in `T_C_W` form. Points on a
/// plane (a calibration board) are solved from the plane's homography, which
/// the control-point formulation cannot do; collinear points are an error.
///
/// # Errors
///
/// Returns [`GeometryError::CountMismatch`] if `world.len() != image.len()`,
/// [`GeometryError::InsufficientData`] if fewer than 4 correspondences are
/// given, [`GeometryError::Degenerate`] if the points are collinear, or
/// [`GeometryError::Singular`] if the intrinsics `k` are non-invertible or the
/// control-point system is degenerate.
pub fn epnp(world: &[Pt3], image: &[Pt2], k: &FxFyCxCy<Real>) -> Result<Iso3, GeometryError> {
    ensure_finite("world point", world.iter().flat_map(|p| p.coords.iter()))?;
    ensure_finite("image point", image.iter().flat_map(|p| p.coords.iter()))?;
    ensure_finite("intrinsics", [&k.fx, &k.fy, &k.cx, &k.cy])?;
    let n = world.len();
    if image.len() != n {
        return Err(GeometryError::CountMismatch {
            expected: n,
            got: image.len(),
        });
    }
    if n < 4 {
        return Err(GeometryError::InsufficientData { need: 4, got: n });
    }

    let kmtx: Mat3 = k.k_matrix();
    let k_inv = kmtx.try_inverse().ok_or(GeometryError::Singular)?;

    let mut img_norm = Vec::with_capacity(n);
    for pi in image {
        let v = k_inv * Vector3::new(pi.x, pi.y, 1.0);
        img_norm.push(Pt2::new(v.x / v.z, v.y / v.z));
    }

    let mut centroid = Vec3::zeros();
    for p in world {
        centroid += p.coords;
    }
    centroid /= n as Real;

    let mut cov = Mat3::zeros();
    for p in world {
        let d = p.coords - centroid;
        cov += d * d.transpose();
    }
    cov /= n as Real;

    let eig = SymmetricEigen::new(cov);
    let axes = eig.eigenvectors;
    let vals = eig.eigenvalues;

    // The control points are the centroid and the principal axes scaled by
    // the spread along them, so a flat point set has a degenerate fourth
    // control point: a plane is solved from its homography instead.
    let mut order = [0, 1, 2];
    order.sort_by(|&a, &b| vals[b].total_cmp(&vals[a]));
    let [l0, l1, l2] = order.map(|i| vals[i].max(0.0));
    if l0 <= 0.0 || l1 <= FLAT_RATIO * l0 {
        return Err(GeometryError::Degenerate {
            reason: "the world points are collinear".into(),
        });
    }
    if l2 <= FLAT_RATIO * l0 {
        let e1 = axes.column(order[0]).into_owned();
        let e2 = axes.column(order[1]).into_owned();
        let plane_axes = Mat3::from_columns(&[e1, e2, e1.cross(&e2)]);
        return planar_pose(world, &img_norm, &centroid, &plane_axes);
    }

    let mut control_w = [Vec3::zeros(); 4];
    control_w[0] = centroid;
    for i in 0..3 {
        let scale = vals[i].abs().sqrt();
        let axis = axes.column(i).into_owned();
        control_w[i + 1] = centroid + axis * scale;
    }

    let basis = Mat3::from_columns(&[
        control_w[1] - control_w[0],
        control_w[2] - control_w[0],
        control_w[3] - control_w[0],
    ]);
    let basis_inv = basis.try_inverse().ok_or(GeometryError::Singular)?;

    let mut alphas = Vec::with_capacity(n);
    for p in world {
        let coeff = basis_inv * (p.coords - control_w[0]);
        let a0 = 1.0 - coeff.x - coeff.y - coeff.z;
        alphas.push([a0, coeff.x, coeff.y, coeff.z]);
    }

    let mut ata = SMatrix::<Real, 12, 12>::zeros();
    for (a, uv) in alphas.iter().zip(img_norm.iter()) {
        let u = uv.x;
        let v = uv.y;
        let mut r0 = SVector::<Real, 12>::zeros();
        let mut r1 = SVector::<Real, 12>::zeros();
        for (j, &alpha) in a.iter().enumerate().take(4) {
            let c = 3 * j;
            r0[c] = alpha;
            r0[c + 2] = -u * alpha;
            r1[c + 1] = alpha;
            r1[c + 2] = -v * alpha;
        }
        ata += r0 * r0.transpose() + r1 * r1.transpose();
    }

    if ata.iter().any(|v| !v.is_finite()) {
        return Err(GeometryError::Singular);
    }
    let eig = SymmetricEigen::new(ata);
    if eig.eigenvalues.iter().any(|v| !v.is_finite()) {
        return Err(GeometryError::Singular);
    }

    // Order basis vectors from smallest to largest eigenvalue (null space first).
    let mut null_order: [usize; 12] = std::array::from_fn(|i| i);
    null_order.sort_by(|&a, &b| eig.eigenvalues[a].total_cmp(&eig.eigenvalues[b]));
    let max_eig = eig.eigenvalues[null_order[11]];
    if !max_eig.is_finite() || max_eig <= Real::EPSILON {
        return Err(GeometryError::Singular);
    }

    let mut basis_vecs = [[Vec3::zeros(); 4]; 4];
    for (k_idx, &col_idx) in null_order.iter().take(4).enumerate() {
        let col = eig.eigenvectors.column(col_idx);
        for j in 0..4 {
            basis_vecs[k_idx][j] = Vec3::new(col[3 * j], col[3 * j + 1], col[3 * j + 2]);
        }
    }

    const PAIRS: [(usize, usize); 6] = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
    let mut rho = [0.0; 6];
    let mut dv = [[Vec3::zeros(); 6]; 4];
    for (p, &(i, j)) in PAIRS.iter().enumerate() {
        let d = control_w[i] - control_w[j];
        rho[p] = d.norm_squared();
        for k_idx in 0..4 {
            dv[k_idx][p] = basis_vecs[k_idx][i] - basis_vecs[k_idx][j];
        }
    }

    let sum_rho: Real = rho.iter().sum();
    if sum_rho <= Real::EPSILON {
        return Err(GeometryError::Singular);
    }

    // Collect candidate beta vectors from N = 1, 2, 3 (and N = 4 when the 4th
    // null vector is near zero, e.g. n = 4 non-coplanar points, or weak perspective).
    let mut candidates: Vec<[Real; 4]> = Vec::new();

    // N = 1: closed-form beta_0^2 minimizing sum (beta_0^2 * ||dv_0||^2 - rho)^2.
    {
        let mut num = 0.0;
        let mut den = 0.0;
        for p in 0..6 {
            let d2 = dv[0][p].norm_squared();
            num += d2 * rho[p];
            den += d2 * d2;
        }
        if den > Real::EPSILON && num > 0.0 {
            let b0 = [(num / den).sqrt(), 0.0, 0.0, 0.0];
            candidates.push(refine_betas(&dv, &rho, b0, 1));
        }
    }

    // N = 2: solve 6x3 system for [b0^2, b0*b1, b1^2].
    {
        let mut l63 = SMatrix::<Real, 6, 3>::zeros();
        let rho_vec = SVector::<Real, 6>::from_column_slice(&rho);
        for p in 0..6 {
            l63[(p, 0)] = dv[0][p].norm_squared();
            l63[(p, 1)] = 2.0 * dv[0][p].dot(&dv[1][p]);
            l63[(p, 2)] = dv[1][p].norm_squared();
        }
        if let Ok(b) = l63.svd(true, true).solve(&rho_vec, 1e-12) {
            let b00 = b[0].max(0.0);
            let b11 = b[2].max(0.0);
            let s1 = if b[1] >= 0.0 { 1.0 } else { -1.0 };
            let init = [b00.sqrt(), s1 * b11.sqrt(), 0.0, 0.0];
            if init[0] > 0.0 || init[1].abs() > 0.0 {
                candidates.push(refine_betas(&dv, &rho, init, 2));
            }
        }
    }

    // N = 3: solve 6x6 system for [b0^2, b0*b1, b1^2, b0*b2, b1*b2, b2^2].
    {
        let mut l66 = SMatrix::<Real, 6, 6>::zeros();
        let rho_vec = SVector::<Real, 6>::from_column_slice(&rho);
        for p in 0..6 {
            l66[(p, 0)] = dv[0][p].norm_squared();
            l66[(p, 1)] = 2.0 * dv[0][p].dot(&dv[1][p]);
            l66[(p, 2)] = dv[1][p].norm_squared();
            l66[(p, 3)] = 2.0 * dv[0][p].dot(&dv[2][p]);
            l66[(p, 4)] = 2.0 * dv[1][p].dot(&dv[2][p]);
            l66[(p, 5)] = dv[2][p].norm_squared();
        }
        if let Ok(b) = l66.svd(true, true).solve(&rho_vec, 1e-12) {
            let diag = [b[0].max(0.0), b[2].max(0.0), b[5].max(0.0)];
            let (anchor, _) = diag
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap();
            let mut init = [diag[0].sqrt(), diag[1].sqrt(), diag[2].sqrt(), 0.0];
            if anchor == 0 {
                if b[1] < 0.0 {
                    init[1] = -init[1];
                }
                if b[3] < 0.0 {
                    init[2] = -init[2];
                }
            } else if anchor == 1 {
                if b[1] < 0.0 {
                    init[0] = -init[0];
                }
                if b[4] < 0.0 {
                    init[2] = -init[2];
                }
            } else {
                if b[3] < 0.0 {
                    init[0] = -init[0];
                }
                if b[4] < 0.0 {
                    init[1] = -init[1];
                }
            }
            if init.iter().any(|v| v.abs() > 0.0) {
                candidates.push(refine_betas(&dv, &rho, init, 3));
            }
        }
    }

    // N = 4: when the 4D null space is genuinely active (n == 4 or small 4th eigenvalue).
    let use_n4 = n == 4 || eig.eigenvalues[null_order[3]] <= 1e-2 * max_eig;
    if use_n4 {
        let rho_vec = SVector::<Real, 6>::from_column_slice(&rho);
        // Four anchor linearizations [b_{aa}, b_{aj} for j != a].
        for a in 0..4 {
            let mut l64 = SMatrix::<Real, 6, 4>::zeros();
            let others: [usize; 3] = match a {
                0 => [1, 2, 3],
                1 => [0, 2, 3],
                2 => [0, 1, 3],
                _ => [0, 1, 2],
            };
            for p in 0..6 {
                l64[(p, 0)] = dv[a][p].norm_squared();
                for (c, &j) in others.iter().enumerate() {
                    l64[(p, c + 1)] = 2.0 * dv[a][p].dot(&dv[j][p]);
                }
            }
            if let Ok(b) = l64.svd(true, true).solve(&rho_vec, 1e-12)
                && b[0] > 0.0
            {
                let mut init = [0.0; 4];
                init[a] = b[0].sqrt();
                for (c, &j) in others.iter().enumerate() {
                    init[j] = b[c + 1] / init[a];
                }
                candidates.push(refine_betas(&dv, &rho, init, 4));
            }
        }
        // Plus all 8 sign orthants (up to global sign ±beta) for small minimal sets.
        if n == 4 {
            for mask in 0..8usize {
                let s = [
                    1.0,
                    if (mask & 1) != 0 { -1.0 } else { 1.0 },
                    if (mask & 2) != 0 { -1.0 } else { 1.0 },
                    if (mask & 4) != 0 { -1.0 } else { 1.0 },
                ];
                let mut sum_dv2 = 0.0;
                for p in 0..6 {
                    let comb =
                        dv[0][p] * s[0] + dv[1][p] * s[1] + dv[2][p] * s[2] + dv[3][p] * s[3];
                    sum_dv2 += comb.norm_squared();
                }
                if sum_dv2 > Real::EPSILON {
                    let scale = (sum_rho / sum_dv2).sqrt();
                    let init = s.map(|v| v * scale);
                    candidates.push(refine_betas(&dv, &rho, init, 4));
                }
            }
        }
    }

    let mut best_pose: Option<(Real, Iso3)> = None;
    for beta in candidates {
        let mut control_c = [Vec3::zeros(); 4];
        for j in 0..4 {
            let mut pt = Vec3::zeros();
            for k_idx in 0..4 {
                pt += basis_vecs[k_idx][j] * beta[k_idx];
            }
            control_c[j] = pt;
        }

        let mut sum_c = 0.0;
        for &(i, j) in &PAIRS {
            sum_c += (control_c[i] - control_c[j]).norm_squared();
        }
        if !sum_c.is_finite() || sum_c <= Real::EPSILON {
            continue;
        }
        let scale = (sum_rho / sum_c).sqrt();
        for cc in &mut control_c {
            *cc *= scale;
        }

        let mut camera_pts = Vec::with_capacity(n);
        for a in &alphas {
            let mut pc = Vec3::zeros();
            for (j, &alpha) in a.iter().enumerate().take(4) {
                pc += control_c[j] * alpha;
            }
            camera_pts.push(pc);
        }

        // The null vector is defined up to sign; pick the sign with positive sum of depths.
        if camera_pts.iter().map(|p| p.z).sum::<Real>() < 0.0 {
            for p in &mut camera_pts {
                *p = -*p;
            }
        }

        let Ok(pose) = pose_from_points(world, &camera_pts) else {
            continue;
        };

        // Score using normalized reprojection error and require positive depth.
        let mut err2 = 0.0;
        let mut valid = true;
        for (pw, uv) in world.iter().zip(img_norm.iter()) {
            let pc = pose * pw;
            if !pc.z.is_finite() || pc.z <= 1e-9 {
                valid = false;
                break;
            }
            let du = pc.x / pc.z - uv.x;
            let dv_err = pc.y / pc.z - uv.y;
            err2 += du * du + dv_err * dv_err;
        }
        if valid && err2.is_finite() && best_pose.as_ref().is_none_or(|(b, _)| err2 < *b) {
            best_pose = Some((err2, pose));
        }
    }

    best_pose.map(|(_, p)| p).ok_or(GeometryError::Singular)
}

/// Damped Gauss-Newton refinement of `beta[0..dim]` minimizing
/// `Σ (||Σ_k beta_k dv_{k,p}||^2 - rho_p)^2`.
fn refine_betas(
    dv: &[[Vec3; 6]; 4],
    rho: &[Real; 6],
    mut beta: [Real; 4],
    dim: usize,
) -> [Real; 4] {
    let mut lambda = 1e-10;
    let residual_cost = |b: &[Real; 4]| -> (SVector<Real, 6>, Real) {
        let mut r = SVector::<Real, 6>::zeros();
        let mut cost = 0.0;
        for p in 0..6 {
            let mut diff = Vec3::zeros();
            for k in 0..dim {
                diff += dv[k][p] * b[k];
            }
            let e = diff.norm_squared() - rho[p];
            r[p] = e;
            cost += e * e;
        }
        (r, cost)
    };

    let (mut r, mut cost) = residual_cost(&beta);
    for _ in 0..15 {
        let mut j_mat = SMatrix::<Real, 6, 4>::zeros();
        for p in 0..6 {
            let mut diff = Vec3::zeros();
            for k in 0..dim {
                diff += dv[k][p] * beta[k];
            }
            for k in 0..dim {
                j_mat[(p, k)] = 2.0 * diff.dot(&dv[k][p]);
            }
        }
        let jtj = j_mat.transpose() * j_mat;
        let g = j_mat.transpose() * r;
        let mut h4 = SMatrix::<Real, 4, 4>::identity();
        for i in 0..dim {
            for j in 0..dim {
                h4[(i, j)] = jtj[(i, j)];
            }
            h4[(i, i)] += lambda * (jtj[(i, i)].abs() + 1.0);
        }
        let mut rhs = SVector::<Real, 4>::zeros();
        for i in 0..dim {
            rhs[i] = -g[i];
        }
        let Some(delta) = h4.lu().solve(&rhs) else {
            break;
        };
        let mut next = beta;
        for i in 0..dim {
            next[i] += delta[i];
        }
        if delta.norm() <= 1e-12 * (beta.iter().map(|v| v * v).sum::<Real>().sqrt() + 1.0) {
            let (_, nc) = residual_cost(&next);
            if nc.is_finite() && nc < cost {
                beta = next;
            }
            break;
        }
        let (nr, nc) = residual_cost(&next);
        if nc.is_finite() && nc < cost {
            beta = next;
            r = nr;
            cost = nc;
            lambda = (lambda * 0.1).max(1e-14);
        } else {
            lambda *= 10.0;
            if lambda > 1e10 {
                break;
            }
        }
    }
    beta
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Isometry3, Rotation3, Translation3};
    use vision_calibration_core::{Camera, IdentitySensor, NoDistortion, Pinhole};

    #[test]
    fn epnp_recovers_pose_synthetic() {
        let k = FxFyCxCy {
            fx: 800.0,
            fy: 780.0,
            cx: 640.0,
            cy: 360.0,
        };
        let cam = Camera::new(Pinhole, NoDistortion, IdentitySensor, k);

        let rot = Rotation3::from_euler_angles(-0.1, 0.05, 0.2);
        let t = Translation3::new(0.1, -0.05, 1.2);
        let iso_gt = Isometry3::from_parts(t, rot.into());

        let mut world = Vec::new();
        let mut image = Vec::new();
        for z in 0..2 {
            for y in 0..3 {
                for x in 0..4 {
                    let pw = Pt3::new(x as Real * 0.1, y as Real * 0.1, 0.6 + z as Real * 0.1);
                    let pc = iso_gt.transform_point(&pw);
                    let uv = cam.project_point(&pc).unwrap();
                    world.push(pw);
                    image.push(uv);
                }
            }
        }

        let est = epnp(&world, &image, &k).unwrap();

        let dt = (est.translation.vector - iso_gt.translation.vector).norm();
        let r_est = est.rotation.to_rotation_matrix();
        let r_gt = iso_gt.rotation.to_rotation_matrix();
        let r_diff = r_est.transpose() * r_gt;
        let trace = r_diff.matrix().trace();
        let cos_theta = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0);
        let ang = cos_theta.acos();

        assert!(dt < 1e-3, "translation error too large: {}", dt);
        assert!(ang < 1e-3, "rotation error too large: {}", ang);
    }

    /// A shallow relief (not flat, so not the planar branch) in front of the
    /// camera: the null vector's sign must not put the points behind it.
    #[test]
    fn epnp_points_ahead_of_the_camera_for_a_shallow_object() {
        let k = FxFyCxCy {
            fx: 1.0,
            fy: 1.0,
            cx: 0.0,
            cy: 0.0,
        };
        let world: Vec<Pt3> = (0..35)
            .map(|i| {
                Pt3::new(
                    0.03 * f64::from(i % 7) - 0.09,
                    0.03 * f64::from(i / 7) - 0.06,
                    0.04 * f64::from(i * 5 % 7) / 6.0,
                )
            })
            .collect();
        let truth = Isometry3::from_parts(
            Translation3::new(0.02, -0.01, 0.7),
            Rotation3::from_euler_angles(0.25, -0.35, 0.1).into(),
        );
        let image: Vec<Pt2> = world
            .iter()
            .map(|p| {
                let c = truth * p;
                Pt2::new(c.x / c.z, c.y / c.z)
            })
            .collect();
        let est = epnp(&world, &image, &k).unwrap();
        assert!((est.translation.vector - truth.translation.vector).norm() < 1e-6);
        assert!(est.rotation.angle_to(&truth.rotation) < 1e-6);
    }

    /// Points on a plane are solved from the plane's homography.
    #[test]
    fn epnp_solves_a_planar_object_and_rejects_a_line() {
        let k = FxFyCxCy {
            fx: 800.0,
            fy: 800.0,
            cx: 640.0,
            cy: 360.0,
        };
        let truth = Isometry3::from_parts(
            Translation3::new(-0.05, 0.02, 0.9),
            Rotation3::from_euler_angles(0.3, 0.2, -0.4).into(),
        );
        let project = |p: &Pt3| {
            let c = truth * p;
            Pt2::new(k.fx * c.x / c.z + k.cx, k.fy * c.y / c.z + k.cy)
        };
        let plane: Vec<Pt3> = (0..20)
            .map(|i| Pt3::new(0.04 * f64::from(i % 5), 0.04 * f64::from(i / 5), 0.0))
            .collect();
        let image: Vec<Pt2> = plane.iter().map(project).collect();
        let est = epnp(&plane, &image, &k).unwrap();
        assert!((est.translation.vector - truth.translation.vector).norm() < 1e-8);
        assert!(est.rotation.angle_to(&truth.rotation) < 1e-8);

        let line: Vec<Pt3> = (0..6)
            .map(|i| Pt3::new(0.04 * f64::from(i), 0.0, 0.0))
            .collect();
        let image: Vec<Pt2> = line.iter().map(project).collect();
        assert!(matches!(
            epnp(&line, &image, &k),
            Err(GeometryError::Degenerate { .. })
        ));
    }

    #[test]
    fn epnp_recovers_pose_from_four_and_five_noncoplanar_points() {
        let k = FxFyCxCy {
            fx: 800.0,
            fy: 780.0,
            cx: 640.0,
            cy: 360.0,
        };
        let truth = Isometry3::from_parts(
            Translation3::new(0.12, -0.08, 1.35),
            Rotation3::from_euler_angles(0.22, -0.18, 0.31).into(),
        );
        let project = |p: &Pt3| {
            let c = truth * p;
            Pt2::new(k.fx * c.x / c.z + k.cx, k.fy * c.y / c.z + k.cy)
        };

        let four = vec![
            Pt3::new(-0.15, -0.12, 0.0),
            Pt3::new(0.20, -0.05, 0.08),
            Pt3::new(-0.04, 0.22, -0.06),
            Pt3::new(0.11, 0.09, 0.24),
        ];
        let img4: Vec<Pt2> = four.iter().map(project).collect();
        let est4 = epnp(&four, &img4, &k).unwrap();
        assert!(
            (est4.translation.vector - truth.translation.vector).norm() < 1e-6,
            "4-pt dt={}",
            (est4.translation.vector - truth.translation.vector).norm()
        );
        assert!(est4.rotation.angle_to(&truth.rotation) < 1e-6);

        let mut five = four;
        five.push(Pt3::new(0.25, 0.18, 0.14));
        let img5: Vec<Pt2> = five.iter().map(project).collect();
        let est5 = epnp(&five, &img5, &k).unwrap();
        assert!(
            (est5.translation.vector - truth.translation.vector).norm() < 1e-6,
            "5-pt dt={}",
            (est5.translation.vector - truth.translation.vector).norm()
        );
        assert!(est5.rotation.angle_to(&truth.rotation) < 1e-6);
    }
}
