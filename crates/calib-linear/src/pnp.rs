//! Perspective-n-Point (PnP) solvers for camera pose estimation.
//!
//! Includes:
//! - DLT (linear) pose estimation with normalization.
//! - P3P minimal solver (3 points, multiple solutions).
//! - EPnP (control-point formulation) for 4+ points.
//! - DLT wrapped in RANSAC for outlier rejection.
//!
//! All methods estimate a pose `T_C_W`: transform from world coordinates into
//! the camera frame.

use calib_core::{
    ransac_fit, Camera, Estimator, FxFyCxCySkew, IdentitySensor, Iso3, Mat3, Mat4, NoDistortion,
    Pinhole, Pt3, RansacOptions, Real, Vec2, Vec3,
};
use nalgebra::{
    linalg::{Schur, SymmetricEigen},
    DMatrix, Isometry3, Rotation3, Translation3, UnitQuaternion, Vector3,
};
use thiserror::Error;

/// Errors that can occur during PnP estimation.
#[derive(Debug, Error)]
pub enum PnpError {
    /// Not enough point correspondences were provided.
    #[error("need at least 6 point correspondences, got {0}")]
    NotEnoughPoints(usize),
    /// Incorrect number of correspondences for a minimal solver.
    #[error("invalid number of correspondences: expected {expected}, got {got}")]
    InvalidPointCount { expected: usize, got: usize },
    /// Intrinsics matrix is not invertible.
    #[error("intrinsics matrix is not invertible")]
    SingularIntrinsics,
    /// 3D points are degenerate for normalization.
    #[error("degenerate 3d point configuration for normalization")]
    DegeneratePoints,
    /// EPnP failed due to degenerate control points.
    #[error("degenerate control point configuration for EPnP")]
    DegenerateControlPoints,
    /// Linear solve (SVD) failed.
    #[error("svd failed in PnP DLT")]
    SvdFailed,
    /// Polynomial solve failed in P3P.
    #[error("failed to solve the P3P polynomial system")]
    PolynomialSolveFailed,
    /// RANSAC could not find a consensus pose.
    #[error("ransac failed to find a consensus PnP solution")]
    RansacFailed,
}

/// Linear PnP solver (DLT) for camera pose estimation.
///
/// This solves for a pose `T_C_W` (world to camera) from 3D points and their
/// 2D projections, using a Direct Linear Transform and optionally wrapping it
/// in a RANSAC loop.
#[derive(Debug, Clone, Copy)]
pub struct PnpSolver;

fn solve_quadratic_real(a: Real, b: Real, c: Real) -> Vec<Real> {
    let eps = 1e-12;
    if a.abs() < eps {
        if b.abs() < eps {
            return Vec::new();
        }
        return vec![-c / b];
    }
    let disc = b * b - 4.0 * a * c;
    if disc.abs() < eps {
        return vec![-b / (2.0 * a)];
    }
    if disc < 0.0 {
        return Vec::new();
    }
    let sqrt_disc = disc.sqrt();
    let r1 = (-b + sqrt_disc) / (2.0 * a);
    let r2 = (-b - sqrt_disc) / (2.0 * a);
    if (r1 - r2).abs() < 1e-8 {
        vec![r1]
    } else {
        vec![r1, r2]
    }
}

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

fn solve_cubic_real(a: Real, b: Real, c: Real, d: Real) -> Vec<Real> {
    let eps = 1e-12;
    if a.abs() < eps {
        return solve_quadratic_real(b, c, d);
    }

    let a_inv = 1.0 / a;
    let b = b * a_inv;
    let c = c * a_inv;
    let d = d * a_inv;

    let p = c - b * b / 3.0;
    let q = 2.0 * b * b * b / 27.0 - b * c / 3.0 + d;

    let disc = (q * 0.5) * (q * 0.5) + (p / 3.0) * (p / 3.0) * (p / 3.0);
    let shift = b / 3.0;

    let mut roots = Vec::new();
    if disc > eps {
        let sqrt_disc = disc.sqrt();
        let u = (-q * 0.5 + sqrt_disc).signum() * (-q * 0.5 + sqrt_disc).abs().powf(1.0 / 3.0);
        let v = (-q * 0.5 - sqrt_disc).signum() * (-q * 0.5 - sqrt_disc).abs().powf(1.0 / 3.0);
        roots.push(u + v - shift);
    } else if disc.abs() <= eps {
        let u = (-q * 0.5).signum() * (-q * 0.5).abs().powf(1.0 / 3.0);
        roots.push(2.0 * u - shift);
        roots.push(-u - shift);
    } else {
        let r = (-p / 3.0).sqrt();
        let phi = ((-q * 0.5) / (r * r * r)).clamp(-1.0, 1.0).acos();
        let two_r = 2.0 * r;
        roots.push(two_r * (phi / 3.0).cos() - shift);
        roots.push(two_r * ((phi + 2.0 * std::f64::consts::PI) / 3.0).cos() - shift);
        roots.push(two_r * ((phi + 4.0 * std::f64::consts::PI) / 3.0).cos() - shift);
    }

    roots.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    roots.dedup_by(|a, b| (*a - *b).abs() < 1e-8);
    roots
}

fn solve_quartic_real(a: Real, b: Real, c: Real, d: Real, e: Real) -> Vec<Real> {
    let eps = 1e-12;
    if a.abs() < eps {
        return solve_cubic_real(b, c, d, e);
    }

    let mut comp = DMatrix::<Real>::zeros(4, 4);
    comp[(0, 0)] = -b / a;
    comp[(0, 1)] = -c / a;
    comp[(0, 2)] = -d / a;
    comp[(0, 3)] = -e / a;
    comp[(1, 0)] = 1.0;
    comp[(2, 1)] = 1.0;
    comp[(3, 2)] = 1.0;

    let schur = Schur::new(comp);
    let eigvals = schur.complex_eigenvalues();

    let mut roots = Vec::new();
    for val in eigvals.iter() {
        if val.im.abs() < 1e-8 {
            roots.push(val.re);
        }
    }

    roots.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    roots.dedup_by(|a, b| (*a - *b).abs() < 1e-8);
    roots
}

fn pose_from_points(world: &[Pt3], camera: &[Vec3]) -> Result<Iso3, PnpError> {
    if world.len() != camera.len() || world.len() < 3 {
        return Err(PnpError::DegeneratePoints);
    }

    let n = world.len() as Real;
    let mut c_w = Vec3::zeros();
    let mut c_c = Vec3::zeros();
    for (pw, pc) in world.iter().zip(camera.iter()) {
        c_w += pw.coords;
        c_c += pc;
    }
    c_w /= n;
    c_c /= n;

    let mut h = Mat3::zeros();
    for (pw, pc) in world.iter().zip(camera.iter()) {
        let dw = pw.coords - c_w;
        let dc = pc - c_c;
        h += dc * dw.transpose();
    }

    let svd = h.svd(true, true);
    let u = svd.u.ok_or(PnpError::SvdFailed)?;
    let v_t = svd.v_t.ok_or(PnpError::SvdFailed)?;
    let mut r = u * v_t;
    if r.determinant() < 0.0 {
        let mut u_fix = u;
        u_fix.column_mut(2).neg_mut();
        r = u_fix * v_t;
    }

    let t = c_c - r * c_w;
    let rot = UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(r));
    let trans = Translation3::from(t);
    Ok(Isometry3::from_parts(trans, rot))
}

impl PnpSolver {
    /// Direct linear PnP on all inliers.
    ///
    /// `world` are 3D points in world coordinates, `image` are their pixel
    /// positions, and `k` are the camera intrinsics. Uses a normalized DLT
    /// solve and projects the rotation onto SO(3).
    pub fn dlt(world: &[Pt3], image: &[Vec2], k: &FxFyCxCySkew<Real>) -> Result<Iso3, PnpError> {
        let n = world.len();
        if n < 6 || image.len() != n {
            return Err(PnpError::NotEnoughPoints(n));
        }

        let kmtx: Mat3 = k.k_matrix();
        let k_inv = kmtx.try_inverse().ok_or(PnpError::SingularIntrinsics)?;

        let mut cx = 0.0;
        let mut cy = 0.0;
        let mut cz = 0.0;
        for p in world {
            cx += p.x;
            cy += p.y;
            cz += p.z;
        }
        let n_real = n as Real;
        cx /= n_real;
        cy /= n_real;
        cz /= n_real;

        let mut mean_dist = 0.0;
        for p in world {
            let dx = p.x - cx;
            let dy = p.y - cy;
            let dz = p.z - cz;
            mean_dist += (dx * dx + dy * dy + dz * dz).sqrt();
        }
        mean_dist /= n_real;
        if mean_dist <= Real::EPSILON {
            return Err(PnpError::DegeneratePoints);
        }

        let scale = (3.0_f64).sqrt() / mean_dist;
        let t_world = Mat4::new(
            scale,
            0.0,
            0.0,
            -scale * cx,
            0.0,
            scale,
            0.0,
            -scale * cy,
            0.0,
            0.0,
            scale,
            -scale * cz,
            0.0,
            0.0,
            0.0,
            1.0,
        );

        // Build 2n x 12 DLT matrix for camera matrix P = [R | t] in normalized coords.
        let mut a = DMatrix::<Real>::zeros(2 * n, 12);

        for (i, (pw, pi)) in world.iter().zip(image.iter()).enumerate() {
            let x = (pw.x - cx) * scale;
            let y = (pw.y - cy) * scale;
            let z = (pw.z - cz) * scale;

            // Normalized image point: x_n = K^{-1} [u,v,1]^T.
            let v_img = k_inv * nalgebra::Vector3::new(pi.x, pi.y, 1.0);
            let u = v_img.x / v_img.z;
            let v = v_img.y / v_img.z;

            let r0 = 2 * i;
            let r1 = 2 * i + 1;

            // Row for x
            a[(r0, 0)] = x;
            a[(r0, 1)] = y;
            a[(r0, 2)] = z;
            a[(r0, 3)] = 1.0;
            a[(r0, 8)] = -u * x;
            a[(r0, 9)] = -u * y;
            a[(r0, 10)] = -u * z;
            a[(r0, 11)] = -u;

            // Row for y
            a[(r1, 4)] = x;
            a[(r1, 5)] = y;
            a[(r1, 6)] = z;
            a[(r1, 7)] = 1.0;
            a[(r1, 8)] = -v * x;
            a[(r1, 9)] = -v * y;
            a[(r1, 10)] = -v * z;
            a[(r1, 11)] = -v;
        }

        // Solve A p = 0 via SVD: take the singular vector for the smallest singular value.
        let svd = a.svd(true, true);
        let v_t = svd.v_t.ok_or(PnpError::SvdFailed)?;
        let p_vec = v_t.row(v_t.nrows() - 1);

        // Reshape into 3x4 matrix P = [R|t] (up to scale).
        let mut p_mtx = nalgebra::Matrix3x4::<Real>::zeros();
        for r in 0..3 {
            for c in 0..4 {
                p_mtx[(r, c)] = p_vec[4 * r + c];
            }
        }

        // De-normalize 3D points: P = P_norm * T_world.
        let p_mtx = p_mtx * t_world;

        let m = p_mtx.fixed_view::<3, 3>(0, 0).into_owned();
        let mut r_approx = m;

        // Normalise scale using average row norm.
        let row0 = r_approx.row(0);
        let row1 = r_approx.row(1);
        let row2 = r_approx.row(2);
        let mut s = (row0.norm() + row1.norm() + row2.norm()) / 3.0;
        if r_approx.determinant() < 0.0 {
            s = -s;
        }
        if s.abs() > 0.0 {
            r_approx /= s;
        }

        // Project onto SO(3).
        let svd = r_approx.svd(true, true);
        let u = svd.u.ok_or(PnpError::SvdFailed)?;
        let v_t = svd.v_t.ok_or(PnpError::SvdFailed)?;
        let mut r_orth = u * v_t;
        if r_orth.determinant() < 0.0 {
            let mut u_flipped = u;
            u_flipped.column_mut(2).neg_mut();
            r_orth = u_flipped * v_t;
        }

        // Translation is the last column, scaled consistently with rotation.
        let mut t = p_mtx.column(3).into_owned();
        if s.abs() > 0.0 {
            t /= s;
        }

        let rot = UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(r_orth));
        let trans = Translation3::from(t);
        Ok(Isometry3::from_parts(trans, rot))
    }

    /// P3P minimal solver: returns up to four pose candidates.
    ///
    /// Requires exactly three non-collinear points and intrinsics `k` to
    /// convert pixels into rays. The resulting poses are in `T_C_W` form.
    pub fn p3p(
        world: &[Pt3],
        image: &[Vec2],
        k: &FxFyCxCySkew<Real>,
    ) -> Result<Vec<Iso3>, PnpError> {
        if world.len() != image.len() {
            return Err(PnpError::InvalidPointCount {
                expected: 3,
                got: world.len().max(image.len()),
            });
        }
        if world.len() != 3 {
            return Err(PnpError::InvalidPointCount {
                expected: 3,
                got: world.len(),
            });
        }

        let kmtx: Mat3 = k.k_matrix();
        let k_inv = kmtx.try_inverse().ok_or(PnpError::SingularIntrinsics)?;

        let mut bearings = Vec::with_capacity(3);
        for pi in image {
            let v = k_inv * Vector3::new(pi.x, pi.y, 1.0);
            bearings.push(v.normalize());
        }

        let a = (world[1] - world[2]).norm(); // BC
        let b = (world[0] - world[2]).norm(); // AC
        let c = (world[0] - world[1]).norm(); // AB

        if a <= Real::EPSILON || b <= Real::EPSILON || c <= Real::EPSILON {
            return Err(PnpError::DegeneratePoints);
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
            return Err(PnpError::PolynomialSolveFailed);
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
            return Err(PnpError::PolynomialSolveFailed);
        }

        solutions.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        Ok(solutions.into_iter().map(|(_, pose)| pose).collect())
    }

    /// EPnP pose estimation for 4+ points.
    ///
    /// Uses a control-point formulation derived from the covariance of the
    /// 3D points. Returns a single pose estimate in `T_C_W` form.
    pub fn epnp(world: &[Pt3], image: &[Vec2], k: &FxFyCxCySkew<Real>) -> Result<Iso3, PnpError> {
        let n = world.len();
        if n < 4 || image.len() != n {
            return Err(PnpError::NotEnoughPoints(n));
        }

        let kmtx: Mat3 = k.k_matrix();
        let k_inv = kmtx.try_inverse().ok_or(PnpError::SingularIntrinsics)?;

        let mut img_norm = Vec::with_capacity(n);
        for pi in image {
            let v = k_inv * Vector3::new(pi.x, pi.y, 1.0);
            img_norm.push(Vec2::new(v.x / v.z, v.y / v.z));
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
        let basis_inv = basis
            .try_inverse()
            .ok_or(PnpError::DegenerateControlPoints)?;

        let mut alphas = Vec::with_capacity(n);
        for p in world {
            let coeff = basis_inv * (p.coords - control_w[0]);
            let a0 = 1.0 - coeff.x - coeff.y - coeff.z;
            alphas.push([a0, coeff.x, coeff.y, coeff.z]);
        }

        let mut m = DMatrix::<Real>::zeros(2 * n, 12);
        for (i, (a, uv)) in alphas.iter().zip(img_norm.iter()).enumerate() {
            let r0 = 2 * i;
            let r1 = 2 * i + 1;
            let u = uv.x;
            let v = uv.y;
            for (j, &alpha) in a.iter().enumerate().take(4) {
                let c = 3 * j;
                m[(r0, c)] = alpha;
                m[(r0, c + 2)] = -u * alpha;
                m[(r1, c + 1)] = alpha;
                m[(r1, c + 2)] = -v * alpha;
            }
        }

        let svd = m.svd(true, true);
        let v_t = svd.v_t.ok_or(PnpError::SvdFailed)?;
        let sol = v_t.row(v_t.nrows() - 1);

        let mut control_c = [Vec3::zeros(); 4];
        for (j, cc) in control_c.iter_mut().enumerate() {
            *cc = Vec3::new(sol[3 * j], sol[3 * j + 1], sol[3 * j + 2]);
        }

        let mut sum_w = 0.0;
        let mut sum_c = 0.0;
        for i in 0..4 {
            for j in (i + 1)..4 {
                let dw = (control_w[i] - control_w[j]).norm();
                let dc = (control_c[i] - control_c[j]).norm();
                sum_w += dw * dw;
                sum_c += dc * dc;
            }
        }

        if sum_c <= Real::EPSILON {
            return Err(PnpError::DegenerateControlPoints);
        }

        let scale = (sum_w / sum_c).sqrt();
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

        pose_from_points(world, &camera_pts)
    }

    /// Robust PnP using DLT inside a RANSAC loop.
    ///
    /// Returns the best pose and inlier indices. The residual is pixel
    /// reprojection error using the provided intrinsics.
    pub fn dlt_ransac(
        world: &[Pt3],
        image: &[Vec2],
        k: &FxFyCxCySkew<Real>,
        opts: &RansacOptions,
    ) -> Result<(Iso3, Vec<usize>), PnpError> {
        let n = world.len();
        if n < 6 || image.len() != n {
            return Err(PnpError::NotEnoughPoints(n));
        }

        #[derive(Clone)]
        struct PnpDatum {
            pw: Pt3,
            pi: Vec2,
            k: FxFyCxCySkew<Real>,
        }

        struct PnpEst;

        impl Estimator for PnpEst {
            type Datum = PnpDatum;
            type Model = Iso3;

            const MIN_SAMPLES: usize = 6;

            fn fit(data: &[Self::Datum], sample_indices: &[usize]) -> Option<Self::Model> {
                let mut world = Vec::with_capacity(sample_indices.len());
                let mut image = Vec::with_capacity(sample_indices.len());
                for &idx in sample_indices {
                    world.push(data[idx].pw);
                    image.push(data[idx].pi);
                }
                let k = data[0].k;
                PnpSolver::dlt(&world, &image, &k).ok()
            }

            fn residual(model: &Self::Model, datum: &Self::Datum) -> f64 {
                let cam = Camera::new(Pinhole, NoDistortion, IdentitySensor, datum.k);
                let pw = datum.pw;
                let pc = model.transform_point(&pw);
                let Some(proj) = cam.project_point(&pc) else {
                    return f64::INFINITY;
                };
                let du = proj.x - datum.pi.x;
                let dv = proj.y - datum.pi.y;
                (du * du + dv * dv).sqrt()
            }

            fn is_degenerate(_data: &[Self::Datum], sample_indices: &[usize]) -> bool {
                sample_indices.len() < Self::MIN_SAMPLES
            }
        }

        let data: Vec<PnpDatum> = world
            .iter()
            .cloned()
            .zip(image.iter().cloned())
            .map(|(pw, pi)| PnpDatum { pw, pi, k: *k })
            .collect();

        let res = ransac_fit::<PnpEst>(&data, opts);
        if !res.success {
            return Err(PnpError::RansacFailed);
        }
        let pose = res.model.expect("success guarantees a model");
        Ok((pose, res.inliers))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pnp_dlt_recovers_pose_synthetic() {
        let k = FxFyCxCySkew {
            fx: 800.0,
            fy: 780.0,
            cx: 640.0,
            cy: 360.0,
            skew: 0.0,
        };
        let cam = Camera::new(Pinhole, NoDistortion, IdentitySensor, k);

        // Ground-truth pose: world -> camera.
        let rot = Rotation3::from_euler_angles(0.1, -0.05, 0.2);
        let t = Translation3::new(0.1, -0.05, 1.0);
        let iso_gt = Isometry3::from_parts(t, rot.into());

        // Generate synthetic 3D points and project.
        let mut world = Vec::new();
        let mut image = Vec::new();
        for z in 0..2 {
            for y in 0..3 {
                for x in 0..4 {
                    let pw = Pt3::new(x as Real * 0.1, y as Real * 0.1, 0.5 + z as Real * 0.1);
                    let pc = iso_gt.transform_point(&pw);
                    let uv = cam.project_point(&pc).unwrap();
                    world.push(pw);
                    image.push(uv);
                }
            }
        }

        let est = PnpSolver::dlt(&world, &image, &k).unwrap();

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

    #[test]
    fn pnp_ransac_handles_outliers() {
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

        let mut world = Vec::new();
        let mut image = Vec::new();
        for z in 0..2 {
            for y in 0..3 {
                for x in 0..4 {
                    let pw = Pt3::new(x as Real * 0.1, y as Real * 0.1, 0.5 + z as Real * 0.1);
                    let pc = iso_gt.transform_point(&pw);
                    let uv = cam.project_point(&pc).unwrap();
                    world.push(pw);
                    image.push(uv);
                }
            }
        }

        let inlier_count = world.len();

        // Add a few mismatched correspondences as outliers.
        for i in 0..4 {
            world.push(Pt3::new(0.5 + i as Real * 0.2, -0.3, 1.2));
            image.push(Vec2::new(
                1200.0 + i as Real * 50.0,
                -100.0 - i as Real * 25.0,
            ));
        }

        let opts = RansacOptions {
            max_iters: 500,
            thresh: 1.0,
            min_inliers: inlier_count.saturating_sub(2),
            confidence: 0.99,
            seed: 77,
            refit_on_inliers: true,
        };

        let (est, inliers) = PnpSolver::dlt_ransac(&world, &image, &k, &opts).unwrap();

        assert!(inliers.len() >= inlier_count.saturating_sub(2));
        assert!(inliers.len() < world.len());

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

        let sols = PnpSolver::p3p(&world, &image, &k).unwrap();
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
    fn epnp_recovers_pose_synthetic() {
        let k = FxFyCxCySkew {
            fx: 800.0,
            fy: 780.0,
            cx: 640.0,
            cy: 360.0,
            skew: 0.0,
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

        let est = PnpSolver::epnp(&world, &image, &k).unwrap();

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
}
