//! Camera matrix estimation and decomposition utilities.
//!
//! Provides a normalized DLT solver for the 3x4 projection matrix `P` and an
//! RQ decomposition to recover intrinsics and rotation.

use crate::math::{mat34_from_svd_row, normalize_points_2d, normalize_points_3d};
use anyhow::Result;
use calib_core::{Mat3, Pt2, Pt3, Real, Vec3};
use nalgebra::{DMatrix, Matrix3x4};

/// 3x4 camera projection matrix `P = K [R | t]`.
pub type Mat34 = Matrix3x4<Real>;

/// Camera matrix decomposition into `K`, `R`, `t` with `K` upper-triangular.
#[derive(Debug, Clone)]
pub struct CameraMatrixDecomposition {
    /// Intrinsics matrix (upper-triangular, positive diagonal).
    pub k: Mat3,
    /// Rotation matrix (orthonormal, det=+1).
    pub r: Mat3,
    /// Translation vector in camera coordinates.
    pub t: Vec3,
}

/// Estimate a camera projection matrix `P` using normalized DLT.
///
/// `world` are 3D points and `image` are their pixel projections. The output is
/// defined up to a global scale.
pub fn dlt_camera_matrix(world: &[Pt3], image: &[Pt2]) -> Result<Mat34> {
    let n = world.len();
    if n < 6 {
        anyhow::bail!("need at least 6 point correspondences, got {}", n);
    }
    if n != image.len() {
        anyhow::bail!(
            "mismatched number of world points ({}) and image points ({})",
            n,
            image.len()
        );
    }

    let (world_n, t_w) = normalize_points_3d(world).ok_or(anyhow::anyhow!(
        "degenerate point configuration for normalization"
    ))?;
    let (image_n, t_i) = normalize_points_2d(image).ok_or(anyhow::anyhow!(
        "degenerate point configuration for normalization"
    ))?;

    let mut a = DMatrix::<Real>::zeros(2 * n, 12);

    for (i, (pw, pi)) in world_n.iter().zip(image_n.iter()).enumerate() {
        let x = pw.x;
        let y = pw.y;
        let z = pw.z;
        let u = pi.x;
        let v = pi.y;

        let r0 = 2 * i;
        let r1 = 2 * i + 1;

        a[(r0, 0)] = x;
        a[(r0, 1)] = y;
        a[(r0, 2)] = z;
        a[(r0, 3)] = 1.0;
        a[(r0, 8)] = -u * x;
        a[(r0, 9)] = -u * y;
        a[(r0, 10)] = -u * z;
        a[(r0, 11)] = -u;

        a[(r1, 4)] = x;
        a[(r1, 5)] = y;
        a[(r1, 6)] = z;
        a[(r1, 7)] = 1.0;
        a[(r1, 8)] = -v * x;
        a[(r1, 9)] = -v * y;
        a[(r1, 10)] = -v * z;
        a[(r1, 11)] = -v;
    }

    let svd = a.svd(true, true);
    let v_t = svd.v_t.ok_or(anyhow::anyhow!("SVD failed"))?;
    let p_norm = mat34_from_svd_row(&v_t, v_t.nrows() - 1);

    let t_i_inv = t_i.try_inverse().ok_or(anyhow::anyhow!("SVD failed"))?;
    let p = t_i_inv * p_norm * t_w;

    Ok(p)
}

/// RQ decomposition of a 3x3 matrix.
///
/// Returns `(K, R)` with `K` upper-triangular and `R` orthonormal.
pub fn rq_decompose(m: &Mat3) -> (Mat3, Mat3) {
    let j = Mat3::new(0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0);

    let m1 = j * m.transpose() * j;
    let qr = m1.qr();

    let mut k = j * qr.r().transpose() * j;
    let mut r = j * qr.q().transpose() * j;

    // Enforce positive diagonal in K.
    let mut d = Mat3::identity();
    for i in 0..3 {
        if k[(i, i)] < 0.0 {
            d[(i, i)] = -1.0;
        }
    }
    k *= d;
    r = d * r;

    (k, r)
}

/// Decompose a camera projection matrix into intrinsics, rotation, and translation.
///
/// Returns `K`, `R`, and `t` such that `P ~ K [R | t]`. The diagonal of `K` is
/// forced positive and `R` is projected onto SO(3).
pub fn decompose_camera_matrix(p: &Mat34) -> Result<CameraMatrixDecomposition> {
    let m = p.fixed_view::<3, 3>(0, 0).into_owned();
    let (mut k, mut r) = rq_decompose(&m);

    // Normalize so that K[2,2] is positive and close to 1 in scale.
    if k[(2, 2)] < 0.0 {
        k = -k;
        r = -r;
    }

    let k_inv = k
        .try_inverse()
        .ok_or_else(|| anyhow::anyhow!("intrinsics matrix is not invertible"))?;
    let mut t = k_inv * p.column(3);

    if r.determinant() < 0.0 {
        r = -r;
        t = -t;
    }

    Ok(CameraMatrixDecomposition {
        k,
        r,
        t: t.into_owned(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Rotation3, Translation3, Vector4};

    fn project(p: &Mat34, point: &Pt3) -> Pt2 {
        let x = p * Vector4::new(point.x, point.y, point.z, 1.0);
        Pt2::new(x.x / x.z, x.y / x.z)
    }

    #[test]
    fn dlt_camera_matrix_recovers_projection() {
        let k = Mat3::new(900.0, 0.0, 640.0, 0.0, 880.0, 360.0, 0.0, 0.0, 1.0);
        let rot = Rotation3::from_euler_angles(0.15, -0.05, 0.1);
        let t = Translation3::new(0.1, -0.05, 1.2);
        let r = rot.matrix();

        let mut p_gt = Mat34::zeros();
        p_gt.fixed_view_mut::<3, 3>(0, 0).copy_from(&(k * r));
        p_gt.set_column(3, &(k * t.vector));

        let mut world = Vec::new();
        let mut image = Vec::new();
        for z in 0..2 {
            for y in 0..3 {
                for x in 0..4 {
                    let pw = Pt3::new(x as Real * 0.2, y as Real * 0.15, 2.0 + z as Real * 0.1);
                    let uv = project(&p_gt, &pw);
                    world.push(pw);
                    image.push(uv);
                }
            }
        }

        let p_est = dlt_camera_matrix(&world, &image).unwrap();
        let dot: Real = p_gt
            .as_slice()
            .iter()
            .zip(p_est.as_slice().iter())
            .map(|(a, b)| a * b)
            .sum();
        let denom: Real = p_est.as_slice().iter().map(|v| v * v).sum();
        let scale = if denom.abs() > Real::EPSILON {
            dot / denom
        } else {
            1.0
        };
        let p_scaled = p_est * scale;

        let diff = (p_scaled - p_gt).norm();
        assert!(diff < 1e-6, "camera matrix diff too large: {}", diff);
    }

    #[test]
    fn rq_decompose_recovers_k_r() {
        let k = Mat3::new(800.0, 1.5, 640.0, 0.0, 780.0, 360.0, 0.0, 0.0, 1.0);
        let rot = Rotation3::from_euler_angles(0.1, 0.2, -0.05);
        let r = rot.matrix();
        let m = k * r;

        let (k_est, r_est) = rq_decompose(&m);

        let scale = k[(2, 2)] / k_est[(2, 2)];
        let diff = (k_est * scale - k).norm();
        assert!(diff < 1e-6, "K mismatch: {}", diff);

        let r_diff = r_est.transpose() * r;
        let trace = r_diff.trace();
        let cos_theta = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0);
        let ang = cos_theta.acos();
        assert!(ang < 1e-6, "R mismatch: {}", ang);
    }

    #[test]
    fn camera_matrix_decomposition_roundtrip() {
        let k = Mat3::new(900.0, -2.0, 640.0, 0.0, 870.0, 360.0, 0.0, 0.0, 1.0);
        let rot = Rotation3::from_euler_angles(-0.1, 0.05, 0.2);
        let t = Translation3::new(-0.2, 0.1, 1.5);
        let mut p_gt = Mat34::zeros();
        p_gt.fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&(k * rot.matrix()));
        p_gt.set_column(3, &(k * t.vector));

        let decomp = decompose_camera_matrix(&p_gt).unwrap();

        let mut p_recon = Mat34::zeros();
        p_recon
            .fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&(decomp.k * decomp.r));
        p_recon.set_column(3, &(decomp.k * decomp.t));

        let dot: Real = p_gt
            .as_slice()
            .iter()
            .zip(p_recon.as_slice().iter())
            .map(|(a, b)| a * b)
            .sum();
        let denom: Real = p_recon.as_slice().iter().map(|v| v * v).sum();
        let scale = if denom.abs() > Real::EPSILON {
            dot / denom
        } else {
            1.0
        };
        let diff = (p_recon * scale - p_gt).norm();
        assert!(diff < 1e-6, "P reconstruction error too large: {}", diff);
    }
}
