//! Direct Linear Transform (DLT) solver for camera pose estimation.
//!
//! Provides a linear least-squares solution to the PnP problem using
//! homogeneous equations. The rotation matrix is projected onto SO(3)
//! via SVD decomposition.

use crate::math::mat34_from_svd_row;
use anyhow::Result;
use calib_core::{FxFyCxCySkew, Iso3, Mat3, Mat4, Pt2, Pt3, Real};
use nalgebra::{DMatrix, Isometry3, Rotation3, Translation3, UnitQuaternion};

/// Direct linear PnP on all input points.
///
/// `world` are 3D points in world coordinates, `image` are their pixel
/// positions, and `k` are the camera intrinsics. Uses a normalized DLT
/// solve and projects the rotation onto SO(3).
///
/// Returns `T_C_W`: the transform from world to camera coordinates.
pub fn dlt(world: &[Pt3], image: &[Pt2], k: &FxFyCxCySkew<Real>) -> Result<Iso3> {
    let n = world.len();
    if n < 6 || image.len() != n {
        anyhow::bail!("need at least 6 point correspondences, got {}", n);
    }

    let kmtx: Mat3 = k.k_matrix();
    let k_inv = kmtx
        .try_inverse()
        .ok_or_else(|| anyhow::anyhow!("intrinsics matrix is not invertible"))?;

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
        anyhow::bail!("degenerate 3d point configuration for normalization");
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
    let v_t = svd
        .v_t
        .ok_or_else(|| anyhow::anyhow!("svd failed in PnP DLT"))?;
    // Reshape into 3x4 matrix P = [R|t] (up to scale).
    let p_mtx = mat34_from_svd_row(&v_t, v_t.nrows() - 1);

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
    let u = svd
        .u
        .ok_or_else(|| anyhow::anyhow!("svd failed in PnP DLT"))?;
    let v_t = svd
        .v_t
        .ok_or_else(|| anyhow::anyhow!("svd failed in PnP DLT"))?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use calib_core::{Camera, IdentitySensor, NoDistortion, Pinhole};
    use nalgebra::{Rotation3, Translation3};

    #[test]
    fn dlt_recovers_pose_synthetic() {
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

        let est = dlt(&world, &image, &k).unwrap();

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
