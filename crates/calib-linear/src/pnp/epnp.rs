//! EPnP (Efficient Perspective-n-Point) solver for camera pose estimation.
//!
//! Implements Lepetit's control-point formulation for 4+ points. Uses
//! eigen decomposition of the world point covariance to define a minimal
//! basis, then solves for control point positions in the camera frame.

use super::pose_utils::pose_from_points;
use anyhow::Result;
use calib_core::{FxFyCxCySkew, Iso3, Mat3, Pt2, Pt3, Real, Vec3};
use nalgebra::{linalg::SymmetricEigen, DMatrix, Vector3};

/// EPnP pose estimation for 4+ points.
///
/// Uses a control-point formulation derived from the covariance of the
/// 3D points. Returns a single pose estimate in `T_C_W` form.
pub fn epnp(world: &[Pt3], image: &[Pt2], k: &FxFyCxCySkew<Real>) -> Result<Iso3> {
    let n = world.len();
    if n < 4 || image.len() != n {
        anyhow::bail!("need at least 4 point correspondences, got {}", n);
    }

    let kmtx: Mat3 = k.k_matrix();
    let k_inv = kmtx
        .try_inverse()
        .ok_or_else(|| anyhow::anyhow!("intrinsics matrix is not invertible"))?;

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
        .ok_or_else(|| anyhow::anyhow!("degenerate control point configuration for EPnP"))?;

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
    let v_t = svd
        .v_t
        .ok_or_else(|| anyhow::anyhow!("svd failed in PnP DLT"))?;
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
        anyhow::bail!("degenerate control point configuration for EPnP");
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

#[cfg(test)]
mod tests {
    use super::*;
    use calib_core::{Camera, IdentitySensor, NoDistortion, Pinhole};
    use nalgebra::{Isometry3, Rotation3, Translation3};

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
}
