use calib_core::{
    ransac_fit, Camera, Estimator, FxFyCxCySkew, IdentitySensor, Iso3, Mat3, NoDistortion, Pinhole,
    Pt3, RansacOptions, Real, Vec2,
};
use nalgebra::{DMatrix, DVector, Isometry3, Rotation3, Translation3, UnitQuaternion};
use thiserror::Error;

/// Errors that can occur during PnP estimation.
#[derive(Debug, Error)]
pub enum PnpError {
    /// Not enough point correspondences were provided.
    #[error("need at least 6 point correspondences, got {0}")]
    NotEnoughPoints(usize),
    /// Intrinsics matrix is not invertible.
    #[error("intrinsics matrix is not invertible")]
    SingularIntrinsics,
    /// Linear solve (SVD) failed.
    #[error("svd failed in PnP DLT")]
    SvdFailed,
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

impl PnpSolver {
    /// Direct linear PnP on all inliers.
    ///
    /// `world` are 3D points in world coordinates, `image` are their
    /// corresponding pixel positions, and `k` are the camera intrinsics.
    pub fn dlt(world: &[Pt3], image: &[Vec2], k: &FxFyCxCySkew<Real>) -> Result<Iso3, PnpError> {
        let n = world.len();
        if n < 6 || image.len() != n {
            return Err(PnpError::NotEnoughPoints(n));
        }

        let kmtx: Mat3 = k.k_matrix();
        let k_inv = kmtx.try_inverse().ok_or(PnpError::SingularIntrinsics)?;

        // Build 2n x 12 DLT matrix for camera matrix P = [R | t] in normalized coords.
        let mut a = DMatrix::<Real>::zeros(2 * n, 12);

        for (i, (pw, pi)) in world.iter().zip(image.iter()).enumerate() {
            let x = pw.x;
            let y = pw.y;
            let z = pw.z;

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

        // Solve A p = 0 by SVD on A^T A.
        let ata = a.transpose() * &a;
        let eig = ata.symmetric_eigen();
        let min_idx = eig
            .eigenvalues
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .ok_or(PnpError::SvdFailed)?;

        let p_vec: DVector<Real> = eig.eigenvectors.column(min_idx).into();

        // Reshape into 3x4 matrix P = [R|t] (up to scale).
        let mut p_mtx = nalgebra::Matrix3x4::<Real>::zeros();
        for r in 0..3 {
            for c in 0..4 {
                p_mtx[(r, c)] = p_vec[4 * r + c];
            }
        }

        let m = p_mtx.fixed_view::<3, 3>(0, 0).into_owned();
        let mut r_approx = m;

        // Normalise scale using average row norm.
        let row0 = r_approx.row(0);
        let row1 = r_approx.row(1);
        let row2 = r_approx.row(2);
        let s = (row0.norm() + row1.norm() + row2.norm()) / 3.0;
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

    /// Robust PnP using DLT inside a RANSAC loop.
    ///
    /// Returns the best pose and inlier indices.
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
}
