use calib_core::{ransac, Estimator, Mat3, Pt2, RansacOptions, Real};
use nalgebra::{DMatrix, SMatrix};
use thiserror::Error;

/// Errors that can occur during fundamental / essential matrix estimation.
#[derive(Debug, Error)]
pub enum EpipolarError {
    /// Not enough point correspondences were provided.
    #[error("need at least 8 point correspondences, got {0}")]
    NotEnoughPoints(usize),
    /// Linear solve (SVD) failed.
    #[error("svd failed in epipolar estimation")]
    SvdFailed,
    /// RANSAC failed to find a consensus model.
    #[error("ransac failed to find a consensus epipolar model")]
    RansacFailed,
}

/// Linear epipolar geometry solvers (fundamental / essential matrices).
#[derive(Debug, Clone, Copy)]
pub struct EpipolarSolver;

impl EpipolarSolver {
    /// Normalised 8-point algorithm for the fundamental matrix.
    ///
    /// `pts1` and `pts2` are corresponding points in two images.
    pub fn fundamental_8point(
        pts1: &[Pt2],
        pts2: &[Pt2],
    ) -> Result<Mat3, EpipolarError> {
        let n = pts1.len();
        if n < 8 || pts2.len() != n {
            return Err(EpipolarError::NotEnoughPoints(n));
        }

        // Build design matrix A (n x 9) for x'^T F x = 0.
        let mut a = DMatrix::<Real>::zeros(n, 9);

        for (i, (p1, p2)) in pts1.iter().zip(pts2.iter()).enumerate() {
            let x = p1.x;
            let y = p1.y;
            let xp = p2.x;
            let yp = p2.y;

            a[(i, 0)] = xp * x;
            a[(i, 1)] = xp * y;
            a[(i, 2)] = xp;
            a[(i, 3)] = yp * x;
            a[(i, 4)] = yp * y;
            a[(i, 5)] = yp;
            a[(i, 6)] = x;
            a[(i, 7)] = y;
            a[(i, 8)] = 1.0;
        }

        // Solve A f = 0 using SVD on A^T A.
        let ata = a.transpose() * &a;
        let eig = ata.symmetric_eigen();
        let min_idx = eig
            .eigenvalues
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .ok_or(EpipolarError::SvdFailed)?;
        let f_vec = eig.eigenvectors.column(min_idx);

        let mut f = Mat3::zeros();
        for r in 0..3 {
            for c in 0..3 {
                f[(r, c)] = f_vec[3 * r + c];
            }
        }

        // Enforce rank-2 constraint on F.
        let svd_f = f.svd(true, true);
        let mut u = svd_f.u.ok_or(EpipolarError::SvdFailed)?;
        let mut s = svd_f.singular_values;
        let v_t = svd_f.v_t.ok_or(EpipolarError::SvdFailed)?;
        s[2] = 0.0;
        let s_mat = SMatrix::<Real, 3, 3>::from_diagonal(&s);
        f = u * s_mat * v_t;

        Ok(f)
    }

    /// Robust fundamental matrix estimation using the 8-point algorithm inside RANSAC.
    pub fn fundamental_8point_ransac(
        pts1: &[Pt2],
        pts2: &[Pt2],
        opts: &RansacOptions,
    ) -> Result<(Mat3, Vec<usize>), EpipolarError> {
        let n = pts1.len();
        if n < 8 || pts2.len() != n {
            return Err(EpipolarError::NotEnoughPoints(n));
        }

        #[derive(Clone)]
        struct FDatum {
            x1: Pt2,
            x2: Pt2,
        }

        struct FundamentalEst;

        impl Estimator for FundamentalEst {
            type Datum = FDatum;
            type Model = Mat3;

            const MIN_SAMPLES: usize = 8;

            fn fit(data: &[Self::Datum], sample_indices: &[usize]) -> Option<Self::Model> {
                let mut p1 = Vec::with_capacity(sample_indices.len());
                let mut p2 = Vec::with_capacity(sample_indices.len());
                for &idx in sample_indices {
                    p1.push(data[idx].x1);
                    p2.push(data[idx].x2);
                }
                EpipolarSolver::fundamental_8point(&p1, &p2).ok()
            }

            fn residual(model: &Self::Model, datum: &Self::Datum) -> f64 {
                // Symmetric epipolar distance (approximate).
                let x = nalgebra::Vector3::new(datum.x1.x, datum.x1.y, 1.0);
                let xp = nalgebra::Vector3::new(datum.x2.x, datum.x2.y, 1.0);

                let fx = model * x;
                let ftxp = model.transpose() * xp;
                let denom = fx.x * fx.x + fx.y * fx.y + ftxp.x * ftxp.x + ftxp.y * ftxp.y;
                let denom = denom.max(1e-12);
                let val = xp.transpose() * model * x;
                let d2 = (val[0] * val[0]) / denom;
                d2.sqrt()
            }

            fn is_degenerate(_data: &[Self::Datum], sample_indices: &[usize]) -> bool {
                sample_indices.len() < Self::MIN_SAMPLES
            }
        }

        let data: Vec<FDatum> = pts1
            .iter()
            .cloned()
            .zip(pts2.iter().cloned())
            .map(|(x1, x2)| FDatum { x1, x2 })
            .collect();

        let res = ransac::<FundamentalEst>(&data, opts);
        if !res.success {
            return Err(EpipolarError::RansacFailed);
        }
        let f = res.model.expect("success guarantees a model");
        Ok((f, res.inliers))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use calib_core::{CameraIntrinsics, Mat3};
    use nalgebra::{Rotation3, Translation3};

    fn make_k() -> (CameraIntrinsics, Mat3) {
        let k = CameraIntrinsics {
            fx: 800.0,
            fy: 780.0,
            cx: 640.0,
            cy: 360.0,
            skew: 0.0,
        };
        (k, k.k_matrix())
    }

    #[test]
    fn fundamental_8point_succeeds_on_simple_data() {
        // Very simple synthetic stereo with small baseline along X.
        let (_k, kmtx) = make_k();

        let rot_l = Rotation3::identity();
        let t_l = Translation3::new(0.0, 0.0, 0.0);
        let rot_r = Rotation3::identity();
        let t_r = Translation3::new(0.1, 0.0, 0.0);

        let p_l = kmtx * Mat3::identity();
        let p_r = kmtx * Mat3::identity();

        let mut pts1 = Vec::new();
        let mut pts2 = Vec::new();

        for z in 1..3 {
            for y in 0..3 {
                for x in 0..4 {
                    let pw =
                        nalgebra::Point3::new(x as Real * 0.1, y as Real * 0.1, z as Real * 0.5);
                    let pc_l = rot_l * pw + t_l.vector;
                    let pc_r = rot_r * pw + t_r.vector;

                    let xl = p_l * pc_l.coords;
                    let xr = p_r * pc_r.coords;

                    let u_l = xl.x / xl.z;
                    let v_l = xl.y / xl.z;
                    let u_r = xr.x / xr.z;
                    let v_r = xr.y / xr.z;

                    pts1.push(Pt2::new(u_l, v_l));
                    pts2.push(Pt2::new(u_r, v_r));
                }
            }
        }

        let f = EpipolarSolver::fundamental_8point(&pts1, &pts2).unwrap();
        // Basic sanity check: F should not be the zero matrix.
        let norm_f = f.norm();
        assert!(norm_f > 0.0);
    }
}

