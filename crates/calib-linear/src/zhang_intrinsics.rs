// crates/calib-linear/src/zhang_intrinsics.rs

use calib_core::{CameraIntrinsics, Mat3, Real};
use nalgebra::DMatrix;

/// Build the 6-vector v_ij(H) as in Zhang's method.
/// i, j are 0- or 1-based column indices (we'll use 0/1 for (1,2) and (1,1),(2,2)).
fn v_ij(hmtx: &Mat3, i: usize, j: usize) -> nalgebra::SVector<Real, 6> {
    let hi = hmtx.column(i);
    let hj = hmtx.column(j);

    nalgebra::SVector::<Real, 6>::from_row_slice(&[
        hi[0] * hj[0],
        hi[0] * hj[1] + hi[1] * hj[0],
        hi[1] * hj[1],
        hi[2] * hj[0] + hi[0] * hj[2],
        hi[2] * hj[1] + hi[1] * hj[2],
        hi[2] * hj[2],
    ])
}

/// Estimate camera intrinsics K from a set of plane homographies H_k using
/// Zhang's closed-form solution (no distortion).
///
/// Requires at least 3 homographies for a stable solution.
pub fn estimate_intrinsics_from_homographies(hmtxs: &[Mat3]) -> CameraIntrinsics {
    assert!(
        hmtxs.len() >= 3,
        "need at least 3 homographies for intrinsics estimation"
    );

    let m = hmtxs.len();
    let mut vmtx = DMatrix::<Real>::zeros(2 * m, 6);

    for (k, hmtx) in hmtxs.iter().enumerate() {
        let v11 = v_ij(hmtx, 0, 0);
        let v22 = v_ij(hmtx, 1, 1);
        let v12 = v_ij(hmtx, 0, 1);

        // Row 2k: v_12^T
        vmtx.row_mut(2 * k).copy_from(&v12.transpose());
        // Row 2k+1: (v_11 - v_22)^T
        vmtx.row_mut(2 * k + 1).copy_from(&(v11 - v22).transpose());
    }

    // Solve V b = 0 via SVD: take the singular vector corresponding to the
    // smallest singular value.
    let svd = vmtx.svd(true, true);
    let v_t = svd.v_t.expect("V^T from SVD");
    let b = v_t.row(v_t.nrows() - 1); // last row

    let b11 = b[0];
    let b12 = b[1];
    let b22 = b[2];
    let b13 = b[3];
    let b23 = b[4];
    let b33 = b[5];

    // From Zhang's paper:
    //
    // v0 = (B12 B13 - B11 B23) / (B11 B22 - B12^2)
    // λ = B33 - (B13^2 + v0 (B12 B13 - B11 B23)) / B11
    // α = sqrt(λ / B11)
    // β = sqrt(λ B11 / (B11 B22 - B12^2))
    // γ = -B12 α^2 β / λ
    // u0 = γ v0 / β - B13 α^2 / λ
    //
    // We name them fx, fy, skew, cx, cy accordingly.

    let denom = b11 * b22 - b12 * b12;
    let denom_norm = b11 * b11 + b22 * b22;
    let denom_rel = if denom_norm > 0.0 {
        denom.abs() / denom_norm
    } else {
        0.0
    };
    assert!(
        denom_rel > 1e-6,
        "degenerate configuration in intrinsics estimation"
    );

    let v0 = (b12 * b13 - b11 * b23) / denom;
    let lambda = b33 - (b13 * b13 + v0 * (b12 * b13 - b11 * b23)) / b11;

    assert!(
        lambda.signum() == b11.signum(),
        "invalid sign for λ; check homographies"
    );

    let alpha = (lambda / b11).sqrt();
    let beta = (lambda * b11 / denom).sqrt();
    let gamma = -b12 * alpha * alpha * beta / lambda;
    let u0 = gamma * v0 / beta - b13 * alpha * alpha / lambda;

    CameraIntrinsics {
        fx: alpha,
        fy: beta,
        cx: u0,
        cy: v0,
        skew: gamma,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Isometry3, Matrix3, Rotation3, Translation3, Vector3};

    fn make_kmtx() -> (CameraIntrinsics, Mat3) {
        let intr = CameraIntrinsics {
            fx: 900.0,
            fy: 880.0,
            cx: 640.0,
            cy: 360.0,
            skew: 0.0,
        };
        let kmtx = Matrix3::new(
            intr.fx, intr.skew, intr.cx, 0.0, intr.fy, intr.cy, 0.0, 0.0, 1.0,
        );
        (intr, kmtx)
    }

    fn synthetic_homography(kmtx: &Mat3, rot: Rotation3<Real>, t: Vector3<Real>) -> Mat3 {
        // Pose of board in camera frame
        let iso = Isometry3::from_parts(Translation3::from(t), rot.into());

        // For Z=0 plane, H = K [r1 r2 t]
        let binding = iso.rotation.to_rotation_matrix();
        let r_mat = binding.matrix();
        let r1 = r_mat.column(0);
        let r2 = r_mat.column(1);

        let mut hmtx = Mat3::zeros();
        hmtx.set_column(0, &(kmtx * r1));
        hmtx.set_column(1, &(kmtx * r2));
        hmtx.set_column(2, &(kmtx * t));
        hmtx
    }

    #[test]
    fn intrinsics_from_homographies_recovers_kmtx() {
        let (intr_gt, kmtx) = make_kmtx();

        // Three different board poses
        let hmts: Vec<Mat3> = vec![
            synthetic_homography(
                &kmtx,
                Rotation3::from_euler_angles(0.1, 0.0, 0.05),
                Vector3::new(0.1, -0.05, 1.0),
            ),
            synthetic_homography(
                &kmtx,
                Rotation3::from_euler_angles(-0.05, 0.15, -0.1),
                Vector3::new(-0.05, 0.1, 1.2),
            ),
            synthetic_homography(
                &kmtx,
                Rotation3::from_euler_angles(0.2, -0.1, 0.0),
                Vector3::new(0.0, 0.0, 0.9),
            ),
        ];

        let intr_est = estimate_intrinsics_from_homographies(&hmts);

        // Tolerances are somewhat arbitrary; adjust based on your geometry
        assert!((intr_est.fx - intr_gt.fx).abs() < 5.0, "fx mismatch");
        assert!((intr_est.fy - intr_gt.fy).abs() < 5.0, "fy mismatch");
        assert!((intr_est.cx - intr_gt.cx).abs() < 10.0, "cx mismatch");
        assert!((intr_est.cy - intr_gt.cy).abs() < 10.0, "cy mismatch");
        assert!(intr_est.skew.abs() < 1e-6, "skew not ~0: {}", intr_est.skew);
    }
}
