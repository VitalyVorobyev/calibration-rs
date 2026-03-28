//! Homography geometry: transfer, decomposition, and construction.
//!
//! A homography `H` is a 3×3 projective transform mapping points between
//! two views of a planar scene (or between any two views under pure rotation).
//!
//! This module provides:
//! - **Transfer**: project points through a homography
//! - **Decomposition**: recover candidate (R, t, n) from H
//! - **Construction**: build H from known pose and plane normal

use anyhow::Result;
use vision_calibration_core::{Mat3, Pt2, Real, Vec3};

use crate::types::HomographyMatrix;

/// Result of decomposing a homography into motion and plane normal.
#[derive(Debug, Clone)]
pub struct HomographyDecomposition {
    /// Rotation from view 1 to view 2.
    pub r: Mat3,
    /// Translation from view 1 to view 2 (up to scale).
    pub t: Vec3,
    /// Plane normal in camera 1 coordinates.
    pub normal: Vec3,
}

/// Transfer a point through a homography: `x' = H * x`.
///
/// Projects the result back to inhomogeneous coordinates.
pub fn homography_transfer(h: &HomographyMatrix, pt: &Pt2) -> Pt2 {
    let p = nalgebra::Vector3::new(pt.x, pt.y, 1.0);
    let hp = h * p;
    Pt2::new(hp.x / hp.z, hp.y / hp.z)
}

/// Transfer a point through the inverse homography: `x = H⁻¹ * x'`.
///
/// Returns an error if `H` is singular.
pub fn homography_transfer_inverse(h: &HomographyMatrix, pt: &Pt2) -> Result<Pt2> {
    let h_inv = h
        .try_inverse()
        .ok_or_else(|| anyhow::anyhow!("singular homography"))?;
    Ok(homography_transfer(&h_inv, pt))
}

/// Decompose a homography into candidate (R, t, n) triples.
///
/// Given a homography `H` between two calibrated views of a planar scene,
/// recovers up to 4 candidate decompositions. The homography relates to
/// pose and plane as: `H ~ R + t * n^T / d`, where `d` is the distance
/// from camera 1 to the plane.
///
/// Returns 2 or 4 candidates; the correct one must be selected using
/// additional constraints (e.g., visible points must have positive depth,
/// plane normal must face the camera).
pub fn decompose_homography(h: &HomographyMatrix) -> Result<Vec<HomographyDecomposition>> {
    // Normalize H by middle singular value so that σ2 = 1.
    let svd_h = h.svd(false, false);
    let s2 = svd_h.singular_values[1];
    let s3 = svd_h.singular_values[2];

    if s3 < 1e-12 {
        anyhow::bail!("degenerate homography (rank < 3)");
    }

    let h_n = h / s2;

    // Eigendecompose H^T H: eigenvalues γ1 ≥ γ2 ≈ 1 ≥ γ3.
    let hth = h_n.transpose() * h_n;
    let eig = hth.symmetric_eigen();

    let mut indices = [0usize, 1, 2];
    indices.sort_by(|&a, &b| {
        eig.eigenvalues[b]
            .partial_cmp(&eig.eigenvalues[a])
            .unwrap()
    });
    let l1 = eig.eigenvalues[indices[0]];
    let l3 = eig.eigenvalues[indices[2]];
    let v1 = eig.eigenvectors.column(indices[0]).into_owned();
    let v3 = eig.eigenvectors.column(indices[2]).into_owned();

    // Pure rotation: all eigenvalues ≈ 1.
    if (l1 - l3).abs() < 1e-8 {
        let mut r = h_n;
        if r.determinant() < 0.0 {
            r = -r;
        }
        return Ok(vec![HomographyDecomposition {
            r,
            t: Vec3::zeros(),
            normal: Vec3::new(0.0, 0.0, 1.0),
        }]);
    }

    // Candidate plane normals from eigendecomposition.
    let c1 = ((1.0 - l3) / (l1 - l3)).sqrt();
    let c3 = ((l1 - 1.0) / (l1 - l3)).sqrt();
    let normals: [Vec3; 2] = [c1 * v1 + c3 * v3, c1 * v1 - c3 * v3];

    let mut results = Vec::with_capacity(4);

    for n_raw in &normals {
        let n = n_raw.normalize();

        // For x ⊥ n: H*x = R*x exactly (the translation term vanishes).
        // Build an orthonormal basis {u1, u2, n}.
        let u1 = find_perpendicular(&n);
        let u2 = n.cross(&u1);

        let hu1: Vec3 = h_n * u1;
        let hu2: Vec3 = h_n * u2;

        // In the noiseless case, hu1 and hu2 are R*u1 and R*u2 (orthonormal).
        // Complete to a 3×3 matrix and project to SO(3).
        let r_n: Vec3 = hu1.cross(&hu2);
        let m = Mat3::from_columns(&[hu1, hu2, r_n]);
        let basis = Mat3::from_columns(&[u1, u2, n]);
        let r_approx = m * basis.transpose();

        let svd_r = r_approx.svd(true, true);
        let u = svd_r.u.ok_or(anyhow::anyhow!("SVD failed"))?;
        let vt = svd_r.v_t.ok_or(anyhow::anyhow!("SVD failed"))?;
        let mut r = u * vt;
        if r.determinant() < 0.0 {
            r = -r;
        }

        // t/d = (H_n - R) * n.
        let t: Vec3 = (h_n - r) * n;

        // Sign ambiguity: (R, t, n) and (R, -t, -n).
        results.push(HomographyDecomposition {
            r,
            t,
            normal: n,
        });
        results.push(HomographyDecomposition {
            r,
            t: -t,
            normal: -n,
        });
    }

    Ok(results)
}

/// Find a unit vector perpendicular to `v`.
fn find_perpendicular(v: &Vec3) -> Vec3 {
    let candidate = if v.x.abs() < 0.9 {
        Vec3::new(1.0, 0.0, 0.0)
    } else {
        Vec3::new(0.0, 1.0, 0.0)
    };
    let perp = v.cross(&candidate);
    perp.normalize()
}

/// Construct a homography from relative pose and plane equation.
///
/// Given rotation `R`, translation `t`, plane normal `n` (in camera 1
/// coordinates), and distance `d` from camera 1 origin to the plane,
/// returns `H = R + (t * n^T) / d`.
///
/// The plane equation is `n^T X = d` for 3D points `X` on the plane.
pub fn homography_from_pose_and_plane(r: &Mat3, t: &Vec3, n: &Vec3, d: Real) -> HomographyMatrix {
    *r + (t * n.transpose()) / d
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Rotation3;
    use vision_calibration_core::Pt3;

    fn make_planar_scene() -> (Mat3, Vec3, Vec3, Real, Vec<Pt3>) {
        let rot = Rotation3::from_euler_angles(0.05, -0.03, 0.02);
        let r = *rot.matrix();
        let t = Vec3::new(0.3, 0.01, 0.005);
        let n = Vec3::new(0.0, 0.0, 1.0); // plane at z = d
        let d = 3.0;

        let points = vec![
            Pt3::new(0.5, 0.3, d),
            Pt3::new(-0.4, 0.2, d),
            Pt3::new(0.6, -0.3, d),
            Pt3::new(-0.3, -0.4, d),
            Pt3::new(0.1, 0.6, d),
            Pt3::new(0.4, -0.5, d),
        ];

        (r, t, n, d, points)
    }

    #[test]
    fn transfer_roundtrip() {
        let (r, t, n, d, points) = make_planar_scene();
        let h = homography_from_pose_and_plane(&r, &t, &n, d);

        for pw in &points {
            let pt1 = Pt2::new(pw.x / pw.z, pw.y / pw.z);
            let pt2_expected = {
                let pc2 = r * pw.coords + t;
                Pt2::new(pc2.x / pc2.z, pc2.y / pc2.z)
            };

            let pt2 = homography_transfer(&h, &pt1);
            let err = ((pt2.x - pt2_expected.x).powi(2) + (pt2.y - pt2_expected.y).powi(2)).sqrt();
            assert!(err < 1e-10, "transfer error: {}", err);

            let pt1_back = homography_transfer_inverse(&h, &pt2).unwrap();
            let err_back =
                ((pt1_back.x - pt1.x).powi(2) + (pt1_back.y - pt1.y).powi(2)).sqrt();
            assert!(err_back < 1e-10, "inverse transfer error: {}", err_back);
        }
    }

    #[test]
    fn homography_from_pose_and_plane_consistent() {
        let (r, t, n, d, points) = make_planar_scene();
        let h = homography_from_pose_and_plane(&r, &t, &n, d);

        for pw in &points {
            let pt1 = Pt2::new(pw.x / pw.z, pw.y / pw.z);
            let pt2_gt = {
                let pc2 = r * pw.coords + t;
                Pt2::new(pc2.x / pc2.z, pc2.y / pc2.z)
            };

            let pt2 = homography_transfer(&h, &pt1);
            let err = ((pt2.x - pt2_gt.x).powi(2) + (pt2.y - pt2_gt.y).powi(2)).sqrt();
            assert!(err < 1e-10, "pose→H→transfer error: {}", err);
        }
    }

    #[test]
    fn decompose_homography_roundtrip() {
        let (r_gt, t_gt, n_gt, d, _) = make_planar_scene();
        let h = homography_from_pose_and_plane(&r_gt, &t_gt, &n_gt, d);

        let decomps = decompose_homography(&h).unwrap();
        assert!(!decomps.is_empty());

        let mut found = false;
        for decomp in &decomps {
            // Check rotation.
            let r_diff = decomp.r.transpose() * r_gt;
            let cos_theta = ((r_diff.trace() - 1.0) * 0.5).clamp(-1.0, 1.0);
            let ang_deg = cos_theta.acos().to_degrees();

            // Check translation direction (scale is unknown).
            let t_cos = if t_gt.norm() > 1e-12 && decomp.t.norm() > 1e-12 {
                decomp.t.normalize().dot(&t_gt.normalize()).abs()
            } else {
                1.0
            };

            // Check normal direction.
            let n_cos = decomp.normal.normalize().dot(&n_gt.normalize()).abs();

            if ang_deg < 1.0 && t_cos > 0.99 && n_cos > 0.99 {
                found = true;
            }
        }

        assert!(
            found,
            "decompose_homography did not recover the correct pose/normal"
        );
    }

    #[test]
    fn decompose_pure_rotation() {
        let rot = Rotation3::from_euler_angles(0.1, -0.05, 0.02);
        let h: HomographyMatrix = *rot.matrix();

        let decomps = decompose_homography(&h).unwrap();
        assert!(!decomps.is_empty());

        // For pure rotation, t should be ~zero.
        let best = decomps
            .iter()
            .min_by(|a, b| a.t.norm().partial_cmp(&b.t.norm()).unwrap())
            .unwrap();
        assert!(best.t.norm() < 1e-6, "expected zero translation for pure rotation");

        let r_diff = best.r.transpose() * rot.matrix();
        let cos_theta = ((r_diff.trace() - 1.0) * 0.5).clamp(-1.0, 1.0);
        let ang_deg = cos_theta.acos().to_degrees();
        assert!(ang_deg < 0.01, "rotation error: {} deg", ang_deg);
    }

    #[test]
    fn identity_homography() {
        let h = HomographyMatrix::identity();
        let pt = Pt2::new(0.5, -0.3);
        let pt2 = homography_transfer(&h, &pt);
        assert!((pt2.x - pt.x).abs() < 1e-15);
        assert!((pt2.y - pt.y).abs() < 1e-15);
    }
}
