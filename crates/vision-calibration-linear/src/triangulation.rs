//! Linear triangulation of 3D points from multiple views.
//!
//! Uses a DLT formulation on the camera projection matrices and image points.

use anyhow::Result;
use nalgebra::DMatrix;
use vision_calibration_core::{Pt2, Pt3, Real};

use crate::camera_matrix::Mat34;

/// Linear triangulation from multiple views using DLT.
///
/// `cameras` are projection matrices `P_i`, and `points` are their corresponding
/// pixel coordinates. The returned 3D point is in the same world frame as the
/// camera matrices.
pub fn triangulate_point_linear(cameras: &[Mat34], points: &[Pt2]) -> Result<Pt3> {
    if cameras.len() < 2 {
        anyhow::bail!("need at least 2 views, got {}", cameras.len());
    }
    if cameras.len() != points.len() {
        anyhow::bail!(
            "mismatched number of cameras ({}) and points ({})",
            cameras.len(),
            points.len()
        );
    }

    let mut a = DMatrix::<Real>::zeros(2 * cameras.len(), 4);
    for (i, (p, cam)) in points.iter().zip(cameras.iter()).enumerate() {
        let u = p.x;
        let v = p.y;

        let r0 = 2 * i;
        let r1 = 2 * i + 1;

        let row0 = cam.row(0);
        let row1 = cam.row(1);
        let row2 = cam.row(2);

        a.row_mut(r0).copy_from(&(u * row2 - row0));
        a.row_mut(r1).copy_from(&(v * row2 - row1));
    }

    let svd = a.svd(true, true);
    let v_t = svd
        .v_t
        .ok_or_else(|| anyhow::anyhow!("svd failed during triangulation"))?;
    let x_h = v_t.row(v_t.nrows() - 1);

    let w = x_h[3];
    if w.abs() <= Real::EPSILON {
        anyhow::bail!("triangulation produced an invalid point");
    }

    let x = x_h[0] / w;
    let y = x_h[1] / w;
    let z = x_h[2] / w;

    Ok(Pt3::new(x, y, z))
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Vector4;

    fn project(cam: &Mat34, p: &Pt3) -> Pt2 {
        let x = cam * Vector4::new(p.x, p.y, p.z, 1.0);
        Pt2::new(x.x / x.z, x.y / x.z)
    }

    #[test]
    fn triangulation_two_views_recovers_point() {
        let cam1 = Mat34::new(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
        let cam2 = Mat34::new(1.0, 0.0, 0.0, -0.2, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);

        let pw = Pt3::new(0.1, -0.05, 2.0);
        let p1 = project(&cam1, &pw);
        let p2 = project(&cam2, &pw);

        let est = triangulate_point_linear(&[cam1, cam2], &[p1, p2]).unwrap();

        let err = (est - pw).norm();
        assert!(err < 1e-6, "triangulation error too large: {}", err);
    }
}
