use calib_core::{Mat3, Pt2};
use nalgebra::DMatrix;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum HomographyError {
    #[error("need at least 4 point correspondences, got {0}")]
    NotEnoughPoints(usize),
    #[error("svd failed")]
    SvdFailed,
}

/// Estimate H such that x' ~ H x using DLT.
pub fn dlt_homography(world: &[Pt2], image: &[Pt2]) -> Result<Mat3, HomographyError> {
    let n = world.len();
    if n < 4 || image.len() != n {
        return Err(HomographyError::NotEnoughPoints(n));
    }

    let mut a = DMatrix::<f64>::zeros(2 * n, 9);

    for (i, (pw, pi)) in world.iter().zip(image.iter()).enumerate() {
        let x = pw.x;
        let y = pw.y;
        let u = pi.x;
        let v = pi.y;

        let r0 = 2 * i;
        let r1 = 2 * i + 1;

        a[(r0, 0)] = -x;
        a[(r0, 1)] = -y;
        a[(r0, 2)] = -1.0;
        a[(r0, 6)] = u * x;
        a[(r0, 7)] = u * y;
        a[(r0, 8)] = u;

        a[(r1, 3)] = -x;
        a[(r1, 4)] = -y;
        a[(r1, 5)] = -1.0;
        a[(r1, 6)] = v * x;
        a[(r1, 7)] = v * y;
        a[(r1, 8)] = v;
    }

    // Solve A h = 0 via SVD (smallest singular value)
    let svd = a.svd(true, false);
    let v_t = svd.v_t.ok_or(HomographyError::SvdFailed)?;
    let h = v_t.row(v_t.nrows() - 1);

    let mut h_mat = Mat3::zeros();
    for r in 0..3 {
        for c in 0..3 {
            h_mat[(r, c)] = h[3 * r + c];
        }
    }

    // normalise such that H[2,2] = 1
    let scale = h_mat[(2, 2)];
    if scale.abs() > f64::EPSILON {
        h_mat /= scale;
    }

    Ok(h_mat)
}

#[cfg(test)]
mod tests {
    use super::*;
    use calib_core::Pt2;

    #[test]
    fn basic_homography() {
        let w = vec![
            Pt2::new(0.0, 0.0),
            Pt2::new(1.0, 0.0),
            Pt2::new(1.0, 1.0),
            Pt2::new(0.0, 1.0),
        ];
        let img = vec![
            Pt2::new(0.0, 0.0),
            Pt2::new(2.0, 0.0),
            Pt2::new(2.0, 2.0),
            Pt2::new(0.0, 2.0),
        ];

        let h = dlt_homography(&w, &img).unwrap();
        let s = h[(0, 0)];
        assert!((s - 2.0).abs() < 1e-6);
    }
}
