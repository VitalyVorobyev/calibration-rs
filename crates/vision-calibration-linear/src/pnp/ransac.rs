//! RANSAC-based robust PnP estimation.
//!
//! Wraps the DLT solver in a RANSAC loop for outlier rejection, using
//! pixel reprojection error as the residual metric.

use super::dlt;
use anyhow::Result;
use vision_calibration_core::{
    Camera, Estimator, FxFyCxCySkew, IdentitySensor, Iso3, NoDistortion, Pinhole, Pt2, Pt3,
    RansacOptions, Real, ransac_fit,
};

/// Robust PnP using DLT inside a RANSAC loop.
///
/// Returns the best pose and inlier indices. The residual is pixel
/// reprojection error using the provided intrinsics.
pub fn dlt_ransac(
    world: &[Pt3],
    image: &[Pt2],
    k: &FxFyCxCySkew<Real>,
    opts: &RansacOptions,
) -> Result<(Iso3, Vec<usize>)> {
    let n = world.len();
    if n < 6 || image.len() != n {
        anyhow::bail!("need at least 6 point correspondences, got {}", n);
    }

    #[derive(Clone)]
    struct PnpDatum {
        pw: Pt3,
        pi: Pt2,
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
            dlt::dlt(&world, &image, &k).ok()
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
        anyhow::bail!("ransac failed to find a consensus PnP solution");
    }
    let pose = res.model.expect("success guarantees a model");
    Ok((pose, res.inliers))
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Isometry3, Rotation3, Translation3};
    use vision_calibration_core::{Camera, IdentitySensor, NoDistortion, Pinhole};

    #[test]
    fn ransac_handles_outliers() {
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
            image.push(Pt2::new(
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

        let (est, inliers) = dlt_ransac(&world, &image, &k, &opts).unwrap();

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
