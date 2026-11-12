//! Bearings: the unit rays a rig camera's own unprojection gives for pixels.
//!
//! Closed-form multiple-view steps work on bearings, so they serve every camera
//! model, including a fisheye that sees beyond 90° from its axis, where
//! normalized coordinates do not exist. Every projection model is central (the
//! ray starts at the camera centre), so the ray's origin is ignored.

use crate::{Error, Result};
use nalgebra::{Matrix2, Matrix3x2};
use vision_calibration_core::{CalibratedRig, CameraModel, Pt2, Real, Vec3};

/// Camera `camera` of the rig, or `InvalidInput` when out of range.
pub(crate) fn camera_model(rig: &CalibratedRig, camera: usize) -> Result<&CameraModel> {
    rig.model(camera).ok_or_else(|| {
        Error::invalid_input(format!(
            "camera {camera} out of range: the rig has {} cameras",
            rig.num_cameras()
        ))
    })
}

/// The unit bearing of `pixel` through `model`. `index` names the pixel in
/// the error when its ray is zero or not finite.
///
/// The unprojection inverts the distortion iteratively, and returns its best
/// effort for a pixel no ray reaches (beyond where a strong barrel
/// distortion folds over). The bearing is accepted only when it projects
/// back onto the pixel within [`ROUND_TRIP_TOL_PX`]; otherwise the pixel is
/// outside the region where the model inverts, and that is an error.
pub(crate) fn bearing(model: &CameraModel, pixel: &Pt2, index: usize) -> Result<Vec3> {
    let d = model.backproject_pixel(pixel).direction;
    let n = d.norm();
    if !(n.is_finite() && n > 0.0) {
        return Err(Error::invalid_input(format!(
            "pixel {index} ({}, {}) has no finite viewing ray",
            pixel.x, pixel.y
        )));
    }
    let b = d / n;
    let back = model.project_point_c(&b);
    match back {
        Some(p) if (p - pixel).norm() <= ROUND_TRIP_TOL_PX => Ok(b),
        _ => Err(Error::invalid_input(format!(
            "pixel {index} ({}, {}) is outside the region where the camera model \
             inverts: no viewing ray projects onto it",
            pixel.x, pixel.y
        ))),
    }
}

/// How far, in pixels, a bearing may project from the pixel it came from.
/// The undistortion converges to rounding where it converges at all; a
/// pixel no ray reaches is off by far more.
const ROUND_TRIP_TOL_PX: Real = 1e-3;

/// The unit bearing `b` of `pixel` through `model` and the Jacobian
/// `J = ∂b/∂pixel` (3×2, `bᵀ J = 0`): a pixel error `δ` moves the bearing by
/// `J δ`, to first order.
///
/// `J` comes from the forward projection `π`, which is closed form, so it does
/// not depend on the iterative undistortion's tolerance. With `T` an
/// orthonormal basis of the plane perpendicular to `b`, central differences
/// give `P = ∂π(b + T s)/∂s` (2×2) and `J = T P⁻¹`. `index` names the pixel in
/// the error.
pub(crate) fn bearing_with_jacobian(
    model: &CameraModel,
    pixel: &Pt2,
    index: usize,
) -> Result<(Vec3, Matrix3x2<Real>)> {
    // |b| = 1, so a step h is an angle of h rad: about 1e-3 px at f = 1000.
    // The central difference's truncation error is O(h²) ≈ 1e-12 relative and
    // its roundoff about ε/h ≈ 1e-10, both far below the noise being mapped.
    const H: Real = 1e-6;

    let b = bearing(model, pixel, index)?;
    let fail = || {
        Error::invalid_input(format!(
            "pixel {index} ({}, {}): the camera's projection has no usable derivative there",
            pixel.x, pixel.y
        ))
    };

    // An orthonormal basis of the plane perpendicular to b: the cross product
    // with the coordinate axis least aligned with b is well conditioned.
    let axis = if b.x.abs() <= b.y.abs() && b.x.abs() <= b.z.abs() {
        Vec3::x()
    } else if b.y.abs() <= b.z.abs() {
        Vec3::y()
    } else {
        Vec3::z()
    };
    let t1 = b.cross(&axis).normalize();
    let t2 = b.cross(&t1);

    let mut p = Matrix2::zeros();
    for (col, t) in [t1, t2].into_iter().enumerate() {
        let plus = model.project_point_c(&(b + t * H)).ok_or_else(fail)?;
        let minus = model.project_point_c(&(b - t * H)).ok_or_else(fail)?;
        p[(0, col)] = (plus.x - minus.x) / (2.0 * H);
        p[(1, col)] = (plus.y - minus.y) / (2.0 * H);
    }
    let p_inv = p.try_inverse().filter(|m| m.iter().all(|v| v.is_finite()));
    let p_inv = p_inv.ok_or_else(fail)?;
    let t = Matrix3x2::from_columns(&[t1, t2]);
    Ok((b, t * p_inv))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{fisheye_params, pinhole_params};
    use vision_calibration_core::{CameraParams, DistortionParams, KannalaBrandt4};

    fn kb4_pinhole_like() -> CameraParams {
        // A fisheye with a larger fourth-order term, away from the axis.
        let mut c = fisheye_params();
        c.distortion = DistortionParams::KannalaBrandt4 {
            params: KannalaBrandt4 {
                k1: 0.03,
                k2: -0.01,
                k3: 0.002,
                k4: -0.0005,
                iters: 0,
            },
        };
        c
    }

    /// `J` against central differences of the backprojection `bearing()`.
    fn check_against_backprojection(params: &CameraParams, pixels: &[Pt2]) {
        let model = params.build().unwrap();
        for px in pixels {
            let (b, j) = bearing_with_jacobian(&model, px, 0).unwrap();
            assert!((b.norm() - 1.0).abs() < 1e-12);
            assert!(
                (b.transpose() * j).norm() < 1e-9,
                "bᵀJ = {:?}",
                b.transpose() * j
            );
            let d = 1e-3;
            let mut fd = Matrix3x2::zeros();
            for (col, step) in [Pt2::new(d, 0.0), Pt2::new(0.0, d)].into_iter().enumerate() {
                let bp = bearing(&model, &(px + step.coords), 0).unwrap();
                let bm = bearing(&model, &(px - step.coords), 0).unwrap();
                fd.set_column(col, &((bp - bm) / (2.0 * d)));
            }
            let rel = (j - fd).norm() / fd.norm();
            assert!(rel < 1e-6, "pixel {px:?}: relative difference {rel}");
        }
    }

    #[test]
    fn jacobian_of_an_undistorted_pinhole_is_closed_form() {
        let mut params = pinhole_params(0.0);
        // No distortion: bearing = normalize((x, y, 1)).
        params.distortion = DistortionParams::None;
        let model = params.build().unwrap();
        let (fx, fy, cx, cy) = (800.0, 790.0, 640.0, 480.0);
        let px = Pt2::new(900.0, 300.0);
        let (b, j) = bearing_with_jacobian(&model, &px, 0).unwrap();
        let d = Vec3::new((px.x - cx) / fx, (px.y - cy) / fy, 1.0);
        let n = d.norm();
        assert!((b - d / n).norm() < 1e-14);
        // ∂(d/|d|)/∂d = (I − b bᵀ)/|d|; ∂d/∂pixel = diag(1/fx, 1/fy) on x, y.
        let proj = Matrix3x2::new(1.0 / fx, 0.0, 0.0, 1.0 / fy, 0.0, 0.0);
        let expected = (nalgebra::Matrix3::identity() - b * b.transpose()) * proj / n;
        assert!(
            (j - expected).norm() < 1e-8 * expected.norm(),
            "{j} vs {expected}"
        );
    }

    #[test]
    fn jacobian_matches_backprojection_for_brown_conrady_and_kb4() {
        check_against_backprojection(
            &pinhole_params(-0.2),
            &[
                Pt2::new(640.0, 480.0),
                Pt2::new(100.0, 80.0),
                Pt2::new(1100.0, 700.0),
            ],
        );
        check_against_backprojection(
            &kb4_pinhole_like(),
            &[
                Pt2::new(640.0, 480.0),
                Pt2::new(800.0, 520.0),
                Pt2::new(300.0, 200.0),
                Pt2::new(1000.0, 900.0),
            ],
        );
    }

    #[test]
    fn jacobian_names_a_pixel_without_a_ray() {
        let model = pinhole_params(0.0).build().unwrap();
        let err = bearing_with_jacobian(&model, &Pt2::new(f64::NAN, 0.0), 7).unwrap_err();
        assert!(err.to_string().contains("pixel 7"), "{err}");
    }

    /// Barrel distortion `k1 = −0.3` folds over at a normalized radius of
    /// `1/√(3·0.3)`: no ray lands beyond a distorted radius of about 0.70
    /// (562 px). A pixel there has no bearing, and the undistortion's best
    /// effort is not one.
    #[test]
    fn a_pixel_outside_the_invertible_region_has_no_bearing() {
        let model = pinhole_params(-0.3).build().unwrap();
        assert!(bearing(&model, &Pt2::new(640.0 + 400.0, 480.0), 0).is_ok());
        let err = bearing(&model, &Pt2::new(640.0 + 700.0, 480.0), 7).unwrap_err();
        assert!(matches!(err, Error::InvalidInput { .. }), "{err:?}");
    }
}
