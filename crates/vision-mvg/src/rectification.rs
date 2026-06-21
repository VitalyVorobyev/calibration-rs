//! Scheimpflug-aware stereo rectification.
//!
//! Computes rectifying homographies for a calibrated stereo pair so that
//! corresponding points fall on the **same image row** — the precondition for
//! 1-D disparity search. Unlike the textbook (Fusiello / Bouguet) construction,
//! this handles **Scheimpflug** cameras whose sensor plane is tilted relative to
//! the optical axis.
//!
//! # Why the tilt is "free"
//!
//! In this project's camera model (ADR 0005) a pixel is formed as
//!
//! ```text
//! pixel = K · H_tilt · x_n            (homogeneous, then dehomogenize)
//! ```
//!
//! where `x_n` is the undistorted normalized projection of the viewing ray and
//! `H_tilt` is the Scheimpflug sensor homography
//! ([`ScheimpflugParams::compile`]). The tilt is therefore just another
//! projective factor on the normalized plane. Pre-multiplying each camera's
//! unprojection by `H_tilt⁻¹` collapses a Scheimpflug camera to an **equivalent
//! frontal pinhole** in `x_n`:
//!
//! ```text
//! x_n = H_tilt⁻¹ · K⁻¹ · pixel
//! ```
//!
//! Standard rectification then applies to the frontal-equivalent cameras. Each
//! rectifying homography is
//!
//! ```text
//! H_left  = K_rect · R_rect          · H_tilt0⁻¹ · K0⁻¹
//! H_right = K_rect · (R_rect · Rᵀ)   · H_tilt1⁻¹ · K1⁻¹
//! ```
//!
//! with `R = R_C1_C0` the inter-camera rotation and `R_rect` the common rectified
//! orientation whose x-axis is the baseline. When the tilt is zero
//! `H_tilt⁻¹ = I` and this reduces exactly to standard pinhole rectification.
//!
//! # Conventions and caveats
//!
//! - Inputs are **undistorted** pixel coordinates — remove lens distortion
//!   (e.g. [`BrownConrady5::undistort`](vision_calibration_core::BrownConrady5)
//!   in normalized space) before applying these maps. The rectifying homography
//!   is the *rotation* part of rectification; distortion is handled separately,
//!   exactly as OpenCV splits `initUndistortRectifyMap`.
//! - The relative pose is `cam1_se3_cam0` (`T_C1_C0`): it maps a point from
//!   camera 0's frame into camera 1's frame (ADR 0009). For a
//!   `RigExtrinsicsExport` with camera 0 as the reference this is simply
//!   `cam_se3_rig[1]`.
//! - Camera 0 is treated as the left/reference camera; its frame is the world
//!   frame of the rectification.

use crate::{MvgError, Result};
use nalgebra::{Matrix3, Vector3};
use vision_calibration_core::{Iso3, Mat3, Pt2, Real, ScheimpflugParams, Vec3};

/// Numerical floor for "this vector is effectively zero".
const EPS: Real = 1e-12;
/// Largest Scheimpflug tilt (radians) we accept before declaring the sensor
/// homography degenerate (~80°; physical tilts are a few degrees).
const MAX_TILT_RAD: Real = 1.4;

/// Frozen calibration of one camera in the stereo pair.
#[derive(Clone, Copy, Debug)]
pub struct RectifyCamera {
    /// Intrinsic matrix `K`.
    pub k: Mat3,
    /// Scheimpflug sensor tilt. Use [`ScheimpflugParams::default`] (zero tilt)
    /// for an ordinary pinhole camera.
    pub tilt: ScheimpflugParams,
}

impl RectifyCamera {
    /// A pinhole camera (no sensor tilt).
    pub fn pinhole(k: Mat3) -> Self {
        Self {
            k,
            tilt: ScheimpflugParams::default(),
        }
    }

    /// A Scheimpflug camera with the given tilt.
    pub fn scheimpflug(k: Mat3, tilt: ScheimpflugParams) -> Self {
        Self { k, tilt }
    }
}

/// Options controlling the shared rectified intrinsics.
#[derive(Clone, Copy, Debug, Default)]
pub struct RectifyOptions {
    /// Explicit shared rectified intrinsics `K_rect`. When `None` (the default)
    /// the two input cameras' focal lengths and principal points are averaged
    /// (with zero skew). Both rectified cameras must share `K_rect` for rows to
    /// align, so this is a single matrix rather than one per camera.
    pub k_rect: Option<Mat3>,
}

/// Rectifying maps for a stereo pair.
#[derive(Clone, Debug)]
pub struct StereoRectification {
    /// Homography mapping **undistorted** left (camera 0) pixels to rectified
    /// pixels.
    pub h_left: Mat3,
    /// Homography mapping **undistorted** right (camera 1) pixels to rectified
    /// pixels.
    pub h_right: Mat3,
    /// The shared rectified intrinsics actually used.
    pub k_rect: Mat3,
    /// Common rectified orientation: maps a direction in camera 0's frame into
    /// the rectified frame (rows are the new x/y/z axes).
    pub r_rect: Mat3,
    /// Baseline length `‖t‖` between the two optical centres (calibration units).
    pub baseline: Real,
}

impl StereoRectification {
    /// Apply the left rectifying homography to an undistorted pixel.
    pub fn rectify_left(&self, px: &Pt2) -> Pt2 {
        apply_homography(&self.h_left, px)
    }

    /// Apply the right rectifying homography to an undistorted pixel.
    pub fn rectify_right(&self, px: &Pt2) -> Pt2 {
        apply_homography(&self.h_right, px)
    }
}

/// Compute Scheimpflug-aware rectifying homographies for a calibrated stereo
/// pair.
///
/// `left` / `right` carry each camera's intrinsics and Scheimpflug tilt;
/// `cam1_se3_cam0` is the relative pose `T_C1_C0` (camera 0 → camera 1).
///
/// # Errors
///
/// - [`MvgError::Degenerate`] if the baseline is ~zero (coincident centres) or a
///   Scheimpflug tilt is out of range (≳ 80°, where the sensor homography
///   degenerates).
/// - [`MvgError::Numerical`] if an intrinsic matrix is not invertible.
pub fn rectify_stereo_pair(
    left: &RectifyCamera,
    right: &RectifyCamera,
    cam1_se3_cam0: &Iso3,
    opts: &RectifyOptions,
) -> Result<StereoRectification> {
    // Inter-camera rotation R = R_C1_C0 and translation t = t_C1_C0.
    let r = *cam1_se3_cam0.rotation.to_rotation_matrix().matrix();
    let t = cam1_se3_cam0.translation.vector;
    let baseline = t.norm();
    if baseline < EPS {
        return Err(MvgError::degenerate(
            "zero baseline: the two optical centres coincide",
        ));
    }

    // Camera-1 centre expressed in camera-0's frame: c1 = -Rᵀ t (camera-0 centre
    // is the origin). The rectified x-axis points along the baseline c1 - c0.
    let c1 = -(r.transpose() * t);
    let v1 = c1 / c1.norm();

    // y-axis ⟂ to the baseline and the original (camera-0) optical axis z.
    let z = Vec3::new(0.0, 0.0, 1.0);
    let mut v2 = z.cross(&v1);
    if v2.norm() < EPS {
        // Baseline parallel to the optical axis (forward stereo): fall back to a
        // different seed so the cross product is well-defined.
        v2 = Vec3::new(0.0, 1.0, 0.0).cross(&v1);
    }
    let v2 = v2 / v2.norm();
    let v3 = v1.cross(&v2);
    let r_rect = Matrix3::from_rows(&[v1.transpose(), v2.transpose(), v3.transpose()]);

    let k0_inv = left
        .k
        .try_inverse()
        .ok_or_else(|| MvgError::numerical("left camera matrix K is not invertible"))?;
    let k1_inv = right
        .k
        .try_inverse()
        .ok_or_else(|| MvgError::numerical("right camera matrix K is not invertible"))?;
    let h_tilt0_inv = tilt_inverse(&left.tilt)?;
    let h_tilt1_inv = tilt_inverse(&right.tilt)?;

    let k_rect = opts
        .k_rect
        .unwrap_or_else(|| average_intrinsics(&left.k, &right.k));

    let h_left = k_rect * r_rect * h_tilt0_inv * k0_inv;
    let h_right = k_rect * (r_rect * r.transpose()) * h_tilt1_inv * k1_inv;

    Ok(StereoRectification {
        h_left,
        h_right,
        k_rect,
        r_rect,
        baseline,
    })
}

/// Inverse of a Scheimpflug sensor homography, with a range guard so we never
/// trip [`ScheimpflugParams::compile`]'s internal panic on a singular tilt.
fn tilt_inverse(p: &ScheimpflugParams) -> Result<Mat3> {
    if p.tilt_x.abs() >= MAX_TILT_RAD || p.tilt_y.abs() >= MAX_TILT_RAD {
        return Err(MvgError::degenerate(format!(
            "Scheimpflug tilt out of range: (tilt_x={}, tilt_y={})",
            p.tilt_x, p.tilt_y
        )));
    }
    Ok(p.compile().h_inv)
}

/// Average two intrinsic matrices (focal lengths + principal points), forcing
/// the structural zeros and `K[2,2] = 1`.
fn average_intrinsics(a: &Mat3, b: &Mat3) -> Mat3 {
    Matrix3::new(
        0.5 * (a[(0, 0)] + b[(0, 0)]),
        0.0,
        0.5 * (a[(0, 2)] + b[(0, 2)]),
        0.0,
        0.5 * (a[(1, 1)] + b[(1, 1)]),
        0.5 * (a[(1, 2)] + b[(1, 2)]),
        0.0,
        0.0,
        1.0,
    )
}

/// Apply a homography to a pixel and dehomogenize.
fn apply_homography(h: &Mat3, px: &Pt2) -> Pt2 {
    let v = h * Vector3::new(px.x, px.y, 1.0);
    Pt2::new(v.x / v.z, v.y / v.z)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Rotation3, Translation3, UnitQuaternion};
    use vision_calibration_core::{
        BrownConrady5, Camera, FxFyCxCySkew, IdentitySensor, Pinhole, Pt3,
    };

    type ScheimpflugCamera = Camera<
        Real,
        Pinhole,
        BrownConrady5<Real>,
        vision_calibration_core::HomographySensor<Real>,
        FxFyCxCySkew<Real>,
    >;

    fn no_distortion() -> BrownConrady5<Real> {
        BrownConrady5 {
            k1: 0.0,
            k2: 0.0,
            k3: 0.0,
            p1: 0.0,
            p2: 0.0,
            iters: 5,
        }
    }

    fn intr(fx: Real, fy: Real, cx: Real, cy: Real) -> FxFyCxCySkew<Real> {
        FxFyCxCySkew {
            fx,
            fy,
            cx,
            cy,
            skew: 0.0,
        }
    }

    /// Build the full Scheimpflug projection camera (zero distortion) used to
    /// generate synthetic observations through the *real* core model.
    fn scheimpflug_camera(k: &FxFyCxCySkew<Real>, tilt: ScheimpflugParams) -> ScheimpflugCamera {
        Camera::new(Pinhole, no_distortion(), tilt.compile(), *k)
    }

    /// A frontal pinhole projection camera (identity sensor).
    fn pinhole_camera(
        k: &FxFyCxCySkew<Real>,
    ) -> Camera<Real, Pinhole, BrownConrady5<Real>, IdentitySensor, FxFyCxCySkew<Real>> {
        Camera::new(Pinhole, no_distortion(), IdentitySensor, *k)
    }

    fn relative_pose(euler: (Real, Real, Real), t: (Real, Real, Real)) -> Iso3 {
        let rot = Rotation3::from_euler_angles(euler.0, euler.1, euler.2);
        Iso3::from_parts(Translation3::new(t.0, t.1, t.2), UnitQuaternion::from(rot))
    }

    /// A spread of 3D points, all comfortably in front of both cameras.
    fn scene_points() -> Vec<Pt3> {
        let mut pts = Vec::new();
        for i in 0..5 {
            for j in 0..5 {
                let x = -0.5 + 0.25 * i as Real;
                let y = -0.5 + 0.25 * j as Real;
                let z = 4.0 + 0.4 * ((i * 3 + j) % 4) as Real;
                pts.push(Pt3::new(x, y, z));
            }
        }
        pts
    }

    /// Project every point into both cameras (camera 1 sees `R·p + t`), keeping
    /// only points in front of both. Returns the rectified-row check inputs.
    fn project_pair<C0, C1>(cam0: &C0, cam1: &C1, pose: &Iso3, pts: &[Pt3]) -> Vec<(Pt2, Pt2)>
    where
        C0: ProjectLike,
        C1: ProjectLike,
    {
        let r = *pose.rotation.to_rotation_matrix().matrix();
        let t = pose.translation.vector;
        let mut out = Vec::new();
        for p in pts {
            let p0 = Vector3::new(p.x, p.y, p.z);
            let p1 = r * p0 + t;
            if let (Some(a), Some(b)) = (cam0.project(&p0), cam1.project(&p1)) {
                out.push((a, b));
            }
        }
        out
    }

    /// Minimal projection abstraction so the two concrete `Camera<...>` types can
    /// share `project_pair`.
    trait ProjectLike {
        fn project(&self, p_c: &Vector3<Real>) -> Option<Pt2>;
    }
    impl<Sm> ProjectLike for Camera<Real, Pinhole, BrownConrady5<Real>, Sm, FxFyCxCySkew<Real>>
    where
        Sm: vision_calibration_core::SensorModel<Real>,
    {
        fn project(&self, p_c: &Vector3<Real>) -> Option<Pt2> {
            self.project_point_c(p_c)
        }
    }

    fn max_row_disagreement(pairs: &[(Pt2, Pt2)], rect: &StereoRectification) -> Real {
        pairs
            .iter()
            .map(|(a, b)| {
                let ra = rect.rectify_left(a);
                let rb = rect.rectify_right(b);
                (ra.y - rb.y).abs()
            })
            .fold(0.0, Real::max)
    }

    #[test]
    fn scheimpflug_pair_rows_align() {
        let k0 = intr(800.0, 810.0, 320.0, 240.0);
        let k1 = intr(790.0, 805.0, 318.0, 242.0);
        let tilt0 = ScheimpflugParams {
            tilt_x: 0.09,
            tilt_y: -0.04,
        };
        let tilt1 = ScheimpflugParams {
            tilt_x: -0.06,
            tilt_y: 0.05,
        };
        let pose = relative_pose((0.02, 0.04, -0.01), (-1.0, 0.05, 0.1));

        let cam0 = scheimpflug_camera(&k0, tilt0);
        let cam1 = scheimpflug_camera(&k1, tilt1);
        let pairs = project_pair(&cam0, &cam1, &pose, &scene_points());
        assert!(pairs.len() >= 20, "too few visible points: {}", pairs.len());

        let rect = rectify_stereo_pair(
            &RectifyCamera::scheimpflug(k0.k_matrix(), tilt0),
            &RectifyCamera::scheimpflug(k1.k_matrix(), tilt1),
            &pose,
            &RectifyOptions::default(),
        )
        .unwrap();

        let max_dv = max_row_disagreement(&pairs, &rect);
        assert!(max_dv < 1e-6, "rows not aligned: max |Δv| = {max_dv}");
    }

    #[test]
    fn zero_tilt_matches_standard_rectification() {
        // With zero tilt the Scheimpflug path must coincide with plain pinhole
        // rectification (H_tilt⁻¹ = I) and still row-align.
        let k0 = intr(800.0, 800.0, 320.0, 240.0);
        let k1 = intr(800.0, 800.0, 320.0, 240.0);
        let pose = relative_pose((0.0, 0.03, 0.0), (-1.2, 0.0, 0.0));

        let cam0 = pinhole_camera(&k0);
        let cam1 = pinhole_camera(&k1);
        let pairs = project_pair(&cam0, &cam1, &pose, &scene_points());

        let rect = rectify_stereo_pair(
            &RectifyCamera::pinhole(k0.k_matrix()),
            &RectifyCamera::pinhole(k1.k_matrix()),
            &pose,
            &RectifyOptions::default(),
        )
        .unwrap();

        // A pinhole RectifyCamera and a zero-tilt Scheimpflug one must produce
        // identical maps.
        let rect_via_tilt = rectify_stereo_pair(
            &RectifyCamera::scheimpflug(k0.k_matrix(), ScheimpflugParams::default()),
            &RectifyCamera::scheimpflug(k1.k_matrix(), ScheimpflugParams::default()),
            &pose,
            &RectifyOptions::default(),
        )
        .unwrap();
        assert!((rect.h_left - rect_via_tilt.h_left).abs().max() < 1e-12);
        assert!((rect.h_right - rect_via_tilt.h_right).abs().max() < 1e-12);

        let max_dv = max_row_disagreement(&pairs, &rect);
        assert!(max_dv < 1e-6, "rows not aligned: max |Δv| = {max_dv}");
    }

    #[test]
    fn baseline_is_recovered() {
        let k = intr(800.0, 800.0, 320.0, 240.0);
        let pose = relative_pose((0.01, 0.0, 0.0), (-0.8, 0.3, 0.2));
        let rect = rectify_stereo_pair(
            &RectifyCamera::pinhole(k.k_matrix()),
            &RectifyCamera::pinhole(k.k_matrix()),
            &pose,
            &RectifyOptions::default(),
        )
        .unwrap();
        let expected = (0.8f64 * 0.8 + 0.3 * 0.3 + 0.2 * 0.2).sqrt();
        assert!((rect.baseline - expected).abs() < 1e-12);
    }

    #[test]
    fn explicit_k_rect_is_used() {
        let k = intr(800.0, 800.0, 320.0, 240.0);
        let pose = relative_pose((0.0, 0.0, 0.0), (-1.0, 0.0, 0.0));
        let k_rect = intr(1000.0, 1000.0, 256.0, 256.0).k_matrix();
        let rect = rectify_stereo_pair(
            &RectifyCamera::pinhole(k.k_matrix()),
            &RectifyCamera::pinhole(k.k_matrix()),
            &pose,
            &RectifyOptions {
                k_rect: Some(k_rect),
            },
        )
        .unwrap();
        assert!((rect.k_rect - k_rect).abs().max() < 1e-15);
    }

    #[test]
    fn rejects_zero_baseline() {
        let k = intr(800.0, 800.0, 320.0, 240.0);
        let pose = relative_pose((0.05, 0.0, 0.0), (0.0, 0.0, 0.0));
        let err = rectify_stereo_pair(
            &RectifyCamera::pinhole(k.k_matrix()),
            &RectifyCamera::pinhole(k.k_matrix()),
            &pose,
            &RectifyOptions::default(),
        )
        .unwrap_err();
        assert!(matches!(err, MvgError::Degenerate { .. }), "got {err:?}");
    }

    #[test]
    fn rejects_extreme_tilt() {
        let k = intr(800.0, 800.0, 320.0, 240.0);
        let pose = relative_pose((0.0, 0.0, 0.0), (-1.0, 0.0, 0.0));
        let err = rectify_stereo_pair(
            &RectifyCamera::scheimpflug(
                k.k_matrix(),
                ScheimpflugParams {
                    tilt_x: 1.5,
                    tilt_y: 0.0,
                },
            ),
            &RectifyCamera::pinhole(k.k_matrix()),
            &pose,
            &RectifyOptions::default(),
        )
        .unwrap_err();
        assert!(matches!(err, MvgError::Degenerate { .. }), "got {err:?}");
    }
}
