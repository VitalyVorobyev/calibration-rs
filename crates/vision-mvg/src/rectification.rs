//! Stereo from a calibration: rectification, disparity to depth, and the
//! uncertainty of the points.
//!
//! [`rectify`] turns two cameras of a [`CalibratedRig`] into two virtual
//! pinholes that share one set of intrinsics and an orientation whose x axis is
//! the baseline, so matching pixels lie on one image row. The maps go through
//! each camera's own unprojection and projection: raw pixels go in and any
//! camera model works (lens distortion, a fisheye, a tilted sensor), with no
//! separate undistortion step.
//!
//! On the rectified pair a match is a left pixel `(x, y)` and a disparity
//! `d = x_left − x_right > 0`. [`StereoRectification::point`] gives the 3D
//! point, [`StereoRectification::disparity_to_depth`] the matrix `Q` that does
//! it for a whole map, and [`StereoRectification::disparity_covariance`] the
//! covariance that the disparity's noise alone gives it.
//! [`StereoRectification::triangulate`] gives measurements that also carry the
//! calibration's uncertainty.
//!
//! # Frames
//!
//! Two frames appear. [`point`](StereoRectification::point) and
//! [`disparity_covariance`](StereoRectification::disparity_covariance) are in
//! the **left camera's frame** (`cameras[0]`), as a dense point cloud is.
//! [`triangulate`](StereoRectification::triangulate) works on the rig, so its
//! points are in the **rig frame**.
//!
//! # Rectified geometry
//!
//! With `B` the baseline, `(fx, fy, cx, cy)` the rectified intrinsics and
//! `R₀` the rotation from the left camera into the rectified frame, a point
//! `P` of the rectified frame projects to `x_l = fx X/Z + cx` on the left and
//! `x_r = fx (X − B)/Z + cx` on the right, both on row `y = fy Y/Z + cy`.
//! Hence `d = fx B/Z`, and a disparity error `δd` moves the point along its
//! viewing ray: `δP = −P δd/d`.

use crate::bearing::{bearing, camera_model};
use crate::dense::GrayImage;
use crate::triangulation::{RigObservation, Triangulation, triangulate};
use crate::{MvgError, Result};
use nalgebra::Matrix4;
use serde::{Deserialize, Serialize};
use vision_calibration_core::{
    CalibratedRig, CameraModel, FxFyCxCy, Iso3, Mat3, PointCovariance, Pt2, Pt3, Real, Vec3,
};

/// Numerical floor for "this vector is effectively zero".
const EPS: Real = 1e-12;

/// How to rectify.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct RectifyOptions {
    /// The rectified pinhole's intrinsics. `None` takes the mean of the two
    /// cameras' `fx`, `fy`, `cx`, `cy`, multiplied by [`scale`](Self::scale).
    pub k_rect: Option<FxFyCxCy<Real>>,
    /// Scale of the default rectified intrinsics: 1 keeps the cameras' mean
    /// focal length, 0.5 gives a rectified image of half the size. Finite and
    /// positive; ignored when [`k_rect`](Self::k_rect) is set.
    pub scale: Real,
}

impl Default for RectifyOptions {
    fn default() -> Self {
        Self {
            k_rect: None,
            scale: 1.0,
        }
    }
}

/// A match on the rectified pair: the left rectified pixel and its disparity.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RectifiedMatch {
    /// The pixel in the rectified left image.
    pub left: Pt2,
    /// `x_left − x_right`, in pixels; positive for a point ahead of the pair.
    pub disparity: Real,
}

impl RectifiedMatch {
    /// A match at rectified left pixel `left` with disparity `disparity`.
    pub fn new(left: Pt2, disparity: Real) -> Self {
        Self { left, disparity }
    }
}

/// A rectified stereo pair of rig cameras: two virtual pinholes sharing one
/// set of intrinsics and an orientation whose x axis is the baseline, so
/// matches lie on one row. Built by [`rectify`].
#[derive(Debug, Clone)]
pub struct StereoRectification {
    cameras: [usize; 2],
    models: [CameraModel; 2],
    k_rect: FxFyCxCy<Real>,
    /// Rotation from camera `side`'s frame into the rectified frame.
    rect_r_cam: [Mat3; 2],
    baseline: Real,
    /// What the rectification was built from, to check a rig against.
    num_cameras: usize,
    cam_se3_rig: [Iso3; 2],
}

/// Rectify cameras `cameras = [left, right]` of `rig`.
///
/// The rectified orientation has its x axis along the baseline (from the left
/// camera's centre to the right's), its y axis perpendicular to x and to the
/// left optical axis, and z = x × y. A baseline along the optical axis
/// (forward stereo) takes the y axis from the left camera's y instead.
///
/// ```
/// use vision_calibration_core::{
///     BrownConrady5, CalibratedRig, FxFyCxCy, Iso3, Pt3, Uncertainty, make_pinhole_camera,
///     pinhole_camera_params,
/// };
/// use vision_mvg::rectification::{RectifyOptions, rectify};
///
/// // A stereo pair, camera 1 12 cm to the right of camera 0, with barrel distortion.
/// let k = FxFyCxCy { fx: 800.0, fy: 800.0, cx: 320.0, cy: 240.0 };
/// let distortion = BrownConrady5 { k1: -0.1, iters: 30, ..Default::default() };
/// let camera = pinhole_camera_params(&make_pinhole_camera(k, distortion));
/// let rig = CalibratedRig::new(
///     vec![camera.clone(), camera],
///     vec![Iso3::identity(), Iso3::translation(-0.12, 0.0, 0.0)],
///     Uncertainty::default(),
/// )?;
/// let rect = rectify(&rig, [0, 1], &RectifyOptions::default())?;
///
/// // A point 1.5 m ahead: its raw pixels share a row once rectified, and the
/// // disparity gives its position back.
/// let p = Pt3::new(0.1, -0.05, 1.5);
/// let raw = |i: usize| rig.model(i).unwrap().project_point(&(rig.cam_se3_rig()[i] * p)).unwrap();
/// let (left, right) = (
///     rect.rectified_pixel(0, &raw(0)).unwrap(),
///     rect.rectified_pixel(1, &raw(1)).unwrap(),
/// );
/// assert!((left.y - right.y).abs() < 1e-9);
/// let found = rect.point(&left, left.x - right.x).unwrap();
/// assert!((found - p).norm() < 1e-9);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// # Errors
///
/// [`MvgError::InvalidInput`] for equal or out-of-range camera indices, or a
/// `scale` or `k_rect` that is not finite and positive;
/// [`MvgError::Degenerate`] for coincident camera centres.
pub fn rectify(
    rig: &CalibratedRig,
    cameras: [usize; 2],
    opts: &RectifyOptions,
) -> Result<StereoRectification> {
    let [a, b] = cameras;
    if a == b {
        return Err(MvgError::invalid_input(format!(
            "a stereo pair needs two different cameras, got {a} twice"
        )));
    }
    let models = [camera_model(rig, a)?.clone(), camera_model(rig, b)?.clone()];
    let (ka, kb) = (rig.cameras()[a].intrinsics, rig.cameras()[b].intrinsics);

    let k_rect = match opts.k_rect {
        Some(k) => k,
        None => {
            if !(opts.scale.is_finite() && opts.scale > 0.0) {
                return Err(MvgError::invalid_input(format!(
                    "rectification scale must be finite and positive, got {}",
                    opts.scale
                )));
            }
            let mean = |p: Real, q: Real| 0.5 * (p + q) * opts.scale;
            FxFyCxCy {
                fx: mean(ka.fx, kb.fx),
                fy: mean(ka.fy, kb.fy),
                cx: mean(ka.cx, kb.cx),
                cy: mean(ka.cy, kb.cy),
            }
        }
    };
    let valid = [k_rect.fx, k_rect.fy, k_rect.cx, k_rect.cy]
        .iter()
        .all(|v| v.is_finite())
        && k_rect.fx > 0.0
        && k_rect.fy > 0.0;
    if !valid {
        return Err(MvgError::invalid_input(
            "the rectified intrinsics must be finite with positive focal lengths",
        ));
    }

    // The right camera's frame in the left's: p_left = left_se3_right · p_right.
    let cam_se3_rig = [rig.cam_se3_rig()[a], rig.cam_se3_rig()[b]];
    let left_se3_right = cam_se3_rig[0] * cam_se3_rig[1].inverse();
    let r_left_right = *left_se3_right.rotation.to_rotation_matrix().matrix();
    // The right camera's centre in the left frame is the translation.
    let centre = left_se3_right.translation.vector;
    let baseline = centre.norm();
    if !(baseline.is_finite() && baseline > EPS) {
        return Err(MvgError::degenerate(
            "zero baseline: the two optical centres coincide",
        ));
    }

    let x = centre / baseline;
    let mut y = Vec3::z().cross(&x);
    if y.norm() < EPS {
        // The baseline lies along the optical axis (forward stereo): seed with
        // the left camera's y axis so the cross product is defined.
        y = Vec3::y().cross(&x);
    }
    let y = y / y.norm();
    let z = x.cross(&y);
    let rect_r_left = Mat3::from_rows(&[x.transpose(), y.transpose(), z.transpose()]);

    Ok(StereoRectification {
        cameras,
        models,
        k_rect,
        rect_r_cam: [rect_r_left, rect_r_left * r_left_right],
        baseline,
        num_cameras: rig.num_cameras(),
        cam_se3_rig,
    })
}

impl StereoRectification {
    /// The rig cameras `[left, right]`.
    pub fn cameras(&self) -> [usize; 2] {
        self.cameras
    }

    /// The rectified pinhole's intrinsics, shared by both images.
    pub fn k_rect(&self) -> FxFyCxCy<Real> {
        self.k_rect
    }

    /// The distance between the two camera centres, in the rig's length unit
    /// (metres for a calibration in metres).
    pub fn baseline(&self) -> Real {
        self.baseline
    }

    /// The rotation from camera `side` (0 = left = `cameras()[0]`, 1 = right)
    /// into the rectified frame.
    ///
    /// # Panics
    ///
    /// When `side` is not 0 or 1.
    pub fn rect_r_cam(&self, side: usize) -> Mat3 {
        self.rect_r_cam[side]
    }

    /// The rectified pixel of a raw pixel of camera `side`; `None` when its
    /// ray has no finite bearing or points behind the rectified image plane.
    ///
    /// # Panics
    ///
    /// When `side` is not 0 or 1.
    pub fn rectified_pixel(&self, side: usize, raw: &Pt2) -> Option<Pt2> {
        let b = bearing(&self.models[side], raw, 0).ok()?;
        let v = self.rect_r_cam[side] * b;
        if !v.z.is_finite() || v.z <= 0.0 {
            return None;
        }
        let k = &self.k_rect;
        let p = Pt2::new(k.fx * v.x / v.z + k.cx, k.fy * v.y / v.z + k.cy);
        (p.x.is_finite() && p.y.is_finite()).then_some(p)
    }

    /// The raw pixel of camera `side` that a rectified pixel comes from (the
    /// remapping direction); `None` outside the camera's projection domain.
    ///
    /// # Panics
    ///
    /// When `side` is not 0 or 1.
    pub fn raw_pixel(&self, side: usize, rectified: &Pt2) -> Option<Pt2> {
        let k = &self.k_rect;
        let ray = Vec3::new(
            (rectified.x - k.cx) / k.fx,
            (rectified.y - k.cy) / k.fy,
            1.0,
        );
        let p = self.models[side].project_point_c(&(self.rect_r_cam[side].transpose() * ray))?;
        (p.x.is_finite() && p.y.is_finite()).then_some(p)
    }

    /// The rectified image of camera `side`: `width × height` pixels sampled
    /// bilinearly from the raw image `src`, 0 where the source has no pixel.
    ///
    /// The output size is free; with `scale` in [`RectifyOptions`] it is the
    /// source size times `scale`. The source is sampled, not filtered: a
    /// large reduction aliases, so reduce the source first or keep `scale`
    /// near 1.
    ///
    /// # Panics
    ///
    /// When `side` is not 0 or 1.
    pub fn rectify_image(
        &self,
        side: usize,
        src: &GrayImage,
        width: usize,
        height: usize,
    ) -> GrayImage {
        let mut data = vec![0u8; width * height];
        let (w, h) = (src.width as Real, src.height as Real);
        for y in 0..height {
            for x in 0..width {
                let Some(raw) = self.raw_pixel(side, &Pt2::new(x as Real, y as Real)) else {
                    continue;
                };
                if raw.x >= 0.0 && raw.y >= 0.0 && raw.x <= w - 1.0 && raw.y <= h - 1.0 {
                    data[y * width + x] =
                        src.bilinear(raw.x, raw.y).round().clamp(0.0, 255.0) as u8;
                }
            }
        }
        GrayImage::new(width, height, data)
    }

    /// The 4×4 disparity-to-depth matrix `Q`: `Q·[x, y, d, 1]ᵀ`, dehomogenized,
    /// is the point in the **rectified left frame** (OpenCV's convention, with
    /// `d = x_left − x_right > 0`).
    ///
    /// [`point`](Self::point) additionally rotates it into the left camera's
    /// frame.
    pub fn disparity_to_depth(&self) -> Matrix4<Real> {
        let k = &self.k_rect;
        let aspect = k.fx / k.fy;
        Matrix4::new(
            1.0,
            0.0,
            0.0,
            -k.cx,
            0.0,
            aspect,
            0.0,
            -k.cy * aspect,
            0.0,
            0.0,
            0.0,
            k.fx,
            0.0,
            0.0,
            1.0 / self.baseline,
            0.0,
        )
    }

    /// The point at rectified left pixel `left` and disparity `disparity`, in
    /// the **left camera's frame**; `None` for `disparity ≤ 0` or a value
    /// that is not finite.
    pub fn point(&self, left: &Pt2, disparity: Real) -> Option<Pt3> {
        self.rectified_point(left, disparity)
            .map(|p| Pt3::from(self.rect_r_cam[0].transpose() * p))
    }

    /// The point in the rectified left frame (`Z = fx B/d`).
    fn rectified_point(&self, left: &Pt2, disparity: Real) -> Option<Vec3> {
        if !(disparity.is_finite() && disparity > 0.0 && left.x.is_finite() && left.y.is_finite()) {
            return None;
        }
        let k = &self.k_rect;
        let z = k.fx * self.baseline / disparity;
        Some(Vec3::new(
            (left.x - k.cx) * z / k.fx,
            (left.y - k.cy) * z / k.fy,
            z,
        ))
    }

    /// The covariance, m², of [`point`](Self::point) that a disparity noise of
    /// `sigma_d` pixels alone gives it, in the left camera's frame: first
    /// order through `Q`, with the rectified row and column exact. It has rank
    /// 1, along the viewing ray, and its depth σ is `Z² σ_d/(fx B)`.
    ///
    /// It does not include the calibration's uncertainty (see
    /// [`triangulate`](Self::triangulate)). `None` where `point` is, or for a
    /// `sigma_d` that is negative or not finite.
    pub fn disparity_covariance(&self, left: &Pt2, disparity: Real, sigma_d: Real) -> Option<Mat3> {
        if !(sigma_d.is_finite() && sigma_d >= 0.0) {
            return None;
        }
        // δP = −P δd/d, so Σ = (σ_d/d)² P Pᵀ.
        let p = self.point(left, disparity)?.coords;
        Some(p * p.transpose() * (sigma_d / disparity).powi(2))
    }

    /// Measurements from matches on the rectified pair, with the
    /// calibration's uncertainty.
    ///
    /// Each match is mapped back to the raw pixels of the two cameras (the
    /// right one on the left's row) and triangulated with
    /// [`triangulate()`](crate::triangulation::triangulate()) as one stereo
    /// capture: the points are in the **rig frame**, refined on the
    /// reprojection error of the stated `pixel_noise` of each raw pixel, and
    /// the covariance is joint over the points with the calibration's share
    /// included.
    ///
    /// `rig` must be the one this rectification was built from.
    ///
    /// # Errors
    ///
    /// [`MvgError::InvalidInput`] when `rig` differs from the one rectified,
    /// or a match has no raw pixel in a camera; otherwise the errors of
    /// [`triangulate()`](crate::triangulation::triangulate()), whose messages
    /// number the matches as tracks.
    pub fn triangulate(
        &self,
        rig: &CalibratedRig,
        matches: &[RectifiedMatch],
        pixel_noise: PointCovariance,
    ) -> Result<Triangulation> {
        let [a, b] = self.cameras;
        let same_rig = rig.num_cameras() == self.num_cameras
            && rig
                .cam_se3_rig()
                .get(a)
                .zip(rig.cam_se3_rig().get(b))
                .is_some_and(|(pa, pb)| {
                    same_pose(pa, &self.cam_se3_rig[0]) && same_pose(pb, &self.cam_se3_rig[1])
                });
        if !same_rig {
            return Err(MvgError::invalid_input(
                "the rig is not the one this rectification was built from",
            ));
        }
        let tracks = matches
            .iter()
            .enumerate()
            .map(|(i, m)| {
                let right = Pt2::new(m.left.x - m.disparity, m.left.y);
                let missing = |side: usize| {
                    MvgError::invalid_input(format!(
                        "match {i}: no raw pixel in camera {} for rectified pixel ({}, {}) \
                         with disparity {}",
                        self.cameras[side], m.left.x, m.left.y, m.disparity
                    ))
                };
                let raw_left = self.raw_pixel(0, &m.left).ok_or_else(|| missing(0))?;
                let raw_right = self.raw_pixel(1, &right).ok_or_else(|| missing(1))?;
                Ok(vec![
                    RigObservation::new(0, a, raw_left, pixel_noise),
                    RigObservation::new(0, b, raw_right, pixel_noise),
                ])
            })
            .collect::<Result<Vec<_>>>()?;
        triangulate(rig, &[Iso3::identity()], &tracks)
    }
}

/// Whether two poses are the same to rounding.
fn same_pose(a: &Iso3, b: &Iso3) -> bool {
    (a.to_homogeneous() - b.to_homogeneous()).abs().max() <= 1e-12
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{iso, pinhole_params};
    use vision_calibration_core::Uncertainty;

    fn rig(baseline: Real) -> CalibratedRig {
        CalibratedRig::new(
            vec![pinhole_params(-0.1), pinhole_params(-0.1)],
            vec![
                Iso3::identity(),
                iso(0.0, 0.02, 0.0, Vec3::new(-baseline, 0.0, 0.0)),
            ],
            Uncertainty::default(),
        )
        .unwrap()
    }

    #[test]
    fn baseline_and_default_intrinsics() {
        let rect = rectify(&rig(0.12), [0, 1], &RectifyOptions::default()).unwrap();
        // Camera 1's centre is R^T(-t) away; rotation does not change the norm.
        assert!((rect.baseline() - 0.12).abs() < 1e-12);
        let k = rect.k_rect();
        assert_eq!((k.fx, k.fy, k.cx, k.cy), (800.0, 790.0, 640.0, 480.0));
        let half = rectify(
            &rig(0.12),
            [0, 1],
            &RectifyOptions {
                scale: 0.5,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(half.k_rect().fx, 400.0);
    }

    #[test]
    fn explicit_k_rect_is_used() {
        let k = FxFyCxCy {
            fx: 1000.0,
            fy: 1000.0,
            cx: 256.0,
            cy: 256.0,
        };
        let opts = RectifyOptions {
            k_rect: Some(k),
            scale: f64::NAN, // ignored
        };
        let rect = rectify(&rig(0.1), [0, 1], &opts).unwrap();
        let got = rect.k_rect();
        assert_eq!((got.fx, got.fy, got.cx, got.cy), (k.fx, k.fy, k.cx, k.cy));
    }

    #[test]
    fn rejects_bad_inputs() {
        let r = rig(0.1);
        let opts = RectifyOptions::default();
        let invalid = |e: MvgError| matches!(e, MvgError::InvalidInput { .. });
        assert!(invalid(rectify(&r, [0, 0], &opts).unwrap_err()));
        assert!(invalid(rectify(&r, [0, 2], &opts).unwrap_err()));
        for scale in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let bad = RectifyOptions {
                scale,
                ..Default::default()
            };
            assert!(invalid(rectify(&r, [0, 1], &bad).unwrap_err()), "{scale}");
        }
        let bad_k = RectifyOptions {
            k_rect: Some(FxFyCxCy {
                fx: 0.0,
                fy: 800.0,
                cx: 0.0,
                cy: 0.0,
            }),
            ..Default::default()
        };
        assert!(invalid(rectify(&r, [0, 1], &bad_k).unwrap_err()));
        let err = rectify(&rig(0.0), [0, 1], &opts).unwrap_err();
        assert!(matches!(err, MvgError::Degenerate { .. }), "{err:?}");
    }

    #[test]
    fn forward_baseline_is_defined() {
        // Camera 1 sits straight ahead of camera 0 along its optical axis.
        let rig = CalibratedRig::new(
            vec![pinhole_params(0.0), pinhole_params(0.0)],
            vec![Iso3::identity(), Iso3::translation(0.0, 0.0, -0.3)],
            Uncertainty::default(),
        )
        .unwrap();
        let rect = rectify(&rig, [0, 1], &RectifyOptions::default()).unwrap();
        let r = rect.rect_r_cam(0);
        assert!((r * r.transpose() - Mat3::identity()).abs().max() < 1e-12);
        assert!((r.determinant() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn q_agrees_with_point() {
        let rect = rectify(&rig(0.12), [0, 1], &RectifyOptions::default()).unwrap();
        let (left, d) = (Pt2::new(701.0, 410.5), 37.25);
        let h = rect.disparity_to_depth() * nalgebra::Vector4::new(left.x, left.y, d, 1.0);
        let in_rect = Vec3::new(h.x / h.w, h.y / h.w, h.z / h.w);
        let p = rect.point(&left, d).unwrap();
        let expected = rect.rect_r_cam(0).transpose() * in_rect;
        assert!((p.coords - expected).norm() < 1e-12 * expected.norm());
        for bad in [0.0, -3.0, f64::NAN] {
            assert!(rect.point(&left, bad).is_none());
        }
    }

    #[test]
    fn triangulate_rejects_another_rig() {
        let rect = rectify(&rig(0.12), [0, 1], &RectifyOptions::default()).unwrap();
        let m = [RectifiedMatch::new(Pt2::new(640.0, 480.0), 40.0)];
        let noise = PointCovariance::isotropic(0.3).unwrap();
        let err = rect.triangulate(&rig(0.2), &m, noise).unwrap_err();
        assert!(matches!(err, MvgError::InvalidInput { .. }), "{err:?}");
        assert!(rect.triangulate(&rig(0.12), &m, noise).is_ok());
    }
}
