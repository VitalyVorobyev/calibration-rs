//! A dense point cloud from a disparity map, with each point's covariance on
//! demand.
//!
//! # The measurement
//!
//! A dense match is a pixel `(x, y)` of the rectified left image, a grid
//! point and so exact, and a disparity `d` measured along the row of the
//! rectified right image. The rectified images were resampled with the
//! calibration's values, so both maps are fixed: the left pixel is the raw
//! pixel `u_L = raw_L(x, y)`, and the right measurement says only that the
//! right camera saw the point on the raw-image curve of rectified column
//! `x − d`, at `u_R = raw_R(x − d, y)` to first order along it. A row error
//! there (the vertical disparity a calibration error causes) is not
//! measured: the matcher searched along the row.
//!
//! With the cameras' parameters `θ` and poses `cam_se3_rig · Exp(ξ)` the
//! point `X` (rig frame) is the implicit function
//!
//! ```text
//! F(X; θ, ξ, d) = [ π_L(X) − u_L ;  r (π_R(X) − u_R(d)) ] = 0
//! ```
//!
//! with `π_k(X) = π(cam_se3_rig_k · Exp(ξ_k) · X; θ_k)` and `r = e₁ᵀ M⁻¹`
//! the row of the fixed right map's inverse that gives the rectified column,
//! `M = ∂raw_R/∂(rectified pixel)`. Three equations fix the three
//! coordinates, so with `A = ∂F/∂X`
//!
//! ```text
//! ∂X/∂c = −A⁻¹ ∂F/∂c        ∂X/∂d = −A⁻¹ e₃ = −(X − c_L)/d
//! ```
//!
//! (`c_L` the left camera's centre), the derivatives of `π` exact from the
//! camera's own model (dual numbers, `CameraJacobian`). The rectification
//! is not differentiated: it is how the measurement was taken, not a
//! parameter of the point. Point `i`'s covariance is
//!
//! ```text
//! Σ_i = G_i Σ_c G_iᵀ + σ_d² g_i g_iᵀ,     G_i = ∂X_i/∂c,  g_i = ∂X_i/∂d
//! ```
//!
//! and two points share the calibration, `Cov(X_i, X_j) = G_i Σ_c G_jᵀ`.
//! The disparities' errors are taken as independent.
//!
//! The cloud stores the points and computes `G_i` when a covariance is asked
//! for: at full resolution the `G_i` would take 3 × m × 8 bytes a point.

use super::StereoRectification;
use crate::dense::DisparityMap;
use crate::sensitivity::{camera_coordinates, skew};
use crate::{MvgError, Result};
use nalgebra::{DMatrix, Dyn, Matrix2x3, Matrix3, Matrix3x2, Matrix3x6, OMatrix, U3, Vector3};
use serde::Serialize;
use vision_calibration_core::{
    CalibratedRig, FxFyCxCy, Iso3, Mat3, Pt2, Pt3, Real, UncertaintyUnavailable, Vec3,
};
use vision_calibration_optim::CameraJacobian;

/// One point of a [`DenseCloud`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct DensePoint {
    /// The point in the **rig frame**, in the rig's length unit (metres for
    /// a calibration in metres).
    pub point: Pt3,
    /// The rectified left pixel `[x, y]` it was measured at, the indices of
    /// [`DisparityMap::get`].
    pub pixel: [usize; 2],
    /// Its disparity `x_left − x_right`, pixels.
    pub disparity: Real,
}

/// The points of a disparity map, with the covariance of any of them on
/// demand. Built by [`StereoRectification::dense_cloud`].
///
/// The points are in the **rig frame**, as
/// [`StereoRectification::triangulate`]'s are (the left camera's frame when
/// it is the reference camera). A point's covariance has two sources: the
/// disparity's noise, and the calibration's uncertainty, which every point
/// shares, so the points are correlated. [`point_covariance`](Self::point_covariance)
/// gives one point's, [`covariance`](Self::covariance) the joint covariance
/// of chosen points, the input of the measurements in
/// [`metrology`](crate::metrology) through
/// [`PointSet::from_dense_cloud`](crate::metrology::PointSet::from_dense_cloud).
#[derive(Debug, Clone)]
pub struct DenseCloud {
    points: Vec<DensePoint>,
    sigma_d: Real,
    /// The left camera's centre in the rig frame.
    left_centre: Vec3,
    /// What the calibration's share needs, or why there is none.
    share: std::result::Result<CalibrationShare, UncertaintyUnavailable>,
}

/// The calibration's side of the points' covariance.
#[derive(Debug, Clone)]
struct CalibrationShare {
    /// The left and the right camera.
    sides: [Side; 2],
    /// The rotation from the right camera into the rectified frame.
    rect_r_right: Mat3,
    k_rect: FxFyCxCy<Real>,
    /// `Σ_c`, over the calibration coordinates the cameras use.
    sigma_c: DMatrix<Real>,
}

/// One camera of the pair, as the sensitivity needs it.
#[derive(Debug, Clone)]
struct Side {
    jacobian: CameraJacobian,
    cam_se3_rig: Iso3,
    /// The rotation of `cam_se3_rig`.
    rotation: Matrix3<Real>,
    /// `(column of ∂π/∂θ, column of c)` of each listed camera parameter.
    theta: Vec<(usize, usize)>,
    /// The first of the six columns of the camera's `cam_se3_rig`, when
    /// listed.
    extrinsic: Option<usize>,
}

impl StereoRectification {
    /// The point cloud of a disparity map of this rectified pair, with each
    /// point's covariance on demand.
    ///
    /// Every pixel of `disparity` with a finite, positive disparity becomes
    /// a point, row by row; the others (`NaN` marks the matcher's invalid
    /// pixels) are left out, and [`DensePoint::pixel`] says where each point
    /// came from. `sigma_d` is the disparity's standard deviation, pixels,
    /// one for the whole map. The points are in the **rig frame**.
    ///
    /// The covariance is not computed here: [`DenseCloud::point_covariance`]
    /// and [`DenseCloud::covariance`] compute it for the points asked for.
    /// It includes the calibration's share, the cameras' parameters and
    /// extrinsics with their correlations from the rig's covariance;
    /// parameters the covariance does not list were held fixed by the
    /// calibration and count as exact. A rig without a covariance gives
    /// points without one
    /// ([`UncertaintyUnavailable::CalibrationWithoutCovariance`]): a σ from
    /// the disparity alone, which
    /// [`disparity_covariance`](Self::disparity_covariance) gives, would
    /// leave out the calibration's share.
    ///
    /// `rig` must be the one this rectification was built from.
    ///
    /// ```
    /// use vision_calibration_core::{
    ///     BrownConrady5, CalibratedRig, FxFyCxCy, Iso3, Uncertainty, make_pinhole_camera,
    ///     pinhole_camera_params,
    /// };
    /// use vision_mvg::dense::DisparityMap;
    /// use vision_mvg::rectification::{RectifyOptions, rectify};
    ///
    /// let k = FxFyCxCy { fx: 800.0, fy: 800.0, cx: 320.0, cy: 240.0 };
    /// let camera = pinhole_camera_params(&make_pinhole_camera(k, BrownConrady5::default()));
    /// let rig = CalibratedRig::new(
    ///     vec![camera.clone(), camera],
    ///     vec![Iso3::identity(), Iso3::translation(-0.12, 0.0, 0.0)],
    ///     Uncertainty::default(),
    /// )?;
    /// let rect = rectify(&rig, [0, 1], &RectifyOptions::default())?;
    ///
    /// // A 640 × 480 map, 48 px everywhere but one invalid pixel.
    /// let mut map = DisparityMap::filled(640, 480, 48.0);
    /// map.set(0, 0, f32::NAN);
    /// let cloud = rect.dense_cloud(&rig, &map, 0.2)?;
    /// assert_eq!(cloud.len(), 640 * 480 - 1);
    /// // Depth f B / d = 800 · 0.12 / 48 = 2 m.
    /// assert!((cloud.points()[0].point.z - 2.0).abs() < 1e-12);
    /// // This rig has no covariance, so the points have none; one with a
    /// // covariance gives `cloud.point_covariance(i)`, 3×3 in m².
    /// assert!(cloud.point_covariance(0).is_none());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// [`MvgError::InvalidInput`] when `rig` differs from the one rectified,
    /// for a `sigma_d` that is negative or not finite, for a camera model
    /// with a general sensor homography (its parameters have no derivative),
    /// and for a covariance that lists part of a camera's extrinsic (a pose
    /// is estimated whole or held).
    pub fn dense_cloud(
        &self,
        rig: &CalibratedRig,
        disparity: &DisparityMap,
        sigma_d: Real,
    ) -> Result<DenseCloud> {
        self.check_rig(rig)?;
        if !(sigma_d.is_finite() && sigma_d >= 0.0) {
            return Err(MvgError::invalid_input(format!(
                "the disparity's sigma must be finite and non-negative, got {sigma_d}"
            )));
        }
        let share = match rig.uncertainty().covariance() {
            None => Err(UncertaintyUnavailable::CalibrationWithoutCovariance),
            Some(covariance) => Ok(self.calibration_share(rig, covariance)?),
        };

        // X = rig_se3_left · rect_r_leftᵀ · P_rect, one rotation for both.
        let rig_se3_left = self.cam_se3_rig[0].inverse();
        let rotation = rig_se3_left.rotation.to_rotation_matrix().into_inner()
            * self.rect_r_cam[0].transpose();
        let left_centre = rig_se3_left.translation.vector;
        let mut points = Vec::with_capacity(disparity.valid_count());
        for y in 0..disparity.height {
            for x in 0..disparity.width {
                let d = Real::from(disparity.get(x, y));
                let Some(p) = self.rectified_point(&Pt2::new(x as Real, y as Real), d) else {
                    continue;
                };
                points.push(DensePoint {
                    point: Pt3::from(rotation * p + left_centre),
                    pixel: [x, y],
                    disparity: d,
                });
            }
        }
        Ok(DenseCloud {
            points,
            sigma_d,
            left_centre,
            share,
        })
    }

    /// The calibration coordinates of the pair's two cameras and their
    /// covariance.
    fn calibration_share(
        &self,
        rig: &CalibratedRig,
        covariance: &vision_calibration_core::ParameterCovariance,
    ) -> Result<CalibrationShare> {
        let mut index: Vec<usize> = Vec::new();
        let mut side = |s: usize| -> Result<Side> {
            let k = self.cameras[s];
            let jacobian = CameraJacobian::new(&rig.cameras()[k])
                .map_err(|e| MvgError::invalid_input(format!("camera {k}: {e}")))?;
            let coordinates = camera_coordinates(rig, covariance, k, &jacobian)?;
            let mut column = |at: usize| {
                index.push(at);
                index.len() - 1
            };
            let theta = coordinates
                .theta
                .into_iter()
                .map(|(j, _, at)| (j, column(at)))
                .collect();
            let extrinsic = coordinates.extrinsic.map(|pose| {
                let columns: Vec<usize> = pose.into_iter().map(|(_, at)| column(at)).collect();
                columns[0]
            });
            let cam_se3_rig = self.cam_se3_rig[s];
            Ok(Side {
                jacobian,
                cam_se3_rig,
                rotation: cam_se3_rig.rotation.to_rotation_matrix().into_inner(),
                theta,
                extrinsic,
            })
        };
        let sides = [side(0)?, side(1)?];
        let m = index.len();
        Ok(CalibrationShare {
            sides,
            rect_r_right: self.rect_r_cam[1],
            k_rect: self.k_rect,
            sigma_c: DMatrix::from_fn(m, m, |a, b| covariance.covariance[index[a]][index[b]]),
        })
    }
}

impl CalibrationShare {
    /// `G = ∂X/∂c` (3 × m) at the rig-frame point `x`, and `−A⁻¹ e₃`;
    /// `None` when a camera's projection has no usable derivative there or
    /// `A` is singular.
    fn jacobian(&self, x: &Vec3) -> Option<(OMatrix<Real, U3, Dyn>, Vector3<Real>)> {
        let [left, right] = &self.sides;
        let at = |side: &Side| (side.cam_se3_rig * Pt3::from(*x)).coords;
        let (p_left, p_right) = (at(left), at(right));
        let jl = left.jacobian.project(&p_left)?;
        let jr = right.jacobian.project(&p_right)?;

        // r = e₁ᵀ M⁻¹, M = ∂raw_R/∂(rectified pixel) = v_z D_R R₁ᵀ E at the
        // rectified ray v = R₁ p_R (π is homogeneous of degree 0, so its
        // derivative at p_R/v_z is v_z times that at p_R).
        let k = &self.k_rect;
        let e = Matrix3x2::new(1.0 / k.fx, 0.0, 0.0, 1.0 / k.fy, 0.0, 0.0);
        let v_z = (self.rect_r_right * p_right).z;
        let m = (jr.d_point * self.rect_r_right.transpose() * e) * v_z;
        let m_inv = m.try_inverse()?;
        let r = m_inv.row(0).into_owned();

        // A = [D_L R_L; r D_R R_R].
        let top: Matrix2x3<Real> = jl.d_point * left.rotation;
        let bottom = r * jr.d_point * right.rotation;
        let a = Matrix3::from_rows(&[top.row(0).into_owned(), top.row(1).into_owned(), bottom]);
        let a_inv = a
            .try_inverse()
            .filter(|m| m.iter().all(|v| v.is_finite()))?;

        let mut g = OMatrix::<Real, U3, Dyn>::zeros(self.sigma_c.nrows());
        // ∂F/∂θ_L = [∂π_L/∂θ_L; 0], ∂F/∂θ_R = [0; 0; r ∂π_R/∂θ_R].
        for &(j, c) in &left.theta {
            let f = Vector3::new(jl.d_params[(0, j)], jl.d_params[(1, j)], 0.0);
            g.set_column(c, &(-(a_inv * f)));
        }
        for &(j, c) in &right.theta {
            let f = Vector3::new(0.0, 0.0, r.dot(&jr.d_params.column(j).transpose()));
            g.set_column(c, &(-(a_inv * f)));
        }
        // cam_se3_rig · Exp(ω, v) moves the camera-frame point by
        // R (−[X]ₓ ω + v).
        let mut d_xi = Matrix3x6::<Real>::zeros();
        d_xi.fixed_view_mut::<3, 3>(0, 0).copy_from(&(-skew(x)));
        d_xi.fixed_view_mut::<3, 3>(0, 3)
            .copy_from(&Matrix3::identity());
        if let Some(first) = left.extrinsic {
            let f = jl.d_point * (left.rotation * d_xi);
            for c in 0..6 {
                let column = Vector3::new(f[(0, c)], f[(1, c)], 0.0);
                g.set_column(first + c, &(-(a_inv * column)));
            }
        }
        if let Some(first) = right.extrinsic {
            let f = r * jr.d_point * (right.rotation * d_xi);
            for c in 0..6 {
                let column = Vector3::new(0.0, 0.0, f[c]);
                g.set_column(first + c, &(-(a_inv * column)));
            }
        }
        Some((g, -a_inv.column(2).into_owned()))
    }
}

impl DenseCloud {
    /// The points, row by row of the disparity map.
    pub fn points(&self) -> &[DensePoint] {
        &self.points
    }

    /// The number of points.
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Whether the cloud has no point.
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// The disparity's standard deviation the cloud was built with, pixels.
    pub fn sigma_d(&self) -> Real {
        self.sigma_d
    }

    /// Why the points have no covariance; `None` when they have one.
    pub fn covariance_unavailable(&self) -> Option<&UncertaintyUnavailable> {
        self.share.as_ref().err()
    }

    /// Point `i`'s 3×3 covariance, m²: the disparity's share and the
    /// calibration's. `None` when the points have no covariance
    /// ([`covariance_unavailable`](Self::covariance_unavailable)), `i` is
    /// out of range or a camera's projection has no usable derivative at
    /// the point.
    ///
    /// For several points that one measurement combines, take their joint
    /// [`covariance`](Self::covariance): the calibration correlates them.
    pub fn point_covariance(&self, i: usize) -> Option<Mat3> {
        let joint = self.covariance(&[i]).ok()?;
        Some(joint.fixed_view::<3, 3>(0, 0).into_owned())
    }

    /// The joint covariance of the points `indices`, m², 3k × 3k for k
    /// indices, ordered `x, y, z` per point in the order given: the
    /// calibration's share, which correlates every pair of points, plus each
    /// point's disparity share.
    ///
    /// The cost is `O(k m)` derivatives and an `O(k² m)` product for `m`
    /// calibration coordinates (about 30 for a stereo pair); choose the
    /// points a measurement needs, not the whole cloud.
    ///
    /// # Errors
    ///
    /// [`MvgError::InvalidInput`] when the points have no covariance (the
    /// message gives the reason) or an index is out of range;
    /// [`MvgError::Numerical`] when a camera's projection has no usable
    /// derivative at a point. The message names the point.
    pub fn covariance(&self, indices: &[usize]) -> Result<DMatrix<Real>> {
        let share = self.share.as_ref().map_err(|reason| {
            MvgError::invalid_input(format!(
                "the points have no covariance: the uncertainty is unavailable ({reason:?})"
            ))
        })?;
        let (k, m) = (indices.len(), share.sigma_c.nrows());
        let mut g = DMatrix::<Real>::zeros(3 * k, m);
        let mut g_d = Vec::with_capacity(k);
        for (row, &i) in indices.iter().enumerate() {
            let p = self.points.get(i).ok_or_else(|| {
                MvgError::invalid_input(format!(
                    "point index {i} is out of range for {} points",
                    self.len()
                ))
            })?;
            let (g_i, _) = share.jacobian(&p.point.coords).ok_or_else(|| {
                MvgError::numerical(format!(
                    "point {i}: a camera's projection has no usable derivative there"
                ))
            })?;
            g.view_mut((3 * row, 0), (3, m)).copy_from(&g_i);
            // ∂X/∂d = −(X − c_L)/d: along the left viewing ray.
            g_d.push(-(p.point.coords - self.left_centre) / p.disparity);
        }
        let mut sigma = &g * &share.sigma_c * g.transpose();
        let var_d = self.sigma_d * self.sigma_d;
        for (row, g_i) in g_d.iter().enumerate() {
            let mut block = sigma.fixed_view_mut::<3, 3>(3 * row, 3 * row);
            block += g_i * g_i.transpose() * var_d;
        }
        // Symmetrise away the rounding of the products.
        Ok((&sigma + sigma.transpose()) * 0.5)
    }
}

#[cfg(test)]
mod tests;
