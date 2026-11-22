//! The plane of sharp focus and the depth of field, from the blur each
//! feature carries.
//!
//! A lens images one plane sharply, the plane of sharp focus. A point off
//! it is imaged as a spot whose size grows with its distance from the plane
//! in inverse depth. [`fit_focus`] fits that law to the per-point blur of a
//! calibrated planar dataset (`CorrespondenceView::blurs`), with the camera
//! and the target poses held at the calibration, and returns the plane, the
//! depth of field for a stated blur limit, and how well the law explains
//! the blur.
//!
//! # Model
//!
//! Let `n` be the sensor normal in the camera frame (the optical axis for
//! an untilted sensor, the axis tilted with the sensor for a Scheimpflug
//! one) and `X` a target point in the camera frame, at depth `z = n·X`
//! along it. With the plane of sharp focus `π·X = 1` (`π` in 1/m, the
//! camera frame), the point's defocus is
//!
//! ```text
//! d = 1/z − π·X/z = (1 − π·X)/z      (1/m)
//! ```
//!
//! and the blur the image shows, the width `σ = √(tr Σ / 2)` of the
//! measured point-spread function, is predicted as
//!
//! ```text
//! σ_pred = √(σ0² + 1/12 + (κ·s·d)²) + δ_view
//! ```
//!
//! - `κ` (px·m) is the blur per unit defocus. For a geometric spot from a
//!   uniformly lit circular entrance pupil of diameter `D`, read as the
//!   per-axis σ of a filled disc, `κ = D·f/4` with `f` the focal length in
//!   pixels.
//! - `s` is the camera's local image scale at the point relative to its
//!   distortion-free scale: 1 without distortion, and the change of
//!   magnification distortion brings elsewhere. It is computed from the
//!   calibrated camera at the point's own projection.
//! - `σ0²` (px²) is the optical blur at focus; `1/12` is the pixel's area
//!   integration, which the measured blur contains, so the prediction adds
//!   it rather than the observation dropping it.
//! - `δ_view` (px) is one blur offset per view, the offsets summing to
//!   zero. A detector's per-point blur covariance holds for differences
//!   within one image; every point of an image can share an offset it does
//!   not contain, which this term takes. A level common to every view is
//!   not separable from the in-focus blur except through the shape of the
//!   blur near focus, so it is left to `σ0²`.
//!
//! The model is the thin-lens defocus law in object space, which holds for
//! a thick lens with `D` its entrance pupil and `π` the true plane of sharp
//! focus. It is fitted by weighted least squares in `σ`, each point weighted
//! by the inverse variance of its `σ` from the detector's covariance (all
//! points equally when any point has none), with the noise scale estimated
//! from the residuals.
//!
//! # What is determined
//!
//! Within one view the defocus is affine in the image, so per view the blur
//! is `σ² ≈ σ0² + (αu + βv + γ)²`. The views' different poses separate `κ`
//! from the plane's tilt. The plane's position along the axis needs the
//! fold where a view crosses the plane: far from focus, `σ ≈ |αu + βv + γ|
//! + δ`, and the view's offset takes `γ`. So at least one view must hold
//! points on both sides of the plane ([`FocusDiagnostic::NotCrossed`]).
//! Views all at one depth facing the camera cannot separate `κ` from the
//! plane's tilt ([`FocusDiagnostic::Undetermined`]).
//!
//! # What it does not give
//!
//! The lens tilt that the plane of sharp focus implies needs two metric
//! data this fit does not have: the projection centre's distance to the
//! sensor (the focal length in pixels times the pixel pitch) and the lens's
//! pupil magnification. The plane, its line in the image and the depth of
//! field need neither. The calibration's own uncertainty (camera and poses)
//! is not propagated: the covariance is conditional on them.

use nalgebra::{DMatrix, DVector, Matrix3, SMatrix, Vector3};
use serde::{Deserialize, Serialize};

use vision_calibration_core::{
    AnySensor, CameraModel, CameraParams, CameraProject, Iso3, PlanarDataset, ProjectionKind, Real,
};

use crate::Error;

/// The fewest blur points a view needs to enter the fit: its offset takes
/// one, so a view with fewer carries next to nothing.
const MIN_VIEW_POINTS: usize = 3;

/// The pixel's area integration, px²: the variance of a uniform unit
/// square along each axis.
const PIXEL_VARIANCE: Real = 1.0 / 12.0;

/// Central-difference step of the local image scale, in normalized
/// coordinates (about a millipixel at f = 1000 px): small against any
/// distortion's curvature, large against rounding.
const SCALE_STEP: Real = 1e-6;

/// The reciprocal condition number (after scaling to a unit diagonal) below
/// which the normal equations count as singular.
const SINGULAR_RCOND: Real = 1e-10;

/// Levenberg–Marquardt iteration budget; the fit has at most a few hundred
/// parameters and converges in tens of iterations from its seed.
const MAX_ITERATIONS: usize = 200;

/// Options of [`fit_focus`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[serde(default, deny_unknown_fields)]
#[non_exhaustive]
pub struct FocusOptions {
    /// The largest acceptable blur `σ` (px, the measured width
    /// `√(tr Σ / 2)`, pixel integration included) for the depth of field;
    /// `None` (the default) computes none. Must be finite and positive. A
    /// limit at or below the in-focus blur `√(σ0² + 1/12)` has no depth of
    /// field.
    pub blur_limit_px: Option<Real>,
}

impl FocusOptions {
    /// Options with the depth of field for `blur_limit_px` (`None`: none).
    pub fn new(blur_limit_px: Option<Real>) -> Self {
        Self { blur_limit_px }
    }
}

/// The result of [`fit_focus`]: the fitted focus model, or why there is
/// none.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[serde(tag = "status", rename_all = "snake_case")]
#[non_exhaustive]
// Returned once per call and matched by value; boxing the fit would only
// add a dereference to every use.
#[allow(clippy::large_enum_variant)]
pub enum FocusDiagnostic {
    /// The model was fitted and determines the plane of sharp focus.
    Fitted(FocusFit),
    /// No point of the dataset carries a blur.
    NoBlur,
    /// Too few blur points for the parameters: the five of the model plus
    /// one offset per view with at least three blur points but the first
    /// (the offsets sum to zero), and one more.
    TooFewPoints {
        /// Blur points needed.
        need: usize,
        /// Blur points in views with at least three.
        got: usize,
    },
    /// No view has points on both sides of the fitted plane, so the plane's
    /// position along the axis is not determined: each view's blur offset
    /// absorbs it. Views that cross the plane of sharp focus (a target
    /// tilted through it) determine it.
    NotCrossed,
    /// The views do not determine the model: its normal equations are
    /// singular (for one, every view at one depth facing the camera, where
    /// the blur scale and the plane's tilt trade), or the fit does not
    /// converge.
    Undetermined,
}

/// The focus model fitted to a calibrated dataset's blur.
///
/// Planes are written `p·X = 1` with `p` in 1/m and `X` in the camera
/// frame, metres; the in-focus distance along a ray `r` is `1 / (π·r)` in
/// units of `r`'s length.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[non_exhaustive]
pub struct FocusFit {
    /// The plane of sharp focus `π`, 1/m, camera frame: `π·X = 1`. On the
    /// optical axis the camera is focused at `1 / π_z` metres.
    pub plane_inv_m: [Real; 3],
    /// The blur per unit defocus `κ`, px·m (> 0).
    pub blur_scale_px_m: Real,
    /// The optical blur variance at focus `σ0²`, px², pixel integration
    /// excluded. Estimated freely: noise can make it slightly negative
    /// when the optics are sharper than a pixel.
    pub in_focus_variance_px2: Real,
    /// Covariance of `(π_x, π_y, π_z, κ, σ0²)`, in their units, scaled by
    /// [`noise_scale`](Self::noise_scale)`²` and marginal over the view
    /// offsets.
    pub covariance: [[Real; 5]; 5],
    /// The unit sensor normal `n` in the camera frame, from the camera:
    /// `(0, 0, 1)` for an untilted sensor.
    pub sensor_normal: [Real; 3],
    /// Where the plane of sharp focus meets infinity in the image; `None`
    /// when the plane is parallel to the sensor.
    pub image_line: Option<FocusImageLine>,
    /// The depth of field for [`FocusOptions::blur_limit_px`]; `None` when
    /// no limit was given or the limit is not above the in-focus blur.
    pub depth_of_field: Option<DepthOfField>,
    /// Blur points the fit used (in views with at least three).
    pub points: usize,
    /// Residual degrees of freedom: `points` minus the five model
    /// parameters and one offset per fitted view but one.
    pub residual_dof: usize,
    /// Whether the points were weighted by their stated blur covariance
    /// (true) or equally (false, when any point's covariance gives its `σ`
    /// no variance).
    pub stated_weights: bool,
    /// `√(χ² / residual_dof)` of the weighted residuals: about 1 when the
    /// model explains the blur to the detector's stated precision, larger
    /// when the model misfits (field curvature, aberrations) or the stated
    /// covariance is short; in px with equal weights. The covariance is
    /// scaled by its square.
    pub noise_scale: Real,
    /// RMS of the blur residuals `σ − σ_pred`, px.
    pub residual_rms_px: Real,
    /// One entry per input view, in input order.
    pub views: Vec<FocusViewStats>,
}

/// The image line where the plane of sharp focus is at infinity.
///
/// The pixels `(u, v)` with `u·cos θ + v·sin θ = ρ`, in the camera's
/// distortion-free projection (distortion bends the line by its own
/// amount). On the side `u·cos θ + v·sin θ < ρ` the in-focus distance is
/// finite and grows towards the line; beyond it the plane lies behind the
/// camera. `(cos θ, sin θ)` is the image direction in which the focus
/// recedes, the tilt direction of the plane of sharp focus.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[non_exhaustive]
pub struct FocusImageLine {
    /// `θ`, radians, in `(−π, π]`.
    pub angle_rad: Real,
    /// `ρ`, pixels.
    pub offset_px: Real,
    /// Covariance of `(θ, ρ)`, from the plane's.
    pub covariance: [[Real; 2]; 2],
}

/// The depth of field for a blur limit `σ_max`: the wedge between a near
/// and a far plane.
///
/// A point is within it when its defocus satisfies `|d| ≤ ε`,
/// `ε = √(σ_max² − σ0² − 1/12) / κ`. The limits are the planes
/// `(π ± ε·n)·X = 1`, which meet the plane of sharp focus on the line where
/// it crosses the plane through the projection centre parallel to the
/// sensor. Distortion that magnifies by a factor `s` tightens the limit
/// there to `ε/s`; the planes are those of the distortion-free scale.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[non_exhaustive]
pub struct DepthOfField {
    /// The blur limit `σ_max`, px.
    pub blur_limit_px: Real,
    /// The defocus limit `ε`, 1/m.
    pub defocus_limit_inv_m: Real,
    /// The standard deviation of `ε`, 1/m.
    pub defocus_limit_sigma_inv_m: Real,
    /// The near limit `π + ε·n`, 1/m, camera frame.
    pub near_plane_inv_m: [Real; 3],
    /// Covariance of the near plane's coefficients, 1/m².
    pub near_plane_covariance: [[Real; 3]; 3],
    /// The far limit `π − ε·n`, 1/m, camera frame. Where its `p·X` is
    /// not positive along a ray, the depth of field extends to infinity.
    pub far_plane_inv_m: [Real; 3],
    /// Covariance of the far plane's coefficients, 1/m².
    pub far_plane_covariance: [[Real; 3]; 3],
}

/// The blur fit in one view.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[non_exhaustive]
pub struct FocusViewStats {
    /// Points of the view that carry a blur.
    pub points: usize,
    /// Whether the fit used the view (at least three blur points). An
    /// unused view's other fields are 0 and `false`.
    pub used: bool,
    /// The view's blur offset `δ`, px, relative to the mean over the
    /// fitted views (the offsets sum to zero).
    pub offset_px: Real,
    /// The standard deviation of `δ`, px.
    pub offset_sigma_px: Real,
    /// RMS of the view's blur residuals, px.
    pub rms_px: Real,
    /// The largest absolute blur residual of the view, px.
    pub max_abs_px: Real,
    /// Whether the view has points on both sides of the plane of sharp
    /// focus.
    pub crosses_focus: bool,
}

/// Fit the focus model to the blur of a calibrated planar dataset (see the
/// module documentation).
///
/// `camera` and `cam_se3_target` (one pose per view of `dataset`) are the
/// calibration's and are held fixed; only the points' 3D positions and
/// blur enter, never the observed pixels. Points without a blur are
/// skipped, and views with fewer than three blur points are not fitted.
/// The result is deterministic.
///
/// # Errors
///
/// [`Error::InvalidInput`] when the pose count differs from the view count,
/// the camera's projection is not a pinhole (the defocus law is the
/// perspective one), its sensor is not one this fit knows, the sensor faces
/// away from the scene, a blur point does not project, or the blur limit is
/// not finite and positive. A dataset the model cannot be fitted to is not
/// an error: it is a [`FocusDiagnostic`] other than `Fitted`.
///
/// # Example
///
/// ```no_run
/// use vision_calibration_pipeline::analysis::{FocusDiagnostic, FocusOptions, fit_focus};
/// use vision_calibration_pipeline::planar_intrinsics::{
///     PlanarIntrinsicsExport, PlanarIntrinsicsInput,
/// };
/// # fn main() -> Result<(), vision_calibration_pipeline::Error> {
/// # let input: PlanarIntrinsicsInput = unimplemented!("views with blur");
/// # let export: PlanarIntrinsicsExport = unimplemented!("their calibration");
/// let options = FocusOptions::new(Some(1.5)); // depth of field at σ ≤ 1.5 px
/// match fit_focus(&input, &export.camera, &export.cam_se3_target, &options)? {
///     FocusDiagnostic::Fitted(fit) => {
///         println!("focused at {:.3} m on the axis", 1.0 / fit.plane_inv_m[2]);
///     }
///     other => println!("no focus model: {other:?}"),
/// }
/// # Ok(())
/// # }
/// ```
pub fn fit_focus(
    dataset: &PlanarDataset,
    camera: &CameraParams,
    cam_se3_target: &[Iso3],
    opts: &FocusOptions,
) -> Result<FocusDiagnostic, Error> {
    if cam_se3_target.len() != dataset.views.len() {
        return Err(Error::invalid_input(format!(
            "{} poses for {} views",
            cam_se3_target.len(),
            dataset.views.len()
        )));
    }
    if let Some(limit) = opts.blur_limit_px
        && !(limit.is_finite() && limit > 0.0)
    {
        return Err(Error::invalid_input(format!(
            "the blur limit must be finite and positive, got {limit}"
        )));
    }
    if camera.projection.kind() != ProjectionKind::Pinhole {
        return Err(Error::invalid_input(
            "the focus model needs a pinhole projection (the defocus law is the perspective one)",
        ));
    }
    let model = camera.build()?;
    let geometry = SensorGeometry::new(camera, &model)?;

    let mut obs = Vec::new();
    let mut view_points = vec![0usize; dataset.views.len()];
    for (v, (view, pose)) in dataset.views.iter().zip(cam_se3_target).enumerate() {
        let Some(blurs) = view.obs.blurs() else {
            continue;
        };
        for (i, (p3, blur)) in view.obs.points_3d.iter().zip(blurs).enumerate() {
            let Some(blur) = blur else {
                continue;
            };
            let x = (pose * p3).coords;
            let point = geometry.point(&model, &x).ok_or_else(|| {
                Error::invalid_input(format!(
                    "point {i} of view {v} does not project at its calibrated pose"
                ))
            })?;
            let sigma = blur.sigma_px();
            let c = blur.cov();
            // σ² = (xx + yy)/2, so var(σ²) = (c_xx,xx + 2·c_xx,yy + c_yy,yy)/4
            // and var(σ) = var(σ²)/(4σ²).
            let var_sq = (c[0][0] + 2.0 * c[0][2] + c[2][2]) / 4.0;
            obs.push(Obs {
                view: v,
                r: point.r,
                scale: point.scale,
                sigma,
                var: var_sq / (4.0 * sigma * sigma),
            });
            view_points[v] += 1;
        }
    }
    if obs.is_empty() {
        return Ok(FocusDiagnostic::NoBlur);
    }

    // Views with enough points get an offset column; the rest are dropped.
    let mut column = vec![None; dataset.views.len()];
    let mut fitted_views = 0;
    for (v, &n) in view_points.iter().enumerate() {
        if n >= MIN_VIEW_POINTS {
            column[v] = Some(fitted_views);
            fitted_views += 1;
        }
    }
    obs.retain(|o| column[o.view].is_some());
    for o in &mut obs {
        // Checked by the retain above.
        o.view = column[o.view].unwrap_or(0);
    }
    // One offset per fitted view, their sum zero: a blur level common to
    // every view is the in-focus blur's.
    let params = MODEL_PARAMS + fitted_views.saturating_sub(1);
    if obs.len() <= params {
        return Ok(FocusDiagnostic::TooFewPoints {
            need: params + 1,
            got: obs.len(),
        });
    }
    let stated_weights = obs.iter().all(|o| o.var.is_finite() && o.var > 0.0);
    let weights: Vec<Real> = obs
        .iter()
        .map(|o| if stated_weights { 1.0 / o.var } else { 1.0 })
        .collect();

    let Some(seed) = seed(&obs) else {
        return Ok(FocusDiagnostic::Undetermined);
    };
    let mut theta = vec![0.0; params];
    theta[..MODEL_PARAMS].copy_from_slice(&seed);
    let Some(theta) = levenberg_marquardt(&obs, &weights, theta) else {
        return Ok(FocusDiagnostic::Undetermined);
    };
    let Some(solution) = Solution::new(&obs, &weights, theta) else {
        return Ok(FocusDiagnostic::Undetermined);
    };
    let fit = solution.into_fit(
        &geometry,
        &obs,
        &view_points,
        &column,
        stated_weights,
        opts.blur_limit_px,
    );
    if fit.views.iter().any(|v| v.crosses_focus) {
        Ok(FocusDiagnostic::Fitted(fit))
    } else {
        Ok(FocusDiagnostic::NotCrossed)
    }
}

/// `β0..β3` (the defocus coefficients `κ·(1, −π')` in the sensor frame)
/// and `σ0²`; the view offsets follow.
const MODEL_PARAMS: usize = 5;

/// One blur point.
struct Obs {
    /// The view's offset column (after the retain: its index among the
    /// fitted views).
    view: usize,
    /// Regressors `(1/z, x/z, y/z, 1)` in the sensor frame.
    r: [Real; 4],
    /// The local image scale relative to the distortion-free one.
    scale: Real,
    /// The measured blur `σ`, px.
    sigma: Real,
    /// The stated variance of `σ`, px².
    var: Real,
}

/// The sensor's frame and the camera's distortion-free scale.
struct SensorGeometry {
    /// Rows: the sensor frame's axes in the camera frame, the third the
    /// sensor normal `n`.
    rotation: Matrix3<Real>,
    /// The distortion-free pixel map of camera-frame directions, `K·H`.
    pixel_map: Matrix3<Real>,
    /// The distortion-free image scale, px per unit of sensor-frame
    /// normalized coordinate (`√(‖J‖²_F / 2)`).
    reference_scale: Real,
}

/// A blur point's regressors and relative image scale.
struct PointGeometry {
    r: [Real; 4],
    scale: Real,
}

impl SensorGeometry {
    fn new(camera: &CameraParams, model: &CameraModel) -> Result<Self, Error> {
        let h = match &model.sensor {
            AnySensor::Identity(_) => Matrix3::identity(),
            AnySensor::Homography(sensor) => sensor.h,
            _ => {
                return Err(Error::invalid_input(
                    "the focus model does not know this camera's sensor",
                ));
            }
        };
        // K·H = K'·R with K' upper triangular: H's third row is the sensor
        // normal, up to scale.
        let h2 = Vector3::new(h[(2, 0)], h[(2, 1)], h[(2, 2)]);
        let scale = h2.norm();
        if !(scale.is_finite() && scale > 0.0) || h2.z <= 0.0 {
            return Err(Error::invalid_input(
                "the camera's sensor plane does not face the scene",
            ));
        }
        let n = h2 / scale;
        // Any rotation whose third row is n: the fit's scalars are
        // invariant to the roll about n.
        let u1 = (Vector3::x() - n * n.x).normalize();
        let u2 = n.cross(&u1);
        let rotation = Matrix3::from_rows(&[u1.transpose(), u2.transpose(), n.transpose()]);
        let k = &camera.intrinsics;
        let kmat = Matrix3::new(k.fx, 0.0, k.cx, 0.0, k.fy, k.cy, 0.0, 0.0, 1.0);
        let pixel_map = kmat * h;
        // In the sensor frame the map is K·H·Rᵀ, whose third row is
        // ‖H₂‖·(0, 0, 1): its upper-left block over ‖H₂‖ is the Jacobian.
        let jac = pixel_map * rotation.transpose();
        let reference_scale = ((jac[(0, 0)].powi(2)
            + jac[(0, 1)].powi(2)
            + jac[(1, 0)].powi(2)
            + jac[(1, 1)].powi(2))
            / 2.0)
            .sqrt()
            / jac[(2, 2)];
        Ok(Self {
            rotation,
            pixel_map,
            reference_scale,
        })
    }

    fn normal(&self) -> Vector3<Real> {
        self.rotation.row(2).transpose()
    }

    /// The regressors and relative image scale of the camera-frame point
    /// `x`; `None` when it is not in front of the sensor or does not
    /// project.
    fn point(&self, model: &impl CameraProject, x: &Vector3<Real>) -> Option<PointGeometry> {
        let xs = self.rotation * x;
        if !(xs.z.is_finite() && xs.z > 0.0) {
            return None;
        }
        let (a, b) = (xs.x / xs.z, xs.y / xs.z);
        let project = |a: Real, b: Real| {
            model
                .project_camera_point(&(self.rotation.transpose() * Vector3::new(a, b, 1.0)))
                .map(|p| p.coords)
                .filter(|p| p.x.is_finite() && p.y.is_finite())
        };
        let h = SCALE_STEP;
        let du = (project(a + h, b)? - project(a - h, b)?) / (2.0 * h);
        let dv = (project(a, b + h)? - project(a, b - h)?) / (2.0 * h);
        let local = ((du.norm_squared() + dv.norm_squared()) / 2.0).sqrt();
        Some(PointGeometry {
            r: [1.0 / xs.z, a, b, 1.0],
            scale: local / self.reference_scale,
        })
    }
}

/// The defocus term `g = β·r` (signed, px).
fn defocus(beta: &[Real], r: &[Real; 4]) -> Real {
    beta[0] * r[0] + beta[1] * r[1] + beta[2] * r[2] + beta[3] * r[3]
}

/// The predicted blur of `o` at `theta` without its view offset, and the
/// root's argument.
fn optical_blur(theta: &[Real], o: &Obs) -> (Real, Real) {
    let g = o.scale * defocus(&theta[..4], &o.r);
    let m2 = (theta[4] + PIXEL_VARIANCE + g * g).max(1e-12);
    (m2.sqrt(), g)
}

/// The blur offset of fitted view `view`: a parameter for every view but
/// the last, whose offset is minus their sum (the offsets have zero mean).
fn offset(theta: &[Real], view: usize) -> Real {
    if MODEL_PARAMS + view < theta.len() {
        theta[MODEL_PARAMS + view]
    } else {
        -theta[MODEL_PARAMS..].iter().sum::<Real>()
    }
}

/// The variance of [`offset`] from the parameters' covariance `cov`.
fn offset_variance(cov: &DMatrix<Real>, view: usize) -> Real {
    let p = cov.nrows();
    if MODEL_PARAMS + view < p {
        cov[(MODEL_PARAMS + view, MODEL_PARAMS + view)]
    } else {
        cov.view(
            (MODEL_PARAMS, MODEL_PARAMS),
            (p - MODEL_PARAMS, p - MODEL_PARAMS),
        )
        .sum()
    }
}

/// The residual `σ − σ_pred` of `o` at `theta`.
fn residual(theta: &[Real], o: &Obs) -> Real {
    o.sigma - optical_blur(theta, o).0 - offset(theta, o.view)
}

fn cost(obs: &[Obs], weights: &[Real], theta: &[Real]) -> Real {
    obs.iter()
        .zip(weights)
        .map(|(o, w)| w * residual(theta, o).powi(2))
        .sum()
}

/// `JᵀWJ` and `JᵀWe` of the residuals `e = σ − σ_pred`, with `J` the
/// Jacobian of `σ_pred`.
fn normal_equations(
    obs: &[Obs],
    weights: &[Real],
    theta: &[Real],
) -> (DMatrix<Real>, DVector<Real>) {
    let p = theta.len();
    let mut jtj = DMatrix::zeros(p, p);
    let mut jte = DVector::zeros(p);
    // The nonzero entries of one Jacobian row: the model's five and the
    // view's offset, or every offset for the last view.
    let mut row: Vec<(usize, Real)> = Vec::with_capacity(p);
    for (o, &w) in obs.iter().zip(weights) {
        let (m, g) = optical_blur(theta, o);
        let e = o.sigma - m - offset(theta, o.view);
        row.clear();
        row.extend((0..4).map(|k| (k, o.scale * g * o.r[k] / m)));
        row.push((4, 0.5 / m));
        if MODEL_PARAMS + o.view < p {
            row.push((MODEL_PARAMS + o.view, 1.0));
        } else {
            row.extend((MODEL_PARAMS..p).map(|k| (k, -1.0)));
        }
        for &(a, ra) in &row {
            jte[a] += w * ra * e;
            for &(b, rb) in &row {
                jtj[(a, b)] += w * ra * rb;
            }
        }
    }
    (jtj, jte)
}

/// The seed of `β` and `σ0²`: the rank-one part of the free quadratic of
/// `σ² − 1/12` in the regressors (scaled to unit RMS), `σ0²` the rest of its
/// constant. Offsets and the image scale are left out.
fn seed(obs: &[Obs]) -> Option<[Real; MODEL_PARAMS]> {
    let quad = |r: &[Real; 4]| {
        let mut q = [0.0; 10];
        let mut k = 0;
        for i in 0..4 {
            for j in i..4 {
                q[k] = r[i] * r[j];
                k += 1;
            }
        }
        q
    };
    let mut ata = SMatrix::<Real, 10, 10>::zeros();
    let mut atb = SMatrix::<Real, 10, 1>::zeros();
    for o in obs {
        let q = SMatrix::<Real, 10, 1>::from_row_slice(&quad(&o.r));
        ata += q * q.transpose();
        atb += q * (o.sigma * o.sigma - PIXEL_VARIANCE);
    }
    let c = ata.cholesky()?.solve(&atb);
    let mut m = SMatrix::<Real, 4, 4>::zeros();
    let mut k = 0;
    for i in 0..4 {
        for j in i..4 {
            let v = if i == j { c[k] } else { c[k] / 2.0 };
            m[(i, j)] = v;
            m[(j, i)] = v;
            k += 1;
        }
    }
    let n = obs.len() as Real;
    let s: [Real; 4] = std::array::from_fn(|i| {
        (obs.iter().map(|o| o.r[i] * o.r[i]).sum::<Real>() / n)
            .sqrt()
            .max(1e-12)
    });
    let ms = SMatrix::<Real, 4, 4>::from_fn(|i, j| m[(i, j)] * s[i] * s[j]);
    let eigen = ms.symmetric_eigen();
    let (top, lambda) = eigen
        .eigenvalues
        .iter()
        .copied()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(&b.1))?;
    if !(lambda.is_finite() && lambda > 0.0) {
        return None;
    }
    let v = eigen.eigenvectors.column(top);
    let mut beta: [Real; 4] = std::array::from_fn(|i| lambda.sqrt() * v[i] / s[i]);
    if beta[0] < 0.0 {
        beta.iter_mut().for_each(|b| *b = -*b);
    }
    let s0sq = (m[(3, 3)] - beta[3] * beta[3]).max(0.04);
    Some([beta[0], beta[1], beta[2], beta[3], s0sq])
}

/// Levenberg–Marquardt (Marquardt's diagonal damping) on the weighted
/// residuals from `theta`; `None` when it does not converge in
/// [`MAX_ITERATIONS`] or a step cannot be solved at any damping.
fn levenberg_marquardt(obs: &[Obs], weights: &[Real], mut theta: Vec<Real>) -> Option<Vec<Real>> {
    let p = theta.len();
    let mut lambda = 1e-3;
    let mut current = cost(obs, weights, &theta);
    for _ in 0..MAX_ITERATIONS {
        let (jtj, jte) = normal_equations(obs, weights, &theta);
        let mut accepted = false;
        for _ in 0..30 {
            let mut a = jtj.clone();
            for i in 0..p {
                a[(i, i)] += lambda * jtj[(i, i)].max(1e-12);
            }
            let Some(chol) = a.cholesky() else {
                lambda *= 10.0;
                continue;
            };
            let step = chol.solve(&jte);
            let candidate: Vec<Real> = theta.iter().zip(step.iter()).map(|(t, s)| t + s).collect();
            let next = cost(obs, weights, &candidate);
            if next.is_finite() && next <= current {
                let decrease = current - next;
                theta = candidate;
                lambda = (lambda * 0.3).max(1e-12);
                let small_step =
                    step.norm() <= 1e-12 * (1.0 + DVector::from_column_slice(&theta).norm());
                if decrease <= 1e-14 * current.max(1e-300) || small_step {
                    return Some(theta);
                }
                current = next;
                accepted = true;
                break;
            }
            lambda *= 10.0;
        }
        if !accepted {
            // No damping lowers the cost: at a minimum to rounding.
            return Some(theta);
        }
    }
    None
}

/// The converged fit with its covariance.
struct Solution {
    /// `β0..β3`, `σ0²`, the offsets; `β0 > 0`.
    theta: Vec<Real>,
    /// Covariance of `theta`, scaled by the noise scale's square.
    cov: DMatrix<Real>,
    residual_dof: usize,
    noise_scale: Real,
}

impl Solution {
    /// The covariance at `theta`; `None` when the normal equations are
    /// singular or the blur scale vanishes.
    fn new(obs: &[Obs], weights: &[Real], mut theta: Vec<Real>) -> Option<Self> {
        if theta[0] < 0.0 {
            // The model sees g², so −β is the same fit.
            theta[..4].iter_mut().for_each(|b| *b = -*b);
        }
        if !(theta[0].is_finite() && theta[0] > 0.0) {
            return None;
        }
        let (jtj, _) = normal_equations(obs, weights, &theta);
        let p = theta.len();
        let d = DVector::from_iterator(p, (0..p).map(|i| 1.0 / jtj[(i, i)].max(1e-300).sqrt()));
        let scaled = DMatrix::from_fn(p, p, |i, j| jtj[(i, j)] * d[i] * d[j]);
        let eig = scaled.clone().symmetric_eigen().eigenvalues;
        let (lo, hi) = eig.iter().fold((Real::INFINITY, 0.0_f64), |(lo, hi), &e| {
            (lo.min(e), hi.max(e))
        });
        if !(lo.is_finite() && lo > SINGULAR_RCOND * hi) {
            return None;
        }
        let inv = scaled.cholesky()?.inverse();
        let residual_dof = obs.len() - p;
        let chi2 = cost(obs, weights, &theta);
        let noise_var = chi2 / residual_dof as Real;
        let cov = DMatrix::from_fn(p, p, |i, j| noise_var * inv[(i, j)] * d[i] * d[j]);
        Some(Self {
            theta,
            cov,
            residual_dof,
            noise_scale: noise_var.sqrt(),
        })
    }

    fn into_fit(
        self,
        geometry: &SensorGeometry,
        obs: &[Obs],
        view_points: &[usize],
        column: &[Option<usize>],
        stated_weights: bool,
        blur_limit_px: Option<Real>,
    ) -> FocusFit {
        let th = &self.theta;
        let kappa = th[0];
        let pi_s = Vector3::new(-th[1] / kappa, -th[2] / kappa, -th[3] / kappa);
        let rt = geometry.rotation.transpose();
        let pi = rt * pi_s;

        // (β, σ0²) → (π, κ, σ0²): π' = −β₁..₃/β₀ in the sensor frame, then
        // π = Rᵀ·π'.
        let mut d_pis = SMatrix::<Real, 3, 5>::zeros();
        for i in 0..3 {
            d_pis[(i, 0)] = th[i + 1] / (kappa * kappa);
            d_pis[(i, i + 1)] = -1.0 / kappa;
        }
        let mut g = SMatrix::<Real, 5, 5>::zeros();
        g.fixed_view_mut::<3, 5>(0, 0).copy_from(&(rt * d_pis));
        g[(3, 0)] = 1.0;
        g[(4, 4)] = 1.0;
        let cb = SMatrix::<Real, 5, 5>::from_fn(|i, j| self.cov[(i, j)]);
        let cov5 = g * cb * g.transpose();
        let cov_pi = cov5.fixed_view::<3, 3>(0, 0).into_owned();
        let n = geometry.normal();

        let image_line = image_line(&geometry.pixel_map, &pi, &cov_pi);
        let depth_of_field =
            blur_limit_px.and_then(|limit| depth_of_field(limit, &pi, kappa, th[4], &n, &cov5));

        // Per-view residuals and crossing.
        let fitted = column.iter().flatten().count();
        let mut ss = vec![0.0; fitted];
        let mut max_abs = vec![0.0_f64; fitted];
        let mut count = vec![0usize; fitted];
        let mut sign = vec![(false, false); fitted];
        let mut total_ss = 0.0;
        for o in obs {
            let e = residual(th, o);
            ss[o.view] += e * e;
            total_ss += e * e;
            max_abs[o.view] = max_abs[o.view].max(e.abs());
            count[o.view] += 1;
            let d = defocus(&th[..4], &o.r);
            if d > 0.0 {
                sign[o.view].0 = true;
            } else if d < 0.0 {
                sign[o.view].1 = true;
            }
        }
        let views = column
            .iter()
            .zip(view_points)
            .map(|(col, &points)| match *col {
                Some(c) => FocusViewStats {
                    points,
                    used: true,
                    offset_px: offset(th, c),
                    offset_sigma_px: offset_variance(&self.cov, c).max(0.0).sqrt(),
                    rms_px: (ss[c] / count[c] as Real).sqrt(),
                    max_abs_px: max_abs[c],
                    crosses_focus: sign[c].0 && sign[c].1,
                },
                None => FocusViewStats {
                    points,
                    used: false,
                    offset_px: 0.0,
                    offset_sigma_px: 0.0,
                    rms_px: 0.0,
                    max_abs_px: 0.0,
                    crosses_focus: false,
                },
            })
            .collect();

        FocusFit {
            plane_inv_m: pi.into(),
            blur_scale_px_m: kappa,
            in_focus_variance_px2: th[4],
            covariance: std::array::from_fn(|i| std::array::from_fn(|j| cov5[(i, j)])),
            sensor_normal: n.into(),
            image_line,
            depth_of_field,
            points: obs.len(),
            residual_dof: self.residual_dof,
            stated_weights,
            noise_scale: self.noise_scale,
            residual_rms_px: (total_ss / obs.len() as Real).sqrt(),
            views,
        }
    }
}

/// The image line of the plane's points at infinity through the
/// distortion-free pixel map `m`, with its covariance; `None` when the plane
/// is parallel to the sensor.
fn image_line(
    m: &Matrix3<Real>,
    pi: &Vector3<Real>,
    cov_pi: &Matrix3<Real>,
) -> Option<FocusImageLine> {
    // A pixel p sees the direction m⁻¹·p, which meets π·X = 1 at infinity
    // where π·m⁻¹·p = 0: the line l = m⁻ᵀ·π, with l·p > 0 on the side where
    // the plane is in front.
    let m_inv_t = m.try_inverse()?.transpose();
    let l = m_inv_t * pi;
    let n2 = l.x * l.x + l.y * l.y;
    let n = n2.sqrt();
    if !(n > 0.0 && n.is_finite()) {
        return None;
    }
    let angle = (-l.y).atan2(-l.x);
    let offset = l.z / n;
    // dθ = (l_a·dl_b − l_b·dl_a)/n², dρ = dl_c/n − l_c·(l_a·dl_a + l_b·dl_b)/n³.
    let d_l = SMatrix::<Real, 2, 3>::new(
        -l.y / n2,
        l.x / n2,
        0.0,
        -l.z * l.x / (n2 * n),
        -l.z * l.y / (n2 * n),
        1.0 / n,
    );
    let jac = d_l * m_inv_t;
    let cov = jac * cov_pi * jac.transpose();
    Some(FocusImageLine {
        angle_rad: angle,
        offset_px: offset,
        covariance: [[cov[(0, 0)], cov[(0, 1)]], [cov[(1, 0)], cov[(1, 1)]]],
    })
}

/// The depth-of-field wedge for the blur limit `limit`; `None` when the
/// limit is not above the in-focus blur.
fn depth_of_field(
    limit: Real,
    pi: &Vector3<Real>,
    kappa: Real,
    s0sq: Real,
    n: &Vector3<Real>,
    cov5: &SMatrix<Real, 5, 5>,
) -> Option<DepthOfField> {
    let room = limit * limit - s0sq - PIXEL_VARIANCE;
    if !(room.is_finite() && room > 0.0) {
        return None;
    }
    let eps = room.sqrt() / kappa;
    // ε = √(σ_max² − σ0² − 1/12)/κ: ∂ε/∂κ = −ε/κ, ∂ε/∂σ0² = −1/(2κ²ε).
    let d_eps = SMatrix::<Real, 1, 5>::new(
        0.0,
        0.0,
        0.0,
        -eps / kappa,
        -1.0 / (2.0 * kappa * kappa * eps),
    );
    let var_eps = (d_eps * cov5 * d_eps.transpose())[(0, 0)];
    let plane = |sign: Real| {
        let mut jac = SMatrix::<Real, 3, 5>::zeros();
        jac.fixed_view_mut::<3, 3>(0, 0).fill_with_identity();
        for i in 0..3 {
            jac[(i, 3)] = sign * n[i] * d_eps[(0, 3)];
            jac[(i, 4)] = sign * n[i] * d_eps[(0, 4)];
        }
        let c = jac * cov5 * jac.transpose();
        (
            (pi + n * (sign * eps)).into(),
            std::array::from_fn(|i| std::array::from_fn(|j| c[(i, j)])),
        )
    };
    let (near, near_cov) = plane(1.0);
    let (far, far_cov) = plane(-1.0);
    Some(DepthOfField {
        blur_limit_px: limit,
        defocus_limit_inv_m: eps,
        defocus_limit_sigma_inv_m: var_eps.max(0.0).sqrt(),
        near_plane_inv_m: near,
        near_plane_covariance: near_cov,
        far_plane_inv_m: far,
        far_plane_covariance: far_cov,
    })
}

#[cfg(test)]
mod tests;
