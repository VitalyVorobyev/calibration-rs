//! The pose of a known object from its pixels, with the covariance of the
//! pose.
//!
//! The object's points are known in its own frame; the pixels are seen by the
//! cameras of a [`CalibratedRig`], each with its stated noise. The pose comes
//! out in the **rig frame** (`rig_se3_object`), however many cameras saw the
//! object.
//!
//! Pixels become bearings through each camera's own model. The camera with the
//! most observations starts the pose in closed form (EPnP, or a plane
//! homography for a planar object) on the normalized coordinates of its
//! bearings; with [`PoseOptions::ransac`] the start comes from P3P inside a
//! RANSAC whose residual is the reprojection distance in σ of the stated
//! noise. The pose is then refined on the reprojection error of **every**
//! inlier of every camera (a generalized PnP), through the public problem
//! builder of `vision-calibration-optim`, with the rig's calibration as
//! consider parameters: the covariance of the pose has the pixels' share and
//! the calibration's share.

use crate::bearing::{bearing, camera_model};
use crate::triangulation::sub_covariance;
use crate::{Error, Result};
use nalgebra::{Matrix6, Vector2};
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;
use vision_calibration_core::RunControl;
use vision_calibration_core::{
    CalibratedRig, CameraModel, Estimator, Iso3, PointCovariance, Pt2, Pt3, RansacOptions, Real,
    Uncertainty, ransac_fit,
};
use vision_calibration_optim::RobustLoss;
use vision_calibration_optim::problem::{
    PixelNoise, Problem, SolveOptions, TargetPoint, Use, solve,
};
use vision_geometry::pnp::{epnp, p3p};

/// Bearings with a smaller `z` than this have no usable normalized
/// coordinates and do not start the pose.
const MIN_BEARING_Z: Real = 1e-3;

/// Refit-and-reselect rounds after the first refinement; the inlier set
/// settles in one or two.
const SETTLE_ROUNDS: usize = 3;

/// One known object point seen by rig camera `camera`, with its pixel's
/// stated noise.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PoseObservation {
    /// Index of the rig camera that saw the point.
    pub camera: usize,
    /// The point in the object's frame, metres.
    pub object: Pt3,
    /// The observed pixel.
    pub pixel: Pt2,
    /// The pixel's covariance, px²: [`PointCovariance::new`] for a general
    /// one, [`PointCovariance::isotropic`] for one σ, or
    /// [`CalibratedRig::reprojection_noise`] for the calibration's noise
    /// level. It is stated, not estimated.
    pub noise: PointCovariance,
}

impl PoseObservation {
    /// The observation of the object point `object` by rig camera `camera` at
    /// `pixel`, with the covariance `noise` of the pixel.
    pub fn new(camera: usize, object: Pt3, pixel: Pt2, noise: PointCovariance) -> Self {
        Self {
            camera,
            object,
            pixel,
            noise,
        }
    }
}

/// Options of [`estimate_pose`].
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct PoseOptions {
    /// RANSAC over the observations of the initializing camera (the one with
    /// the most). `thresh` is in σ of the stated noise: the Mahalanobis
    /// reprojection distance, `3.0` keeps about 99.7 % of the inliers
    /// whatever the focal length or camera model. `None`: every observation
    /// is an inlier.
    pub ransac: Option<RansacOptions>,
    /// Robust loss of the refinement, its scale in σ units (the residuals are
    /// whitened by their stated noise). The covariance takes the residuals
    /// at their stated variance.
    pub loss: RobustLoss,
    /// Progress observers and the cancellation token of the refinement; the
    /// default reports nothing and never cancels. Not part of the result: an
    /// uncancelled run with observers equals one without.
    pub control: RunControl,
}

impl Default for PoseOptions {
    /// No RANSAC and no robust loss: every observation counts, squared.
    fn default() -> Self {
        Self {
            ransac: None,
            loss: RobustLoss::None,
            control: RunControl::default(),
        }
    }
}

/// The pose of an object and its uncertainty.
#[derive(Debug, Clone, Serialize)]
#[non_exhaustive]
pub struct ObjectPose {
    /// The object's pose in the rig frame (object → rig).
    pub rig_se3_object: Iso3,
    /// Indices into the observations given of those the pose uses: all of
    /// them without RANSAC.
    pub inliers: Vec<usize>,
    /// The covariance of the pose over the right-perturbation tangent
    /// (`T·Exp(δ)`) with the labels `object.rx`, `.ry`, `.rz` (rad) and
    /// `object.tx`, `.ty`, `.tz` (m): from the observations' stated noise plus
    /// the calibration's share. Or why it is unavailable; a rig without a
    /// covariance gives
    /// [`UncertaintyUnavailable::CalibrationWithoutCovariance`](vision_calibration_core::UncertaintyUnavailable::CalibrationWithoutCovariance).
    pub uncertainty: Uncertainty,
    /// RMS reprojection error over the inliers, pixels: the root of the mean
    /// squared distance between a pixel and its point's projection (`√2 σ`
    /// for isotropic noise of `σ` per axis).
    pub reprojection_rms_px: Real,
}

impl ObjectPose {
    /// The 6×6 covariance of the pose over `[rx ry rz tx ty tz]`, rad and m,
    /// right-perturbation tangent; `None` when the uncertainty is
    /// unavailable.
    pub fn pose_covariance(&self) -> Option<Matrix6<Real>> {
        let axes = ["rx", "ry", "rz", "tx", "ty", "tz"].map(|a| format!("object.{a}"));
        sub_covariance(&self.uncertainty, axes.clone(), axes)
    }
}

/// Estimate the pose of an object from its points seen by the rig's cameras,
/// refine it and give its covariance.
///
/// `observations` are known object points with the pixels they were seen at.
/// The initializing camera, the one with the most observations (the lowest
/// index on a tie), starts the pose in closed form from the bearings of its
/// pixels. The pose is then refined on the reprojection error of every
/// inlier of every camera, each pixel weighted by its stated noise; the result
/// is in the rig frame whatever camera started it.
///
/// With [`PoseOptions::ransac`] the start is a P3P-based RANSAC on the
/// initializing camera whose residual is the reprojection distance in σ of
/// the stated noise, so a threshold of `3.0` means the same for every
/// camera. The inliers are then selected by the refined pose over every
/// camera, and refined again until the set settles (at most three rounds).
///
/// The covariance has two sources: the pixels' stated noise, and the rig's
/// calibration, which enters as consider parameters (its parameters stay at
/// their values and its covariance is propagated into the pose's).
///
/// ```
/// use vision_calibration_core::{
///     BrownConrady5, CalibratedRig, FxFyCxCy, Iso3, PointCovariance, Pt3, Uncertainty,
///     make_pinhole_camera, pinhole_camera_params,
/// };
/// use vision_mvg::pose::{PoseObservation, PoseOptions, estimate_pose};
///
/// // One 800 px camera is the rig.
/// let k = FxFyCxCy { fx: 800.0, fy: 800.0, cx: 320.0, cy: 240.0 };
/// let camera = pinhole_camera_params(&make_pinhole_camera(k, BrownConrady5::default()));
/// let rig = CalibratedRig::single(camera, Uncertainty::default())?;
///
/// // A 4 x 3 grid of points on a 10 cm pitch, 0.8 m ahead; pixels to 0.3 px.
/// let truth = Iso3::translation(-0.1, -0.05, 0.8);
/// let noise = PointCovariance::isotropic(0.3)?;
/// let observations: Vec<PoseObservation> = (0..12)
///     .map(|i| {
///         let p = Pt3::new(0.1 * f64::from(i % 4), 0.1 * f64::from(i / 4), 0.0);
///         let pixel = rig.model(0).unwrap().project_point(&(truth * p)).unwrap();
///         PoseObservation::new(0, p, pixel, noise)
///     })
///     .collect();
///
/// let pose = estimate_pose(&rig, &observations, &PoseOptions::default())?;
/// assert!((pose.rig_se3_object.translation.vector - truth.translation.vector).norm() < 1e-6);
/// assert_eq!(pose.inliers.len(), 12);
/// // This rig has no covariance, so neither does the pose; one with a
/// // covariance gives `pose.pose_covariance()`, a 6 x 6 matrix.
/// assert!(pose.pose_covariance().is_none());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// # Errors
///
/// [`Error::InvalidInput`], before any solve, for a camera index out of
/// range, a non-finite object point or pixel, a pixel without a finite
/// viewing ray, and fewer than 4 observations in the initializing camera (or
/// fewer than 4 of them ahead of it); [`Error::Core`] for invalid RANSAC
/// options; [`Error::NoConsensus`] when RANSAC finds no model;
/// [`Error::Geometry`] when the closed-form start is degenerate (collinear
/// points); [`Error::Numerical`] when the refinement fails to run;
/// [`Error::Cancelled`] when `opts.control`'s token is cancelled. A
/// refinement that stops short of convergence is not an error: the pose
/// stands and the uncertainty says
/// [`UncertaintyUnavailable::NotConverged`](vision_calibration_core::UncertaintyUnavailable::NotConverged).
pub fn estimate_pose(
    rig: &CalibratedRig,
    observations: &[PoseObservation],
    opts: &PoseOptions,
) -> Result<ObjectPose> {
    let start_camera = validate(rig, observations)?;
    if let Some(ransac) = &opts.ransac {
        ransac.validate()?;
    }

    // The closed-form start, from the initializing camera.
    let model = camera_model(rig, start_camera)?;
    let mut data = Vec::new();
    for (i, obs) in observations.iter().enumerate() {
        if obs.camera != start_camera {
            continue;
        }
        let b = bearing(model, &obs.pixel, i)?;
        if b.z > MIN_BEARING_Z {
            data.push(StartDatum {
                object: obs.object,
                normalized: Pt2::new(b.x / b.z, b.y / b.z),
                pixel: obs.pixel,
                noise: obs.noise,
                model,
            });
        }
    }
    if data.len() < 4 {
        return Err(Error::invalid_input(format!(
            "only {} observation(s) of camera {start_camera} lie ahead of it; a pose needs at least 4",
            data.len()
        )));
    }

    let start = match &opts.ransac {
        None => fit_pose(&data, &(0..data.len()).collect::<Vec<_>>())?,
        Some(ransac) => {
            let result = ransac_fit::<StartEstimator>(&data, ransac);
            match (result.success, result.model) {
                (true, Some(pose)) => pose,
                _ => return Err(Error::NoConsensus),
            }
        }
    };
    let mut rig_se3_object = rig.cam_se3_rig()[start_camera].inverse() * start;

    // Without RANSAC every observation counts. With it the inliers are those
    // of the start pose over every camera: the closed-form start only saw
    // one.
    let mut inliers: Vec<usize> = match &opts.ransac {
        None => (0..observations.len()).collect(),
        Some(ransac) => select(rig, observations, &rig_se3_object, ransac.thresh),
    };
    if inliers.len() < 4 {
        return Err(Error::NoConsensus);
    }

    // Refine; with RANSAC re-select by the refined pose and refine again
    // until the set settles.
    let mut solved = refine(
        rig,
        observations,
        &inliers,
        &rig_se3_object,
        opts.loss,
        &opts.control,
    )?;
    if let Some(ransac) = &opts.ransac {
        for _ in 0..SETTLE_ROUNDS {
            let next = select(rig, observations, &solved.0, ransac.thresh);
            if next == inliers || next.len() < 4 || next.len() < ransac.min_inliers {
                break;
            }
            inliers = next;
            rig_se3_object = solved.0;
            solved = refine(
                rig,
                observations,
                &inliers,
                &rig_se3_object,
                opts.loss,
                &opts.control,
            )?;
        }
    }
    let (rig_se3_object, uncertainty) = solved;

    let sum_sq: Real = inliers
        .iter()
        .map(|&i| {
            let obs = &observations[i];
            projection_error(rig, &rig_se3_object, obs).map_or(Real::INFINITY, |e| e.norm_squared())
        })
        .sum();
    Ok(ObjectPose {
        rig_se3_object,
        reprojection_rms_px: (sum_sq / inliers.len() as Real).sqrt(),
        inliers,
        uncertainty,
    })
}

/// Everything checked before any work. Returns the initializing camera.
fn validate(rig: &CalibratedRig, observations: &[PoseObservation]) -> Result<usize> {
    let mut counts = vec![0_usize; rig.num_cameras()];
    for (i, obs) in observations.iter().enumerate() {
        camera_model(rig, obs.camera).map_err(|e| match e {
            Error::InvalidInput { reason } => {
                Error::invalid_input(format!("observation {i}: {reason}"))
            }
            other => other,
        })?;
        let finite = obs.object.coords.iter().all(|v| v.is_finite())
            && obs.pixel.coords.iter().all(|v| v.is_finite());
        if !finite {
            return Err(Error::invalid_input(format!(
                "observation {i}: the object point or the pixel is not finite"
            )));
        }
        counts[obs.camera] += 1;
    }
    // The most observations; the lowest index on a tie.
    let (camera, &count) = counts
        .iter()
        .enumerate()
        .rev()
        .max_by_key(|&(_, &c)| c)
        .ok_or_else(|| Error::invalid_input("the rig has no cameras"))?;
    if count < 4 {
        return Err(Error::invalid_input(format!(
            "the camera with the most observations (camera {camera}) has {count}; a pose needs at least 4"
        )));
    }
    Ok(camera)
}

/// The pixel error `projection − pixel` of `obs` under `rig_se3_object`;
/// `None` when the point does not project in its camera.
fn projection_error(
    rig: &CalibratedRig,
    rig_se3_object: &Iso3,
    obs: &PoseObservation,
) -> Option<Vector2<Real>> {
    let model = rig.model(obs.camera)?;
    let p = rig.cam_se3_rig()[obs.camera] * rig_se3_object * obs.object;
    let px = model.project_point(&p)?;
    Some(px - obs.pixel)
}

/// The reprojection distance of `obs` in σ of its stated noise.
fn residual(rig: &CalibratedRig, rig_se3_object: &Iso3, obs: &PoseObservation) -> Real {
    projection_error(rig, rig_se3_object, obs)
        .map_or(Real::INFINITY, |e| obs.noise.whitened_norm([e.x, e.y]))
}

/// Indices of the observations within `thresh` σ of `rig_se3_object`.
fn select(
    rig: &CalibratedRig,
    observations: &[PoseObservation],
    rig_se3_object: &Iso3,
    thresh: Real,
) -> Vec<usize> {
    (0..observations.len())
        .filter(|&i| residual(rig, rig_se3_object, &observations[i]) <= thresh)
        .collect()
}

/// Refine `rig_se3_object` on the reprojection error of the observations
/// `used`. Returns the pose and its uncertainty.
fn refine(
    rig: &CalibratedRig,
    observations: &[PoseObservation],
    used: &[usize],
    rig_se3_object: &Iso3,
    loss: RobustLoss,
    control: &RunControl,
) -> Result<(Iso3, Uncertainty)> {
    let fail = |e: vision_calibration_optim::Error| match e {
        vision_calibration_optim::Error::Cancelled => Error::Cancelled,
        e => Error::numerical(format!("refining the pose: {e}")),
    };
    let mut problem = Problem::new();
    let blocks = problem
        .add_calibrated_rig(rig, Use::Consider)
        .map_err(fail)?;
    let object = problem.add_pose("object", rig_se3_object).map_err(fail)?;
    for &i in used {
        let obs = &observations[i];
        let (Some(camera), Some(chain)) =
            (blocks.camera(obs.camera), blocks.chain(obs.camera, object))
        else {
            return Err(Error::numerical(format!(
                "observation {i}: camera {} missing from the problem",
                obs.camera
            )));
        };
        problem
            .add_reprojection(
                camera,
                chain,
                TargetPoint::Known(obs.object),
                obs.pixel,
                PixelNoise::Stated(obs.noise),
                loss,
            )
            .map_err(|e| Error::numerical(format!("observation {i}: refining the pose: {e}")))?;
    }
    // `SolveOptions` is non-exhaustive: start from its default.
    let mut solve_options = SolveOptions::default();
    solve_options.control = control.clone();
    let solution = solve(&problem, &solve_options).map_err(fail)?;
    let pose = solution
        .pose(object)
        .ok_or_else(|| Error::numerical("the object pose is missing from the solution"))?;
    Ok((pose, solution.uncertainty))
}

/// One observation of the initializing camera as the closed-form start needs
/// it.
struct StartDatum<'a> {
    object: Pt3,
    /// `(x/z, y/z)` of the bearing.
    normalized: Pt2,
    pixel: Pt2,
    noise: PointCovariance,
    model: &'a CameraModel,
}

/// `cam_se3_object` from the normalized coordinates of the `sample` data: P3P
/// plus the fourth point for a minimal sample, otherwise EPnP (or the plane
/// homography for a planar object).
struct StartEstimator<'a>(PhantomData<&'a ()>);

impl<'a> Estimator for StartEstimator<'a> {
    type Datum = StartDatum<'a>;
    type Model = Iso3;

    const MIN_SAMPLES: usize = 4;

    fn fit(data: &[Self::Datum], sample_indices: &[usize]) -> Option<Self::Model> {
        if sample_indices.len() == Self::MIN_SAMPLES {
            // P3P returns up to four poses; the fourth point picks one.
            let world: Vec<Pt3> = sample_indices[..3]
                .iter()
                .map(|&i| data[i].object)
                .collect();
            let image: Vec<Pt2> = sample_indices[..3]
                .iter()
                .map(|&i| data[i].normalized)
                .collect();
            let candidates = p3p(&world, &image, &identity_k()).ok()?;
            let check = &data[sample_indices[3]];
            return candidates
                .into_iter()
                .map(|pose| (<Self as Estimator>::residual(&pose, check), pose))
                .filter(|(r, _)| r.is_finite())
                .min_by(|a, b| a.0.total_cmp(&b.0))
                .map(|(_, pose)| pose);
        }
        fit_pose(data, sample_indices).ok()
    }

    fn residual(model: &Self::Model, datum: &Self::Datum) -> f64 {
        let p = model * datum.object;
        if p.z <= 0.0 {
            return f64::INFINITY;
        }
        datum.model.project_point(&p).map_or(f64::INFINITY, |px| {
            datum
                .noise
                .whitened_norm([px.x - datum.pixel.x, px.y - datum.pixel.y])
        })
    }
}

fn identity_k() -> vision_calibration_core::FxFyCxCy<Real> {
    vision_calibration_core::FxFyCxCy {
        fx: 1.0,
        fy: 1.0,
        cx: 0.0,
        cy: 0.0,
    }
}

/// The closed-form pose of the `indices` data (EPnP; it handles a planar
/// object).
fn fit_pose(data: &[StartDatum<'_>], indices: &[usize]) -> Result<Iso3> {
    let world: Vec<Pt3> = indices.iter().map(|&i| data[i].object).collect();
    let image: Vec<Pt2> = indices.iter().map(|&i| data[i].normalized).collect();
    Ok(epnp(&world, &image, &identity_k())?)
}
