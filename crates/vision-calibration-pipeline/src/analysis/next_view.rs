//! Planning the next view: the covariance a calibration would have with more
//! views of its target, predicted without a re-solve, and a ranking of
//! candidate target poses by how much each would reduce the projection
//! uncertainty.
//!
//! # Prediction
//!
//! A candidate is a target pose `camera_se3_target`. The target's points are
//! projected through the current estimate; those behind the camera or
//! outside the image are dropped. The candidate's information about the
//! camera is `JᵀJ` over the camera parameters and the candidate's own pose,
//! with the pose marginalized (a Schur complement), since a re-solve would
//! estimate it ([`view_information`]). Views are independent given the
//! camera, so the predicted information is the current one plus each
//! candidate's:
//!
//! ```text
//! Σ′ = (Σ⁻¹ + Σ_c S_c / σ²)⁻¹
//! ```
//!
//! over the parameters the result estimated. `Σ` is the result's covariance
//! of them: its inverse is the information the result reports, with every
//! existing pose already marginalized and conditional on the same fixed
//! parameters. `σ²` is the variance of a pixel residual that covariance
//! implies, read from the result itself: the existing views' information
//! `S_A` is rebuilt from the export's observed target points and poses, and
//! `σ² = tr(Σ S_A) / n`, since `Σ = σ² S_A⁻¹` for a planar calibration (one
//! residual family, no priors, a robust loss entering as one scalar). The
//! prediction is thus on the covariance's own scale, robust loss included,
//! and adding no information returns `Σ` unchanged.
//!
//! A candidate whose visible points cannot determine its pose (fewer than
//! four, the least a planar view needs, or a degenerate configuration) adds
//! nothing, as a re-solve would reject the view.
//!
//! # Ranking
//!
//! [`rank_candidates`] scores each candidate alone by the largest projection
//! σ over the image after adding it, on the grid
//! [`projection_uncertainty`](crate::common::CalibratedCameras::projection_uncertainty)
//! uses (the figure the quality check bounds), and reports the reduction of
//! `ln det Σ` alongside: a unit-free measure of the volume of the whole
//! covariance, where a trace would add pixels² to squared distortion
//! coefficients.

use serde::{Deserialize, Serialize};

use nalgebra::{DMatrix, Rotation3, Translation3, UnitQuaternion, Vector3};
use vision_calibration_core::{
    CameraModel, CameraParams, Iso3, ParameterCovariance, Pt2, Pt3, Uncertainty,
};
use vision_calibration_optim::{
    Error as OptimError, ProjectionUncertainty, ViewPose, camera_parameters,
    projection_uncertainty, view_information,
};

use crate::Error;
// A candidate seeing fewer points than a planar view needs adds nothing.
use crate::planar_intrinsics::{MIN_VIEW_POINTS, PlanarIntrinsicsExport};

/// The covariance a calibration is predicted to have with candidate views
/// added.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[non_exhaustive]
pub struct PredictedUncertainty {
    /// The predicted covariance of the parameters the result estimated, with
    /// the result's labels, fixed list and noise levels. Per-view blocks are
    /// not predicted (`views` is empty); `num_residuals` and `dof` count the
    /// candidates' visible points and poses.
    pub covariance: ParameterCovariance,
    /// The predicted projection uncertainty over the image.
    pub projection: ProjectionUncertainty,
    /// Per candidate, in order, the target points that land in the image.
    /// A candidate with fewer than four adds no information.
    pub visible_points: Vec<usize>,
    /// The standard deviation of a pixel residual (per axis) the result's
    /// covariance implies, which scales the candidates' information, in
    /// pixels.
    pub effective_noise_px: f64,
}

/// One candidate view, scored by the uncertainty it would leave.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[non_exhaustive]
pub struct RankedCandidate {
    /// The candidate's position in the list given to [`rank_candidates`].
    pub index: usize,
    /// Target points that land in the image. Fewer than four add no
    /// information.
    pub visible_points: usize,
    /// The largest projection σ over the image with this view added, in
    /// pixels.
    pub max_projection_sigma_px: f64,
    /// The current largest projection σ minus `max_projection_sigma_px`, in
    /// pixels (≥ 0).
    pub projection_sigma_reduction_px: f64,
    /// `ln det Σ − ln det Σ′`, the reduction of the log-volume of the whole
    /// covariance (≥ 0, unit-free).
    pub log_det_reduction: f64,
}

/// A deterministic set of candidate target poses: the target's centroid on
/// the ray through the centre of each region of an image grid, at a given
/// distance, facing the camera and tilted about each of its axes.
///
/// For each region, row by row from the top-left, the poses are the
/// fronto-parallel one, then tilts of `+tilt_rad` and `−tilt_rad` about the
/// target's x axis and about its y axis (only the fronto-parallel one when
/// `tilt_rad` is zero).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct CandidateGrid {
    /// Distance from the camera centre to the target's centroid, metres
    /// (> 0). No default: it depends on the target's size and the lens.
    pub distance_m: f64,
    /// Tilt out of the fronto-parallel pose, radians, in `[0, π/2)`. Tilted
    /// views are what determine the focal length and the principal point;
    /// 0.3–0.6 rad (about 15–35°) is typical.
    pub tilt_rad: f64,
    /// Regions of the image grid, `[columns, rows]`, each at least 1.
    pub regions: [usize; 2],
}

impl CandidateGrid {
    /// A grid of `regions` at `distance_m` metres, tilted by `tilt_rad`.
    pub fn new(distance_m: f64, tilt_rad: f64, regions: [usize; 2]) -> Self {
        Self {
            distance_m,
            tilt_rad,
            regions,
        }
    }
}

/// The covariance `export` is predicted to have with the views `candidates`
/// of the target `target_points` added, and its projection uncertainty over
/// an image of `image_size` pixels (see the module documentation).
///
/// `candidates` are `camera_se3_target` poses; `target_points` are the
/// target's points in its own frame, metres, as the calibration observed
/// them.
///
/// # Errors
///
/// - [`Error::InsufficientData`] when `candidates` or `target_points` is
///   empty;
/// - [`Error::CovarianceUnavailable`] when the result has no covariance;
/// - [`Error::InvalidInput`] when the image is smaller than 2×2 pixels, the
///   result holds every camera parameter fixed, its covariance names a
///   parameter that is not the camera's, or it carries no per-feature
///   residuals or poses to rebuild its information from;
/// - [`Error::Numerical`] when the covariance is not positive definite.
pub fn predict_with_views(
    export: &PlanarIntrinsicsExport,
    candidates: &[Iso3],
    target_points: &[Pt3],
    image_size: [u32; 2],
) -> Result<PredictedUncertainty, Error> {
    if candidates.is_empty() {
        return Err(Error::InsufficientData { need: 1, got: 0 });
    }
    let planner = Planner::new(export, target_points, image_size)?;
    let mut added = DMatrix::zeros(planner.free.len(), planner.free.len());
    let mut visible_points = Vec::with_capacity(candidates.len());
    let mut points_added = 0;
    let mut views_added = 0;
    for candidate in candidates {
        let (visible, information) = planner.candidate_information(candidate)?;
        visible_points.push(visible);
        if let Some(information) = information {
            added += information;
            points_added += visible;
            views_added += 1;
        }
    }
    let predicted = planner.predict(&added)?;
    let current = &planner.covariance;
    let covariance = ParameterCovariance {
        labels: current.labels.clone(),
        covariance: rows(&predicted),
        views: Vec::new(),
        fixed: current.fixed.clone(),
        noise: current.noise.clone(),
        num_residuals: current.num_residuals + 2 * points_added,
        dof: (current.dof + 2 * points_added).saturating_sub(6 * views_added),
        view_scatter: current.view_scatter,
    };
    let projection = planner.projection(&covariance)?;
    Ok(PredictedUncertainty {
        covariance,
        projection,
        visible_points,
        effective_noise_px: planner.sigma2.sqrt(),
    })
}

/// Every candidate of `candidates`, each added alone to `export`, ranked
/// from the largest reduction of the maximum projection σ over an image of
/// `image_size` pixels to the smallest; ties keep the candidates' order.
///
/// # Errors
///
/// As [`predict_with_views`].
pub fn rank_candidates(
    export: &PlanarIntrinsicsExport,
    candidates: &[Iso3],
    target_points: &[Pt3],
    image_size: [u32; 2],
) -> Result<Vec<RankedCandidate>, Error> {
    if candidates.is_empty() {
        return Err(Error::InsufficientData { need: 1, got: 0 });
    }
    let planner = Planner::new(export, target_points, image_size)?;
    let current_max = planner.projection(&planner.covariance)?.max;
    let current_log_det = log_det(&planner.covariance_matrix)
        .ok_or_else(|| Error::numerical("the result's covariance is not positive definite"))?;
    let mut ranked = Vec::with_capacity(candidates.len());
    for (index, candidate) in candidates.iter().enumerate() {
        let (visible, information) = planner.candidate_information(candidate)?;
        let (max, log_det_reduction) = match information {
            None => (current_max, 0.0),
            Some(information) => {
                let predicted = planner.predict(&information)?;
                let mut covariance = planner.covariance.clone();
                covariance.covariance = rows(&predicted);
                let max = planner.projection(&covariance)?.max;
                let reduction = log_det(&predicted)
                    .map(|l| current_log_det - l)
                    .ok_or_else(|| Error::numerical("a predicted covariance is singular"))?;
                (max, reduction)
            }
        };
        ranked.push(RankedCandidate {
            index,
            visible_points: visible,
            max_projection_sigma_px: max,
            projection_sigma_reduction_px: current_max - max,
            log_det_reduction,
        });
    }
    ranked.sort_by(|a, b| {
        a.max_projection_sigma_px
            .total_cmp(&b.max_projection_sigma_px)
            .then(a.index.cmp(&b.index))
    });
    Ok(ranked)
}

/// The candidate poses of `grid` for the target `target_points` (its own
/// frame, metres) seen by `camera` over an image of `image_size` pixels;
/// see [`CandidateGrid`].
///
/// # Errors
///
/// [`Error::InsufficientData`] when `target_points` is empty;
/// [`Error::InvalidInput`] when the image is smaller than 2×2 pixels or a
/// grid setting is out of its range; [`Error::Numerical`] when a region
/// centre has no camera ray in front of the camera.
pub fn candidate_poses(
    camera: &CameraParams,
    target_points: &[Pt3],
    image_size: [u32; 2],
    grid: &CandidateGrid,
) -> Result<Vec<Iso3>, Error> {
    check_target_and_image(target_points, image_size)?;
    if !(grid.distance_m > 0.0 && grid.distance_m.is_finite()) {
        return Err(Error::invalid_input(format!(
            "candidate distance {} m must be positive",
            grid.distance_m
        )));
    }
    if !(0.0..std::f64::consts::FRAC_PI_2).contains(&grid.tilt_rad) {
        return Err(Error::invalid_input(format!(
            "candidate tilt {} rad must be in [0, π/2)",
            grid.tilt_rad
        )));
    }
    let [columns, rows] = grid.regions;
    if columns == 0 || rows == 0 {
        return Err(Error::invalid_input(format!(
            "candidate regions {columns}×{rows} must be at least 1×1"
        )));
    }
    let model = camera.build()?;
    let centroid =
        target_points.iter().map(|p| p.coords).sum::<Vector3<f64>>() / target_points.len() as f64;
    let t = grid.tilt_rad;
    let tilts: &[(f64, f64)] = if t == 0.0 {
        &[(0.0, 0.0)]
    } else {
        &[(0.0, 0.0), (t, 0.0), (-t, 0.0), (0.0, t), (0.0, -t)]
    };
    let [width, height] = image_size.map(f64::from);
    let mut poses = Vec::with_capacity(columns * rows * tilts.len());
    for row in 0..rows {
        for column in 0..columns {
            let pixel = Pt2::new(
                (column as f64 + 0.5) * width / columns as f64,
                (row as f64 + 0.5) * height / rows as f64,
            );
            let ray = model
                .backproject_pixel(&pixel)
                .point_at_unit_depth()
                .ok_or_else(|| Error::numerical(format!("no camera ray reaches pixel {pixel}")))?
                .normalize();
            for &(about_x, about_y) in tilts {
                let rotation: UnitQuaternion<f64> =
                    Rotation3::from_euler_angles(about_x, about_y, 0.0).into();
                let translation = ray * grid.distance_m - rotation * centroid;
                poses.push(Iso3::from_parts(Translation3::from(translation), rotation));
            }
        }
    }
    Ok(poses)
}

/// What a prediction needs from the result: its covariance and information
/// over the estimated camera parameters, and the scale of a pixel residual.
struct Planner<'a> {
    camera: &'a CameraParams,
    model: CameraModel,
    target_points: &'a [Pt3],
    image_size: [u32; 2],
    covariance: ParameterCovariance,
    covariance_matrix: DMatrix<f64>,
    /// `Σ⁻¹`.
    information: DMatrix<f64>,
    /// Per covariance label, its index among the camera's parameters.
    free: Vec<usize>,
    /// The variance of a pixel residual, px².
    sigma2: f64,
}

impl<'a> Planner<'a> {
    fn new(
        export: &'a PlanarIntrinsicsExport,
        target_points: &'a [Pt3],
        image_size: [u32; 2],
    ) -> Result<Self, Error> {
        check_target_and_image(target_points, image_size)?;
        let covariance = match &export.uncertainty {
            Uncertainty::Available(covariance) => covariance.clone(),
            Uncertainty::Unavailable { reason } => {
                return Err(Error::CovarianceUnavailable {
                    reason: reason.clone(),
                });
            }
        };
        let camera = &export.params.camera;
        let names: Vec<&'static str> = camera_parameters(camera)
            .map_err(input_error)?
            .into_iter()
            .map(|(label, _)| label)
            .collect();
        let free = covariance
            .labels
            .iter()
            .map(|label| {
                names.iter().position(|n| n == label).ok_or_else(|| {
                    Error::invalid_input(format!(
                        "covariance label `{label}` is not a parameter of the camera"
                    ))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        if free.is_empty() {
            return Err(Error::invalid_input(
                "the result held every camera parameter fixed: there is no uncertainty to plan for",
            ));
        }
        let n = free.len();
        let covariance_matrix = DMatrix::from_fn(n, n, |i, j| covariance.covariance[i][j]);
        let information = invert_spd(&covariance_matrix)
            .ok_or_else(|| Error::numerical("the result's covariance is not positive definite"))?;

        let existing = existing_information(export, &covariance, &free)?;
        let sigma2 = (&covariance_matrix * existing).trace() / n as f64;
        if !(sigma2 > 0.0 && sigma2.is_finite()) {
            return Err(Error::numerical(format!(
                "the result's covariance implies a pixel variance of {sigma2}"
            )));
        }
        Ok(Self {
            camera,
            model: camera.build()?,
            target_points,
            image_size,
            covariance,
            covariance_matrix,
            information,
            free,
            sigma2,
        })
    }

    /// The candidate's visible points, and its information over the free
    /// parameters per unit pixel variance; `None` when it adds none.
    fn candidate_information(
        &self,
        camera_se3_target: &Iso3,
    ) -> Result<(usize, Option<DMatrix<f64>>), Error> {
        let [width, height] = self.image_size.map(f64::from);
        let visible: Vec<Pt3> = self
            .target_points
            .iter()
            .filter(|point| {
                let p_c = camera_se3_target.transform_point(point).coords;
                p_c.z > 0.0
                    && self.model.project_point_c(&p_c).is_some_and(|q| {
                        (0.0..=width).contains(&q.x) && (0.0..=height).contains(&q.y)
                    })
            })
            .copied()
            .collect();
        if visible.len() < MIN_VIEW_POINTS {
            return Ok((visible.len(), None));
        }
        let information = match view_information(
            self.camera,
            camera_se3_target,
            &visible,
            ViewPose::Estimated,
        ) {
            Ok(information) => information,
            // The points do not determine the pose: a re-solve would not
            // learn the camera from this view.
            Err(OptimError::Numerical(_)) => return Ok((visible.len(), None)),
            Err(err) => return Err(input_error(err)),
        };
        Ok((visible.len(), Some(self.restrict(&information.matrix))))
    }

    /// The block of a camera-parameter matrix over the free parameters.
    fn restrict(&self, matrix: &DMatrix<f64>) -> DMatrix<f64> {
        let n = self.free.len();
        DMatrix::from_fn(n, n, |i, j| matrix[(self.free[i], self.free[j])])
    }

    /// `(Σ⁻¹ + added / σ²)⁻¹`.
    fn predict(&self, added: &DMatrix<f64>) -> Result<DMatrix<f64>, Error> {
        let information = &self.information + added / self.sigma2;
        invert_spd(&information)
            .ok_or_else(|| Error::numerical("the predicted information is not positive definite"))
    }

    fn projection(&self, covariance: &ParameterCovariance) -> Result<ProjectionUncertainty, Error> {
        projection_uncertainty(self.camera, covariance, 0, self.image_size).map_err(input_error)
    }
}

/// `S_A`: the information of the result's own views over the free
/// parameters, per unit pixel variance, rebuilt from its observed target
/// points and poses. A view whose pose the covariance does not list was held
/// fixed and keeps its camera block whole.
fn existing_information(
    export: &PlanarIntrinsicsExport,
    covariance: &ParameterCovariance,
    free: &[usize],
) -> Result<DMatrix<f64>, Error> {
    let poses = &export.params.camera_se3_target;
    let mut points: Vec<Vec<Pt3>> = vec![Vec::new(); poses.len()];
    for record in &export.per_feature_residuals.target {
        let view = points.get_mut(record.pose).ok_or_else(|| {
            Error::invalid_input(format!(
                "a feature residual names view {}, but the result has {} poses",
                record.pose,
                poses.len()
            ))
        })?;
        view.push(Pt3::from(record.target_xyz_m));
    }
    if points.iter().all(Vec::is_empty) {
        return Err(Error::invalid_input(
            "the result carries no per-feature residuals to rebuild its information from",
        ));
    }
    let n = free.len();
    let mut total = DMatrix::zeros(n, n);
    for (view, (pose, view_points)) in poses.iter().zip(&points).enumerate() {
        if view_points.is_empty() {
            continue;
        }
        let estimated = covariance.views.iter().any(|v| v.view == view);
        let information = view_information(
            &export.params.camera,
            pose,
            view_points,
            if estimated {
                ViewPose::Estimated
            } else {
                ViewPose::Fixed
            },
        )
        .map_err(input_error)?;
        total += DMatrix::from_fn(n, n, |i, j| information.matrix[(free[i], free[j])]);
    }
    Ok(total)
}

fn check_target_and_image(target_points: &[Pt3], image_size: [u32; 2]) -> Result<(), Error> {
    if target_points.is_empty() {
        return Err(Error::InsufficientData { need: 1, got: 0 });
    }
    let [width, height] = image_size;
    if width < 2 || height < 2 {
        return Err(Error::invalid_input(format!(
            "image size {width}×{height} is smaller than 2×2 pixels"
        )));
    }
    Ok(())
}

/// An optim input error as the pipeline's own; others propagate.
fn input_error(err: OptimError) -> Error {
    match err {
        OptimError::InvalidInput { reason } => Error::InvalidInput { reason },
        err => err.into(),
    }
}

fn rows(m: &DMatrix<f64>) -> Vec<Vec<f64>> {
    m.row_iter().map(|r| r.iter().copied().collect()).collect()
}

/// The unit-diagonal scaling `d` of a symmetric matrix, `dᵢ = 1/√aᵢᵢ`;
/// `None` when a diagonal entry is not positive.
fn unit_diagonal(a: &DMatrix<f64>) -> Option<Vec<f64>> {
    let d: Vec<f64> = a.diagonal().iter().map(|v| 1.0 / v.sqrt()).collect();
    d.iter().all(|v| v.is_finite() && *v > 0.0).then_some(d)
}

/// The inverse of a symmetric positive-definite matrix, factored in its
/// unit-diagonal form so that parameters of very different units (pixels,
/// distortion coefficients) do not limit the precision.
fn invert_spd(a: &DMatrix<f64>) -> Option<DMatrix<f64>> {
    let n = a.nrows();
    let d = unit_diagonal(a)?;
    let scaled = DMatrix::from_fn(n, n, |i, j| a[(i, j)] * d[i] * d[j]);
    let inverse = scaled.cholesky()?.inverse();
    Some(DMatrix::from_fn(n, n, |i, j| inverse[(i, j)] * d[i] * d[j]))
}

/// `ln det` of a symmetric positive-definite matrix, through its
/// unit-diagonal form.
fn log_det(a: &DMatrix<f64>) -> Option<f64> {
    let n = a.nrows();
    let d = unit_diagonal(a)?;
    let scaled = DMatrix::from_fn(n, n, |i, j| a[(i, j)] * d[i] * d[j]);
    let chol = scaled.cholesky()?;
    let scaled_log_det: f64 = chol.l_dirty().diagonal().iter().map(|v| 2.0 * v.ln()).sum();
    Some(scaled_log_det - 2.0 * d.iter().map(|v| v.ln()).sum::<f64>())
}

#[cfg(test)]
mod tests;
