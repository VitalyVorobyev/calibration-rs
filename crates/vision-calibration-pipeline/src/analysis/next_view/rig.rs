//! Planning the next robot pose of a rig hand-eye calibration: the
//! covariance it would have with views at candidate robot poses added, and
//! a ranking of the candidates.
//!
//! A candidate is a robot pose `base_se3_gripper`. Each camera sees the
//! target through the result's chain at that pose (eye-in-hand: the rig on
//! the gripper and the target in the base; eye-to-hand: the rig in the base
//! and the target on the gripper); the points behind a camera or outside its
//! image are dropped, and a camera that keeps fewer than four adds nothing.
//! The view's information is exact over every parameter the result
//! estimated: each camera's parameters, the rig, the hand-eye and the target.
//! A result that corrected its robot poses corrects the candidate's too: the
//! correction is marginalized under the result's robot pose error, weighed
//! against its reprojection noise as the solve weighed it.
//!
//! As for a planar calibration, the prediction is
//!
//! ```text
//! Σ′ = (Σ⁻¹ + Σ_c S_c / σ²)⁻¹
//! ```
//!
//! with `σ² = tr(Σ S_A) / n` read from the result: `S_A` is the information
//! of its own views, rebuilt from its observed target points at the robot
//! poses it solved with. With exact robot poses, `Σ = σ² S_A⁻¹` holds
//! exactly; with corrected ones it holds when the reprojection noise is
//! Gaussian, since the solve weighs the robot pose priors at a median-based
//! estimate of that noise.
//!
//! # Ranking
//!
//! [`rank_robot_poses`] scores each candidate alone by the reduction of
//! `ln det Σ`: the volume of the whole covariance, so a view that pins down
//! the hand-eye counts as much as one that pins down the lenses, and
//! unit-free, since rescaling a parameter shifts both log-determinants
//! alike. The largest projection σ over every camera's image is reported
//! alongside.

use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};
use vision_calibration_core::{
    CameraModel, Iso3, NoiseLevel, ParameterCovariance, Pt3, ResidualKind, Uncertainty,
};
use vision_calibration_optim::{
    HandeyeMode, ProjectionUncertainty, RigHandeyeGeometry, RobotPose, projection_uncertainty,
    rig_handeye_view_information,
};

use super::{input_error, invert_spd, log_det, rows};
use crate::Error;
use crate::planar_intrinsics::MIN_VIEW_POINTS;
use crate::rig_handeye::RigHandeyeExport;

/// The covariance a rig hand-eye calibration is predicted to have with
/// views at candidate robot poses added.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[non_exhaustive]
pub struct PredictedRigUncertainty {
    /// The predicted covariance of the parameters the result estimated, with
    /// the result's labels, fixed list and noise levels. Per-view blocks are
    /// not predicted (`views` is empty); `num_residuals` and `dof` count the
    /// candidates' visible points.
    pub covariance: ParameterCovariance,
    /// Each camera's predicted projection uncertainty over its image.
    pub projection: Vec<ProjectionUncertainty>,
    /// Per candidate, in order, per camera, the target points that land in
    /// that camera's image. A camera that sees fewer than four adds no
    /// information.
    pub visible_points: Vec<Vec<usize>>,
    /// The standard deviation of a pixel residual (per axis) the result's
    /// covariance implies, which scales the candidates' information, in
    /// pixels.
    pub effective_noise_px: f64,
}

/// One candidate robot pose, scored by the uncertainty its view would
/// leave.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[non_exhaustive]
pub struct RankedRobotPose {
    /// The candidate's position in the list given to [`rank_robot_poses`].
    pub index: usize,
    /// Per camera, the target points that land in its image. A camera that
    /// sees fewer than four adds no information.
    pub visible_points: Vec<usize>,
    /// `ln det Σ − ln det Σ′`, the reduction of the log-volume of the whole
    /// covariance (≥ 0, unit-free).
    pub log_det_reduction: f64,
    /// The largest projection σ over every camera's image with this view
    /// added, in pixels.
    pub max_projection_sigma_px: f64,
    /// The current largest projection σ minus `max_projection_sigma_px`, in
    /// pixels (≥ 0).
    pub projection_sigma_reduction_px: f64,
}

/// The covariance `export` is predicted to have with views of the target
/// `target_points` at the robot poses `candidates` added, and each camera's
/// projection uncertainty over its image of `image_sizes` pixels (see the
/// module documentation).
///
/// `candidates` are `base_se3_gripper` poses; `target_points` are the
/// target's points in its own frame, metres, as the calibration observed
/// them; `image_sizes` has one `[width, height]` per camera.
///
/// # Errors
///
/// - [`Error::InsufficientData`] when `candidates` or `target_points` is
///   empty;
/// - [`Error::CovarianceUnavailable`] when the result has no covariance;
/// - [`Error::InvalidInput`] when `image_sizes` does not have one size per
///   camera or a size is smaller than 2×2 pixels, the export misses the
///   hand-eye or target pose of its mode or the robot pose error of its
///   corrections, its covariance names a parameter no view observes, its
///   observations were whitened by stated per-point covariances, or it
///   carries no per-feature residuals to rebuild its information from;
/// - [`Error::Numerical`] when the covariance is not positive definite.
pub fn predict_with_robot_poses(
    export: &RigHandeyeExport,
    candidates: &[Iso3],
    target_points: &[Pt3],
    image_sizes: &[[u32; 2]],
) -> Result<PredictedRigUncertainty, Error> {
    if candidates.is_empty() {
        return Err(Error::InsufficientData { need: 1, got: 0 });
    }
    let planner = RigPlanner::new(export, target_points, image_sizes)?;
    let n = planner.index.len();
    let mut added = DMatrix::zeros(n, n);
    let mut visible_points = Vec::with_capacity(candidates.len());
    let mut points_added = 0;
    for candidate in candidates {
        let (visible, information) = planner.candidate_information(candidate)?;
        if let Some(information) = information {
            added += information;
            points_added += visible
                .iter()
                .filter(|&&v| v >= MIN_VIEW_POINTS)
                .sum::<usize>();
        }
        visible_points.push(visible);
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
        dof: current.dof + 2 * points_added,
        view_scatter: current.view_scatter,
        second_order: None,
    };
    let projection = planner.projections(&covariance)?;
    Ok(PredictedRigUncertainty {
        covariance,
        projection,
        visible_points,
        effective_noise_px: planner.sigma2.sqrt(),
    })
}

/// Every robot pose of `candidates`, each added alone to `export`, ranked
/// from the largest reduction of `ln det Σ` to the smallest; ties keep the
/// candidates' order.
///
/// # Errors
///
/// As [`predict_with_robot_poses`].
pub fn rank_robot_poses(
    export: &RigHandeyeExport,
    candidates: &[Iso3],
    target_points: &[Pt3],
    image_sizes: &[[u32; 2]],
) -> Result<Vec<RankedRobotPose>, Error> {
    if candidates.is_empty() {
        return Err(Error::InsufficientData { need: 1, got: 0 });
    }
    let planner = RigPlanner::new(export, target_points, image_sizes)?;
    let current_max = max_sigma(&planner.projections(&planner.covariance)?);
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
                let max = max_sigma(&planner.projections(&covariance)?);
                let reduction = log_det(&predicted)
                    .map(|l| current_log_det - l)
                    .ok_or_else(|| Error::numerical("a predicted covariance is singular"))?;
                (max, reduction)
            }
        };
        ranked.push(RankedRobotPose {
            index,
            visible_points: visible,
            log_det_reduction,
            max_projection_sigma_px: max,
            projection_sigma_reduction_px: current_max - max,
        });
    }
    ranked.sort_by(|a, b| {
        b.log_det_reduction
            .total_cmp(&a.log_det_reduction)
            .then(a.index.cmp(&b.index))
    });
    Ok(ranked)
}

fn max_sigma(projections: &[ProjectionUncertainty]) -> f64 {
    projections.iter().map(|p| p.max).fold(0.0, f64::max)
}

/// What a prediction needs from a rig hand-eye result.
struct RigPlanner<'a> {
    export: &'a RigHandeyeExport,
    geometry: RigHandeyeGeometry<'a>,
    models: Vec<CameraModel>,
    robot: RobotPose,
    target_points: &'a [Pt3],
    image_sizes: &'a [[u32; 2]],
    covariance: ParameterCovariance,
    covariance_matrix: DMatrix<f64>,
    /// `Σ⁻¹`.
    information: DMatrix<f64>,
    /// Per covariance label, its index among a view's parameters.
    index: Vec<usize>,
    /// The variance of a pixel residual, px².
    sigma2: f64,
}

impl<'a> RigPlanner<'a> {
    fn new(
        export: &'a RigHandeyeExport,
        target_points: &'a [Pt3],
        image_sizes: &'a [[u32; 2]],
    ) -> Result<Self, Error> {
        if target_points.is_empty() {
            return Err(Error::InsufficientData { need: 1, got: 0 });
        }
        if image_sizes.len() != export.cameras.len() {
            return Err(Error::invalid_input(format!(
                "{} image sizes for {} cameras",
                image_sizes.len(),
                export.cameras.len()
            )));
        }
        if let Some([width, height]) = image_sizes.iter().find(|[w, h]| *w < 2 || *h < 2) {
            return Err(Error::invalid_input(format!(
                "image size {width}×{height} is smaller than 2×2 pixels"
            )));
        }
        let covariance = match &export.uncertainty {
            Uncertainty::Available(covariance) => covariance.clone(),
            Uncertainty::Unavailable { reason } => {
                return Err(Error::CovarianceUnavailable {
                    reason: reason.clone(),
                });
            }
        };
        let geometry = geometry(export)?;
        let robot = robot_pose(export, &covariance)?;
        let models = export
            .cameras
            .iter()
            .map(|c| c.build().map_err(Error::from))
            .collect::<Result<Vec<_>, _>>()?;

        // A view's parameters, to place the covariance's labels among them.
        let probe = rig_handeye_view_information(
            &geometry,
            &Iso3::identity(),
            &vec![Vec::new(); export.cameras.len()],
            RobotPose::Exact,
        )
        .map_err(input_error)?;
        let index = covariance
            .labels
            .iter()
            .map(|label| {
                probe.labels.iter().position(|l| l == label).ok_or_else(|| {
                    Error::invalid_input(format!(
                        "covariance label `{label}` is not a parameter a view observes"
                    ))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let n = index.len();
        if n == 0 {
            return Err(Error::invalid_input(
                "the result held every parameter fixed: there is no uncertainty to plan for",
            ));
        }
        let covariance_matrix = DMatrix::from_fn(n, n, |i, j| covariance.covariance[i][j]);
        let information = invert_spd(&covariance_matrix)
            .ok_or_else(|| Error::numerical("the result's covariance is not positive definite"))?;

        let mut planner = Self {
            export,
            geometry,
            models,
            robot,
            target_points,
            image_sizes,
            covariance,
            covariance_matrix,
            information,
            index,
            sigma2: f64::NAN,
        };
        let existing = planner.existing_information()?;
        let sigma2 = (&planner.covariance_matrix * existing).trace() / n as f64;
        if !(sigma2 > 0.0 && sigma2.is_finite()) {
            return Err(Error::numerical(format!(
                "the result's covariance implies a pixel variance of {sigma2}"
            )));
        }
        planner.sigma2 = sigma2;
        Ok(planner)
    }

    /// `S_A`: the information of the result's own views over the estimated
    /// parameters, per unit pixel variance, rebuilt from its observed target
    /// points at the robot poses it solved with.
    fn existing_information(&self) -> Result<DMatrix<f64>, Error> {
        let export = self.export;
        let views = export.rig_se3_target.len();
        let cameras = export.cameras.len();
        let mut points: Vec<Vec<Vec<Pt3>>> = vec![vec![Vec::new(); cameras]; views];
        for record in &export.diagnostics.per_feature_residuals.target {
            let slot = points
                .get_mut(record.view)
                .and_then(|view| view.get_mut(record.camera))
                .ok_or_else(|| {
                    Error::invalid_input(format!(
                        "a feature residual names view {} of camera {}, but the result has {views} \
                         views of {cameras} cameras",
                        record.view, record.camera
                    ))
                })?;
            slot.push(Pt3::from(record.target_xyz_m));
        }
        if points.iter().flatten().all(Vec::is_empty) {
            return Err(Error::invalid_input(
                "the result carries no per-feature residuals to rebuild its information from",
            ));
        }
        let n = self.index.len();
        let mut total = DMatrix::zeros(n, n);
        for (rig_se3_target, view_points) in export.rig_se3_target.iter().zip(&points) {
            if view_points.iter().all(Vec::is_empty) {
                continue;
            }
            let robot = self.solved_robot_pose(rig_se3_target);
            let information =
                rig_handeye_view_information(&self.geometry, &robot, view_points, self.robot)
                    .map_err(input_error)?;
            total += self.restrict(&information.matrix);
        }
        Ok(total)
    }

    /// The robot pose a view was solved at (its correction applied), from
    /// its `rig_se3_target`.
    fn solved_robot_pose(&self, rig_se3_target: &Iso3) -> Iso3 {
        let (h, t) = (&self.geometry.handeye, &self.geometry.target);
        match self.geometry.mode {
            // rig_se3_target = H⁻¹ · G⁻¹ · B.
            HandeyeMode::EyeInHand => t * rig_se3_target.inverse() * h.inverse(),
            // rig_se3_target = H · G · T.
            HandeyeMode::EyeToHand => h.inverse() * rig_se3_target * t.inverse(),
        }
    }

    /// Per camera, the candidate's visible points, and its information over
    /// the estimated parameters per unit pixel variance; `None` when it adds
    /// none.
    fn candidate_information(
        &self,
        base_se3_gripper: &Iso3,
    ) -> Result<(Vec<usize>, Option<DMatrix<f64>>), Error> {
        let (h, t) = (&self.geometry.handeye, &self.geometry.target);
        let rig_se3_target = match self.geometry.mode {
            HandeyeMode::EyeInHand => h.inverse() * base_se3_gripper.inverse() * t,
            HandeyeMode::EyeToHand => h * base_se3_gripper * t,
        };
        let mut counts = Vec::with_capacity(self.models.len());
        let mut observed = Vec::with_capacity(self.models.len());
        for (i, model) in self.models.iter().enumerate() {
            let cam_se3_target = self.geometry.cam_se3_rig[i] * rig_se3_target;
            let [width, height] = self.image_sizes[i].map(f64::from);
            let visible: Vec<Pt3> = self
                .target_points
                .iter()
                .filter(|point| {
                    let p_c = cam_se3_target.transform_point(point).coords;
                    // `project_point_c` is `None` outside the projection's
                    // domain.
                    model.project_point_c(&p_c).is_some_and(|q| {
                        (0.0..=width).contains(&q.x) && (0.0..=height).contains(&q.y)
                    })
                })
                .copied()
                .collect();
            counts.push(visible.len());
            observed.push(if visible.len() < MIN_VIEW_POINTS {
                Vec::new()
            } else {
                visible
            });
        }
        if observed.iter().all(Vec::is_empty) {
            return Ok((counts, None));
        }
        let information =
            rig_handeye_view_information(&self.geometry, base_se3_gripper, &observed, self.robot)
                .map_err(input_error)?;
        Ok((counts, Some(self.restrict(&information.matrix))))
    }

    /// The block of a view's information over the estimated parameters.
    fn restrict(&self, matrix: &DMatrix<f64>) -> DMatrix<f64> {
        let n = self.index.len();
        DMatrix::from_fn(n, n, |i, j| matrix[(self.index[i], self.index[j])])
    }

    /// `(Σ⁻¹ + added / σ²)⁻¹`.
    fn predict(&self, added: &DMatrix<f64>) -> Result<DMatrix<f64>, Error> {
        let information = &self.information + added / self.sigma2;
        invert_spd(&information)
            .ok_or_else(|| Error::numerical("the predicted information is not positive definite"))
    }

    fn projections(
        &self,
        covariance: &ParameterCovariance,
    ) -> Result<Vec<ProjectionUncertainty>, Error> {
        self.export
            .cameras
            .iter()
            .zip(self.image_sizes)
            .enumerate()
            .map(|(i, (camera, size))| {
                projection_uncertainty(camera, covariance, i, *size).map_err(input_error)
            })
            .collect()
    }
}

/// The export's chain: its cameras, rig, hand-eye and target pose.
fn geometry(export: &RigHandeyeExport) -> Result<RigHandeyeGeometry<'_>, Error> {
    let (handeye, target) = (
        *export.handeye.handeye().pose(),
        *export.handeye.target_pose(),
    );
    if export.cam_se3_rig.len() != export.cameras.len() {
        return Err(Error::invalid_input(format!(
            "{} cam_se3_rig poses for {} cameras",
            export.cam_se3_rig.len(),
            export.cameras.len()
        )));
    }
    Ok(RigHandeyeGeometry {
        cameras: &export.cameras,
        cam_se3_rig: &export.cam_se3_rig,
        mode: export.handeye.mode(),
        handeye,
        target,
    })
}

/// How the export's robot poses enter: exact, or corrected under its robot
/// pose error and weighed against its reprojection noise.
fn robot_pose(
    export: &RigHandeyeExport,
    covariance: &ParameterCovariance,
) -> Result<RobotPose, Error> {
    let pixel = covariance
        .noise
        .iter()
        .find(|n| n.kind == ResidualKind::Reprojection)
        .map(|n| n.level);
    let pixel_sigma = match pixel {
        Some(NoiseLevel::Sigma(sigma)) => sigma,
        Some(NoiseLevel::Sigma0(_)) => {
            return Err(Error::invalid_input(
                "the result's observations were whitened by stated per-point covariances, which \
                 a prediction does not model",
            ));
        }
        None => {
            return Err(Error::invalid_input(
                "the result's covariance states no reprojection noise",
            ));
        }
    };
    if export.robot_deltas.is_none() {
        return Ok(RobotPose::Exact);
    }
    let sigma = export.robot_pose_sigma.ok_or_else(|| {
        Error::invalid_input("the result corrected its robot poses but states no robot pose error")
    })?;
    Ok(RobotPose::Corrected { sigma, pixel_sigma })
}

#[cfg(test)]
mod tests;
