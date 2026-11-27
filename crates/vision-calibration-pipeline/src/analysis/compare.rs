//! [`compare_runs`]: calibrations of one rig side by side, with the
//! evidence each difference has and the test it supports, if any.
//!
//! Two runs differ in what they estimate (the camera models), in the data
//! (shared, nested or independent observations) and in the estimator (the
//! configuration). Which comparisons are valid depends on all three, so the
//! result reports every difference with its size in units of the reference
//! run's stated σ (materiality, never a test) and a statistical test only
//! where the data relation makes one valid.

use std::collections::BTreeSet;

use nalgebra::{Matrix6, UnitQuaternion, Vector3, Vector6};
use serde::{Deserialize, Serialize};
use vision_calibration_core::{
    CameraModelSpec, CameraParams, Compensation, FitRegion, Handeye, ImageGrid, Iso3,
    ParameterCovariance, ProjectionDifferenceOptions, ReprojectionStats, Uncertainty,
    UncertaintyUnavailable, projection_difference,
};
use vision_calibration_dataset::ProblemKind;
use vision_calibration_optim::camera_parameters;

use super::TargetFeatureResidual;
use super::quality::{QualityThresholds, Verdict, assess};
use crate::Error;
use crate::common::CalibratedCameras;
use crate::dataset_runner::CalibrationConfig;

/// A cross-check that a camera pose is the identity: the reference camera
/// of a rig is held at exactly the identity, so any rounding is far below.
const IDENTITY_TOLERANCE: f64 = 1e-12;

/// One run as [`compare_runs`] reads it: its export, the id and image size
/// of each of its cameras, and optionally the configuration it was
/// calibrated with.
#[non_exhaustive]
pub struct ComparedRun<'a> {
    /// The name the comparison shows for the run (`run-003`, or any label).
    pub name: String,
    /// The run's export.
    pub export: &'a dyn CalibratedCameras,
    /// Each camera's id, in the export's camera order. Cameras are matched
    /// across runs by id, not by position.
    pub camera_ids: Vec<String>,
    /// Each camera's image size `[width, height]`, pixels, in the export's
    /// camera order.
    pub image_sizes: Vec<[u32; 2]>,
    /// The configuration the run was calibrated with; when every run has
    /// one, each difference lists the configuration changes.
    pub config: Option<CalibrationConfig>,
}

impl<'a> ComparedRun<'a> {
    /// The run `name` of `export`, its cameras named `camera_ids` and of
    /// `image_sizes`, both in the export's camera order; no configuration.
    pub fn new(
        name: impl Into<String>,
        export: &'a dyn CalibratedCameras,
        camera_ids: Vec<String>,
        image_sizes: Vec<[u32; 2]>,
    ) -> Self {
        Self {
            name: name.into(),
            export,
            camera_ids,
            image_sizes,
            config: None,
        }
    }

    /// The same run with the configuration it was calibrated with.
    #[must_use]
    pub fn with_config(mut self, config: CalibrationConfig) -> Self {
        self.config = Some(config);
        self
    }
}

impl std::fmt::Debug for ComparedRun<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ComparedRun")
            .field("name", &self.name)
            .field("camera_ids", &self.camera_ids)
            .field("image_sizes", &self.image_sizes)
            .field("config", &self.config.is_some())
            .finish_non_exhaustive()
    }
}

/// How the observations of a run relate to the reference run's: what
/// decides which test of a difference is valid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[serde(tag = "kind", rename_all = "snake_case")]
#[non_exhaustive]
pub enum DataRelation {
    /// Not established: no test.
    #[default]
    Unknown,
    /// No observation in common: independent captures of the device.
    Independent,
    /// The same observations and the same configuration.
    Identical,
    /// The same observations under another configuration (camera model,
    /// loss, solver).
    SameData,
    /// One run's observations are a strict subset of the other's, and the
    /// configurations are equal: the runs differ by what was left out.
    Nested {
        /// The run with fewer observations.
        smaller: RunSide,
    },
    /// Shared and differing observations, or nested observations under
    /// different configurations.
    Overlapping,
}

/// One of the two runs of a difference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum RunSide {
    /// The reference run, the first of the comparison.
    Reference,
    /// The run compared with it.
    Other,
}

/// The options of [`compare_runs`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[serde(default, deny_unknown_fields)]
#[non_exhaustive]
pub struct CompareRunsOptions {
    /// How each run after the first relates to the first, one entry per
    /// run after the first; empty for [`DataRelation::Unknown`] everywhere.
    /// The relation decides which test a difference gets, so it is stated,
    /// never guessed.
    pub relations: Vec<DataRelation>,
    /// How the projection difference aligns the two camera frames; default
    /// [`Compensation::Rotation`].
    pub compensation: Compensation,
    /// Grid columns of the projection difference, at least 1; default 64.
    /// The rows follow from the image's aspect.
    pub grid_columns: u32,
    /// The thresholds each run's quality verdict is judged by.
    pub thresholds: QualityThresholds,
}

impl Default for CompareRunsOptions {
    fn default() -> Self {
        Self {
            relations: Vec::new(),
            compensation: Compensation::Rotation,
            grid_columns: 64,
            thresholds: QualityThresholds::default(),
        }
    }
}

impl CompareRunsOptions {
    /// The default options: every relation unknown, rotation-compensated
    /// projection differences on a 64-column grid, the default thresholds.
    pub fn new() -> Self {
        Self::default()
    }

    /// The options with the relation of each run after the first to the
    /// first.
    #[must_use]
    pub fn with_relations(mut self, relations: Vec<DataRelation>) -> Self {
        self.relations = relations;
        self
    }
}

/// Runs side by side against the first, the reference: what
/// [`compare_runs`] returns.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[non_exhaustive]
pub struct RunComparison {
    /// What each run says of itself, in the given order; the first is the
    /// reference.
    pub runs: Vec<RunEvidence>,
    /// Each run after the first against the reference, in run order.
    pub differences: Vec<RunDifference>,
}

/// One run's own evidence: its quality verdict, its covariance and the fit
/// of each camera.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[non_exhaustive]
pub struct RunEvidence {
    /// The run's name.
    pub name: String,
    /// The workflow that produced it.
    pub kind: ProblemKind,
    /// Its quality verdict, judged by the options' thresholds.
    pub verdict: Verdict,
    /// Whether it has a covariance.
    pub covariance: CovarianceStatus,
    /// The camera whose frame is the rig frame (its pose in the rig is the
    /// identity), by id; for one camera, that camera.
    pub reference_camera: Option<String>,
    /// Each camera, in the run's camera order.
    pub cameras: Vec<CameraEvidence>,
}

/// Whether a run has a covariance of its estimates.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[serde(tag = "status", rename_all = "snake_case")]
#[non_exhaustive]
pub enum CovarianceStatus {
    /// It has one.
    Available {
        /// The number of shared (not per-view) parameters it covers.
        shared_parameters: usize,
        /// The factor the covariance was scaled by because the views
        /// scatter more than the residual noise predicts; `None` when it
        /// was not scaled.
        view_scatter: Option<f64>,
    },
    /// It has none, for the reason given.
    Unavailable {
        /// Why.
        reason: UncertaintyUnavailable,
    },
}

/// One camera's fit in one run. The residual statistics are in-sample:
/// they compare like with like only between runs of the same observations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[non_exhaustive]
pub struct CameraEvidence {
    /// The camera's id.
    pub id: String,
    /// Its model; `None` for a general sensor homography, which no model
    /// spec describes.
    pub model: Option<CameraModelSpec>,
    /// Its image size `[width, height]`, pixels.
    pub image_size: [u32; 2],
    /// The views it observed features in.
    pub views: usize,
    /// Its features in the calibration.
    pub features: usize,
    /// Features the calibrated camera could not project.
    pub unprojected: usize,
    /// The statistics of its reprojection error lengths (mean, RMS, median,
    /// 95th percentile, largest), pixels; `None` without projected features.
    pub reprojection: Option<ReprojectionStats>,
    /// The largest major-axis projection σ over a grid across the image,
    /// pixels; `None` without a covariance of the camera's parameters.
    pub max_projection_sigma_px: Option<f64>,
}

/// One run against the reference.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[non_exhaustive]
pub struct RunDifference {
    /// The run's position in the comparison (1 for the first run after the
    /// reference).
    pub run: usize,
    /// How its observations relate to the reference's, as stated.
    pub relation: DataRelation,
    /// Whether the two runs' in-sample residual statistics compare like
    /// with like: only for the same observations ([`DataRelation::Identical`],
    /// [`DataRelation::SameData`]). A run that left out its worst features
    /// always has a lower residual, which says nothing of its calibration.
    pub fit_comparable: bool,
    /// The cameras, by id, whose camera models differ between the runs.
    pub model_changes: Vec<String>,
    /// Where the two configurations differ; `None` when either run's
    /// configuration was not given.
    pub config_changes: Option<Vec<ConfigChange>>,
    /// Each camera both runs have, in the reference's camera order.
    pub cameras: Vec<CameraDifference>,
    /// Each camera's pose relative to the reference run's reference camera,
    /// for the cameras both runs have (a rig).
    pub rig: Vec<PoseDifference>,
    /// The hand-eye, expressed for the reference run's reference camera;
    /// `None` unless both runs calibrated one in the same mode.
    pub handeye: Option<PoseDifference>,
    /// Each camera's laser plane, for the cameras both runs have one for.
    pub laser_planes: Vec<PlaneDifference>,
    /// The cameras of the run the reference does not have, and the
    /// reference's cameras it does not have, by id.
    pub unmatched_cameras: Vec<String>,
}

/// One setting where two configurations differ.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[non_exhaustive]
pub struct ConfigChange {
    /// The setting, as a path into the configuration's JSON
    /// (`config.optimize.reprojection_loss`, `config.camera.fix[2]`).
    pub path: String,
    /// The reference's value as JSON text; `None` where it has none.
    pub reference: Option<String>,
    /// The other run's value as JSON text; `None` where it has none.
    pub other: Option<String>,
}

/// One camera both runs have.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[non_exhaustive]
pub struct CameraDifference {
    /// The camera's id.
    pub id: String,
    /// Its parameters side by side, when both runs use one camera model.
    pub parameters: ParameterComparison,
    /// Where the two calibrations project the same rays.
    pub projection: ProjectionComparison,
}

/// A camera's parameters in two runs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[serde(tag = "kind", rename_all = "snake_case")]
#[non_exhaustive]
pub enum ParameterComparison {
    /// Both runs use one model: each parameter, in the solver's order.
    SameModel {
        /// The parameters.
        parameters: Vec<ParameterDifference>,
    },
    /// The models differ, so a parameter of one is not a parameter of the
    /// other; the projection difference compares them.
    DifferentModels,
    /// A camera whose parameters the solver does not name (a general
    /// sensor homography).
    Unavailable {
        /// Why.
        reason: String,
    },
}

/// One parameter in two runs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[non_exhaustive]
pub struct ParameterDifference {
    /// The parameter, as the covariance labels it without its camera
    /// prefix (`fx`, `k1`, `tilt_x`).
    pub label: String,
    /// Its value in the reference.
    pub reference: f64,
    /// Its value in the other run.
    pub other: f64,
    /// `other − reference`.
    pub delta: f64,
    /// The reference's stated σ; `None` when it did not estimate the
    /// parameter or has no covariance.
    pub reference_sigma: Option<f64>,
    /// The other run's stated σ, likewise.
    pub other_sigma: Option<f64>,
    /// `delta / reference_sigma`, signed: whether the change matters at
    /// the precision the reference claims. Not a test: two runs of the same
    /// observations can differ by many σ of either without disagreeing, and
    /// by a fraction of one while disagreeing.
    pub delta_over_reference_sigma: Option<f64>,
    /// The test of the difference, when the data relation supports one.
    pub test: DifferenceTest,
}

/// The test of a scalar difference `Δ = other − reference`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[serde(tag = "kind", rename_all = "snake_case")]
#[non_exhaustive]
pub enum DifferenceTest {
    /// Independent captures: `σ_Δ = √(σ_ref² + σ_other²)`, `z = Δ / σ_Δ`.
    /// Under the hypothesis that both runs estimate the same value, `z` is
    /// standard normal.
    Independent {
        /// `σ_Δ`.
        sigma: f64,
        /// `Δ / σ_Δ`.
        z: f64,
        /// The two-sided p-value of `z`.
        p_value: f64,
    },
    /// Nested observations under one configuration:
    /// `σ_Δ = √(σ_smaller² − σ_larger²)` (the Hausman form, each variance
    /// taken at the larger run's view-scatter factor), `z = Δ / σ_Δ`. Under
    /// the hypothesis that the left-out observations fit the model as the
    /// others do, `z` is standard normal.
    Nested {
        /// `σ_Δ`.
        sigma: f64,
        /// `Δ / σ_Δ`.
        z: f64,
        /// The two-sided p-value of `z`.
        p_value: f64,
    },
    /// No valid test.
    NotAvailable {
        /// Why.
        reason: NoTest,
    },
}

/// Why a difference has no test.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum NoTest {
    /// The data relation is not known.
    UnknownRelation,
    /// The runs have the same observations and configuration.
    Identical,
    /// The runs share observations ([`DataRelation::SameData`],
    /// [`DataRelation::Overlapping`]): their errors are correlated by an
    /// amount neither export states.
    SharedData,
    /// A run has no covariance.
    NoCovariance,
    /// A run held the quantity fixed, so states no σ of it.
    NotEstimated,
    /// Nested runs whose stated variances do not order (the smaller run's
    /// variance is not the larger): the Hausman form has no σ.
    NestedVarianceNotPositive,
    /// The runs' rigs have different reference cameras, so the other run's
    /// covariance is not of the compared pose.
    GaugeDiffers,
    /// The covariance of the difference is singular.
    SingularCovariance,
}

/// A pose in two runs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[non_exhaustive]
pub struct PoseDifference {
    /// The pose, its frames named by camera id: `<camera>_se3_<reference>`
    /// for a camera's pose relative to the reference run's reference
    /// camera; `gripper_se3_<reference>` or `<reference>_se3_base` for the
    /// hand-eye.
    pub label: String,
    /// The angle of the rotation between the two, degrees.
    pub rotation_deg: f64,
    /// The translations' difference, `other − reference`, in the pose's
    /// target frame, metres.
    pub translation_m: [f64; 3],
    /// Its length, metres.
    pub translation_norm_m: f64,
    /// `Log(P_ref⁻¹ · P_other)` as `[ωx, ωy, ωz, vx, vy, vz]` (radians,
    /// metres): the difference in the covariance's right-perturbation
    /// tangent, in the pose's source frame.
    pub tangent: [f64; 6],
    /// The tangent's length in the reference's covariance,
    /// `√(τᵀ Σ_ref⁻¹ τ)`: the change in units of the reference's stated
    /// uncertainty. Not a test.
    pub delta_over_reference_sigma: Option<f64>,
    /// The test of the difference, when the data relation supports one.
    pub test: PoseTest,
}

/// The test of a pose difference, its tangent `τ` against the covariance
/// of the difference.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[serde(tag = "kind", rename_all = "snake_case")]
#[non_exhaustive]
pub enum PoseTest {
    /// Independent captures: `χ² = τᵀ (Σ_ref + Σ_other)⁻¹ τ`.
    Independent {
        /// `χ²`.
        chi2: f64,
        /// Its degrees of freedom (6).
        dof: usize,
        /// The upper-tail p-value of `χ²`.
        p_value: f64,
    },
    /// Nested observations under one configuration:
    /// `χ² = τᵀ (Σ_smaller − Σ_larger)⁻¹ τ` (the Hausman form).
    Nested {
        /// `χ²`.
        chi2: f64,
        /// Its degrees of freedom (6).
        dof: usize,
        /// The upper-tail p-value of `χ²`.
        p_value: f64,
    },
    /// No valid test.
    NotAvailable {
        /// Why.
        reason: NoTest,
    },
}

/// A camera's laser plane in two runs, in the camera's frame.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[non_exhaustive]
pub struct PlaneDifference {
    /// The camera's id.
    pub camera: String,
    /// The angle between the two normals, degrees.
    pub normal_angle_deg: f64,
    /// The reference's distance `d` of `n · p + d = 0`, metres.
    pub distance_reference_m: f64,
    /// The other run's distance, its plane's sign matched to the
    /// reference's normal, metres.
    pub distance_other_m: f64,
    /// `other − reference`, metres.
    pub distance_delta_m: f64,
    /// `distance_delta_m` over the reference's σ of `d`. Not a test.
    pub distance_over_reference_sigma: Option<f64>,
    /// The test of the distance difference, when the data relation
    /// supports one.
    pub distance_test: DifferenceTest,
}

/// Where two calibrations of a camera project the same rays.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[serde(tag = "status", rename_all = "snake_case")]
#[non_exhaustive]
pub enum ProjectionComparison {
    /// The projection difference, `other − reference` at the reference's
    /// rays, after the compensation; fitted on the grid blocks either run's
    /// features cover.
    Compared {
        /// How the other camera's frame was aligned.
        compensation: Compensation,
        /// The angle of the compensating rotation, degrees.
        rotation_deg: f64,
        /// Grid nodes the rotation was fitted on and the `*_fit_px`
        /// statistics are taken over: the covered region.
        fit_nodes: usize,
        /// Grid nodes where both cameras have a value.
        valid_nodes: usize,
        /// The root mean square difference over the fitted nodes, pixels.
        rms_fit_px: f64,
        /// The largest difference over the fitted nodes, pixels.
        max_fit_px: f64,
        /// The root mean square over every valid node, pixels: outside the
        /// covered region, an extrapolation of both models.
        rms_valid_px: f64,
        /// The largest difference over every valid node, pixels.
        max_valid_px: f64,
    },
    /// Not computed, for the reason given.
    Unavailable {
        /// Why.
        reason: String,
    },
}

/// Calibrations of one rig side by side against the first, the reference:
/// the evidence each run has, and each difference with its materiality and,
/// where the data relation supports one, its test.
///
/// Every run is of one workflow. Cameras are matched by id. For each run
/// after the first, and each camera both have:
///
/// - **parameters**, when both use one camera model: each value, its stated
///   σ in each run, `Δ = other − reference`, `Δ / σ_ref` (materiality: how
///   large the change is at the precision the reference claims, not a test)
///   and a [`DifferenceTest`];
/// - the **projection difference** over the image
///   ([`projection_difference`](vision_calibration_core::projection_difference),
///   fitted on the blocks either run's features cover), whatever the
///   models;
/// - the **rig** poses relative to the reference run's reference camera, the
///   **hand-eye** expressed for that camera, and the **laser planes**: each
///   rotation as an angle, each translation as a length and its components,
///   and a 6-dof `χ²` where a test is valid. Rig poses are compared
///   relative to one camera, so the comparison does not depend on which
///   camera each run held fixed; a test needs both runs to have held that
///   same camera.
///
/// Per-view poses, target poses and robot-pose corrections describe the
/// capture, not the device, and are not compared.
///
/// # Tests
///
/// The relation of each run to the reference is stated in
/// [`CompareRunsOptions::relations`]; it decides the test:
///
/// | Relation | Test |
/// |---|---|
/// | [`Independent`](DataRelation::Independent) | `σ_Δ² = σ_ref² + σ_other²` |
/// | [`Nested`](DataRelation::Nested) | `σ_Δ² = σ_smaller² − σ_larger²` (Hausman), each variance taken at the larger run's view-scatter factor |
/// | any other | none, with the reason |
///
/// The nested form holds because, when the left-out observations fit the
/// model as the rest do, the estimate from all observations is uncorrelated
/// with its difference from the estimate without them. Runs of the same
/// observations under different configurations have no test: their errors
/// are almost fully correlated by an amount neither export states, so a
/// small difference does not mean agreement and a large one is no
/// significance.
///
/// # Errors
///
/// [`Error::InvalidInput`] for fewer than two runs, runs of different
/// workflows, camera ids or image sizes not one per camera, a camera id
/// repeated in a run, a run sharing no camera id with the reference, a
/// shared camera whose image sizes differ, relations not one per run after
/// the first, or zero grid columns. Errors of a run's quality assessment
/// pass through.
pub fn compare_runs(
    runs: &[ComparedRun<'_>],
    options: &CompareRunsOptions,
) -> Result<RunComparison, Error> {
    let prepared = validate(runs, options)?;
    let evidence = runs
        .iter()
        .zip(&prepared)
        .map(|(run, p)| run_evidence(run, p, &options.thresholds))
        .collect::<Result<Vec<_>, Error>>()?;
    let reference = &prepared[0];
    let differences = (1..runs.len())
        .map(|i| {
            let relation = options
                .relations
                .get(i - 1)
                .copied()
                .unwrap_or(DataRelation::Unknown);
            run_difference(
                i,
                relation,
                (&runs[0], reference),
                (&runs[i], &prepared[i]),
                options,
            )
        })
        .collect();
    Ok(RunComparison {
        runs: evidence,
        differences,
    })
}

/// What [`compare_runs`] reads of one run, once.
struct Prepared {
    cameras: Vec<CameraParams>,
    cam_se3_rig: Vec<Iso3>,
    /// The index of the camera whose rig pose is the identity.
    reference_camera: Option<usize>,
    covariance: Option<ParameterCovariance>,
}

impl Prepared {
    fn multi_camera(&self) -> bool {
        self.cameras.len() > 1
    }

    /// The covariance label of camera `camera`'s parameter `name`.
    fn label(&self, camera: usize, name: &str) -> String {
        if self.multi_camera() {
            format!("cam{camera}.{name}")
        } else {
            name.to_string()
        }
    }

    /// The view-scatter factor the covariance was scaled by (1 when it was
    /// not).
    fn view_scatter(&self) -> f64 {
        self.covariance
            .as_ref()
            .and_then(|c| c.view_scatter)
            .unwrap_or(1.0)
    }
}

fn validate(
    runs: &[ComparedRun<'_>],
    options: &CompareRunsOptions,
) -> Result<Vec<Prepared>, Error> {
    if runs.len() < 2 {
        return Err(Error::invalid_input(format!(
            "compare_runs needs at least two runs, got {}",
            runs.len()
        )));
    }
    if options.grid_columns == 0 {
        return Err(Error::invalid_input("grid_columns must be at least 1"));
    }
    if !options.relations.is_empty() && options.relations.len() != runs.len() - 1 {
        return Err(Error::invalid_input(format!(
            "relations has {} entries; expected one per run after the first ({})",
            options.relations.len(),
            runs.len() - 1
        )));
    }
    let reference_kind = export_kind(runs[0].export);
    let mut prepared = Vec::with_capacity(runs.len());
    for run in runs {
        if export_kind(run.export) != reference_kind {
            return Err(Error::invalid_input(format!(
                "run {} is of another workflow than run {}",
                run.name, runs[0].name
            )));
        }
        let cameras = run.export.camera_params();
        if run.camera_ids.len() != cameras.len() || run.image_sizes.len() != cameras.len() {
            return Err(Error::invalid_input(format!(
                "run {} has {} cameras but {} camera ids and {} image sizes",
                run.name,
                cameras.len(),
                run.camera_ids.len(),
                run.image_sizes.len()
            )));
        }
        let mut seen = BTreeSet::new();
        for id in &run.camera_ids {
            if !seen.insert(id.as_str()) {
                return Err(Error::invalid_input(format!(
                    "run {} names camera {id} twice",
                    run.name
                )));
            }
        }
        let cam_se3_rig = run.export.cam_se3_rig();
        let reference_camera = cam_se3_rig.iter().position(is_identity);
        prepared.push(Prepared {
            cameras,
            cam_se3_rig,
            reference_camera,
            covariance: run.export.parameter_uncertainty().covariance().cloned(),
        });
    }
    for run in &runs[1..] {
        let mut shared = 0;
        for (j, id) in run.camera_ids.iter().enumerate() {
            if let Some(i) = runs[0].camera_ids.iter().position(|r| r == id) {
                shared += 1;
                if runs[0].image_sizes[i] != run.image_sizes[j] {
                    return Err(Error::invalid_input(format!(
                        "camera {id} has images of {:?} in run {} and {:?} in run {}",
                        runs[0].image_sizes[i], runs[0].name, run.image_sizes[j], run.name
                    )));
                }
            }
        }
        if shared == 0 {
            return Err(Error::invalid_input(format!(
                "run {} shares no camera id with run {}",
                run.name, runs[0].name
            )));
        }
    }
    Ok(prepared)
}

fn export_kind(export: &dyn CalibratedCameras) -> ProblemKind {
    export.kind()
}

fn is_identity(pose: &Iso3) -> bool {
    pose.rotation.angle() <= IDENTITY_TOLERANCE
        && pose.translation.vector.norm() <= IDENTITY_TOLERANCE
}

fn run_evidence(
    run: &ComparedRun<'_>,
    prepared: &Prepared,
    thresholds: &QualityThresholds,
) -> Result<RunEvidence, Error> {
    let report = assess(run.export, &run.image_sizes, thresholds)?;
    let covariance = match run.export.parameter_uncertainty() {
        Uncertainty::Available(c) => CovarianceStatus::Available {
            shared_parameters: c.labels.len(),
            view_scatter: c.view_scatter,
        },
        Uncertainty::Unavailable { reason } => CovarianceStatus::Unavailable {
            reason: reason.clone(),
        },
    };
    let residuals = &run.export.diagnostics().per_feature_residuals.target;
    let cameras = prepared
        .cameras
        .iter()
        .enumerate()
        .map(|(i, params)| {
            let own: Vec<&TargetFeatureResidual> =
                residuals.iter().filter(|r| r.camera == i).collect();
            let views: BTreeSet<usize> = own.iter().map(|r| r.view).collect();
            let errors: Vec<f64> = own
                .iter()
                .filter_map(|r| r.error_px)
                .filter(|e| e.is_finite())
                .collect();
            CameraEvidence {
                id: run.camera_ids[i].clone(),
                model: params.spec(),
                image_size: run.image_sizes[i],
                views: views.len(),
                features: own.len(),
                unprojected: own.iter().filter(|r| r.error_px.is_none()).count(),
                reprojection: (!errors.is_empty()).then(|| ReprojectionStats::from_errors(&errors)),
                max_projection_sigma_px: run
                    .export
                    .projection_uncertainty(i, run.image_sizes[i])
                    .ok()
                    .map(|u| u.max),
            }
        })
        .collect();
    Ok(RunEvidence {
        name: run.name.clone(),
        kind: export_kind(run.export),
        verdict: report.verdict,
        covariance,
        reference_camera: prepared.reference_camera.map(|i| run.camera_ids[i].clone()),
        cameras,
    })
}

/// The two runs of a difference, each with what was read of it.
type Side<'r, 'a> = (&'r ComparedRun<'a>, &'r Prepared);

fn run_difference(
    index: usize,
    relation: DataRelation,
    reference: Side<'_, '_>,
    other: Side<'_, '_>,
    options: &CompareRunsOptions,
) -> RunDifference {
    let (ref_run, ref_p) = reference;
    let (oth_run, oth_p) = other;
    let pairs: Vec<(usize, usize)> = ref_run
        .camera_ids
        .iter()
        .enumerate()
        .filter_map(|(i, id)| {
            oth_run
                .camera_ids
                .iter()
                .position(|o| o == id)
                .map(|j| (i, j))
        })
        .collect();
    let mut unmatched: Vec<String> = oth_run
        .camera_ids
        .iter()
        .filter(|id| !ref_run.camera_ids.contains(id))
        .chain(
            ref_run
                .camera_ids
                .iter()
                .filter(|id| !oth_run.camera_ids.contains(id)),
        )
        .cloned()
        .collect();
    unmatched.sort();
    let scale = Scale::new(relation, ref_p, oth_p);
    let cameras: Vec<CameraDifference> = pairs
        .iter()
        .map(|&(i, j)| camera_difference(i, j, reference, other, &scale, options))
        .collect();
    let model_changes = pairs
        .iter()
        .filter(|&&(i, j)| ref_p.cameras[i].spec() != oth_p.cameras[j].spec())
        .map(|&(i, _)| ref_run.camera_ids[i].clone())
        .collect();
    let config_changes = match (&ref_run.config, &oth_run.config) {
        (Some(a), Some(b)) => Some(config_changes(a, b)),
        _ => None,
    };
    let rig = rig_differences(&pairs, reference, other, &scale);
    let handeye = handeye_difference(&pairs, reference, other, &scale);
    let laser_planes = plane_differences(&pairs, reference, other, &scale);
    RunDifference {
        run: index,
        relation,
        fit_comparable: matches!(relation, DataRelation::Identical | DataRelation::SameData),
        model_changes,
        config_changes,
        cameras,
        rig,
        handeye,
        laser_planes,
        unmatched_cameras: unmatched,
    }
}

/// How a difference is tested: the relation, and for nested runs which one
/// is smaller and the view-scatter factors that put both covariances on the
/// larger run's noise scale.
struct Scale<'p> {
    relation: DataRelation,
    reference: &'p Prepared,
    other: &'p Prepared,
}

impl<'p> Scale<'p> {
    fn new(relation: DataRelation, reference: &'p Prepared, other: &'p Prepared) -> Self {
        Self {
            relation,
            reference,
            other,
        }
    }

    /// Why no test applies, before any σ is read; `None` when the relation
    /// supports one.
    fn relation_bar(&self) -> Option<NoTest> {
        match self.relation {
            DataRelation::Independent | DataRelation::Nested { .. } => {
                if self.reference.covariance.is_none() || self.other.covariance.is_none() {
                    Some(NoTest::NoCovariance)
                } else {
                    None
                }
            }
            DataRelation::Unknown => Some(NoTest::UnknownRelation),
            DataRelation::Identical => Some(NoTest::Identical),
            DataRelation::SameData | DataRelation::Overlapping => Some(NoTest::SharedData),
        }
    }

    /// The variance of `Δ` from the two runs' variances of the quantity.
    /// `Err` with the reason when there is none.
    fn variance(&self, reference_var: f64, other_var: f64) -> Result<f64, NoTest> {
        match self.relation {
            DataRelation::Independent => Ok(reference_var + other_var),
            DataRelation::Nested { smaller } => {
                let (small, large) = self.nested_pair(smaller, reference_var, other_var);
                let v = small - large;
                if v > 0.0 && v.is_finite() {
                    Ok(v)
                } else {
                    Err(NoTest::NestedVarianceNotPositive)
                }
            }
            _ => Err(self.relation_bar().unwrap_or(NoTest::UnknownRelation)),
        }
    }

    /// The smaller and larger run's variances, each divided by its own
    /// view-scatter factor and multiplied by the larger run's: both on the
    /// larger run's noise scale.
    fn nested_pair(&self, smaller: RunSide, reference: f64, other: f64) -> (f64, f64) {
        let (vr, vo) = (self.reference.view_scatter(), self.other.view_scatter());
        match smaller {
            RunSide::Reference => (reference / vr * vo, other),
            RunSide::Other => (other / vo * vr, reference),
        }
    }

    fn scalar_test(
        &self,
        delta: f64,
        reference_var: Option<f64>,
        other_var: Option<f64>,
    ) -> DifferenceTest {
        if let Some(reason) = self.relation_bar() {
            return DifferenceTest::NotAvailable { reason };
        }
        let (Some(a), Some(b)) = (reference_var, other_var) else {
            return DifferenceTest::NotAvailable {
                reason: NoTest::NotEstimated,
            };
        };
        match self.variance(a, b) {
            Ok(v) => {
                let sigma = v.sqrt();
                let z = delta / sigma;
                let p_value = chi2_survival(z * z, 1);
                match self.relation {
                    DataRelation::Nested { .. } => DifferenceTest::Nested { sigma, z, p_value },
                    _ => DifferenceTest::Independent { sigma, z, p_value },
                }
            }
            Err(reason) => DifferenceTest::NotAvailable { reason },
        }
    }

    fn pose_test(
        &self,
        tangent: &Vector6<f64>,
        reference: Option<Matrix6<f64>>,
        other: Option<Matrix6<f64>>,
    ) -> PoseTest {
        if let Some(reason) = self.relation_bar() {
            return PoseTest::NotAvailable { reason };
        }
        let (Some(a), Some(b)) = (reference, other) else {
            return PoseTest::NotAvailable {
                reason: NoTest::NotEstimated,
            };
        };
        let (covariance, nested) = match self.relation {
            DataRelation::Nested { smaller } => {
                let (vr, vo) = (self.reference.view_scatter(), self.other.view_scatter());
                let (small, large) = match smaller {
                    RunSide::Reference => (a / vr * vo, b),
                    RunSide::Other => (b / vo * vr, a),
                };
                (small - large, true)
            }
            _ => (a + b, false),
        };
        let Some(chi2) = mahalanobis_squared(&covariance, tangent) else {
            return PoseTest::NotAvailable {
                reason: if nested {
                    NoTest::NestedVarianceNotPositive
                } else {
                    NoTest::SingularCovariance
                },
            };
        };
        let p_value = chi2_survival(chi2, 6);
        if nested {
            PoseTest::Nested {
                chi2,
                dof: 6,
                p_value,
            }
        } else {
            PoseTest::Independent {
                chi2,
                dof: 6,
                p_value,
            }
        }
    }
}

/// `τᵀ Σ⁻¹ τ` through a Cholesky factor; `None` when `Σ` is not positive
/// definite.
fn mahalanobis_squared(covariance: &Matrix6<f64>, tangent: &Vector6<f64>) -> Option<f64> {
    let symmetric = 0.5 * (covariance + covariance.transpose());
    let cholesky = symmetric.cholesky()?;
    let solved = cholesky.solve(tangent);
    let value = tangent.dot(&solved);
    value.is_finite().then_some(value)
}

fn camera_difference(
    i: usize,
    j: usize,
    reference: Side<'_, '_>,
    other: Side<'_, '_>,
    scale: &Scale<'_>,
    options: &CompareRunsOptions,
) -> CameraDifference {
    let (ref_run, ref_p) = reference;
    let (oth_run, oth_p) = other;
    let (a, b) = (&ref_p.cameras[i], &oth_p.cameras[j]);
    let parameters = if a.spec() != b.spec() {
        ParameterComparison::DifferentModels
    } else {
        match (camera_parameters(a), camera_parameters(b)) {
            (Ok(pa), Ok(pb)) => ParameterComparison::SameModel {
                parameters: pa
                    .iter()
                    .zip(&pb)
                    .map(|(&(label, va), &(_, vb))| {
                        let var_a = variance(ref_p, &ref_p.label(i, label));
                        let var_b = variance(oth_p, &oth_p.label(j, label));
                        let delta = vb - va;
                        let reference_sigma = var_a.map(f64::sqrt);
                        ParameterDifference {
                            label: label.to_string(),
                            reference: va,
                            other: vb,
                            delta,
                            reference_sigma,
                            other_sigma: var_b.map(f64::sqrt),
                            delta_over_reference_sigma: ratio(delta, reference_sigma),
                            test: scale.scalar_test(delta, var_a, var_b),
                        }
                    })
                    .collect(),
            },
            (Err(e), _) | (_, Err(e)) => ParameterComparison::Unavailable {
                reason: e.to_string(),
            },
        }
    };
    let projection = projection_comparison(i, j, reference, other, options, ref_run, oth_run);
    CameraDifference {
        id: ref_run.camera_ids[i].clone(),
        parameters,
        projection,
    }
}

fn projection_comparison(
    i: usize,
    j: usize,
    reference: Side<'_, '_>,
    other: Side<'_, '_>,
    options: &CompareRunsOptions,
    ref_run: &ComparedRun<'_>,
    oth_run: &ComparedRun<'_>,
) -> ProjectionComparison {
    let size = ref_run.image_sizes[i];
    let grid = match ImageGrid::with_columns(size, options.grid_columns) {
        Ok(grid) => grid,
        Err(e) => {
            return ProjectionComparison::Unavailable {
                reason: e.to_string(),
            };
        }
    };
    let covered: Vec<[f64; 2]> = observed(ref_run, i).chain(observed(oth_run, j)).collect();
    let fit = ProjectionDifferenceOptions::new(options.compensation, FitRegion::Covered(covered));
    match projection_difference(&reference.1.cameras[i], &other.1.cameras[j], &grid, &fit) {
        Ok(d) => ProjectionComparison::Compared {
            compensation: options.compensation,
            rotation_deg: d.rotation_angle().to_degrees(),
            fit_nodes: d.fit_nodes,
            valid_nodes: d.valid_nodes,
            rms_fit_px: d.rms_fit_px,
            max_fit_px: d.max_fit_px,
            rms_valid_px: d.rms_valid_px,
            max_valid_px: d.max_valid_px,
        },
        Err(e) => ProjectionComparison::Unavailable {
            reason: e.to_string(),
        },
    }
}

/// The observed pixels of camera `camera`'s features in `run`.
fn observed<'r>(run: &'r ComparedRun<'_>, camera: usize) -> impl Iterator<Item = [f64; 2]> + 'r {
    run.export
        .diagnostics()
        .per_feature_residuals
        .target
        .iter()
        .filter(move |r| r.camera == camera)
        .map(|r| r.observed_px)
}

/// The stated variance of the parameter `label`; `None` without a
/// covariance or when the run did not estimate it.
fn variance(prepared: &Prepared, label: &str) -> Option<f64> {
    let c = prepared.covariance.as_ref()?;
    let i = c.index(label)?;
    let v = *c.covariance.get(i)?.get(i)?;
    (v.is_finite() && v >= 0.0).then_some(v)
}

/// The 6×6 covariance of the pose whose labels are `labels`; `None` when
/// any is missing.
fn pose_covariance(prepared: &Prepared, labels: &[String; 6]) -> Option<Matrix6<f64>> {
    let c = prepared.covariance.as_ref()?;
    let index: Vec<usize> = labels
        .iter()
        .map(|l| c.index(l))
        .collect::<Option<Vec<_>>>()?;
    let m = Matrix6::from_fn(|r, s| c.covariance[index[r]][index[s]]);
    m.iter().all(|v| v.is_finite()).then_some(m)
}

fn ratio(delta: f64, sigma: Option<f64>) -> Option<f64> {
    sigma.filter(|s| *s > 0.0).map(|s| delta / s)
}

const POSE_COORDINATES: [&str; 6] = ["rx", "ry", "rz", "tx", "ty", "tz"];

fn pose_labels(name: &str) -> [String; 6] {
    POSE_COORDINATES.map(|c| format!("{name}.{c}"))
}

/// The difference of two poses.
fn pose_difference(
    label: String,
    reference: &Iso3,
    other: &Iso3,
    covariances: (Option<Matrix6<f64>>, Option<Matrix6<f64>>),
    gauge_differs: bool,
    scale: &Scale<'_>,
) -> PoseDifference {
    let relative: UnitQuaternion<f64> = reference.rotation.inverse() * other.rotation;
    let omega = relative.scaled_axis();
    let dt = other.translation.vector - reference.translation.vector;
    let v = reference.rotation.inverse_transform_vector(&dt);
    let tangent = Vector6::new(omega.x, omega.y, omega.z, v.x, v.y, v.z);
    let delta_over_reference_sigma = covariances
        .0
        .as_ref()
        .and_then(|c| mahalanobis_squared(c, &tangent))
        .map(f64::sqrt);
    let test = if gauge_differs && scale.relation_bar().is_none() {
        PoseTest::NotAvailable {
            reason: NoTest::GaugeDiffers,
        }
    } else {
        scale.pose_test(&tangent, covariances.0, covariances.1)
    };
    PoseDifference {
        label,
        rotation_deg: relative.angle().to_degrees(),
        translation_m: [dt.x, dt.y, dt.z],
        translation_norm_m: dt.norm(),
        tangent: [
            tangent[0], tangent[1], tangent[2], tangent[3], tangent[4], tangent[5],
        ],
        delta_over_reference_sigma,
        test,
    }
}

/// The reference run's reference camera and its index in the other run,
/// when both have it; and whether the other run's rig frame is another
/// camera's.
fn reference_pair(
    pairs: &[(usize, usize)],
    reference: Side<'_, '_>,
    other: Side<'_, '_>,
) -> Option<(usize, usize, bool)> {
    let r = reference.1.reference_camera?;
    let &(_, r_other) = pairs.iter().find(|&&(i, _)| i == r)?;
    let gauge_differs = other.1.reference_camera != Some(r_other);
    Some((r, r_other, gauge_differs))
}

fn rig_differences(
    pairs: &[(usize, usize)],
    reference: Side<'_, '_>,
    other: Side<'_, '_>,
    scale: &Scale<'_>,
) -> Vec<PoseDifference> {
    let (ref_run, ref_p) = reference;
    let (_, oth_p) = other;
    if !ref_p.multi_camera() {
        return Vec::new();
    }
    let Some((r, r_other, gauge_differs)) = reference_pair(pairs, reference, other) else {
        return Vec::new();
    };
    pairs
        .iter()
        .filter(|&&(i, _)| i != r)
        .map(|&(i, j)| {
            // `cam_i_se3_cam_r = cam_i_se3_rig · cam_r_se3_rig⁻¹`.
            let a = ref_p.cam_se3_rig[i] * ref_p.cam_se3_rig[r].inverse();
            let b = oth_p.cam_se3_rig[j] * oth_p.cam_se3_rig[r_other].inverse();
            let cov_a = pose_covariance(ref_p, &pose_labels(&ref_p.label(i, "cam_se3_rig")));
            let cov_b = if gauge_differs {
                None
            } else {
                pose_covariance(oth_p, &pose_labels(&oth_p.label(j, "cam_se3_rig")))
            };
            let label = format!("{}_se3_{}", ref_run.camera_ids[i], ref_run.camera_ids[r]);
            pose_difference(label, &a, &b, (cov_a, cov_b), gauge_differs, scale)
        })
        .collect()
}

fn handeye_difference(
    pairs: &[(usize, usize)],
    reference: Side<'_, '_>,
    other: Side<'_, '_>,
    scale: &Scale<'_>,
) -> Option<PoseDifference> {
    let (ref_run, ref_p) = reference;
    let (oth_run, oth_p) = other;
    let (r, r_other, gauge_differs) = reference_pair(pairs, reference, other)?;
    let ha = ref_run.export.handeye()?;
    let hb = oth_run.export.handeye()?;
    let id = &ref_run.camera_ids[r];
    // Each hand-eye expressed for camera r: eye-in-hand `gripper_se3_cam_r
    // = gripper_se3_rig · cam_r_se3_rig⁻¹`, eye-to-hand `cam_r_se3_base =
    // cam_r_se3_rig · rig_se3_base`.
    let (label, a, b, name_a, name_b) = match (ha, hb) {
        (Handeye::EyeInHand { gripper_se3_rig: a }, Handeye::EyeInHand { gripper_se3_rig: b }) => (
            format!("gripper_se3_{id}"),
            a * ref_p.cam_se3_rig[r].inverse(),
            b * oth_p.cam_se3_rig[r_other].inverse(),
            handeye_name(ref_p, "gripper_se3_rig", "gripper_se3_cam"),
            handeye_name(oth_p, "gripper_se3_rig", "gripper_se3_cam"),
        ),
        (Handeye::EyeToHand { rig_se3_base: a }, Handeye::EyeToHand { rig_se3_base: b }) => (
            format!("{id}_se3_base"),
            ref_p.cam_se3_rig[r] * a,
            oth_p.cam_se3_rig[r_other] * b,
            handeye_name(ref_p, "rig_se3_base", "cam_se3_base"),
            handeye_name(oth_p, "rig_se3_base", "cam_se3_base"),
        ),
        _ => return None,
    };
    let cov_a = pose_covariance(ref_p, &pose_labels(name_a));
    let cov_b = if gauge_differs {
        None
    } else {
        pose_covariance(oth_p, &pose_labels(name_b))
    };
    Some(pose_difference(
        label,
        &a,
        &b,
        (cov_a, cov_b),
        gauge_differs,
        scale,
    ))
}

/// The hand-eye's covariance name: the rig's, or the camera's for one
/// camera (as its export names it).
fn handeye_name(prepared: &Prepared, rig: &'static str, camera: &'static str) -> &'static str {
    if prepared.multi_camera() { rig } else { camera }
}

fn plane_differences(
    pairs: &[(usize, usize)],
    reference: Side<'_, '_>,
    other: Side<'_, '_>,
    scale: &Scale<'_>,
) -> Vec<PlaneDifference> {
    let (ref_run, ref_p) = reference;
    let (oth_run, oth_p) = other;
    let planes_a = ref_run.export.laser_planes_cam();
    let planes_b = oth_run.export.laser_planes_cam();
    pairs
        .iter()
        .filter_map(|&(i, j)| {
            let (pa, pb) = (planes_a.get(i)?, planes_b.get(j)?);
            let (na, nb) = (pa.normal().into_inner(), pb.normal().into_inner());
            // A plane is `n · p + d = 0` up to the sign of `(n, d)`: match
            // the other's sign to the reference's normal.
            let sign = if na.dot(&nb) < 0.0 { -1.0 } else { 1.0 };
            let (nb, db) = (nb * sign, pb.distance() * sign);
            let da = pa.distance();
            let label_a =
                vision_calibration_core::CalibratedRig::plane_labels(i, ref_p.multi_camera());
            let label_b =
                vision_calibration_core::CalibratedRig::plane_labels(j, oth_p.multi_camera());
            let var_a = variance(ref_p, &label_a[3]);
            let var_b = variance(oth_p, &label_b[3]);
            let delta = db - da;
            Some(PlaneDifference {
                camera: ref_run.camera_ids[i].clone(),
                normal_angle_deg: angle_between(&na, &nb).to_degrees(),
                distance_reference_m: da,
                distance_other_m: db,
                distance_delta_m: delta,
                distance_over_reference_sigma: ratio(delta, var_a.map(f64::sqrt)),
                distance_test: scale.scalar_test(delta, var_a, var_b),
            })
        })
        .collect()
}

/// The angle between two unit vectors, radians, accurate near zero.
fn angle_between(a: &Vector3<f64>, b: &Vector3<f64>) -> f64 {
    a.cross(b).norm().atan2(a.dot(b))
}

/// Where two configurations differ, as paths into their JSON.
fn config_changes(a: &CalibrationConfig, b: &CalibrationConfig) -> Vec<ConfigChange> {
    let to_json =
        |c: &CalibrationConfig| serde_json::to_value(c).unwrap_or(serde_json::Value::Null);
    let mut out = Vec::new();
    json_diff("", Some(&to_json(a)), Some(&to_json(b)), &mut out);
    out
}

fn json_diff(
    path: &str,
    a: Option<&serde_json::Value>,
    b: Option<&serde_json::Value>,
    out: &mut Vec<ConfigChange>,
) {
    use serde_json::Value;
    match (a, b) {
        (Some(Value::Object(ma)), Some(Value::Object(mb))) => {
            let keys: BTreeSet<&String> = ma.keys().chain(mb.keys()).collect();
            for key in keys {
                let child = if path.is_empty() {
                    key.clone()
                } else {
                    format!("{path}.{key}")
                };
                json_diff(&child, ma.get(key), mb.get(key), out);
            }
        }
        (Some(Value::Array(va)), Some(Value::Array(vb))) if va.len() == vb.len() => {
            for (k, (x, y)) in va.iter().zip(vb).enumerate() {
                json_diff(&format!("{path}[{k}]"), Some(x), Some(y), out);
            }
        }
        (a, b) if a == b => {}
        (a, b) => out.push(ConfigChange {
            path: path.to_string(),
            reference: a.map(Value::to_string),
            other: b.map(Value::to_string),
        }),
    }
}

/// The upper tail `P(X > x)` of a χ² variable of `dof` degrees of freedom:
/// the regularized upper incomplete gamma `Q(dof/2, x/2)`.
pub(crate) fn chi2_survival(x: f64, dof: usize) -> f64 {
    if !x.is_finite() {
        return if x > 0.0 { 0.0 } else { 1.0 };
    }
    if x <= 0.0 {
        return 1.0;
    }
    gamma_q(dof as f64 / 2.0, x / 2.0)
}

/// The regularized upper incomplete gamma function `Q(a, x)`, `a > 0`,
/// `x ≥ 0`: the series below `a + 1`, Lentz's continued fraction above
/// (Press et al., *Numerical Recipes*, §6.2), each to `1e-15`.
fn gamma_q(a: f64, x: f64) -> f64 {
    const EPS: f64 = 1e-15;
    const MAX_TERMS: usize = 1000;
    const TINY: f64 = 1e-300;
    let log_prefix = a * x.ln() - x - ln_gamma(a);
    if x < a + 1.0 {
        let (mut term, mut sum, mut ap) = (1.0 / a, 1.0 / a, a);
        for _ in 0..MAX_TERMS {
            ap += 1.0;
            term *= x / ap;
            sum += term;
            if term.abs() < sum.abs() * EPS {
                break;
            }
        }
        (1.0 - sum * log_prefix.exp()).clamp(0.0, 1.0)
    } else {
        let mut b = x + 1.0 - a;
        let mut c = 1.0 / TINY;
        let mut d = 1.0 / b;
        let mut h = d;
        for i in 1..=MAX_TERMS {
            let an = -(i as f64) * (i as f64 - a);
            b += 2.0;
            d = an * d + b;
            if d.abs() < TINY {
                d = TINY;
            }
            c = b + an / c;
            if c.abs() < TINY {
                c = TINY;
            }
            d = 1.0 / d;
            let step = d * c;
            h *= step;
            if (step - 1.0).abs() < EPS {
                break;
            }
        }
        (log_prefix.exp() * h).clamp(0.0, 1.0)
    }
}

/// `ln Γ(x)`, `x > 0`, by the Lanczos approximation (`g = 7`, 9 terms;
/// relative error below `1e-15`).
fn ln_gamma(x: f64) -> f64 {
    const G: f64 = 7.0;
    const C: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        // Reflection: Γ(x) Γ(1 − x) = π / sin(πx).
        let pi = std::f64::consts::PI;
        return (pi / (pi * x).sin()).ln() - ln_gamma(1.0 - x);
    }
    let x = x - 1.0;
    let mut sum = C[0];
    for (k, c) in C.iter().enumerate().skip(1) {
        sum += c / (x + k as f64);
    }
    let t = x + G + 0.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + sum.ln()
}

#[cfg(test)]
mod tests;
