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

use nalgebra::{DMatrix, DVector, UnitQuaternion, Vector3, Vector6};
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

mod stats;
#[cfg(test)]
use stats::chi2_survival;
use stats::{Scale, mahalanobis_squared};

/// A cross-check that a camera pose is the identity: the reference camera
/// of a rig is held at exactly the identity, so any rounding is far below.
const IDENTITY_TOLERANCE: f64 = 1e-12;

/// One run as [`compare_runs`] reads it: its export, the id and image size
/// of each of its cameras, and optionally the configuration it was
/// calibrated with and how its observations relate to the reference's.
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
    /// The configuration the run was calibrated with; when both runs of a
    /// difference have one, the difference lists the configuration changes.
    pub config: Option<CalibrationConfig>,
    /// How the run's observations relate to the reference's (the first
    /// run's): it decides which test a difference gets, so it is stated,
    /// never guessed. `None` is [`DataRelation::Unknown`]. The reference
    /// itself has none.
    pub relation: Option<DataRelation>,
}

impl<'a> ComparedRun<'a> {
    /// The run `name` of `export`, its cameras named `camera_ids` and of
    /// `image_sizes`, both in the export's camera order; no configuration,
    /// the relation unknown.
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
            relation: None,
        }
    }

    /// The same run with the configuration it was calibrated with.
    #[must_use]
    pub fn with_config(mut self, config: CalibrationConfig) -> Self {
        self.config = Some(config);
        self
    }

    /// The same run with the relation of its observations to the
    /// reference's.
    #[must_use]
    pub fn with_relation(mut self, relation: DataRelation) -> Self {
        self.relation = Some(relation);
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
            .field("relation", &self.relation)
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

/// A stated relation the runs contradict, and why: the difference is then
/// treated as [`DataRelation::Overlapping`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[non_exhaustive]
pub struct RelationRefusal {
    /// The relation as stated.
    pub stated: DataRelation,
    /// Why it does not hold.
    pub reason: RefusalReason,
}

/// Why a stated relation does not hold.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum RefusalReason {
    /// Stated independent, the runs share bit-identical features (the same
    /// target point at the same observed pixel): they read the same images.
    SharedFeatures,
    /// Stated nested, the smaller run has a feature the larger lacks, or
    /// the two have the same features.
    NotASubset,
    /// Stated nested, the two configurations differ.
    ConfigurationsDiffer,
}

/// The options of [`compare_runs`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[serde(default, deny_unknown_fields)]
#[non_exhaustive]
pub struct CompareRunsOptions {
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
            compensation: Compensation::Rotation,
            grid_columns: 64,
            thresholds: QualityThresholds::default(),
        }
    }
}

impl CompareRunsOptions {
    /// The default options: rotation-compensated projection differences on
    /// a 64-column grid, the default thresholds.
    pub fn new() -> Self {
        Self::default()
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
        /// Its a-posteriori variance factor, when it comes from one
        /// residual family with an estimated noise level; what a nested
        /// test needs.
        variance_factor: Option<f64>,
    },
    /// It has none, for the reason given.
    Unavailable {
        /// Why.
        reason: UncertaintyUnavailable,
    },
}

/// One camera's fit in one run. The residual statistics are in-sample:
/// they compare like with like only between runs of the same observations
/// ([`DataRelation::Identical`], [`DataRelation::SameData`]); a run that
/// left out its worst features always has a lower residual.
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
    /// How its observations relate to the reference's: the stated relation,
    /// or [`DataRelation::Overlapping`] when the runs contradict it.
    pub relation: DataRelation,
    /// The stated relation the runs contradict, and why; `None` when the
    /// stated relation was used.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub relation_refused: Option<RelationRefusal>,
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
    /// What the two runs' observations share, by image content and
    /// feature: the evidence of the relation. `None` unless both runs
    /// recorded the content of the images they used (project runs).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data: Option<DataOverlap>,
    /// The exclusions one run made and the other did not. `None` unless
    /// both are runs a project calibrated.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exclusion_changes: Option<ExclusionChanges>,
}

/// What two runs' observations share. A feature is one target point (or
/// laser pixel) of one camera in one image, identified by the image's
/// content and its observed pixel: two runs share it only when they read
/// the same image and their detections agree.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[non_exhaustive]
pub struct DataOverlap {
    /// Images both runs used.
    pub shared_images: usize,
    /// Images only the reference used.
    pub reference_only_images: usize,
    /// Images only the other run used.
    pub other_only_images: usize,
    /// Features both runs used.
    pub shared_features: usize,
    /// Features only the reference used.
    pub reference_only_features: usize,
    /// Features only the other run used.
    pub other_only_features: usize,
}

/// The exclusions that differ between two project runs, notes left out.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[non_exhaustive]
pub struct ExclusionChanges {
    /// What only the reference left out.
    pub reference_only: crate::project::Exclusions,
    /// What only the other run left out.
    pub other_only: crate::project::Exclusions,
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
    /// Both runs use one model.
    SameModel {
        /// Each parameter, in the solver's order, with its own test.
        parameters: Vec<ParameterDifference>,
        /// One test of every parameter both runs estimated, over their full
        /// covariance: a `χ²` of as many degrees of freedom, for independent
        /// captures. Read it before the per-parameter tests: of eight
        /// parameters tested at 5 % each, one rejects by chance in about a
        /// quarter of comparisons. Not offered for nested runs
        /// ([`NoTest::NestedJoint`]).
        joint: JointTest,
    },
    /// The models differ, so a parameter of one is not a parameter of the
    /// other; the projection difference compares them.
    DifferentModels,
    /// A camera whose parameters the solver does not name: a general sensor
    /// homography.
    Unparameterized,
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
    /// Nested observations under one configuration (the Hausman form):
    /// `σ_Δ² = c · k_larger · (a_smaller − a_larger)`, `a` each run's
    /// variance per unit noise (its stated variance over its view-scatter
    /// factor and variance factor), `c` the smaller run's variance factor
    /// and `k_larger` the larger run's view-scatter factor: both on one
    /// noise level. `z = Δ / σ_Δ` is standard normal when the left-out
    /// observations fit the model as the others do and were not chosen by
    /// their residuals.
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
    /// Nested runs whose covariance combines residual families (laser and
    /// reprojection), robot-pose priors or per-camera noise levels: the
    /// nested form's premise, one noise level for every residual, fails,
    /// and no test of such runs has been validated.
    MixedResiduals,
    /// Nested runs of which one export states no variance factor (written
    /// before it was recorded, or its covariance was derived from several
    /// solves): their noise levels cannot be put on one scale.
    NoVarianceFactor,
    /// Nested runs whose variance of the difference is not positive (the
    /// smaller run's information is not the smaller).
    NestedVarianceNotPositive,
    /// The joint test of a camera's parameters between nested runs, which
    /// is not offered: it rejects well above its level when the left-out
    /// data are a small share (measured: 7.7 % with 1 of 8 views left out,
    /// 15.7 % with 6 of about 2000 features, at the 5 % level). Each
    /// parameter's own nested test remains.
    NestedJoint,
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
    pub test: JointTest,
}

/// The test of several quantities at once (a pose's tangent, a camera's
/// parameters), their difference `δ` against the covariance of the
/// difference: `χ² = δᵀ Σ_Δ⁻¹ δ`, `Σ_Δ` as each [`DifferenceTest`] forms
/// a variance.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[serde(tag = "kind", rename_all = "snake_case")]
#[non_exhaustive]
pub enum JointTest {
    /// Independent captures: `Σ_Δ = Σ_ref + Σ_other`.
    Independent {
        /// `χ²`.
        chi2: f64,
        /// Its degrees of freedom.
        dof: usize,
        /// The upper-tail p-value of `χ²`.
        p_value: f64,
    },
    /// Nested observations under one configuration (the Hausman form, on
    /// one noise level as [`DifferenceTest::Nested`]), through a generalized
    /// inverse over the directions whose variance grows without the
    /// left-out data, `dof` their number. Offered for poses; a camera's
    /// parameters have no nested joint test ([`NoTest::NestedJoint`]).
    Nested {
        /// `χ²`.
        chi2: f64,
        /// Its degrees of freedom.
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
        reason: ProjectionUnavailable,
    },
}

/// Why a projection difference was not computed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ProjectionUnavailable {
    /// The image or grid is empty.
    InvalidImage,
    /// Fewer than three nodes of the covered region have a value.
    TooFewNodes,
    /// A sensor homography does not invert, or the rotation fit is
    /// singular.
    Singular,
    /// A camera's projection is not central.
    NotCentral,
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
///   and a [`DifferenceTest`], and one [`JointTest`] of every parameter both
///   estimated;
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
/// Each run's relation to the reference is stated on the run
/// ([`ComparedRun::with_relation`]); it decides the test:
///
/// | Relation | Variance of `Δ` |
/// |---|---|
/// | [`Independent`](DataRelation::Independent) | `σ_ref² + σ_other²` |
/// | [`Nested`](DataRelation::Nested) | `c · k_larger · (a_smaller − a_larger)`, both runs on the smaller run's noise level (the Hausman form) |
/// | any other | none, with the reason |
///
/// The nested form holds because, when the left-out observations fit the
/// model as the rest do, the estimate from all observations is uncorrelated
/// with its difference from the estimate without them. It needs both runs'
/// covariances on one noise level, so it needs each export's variance
/// factor, and a covariance from one residual family without priors
/// ([`NoTest::MixedResiduals`] otherwise). It is **not** valid when the
/// left-out observations were chosen by their residuals (the worst features
/// or views of a fit): a test of the remaining fit against the full one
/// then rejects far more often than its level even when the model is
/// right. Runs of the same observations under different configurations have
/// no test: their errors are almost fully correlated by an amount neither
/// export states, so a small difference does not mean agreement and a large
/// one is no significance.
///
/// A stated relation the exports contradict is not used: a stated
/// independent pair that shares bit-identical features (the same target
/// point at the same observed pixel), and a stated nested pair whose
/// smaller run is not a strict subset of the larger, or whose given
/// configurations differ, are compared as
/// [`Overlapping`](DataRelation::Overlapping), with the refusal in
/// [`RunDifference::relation_refused`].
///
/// # Errors
///
/// [`Error::InvalidInput`] for fewer than two runs, runs of different
/// workflows, camera ids or image sizes not one per camera, a camera id
/// repeated in a run, a run sharing no camera id with the reference, a
/// shared camera whose image sizes differ, a relation stated on the
/// reference, or zero grid columns. Errors of a run's quality assessment
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
    let differences = (1..runs.len())
        .map(|i| {
            run_difference(
                i,
                (&runs[0], &prepared[0]),
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
    /// Each feature: camera id, target (0) or laser (1), the target point
    /// and the observed pixel, as exact bits.
    features: BTreeSet<(String, u8, [u64; 3], [u64; 2])>,
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
    if runs[0].relation.is_some() {
        return Err(Error::invalid_input(format!(
            "run {} is the reference; a relation is stated on the runs compared with it",
            runs[0].name
        )));
    }
    let reference_kind = runs[0].export.kind();
    let mut prepared = Vec::with_capacity(runs.len());
    for run in runs {
        if run.export.kind() != reference_kind {
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
            features: features(run),
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

/// The run's features as exact bits, by camera id.
fn features(run: &ComparedRun<'_>) -> BTreeSet<(String, u8, [u64; 3], [u64; 2])> {
    let id = |c: usize| run.camera_ids.get(c).cloned().unwrap_or_default();
    let residuals = &run.export.diagnostics().per_feature_residuals;
    residuals
        .target
        .iter()
        .map(|r| {
            (
                id(r.camera),
                0,
                r.target_xyz_m.map(f64::to_bits),
                r.observed_px.map(f64::to_bits),
            )
        })
        .chain(
            residuals
                .laser
                .iter()
                .map(|r| (id(r.camera), 1, [0; 3], r.observed_px.map(f64::to_bits))),
        )
        .collect()
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
            variance_factor: c.variance_factor,
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
        kind: run.export.kind(),
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
    let config_changes = match (&ref_run.config, &oth_run.config) {
        (Some(a), Some(b)) => Some(config_changes(a, b)),
        _ => None,
    };
    let stated = oth_run.relation.unwrap_or_default();
    let refusal = refusal(stated, ref_p, oth_p, config_changes.as_deref());
    let relation = if refusal.is_some() {
        DataRelation::Overlapping
    } else {
        stated
    };
    let scale = Scale::new(
        relation,
        ref_p.covariance.as_ref(),
        oth_p.covariance.as_ref(),
    );
    let cameras: Vec<CameraDifference> = pairs
        .iter()
        .map(|&(i, j)| camera_difference(i, j, reference, other, &scale, options))
        .collect();
    let model_changes = pairs
        .iter()
        .filter(|&&(i, j)| ref_p.cameras[i].spec() != oth_p.cameras[j].spec())
        .map(|&(i, _)| ref_run.camera_ids[i].clone())
        .collect();
    let rig = rig_differences(&pairs, reference, other, &scale);
    let handeye = handeye_difference(&pairs, reference, other, &scale);
    let laser_planes = plane_differences(&pairs, reference, other, &scale);
    RunDifference {
        run: index,
        relation,
        relation_refused: refusal.map(|reason| RelationRefusal { stated, reason }),
        model_changes,
        config_changes,
        cameras,
        rig,
        handeye,
        laser_planes,
        unmatched_cameras: unmatched,
        data: None,
        exclusion_changes: None,
    }
}

/// Why the exports contradict the stated relation, if they do.
fn refusal(
    stated: DataRelation,
    reference: &Prepared,
    other: &Prepared,
    config_changes: Option<&[ConfigChange]>,
) -> Option<RefusalReason> {
    match stated {
        DataRelation::Independent => (!reference.features.is_disjoint(&other.features))
            .then_some(RefusalReason::SharedFeatures),
        DataRelation::Nested { smaller } => {
            if config_changes.is_some_and(|c| !c.is_empty()) {
                return Some(RefusalReason::ConfigurationsDiffer);
            }
            let (small, large) = match smaller {
                RunSide::Reference => (&reference.features, &other.features),
                RunSide::Other => (&other.features, &reference.features),
            };
            (!(small.is_subset(large) && small.len() < large.len()))
                .then_some(RefusalReason::NotASubset)
        }
        _ => None,
    }
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
    let oth_p = other.1;
    let (a, b) = (&ref_p.cameras[i], &oth_p.cameras[j]);
    let parameters = if a.spec() != b.spec() {
        ParameterComparison::DifferentModels
    } else {
        match (camera_parameters(a), camera_parameters(b)) {
            (Ok(pa), Ok(pb)) => {
                let mut joint_labels = Vec::new();
                let parameters = pa
                    .iter()
                    .zip(&pb)
                    .map(|(&(label, va), &(_, vb))| {
                        let (la, lb) = (ref_p.label(i, label), oth_p.label(j, label));
                        let var_a = variance(ref_p, &la);
                        let var_b = variance(oth_p, &lb);
                        let delta = vb - va;
                        if var_a.is_some() && var_b.is_some() {
                            joint_labels.push((la, lb, delta));
                        }
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
                    .collect();
                let joint = joint_camera_test(&joint_labels, ref_p, oth_p, scale);
                ParameterComparison::SameModel { parameters, joint }
            }
            _ => ParameterComparison::Unparameterized,
        }
    };
    let projection = projection_comparison(i, j, reference, other, options);
    CameraDifference {
        id: ref_run.camera_ids[i].clone(),
        parameters,
        projection,
    }
}

/// The joint test of the parameters both runs estimated, `(reference label,
/// other label, delta)` each.
fn joint_camera_test(
    labels: &[(String, String, f64)],
    reference: &Prepared,
    other: &Prepared,
    scale: &Scale<'_>,
) -> JointTest {
    if labels.is_empty() {
        return scale.joint_test(&DVector::zeros(0), None, None, false);
    }
    let a: Vec<&str> = labels.iter().map(|(l, _, _)| l.as_str()).collect();
    let b: Vec<&str> = labels.iter().map(|(_, l, _)| l.as_str()).collect();
    let delta = DVector::from_iterator(labels.len(), labels.iter().map(|(_, _, d)| *d));
    scale.joint_test(
        &delta,
        block_covariance(reference, &a),
        block_covariance(other, &b),
        false,
    )
}

fn projection_comparison(
    i: usize,
    j: usize,
    reference: Side<'_, '_>,
    other: Side<'_, '_>,
    options: &CompareRunsOptions,
) -> ProjectionComparison {
    let size = reference.0.image_sizes[i];
    let Ok(grid) = ImageGrid::with_columns(size, options.grid_columns) else {
        return ProjectionComparison::Unavailable {
            reason: ProjectionUnavailable::InvalidImage,
        };
    };
    let covered: Vec<[f64; 2]> = observed(reference.0, i)
        .chain(observed(other.0, j))
        .collect();
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
            reason: match e {
                vision_calibration_core::Error::InsufficientData { .. } => {
                    ProjectionUnavailable::TooFewNodes
                }
                vision_calibration_core::Error::InvalidInput { .. } => {
                    ProjectionUnavailable::NotCentral
                }
                _ => ProjectionUnavailable::Singular,
            },
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

/// The covariance block of `labels`; `None` when any is missing.
fn block_covariance(prepared: &Prepared, labels: &[&str]) -> Option<DMatrix<f64>> {
    let c = prepared.covariance.as_ref()?;
    let index: Vec<usize> = labels
        .iter()
        .map(|l| c.index(l))
        .collect::<Option<Vec<_>>>()?;
    let m = DMatrix::from_fn(index.len(), index.len(), |r, s| {
        c.covariance[index[r]][index[s]]
    });
    m.iter().all(|v| v.is_finite()).then_some(m)
}

fn ratio(delta: f64, sigma: Option<f64>) -> Option<f64> {
    sigma.filter(|s| *s > 0.0).map(|s| delta / s)
}

const POSE_COORDINATES: [&str; 6] = ["rx", "ry", "rz", "tx", "ty", "tz"];

fn pose_labels(name: &str) -> [String; 6] {
    POSE_COORDINATES.map(|c| format!("{name}.{c}"))
}

fn pose_covariance(prepared: &Prepared, name: &str) -> Option<DMatrix<f64>> {
    let labels = pose_labels(name);
    let refs: Vec<&str> = labels.iter().map(String::as_str).collect();
    block_covariance(prepared, &refs)
}

/// The difference of two poses.
fn pose_difference(
    label: String,
    reference: &Iso3,
    other: &Iso3,
    covariances: (Option<DMatrix<f64>>, Option<DMatrix<f64>>),
    gauge_differs: bool,
    scale: &Scale<'_>,
) -> PoseDifference {
    let relative: UnitQuaternion<f64> = reference.rotation.inverse() * other.rotation;
    let omega = relative.scaled_axis();
    let dt = other.translation.vector - reference.translation.vector;
    let v = reference.rotation.inverse_transform_vector(&dt);
    let tangent = Vector6::new(omega.x, omega.y, omega.z, v.x, v.y, v.z);
    let delta = DVector::from_column_slice(tangent.as_slice());
    let delta_over_reference_sigma = covariances
        .0
        .as_ref()
        .and_then(|c| mahalanobis_squared(c, &delta))
        .map(f64::sqrt);
    let test = if gauge_differs && scale.relation_bar().is_none() {
        JointTest::NotAvailable {
            reason: NoTest::GaugeDiffers,
        }
    } else {
        scale.joint_test(&delta, covariances.0, covariances.1, true)
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
            let cov_a = pose_covariance(ref_p, &ref_p.label(i, "cam_se3_rig"));
            let cov_b = if gauge_differs {
                None
            } else {
                pose_covariance(oth_p, &oth_p.label(j, "cam_se3_rig"))
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
    let cov_a = pose_covariance(ref_p, name_a);
    let cov_b = if gauge_differs {
        None
    } else {
        pose_covariance(oth_p, name_b)
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

#[cfg(test)]
mod tests;
