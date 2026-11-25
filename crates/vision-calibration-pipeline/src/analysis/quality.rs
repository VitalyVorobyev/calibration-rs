//! A verdict on a calibration result: whether its data determined the
//! parameters well enough to be used.

use std::collections::BTreeMap;

use nalgebra::{DMatrix, DVector, SymmetricEigen};
use serde::{Deserialize, Serialize};
use vision_calibration_core::{
    CameraParams, ImageManifest, NoiseLevel, ParameterCovariance, ResidualKind, ResidualNoise,
    TargetFeatureResidual, Uncertainty, UncertaintyUnavailable,
};
use vision_calibration_optim::camera_parameters;

use crate::Error;
use crate::common::{CalibratedCameras, split_camera_label};

/// Cells of the coverage grid, `[columns, rows]`: an image is split into
/// 8 × 6 equal cells, each an eighth of its width and a sixth of its height.
const COVERAGE_GRID: [usize; 2] = [8, 6];

/// The asymptotic efficiency of the median-based noise estimate relative to
/// the sample standard deviation under Gaussian noise: σ̂ from `1.4826 ·
/// median |r|` scatters as a sample variance over this share of the
/// residuals.
const MEDIAN_SIGMA_EFFICIENCY: f64 = 0.3675;

/// Thresholds of the quality checks.
///
/// Every default is a limit a well-planned calibration of a perspective
/// camera clears, and that a calibration with a known planning flaw (views
/// bunched in one part of the image, near-identical board poses) does not.
/// The expected feature noise has no default: it depends on the detector,
/// the target and the optics.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[serde(default, deny_unknown_fields)]
#[non_exhaustive]
pub struct QualityThresholds {
    /// Expected noise of one feature coordinate, σ₀, in pixels per axis
    /// (> 0). `None` (the default) leaves the variance-factor check not
    /// applicable: there is no universal value, since it depends on the
    /// detector, the target and the image quality. State it to test whether
    /// the residuals are consistent with it.
    pub expected_feature_noise_px: Option<f64>,
    /// Significance level α of the two-sided χ² variance-factor test, in
    /// `(0, 1)`. Default `0.01`: a result consistent with σ₀ fails the test
    /// once in a hundred calibrations.
    pub variance_test_alpha: f64,
    /// Largest acceptable projection σ anywhere in the image, in pixels
    /// (> 0); above it the check warns. Default `1.0`: where the views
    /// covered the image, σ stays a fraction of a pixel; it exceeds a pixel
    /// where the distortion is extrapolated, typically in corners the board
    /// never reached. The map is conservative for uses that re-estimate the
    /// camera's pose, so the check warns rather than fails.
    pub max_projection_sigma_px: f64,
    /// Share of the image's coverage cells, in `[0, 1]`, below which the
    /// coverage check warns. Default `0.5`: features in fewer than half of
    /// the cells leave the distortion to be extrapolated over the rest.
    pub coverage_warn_below: f64,
    /// Share of the image's coverage cells, in `[0, 1]` and at most
    /// `coverage_warn_below`, below which the coverage check fails. Default
    /// `0.3`: the views reached little more than one quadrant of the image,
    /// so the rest of it, corners included, is extrapolated.
    pub coverage_fail_below: f64,
    /// Largest acceptable mean normalized residual of one view, in units of
    /// the noise level σ̂ (> 0); a view above it is named and the check warns.
    /// Default `3.0`: the mean of `|r| / σ̂` over Gaussian inliers is about
    /// 1.25, so a view at 3 has residuals well over twice the typical ones,
    /// as from a misdetection, motion blur or a wrong robot pose.
    pub max_view_residual: f64,
    /// Largest acceptable absolute correlation of two parameters of one
    /// camera, in `(0, 1]`; a pair above it is named and the check warns.
    /// Default `0.99`: at that level the data determines the pair's sum or
    /// difference, not the two parameters, so each one's value is mostly the
    /// trade-off's choice. The two focal lengths `fx` and `fy` are left out:
    /// the board's distance scales both, so they correlate almost fully in
    /// every planar calibration, while their ratio is well determined.
    pub max_parameter_correlation: f64,
    /// Largest acceptable predicted scatter of the worst combination of the
    /// shared parameters over its stated σ, once the error's second-order
    /// term is counted (≥ 1); above it the covariance-linearity check warns,
    /// naming the combination. Default `1.2`: on synthetic stereo rigs and
    /// single cameras whose error the covariance describes the prediction
    /// stayed at or below 1.06, and every stereo capture predicted above 1.3
    /// scattered 1.4–3 times its σ along that combination.
    pub max_second_order_scatter: f64,
    /// Largest acceptable σ inflation `1/√(1 − R²)` of one camera parameter,
    /// R its multiple correlation with the camera's other parameters (≥ 1);
    /// above it the conditioning check warns, naming the parameter and the
    /// partners it trades off against. The other focal length is left out
    /// of a focal length's regressors, and the other radial coefficients out
    /// of a radial coefficient's. Default `7.0`: a pair at the correlation
    /// limit 0.99 inflates both by 7.09; well-determined sensor tilts
    /// measured 1.2–5, tilts the data cannot separate from the principal
    /// point or the tangential terms 14–2900.
    pub max_sigma_inflation: f64,
}

impl QualityThresholds {
    /// Construct default calibration quality-assessment thresholds.
    pub fn new() -> Self {
        Self::default()
    }
}

impl Default for QualityThresholds {
    fn default() -> Self {
        Self {
            expected_feature_noise_px: None,
            variance_test_alpha: 0.01,
            max_projection_sigma_px: 1.0,
            coverage_warn_below: 0.5,
            coverage_fail_below: 0.3,
            max_view_residual: 3.0,
            max_parameter_correlation: 0.99,
            max_second_order_scatter: 1.2,
            max_sigma_inflation: 7.0,
        }
    }
}

impl QualityThresholds {
    fn validate(&self) -> Result<(), Error> {
        let positive = |name: &str, value: f64| {
            if value.is_finite() && value > 0.0 {
                Ok(())
            } else {
                Err(Error::invalid_input(format!(
                    "quality threshold {name} must be finite and > 0, got {value}"
                )))
            }
        };
        if let Some(sigma) = self.expected_feature_noise_px {
            positive("expected_feature_noise_px", sigma)?;
        }
        positive("max_projection_sigma_px", self.max_projection_sigma_px)?;
        positive("max_view_residual", self.max_view_residual)?;
        if !(self.variance_test_alpha > 0.0 && self.variance_test_alpha < 1.0) {
            return Err(Error::invalid_input(format!(
                "quality threshold variance_test_alpha must be in (0, 1), got {}",
                self.variance_test_alpha
            )));
        }
        if !(self.max_second_order_scatter.is_finite() && self.max_second_order_scatter >= 1.0) {
            return Err(Error::invalid_input(format!(
                "quality threshold max_second_order_scatter must be finite and ≥ 1, got {}",
                self.max_second_order_scatter
            )));
        }
        if !(self.max_sigma_inflation.is_finite() && self.max_sigma_inflation >= 1.0) {
            return Err(Error::invalid_input(format!(
                "quality threshold max_sigma_inflation must be finite and ≥ 1, got {}",
                self.max_sigma_inflation
            )));
        }
        if !(self.max_parameter_correlation > 0.0 && self.max_parameter_correlation <= 1.0) {
            return Err(Error::invalid_input(format!(
                "quality threshold max_parameter_correlation must be in (0, 1], got {}",
                self.max_parameter_correlation
            )));
        }
        if !(0.0 <= self.coverage_fail_below
            && self.coverage_fail_below <= self.coverage_warn_below
            && self.coverage_warn_below <= 1.0)
        {
            return Err(Error::invalid_input(format!(
                "quality thresholds need 0 ≤ coverage_fail_below ≤ coverage_warn_below ≤ 1, \
                 got {} and {}",
                self.coverage_fail_below, self.coverage_warn_below
            )));
        }
        Ok(())
    }
}

/// The judgment of a result, or of one check.
///
/// Ordered by severity: `Pass < Warn < Fail`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum Verdict {
    /// Nothing found that limits the result.
    Pass,
    /// Usable, with a weakness worth knowing; the message says which.
    Warn,
    /// The data does not support the result.
    Fail,
}

/// The outcome of one check.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum CheckStatus {
    /// Assessed, and nothing found.
    Pass,
    /// Assessed, with a weakness.
    Warn,
    /// Assessed, and failed.
    Fail,
    /// Not assessed: the result lacks the check's inputs. The message says
    /// which.
    NotApplicable,
}

impl CheckStatus {
    /// The verdict of an assessed check; `None` when it was not applicable.
    pub fn verdict(self) -> Option<Verdict> {
        match self {
            Self::Pass => Some(Verdict::Pass),
            Self::Warn => Some(Verdict::Warn),
            Self::Fail => Some(Verdict::Fail),
            Self::NotApplicable => None,
        }
    }
}

/// What a check examines; [`assess`] describes each check.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum CheckKind {
    /// Whether the data determines every estimated parameter.
    Observability,
    /// Whether the final solve converged.
    Convergence,
    /// Whether the reprojection noise matches the stated feature noise.
    VarianceFactor,
    /// How precisely the camera places a ray, at worst, over its image.
    ProjectionSigma,
    /// How much of the image the observed features reached.
    FeatureCoverage,
    /// Views whose residuals are far above the noise level.
    ViewOutliers,
    /// Camera parameters the data cannot tell apart.
    ParameterCorrelation,
    /// Whether the covariance describes the joint error of the shared
    /// parameters, or a combination of them errs well beyond its σ at
    /// second order.
    CovarianceLinearity,
    /// Camera parameters whose σ the data inflates by trading them off
    /// against a combination of the camera's other parameters.
    ParameterConditioning,
}

/// The unit of a measured value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum Unit {
    /// Pixels.
    Pixel,
    /// A share in `[0, 1]`.
    Fraction,
    /// A dimensionless ratio, such as σ̂²/σ₀².
    Ratio,
    /// Multiples of the estimated noise level σ̂.
    Sigma,
    /// An absolute correlation coefficient in `[0, 1]`.
    Correlation,
}

/// A check's measured value and the limit that decided its status.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[non_exhaustive]
pub struct Measurement {
    /// The measured value; for a check over several items (views, parameter
    /// pairs), the worst one.
    pub value: f64,
    /// The limit it was compared to: the one it crossed, or, for a pass, the
    /// nearest one.
    pub threshold: f64,
    /// The unit of both.
    pub unit: Unit,
}

/// One check of a [`QualityReport`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[non_exhaustive]
pub struct QualityCheck {
    /// What the check examines.
    pub kind: CheckKind,
    /// The camera, for a check made per camera; `None` for one about the
    /// whole result.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub camera: Option<usize>,
    /// Its outcome.
    pub status: CheckStatus,
    /// The measured value; absent for a check not applicable, and for the
    /// observability and convergence checks, which read a status.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub measurement: Option<Measurement>,
    /// What the check names: unobservable or at-bound parameter labels,
    /// correlated parameter pairs (`fx/cx`), or outlier views (`view 3`).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub subjects: Vec<String>,
    /// One sentence on the outcome, or on why the check was not applicable.
    pub message: String,
}

impl QualityCheck {
    fn new(kind: CheckKind, camera: Option<usize>, status: CheckStatus, message: String) -> Self {
        Self {
            kind,
            camera,
            status,
            measurement: None,
            subjects: Vec::new(),
            message,
        }
    }

    fn not_applicable(kind: CheckKind, camera: Option<usize>, reason: impl Into<String>) -> Self {
        Self::new(kind, camera, CheckStatus::NotApplicable, reason.into())
    }

    fn measured(mut self, value: f64, threshold: f64, unit: Unit) -> Self {
        self.measurement = Some(Measurement {
            value,
            threshold,
            unit,
        });
        self
    }

    fn naming(mut self, subjects: Vec<String>) -> Self {
        self.subjects = subjects;
        self
    }
}

/// The verdict on a calibration result and the checks behind it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[non_exhaustive]
pub struct QualityReport {
    /// The worst verdict of the assessed checks; `warn` when no check could
    /// be assessed, since then nothing vouches for the result.
    pub verdict: Verdict,
    /// Every check, the result-wide ones first, then each camera's.
    pub checks: Vec<QualityCheck>,
}

/// Judge a calibration result.
///
/// Runs a fixed set of checks, each against a threshold of
/// [`QualityThresholds`]:
///
/// | Check | Reads | Pass / Warn / Fail |
/// |---|---|---|
/// | [`Observability`](CheckKind::Observability) | the uncertainty status | fails when the data leaves parameters undetermined (`rank_deficient`) or has no redundancy; warns when a parameter ended on a bound |
/// | [`Convergence`](CheckKind::Convergence) | the uncertainty status | fails when the final solve stopped on its iteration budget |
/// | [`VarianceFactor`](CheckKind::VarianceFactor) | the reprojection noise σ̂ | χ² test of σ̂²/σ₀² against the stated feature noise σ₀; fails above the range, warns below it |
/// | [`ProjectionSigma`](CheckKind::ProjectionSigma) | the projection-uncertainty map, per camera | warns when σ somewhere in the image exceeds the limit |
/// | [`FeatureCoverage`](CheckKind::FeatureCoverage) | the observed features, per camera | warns, then fails, as the share of the image the features reached falls |
/// | [`ViewOutliers`](CheckKind::ViewOutliers) | the per-feature residuals, per camera | warns, naming the views whose mean residual is several σ̂ |
/// | [`ParameterCorrelation`](CheckKind::ParameterCorrelation) | the covariance, per camera | warns, naming the camera parameters that trade off almost fully |
/// | [`CovarianceLinearity`](CheckKind::CovarianceLinearity) | the covariance's second-order error | warns when the worst combination of the shared parameters is predicted to scatter well beyond its σ, naming its parameters |
/// | [`ParameterConditioning`](CheckKind::ParameterConditioning) | the covariance, per camera | warns, naming the camera parameters whose σ the data inflates by trading them off against a combination of the camera's other parameters, with those partners and, for a sensor tilt, the cause |
///
/// A check whose inputs the result does not carry is
/// [`NotApplicable`](CheckStatus::NotApplicable), with the reason in its
/// message: no image size, no stated feature noise, no covariance, or
/// intrinsics the final solve held fixed, no second-order prediction. It is
/// never reported as a pass.
/// The report's [`Verdict`] is the worst verdict of the assessed checks.
///
/// `image_sizes` gives each camera's `[width, height]` in pixels, in camera
/// order, or is empty. A camera's size comes from the export's image manifest
/// when its frames are regions of a larger image (their common region size),
/// else from `image_sizes`; with neither, the projection and coverage checks
/// of that camera are not applicable.
///
/// The function reads the export only; it does not re-run any solve.
///
/// # Errors
///
/// [`Error::InvalidInput`] when a threshold is out of its range, a size is
/// zero, `image_sizes` is neither empty nor one entry per camera, or a
/// camera's size is ambiguous: its frames' regions differ, mix whole images
/// and regions, or differ from the size given.
pub fn assess<E: CalibratedCameras + ?Sized>(
    export: &E,
    image_sizes: &[[u32; 2]],
    thresholds: &QualityThresholds,
) -> Result<QualityReport, Error> {
    thresholds.validate()?;
    let cameras = export.camera_params();
    let sizes = resolve_image_sizes(
        export.diagnostics().image_manifest.as_ref(),
        cameras.len(),
        image_sizes,
    )?;
    let uncertainty = export.parameter_uncertainty();
    let target = &export.diagnostics().per_feature_residuals.target;

    let mut checks = vec![observability(uncertainty), convergence(uncertainty)];
    checks.extend(variance_factor(uncertainty, thresholds));
    checks.push(covariance_linearity(uncertainty, thresholds));
    for (camera, (params, size)) in cameras.iter().zip(&sizes).enumerate() {
        checks.push(projection_sigma(export, params, camera, *size, thresholds));
        checks.push(feature_coverage(target, camera, *size, thresholds));
        checks.push(view_outliers(target, uncertainty, camera, thresholds));
        checks.push(parameter_correlation(
            params,
            uncertainty,
            camera,
            thresholds,
        ));
        checks.push(parameter_conditioning(
            params,
            uncertainty,
            camera,
            thresholds,
        ));
    }
    let verdict = checks
        .iter()
        .filter_map(|check| check.status.verdict())
        .max()
        .unwrap_or(Verdict::Warn);
    Ok(QualityReport { verdict, checks })
}

// ─────────────────────────────────────────────────────────────────────────────
// Image sizes
// ─────────────────────────────────────────────────────────────────────────────

fn resolve_image_sizes(
    manifest: Option<&ImageManifest>,
    num_cameras: usize,
    given: &[[u32; 2]],
) -> Result<Vec<Option<[u32; 2]>>, Error> {
    if !given.is_empty() && given.len() != num_cameras {
        return Err(Error::invalid_input(format!(
            "image_sizes has {} entries; the export has {num_cameras} cameras",
            given.len()
        )));
    }
    if let Some(size) = given.iter().find(|[w, h]| *w == 0 || *h == 0) {
        return Err(Error::invalid_input(format!(
            "image size {}×{} is empty",
            size[0], size[1]
        )));
    }
    (0..num_cameras)
        .map(|camera| {
            let recorded = manifest_image_size(manifest, camera)?;
            match (recorded, given.get(camera).copied()) {
                (Some(r), Some(g)) if r != g => Err(Error::invalid_input(format!(
                    "camera {camera}: the image manifest's regions are {}×{} pixels, \
                     the given image size {}×{}",
                    r[0], r[1], g[0], g[1]
                ))),
                (recorded, given) => Ok(recorded.or(given)),
            }
        })
        .collect()
}

/// A camera's image size from its target frames' regions; `None` when the
/// manifest is absent or its frames are whole images, whose size it does not
/// record.
fn manifest_image_size(
    manifest: Option<&ImageManifest>,
    camera: usize,
) -> Result<Option<[u32; 2]>, Error> {
    let Some(manifest) = manifest else {
        return Ok(None);
    };
    let mut region: Option<[u32; 2]> = None;
    let mut whole = false;
    for frame in manifest.target_frames().filter(|f| f.camera == camera) {
        match frame.roi {
            None => whole = true,
            Some(roi) => {
                let size = [roi.w, roi.h];
                if size[0] == 0 || size[1] == 0 {
                    return Err(Error::invalid_input(format!(
                        "camera {camera}: the image manifest has an empty region"
                    )));
                }
                match region {
                    Some(previous) if previous != size => {
                        return Err(Error::invalid_input(format!(
                            "camera {camera}: the image manifest's regions differ in size \
                             ({}×{} and {}×{})",
                            previous[0], previous[1], size[0], size[1]
                        )));
                    }
                    _ => region = Some(size),
                }
            }
        }
    }
    if whole && region.is_some() {
        return Err(Error::invalid_input(format!(
            "camera {camera}: the image manifest mixes whole images and regions"
        )));
    }
    Ok(region)
}

// ─────────────────────────────────────────────────────────────────────────────
// Result-wide checks
// ─────────────────────────────────────────────────────────────────────────────

fn observability(uncertainty: &Uncertainty) -> QualityCheck {
    let kind = CheckKind::Observability;
    match uncertainty {
        Uncertainty::Available(covariance) => QualityCheck::new(
            kind,
            None,
            CheckStatus::Pass,
            format!(
                "the data determines every estimated parameter ({} shared), with {} \
                 residual degrees of freedom",
                covariance.labels.len(),
                covariance.dof
            ),
        ),
        Uncertainty::Unavailable { reason } => match reason {
            UncertaintyUnavailable::RankDeficient { unobservable } => QualityCheck::new(
                kind,
                None,
                CheckStatus::Fail,
                format!(
                    "the data does not determine {}, alone or in combination",
                    unobservable.join(", ")
                ),
            )
            .naming(unobservable.clone()),
            UncertaintyUnavailable::NoRedundancy => QualityCheck::new(
                kind,
                None,
                CheckStatus::Fail,
                "there are no more observations than estimated parameters".into(),
            ),
            UncertaintyUnavailable::AtBound { labels } => QualityCheck::new(
                kind,
                None,
                CheckStatus::Warn,
                format!(
                    "{} ended on a bound, where the data may not determine them",
                    labels.join(", ")
                ),
            )
            .naming(labels.clone()),
            UncertaintyUnavailable::NotConverged => QualityCheck::not_applicable(
                kind,
                None,
                "the final solve did not converge, so observability was not analysed",
            ),
            UncertaintyUnavailable::NotComputed => {
                QualityCheck::not_applicable(kind, None, "the result carries no uncertainty")
            }
            UncertaintyUnavailable::CalibrationWithoutCovariance => QualityCheck::not_applicable(
                kind,
                None,
                "the result depends on a calibration without covariance",
            ),
            _ => QualityCheck::not_applicable(
                kind,
                None,
                "the result has no covariance, for a reason this version does not assess",
            ),
        },
    }
}

fn convergence(uncertainty: &Uncertainty) -> QualityCheck {
    let kind = CheckKind::Convergence;
    match uncertainty {
        Uncertainty::Unavailable {
            reason: UncertaintyUnavailable::NotComputed,
        } => QualityCheck::not_applicable(
            kind,
            None,
            "the result carries no uncertainty, which records the solve's convergence",
        ),
        Uncertainty::Unavailable {
            reason: UncertaintyUnavailable::NotConverged,
        } => QualityCheck::new(
            kind,
            None,
            CheckStatus::Fail,
            "the final solve stopped on its iteration budget, short of a minimum".into(),
        ),
        _ => QualityCheck::new(
            kind,
            None,
            CheckStatus::Pass,
            "the final solve converged".into(),
        ),
    }
}

fn variance_factor(uncertainty: &Uncertainty, thresholds: &QualityThresholds) -> Vec<QualityCheck> {
    let kind = CheckKind::VarianceFactor;
    let whitened = uncertainty.covariance().is_some_and(|c| {
        c.noise
            .iter()
            .any(|n| n.kind == ResidualKind::Reprojection && n.level.is_whitened())
    });
    if !whitened && thresholds.expected_feature_noise_px.is_none() {
        return vec![QualityCheck::not_applicable(kind, None, NO_EXPECTED_NOISE)];
    }
    let Some(covariance) = uncertainty.covariance() else {
        return vec![QualityCheck::not_applicable(
            kind,
            None,
            no_covariance(uncertainty),
        )];
    };
    let checks: Vec<QualityCheck> = covariance
        .noise
        .iter()
        .filter(|noise| noise.kind == ResidualKind::Reprojection)
        .map(|noise| {
            variance_test(
                noise,
                covariance,
                thresholds.expected_feature_noise_px,
                thresholds.variance_test_alpha,
            )
        })
        .collect();
    if checks.is_empty() {
        return vec![QualityCheck::not_applicable(
            kind,
            None,
            "the final solve has no reprojection residuals",
        )];
    }
    checks
}

/// The χ² test of the variance factor for one family's noise, over its
/// share of the residual degrees of freedom, scaled by the efficiency of the
/// median-based σ̂: `σ̂²/σ₀²` against the stated `expected_px` for pixel
/// residuals, `σ̂₀²` against 1 for residuals whitened by stated covariances.
fn variance_test(
    noise: &ResidualNoise,
    covariance: &ParameterCovariance,
    expected_px: Option<f64>,
    alpha: f64,
) -> QualityCheck {
    let kind = CheckKind::VarianceFactor;
    let camera = noise.camera;
    let (ratio, measured) = match (noise.level, expected_px) {
        (NoiseLevel::Sigma0(s), _) => (
            s * s,
            format!(
                "σ̂₀ = {s:.3} of the residuals whitened by the stated point covariances, \
                 expected 1: σ̂₀²"
            ),
        ),
        (NoiseLevel::Sigma(s), Some(sigma0)) => (
            (s / sigma0).powi(2),
            format!("σ̂ = {s:.3} px against the expected {sigma0:.3} px: σ̂²/σ₀²"),
        ),
        (NoiseLevel::Sigma(_), None) => {
            return QualityCheck::not_applicable(kind, camera, NO_EXPECTED_NOISE);
        }
        _ => {
            return QualityCheck::not_applicable(
                kind,
                camera,
                "the result's noise level has no variance test",
            );
        }
    };
    if covariance.num_residuals == 0 {
        return QualityCheck::not_applicable(kind, camera, "the final solve has no residuals");
    }
    let dof = noise.count as f64 * covariance.dof as f64 / covariance.num_residuals as f64;
    let effective_dof = dof * MEDIAN_SIGMA_EFFICIENCY;
    if effective_dof < 1.0 {
        return QualityCheck::not_applicable(
            kind,
            camera,
            "too few residual degrees of freedom to test the noise level",
        );
    }
    let lower = chi_square_quantile(alpha / 2.0, effective_dof) / effective_dof;
    let upper = chi_square_quantile(1.0 - alpha / 2.0, effective_dof) / effective_dof;
    let summary =
        format!("{measured} = {ratio:.3}, accepted range [{lower:.3}, {upper:.3}] at α = {alpha}");
    let (status, threshold, message) = if !ratio.is_finite() {
        (
            CheckStatus::Fail,
            upper,
            format!("{summary}; the measured variance ratio is not finite"),
        )
    } else if ratio > upper {
        (
            CheckStatus::Fail,
            upper,
            format!("{summary}; the residuals exceed the stated noise (model or detection error)"),
        )
    } else if ratio < lower {
        (
            CheckStatus::Warn,
            lower,
            format!("{summary}; the residuals are below the stated noise, which is overstated"),
        )
    } else {
        (CheckStatus::Pass, upper, summary)
    };
    QualityCheck::new(kind, camera, status, message).measured(ratio, threshold, Unit::Ratio)
}

fn covariance_linearity(uncertainty: &Uncertainty, thresholds: &QualityThresholds) -> QualityCheck {
    let kind = CheckKind::CovarianceLinearity;
    let Some(covariance) = uncertainty.covariance() else {
        return QualityCheck::not_applicable(kind, None, no_covariance(uncertainty));
    };
    let Some(second) = &covariance.second_order else {
        return QualityCheck::not_applicable(
            kind,
            None,
            "the result carries no second-order prediction of its error",
        );
    };
    let limit = thresholds.max_second_order_scatter;
    let ratio = second.max_scatter_ratio;
    let combination = second
        .combination
        .iter()
        .map(|t| format!("{:+.2} {}", t.weight, t.label))
        .collect::<Vec<_>>()
        .join(" ");
    let summary = format!(
        "at second order the worst combination of the shared parameters is predicted to \
         scatter {ratio:.2}× its stated σ (NEES per parameter {:.2})",
        second.nees_ratio
    );
    let (status, message) = if !ratio.is_finite() {
        (
            CheckStatus::Fail,
            format!("{summary}: second-order scatter prediction is not finite"),
        )
    } else if ratio > limit {
        (
            CheckStatus::Warn,
            format!(
                "{summary}, above {limit}: the views pin {combination} (each parameter over \
                 its σ) down only through larger errors that cancel in it, so each parameter's \
                 σ holds but the joint covariance does not; tilt the board more (15–30°) and \
                 move it off the image centre, into the corners"
            ),
        )
    } else {
        (
            CheckStatus::Pass,
            format!("{summary}, within {limit}: the covariance describes the joint error"),
        )
    };
    let subjects = if status == CheckStatus::Warn {
        second.combination.iter().map(|t| t.label.clone()).collect()
    } else {
        Vec::new()
    };
    QualityCheck::new(kind, None, status, message)
        .measured(ratio, limit, Unit::Ratio)
        .naming(subjects)
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-camera checks
// ─────────────────────────────────────────────────────────────────────────────

fn projection_sigma<E: CalibratedCameras + ?Sized>(
    export: &E,
    params: &CameraParams,
    camera: usize,
    size: Option<[u32; 2]>,
    thresholds: &QualityThresholds,
) -> QualityCheck {
    let kind = CheckKind::ProjectionSigma;
    let uncertainty = export.parameter_uncertainty();
    let Some(covariance) = uncertainty.covariance() else {
        return QualityCheck::not_applicable(kind, Some(camera), no_covariance(uncertainty));
    };
    if camera_indices(params, covariance, camera).is_empty() {
        return QualityCheck::not_applicable(
            kind,
            Some(camera),
            format!(
                "the final solve held camera {camera}'s parameters fixed, so the result \
                 has no covariance of them to project"
            ),
        );
    }
    let Some(size) = size else {
        return QualityCheck::not_applicable(kind, Some(camera), NO_IMAGE_SIZE);
    };
    let map = match export.projection_uncertainty(camera, size) {
        Ok(map) => map,
        Err(err) => {
            return QualityCheck::not_applicable(
                kind,
                Some(camera),
                format!("the projection uncertainty could not be computed: {err}"),
            );
        }
    };
    let limit = thresholds.max_projection_sigma_px;
    let status = if !map.max.is_finite() {
        CheckStatus::Fail
    } else if map.max > limit {
        CheckStatus::Warn
    } else {
        CheckStatus::Pass
    };
    let corner = map
        .corner
        .map_or_else(|| "none reached".to_string(), |c| format!("{c:.3} px"));
    QualityCheck::new(
        kind,
        Some(camera),
        status,
        format!(
            "projection σ up to {:.3} px over the {}×{} image (centre {:.3} px, corners {corner})",
            map.max, size[0], size[1], map.center
        ),
    )
    .measured(map.max, limit, Unit::Pixel)
}

fn feature_coverage(
    target: &[TargetFeatureResidual],
    camera: usize,
    size: Option<[u32; 2]>,
    thresholds: &QualityThresholds,
) -> QualityCheck {
    let kind = CheckKind::FeatureCoverage;
    let Some([width, height]) = size else {
        return QualityCheck::not_applicable(kind, Some(camera), NO_IMAGE_SIZE);
    };
    let (width, height) = (f64::from(width), f64::from(height));
    let [columns, rows] = COVERAGE_GRID;
    let mut covered = vec![false; columns * rows];
    let mut features = 0usize;
    for residual in target.iter().filter(|r| r.camera == camera) {
        let [x, y] = residual.observed_px;
        // `contains` is false for NaN, so non-finite points drop out here.
        if !((0.0..=width).contains(&x) && (0.0..=height).contains(&y)) {
            continue;
        }
        features += 1;
        let column = ((x / width * columns as f64) as usize).min(columns - 1);
        let row = ((y / height * rows as f64) as usize).min(rows - 1);
        covered[row * columns + column] = true;
    }
    if features == 0 {
        return QualityCheck::not_applicable(
            kind,
            Some(camera),
            format!("camera {camera} has no observed target features in its image"),
        );
    }
    let cells = covered.iter().filter(|&&c| c).count();
    let fraction = cells as f64 / covered.len() as f64;
    let (status, threshold) = if fraction < thresholds.coverage_fail_below {
        (CheckStatus::Fail, thresholds.coverage_fail_below)
    } else if fraction < thresholds.coverage_warn_below {
        (CheckStatus::Warn, thresholds.coverage_warn_below)
    } else {
        (CheckStatus::Pass, thresholds.coverage_warn_below)
    };
    QualityCheck::new(
        kind,
        Some(camera),
        status,
        format!(
            "features reached {cells} of {} cells ({columns} × {rows}) of the image, {:.0} %",
            covered.len(),
            100.0 * fraction
        ),
    )
    .measured(fraction, threshold, Unit::Fraction)
}

fn view_outliers(
    target: &[TargetFeatureResidual],
    uncertainty: &Uncertainty,
    camera: usize,
    thresholds: &QualityThresholds,
) -> QualityCheck {
    let kind = CheckKind::ViewOutliers;
    let Some(covariance) = uncertainty.covariance() else {
        return QualityCheck::not_applicable(kind, Some(camera), no_covariance(uncertainty));
    };
    let Some(level) = reprojection_sigma(covariance, camera) else {
        return QualityCheck::not_applicable(
            kind,
            Some(camera),
            "the result estimates no reprojection noise for this camera",
        );
    };
    let sigma = level.value();
    if !sigma.is_finite() || sigma <= 0.0 {
        return QualityCheck::not_applicable(
            kind,
            Some(camera),
            "the reprojection noise is not finite or zero, so residuals cannot be normalized",
        );
    }
    // A whitened noise level normalizes the whitened errors, a pixel one the
    // pixel errors.
    let (error_of, sigma_text): (fn(&TargetFeatureResidual) -> Option<f64>, String) = match level {
        NoiseLevel::Sigma(_) => (|r| r.error_px, format!("σ̂ = {sigma:.3} px")),
        NoiseLevel::Sigma0(_) => (|r| r.error_whitened, format!("σ̂₀ = {sigma:.3}, whitened")),
        _ => {
            return QualityCheck::not_applicable(
                kind,
                Some(camera),
                "the result's noise level has no normalized residuals",
            );
        }
    };
    let mut views: BTreeMap<usize, (f64, usize)> = BTreeMap::new();
    for residual in target.iter().filter(|r| r.camera == camera) {
        if let Some(error) = error_of(residual).filter(|e| e.is_finite()) {
            let entry = views.entry(residual.view).or_default();
            entry.0 += error / sigma;
            entry.1 += 1;
        }
    }
    if views.is_empty() {
        return QualityCheck::not_applicable(
            kind,
            Some(camera),
            format!("camera {camera} has no reprojection residuals"),
        );
    }
    let limit = thresholds.max_view_residual;
    let means: Vec<(usize, f64)> = views
        .into_iter()
        .map(|(view, (sum, count))| (view, sum / count as f64))
        .collect();
    let worst = means
        .iter()
        .map(|&(_, m)| m)
        .fold(f64::NEG_INFINITY, f64::max);
    let outliers: Vec<String> = means
        .iter()
        .filter(|&&(_, m)| m > limit)
        .map(|&(view, _)| format!("view {view}"))
        .collect();
    let (status, message) = if outliers.is_empty() {
        (
            CheckStatus::Pass,
            format!(
                "every view's mean residual is within {limit} σ̂ (worst {worst:.2} σ̂, {sigma_text})"
            ),
        )
    } else {
        (
            CheckStatus::Warn,
            format!(
                "mean residual above {limit} σ̂ in {} of {} views (worst {worst:.2} σ̂, \
                 {sigma_text}): {}",
                outliers.len(),
                means.len(),
                outliers.join(", ")
            ),
        )
    };
    QualityCheck::new(kind, Some(camera), status, message)
        .measured(worst, limit, Unit::Sigma)
        .naming(outliers)
}

fn parameter_correlation(
    params: &CameraParams,
    uncertainty: &Uncertainty,
    camera: usize,
    thresholds: &QualityThresholds,
) -> QualityCheck {
    let kind = CheckKind::ParameterCorrelation;
    let Some(covariance) = uncertainty.covariance() else {
        return QualityCheck::not_applicable(kind, Some(camera), no_covariance(uncertainty));
    };
    let indices = camera_indices(params, covariance, camera);
    if indices.len() < 2 {
        return QualityCheck::not_applicable(
            kind,
            Some(camera),
            format!("the final solve estimated fewer than two of camera {camera}'s parameters"),
        );
    }
    let limit = thresholds.max_parameter_correlation;
    let c = &covariance.covariance;
    let mut worst = 0.0_f64;
    let mut pairs = Vec::new();
    let mut non_finite = false;
    for (n, &i) in indices.iter().enumerate() {
        for &j in &indices[n + 1..] {
            if !c[i][i].is_finite() || !c[j][j].is_finite() || !c[i][j].is_finite() {
                non_finite = true;
                pairs.push(format!("{}/{}", covariance.labels[i], covariance.labels[j]));
                continue;
            }
            let denom = (c[i][i] * c[j][j]).sqrt();
            if denom.is_nan()
                || denom <= 0.0
                || is_focal_pair(&covariance.labels[i], &covariance.labels[j])
            {
                continue;
            }
            let rho = (c[i][j] / denom).clamp(-1.0, 1.0).abs();
            worst = worst.max(rho);
            if rho > limit {
                pairs.push(format!("{}/{}", covariance.labels[i], covariance.labels[j]));
            }
        }
    }
    let (status, message) = if non_finite {
        (
            CheckStatus::Fail,
            format!(
                "non-finite covariance entries in parameter pair(s): {}",
                pairs.join(", ")
            ),
        )
    } else if pairs.is_empty() {
        (
            CheckStatus::Pass,
            format!("no two parameters correlate above {limit} (largest |ρ| = {worst:.3})"),
        )
    } else {
        (
            CheckStatus::Warn,
            format!(
                "{} parameter pair(s) correlate above |ρ| = {limit}, so the data determines \
                 their trade-off rather than each: {}",
                pairs.len(),
                pairs.join(", ")
            ),
        )
    };
    QualityCheck::new(kind, Some(camera), status, message)
        .measured(worst, limit, Unit::Correlation)
        .naming(pairs)
}

/// A camera parameter's σ inflation by its correlations with the camera's
/// other parameters is checked against the threshold, and a flagged one is
/// named with its partners: the camera's parameters it trades off against,
/// those whose partial correlation with it reaches `√0.75` (without one among
/// the regressors, its conditional σ would at least double). The camera's
/// marginal covariance is read, so trade-offs with the poses, rig extrinsics,
/// a hand-eye or a laser plane do not count: they show in those parameters'
/// own σ, and no change to the camera model or its capture moves them. A
/// flagged sensor tilt gets a named cause: the camera's free tangential or
/// thin-prism terms, which absorb a tilt to first order, when holding them
/// would at least halve its σ; else the lens distortion, which alone tells a
/// tilt from the principal point.
fn parameter_conditioning(
    params: &CameraParams,
    uncertainty: &Uncertainty,
    camera: usize,
    thresholds: &QualityThresholds,
) -> QualityCheck {
    let kind = CheckKind::ParameterConditioning;
    let Some(covariance) = uncertainty.covariance() else {
        return QualityCheck::not_applicable(kind, Some(camera), no_covariance(uncertainty));
    };
    let indices = camera_indices(params, covariance, camera);
    if indices.is_empty() {
        return QualityCheck::not_applicable(
            kind,
            Some(camera),
            format!(
                "the final solve held camera {camera}'s parameters fixed, so the result \
                 has no covariance of them"
            ),
        );
    }
    let Some(conditioning) = Conditioning::new(covariance, &indices) else {
        return QualityCheck::not_applicable(
            kind,
            Some(camera),
            "the covariance has a parameter without a finite, positive variance",
        );
    };
    let labels = &covariance.labels;
    // Positions `a` index the camera's parameters, `indices[a]` the labels.
    let name = |a: usize| split_camera_label(&labels[indices[a]]).1;
    let n = indices.len();
    let limit = thresholds.max_sigma_inflation;

    let mut worst: Option<(usize, f64)> = None;
    let mut clauses = Vec::new();
    let mut subjects = Vec::new();
    let mut non_finite = false;
    for a in 0..n {
        let excluded = left_out(a, n, &name);
        let inflation = conditioning.inflation_without(a, &excluded);
        if !inflation.is_finite() {
            non_finite = true;
            subjects.push(labels[indices[a]].clone());
            clauses.push(format!("{} has non-finite σ inflation", labels[indices[a]]));
            continue;
        }
        if worst.is_none_or(|(_, w)| inflation > w) {
            worst = Some((a, inflation));
        }
        if inflation <= limit {
            continue;
        }
        let mut partners: Vec<(usize, f64)> = (0..n)
            .filter(|&b| b != a && !excluded.contains(&b))
            .map(|b| (b, conditioning.partial_without(a, b, &excluded).abs()))
            .filter(|&(_, rho)| rho >= PARTNER_PARTIAL)
            .collect();
        partners.sort_by(|x, y| y.1.total_cmp(&x.1).then(x.0.cmp(&y.0)));
        let partners: Vec<usize> = partners.into_iter().map(|(b, _)| b).collect();
        clauses.push(conditioning_clause(
            covariance,
            &indices,
            &conditioning,
            a,
            inflation,
            &partners,
        ));
        subjects.push(labels[indices[a]].clone());
    }
    let Some((worst_at, worst_inflation)) = worst else {
        if non_finite {
            return QualityCheck::new(kind, Some(camera), CheckStatus::Fail, clauses.join("; "))
                .naming(subjects);
        }
        // `indices` is not empty, so the loop always sets a worst parameter.
        return QualityCheck::not_applicable(kind, Some(camera), "no parameter to assess");
    };
    let (status, message) = if non_finite {
        (CheckStatus::Fail, clauses.join("; "))
    } else if clauses.is_empty() {
        (
            CheckStatus::Pass,
            format!(
                "no parameter's σ is inflated above {limit}× (largest: {} {})",
                labels[indices[worst_at]],
                times(worst_inflation)
            ),
        )
    } else {
        (CheckStatus::Warn, clauses.join("; "))
    };
    QualityCheck::new(kind, Some(camera), status, message)
        .measured(worst_inflation, limit, Unit::Ratio)
        .naming(subjects)
}

/// The camera parameters left out of parameter `a`'s regressors:
///
/// - for a focal length the other one: the board's distance scales both
///   alike, and their ratio is what the data separates;
/// - for a radial coefficient (`k1` … `k6`) the camera's other ones: the
///   terms of one polynomial are collinear by construction, the pairwise
///   check reports a pair that is, and the remedy is a model with fewer
///   terms, not another capture.
///
/// Radial coefficients stay regressors of every other parameter: a sensor
/// tilt is determined only through them.
fn left_out<'a>(a: usize, n: usize, name: &impl Fn(usize) -> &'a str) -> Vec<usize> {
    let partner = match name(a) {
        "fx" => Some("fy"),
        "fy" => Some("fx"),
        _ => None,
    };
    if let Some(partner) = partner {
        return (0..n).filter(|&b| name(b) == partner).collect();
    }
    if is_radial(name(a)) {
        return (0..n).filter(|&b| b != a && is_radial(name(b))).collect();
    }
    Vec::new()
}

/// A radial polynomial coefficient (`k1` … `k6`).
fn is_radial(name: &str) -> bool {
    name.strip_prefix('k')
        .is_some_and(|rest| !rest.is_empty() && rest.bytes().all(|b| b.is_ascii_digit()))
}

/// Partial correlation at which a camera parameter is a partner of a flagged
/// one: without it among the regressors, the flagged parameter's conditional
/// σ would at least double (`√(1 − ρ²) ≤ ½`, `ρ ≥ √0.75`).
const PARTNER_PARTIAL: f64 = 0.866_025_403_784_438_6;

/// The tangential and thin-prism terms: to first order each absorbs a sensor
/// tilt, as a decentred radial distortion is a tangential term.
const DECENTRING_TERMS: [&str; 6] = ["p1", "p2", "s1", "s2", "s3", "s4"];

/// σ gain from holding a flagged tilt's free decentring terms at which they
/// are named its cause: holding them would at least halve its σ, the
/// partner threshold applied to the group.
const CAUSE_GAIN: f64 = 2.0;

/// One flagged parameter's sentence: its inflation, its partners and what
/// the trade-off means. `a` and `partners` are positions among the camera's
/// parameters `indices`.
fn conditioning_clause(
    covariance: &ParameterCovariance,
    indices: &[usize],
    conditioning: &Conditioning,
    a: usize,
    inflation: f64,
    partners: &[usize],
) -> String {
    let labels = &covariance.labels;
    let label_of = |b: usize| labels[indices[b]].as_str();
    let name = |b: usize| split_camera_label(label_of(b)).1;
    let label = label_of(a);
    let with = if partners.is_empty() {
        String::new()
    } else {
        let names: Vec<&str> = partners.iter().map(|&b| label_of(b)).collect();
        format!(" (with {})", names.join(", "))
    };
    let head = format!("{label} {}{with}", times(inflation));
    if !matches!(name(a), "tilt_x" | "tilt_y") {
        return if partners.is_empty() {
            format!(
                "{head}: the data determines it only in combination with several other \
                 parameters"
            )
        } else {
            format!("{head}: the data determines their combination, not each")
        };
    }
    // The camera's free decentring terms are the cause when holding them
    // would at least halve the tilt's σ. Not by partial correlation: with the
    // principal point beside them, the tilt's last-step partner can be the
    // principal point alone while the terms still absorb most of it.
    let terms_of = |terms: &[&str]| -> Vec<usize> {
        (0..indices.len())
            .filter(|&b| terms.contains(&name(b)))
            .collect()
    };
    let tangential = terms_of(&DECENTRING_TERMS[..2]);
    let prism = terms_of(&DECENTRING_TERMS[2..]);
    let held: Vec<usize> = tangential.iter().chain(&prism).copied().collect();
    let gain = conditioning.sigma_gain_holding(a, &held);
    if held.is_empty() || gain < CAUSE_GAIN {
        let i = indices[a];
        let sigma_deg = covariance.covariance[i][i].max(0.0).sqrt().to_degrees();
        return format!(
            "{head}, σ {}°: a Scheimpflug tilt is told apart from the principal point only by \
             the lens distortion it decentres; cover the image corners, or hold the tilt at \
             the known mount angle",
            significant(sigma_deg)
        );
    }
    let held_names: Vec<&str> = held.iter().map(|&b| label_of(b)).collect();
    let absorbs = |group: &[usize]| {
        !group.is_empty() && conditioning.sigma_gain_holding(a, group) >= CAUSE_GAIN
    };
    let terms = match (absorbs(&tangential), absorbs(&prism)) {
        (true, false) => "tangential",
        (false, true) => "thin-prism",
        _ => "tangential and thin-prism",
    };
    format!(
        "{head}: free {terms} terms absorb a tilt to first order (holding {} cuts σ({label}) \
         {}×); hold them to estimate the tilt, at the cost of reading a real decentring as \
         tilt, or hold the tilt and keep them free",
        held_names.join(", "),
        factor(gain)
    )
}

/// `×3.0`, `×43`: an inflation as the messages print it.
fn times(value: f64) -> String {
    format!("×{}", factor(value))
}

/// A factor to one decimal below 10, else whole.
fn factor(value: f64) -> String {
    if value < 10.0 {
        format!("{value:.1}")
    } else {
        format!("{value:.0}")
    }
}

/// `value` to two significant digits (0.57, 0.030, 1.2, 32).
fn significant(value: f64) -> String {
    if !(value.is_finite() && value > 0.0) {
        return format!("{value}");
    }
    let decimals = (1 - value.log10().floor() as i32).max(0) as usize;
    format!("{value:.decimals$}")
}

/// Relative floor of a correlation matrix's eigenvalues before it is
/// inverted: each is raised to at least this share of the largest, the rank
/// tolerance of the solver's covariance. It caps a σ inflation at 1e6, so
/// even a matrix the solver only just accepted gives finite values.
const EIGEN_FLOOR: f64 = 1e-12;

/// The conditioning of a set of parameters: their correlation matrix `C`
/// and its inverse `P`, the precision in units of each parameter's σ.
/// `P_ii = 1/(1 − R_i²)`, `R_i` the multiple correlation of parameter `i`
/// with the others of the set.
struct Conditioning {
    correlation: DMatrix<f64>,
    precision: DMatrix<f64>,
}

impl Conditioning {
    /// The conditioning of the parameters at `indices` of `cov`, from their
    /// block of it (their marginal covariance); position `a` here is
    /// `indices[a]` there. `None` when an index is out of range, or an entry
    /// of the block is not finite, or a variance not positive.
    fn new(cov: &ParameterCovariance, indices: &[usize]) -> Option<Self> {
        let n = indices.len();
        let total = cov.labels.len();
        if n == 0
            || indices.iter().any(|&i| i >= total)
            || cov.covariance.len() != total
            || cov.covariance.iter().any(|row| row.len() != total)
        {
            return None;
        }
        let entry = |a: usize, b: usize| cov.covariance[indices[a]][indices[b]];
        let scale: Vec<f64> = (0..n)
            .map(|a| {
                let variance = entry(a, a);
                (variance.is_finite() && variance > 0.0).then(|| 1.0 / variance.sqrt())
            })
            .collect::<Option<_>>()?;
        let mut finite = true;
        let correlation = DMatrix::from_fn(n, n, |a, b| {
            if a == b {
                return 1.0;
            }
            let value = 0.5 * (entry(a, b) + entry(b, a)) * scale[a] * scale[b];
            finite &= value.is_finite();
            value
        });
        if !finite {
            return None;
        }
        let precision = clamped_inverse(&correlation)?;
        Some(Self {
            correlation,
            precision,
        })
    }

    /// The precision of the parameters other than `excluded` at `(a, b)`:
    /// `P_ab − P_aE P_EE⁻¹ P_Eb`, the Schur complement that drops `excluded`
    /// from the regressors.
    fn reduced(&self, a: usize, b: usize, excluded: &[usize]) -> f64 {
        let p = &self.precision;
        let e = excluded.len();
        if e == 0 {
            return p[(a, b)];
        }
        let block = DMatrix::from_fn(e, e, |r, c| p[(excluded[r], excluded[c])]);
        let Some(inverse) = clamped_inverse(&block) else {
            return p[(a, b)];
        };
        let pa = DVector::from_fn(e, |r, _| p[(a, excluded[r])]);
        let pb = DVector::from_fn(e, |r, _| p[(b, excluded[r])]);
        p[(a, b)] - pa.dot(&(inverse * pb))
    }

    /// The σ inflation of parameter `a` by its correlations with the others
    /// but `excluded`: `√P'_aa = 1/√(1 − R²)` of the reduced precision, at
    /// least 1.
    fn inflation_without(&self, a: usize, excluded: &[usize]) -> f64 {
        self.reduced(a, a, excluded).max(1.0).sqrt()
    }

    /// The partial correlation of `a` and `b` given the others but
    /// `excluded`: `−P'_ab/√(P'_aa P'_bb)`.
    fn partial_without(&self, a: usize, b: usize, excluded: &[usize]) -> f64 {
        let (ab, aa, bb) = (
            self.reduced(a, b, excluded),
            self.reduced(a, a, excluded),
            self.reduced(b, b, excluded),
        );
        (-ab / (aa * bb).sqrt()).clamp(-1.0, 1.0)
    }

    /// How many times smaller the σ of parameter `a` becomes when the
    /// parameters `held` are held at their values:
    /// `1/√(1 − c_aGᵀ C_GG⁻¹ c_aG)`, the eigenvalues of `C_GG` floored as for
    /// the full matrix (two decentring terms can correlate fully,
    /// `corr(p1, s3) = −1`).
    fn sigma_gain_holding(&self, a: usize, held: &[usize]) -> f64 {
        let h = held.len();
        if h == 0 {
            return 1.0;
        }
        let block = DMatrix::from_fn(h, h, |r, c| self.correlation[(held[r], held[c])]);
        let cross = DVector::from_fn(h, |r, _| self.correlation[(a, held[r])]);
        let Some(inverse) = clamped_inverse(&block) else {
            return 1.0;
        };
        let explained = cross.dot(&(inverse * &cross));
        1.0 / (1.0 - explained).max(EIGEN_FLOOR).sqrt()
    }
}

/// The inverse of a symmetric positive semi-definite matrix, each eigenvalue
/// raised to at least [`EIGEN_FLOOR`] times the largest; `None` when the
/// largest is not finite and positive.
fn clamped_inverse(matrix: &DMatrix<f64>) -> Option<DMatrix<f64>> {
    let eigen = SymmetricEigen::new(matrix.clone());
    let largest = eigen.eigenvalues.max();
    if !(largest.is_finite() && largest > 0.0) {
        return None;
    }
    let floor = EIGEN_FLOOR * largest;
    let inverse = eigen.eigenvalues.map(|l| 1.0 / l.max(floor));
    let v = &eigen.eigenvectors;
    Some(v * DMatrix::from_diagonal(&inverse) * v.transpose())
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

const NO_IMAGE_SIZE: &str =
    "the image size is unknown: the image manifest does not record it and none was given";

const NO_EXPECTED_NOISE: &str = "no expected feature noise was stated (expected_feature_noise_px)";

fn no_covariance(uncertainty: &Uncertainty) -> String {
    let reason = match uncertainty {
        Uncertainty::Available(_) => return "the result has a covariance".into(),
        Uncertainty::Unavailable { reason } => reason,
    };
    let why = match reason {
        UncertaintyUnavailable::NotConverged => "the final solve did not converge",
        UncertaintyUnavailable::RankDeficient { .. } => {
            "the data does not determine every parameter"
        }
        UncertaintyUnavailable::AtBound { .. } => "a parameter ended on a bound",
        UncertaintyUnavailable::NoRedundancy => "there is no redundancy",
        UncertaintyUnavailable::NotComputed => "none was computed",
        UncertaintyUnavailable::CalibrationWithoutCovariance => {
            "the calibration it depends on has none"
        }
        _ => "of a reason this version does not describe",
    };
    format!("the result has no parameter covariance: {why}")
}

/// Positions in the covariance of camera `camera`'s own parameters
/// (intrinsics, distortion, sensor tilt), in label order.
fn camera_indices(
    params: &CameraParams,
    covariance: &ParameterCovariance,
    camera: usize,
) -> Vec<usize> {
    let names: Vec<&str> = camera_parameters(params)
        .map(|parameters| parameters.into_iter().map(|(label, _)| label).collect())
        .unwrap_or_default();
    covariance
        .labels
        .iter()
        .enumerate()
        .filter(|(_, label)| {
            let (c, name) = split_camera_label(label);
            c == camera && names.contains(&name)
        })
        .map(|(i, _)| i)
        .collect()
}

/// Whether two labels of one camera are its two focal lengths. The board's
/// distance scales `fx` and `fy` alike, so they correlate almost fully in
/// every planar calibration, while their ratio, the pixel aspect, is what the
/// data separates; the correlation check leaves the pair out.
fn is_focal_pair(a: &str, b: &str) -> bool {
    let (a, b) = (split_camera_label(a).1, split_camera_label(b).1);
    matches!((a, b), ("fx", "fy") | ("fy", "fx"))
}

/// The reprojection noise level of `camera`: its own when the result
/// estimates each camera's apart, else the shared one.
fn reprojection_sigma(covariance: &ParameterCovariance, camera: usize) -> Option<NoiseLevel> {
    let reprojection = || {
        covariance
            .noise
            .iter()
            .filter(|n| n.kind == ResidualKind::Reprojection)
    };
    reprojection()
        .find(|n| n.camera == Some(camera))
        .or_else(|| reprojection().find(|n| n.camera.is_none()))
        .map(|n| n.level)
}

/// The `p`-quantile of the χ² distribution with `k` degrees of freedom, by
/// the Wilson–Hilferty cube-root normal approximation (in the 0.5 % tails,
/// within 3 % of the exact quantile at `k = 10` and 1 % from `k = 30`; the
/// test runs at hundreds of degrees of freedom or more).
fn chi_square_quantile(p: f64, k: f64) -> f64 {
    let a = 2.0 / (9.0 * k);
    let cube_root = 1.0 - a + normal_quantile(p) * a.sqrt();
    k * cube_root.max(0.0).powi(3)
}

/// The `p`-quantile of the standard normal distribution, `p` in `(0, 1)`, by
/// Acklam's rational approximation (relative error below 1.2e-9).
fn normal_quantile(p: f64) -> f64 {
    const A: [f64; 6] = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_69e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239,
    ];
    const B: [f64; 5] = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    const C: [f64; 6] = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838,
        -2.549_732_539_343_734,
        4.374_664_141_464_968,
        2.938_163_982_698_783,
    ];
    const D: [f64; 4] = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996,
        3.754_408_661_907_416,
    ];
    const P_LOW: f64 = 0.024_25;
    let tail = |q: f64| {
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    };
    if p < P_LOW {
        tail((-2.0 * p.ln()).sqrt())
    } else if p <= 1.0 - P_LOW {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        -tail((-2.0 * (1.0 - p).ln()).sqrt())
    }
}

#[cfg(test)]
mod tests;
