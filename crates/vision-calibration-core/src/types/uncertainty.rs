//! The uncertainty of a calibration result's estimated parameters.
//!
//! A result's [`Uncertainty`] is the first-order covariance of the parameters
//! the final solve estimated, evaluated at the solution. Parameters are named
//! by labels such as `fx`, `k1`, `tilt_x` or, in multi-camera results,
//! `cam1.fx`. A pose's six labels end in `rx, ry, rz` (rotation, radians) and
//! `tx, ty, tz` (translation, metres): the right perturbation `T · Exp(δ)`,
//! so both are expressed in the pose's source frame (the target frame for
//! `camera_se3_target`).
//!
//! The covariance is conditional on every parameter the solve held fixed
//! ([`ParameterCovariance::fixed`]): it does not include their uncertainty.
//! Per-view parameters (a view's target pose, its robot-pose correction) are
//! reported as one block per view, without their correlation with the
//! shared parameters.

use serde::{Deserialize, Serialize};

/// The uncertainty of a result's estimated parameters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum Uncertainty {
    /// The covariance of the estimated parameters.
    Available(ParameterCovariance),
    /// No covariance, for the reason given.
    Unavailable {
        /// Why the covariance is not available.
        reason: UncertaintyUnavailable,
    },
}

impl Default for Uncertainty {
    /// Unavailable, not computed: the state of a result loaded from a file
    /// written without uncertainty.
    fn default() -> Self {
        Self::Unavailable {
            reason: UncertaintyUnavailable::NotComputed,
        }
    }
}

impl Uncertainty {
    /// The covariance, if available.
    pub fn covariance(&self) -> Option<&ParameterCovariance> {
        match self {
            Self::Available(covariance) => Some(covariance),
            Self::Unavailable { .. } => None,
        }
    }
}

/// Why a result has no covariance.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum UncertaintyUnavailable {
    /// The solve stopped on its iteration budget, short of a minimum.
    NotConverged,
    /// The data does not determine these parameters (or combinations of
    /// them): too few or too similar views, or a free gauge.
    RankDeficient {
        /// Labels of the parameters involved.
        unobservable: Vec<String>,
    },
    /// These parameters ended on a bound, where a Gaussian covariance does
    /// not describe them.
    AtBound {
        /// Labels of the parameters on a bound.
        labels: Vec<String>,
    },
    /// No more observations than estimated parameters.
    NoRedundancy,
    /// The result was produced or stored without uncertainty.
    NotComputed,
}

/// The covariance of a result's estimated parameters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
pub struct ParameterCovariance {
    /// Labels of the shared parameters, in the order of `covariance`.
    pub labels: Vec<String>,
    /// Covariance of the shared parameters: one row per label, symmetric.
    pub covariance: Vec<Vec<f64>>,
    /// Per-view parameters, one block per view that has free ones.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub views: Vec<ViewCovariance>,
    /// Labels of the parameters held fixed. The covariance is conditional on
    /// them.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub fixed: Vec<String>,
    /// The noise level of each kind of residual, estimated from the
    /// residuals at the solution.
    pub noise: Vec<ResidualNoise>,
    /// Number of scalar residuals, priors included.
    pub num_residuals: usize,
    /// Residuals minus estimated parameters.
    pub dof: usize,
    /// How many times the variance of the shared parameters across views
    /// exceeds what the residual noise predicts, from the scatter of the
    /// views' contributions to the fit. The residual noise model takes every
    /// corner's error as independent; errors shared by a view's corners (a
    /// lens model or target that does not fit exactly, a detector bias) are
    /// not in it, and on real images make the estimate vary between view
    /// sets several times more than predicted. When above 1, `covariance`
    /// and `views` are already scaled by it. `None` below six views.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub view_scatter: Option<f64>,
}

impl ParameterCovariance {
    /// Position of `label` among the shared parameters.
    pub fn index(&self, label: &str) -> Option<usize> {
        self.labels.iter().position(|l| l == label)
    }

    /// Standard deviation of a shared parameter.
    pub fn std_dev(&self, label: &str) -> Option<f64> {
        let i = self.index(label)?;
        Some(self.covariance[i][i].max(0.0).sqrt())
    }

    /// Correlation coefficient of two shared parameters, in `[-1, 1]`.
    pub fn correlation(&self, a: &str, b: &str) -> Option<f64> {
        let (i, j) = (self.index(a)?, self.index(b)?);
        let denom = (self.covariance[i][i] * self.covariance[j][j]).sqrt();
        (denom > 0.0).then(|| (self.covariance[i][j] / denom).clamp(-1.0, 1.0))
    }
}

/// The covariance of one view's own parameters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
pub struct ViewCovariance {
    /// View index, as in the result's per-view lists.
    pub view: usize,
    /// Labels of the view's parameters, e.g. `camera_se3_target.rx`.
    pub labels: Vec<String>,
    /// Their covariance: one row per label, symmetric.
    pub covariance: Vec<Vec<f64>>,
}

/// The estimated noise level of one kind of residual.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
pub struct ResidualNoise {
    /// The kind of residual.
    pub kind: ResidualKind,
    /// The noise level, in the residual kind's unit or, for residuals
    /// whitened by stated per-point covariances, as the dimensionless
    /// `σ̂₀`.
    #[serde(flatten)]
    pub level: NoiseLevel,
    /// Number of scalar residuals of this kind.
    pub count: usize,
    /// The camera, for a result that estimates each camera's noise apart
    /// (the rig laserline, which solves each camera's plane on its own);
    /// `None` for one level shared by every camera.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub camera: Option<usize>,
}

/// The estimated noise level of one kind of residual: the
/// Gaussian-equivalent standard deviation of one scalar residual (one pixel
/// axis for reprojection), from the median absolute residual so that
/// outliers do not inflate it.
///
/// In JSON it is one key of the [`ResidualNoise`] entry: `sigma` or
/// `sigma0`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
pub enum NoiseLevel {
    /// `σ̂` in the residual kind's unit: pixels for reprojection and the
    /// laser line, metres for the laser plane.
    #[serde(rename = "sigma")]
    Sigma(f64),
    /// `σ̂₀` of residuals whitened by their stated per-point covariances:
    /// dimensionless, the square root of the a-posteriori variance factor.
    /// About 1 when the stated covariances describe the noise; 2 means the
    /// actual noise is twice what they state.
    #[serde(rename = "sigma0")]
    Sigma0(f64),
}

impl NoiseLevel {
    /// The value, in whichever unit the variant states.
    pub fn value(&self) -> f64 {
        match *self {
            NoiseLevel::Sigma(v) | NoiseLevel::Sigma0(v) => v,
        }
    }

    /// Whether the residuals were whitened by stated covariances.
    pub fn is_whitened(&self) -> bool {
        matches!(self, NoiseLevel::Sigma0(_))
    }
}

/// A kind of residual, which sets its unit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum ResidualKind {
    /// Target reprojection error, pixels.
    Reprojection,
    /// Laser point-to-plane distance, metres.
    LaserPlane,
    /// Laser line distance in the image, pixels.
    LaserLine,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> ParameterCovariance {
        ParameterCovariance {
            labels: vec!["fx".into(), "cx".into()],
            covariance: vec![vec![4.0, -1.2], vec![-1.2, 1.0]],
            views: vec![ViewCovariance {
                view: 0,
                labels: vec!["camera_se3_target.tz".into()],
                covariance: vec![vec![1e-8]],
            }],
            fixed: vec!["k3".into()],
            noise: vec![ResidualNoise {
                kind: ResidualKind::Reprojection,
                level: NoiseLevel::Sigma(0.2),
                count: 400,
                camera: None,
            }],
            num_residuals: 400,
            dof: 380,
            view_scatter: Some(3.5),
        }
    }

    #[test]
    fn std_dev_and_correlation_read_the_matrix() {
        let c = sample();
        assert_eq!(c.std_dev("fx"), Some(2.0));
        assert!((c.correlation("fx", "cx").unwrap() + 0.6).abs() < 1e-12);
        assert_eq!(c.std_dev("fy"), None);
    }

    #[test]
    fn serializes_with_a_status_tag() {
        let json = serde_json::to_value(Uncertainty::Available(sample())).unwrap();
        assert_eq!(json["status"], "available");
        assert_eq!(json["labels"][0], "fx");
        let back: Uncertainty = serde_json::from_value(json).unwrap();
        assert_eq!(back, Uncertainty::Available(sample()));

        let unavailable = Uncertainty::Unavailable {
            reason: UncertaintyUnavailable::RankDeficient {
                unobservable: vec!["cx".into()],
            },
        };
        let json = serde_json::to_value(&unavailable).unwrap();
        assert_eq!(json["status"], "unavailable");
        assert_eq!(json["reason"]["kind"], "rank_deficient");
        assert_eq!(
            serde_json::from_value::<Uncertainty>(json).unwrap(),
            unavailable
        );
    }

    /// A pixel level keeps its `sigma` key; a whitened one is `sigma0`, so a
    /// reader cannot take one for the other.
    #[test]
    fn noise_level_is_keyed_by_its_unit() {
        let pixel = ResidualNoise {
            kind: ResidualKind::Reprojection,
            level: NoiseLevel::Sigma(0.25),
            count: 10,
            camera: None,
        };
        let json = serde_json::to_value(&pixel).unwrap();
        assert_eq!(
            json,
            serde_json::json!({"kind": "reprojection", "sigma": 0.25, "count": 10})
        );
        assert_eq!(
            serde_json::from_value::<ResidualNoise>(json).unwrap(),
            pixel
        );

        let whitened = ResidualNoise {
            level: NoiseLevel::Sigma0(1.04),
            camera: Some(1),
            ..pixel
        };
        let json = serde_json::to_value(&whitened).unwrap();
        assert_eq!(
            json,
            serde_json::json!({"kind": "reprojection", "sigma0": 1.04, "count": 10, "camera": 1})
        );
        let back = serde_json::from_value::<ResidualNoise>(json).unwrap();
        assert_eq!(back, whitened);
        assert!(back.level.is_whitened());
        assert_eq!(back.level.value(), 1.04);
    }

    #[test]
    fn default_is_not_computed() {
        assert_eq!(
            Uncertainty::default(),
            Uncertainty::Unavailable {
                reason: UncertaintyUnavailable::NotComputed
            }
        );
    }
}
