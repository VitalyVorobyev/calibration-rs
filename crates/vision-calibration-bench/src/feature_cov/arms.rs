//! The study's arms: what each one asks the detector for, and the corner
//! sets they share.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use vision_calibration_core::{CorrespondenceView, PlanarDataset, Pt2};
use vision_calibration_dataset::{CornerCovarianceSpec, DatasetSpec, ImageNoiseSpec};
use vision_calibration_pipeline::dataset_runner::{CovarianceFault, RunError};

use super::stats::{Rng, seed_of};

/// Standard deviation of the power check's added noise, px per axis.
pub const NOISE_SIGMA: f64 = 0.05;

/// One arm of the study.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Arm {
    /// No covariance: the default.
    Uniform,
    /// The detector covariance at a known image noise of 1 gray level.
    Known1,
    /// The detector covariance at each image's estimated noise.
    Estimated,
    /// `uniform`'s corners plus N(0, 0.05²) px per axis (power check).
    UniformNoise,
}

impl Arm {
    /// Every arm, the reference first.
    pub const ALL: [Arm; 4] = [Arm::Uniform, Arm::Known1, Arm::Estimated, Arm::UniformNoise];
    /// The arms a detector run produces (the power check derives from
    /// `uniform`).
    pub const DETECTED: [Arm; 3] = [Arm::Uniform, Arm::Known1, Arm::Estimated];
    /// The weighting arms the decision rule judges.
    pub const WEIGHTED: [Arm; 2] = [Arm::Known1, Arm::Estimated];

    /// The arm's name in the outputs.
    pub fn name(self) -> &'static str {
        match self {
            Arm::Uniform => "uniform",
            Arm::Known1 => "known1",
            Arm::Estimated => "estimated",
            Arm::UniformNoise => "uniform_noise",
        }
    }

    /// The image noise the arm asks the detector to state its covariance in;
    /// `None` for an arm without covariance.
    fn image_noise(self) -> Option<ImageNoiseSpec> {
        match self {
            Arm::Known1 => Some(ImageNoiseSpec::Known(1.0)),
            Arm::Estimated => Some(ImageNoiseSpec::Estimate),
            Arm::Uniform | Arm::UniformNoise => None,
        }
    }

    /// `spec` with this arm's detector covariance request: the manifest's
    /// own detector settings otherwise unchanged, so every arm detects the
    /// same corners.
    pub fn spec(self, spec: &DatasetSpec) -> DatasetSpec {
        let mut spec = spec.clone();
        let detector = spec.detector.get_or_insert_default();
        let chess = detector.chess_corners.get_or_insert_default();
        chess.covariance = self
            .image_noise()
            .map(|image_noise| CornerCovarianceSpec { image_noise });
        spec
    }
}

impl std::fmt::Display for Arm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// Why a covariance arm is absent for a dataset: the run's typed error,
/// without the image it names.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Declined {
    /// Some corners carry no covariance ([`CovarianceFault::Missing`]).
    Missing,
    /// A reported covariance is not positive definite.
    NotPositiveDefinite,
    /// Another fault the runner reports.
    Other,
}

/// The arm's fault when `err` is the runner's corner-covariance error.
pub fn declined(err: &anyhow::Error) -> Option<Declined> {
    err.chain()
        .find_map(|e| e.downcast_ref::<RunError>())
        .and_then(|e| match e {
            RunError::CornerCovariance { fault, .. } => Some(match fault {
                CovarianceFault::Missing { .. } => Declined::Missing,
                CovarianceFault::NotPositiveDefinite { .. } => Declined::NotPositiveDefinite,
                _ => Declined::Other,
            }),
            _ => None,
        })
}

/// Run `detect` for `arm`, mapping a declined covariance to `Ok(Err(_))`.
pub fn detect_arm<T>(
    arm: Arm,
    detect: impl FnOnce(&DatasetSpec) -> Result<T>,
    spec: &DatasetSpec,
) -> Result<Result<T, Declined>> {
    match detect(&arm.spec(spec)) {
        Ok(value) => Ok(Ok(value)),
        Err(err) => match declined(&err) {
            Some(fault) if arm.image_noise().is_some() => Ok(Err(fault)),
            _ => Err(err.context(format!("detection for arm {arm}"))),
        },
    }
}

/// Check that `other` holds exactly `reference`'s corners, view by view:
/// the covariance must not move them.
pub fn ensure_same_corners<'a>(
    reference: impl IntoIterator<Item = Option<&'a CorrespondenceView>>,
    other: impl IntoIterator<Item = Option<&'a CorrespondenceView>>,
    what: &str,
) -> Result<()> {
    let (reference, other): (Vec<_>, Vec<_>) =
        (reference.into_iter().collect(), other.into_iter().collect());
    anyhow::ensure!(
        reference.len() == other.len(),
        "{what}: {} views, the reference has {}",
        other.len(),
        reference.len()
    );
    for (i, (a, b)) in reference.iter().zip(&other).enumerate() {
        let same = match (a, b) {
            (None, None) => true,
            (Some(a), Some(b)) => a.points_2d == b.points_2d && a.points_3d == b.points_3d,
            _ => false,
        };
        anyhow::ensure!(same, "{what}: view {i} has different corners");
    }
    Ok(())
}

/// The power-check arm's dataset: `uniform`'s views with seeded isotropic
/// N(0, σ²) px per axis added to every corner (training and held-out
/// roles alike, as each view enters both).
pub fn with_noise(uniform: &PlanarDataset, label: &str) -> Result<PlanarDataset> {
    let mut rng = Rng::new(seed_of(&format!("{label}/uniform_noise")));
    let mut noisy = uniform.clone();
    for view in &mut noisy.views {
        anyhow::ensure!(
            view.obs.covariances().is_none(),
            "{label}: the uniform arm carries a covariance"
        );
        for p in &mut view.obs.points_2d {
            *p = Pt2::new(
                p.x + NOISE_SIGMA * rng.normal(),
                p.y + NOISE_SIGMA * rng.normal(),
            );
        }
    }
    Ok(noisy)
}
