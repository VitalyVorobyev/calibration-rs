//! The diagnostics every `*Export` carries next to its result.

use serde::{Deserialize, Serialize};
use vision_calibration_core::{ImageManifest, PerFeatureResiduals};

/// Evidence about how an export's result was reached: what a viewer or a
/// debugging session reads, not what a consumer needs to use the
/// calibration.
///
/// The result fields of an export (camera parameters, poses, uncertainty,
/// mean reprojection error) are the stable contract; this group may gain
/// detail between releases.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[non_exhaustive]
pub struct ExportDiagnostics {
    /// The reprojection (and, for a laser workflow, laser) residual of every
    /// feature, with per-camera histograms.
    #[serde(default)]
    pub per_feature_residuals: PerFeatureResiduals,
    /// Where each view's source images are, for a viewer that overlays the
    /// residuals on them. The calibration never reads it; `None` means no
    /// images were recorded (the dataset runner and the desktop app fill it).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image_manifest: Option<ImageManifest>,
}

impl ExportDiagnostics {
    /// Diagnostics holding the per-feature residuals and no image manifest.
    pub fn new(per_feature_residuals: PerFeatureResiduals) -> Self {
        Self {
            per_feature_residuals,
            image_manifest: None,
        }
    }
}
