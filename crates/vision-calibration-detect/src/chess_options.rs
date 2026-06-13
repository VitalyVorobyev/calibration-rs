//! Shared ChESS corner-stage options for chessboard-like detectors.

use calib_targets::core::DetectorConfig;
use calib_targets::detect::default_chess_config;
use chess_corners::Threshold;
use serde::{Deserialize, Serialize};

#[cfg(feature = "schemars")]
use schemars::JsonSchema;

/// ChESS corner extractor overrides.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(JsonSchema))]
#[serde(deny_unknown_fields, default)]
pub struct ChessCornersConfig {
    /// Acceptance threshold mode. `None` keeps the detector default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub threshold_mode: Option<ChessThresholdMode>,
    /// Acceptance threshold value. `None` keeps the detector default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub threshold_value: Option<f32>,
}

/// ChESS threshold interpretation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum ChessThresholdMode {
    /// Threshold in native ChESS response units.
    Absolute,
    /// Threshold as a fraction of the image maximum response.
    Relative,
}

pub(crate) fn chess_config_for_override(
    override_cfg: Option<ChessCornersConfig>,
) -> DetectorConfig {
    let mut config = default_chess_config();
    let Some(override_cfg) = override_cfg else {
        return config;
    };
    if override_cfg.threshold_mode.is_none() && override_cfg.threshold_value.is_none() {
        return config;
    }

    let (current_mode, current_value) = match config.threshold {
        Threshold::Absolute(v) => (ChessThresholdMode::Absolute, v),
        Threshold::Relative(v) => (ChessThresholdMode::Relative, v),
        _ => (ChessThresholdMode::Absolute, 15.0),
    };
    let mode = override_cfg.threshold_mode.unwrap_or(current_mode);
    let value = override_cfg.threshold_value.unwrap_or(current_value);
    config = config.with_threshold(match mode {
        ChessThresholdMode::Absolute => Threshold::Absolute(value),
        ChessThresholdMode::Relative => Threshold::Relative(value),
    });
    config
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn threshold_override_preserves_unspecified_mode() {
        let config = chess_config_for_override(Some(ChessCornersConfig {
            threshold_mode: None,
            threshold_value: Some(30.0),
        }));
        assert_eq!(config.threshold, Threshold::Absolute(30.0));
    }

    #[test]
    fn relative_threshold_override_applies() {
        let config = chess_config_for_override(Some(ChessCornersConfig {
            threshold_mode: Some(ChessThresholdMode::Relative),
            threshold_value: Some(0.25),
        }));
        assert_eq!(config.threshold, Threshold::Relative(0.25));
    }
}
