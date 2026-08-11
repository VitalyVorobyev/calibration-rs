//! Shared ChESS corner-stage options for chessboard-like detectors.

use calib_targets::core::DetectorConfig;
use calib_targets::detect::default_chess_config;
use serde::{Deserialize, Serialize};

#[cfg(feature = "schemars")]
use schemars::JsonSchema;

/// ChESS corner extractor overrides.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(JsonSchema))]
#[serde(deny_unknown_fields, default)]
pub struct ChessCornersConfig {
    /// Absolute acceptance threshold on the raw ChESS response: a corner is
    /// kept when its response exceeds this value. `None` keeps
    /// [`default_chess_config`]'s noise-floor cutoff.
    ///
    /// The threshold is always absolute. `chess-corners` 1.0 collapsed its
    /// former `Threshold::{Absolute, Relative}` enum into a single `f32`;
    /// there is no longer a "fraction of the image maximum" mode.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub threshold_value: Option<f32>,
}

pub(crate) fn chess_config_for_override(
    override_cfg: Option<ChessCornersConfig>,
) -> DetectorConfig {
    let config = default_chess_config();
    match override_cfg.and_then(|cfg| cfg.threshold_value) {
        Some(threshold) => config.with_threshold(threshold),
        None => config,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn threshold_override_applies() {
        let config = chess_config_for_override(Some(ChessCornersConfig {
            threshold_value: Some(30.0),
        }));
        assert_eq!(config.threshold, 30.0);
    }

    #[test]
    fn absent_override_keeps_detector_default() {
        assert_eq!(
            chess_config_for_override(None).threshold,
            default_chess_config().threshold
        );
        assert_eq!(
            chess_config_for_override(Some(ChessCornersConfig::default())).threshold,
            default_chess_config().threshold
        );
    }
}
