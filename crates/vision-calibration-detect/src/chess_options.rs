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
    /// Acceptance threshold value. `None` keeps the detector default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub threshold_value: Option<f32>,
}

pub(crate) fn chess_config_for_override(
    override_cfg: Option<ChessCornersConfig>,
) -> DetectorConfig {
    let mut config = default_chess_config();
    let Some(override_cfg) = override_cfg else {
        return config;
    };
    if override_cfg.threshold_value.is_some() {
        config = config.with_threshold(override_cfg.threshold_value.unwrap());
    }
    config
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn threshold_override_preserves_unspecified_mode() {
        let config = chess_config_for_override(Some(ChessCornersConfig {
            threshold_value: Some(30.0),
        }));
        assert_eq!(config.threshold, 30.0);
    }
}
