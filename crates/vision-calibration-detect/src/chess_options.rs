//! Shared ChESS corner-stage options for chessboard-like detectors.
//!
//! Two thresholds, at two different stages of the same pipeline:
//!
//! 1. `threshold_value` gates the **corner detector** — which response peaks
//!    become corners at all.
//! 2. `min_corner_strength` gates the **grid builder** — which of those
//!    corners are strong enough to be trusted as lattice nodes.
//!
//! Both default to `calib-targets`' tuned values and only need overriding for
//! inputs at the edges of what those values were tuned for.

use calib_targets::chessboard::ChessboardParams;
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
    /// [`default_chess_config`]'s noise-floor cutoff (`15.0`).
    ///
    /// The threshold is always absolute. `chess-corners` 1.0 collapsed its
    /// former `Threshold::{Absolute, Relative}` enum into a single `f32`;
    /// there is no longer a "fraction of the image maximum" mode.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub threshold_value: Option<f32>,

    /// Minimum corner strength for a detected corner to enter the grid
    /// builder. `None` keeps the detector default (`33.0`).
    ///
    /// The default drops weakly-firing corners — defocused board edges and
    /// marker-bit saddles fire at ≈15–30 against a sharp board's ≈90+ — which
    /// are grid-consistent but low-confidence, and pollute the fit. That is
    /// the right trade whenever corners are plentiful.
    ///
    /// Set `0.0` to disable the filter when they are not. On small or soft
    /// image tiles the floor can remove a third of the board, and the
    /// resulting sparse, unevenly-covered view conditions far worse than the
    /// weak corners did — the loss shows up as a diverged camera, not as a
    /// slightly worse residual.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_corner_strength: Option<f32>,
}

/// Lower the corner-detector half of the override.
pub(crate) fn chess_config_for_override(
    override_cfg: Option<ChessCornersConfig>,
) -> DetectorConfig {
    let config = default_chess_config();
    match override_cfg.and_then(|cfg| cfg.threshold_value) {
        Some(threshold) => config.with_threshold(threshold),
        None => config,
    }
}

/// Lower the grid-builder half of the override, in place.
///
/// Takes `ChessboardParams` because that is where the floor lives for both
/// detectors that honour it — plain chessboard uses it directly, and ChArUco
/// reaches it through `CharucoParams::chessboard`.
pub(crate) fn apply_board_override(
    override_cfg: Option<ChessCornersConfig>,
    params: &mut ChessboardParams,
) {
    if let Some(strength) = override_cfg.and_then(|cfg| cfg.min_corner_strength) {
        params.min_corner_strength = strength;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn threshold_override_applies() {
        let config = chess_config_for_override(Some(ChessCornersConfig {
            threshold_value: Some(30.0),
            ..ChessCornersConfig::default()
        }));
        assert_eq!(config.threshold, 30.0);
    }

    #[test]
    fn min_corner_strength_override_applies() {
        let mut params = ChessboardParams::default();
        apply_board_override(
            Some(ChessCornersConfig {
                min_corner_strength: Some(0.0),
                ..ChessCornersConfig::default()
            }),
            &mut params,
        );
        assert_eq!(params.min_corner_strength, 0.0);
    }

    #[test]
    fn absent_override_keeps_detector_defaults() {
        assert_eq!(
            chess_config_for_override(None).threshold,
            default_chess_config().threshold
        );
        assert_eq!(
            chess_config_for_override(Some(ChessCornersConfig::default())).threshold,
            default_chess_config().threshold
        );

        let default_strength = ChessboardParams::default().min_corner_strength;
        for cfg in [None, Some(ChessCornersConfig::default())] {
            let mut params = ChessboardParams::default();
            apply_board_override(cfg, &mut params);
            assert_eq!(params.min_corner_strength, default_strength);
        }
    }
}
