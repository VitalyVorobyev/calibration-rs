//! Shared ChESS corner-stage options for chessboard-like detectors.
//!
//! The corner detector's `strategy` (ChESS or Radon), and two thresholds at
//! two different stages of the same pipeline:
//!
//! 1. `threshold_value` gates the **corner detector** — which response peaks
//!    become corners at all.
//! 2. `min_corner_strength` gates the **grid builder** — which of those
//!    corners are strong enough to be trusted as lattice nodes.
//!
//! All three default to `calib-targets`' tuned values and only need
//! overriding for inputs at the edges of what those values were tuned for.

use calib_targets::chessboard::ChessboardParams;
use calib_targets::core::DetectorConfig;
use calib_targets::detect::default_chess_config;
use serde::{Deserialize, Serialize};

#[cfg(feature = "schemars")]
use schemars::JsonSchema;

/// The `chess-corners` corner detector a chessboard-like detector runs.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum CornerStrategy {
    /// The ChESS response ([`default_chess_config`]). Fast; the default.
    #[default]
    Chess,
    /// The whole-image Radon response (`DetectorConfig::radon`). Several
    /// times slower than ChESS, and less biased at a corner's sub-pixel
    /// position on sharp, finely sampled boards (e.g. rendered ones).
    Radon,
}

/// ChESS corner extractor overrides.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(JsonSchema))]
#[serde(deny_unknown_fields, default)]
pub struct ChessCornersConfig {
    /// Corner detector. `None` keeps ChESS.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub strategy: Option<CornerStrategy>,

    /// Acceptance threshold on the corner response. `None` keeps the
    /// strategy's default.
    ///
    /// Its meaning depends on the strategy, as in `chess-corners`:
    /// - ChESS: an absolute floor on the raw response; a corner is kept when
    ///   its response exceeds it. The default is [`default_chess_config`]'s
    ///   noise-floor cutoff (`15.0`).
    /// - Radon: a fraction in `(0, 1]` of the frame's maximum response,
    ///   because the Radon score has no portable absolute scale. The default
    ///   is `chess-corners`' (`0.28`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub threshold_value: Option<f32>,

    /// Minimum corner strength for a detected corner to enter the grid
    /// builder. `None` keeps the detector default (`33.0`).
    ///
    /// The floor is on the strategy's response scale. Radon's scores are
    /// orders of magnitude larger than ChESS's, so the default never binds
    /// under Radon.
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
    let config = match override_cfg.and_then(|cfg| cfg.strategy) {
        None | Some(CornerStrategy::Chess) => default_chess_config(),
        Some(CornerStrategy::Radon) => DetectorConfig::radon(),
    };
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
    fn radon_strategy_selects_the_radon_detector() {
        let radon = chess_config_for_override(Some(ChessCornersConfig {
            strategy: Some(CornerStrategy::Radon),
            ..ChessCornersConfig::default()
        }));
        assert_eq!(radon, DetectorConfig::radon());

        // The threshold is still the caller's, read on Radon's relative scale.
        let tuned = chess_config_for_override(Some(ChessCornersConfig {
            strategy: Some(CornerStrategy::Radon),
            threshold_value: Some(0.1),
            ..ChessCornersConfig::default()
        }));
        assert_eq!(tuned, DetectorConfig::radon().with_threshold(0.1));

        let chess = chess_config_for_override(Some(ChessCornersConfig {
            strategy: Some(CornerStrategy::Chess),
            ..ChessCornersConfig::default()
        }));
        assert_eq!(chess, default_chess_config());
    }

    #[test]
    fn strategy_serializes_snake_case() {
        let cfg: ChessCornersConfig = serde_json::from_str(r#"{"strategy": "radon"}"#).unwrap();
        assert_eq!(cfg.strategy, Some(CornerStrategy::Radon));
        assert_eq!(
            serde_json::to_string(&cfg).unwrap(),
            r#"{"strategy":"radon"}"#
        );
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
