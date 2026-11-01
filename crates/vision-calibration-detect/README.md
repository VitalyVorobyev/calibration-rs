# vision-calibration-detect

Calibration-target feature detectors (chessboard, charuco, puzzleboard,
ringgrid) plus a content-addressed [`DetectionCache`] keyed on
`(image_content_hash, detector_name, canonical_config_hash)`. The cache
makes solver iteration sub-second on a cache hit.

A `DetectorConfig` selects the detector and carries its typed config, and
detects in an 8-bit grayscale `GrayImageView`. Each detector is gated by its
own cargo feature (all on by `default`): `chessboard` and `charuco` (via
`chess-corners` + `calib-targets`), `puzzleboard` (via `calib-targets`), and
`ringgrid` (via the `ringgrid` crate). Neither `image` nor `serde_json`
appears in this crate's signatures.

The dataset runner in `vision-calibration-pipeline` picks the detector from a
dataset manifest and runs it through the cache; call a detector or the cache
directly only for custom workflows. What each detector returns, the cache key
and the per-detector options are in the book's
[Calibration Targets and Detection](https://vitalyvorobyev.github.io/calibration-rs/detection.html)
chapter.
