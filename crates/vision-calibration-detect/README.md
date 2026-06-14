# vision-calibration-detect

Calibration-target feature detectors (chessboard, charuco, puzzleboard,
ringgrid) plus a content-addressed [`DetectionCache`] keyed on
`(image_content_hash, detector_name, canonical_config_hash)`. The cache
makes solver iteration sub-second on cache hit; see ADR 0017.

All four detectors are wired behind the sealed `Detector` trait, each
gated by its own cargo feature (all on by `default`): `chessboard` and
`charuco` (via `chess-corners` + `calib-targets`), `puzzleboard` (via
`calib-targets`), and `ringgrid` (via the `ringgrid` crate). Add a new
detector by implementing `Detector` for a feature-gated module and
registering it in the `dataset_runner` dispatch — existing detectors are
untouched (open for extension).
