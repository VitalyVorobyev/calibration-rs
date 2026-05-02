# vision-calibration-detect

Calibration-target feature detectors (chessboard, charuco, puzzleboard,
ringgrid) plus a content-addressed [`DetectionCache`] keyed on
`(image_content_hash, detector_name, canonical_config_hash)`. The cache
makes solver iteration sub-second on cache hit; see ADR 0017.

PR 1 ships the chessboard detector only (wrapping `chess-corners` +
`calib-targets`). PR 2 adds charuco / puzzleboard / ringgrid.
