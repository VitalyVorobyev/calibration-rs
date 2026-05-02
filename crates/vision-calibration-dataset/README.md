# vision-calibration-dataset

Canonical input-data manifest (`DatasetSpec`) for the calibration-rs
workflow. The manifest is the single on-disk wire format users author
(by hand or via AI heuristics) to describe a foreign dataset's layout
without copying or renaming images.

See ADR 0016 for design rationale. Per-problem-type converters
(`DatasetSpec` → existing `*Input` IR) live in
`vision-calibration-pipeline`.
