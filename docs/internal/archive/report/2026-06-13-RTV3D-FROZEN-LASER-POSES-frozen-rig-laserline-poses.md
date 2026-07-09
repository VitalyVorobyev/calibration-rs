# RTV3D-FROZEN-LASER-POSES - Frozen Rig Laserline Poses

## Scope

Fixed the frozen `RigLaserlineDevice` dataset runner path so it preserves the
upstream `RigHandeyeExport.rig_se3_target` poses when they are available. Older
upstream exports still fall back to recomputing the hand-eye chain, but now use
the upstream per-view robot pose deltas with the same left-multiplied convention
as the optimizer.

This addresses the rtv3d app symptom where the target observations matched the
hand-eye export but projected pixels shifted coherently, yielding a misleading
~54 px mean reprojection diagnostic in the laser-plane preset.

## Files Changed

- `crates/vision-calibration-pipeline/src/dataset_runner/laser.rs`
- `docs/backlog.md`

## Validation Run

- PASS: `cargo test -p vision-calibration-pipeline dataset_runner::laser::tests::rig_laserline_ -- --nocapture`

## Follow-ups / Remaining Risks

- Re-run the app ignored rtv3d laser test after the joint app topology lands.
- The frozen `RigLaserlineDevice` path remains a plane-only diagnostic; rtv3d
  quality calibration should use the joint hand-eye laserline path.
