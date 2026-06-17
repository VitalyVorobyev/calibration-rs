# P5-STAGE-TIMING — per-stage timing in the bench (2026-06-16)

## Context

The [from-scratch profiling report](2026-06-16-perf-from-scratch-rig-profiling.md)
needs a per-stage cost breakdown to target the joint-BA work (P3:
autodiff-Jacobian vs `JᵀJ` assembly vs linear solve). But the bench `Timing`
struct lumped **every** optimization stage into a single `optimize_ms` — for the
rig pipelines that's per-camera intrinsics BA + rig init + rig BA + hand-eye init
+ hand-eye BA collapsed into one number, hiding exactly the split P3 cares about.

## What changed

### `record.rs` — additive, back-compatible `StageTiming`

- New `StageTiming` struct: six `Option<u64>` fields (`intrinsics_init_ms`,
  `intrinsics_optimize_ms`, `rig_init_ms`, `rig_optimize_ms`, `handeye_init_ms`,
  `handeye_optimize_ms`). A field is `None` for pipelines that don't run that
  stage; the present fields sum to `optimize_ms`.
- `Timing` gains `stages: Option<StageTiming>`, `#[serde(default,
  skip_serializing_if = "Option::is_none")]`. **No `BENCH_SCHEMA_VERSION` bump:**
  the field is additive and optional — pre-P5 records (no `stages` key)
  deserialize to `None`, and a `None`-stages record serializes byte-identically
  to a legacy one (the key is omitted). Existing frozen records and the registry
  stay comparable.

### `run.rs` — split the rig optimize blocks + DRY helper

- New `ms_since(Instant) -> u64` helper replaces the
  `start.elapsed().as_millis() as u64` idiom repeated across the runners.
- `run_rig_extrinsics` and `run_rig_handeye` now time each optimize sub-stage
  individually (3 and 5 stages respectively) and attach a populated
  `StageTiming` (including `intrinsics_init_ms` from the existing init timer).
  `optimize_ms` is now the saturating sum of the stage timers (was one wall
  measurement — numerically the same up to sub-ms rounding).
- The single-stage runners (`run_planar_intrinsics`, `run_single_cam_handeye`)
  carry `stages: None`.

## Verification

- `cargo test -p vision-calibration-bench` — 22 tests green, including a new
  `timing_stages_back_compat_and_roundtrip` that asserts (a) legacy JSON without
  `stages` deserializes to `None`, (b) a populated `StageTiming` round-trips, and
  (c) a `None`-stages `Timing` omits the key on serialization.
- `cargo clippy -p vision-calibration-bench --all-targets --all-features
  -D warnings` clean.

## Follow-ups

- Surfacing the per-stage split in the dashboard / `report` rendering is a small
  UI follow-up; the data is now captured in every rig record, which is what P3
  needs.
- The `rtv3d_ref_rig` example already prints per-stage timing ad-hoc; it could
  adopt the same `ms_since` idiom, but it lives in `examples-private` and is out
  of the bench crate's scope.
