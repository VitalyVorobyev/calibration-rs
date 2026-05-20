# ADR 0017: Content-Addressed Detection Cache

- Status: Accepted
- Date: 2026-05-02

## Context

The diagnose-first viewer (B0–B2) is read-only: every iteration round-
trips through `cargo run --example` to produce a fresh export. The new
"complete workflow in the app" commitment (ADR 0016) introduces
in-process re-runs, which exposes a latency issue: at typical detection
speeds (chessboard ≈ 200 ms / image, ringgrid slower), a 6-camera ×
20-pose dataset is 24 s of detection per Run click. Without a cache,
the in-app loop is _slower_ than the terminal one (where engineers
manually serialise detections to JSON and only re-run the solver),
which would shred the credibility of the "fast iteration" promise on
the first real dataset.

The grill session settled the principle ("cache aggressively"); this
ADR pins the contract.

## Decision

### 1. Cache key

`(image_content_hash, detector_name, canonical_config_hash)`:

- `image_content_hash` — 64-bit FNV-1a of the raw image bytes. Cheap,
  allocation-free, and crucially **never** invalidates by mtime —
  copying / moving / `touch`-ing an image must not invalidate its
  cached detections.
- `detector_name` — the `Detector::name()` string (e.g. `"chessboard"`).
  Cleanly partitions entries between detectors so adding a new detector
  in PR 2 doesn't touch existing entries.
- `canonical_config_hash` — 64-bit FNV-1a of the detector config
  serialised to JSON _with sorted keys_, so semantically identical
  configs (e.g. `{"a":1,"b":2}` vs `{"b":2,"a":1}`) collide as
  intended.

The composite key is encoded as `<image>-<detector>-<config>.json` for
filesystem-safe addressing under a per-dataset cache root.

### 2. Trait + filesystem default impl

```rust
pub trait DetectionCache: Send + Sync {
    fn get(&self, key: &CacheKey) -> Result<Option<CachedFeatures>, CacheError>;
    fn put(&self, key: &CacheKey, value: &CachedFeatures) -> Result<(), CacheError>;
}
```

- `Send + Sync` so the runner can share one instance across the
  per-image loop running on a tokio blocking task.
- Two methods only — `get` returns `Ok(None)` on miss; `put` is
  responsible for atomic-rename writes (tmp-file + rename pattern,
  safe on every filesystem we target).

The default `FsDetectionCache` writes one JSON file per entry under
`<root>/<key>.json`. In production the runner points it at
`~/.cache/calibration-rs/<dataset_id>/`; tests use `tempfile::tempdir()`.

### 3. Designed in from PR 1

Every detector entry point in the runner goes through the cache from
day one. Bolting a cache on later means changing every call site
twice; the upfront cost is ~1 file (the `cache.rs` module) and zero
extra effort for the runner because `cache.get_or_detect` is the
natural shape anyway.

### 4. Force-redetect escape hatch

A single `force_redetect: bool` flag on the runner (and exposed in the
Run workspace UI in PR 3) bypasses cache reads for clean runs without
clobbering existing entries. Used for "I changed something inside the
detector implementation, not its config" cases.

### 5. Persistence boundary

The cache files are intentionally **disposable**: no migration story,
no version field. Bumping the schema of `Feature`, `CachedFeatures`,
or `CacheKey` invalidates the on-disk corpus by design — content-
addressed caches don't need migrations because regenerating them is
free (just a re-run with the new code). Pre-1.0 we exploit this; post-
1.0 we add a version bump if a real backwards-compat case appears.

## Consequences

- Sub-second iteration on cache hit — the only time the user pays the
  detection cost is the first run on a new dataset or after a
  detector-config change.
- Re-running with the same params is essentially free; A/B comparisons
  become cheap.
- The cache is shared between the in-app runner and the existing
  `cargo run --example` paths once the examples migrate (a bonus, not
  required for PR 1).
- The 64-bit hash is not collision-proof — but we're not defending
  against adversarial collisions, only accidental ones, and FNV-1a is
  more than enough for a dataset's worth of entries.
- Cache root is per-dataset to keep entries scoped; deleting a
  dataset's cache is a single `rm -rf`.

## Status of work

- ✅ `DetectionCache` trait, `CacheKey`, `CachedFeatures`, and
  `FsDetectionCache` in `vision-calibration-detect`.
- ✅ 5 cache unit tests (key-determinism, key-changes-on-change,
  fs roundtrip, miss-on-unknown-root).
- ✅ Wired into `pipeline::dataset_runner::build_planar_input`.
- ⏳ Per-dataset cache root selection in the Tauri runner (PR 1 task #8).
