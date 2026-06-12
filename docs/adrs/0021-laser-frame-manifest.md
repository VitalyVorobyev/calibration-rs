# ADR 0021: Laser-Frame Dataset Manifest

- Status: Accepted
- Date: 2026-06-12

## Context

The Run workspace covers five of the seven topologies; `LaserlineDevice`
and `RigLaserlineDevice` still hit an `unsupported_topology` arm because
the `DatasetSpec` manifest (ADR 0016) cannot describe laser data. Three
gaps block them:

1. **Laser images.** Laserline calibration consumes per-view laser-line
   pixels extracted from a *second* image per `(view, camera)` slot —
   the laser frame, captured with the line projector on (rtv3d:
   `target_N.png` + `laser_N.png` pairs).
2. **Laser extraction.** The validated subpixel extractor lives in
   [`vision-metrology`](https://github.com/VitalyVorobyev/vision-metrology),
   which is **not on crates.io**. `vision-calibration-detect` and
   `-pipeline` are published crates; a git dependency — even optional —
   would break `cargo publish`.
3. **Upstream calibration.** `RigLaserlineDevice` consumes a *frozen*
   rig hand-eye result (`RigUpstreamCalibration`). The manifest needs a
   way to reference it, and the runner needs the per-view
   `rig_se3_target` poses, which the frozen export alone does not carry
   for the laser dataset's views.

Additionally, the rtv3d robot-pose file (`poses.json`) stores each pose
as a nested 4×4 matrix field (`tcp2base`), which the tabular
`PoseColumnMap` (scalar columns only) cannot express.

## Decision

### 1. Manifest extensions (`vision-calibration-dataset`)

- `CameraSource.laser_images: Option<ImagePattern>` — per-camera laser
  image source. The camera's `roi_xywh` applies to laser frames too
  (same physical sensor; rtv3d's 6-tile strips reuse one file per view
  with per-camera ROIs, exactly like the target frames).
- `DatasetSpec.laser: Option<LaserExtractionSpec>` — extraction
  parameters: `scan_axis` (`cols` | `rows`, default `cols`), `sigma`
  (1.2), `pos_thresh` / `neg_thresh` (4.0), `min_points` (20). Like
  detector params, these are algorithm tuning with safe defaults — an
  omitted `[laser]` table is **not** an ADR 0019 ambiguity.
- `DatasetSpec.upstream_calibration: Option<PathBuf>` — path (relative
  to the manifest dir) to a frozen `RigHandeyeExport` JSON. Required
  for `RigLaserlineDevice`, rejected elsewhere.
- `RobotPoseSource.matrix_field: Option<String>` — for `json` / `jsonl`
  pose files whose rows carry the whole pose as one nested 4×4 (or
  flat-16) array field. Mutually exclusive with `columns`; requires
  `rotation_format = matrix4x4_row_major`. Translation comes from the
  matrix's fourth column (scaled by `translation_units`), mirroring
  `rowmajor4x4`. No `pose_id` ⇒ `by_index` pairing only.

Validation (fail-fast, ADR 0019): laser fields on a non-laser topology
are rejected, not ignored; `LaserlineDevice` requires `laser_images` on
its camera; `RigLaserlineDevice` requires `laser_images` on **every**
camera (a camera without laser views has an unconstrained plane), plus
`robot_poses`, `pose_convention`, and `upstream_calibration`.

### 2. Laser-frame pairing

Laser images pair with target views through the existing
`pose_pairing` mechanism — no second pairing config:

- `by_index` — the camera's laser list must have exactly one image per
  paired view (count match enforced).
- `shared_filename_token` — the same regex/group extracts the view
  token from laser filenames. A laser token with no target view is an
  error (the regex is almost certainly wrong); a target view without a
  laser image is a gap (`None` laser slot for rigs; dropped view for
  the single-camera topology, which requires laser pixels per view).

### 3. Injected laser extraction (the publishing constraint)

The pipeline defines an **open** trait (unlike the sealed `Detector`):

```rust
pub trait LaserPixelExtractor: Send + Sync {
    fn name(&self) -> &str; // stable id, part of the cache key
    fn extract(&self, image: &DynamicImage, spec: &LaserExtractionSpec)
        -> anyhow::Result<Vec<[f64; 2]>>;
}
```

The laser converters take `&dyn LaserPixelExtractor`. The pipeline owns
everything around the call: bytes → cache lookup → decode → ROI crop →
extract → lift to source coordinates → cache store. Implementations:

- **Tauri app** (`app/src-tauri`, unpublished): wraps
  `vision_metrology::LaserExtractor` via a git dependency.
- **Tests**: a deterministic fake.

Cached laser pixels reuse the ADR 0017 cache with detector name
`laser:<extractor-name>` and the canonical JSON of
`LaserExtractionSpec` (+ `_roi`) as the config hash. Pixels are stored
as `Feature`s with `world_xyz = [0,0,0]` — laser points have no target
coordinates; the zero is documented, not meaningful.

### 4. Upstream calibration loading

`build_rig_laserline_device_input` reads the `RigHandeyeExport` JSON,
checks its camera count against the manifest, computes per-view
`rig_se3_target` from the laser dataset's **own robot poses** through
the frozen hand-eye chain —

- `EyeInHand`: `T_R_T = T_G_R⁻¹ · T_B_G⁻¹ · T_B_T`
- `EyeToHand`: `T_R_T = T_R_B · T_B_G · T_G_T`

— and calls `RigHandeyeExport::to_upstream_calibration`. The laser
views therefore do **not** need to match the upstream calibration's
views; only the robot must revisit poses with the target in view.
Pinhole rigs work (zero Scheimpflug tilt = identity sensor).

### 5. Export manifests

`LaserlineDeviceExport` gains `image_manifest` (the last `*Export`
without one). Both laser exports get the manifest spliced by the Tauri
runner from the **target** frames. Laser-frame entries in
`ImageManifest` are deferred to B-laser (the visualization slice) —
`FrameRef` has no frame-kind discriminator yet, and nothing renders
laser pixels today.

## Consequences

- The laser topologies run end-to-end in the Run workspace; B-laser
  (overlay + plane residual visualization) is unblocked.
- Published crates stay free of non-crates.io dependencies; if
  vision-metrology is ever published, a default `LaserPixelExtractor`
  impl can move into `vision-calibration-detect` behind a feature.
- Library consumers driving the laser converters must supply an
  extractor; there is deliberately no built-in fallback extractor in
  published code.
- The `laser:<name>` cache namespace means switching extractor
  implementations re-detects rather than serving stale pixels.
