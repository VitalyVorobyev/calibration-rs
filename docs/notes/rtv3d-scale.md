# rtv3d absolute metric scale — Q5-RTV3D-SCALE

Settles the scale discrepancy recorded in `docs/backlog.md` (Q5) and the
2026-06-11 validation report: our calibration measured a hexagon-neighbor
spacing of "90.1 mm" at a 5.2 mm ChArUco cell, while the legacy oracle
(`artifacts.json`) "implied ~98.5 mm" — an apparent ~8.6–9.3 % gap that
prompted speculation about a cell-size mix-up (5.2 vs 4.75/4.8/5.69 mm).

**Verdict: no real scale ambiguity. The gap was a pipeline-stage bookkeeping
artifact in the oracle-comparison tool, not a metric error.** Once the
comparison uses the *final* laser-informed joint-BA extrinsics (the
calibration this project already ships and validates on all other criteria)
instead of the intermediate hand-eye-stage extrinsics, our hexagon measures
98.21 ± 0.41 mm — matching the oracle's own healthy-camera hexagon
(98.13 ± 1.10 mm) to **0.08 %**. 5.2 mm is confirmed as the correct cell
size; no board-spec correction is needed.

## 1. Where "~98.5 mm" comes from (oracle side)

`artifacts.json`'s `extrinsic[i].camera_se3_sensor` is `T_{cam_i}_{rig}` (rig
frame = cam 0, matching our convention). Inverting each and taking pairwise
distances between *mechanical neighbors* (the rig is a hexagon: 0-1-2-3-4-5-0)
gives:

| edge | distance (mm) |
|---|---:|
| 0-1 | 96.25 |
| 1-2 | 98.47 |
| 2-3 | 98.85 |
| 3-4 | 98.96 |
| 4-5 | 173.23 (touches cam 5) |
| 5-0 | 144.97 (touches cam 5) |

Cams 0–4 give four internally-consistent edges: mean **98.13 mm**, std
1.10 mm (1.1 % spread) — this is the precise source of the report's "~98.5 mm"
figure (previously eyeballed/rounded). The two edges touching cam 5 are wildly
off (145–173 mm vs the ~98 mm the other four edges agree on), consistent with
the already-known oracle cam 5 degeneracy (`fx = 51`, 127 px reprojection).
**The 98.5 mm figure never depended on cam 5 or cam 0's own suspect fx**
(cam 0's oracle fx = 1642 vs ~2030 elsewhere is a separate, already-flagged
oddity that doesn't enter a translation-norm calculation) — it is a clean,
self-consistent measurement from the oracle's 4 good cameras.

## 2. Where "90.1 mm" came from (our side) — and why it was wrong

`rtv3d_rig.rs`'s `compare_to_oracle()` printed the extrinsic-scale table using
`rig_export.cam_se3_rig` — the **hand-eye-stage** export, captured *before*
stage 3 (laser-plane recovery) and stage 4 (joint rig + hand-eye + laser-plane
BA). Re-deriving the same hexagon-neighbor-edge computation at that stage,
current dataset/code (2026-07-08):

| edge | hand-eye stage (mm) | joint-BA stage (mm) | oracle (mm) |
|---|---:|---:|---:|
| 0-1 | 88.95 | 98.84 | 96.25 |
| 1-2 | 88.78 | 98.07 | 98.47 |
| 2-3 | 88.41 | 98.44 | 98.85 |
| 3-4 | 89.59 | 98.31 | 98.96 |
| 4-5 | 88.79 | 97.50 | 173.23 |
| 5-0 | 88.73 | 98.13 | 144.97 |
| **mean ± std** | **88.88 ± 0.36 mm (0.40 %)** | **98.21 ± 0.41 mm (0.41 %)** | 98.13 ± 1.10 mm (clean 4) |

The hand-eye-stage hexagon (88.9 mm) is what the 2026-06-11 report rounded to
"90.1 mm" (the dataset/pipeline has moved slightly since; the exact value
drifts run-to-run with unrelated fixes — see §3 for why that drift itself is
evidence against a cell-size bug). Crucially: **the joint BA — which already
runs in this same example, and whose reprojection/laser numbers are what the
"beat the oracle" verdict is graded on — rescales the *entire* rig by a
uniform +10.5 % between the hand-eye stage and the final joint-BA stage**, and
lands within 0.08 % of the oracle's clean hexagon. The rig shape (hexagon
regularity, ~0.4 % edge spread) is preserved at both stages; only the overall
scale moves. This is the signature of an under-constrained scale direction
being resolved by additional metric information, not of noise or a different
local optimum.

**Likely mechanism** (not required to close this ticket, noted for
follow-up): the Scheimpflug tilt ↔ principal-point ↔ rig-pose valley already
documented in the 2026-06-11 report (`cx = −217` for cam 2 when seeded from
the oracle) gives the per-camera/rig-BA stage a shallow direction that trades
scale against tilt/pose without much reprojection cost. Stage 4 freezes the
per-camera tilts (already converged) and adds the laser point-to-plane term
(weight 1e4), which pins the remaining degree of freedom down. The
92-view-BA-only stage's scale is *not wrong because of a modeling bug*; it is
simply less metrically constrained than the full joint solve, and the
oracle-comparison tool was reading the weaker of the two.

## 3. The cell-size-confusion hypothesis — rejected

The dataset ships two cell-size fields for the *same* 22×22 board (both
`ncols`/`nrows` = 22, `marker_scale`/`marker_size_rel` = 0.75):

- `config.json` → `target.cellsize_mm = 5.2` (used throughout; matches the
  Q5 pipeline default).
- `board_charuco.json` → `cell_size_mm = 4.8` (already flagged wrong in the
  2026-06-11 report — a real internal metadata inconsistency, but unrelated
  to the oracle gap discussed here).

No file in the dataset (`spec.json`, `config.json`, `board_charuco.json`,
`dataset.json`, `artifacts.json`) contains any other cell-size value — in
particular neither 4.75 mm nor 5.69 mm (the values the ratio 98.5/90.1 ≈ 1.093
would imply if the gap were a pure cell-size scale factor) appear anywhere.
Those numbers were back-calculated from the ratio, not read from a file.

Two more facts argue against a fixed cell-size bug specifically:

1. **The ratio isn't constant.** 8.6 % (2026-06-11 report, rounded) → 9.3 %
   (90.1/98.5 exactly) → 10.4 % (88.88/98.13, current hand-eye stage) → 0.08 %
   (98.21/98.13, current joint-BA stage) as the *same* 5.2 mm cell size ran
   through different pipeline stages/commits over the past month. A cell-size
   mislabeling would produce a fixed ratio (5.2/4.8 = 1.0833, or 5.2/4.75 =
   1.0947) independent of which BA stage or commit is compared; this one
   moves with pipeline stage and vanishes entirely once the laser-informed
   stage is used.
2. **The joint-BA hexagon is internally tight** (0.41 % edge-length spread,
   as regular as the hand-eye-stage hexagon it superseded) — a real board
   mis-scale would show up as increased reprojection error or degraded
   hexagon regularity at the corrected scale, not as a clean rescale that
   also happens to match an independent oracle.

`artifacts.json`'s `meta` block is entirely empty (no date, device, or
`quick_version`), so its capture provenance can't be independently confirmed
— but its geometry (4 clean cameras, one known-broken) is not itself in
question; only the earlier *comparison* was reading the wrong stage of our
own pipeline.

## 4. What would definitively settle it beyond this analysis

A mechanical measurement of the physical camera-to-camera spacing on the rtv3d
head (calipers/CMM between two adjacent housings, or the drawing dimension if
one exists) would be the independent ground truth neither the oracle nor our
own BA can provide. Given the internal consistency demonstrated here (two
independent computations — ours and the oracle's — agreeing to 0.08 % at
5.2 mm), this is now a confirmation exercise rather than a scale-hunting one.

## 5. Reproduction

```bash
RTV3D_DATA_DIR=privatedata/rtv3d CELL_SIZE_MM=5.2 RTV3D_HANDEYE=eye_to_hand \
  cargo run --manifest-path crates/vision-calibration-examples-private/Cargo.toml \
  --example rtv3d_rig --release
```

The "hexagon neighbor-edge lengths" table (added to `compare_to_oracle()` in
this ticket) prints the per-edge ours-vs-oracle comparison and the mean ±
spread verdict directly; no external script is needed. `compare_to_oracle()`
also now sources the extrinsic-scale table from the joint-BA extrinsics (when
laser data is present) instead of the hand-eye-stage export, which is the
concrete code fix behind this note.

## Files touched

- `crates/vision-calibration-examples-private/examples/rtv3d_rig.rs` — the
  oracle-comparison scale table now prefers the joint-BA (laser-informed)
  extrinsics over the hand-eye-stage export; added the hexagon neighbor-edge
  diagnostic table.
- `privatedata/rtv3d/spec.json` — **local-only, gitignored** — `description`
  field annotated with this finding and a pointer to this note (no numeric
  field changed; the transcribed mount translations remain a valid bootstrap
  seed regardless of the ~10 % staleness, since seeded and generic
  initialization converge to the same optimum — S3-SPEC-EXTRINSICS,
  2026-07-02).
