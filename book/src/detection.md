# Calibration Targets and Detection

The calibration workflows consume 2D-3D correspondences: a pixel in the image and the position of the same point on the target, in metres, in the target's own frame. Turning images into those correspondences is the job of the `vision-calibration-detect` crate. It ships four target detectors behind one interface and a content-addressed cache, so repeated runs over the same images never repeat the detection.

You rarely call a detector yourself. The [dataset runner](manifest.md) picks the detector from the manifest's `[target]` table, runs it over every image through the cache and assembles the problem input. This chapter is for choosing a target, understanding what each detector returns, and using the detectors or the cache directly.

## What a Detector Returns

Every detector turns one image into a list of `Feature` values:

```rust
pub struct Feature {
    pub image_xy: [f64; 2],   // pixel (column, row) in the source image
    pub world_xyz: [f64; 3],  // metres, in the target frame; the target plane is z = 0
}
```

A frame that simply contains no target is **not** an error: every detector returns an empty list. Errors (`DetectError`) are reserved for a malformed configuration (`Config`, `InvalidConfig`) and genuine backend failures (`Backend`), so a misconfigured board cannot masquerade as an unlucky image.

Every detector also passes its result through `reject_ambiguous_detection`. The targets are planar, so the image-to-target map is a homography and therefore injective: one pixel is one board point. A detection in which two correspondences share an image point has mislabelled its grid. The **whole detection is rejected** (an empty result), because a duplicated point is evidence that the grid walk lost track of the labels, and the labels that happen not to collide come from the same broken walk. The caller sees a skipped view, which every workflow handles.

## The Four Targets

The target is chosen by the manifest's `[target]` table, `kind` being one of the four names below. All lengths are in metres.

| `kind` | Fields | What it detects |
|---|---|---|
| `chessboard` | `rows`, `cols` (interior corners), `square_size_m` | Saddle corners of a plain checkerboard |
| `charuco` | `rows`, `cols` (squares), `square_size_m`, `marker_size_m`, `dictionary` | Chessboard corners identified by embedded ArUco markers |
| `puzzleboard` | `layout` (`"puzzle_<R>x<C>"`), `cell_size_m` | Saddles of a self-identifying chessboard variant |
| `ringgrid` | `pitch_m`, `rows`, `long_row_cols`, `marker_outer_radius_m`, `marker_inner_radius_m`, `marker_ring_width_m` | Centres of coded ring markers on a hexagonal lattice |

### Chessboard

`rows` and `cols` count **interior corners**: a 9x6 setting is a 10x7 grid of squares. `cols` corners run across the image and `rows` down; a board seen turned by 90 degrees is also accepted whole.

The underlying detector labels each view on its own: grid index `(0, 0)` is the detection's top-left corner **in that image**, and the corner positions are lifted to `(i * square_size_m, j * square_size_m, 0)`. A board seen whole and the same way up gets the same labels in every view. A partial board is labelled from its own top-left corner, so its target frame is shifted by the rows and columns out of view; a board seen turned by 90 degrees gets labels rotated with it.

That is harmless for single-camera intrinsics and the single-camera laserline device, which estimate one target pose per view: a shifted frame only changes that view's pose, and partial views near the image border are valuable for the distortion model. It is not harmless where views share one target frame: hand-eye (the target is fixed relative to the robot base or gripper) and every rig topology (the cameras of one view see one target pose). A shifted frame there contradicts the other views. On a three-camera hand-eye rig, keeping partial boards raises the worst camera's mean reprojection error from 0.29 px to 4.7 px.

So in the shared-frame topologies the [dataset runner](manifest.md#running-a-manifest) keeps a chessboard detection only when it **spans the whole board as declared**: corners on all four edges, `cols` across and `rows` down. Interior corners the detector missed do not matter, since the edges still fix the labels. Every other view is dropped, and the result's `rejected` list records it with its reason (`PartialChessboard` or `TurnedChessboard`). A camera left with no whole-board view is an error (`NoWholeChessboardViews`) before any solve. There is no switch to turn this off, because a view that fails the check would give a wrong result.

Two cases cannot be detected, because their labels look the same as an upright board's: a board seen upside down (turned by 180 degrees), and a square board (`rows == cols`) seen turned by 90 degrees. For a shared frame, capture the board the same way up in every image, and prefer a non-square board. To use partial views in a shared frame, use a ChArUco or PuzzleBoard target. Their corners are identified by codes printed on the board, so a partial view is labelled correctly.

### ChArUco

A ChArUco board interleaves a chessboard with ArUco markers, and each corner is identified by the markers around it. A partly visible board therefore still yields correctly labelled corners, and the result is **sparse**: occluded or unidentified cells simply contribute no feature, so the per-view feature count varies.

`rows` and `cols` count squares (not corners). `marker_size_m` must be positive and at most `square_size_m`. `dictionary` is an ArUco dictionary name such as `"DICT_4X4_1000"`; the layout is checked before any image is read, because a larger board needs more marker codes than a small dictionary holds (a 12x12 board needs 72, which `DICT_4X4_50` cannot supply). `validate_dictionary` and `validate_charuco_layout` expose these checks. The marker layout follows the OpenCV ChArUco convention, so coordinates are indexed the way an OpenCV-printed board expects.

### PuzzleBoard

A PuzzleBoard is a self-identifying chessboard: each saddle decodes to a unique grid index from a local code window, so a single partial view yields globally consistent correspondences. The manifest names the layout as `"puzzle_<rows>x<cols>"` (the prefix is optional, `"130x130"` also parses); the dimensions are all the detector needs. Coordinates are in the board's own frame with its origin at the top-left, like the chessboard and ChArUco detectors, and decoded saddles outside the printed board are discarded.

Two fixed parts of the recipe are not configurable: the image is upscaled 2x before decoding (saddle decoding degrades on small or blurred tiles; pixel coordinates are divided back down), and the search is restricted to the declared board.

### Ring Grid

A coded ring grid is a hexagonal lattice of ring markers, each carrying a 16-sector code. A decoded marker contributes one correspondence at its centre when its code maps to a known board position. `rows` is the number of marker rows and `long_row_cols` the number of columns of the longest (even-indexed) row; the shorter rows follow from the lattice. The three radii describe one marker (outer radius, inner radius, ring-band width) and are validated against `pitch_m` when the board layout is built. Detection runs in adaptive-scale mode, since the marker pixel size is unknown for an arbitrary image.

The detector fits a flat shape: a hexagonal lattice of coded rings. Rectangular lattices and plain uncoded annuli are not exposed.

## Corner Detector Overrides

The chessboard and ChArUco detectors both start from a ChESS-family corner detector. Three optional overrides live in the manifest's `[detector.chess_corners]` table (and in each detector's `chess_corners` field). They are hashed into the detection-cache key, so changing any of them re-runs detection.

| Field | Meaning |
|---|---|
| `strategy` | `"chess"` (the default) or `"radon"`. Radon is several times slower and less biased at a corner's sub-pixel position on sharp, finely sampled boards such as rendered ones. |
| `threshold_value` | Floor on the corner response, gating which response peaks become corners at all. Absolute under ChESS; under Radon a fraction in `(0, 1]` of the frame's maximum response. |
| `min_corner_strength` | Floor on corner strength for entering the grid builder, gating which corners are trusted as lattice nodes. Defaults to `33.0`, which drops weak, defocused corners. |

**Set `min_corner_strength = 0.0` on small or soft image tiles.** There the floor can remove a third of the board, and the resulting sparse, unevenly covered view conditions worse than the weak corners did; the loss shows up as a diverged camera, not as a slightly worse residual.

`detector.min_features_per_view` (a manifest field, not a detector option) sets the minimum number of features for a view to be kept. It defaults to 4, the homography minimum. Raise it for detectors whose sparse detections are unreliable, such as ring-grid decodes with fewer than 8 markers. A plain chessboard in a shared-frame topology needs no floor: the whole-board rule above already applies.

## Calling a Detector Directly

The detectors implement the sealed `Detector` trait: the set of detectors is closed and owned by the crate. A detector takes its configuration as JSON, which is what lets the runner dispatch from the manifest and derive cache keys without knowing the detector type:

```rust
use vision_calibration_detect::{ChessboardConfig, ChessboardDetector, Detector};

let config = serde_json::to_value(ChessboardConfig {
    rows: 17,
    cols: 28,
    square_size_m: 0.02,
    chess_corners: None,
})?;
let image = image::open("data/kuka_1/01.png")?;
let features = ChessboardDetector.detect_json(&image, &config)?;
println!("{} corners", features.len());
```

`ChessboardConfig`, `CharucoConfig`, `PuzzleboardConfig` and `RinggridConfig` reject unknown fields when deserialized, so a typo in a hand-written JSON config fails loudly. Each detector sits behind its own cargo feature (`chessboard`, `charuco`, `puzzleboard`, `ringgrid`; all on by default), so a build that needs one target does not compile the others. The facade crate's `vision_calibration::detect` module re-exports the cache, `Feature` and the error types; the detectors themselves are reached through `vision-calibration-detect`.

To get features into a calibration without a manifest, build the problem input from `Feature` values yourself: `world_xyz` and `image_xy` are exactly the `points_3d` and `points_2d` of a `CorrespondenceView`.

The facade's `dataset_runner::detect_target` runs a detector from a manifest's `target` (and optional `detector`) section instead of a detector config, lowered exactly as a manifest run lowers it:

```rust
use vision_calibration::dataset::TargetSpec;
use vision_calibration::dataset_runner::detect_target;

let target = TargetSpec::Chessboard { rows: 17, cols: 28, square_size_m: 0.02 };
let features = detect_target(&image::open("data/kuka_1/01.png")?, &target, None)?;
```

### In Python

`detect_target` takes a `(H, W)` `uint8` grayscale numpy image and the target as its manifest payload, and returns a `Features` table: `image_xy` `(N, 2)` and `world_xyz` `(N, 3)` arrays. An impossible target (an unknown ChArUco dictionary) raises `RunError` with `kind == "invalid_target_config"`; an image without a usable board gives an empty table.

```python
import numpy as np
from PIL import Image
import vision_calibration as vc

target = {"kind": "chessboard", "rows": 17, "cols": 28, "square_size_m": 0.02}
views = []
for path in ["data/kuka_1/01.png", "data/kuka_1/02.png", "data/kuka_1/03.png"]:
    gray = np.asarray(Image.open(path).convert("L"))
    features = vc.detect_target(gray, target)
    print(path, len(features), "corners")
    views.append(vc.PlanarView(observation=features.to_observation()))
result = vc.run_planar_intrinsics(vc.PlanarIntrinsicsInput(views=views))
```

`vc.reject_ambiguous_detection(features)` returns the detection, or an empty one when two features share an image point.

## The Detection Cache

Detection dominates the cost of iterating on a calibration: the solvers take milliseconds to seconds, the detectors seconds per image. The `DetectionCache` trait stores detections so a rerun with the same images and the same detector parameters reads files instead of detecting again.

A cache entry is keyed by `CacheKey`:

- **the image content**: a 64-bit FNV-1a digest of the raw file bytes. File modification time is deliberately not used, so moving or copying an image does not invalidate its entry;
- **the detector name**: `"chessboard"`, `"charuco"`, ... (laser extraction uses a `laser:<name>` namespace);
- **the configuration**: a hash of the detector configuration serialized to JSON with sorted keys, together with the crate's release and the detector's output revision (`CacheKey::for_detector`). Semantically equal configurations collide on purpose; changing any detector parameter, upgrading the detector stack or bumping a detector's output revision gives a new key and so recomputes, while the entries of the other detectors stay valid.

```rust
use vision_calibration::detect::{CachedFeatures, DetectionCache, FsDetectionCache, CacheKey};
use vision_calibration_detect::{ChessboardDetector, Detector};

let cache = FsDetectionCache::new("cache/kuka_1");   // created on first write
let bytes = std::fs::read("data/kuka_1/01.png")?;
let key = CacheKey::for_detector(&bytes, &ChessboardDetector, &config);

let features = match cache.get(&key)? {
    Some(hit) => hit.features,
    None => {
        let features = ChessboardDetector.detect_json(&image::load_from_memory(&bytes)?, &config)?;
        cache.put(&key, &CachedFeatures { features: features.clone() })?;
        features
    }
};
```

`FsDetectionCache` writes one JSON file per entry under its root, named `<image>-<detector>-<config>.json` with a portable file-name encoding that is valid on Linux, macOS and Windows. Entries are written atomically, so the cache is safe under concurrent readers and writers. The dataset runner takes any `&dyn DetectionCache`, so an application can supply its own store; every `build_*_input` converter also has a `force_redetect` flag that bypasses lookups.

In Python, `vc.FsDetectionCache(root)` is the filesystem cache, passed as `cache=` to `calibrate_dataset` and the `build_*_input` functions (without one, every image is detected). Its `get(key)` and `put(key, features)` take a `CacheKey`; `CacheKey.from_inputs(image_bytes, detector, config)` keys entries you manage yourself.

## Choosing a Target

The detectors differ in what a *partial* view gives you:

- A **chessboard** is labelled from its own top-left corner in each image, so only complete views of the board are safe in a rig or hand-eye run (see above).
- **ChArUco**, **PuzzleBoard** and **ring-grid** targets identify every corner or marker, so any visible part of the board is usable and the target frame is the same in every view. This matters when a robot occludes the board or the board is larger than the field of view of some cameras.

Whatever the target, the accuracy of the calibration is bounded by the accuracy of the detected points. The reprojection floor on real data is usually set by the detector and the print quality, not by the solver. The [per-feature residuals](per_feature_residuals.md) show whether the error is spread evenly or concentrated on a few views or corners.
