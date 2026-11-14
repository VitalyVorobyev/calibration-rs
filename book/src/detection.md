# Calibration Targets and Detection

The calibration workflows consume 2D-3D correspondences: a pixel in the image and the position of the same point on the target, in metres, in the target's own frame. Turning images into those correspondences is the job of the `vision-calibration-detect` crate. It ships four target detectors behind one interface and a content-addressed cache, so repeated runs over the same images never repeat the detection.

You rarely call a detector yourself. The [dataset runner](manifest.md) picks the detector from the manifest's `[target]` table, runs it over every image through the cache and assembles the problem input. This chapter is for choosing a target, understanding what each detector returns, and using the detectors or the cache directly.

## What a Detector Returns

Every detector turns one image into a list of `Feature` values:

```rust,ignore
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
| `ringgrid` | `pitch_m`, `rows`, `long_row_cols`, `marker_outer_radius_m`, `marker_inner_radius_m`, `marker_ring_width_m`, optional `center_method` | Centres of coded ring markers on a hexagonal lattice |

### Chessboard

`rows` and `cols` count **interior corners**: a 9x6 setting is a 10x7 grid of squares. `cols` corners run across the image and `rows` down; a board seen turned by 90 degrees is also accepted whole.

The underlying detector labels each view on its own: grid index `(0, 0)` is the detection's top-left corner **in that image**, `+u` runs along the board axis closest to the image's `+x` and `+v` along the one closest to `+y`, and the corner positions are lifted to `(i * square_size_m, j * square_size_m, 0)`. A board seen whole and the same way up gets the same labels in every view. A partial board is labelled from its own top-left corner, so its target frame is shifted by the rows and columns out of view; a board seen turned gets labels turned with it.

That is harmless for single-camera intrinsics and the single-camera laserline device, which estimate one target pose per view: a shifted frame only changes that view's pose, and partial views near the image border are valuable for the distortion model. It is not harmless where views share one target frame: hand-eye (the target is fixed relative to the robot base or gripper) and every rig topology (the cameras of one view see one target pose). A shifted frame there contradicts the other views. On a three-camera hand-eye rig, keeping partial boards raises the worst camera's mean reprojection error from 0.29 px to 4.7 px.

So in the shared-frame topologies the [dataset runner](manifest.md#running-a-manifest) keeps a chessboard detection only when it **spans the whole board**: corners on all four edges, each edge holding at least half of its corners (and two). Interior corners the detector missed do not matter, since the edges still fix the labels; an edge held by a stray corner or two does not count, because the detector can pad a partial board that way. A partial board is dropped, and the result's `rejected` list records it with its reason (`PartialChessboard`). There is no switch to turn this off, because a view that fails the check would give a wrong result.

A whole board is the board's own grid up to a turn in its plane: two possible labellings for a rectangular board (`rows != cols`), four for a square one. What the runner does with the turn depends on what the labels must agree with.

- **The cameras of one view (every rig topology).** Each camera's labels are turned to agree with the view's first camera, so a hand-held board may be turned any way in any view. The runner picks the labelling from the board's direction in each image: two cameras of a rigid rig see the board turned by the same relative angle in every view, up to the board's symmetry (half a turn for a rectangular board, a quarter for a square one), and how the board moves through both images across the views tells that angle whole, so a camera mounted turned or upside down relative to the others is matched too. When the views are too few (fewer than three shared views) or the board moves by less than a square, the cameras are taken to be mounted the same way up. A detection whose direction does not single out one labelling, within 45 degrees for a rectangular board and 22.5 for a square one, is dropped as `AmbiguousChessboardTurn`.
- **One target frame through every view (hand-eye and the laser rigs).** The frame is the one camera 0 labels. The board must be seen as declared, `cols` across and `rows` down, by camera 0, or, in a view camera 0 misses, within 45 degrees of how camera 0 would see it; a view where it is turned by 90 degrees is dropped as `TurnedChessboard`. A board seen upside down by camera 0, or a square board seen turned by 90 degrees, cannot be told from upright: capture the board the same way up in every view, and prefer a non-square board.

A camera left with no usable chessboard view is an error (`NoWholeChessboardViews`) before any solve. To use partial views in a shared frame, use a ChArUco or PuzzleBoard target. Their corners are identified by codes printed on the board, so a partial view is labelled correctly.

### ChArUco

A ChArUco board interleaves a chessboard with ArUco markers, and each corner is identified by the markers around it. A partly visible board therefore still yields correctly labelled corners, and the result is **sparse**: occluded or unidentified cells simply contribute no feature, so the per-view feature count varies.

`rows` and `cols` count squares (not corners). `marker_size_m` must be positive and at most `square_size_m`. `dictionary` is an ArUco dictionary name such as `"DICT_4X4_1000"`; the layout is checked before any image is read, because a larger board needs more marker codes than a small dictionary holds (a 12x12 board needs 72, which `DICT_4X4_50` cannot supply). `validate_dictionary` and `validate_charuco_layout` expose these checks. The marker layout follows the OpenCV ChArUco convention, so coordinates are indexed the way an OpenCV-printed board expects.

### PuzzleBoard

A PuzzleBoard is a self-identifying chessboard: each saddle decodes to a unique grid index from a local code window, so a single partial view yields globally consistent correspondences. The manifest names the layout as `"puzzle_<rows>x<cols>"` (the prefix is optional, `"130x130"` also parses); the dimensions are all the detector needs. Coordinates are in the board's own frame with its origin at the top-left, like the chessboard and ChArUco detectors, and decoded saddles outside the printed board are discarded.

Two fixed parts of the recipe are not configurable: the image is upscaled 2x before decoding (saddle decoding degrades on small or blurred tiles; pixel coordinates are divided back down), and the search is restricted to the declared board.

### Ring Grid

A coded ring grid is a hexagonal lattice of ring markers, each carrying a 16-sector code. A decoded marker contributes one correspondence at its centre when its code maps to a known board position. `rows` is the number of marker rows and `long_row_cols` the number of columns of the longest (even-indexed) row; the shorter rows follow from the lattice. The three radii describe one marker (outer radius, inner radius, ring-band width) and are validated against `pitch_m` when the board layout is built. Detection runs in adaptive-scale mode, since the marker pixel size is unknown for an arbitrary image.

The detector fits a flat shape: a hexagonal lattice of coded rings. Rectangular lattices and plain uncoded annuli are not exposed.

The optional `center_method` picks how each marker's centre is estimated. The default, `"ellipse"`, takes the projective centre of the fitted inner and outer ellipses: fast, and accurate on sharp images. On a blurred or low-contrast image the edge points of a ring carry a blur- and code-dependent bias; `"photometric"` instead fits the marker's pixels with the printed ring and code pattern, blurred by a fitted Gaussian, and removes that bias. It costs roughly 2.5 times the detection time, and it returns only the markers the fit refined, so a marker that could not be fitted is dropped rather than mixed in at lower accuracy. It stays opt-in until its benefit on real calibration data is established. It does not report a per-point covariance, so the calibration stays unweighted.

```toml
[target]
kind = "ringgrid"
pitch_m = 0.008
rows = 15
long_row_cols = 14
marker_outer_radius_m = 0.0024
marker_inner_radius_m = 0.0012
marker_ring_width_m = 0.0004
center_method = "photometric"
```

## Corner Detector Overrides

The chessboard and ChArUco detectors both start from a ChESS-family corner detector. Four optional overrides live in the manifest's `[detector.chess_corners]` table (and in each detector's `chess_corners` field). They are hashed into the detection-cache key, so changing any of them re-runs detection.

| Field | Meaning |
|---|---|
| `strategy` | `"chess"` (the default) or `"radon"`. Radon is several times slower and less biased at a corner's sub-pixel position on sharp, finely sampled boards such as rendered ones. |
| `threshold_value` | Floor on the corner response, gating which response peaks become corners at all. Absolute under ChESS; under Radon a fraction in `(0, 1]` of the frame's maximum response. |
| `min_corner_strength` | Floor on corner strength for entering the grid builder, gating which corners are trusted as lattice nodes. Defaults to `33.0`, which drops weak, defocused corners. |
| `covariance` | Report each corner's position covariance; see [below](#corner-covariance-from-the-detector). Off by default. |

**Set `min_corner_strength = 0.0` on small or soft image tiles.** There the floor can remove a third of the board, and the resulting sparse, unevenly covered view conditions worse than the weak corners did; the loss shows up as a diverged camera, not as a slightly worse residual.

### Corner covariance from the detector

The chessboard and ChArUco detectors can report, for every corner, the covariance of its position: the image noise propagated through the corner localizer, plus the localizer's systematic-error model. Ask for it in the manifest:

```toml
[detector.chess_corners.covariance]
image_noise = { stated = 2.0 }
```

`image_noise` is `"estimate"` (the default when the section is present) or `{ stated = σ }`, the standard deviation of the image's pixel noise in gray levels of the 8-bit image (finite and at least 0), for example the standard deviation of the difference of two frames of a static scene divided by √2.

With the section present, every observation carries the covariances in `cov_px` (px², `[xx, xy, yy]` in `Feature::cov_px`), the residuals are whitened by them, and the report gives the unit-weight scale factor σ̂₀ instead of an RMS in pixels; see [Per-Point Covariance](uncertainty_point_covariance.md). Positions are identical with and without it.

- **JPEG and demosaiced images.** The per-image noise estimate is unreliable there, and the detector may then report no covariance. The run stops with an error naming the image and suggesting `image_noise = { known = σ }`. No corner is dropped to get past this.
- **Supported targets.** `chessboard` and `charuco`. The PuzzleBoard detector runs on an upscaled image, for which no covariance is reported, and ring grids have none; a manifest that sets `covariance` for another target is an error, raised before any detection.
- **Not recommended for calibration.** On real chessboard and ChArUco images, weighting a calibration by the detector's covariance did not improve its accuracy on held-out views, and on some cameras it made it slightly worse; its stated uncertainty was no closer to the real scatter either. It stays off unless you ask for it. The setting is part of the detection-cache key.

In Python, `Features.cov_px` is the `(N, 3)` array of covariances (`None` when not requested), and `Features.to_observation()` carries it into the observation.

`detector.min_features_per_view` (a manifest field, not a detector option) sets the minimum number of features for a view to be kept. It defaults to 4, the homography minimum. Raise it for detectors whose sparse detections are unreliable, such as ring-grid decodes with fewer than 8 markers. A plain chessboard in a shared-frame topology needs no floor: the whole-board rule above already applies.

## Calling a Detector Directly

A `DetectorConfig` names the target and carries its typed configuration, one variant per detector; the set is closed and owned by the crate. It detects in an 8-bit grayscale `GrayImageView` (row-major pixels; decode and convert the image with any library):

```rust,no_run
use vision_calibration::detect::{ChessboardConfig, DetectorConfig, GrayImageView};
# fn main() -> anyhow::Result<()> {

let config = DetectorConfig::Chessboard(ChessboardConfig {
    rows: 17,
    cols: 28,
    square_size_m: 0.02,
    chess_corners: None,
});
let gray = image::open("data/kuka_1/01.png")?.to_luma8();
let features = config.detect(GrayImageView::new(gray.width(), gray.height(), gray.as_raw())?)?;
println!("{} corners", features.len());
# Ok(())
# }
```

`ChessboardConfig`, `CharucoConfig`, `PuzzleboardConfig` and `RinggridConfig` reject unknown fields when deserialized, so a typo in a hand-written config fails loudly. Each detector sits behind its own cargo feature of `vision-calibration-detect` (`chessboard`, `charuco`, `puzzleboard`, `ringgrid`; all on by default), so a build that needs one target does not compile the others. The facade crate's `vision_calibration::detect` module re-exports the configs, the cache, `Feature` and the error types.

To get features into a calibration without a manifest, build the problem input from `Feature` values yourself: `world_xyz` and `image_xy` are exactly the `points_3d` and `points_2d` of a `CorrespondenceView`.

The facade's `dataset_runner::detect_target` runs a detector from a manifest's `target` (and optional `detector`) section instead of a detector config, lowered exactly as a manifest run lowers it:

```rust,no_run
use vision_calibration::dataset::TargetSpec;
use vision_calibration::dataset_runner::detect_target;
use vision_calibration::detect::GrayImageView;
# fn main() -> anyhow::Result<()> {

let target = TargetSpec::Chessboard { rows: 17, cols: 28, square_size_m: 0.02 };
let gray = image::open("data/kuka_1/01.png")?.to_luma8();
let features = detect_target(GrayImageView::new(gray.width(), gray.height(), gray.as_raw())?, &target, None)?;
# Ok(())
# }
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
result = vc.planar_intrinsics.run_calibration(vc.PlanarIntrinsicsInput(views=views))
```

`vc.reject_ambiguous_detection(features)` returns the detection, or an empty one when two features share an image point.

## The Detection Cache

Detection dominates the cost of iterating on a calibration: the solvers take milliseconds to seconds, the detectors seconds per image. The `DetectionCache` trait stores detections so a rerun with the same images and the same detector parameters reads files instead of detecting again.

A cache entry is keyed by `CacheKey`:

- **the image content**: a 64-bit FNV-1a digest of the raw file bytes. File modification time is deliberately not used, so moving or copying an image does not invalidate its entry;
- **the detector name**: `"chessboard"`, `"charuco"`, ... (laser extraction uses a `laser:<name>` namespace);
- **the configuration**: a hash of the detector configuration serialized to JSON with sorted keys, together with the crop the image was detected in, the crate's release and the detector's output revision (`CacheKey::for_detection`). Semantically equal configurations collide on purpose; changing any detector parameter, upgrading the detector stack or bumping a detector's output revision gives a new key and so recomputes, while the entries of the other detectors stay valid.

```rust,no_run
use vision_calibration::detect::{
    CacheKey, CachedFeatures, ChessboardConfig, DetectionCache, DetectorConfig, FsDetectionCache,
    GrayImageView,
};
# fn main() -> anyhow::Result<()> {
# let config = DetectorConfig::Chessboard(ChessboardConfig {
#     rows: 17,
#     cols: 28,
#     square_size_m: 0.02,
#     chess_corners: None,
# });

let cache = FsDetectionCache::new("cache/kuka_1");   // created on first write
let bytes = std::fs::read("data/kuka_1/01.png")?;
let key = CacheKey::for_detection(&bytes, &config, None); // no crop

let features = match cache.get(&key)? {
    Some(hit) => hit.features,
    None => {
        let gray = image::load_from_memory(&bytes)?.to_luma8();
        let features = config.detect(GrayImageView::new(gray.width(), gray.height(), gray.as_raw())?)?;
        cache.put(&key, &CachedFeatures { features: features.clone() })?;
        features
    }
};
# Ok(())
# }
```

`FsDetectionCache` writes one JSON file per entry under its root, named `<image>-<detector>-<config>.json` with a portable file-name encoding that is valid on Linux, macOS and Windows. Entries are written atomically, so the cache is safe under concurrent readers and writers. The dataset runner takes a `Detection` mode: `Detection::Cached(&cache)` reuses and stores entries, `Detection::Refresh(&cache)` detects every image again and overwrites its entry, and `Detection::Uncached` uses no cache. The cache is any `&dyn DetectionCache`, so an application can supply its own store.

In Python, `vc.FsDetectionCache(root)` is the filesystem cache, passed as `cache=` to `calibrate_dataset` and the `build_*_input` functions (without one, every image is detected). Its `get(key)` and `put(key, features)` take a `CacheKey`; `CacheKey.from_inputs(image_bytes, detector, config)` keys entries you manage yourself.

## Choosing a Target

The detectors differ in what a *partial* view gives you:

- A **chessboard** is labelled from its own top-left corner in each image, so only complete views of the board are safe in a rig or hand-eye run (see above).
- **ChArUco**, **PuzzleBoard** and **ring-grid** targets identify every corner or marker, so any visible part of the board is usable and the target frame is the same in every view. This matters when a robot occludes the board or the board is larger than the field of view of some cameras.

Whatever the target, the accuracy of the calibration is bounded by the accuracy of the detected points. The reprojection floor on real data is usually set by the detector and the print quality, not by the solver. The [per-feature residuals](per_feature_residuals.md) show whether the error is spread evenly or concentrated on a few views or corners.
