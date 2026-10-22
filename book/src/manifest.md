# The Dataset Manifest

A calibration dataset is a folder of images, usually a robot-pose file, and some knowledge only you have: what the target is, how big it is, which frame convention the robot controller uses. The **dataset manifest** (`DatasetSpec`, in the `vision-calibration-dataset` crate) is the one on-disk format that describes all of it. The manifest is *descriptive*, never prescriptive: your data stays where it is, and the manifest points at it.

The same manifest drives three front ends: the Rust [dataset runner](#running-a-manifest), the desktop app's Run workspace (which edits it in a schema-driven form) and the folder sniffer that writes a first draft.

## An Example

`data/kuka_1/dataset.toml` in the repository describes a single-camera hand-eye dataset: 30 chessboard images and a file with one robot pose per line.

```toml
version = 1
topology = "single_cam_handeye"
description = "kuka_1"

[[cameras]]
id = "cam0"
images = { kind = "glob", pattern = "*.png" }

[target]
kind = "chessboard"
rows = 17
cols = 28
square_size_m = 0.02

[robot_poses]
path = "RobotPosesVec.txt"
format = "rowmajor4x4"

[pose_pairing]
kind = "by_index"

[pose_convention]
transform = "t_base_tcp"
rotation_format = "matrix4x4_row_major"
translation_units = "m"
```

Paths are relative to the directory that holds the manifest unless they are absolute. TOML is the usual authoring format; the same structure is valid JSON, and a field reference generated from the JSON schema is in [Manifest Field Reference](manifest_reference.md).

## Topologies

`topology` selects the problem type the runner dispatches to. It describes the *data* (how many cameras, whether there are robot poses and laser frames), not the camera model: a tilted-sensor camera is a `planar_intrinsics` dataset whose calibration config names the sensor.

| `topology` | Problem type | Cameras | Robot poses | Laser images |
|---|---|---|---|---|
| `planar_intrinsics` | `PlanarIntrinsicsProblem` | exactly 1 | no | no |
| `single_cam_handeye` | `SingleCamHandeyeProblem` | exactly 1 | required | no |
| `laserline_device` | `LaserlineDeviceProblem` | exactly 1 | no | required |
| `rig_extrinsics` | `RigExtrinsicsProblem` | 2 or more | no | no |
| `rig_handeye` | `RigHandeyeProblem` | 2 or more | required | no |
| `rig_laserline_device` | `RigLaserlineDeviceProblem` | 2 or more | required | required |
| `rig_handeye_laserline` | `RigHandeyeLaserlineProblem` | 2 or more | required | required |

A field that the topology does not use is an error, not something silently ignored: `laser`, `upstream_calibration` and per-camera `laser_images` are rejected on the non-laser topologies, and `upstream_calibration` is also rejected on `laserline_device` and `rig_handeye_laserline`, which calibrate their own upstream geometry. `rig_laserline_device` is a second stage that needs `upstream_calibration`, the path to a frozen rig hand-eye export; see [Laser Datasets](manifest_laser.md).

## Cameras and Images

Each `[[cameras]]` entry has an `id` (unique; used in pairings and exports) and an `images` source:

- `{ kind = "glob", pattern = "cam0/*.png" }`: a shell glob, expanded relative to the manifest directory and sorted in natural order (`2.png` before `10.png`);
- `{ kind = "list", paths = ["a.png", "b.png"] }`: an explicit ordered list.

`roi_xywh = [x, y, w, h]` crops the image to a rectangle in source pixels before detection; the width and height must be non-zero. It is how one image holding several camera tiles becomes several cameras. Laser topologies add a `laser_images` source per camera with the same two forms.

## Target

`[target]` is tagged by `kind`: `chessboard`, `charuco`, `puzzleboard` or `ringgrid`, with the fields documented in [Calibration Targets and Detection](detection.md). `[detector]` holds optional detection overrides shared by the target families (`min_features_per_view`, and the `[detector.chess_corners]` table).

## Robot Poses

`[robot_poses]` points at the pose file; `[pose_convention]` says how to read it. **There are no defaults for the convention**: it is required whenever `robot_poses` is set, because a wrong guess (gripper in base or base in gripper, millimetres or metres) produces a plausible but wrong calibration.

| `format` | File | Notes |
|---|---|---|
| `rowmajor4x4` | headerless text, 16 floats per line | One row-major 4x4 homogeneous matrix per pose (KUKA-style exports). Needs `rotation_format = "matrix4x4_row_major"` and no `columns`. |
| `counted4x4` | headerless text, a count then that many 4x4 matrices | The layout of the robot-world hand-eye datasets of Tabb and Ahmad Yousef. Same requirements as `rowmajor4x4`. |
| `csv`, `json`, `jsonl` | tabular | Need a `columns` mapping, or `matrix_field` for rows that store the pose as one nested 4x4. |

The `[pose_convention]` table has three required fields:

- `transform`: which transform a row encodes. `t_base_tcp` is the gripper pose in the robot base frame (KUKA and UR controllers), `t_tcp_base` its inverse (some ABB exports), `t_world_tcp` and `t_tcp_world` the same two for a fixed world frame.
- `rotation_format`: `quat_xyzw`, `quat_wxyz`, `euler_xyz_deg`, `euler_xyz_rad`, `euler_zyx_deg`, `euler_zyx_rad`, `axis_angle_rad` or `matrix4x4_row_major`.
- `translation_units`: `m` or `mm`.

For tabular formats, `[robot_poses.columns]` maps your column names to `tx`, `ty`, `tz`, a `rotation` list whose length matches the rotation format, and an optional `pose_id`.

`[pose_pairing]` aligns images with each other and with pose rows:

- `{ kind = "by_index" }`: the i-th image of every camera pairs with pose row i; counts must match.
- `{ kind = "shared_filename_token", regex = "...", group = "..." }`: a regex with a named capture group extracts a view token from every filename (and the `pose_id` column of the pose file); the cameras and the poses must produce the same set of tokens.

## The Folder Sniffer

Writing a manifest by hand for a folder of 400 images is tedious, so `sniff_folder` writes a first draft:

```rust
use std::path::Path;
use vision_calibration::dataset::sniff_folder;

let spec = sniff_folder(Path::new("data/my_dataset"))?;
println!("{:?}", spec.unresolved);
```

The same function is the `generate-manifest` binary (needs the `cli` feature of `vision-calibration-dataset`) and the **Sniff folder** button of the app's Run workspace:

```bash
cargo run -p vision-calibration-dataset --features cli --bin generate-manifest -- data/my_dataset > data/my_dataset/dataset.toml
```

It looks only at file names and folder structure, never at image content, and infers **only what is structurally unambiguous**:

- cameras: images are grouped by parent directory; a single group is `cam0`, several groups take their directory names; each gets a `*.ext` glob for its dominant extension (`png`, `jpg`, `jpeg`, `tif`, `tiff`, `bmp`; dot-files are skipped);
- a robot-pose file: a non-image file whose name contains `pose`, classified by extension and content (`csv`, `json`, `jsonl`, a 16-float line for `rowmajor4x4`, a count-then-matrices file for `counted4x4`);
- `by_index` pairing, when every camera and the pose file have the same count;
- the topology from the camera count and the presence of poses (one camera, no poses: `planar_intrinsics`; one camera with poses: `single_cam_handeye`; several cameras with poses: `rig_handeye`; several without: `rig_extrinsics`).

## The Unresolved-Fields Rule

Everything that needs domain knowledge is **never guessed**. The sniffer leaves a placeholder in the manifest (an empty chessboard, an incomplete convention) and records the dotted path of the field in the manifest's `_unresolved` list:

```toml
_unresolved = ["pose_convention.transform", "pose_convention.translation_units", "target"]
```

The placeholders exist only so that the manifest deserializes and a form has something to edit; they are never trusted numerically. **The runner refuses a manifest whose `_unresolved` list is not empty.** You resolve each field (fill in the target, confirm the convention) and delete the list.

The sniffer flags, as applicable:

| Path | Why it cannot be inferred |
|---|---|
| `target` | board geometry is never inferred from images |
| `topology` | several cameras without poses can be a rig or independent single-camera calibrations |
| `pose_pairing` | camera and pose counts differ |
| `pose_convention.transform`, `pose_convention.translation_units` | a matrix file does not say which frame it expresses or its units (the sniffer pre-fills `t_base_tcp` and `m`, the common defaults, but still asks you to confirm them) |
| `robot_poses.columns`, `pose_convention` | tabular pose files need a column mapping and a full convention |

The same philosophy applies at run time. When the runner meets an ambiguity it cannot resolve from the manifest (for example a hand-eye manifest with no `pose_pairing`, or token pairing without a `pose_id` column) it returns `RunError::AskUser { field, prompt, suggestions }` instead of choosing. The app turns that into a dialog that asks you to pick one of the suggestions; library code decides what to do with it.

## Validation

`vision_calibration_dataset::validate(&spec)` checks the structural rules without touching the filesystem (it does not expand globs), and every runner converter calls it first. It returns a typed `ValidationError`: unresolved fields; a topology that needs robot poses and has none, or a camera count outside the topology's range; a missing `pose_convention`; duplicate camera ids; empty image sources; a rotation-column count that does not match the rotation format; tabular pose formats without `columns`; matrix formats with a `columns` mapping; degenerate ROIs; laser fields on a non-laser topology and missing laser sources on a laser one; out-of-range laser or detector parameters.

## Running a Manifest

The converters in `vision_calibration::dataset_runner` turn a `DatasetSpec` into the input of a problem type, running detection through a [detection cache](detection.md#the-detection-cache):

| Topology | Converter |
|---|---|
| `planar_intrinsics` | `build_planar_input` |
| `single_cam_handeye` | `build_single_cam_handeye_input` |
| `rig_extrinsics` | `build_rig_extrinsics_input` |
| `rig_handeye` | `build_rig_handeye_input` |
| `laserline_device` | `build_laserline_device_input` |
| `rig_laserline_device` | `build_rig_laserline_device_input` |
| `rig_handeye_laserline` | `build_rig_handeye_laserline_input` |

```rust
use std::path::Path;
use vision_calibration::dataset::DatasetSpec;
use vision_calibration::dataset_runner::build_single_cam_handeye_input;
use vision_calibration::detect::FsDetectionCache;
use vision_calibration::prelude::*;
use vision_calibration::single_cam_handeye::{SingleCamHandeyeProblem, run_calibration};

let base_dir = Path::new("data/kuka_1");
let spec: DatasetSpec = toml::from_str(&std::fs::read_to_string(base_dir.join("dataset.toml"))?)?;
let cache = FsDetectionCache::new("cache/kuka_1");

let run = build_single_cam_handeye_input(&spec, base_dir, &cache, false)?;
println!("{}/{} views usable", run.usable_views, run.total_views);

let mut session = CalibrationSession::<SingleCamHandeyeProblem>::new();
session.set_input(run.input)?;
run_calibration(&mut session)?;
```

Each converter checks the topology and target, validates the manifest and resolves the pairing *before* it reads a single image, so a wrong manifest fails in milliseconds with a message that names the field. A view whose detection yields fewer features than `detector.min_features_per_view` (default 4) is dropped. With a plain chessboard, the topologies whose views share one target frame (`single_cam_handeye` and every rig topology) also drop a view unless its corners span the whole board as declared, `cols` across and `rows` down. A partial or turned chessboard is labelled from its own top-left corner and would move that frame; [Calibration Targets and Detection](detection.md#chessboard) explains why, and recommends ChArUco or PuzzleBoard targets when you need partial views. `planar_intrinsics` and `laserline_device` estimate one target pose per view and keep partial boards.

Each result lists the detections it dropped in `rejected`, each with its camera, image and `RejectReason` (`TooFewFeatures`, `PartialChessboard` or `TurnedChessboard`). A camera left with no usable view is an error rather than a poor calibration: `NoWholeChessboardViews` when the whole-board rule dropped any of its views, `InsufficientUsableViews` otherwise. Each result also carries the source image path of every accepted view, which is how the app shows the right image under each residual.

## In Python

`calibrate_dataset` runs a manifest end to end: it reads the file (TOML or JSON), picks the converter and the session by the manifest's topology, calibrates with the given config and returns the result. The manifest's relative paths resolve against its directory.

```python
import vision_calibration as vc

cache = vc.FsDetectionCache("cache/kuka_1")
result = vc.calibrate_dataset("data/kuka_1/dataset.toml", vc.SingleCamHandeyeConfig(), cache=cache)
print(result.handeye_mode, f"{result.mean_reproj_error:.3f} px")
```

The converters are `vision_calibration.dataset_runner.build_*_input`, with the Rust names, for a run you drive step by step; each returns the run result (`input` as a payload, `view_paths`, `usable_views`, `total_views`). A manifest can also be a `DatasetSpec` payload, with `base_dir=` naming the directory its paths resolve against. `vc.sniff_folder(path)` returns the drafted manifest as a payload.

The runner's errors keep their types. `RunError` carries the Rust variant as `kind` (`"validation"`, `"empty_image_match"`, `"pose_count_mismatch"`, ...); the ask-user ambiguity is its subclass `AskUserError`, with the `field`, `prompt` and `suggestions`:

```python
from vision_calibration.dataset_runner import build_single_cam_handeye_input

try:
    run = build_single_cam_handeye_input(spec, base_dir="data/kuka_1")
except vc.AskUserError as err:
    print(f"set {err.field}: {err.prompt} (e.g. {err.suggestions})")
session = vc.SingleCamHandeyeSession()
session.set_input(run.input)
```

## Device Specifications

A second, optional file describes the *hardware* rather than the data: `spec.json` next to the manifest (`DEVICE_SPEC_FILENAME`), parsed as `DeviceSpec`. It holds datasheet values: per camera the focal length in millimetres, the pixel pitch in micrometres, the image resolution, an optional principal point and an optional Scheimpflug mount tilt in degrees; for rigs the nominal camera mounts, the hand-eye mounting and laser-plane nominals.

The `vision_calibration::device_seed` functions derive [manual-init seeds](manual_init.md) from it (`scheimpflug_seed`, `rig_intrinsics_seed`, `nominal_cam_se3_rig`, `handeye_seed`, `rig_layout_seed`). The intrinsics derivation is `f_px = focal_mm * 1000 / pixel_pitch_um` for both focal lengths with the principal point defaulting to the image centre; all unit conversion (millimetres, micrometres and degrees to pixels, metres and radians) happens there and nowhere else. The functions work per camera id, never by position, and a mismatch is a typed `DeviceSeedError`. The seeds feed the rig family; the single-camera problems take a seed directly (see [Seeding with Prior Knowledge](manual_init.md)).

## JSON Schema

The manifest types derive a JSON Schema, and `cargo xtask emit-schemas` writes it to `app/src/schemas/dataset_spec.json`. The app generates its manifest form from that file, so every field the Rust type accepts is editable without hand-writing TOML; external tools can use it for validation and completion in an editor. [Manifest Field Reference](manifest_reference.md) renders the schema as tables.
