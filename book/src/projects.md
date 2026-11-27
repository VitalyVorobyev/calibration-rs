# Projects

A manifest describes a dataset; a **project** is that dataset with its
history: the changes to the workflow's defaults it is calibrated with, the
images and features it leaves out, and every run calibrated from it. A
project is a folder, and the library alone reads and writes it, so `vcal`,
Python, Rust and the desktop app all see the same runs.

```text
my-camera/
  calibration-project.toml     the project file
  .detections/                 the detection cache (disposable)
  runs/
    run-001/
      run.json                 the record: inputs, their digest, a summary
      export.json              the workflow's export
      quality.json             the quality report
      notes.json               label, notes, star (only when set)
      report.html              the run as one page (when asked for)
```

## The Project File

```toml
version = 2
name = "Bench camera"
description = "Left camera of the bench rig"

[device]
product = "bench rig"        # the device type
id = "SN-0042"               # the device's own identifier
label = "left"

[seed]                       # where every run starts from (optional)
run = "../SN-0041/runs/run-002"

[dataset]
manifest = "captures/dataset.toml"

[config]                     # changes to the workflow's defaults
solver.max_iters = 80
camera.model = { projection = "equidistant", distortion = "kannala_brandt4", sensor = "identity" }

[quality]                    # the thresholds of the quality report
expected_feature_noise_px = 0.1

[exclude]
views  = [{ view = { token = "0012" }, note = "robot pose glitch" }]
images = [{ camera = "cam1", image = "cam1/0003.png" }]

[[exclude.features]]
camera = "cam0"
image  = "cam0/0005.png"
points = [[0.03, 0.06, 0.0]]
note   = "glare"
```

Only `version` and `[dataset]` are required. An unknown field is an error,
so a misspelt section such as `[exlude]` never passes silently.

- `[device]` names the device the project calibrates. Projects of one
  device, wherever they are, form its calibration history
  ([The Library](#the-library)); `id` is compared as written.
- `[seed]` names the seed every run starts from instead of the closed-form
  initialization ([Starting from a Seed](#starting-from-a-seed)).
- `[config]` holds the workflow's configuration fields to change, keyed
  like its `*Config` ([Changes to the Defaults](getting_started.md#changes-to-the-defaults));
  what it leaves out keeps the default.
- `[quality]` sets the thresholds the runs' quality reports use
  ([Judging a Result](quality.md)).

## The Dataset

`[dataset]` holds the manifest in one of two ways.

A **manifest file** of its own, referenced by path (relative to the project
folder unless absolute). Its relative paths resolve against its own folder,
as a manifest's always do, and the library never writes it: edit it by hand
and the next run reads the edit.

```toml
[dataset]
manifest = "../captures/2026-11-20/dataset.toml"
```

Or the manifest **written in the project file**, with `root`, the folder its
relative paths resolve against (relative to the project folder unless
absolute, `.` when left out). The desktop app edits such a manifest.

```toml
[dataset]
root = "images"

[dataset.manifest]
version = 1
topology = "planar_intrinsics"

[[dataset.manifest.cameras]]
id = "cam0"
images = { kind = "glob", pattern = "*.png" }

[dataset.manifest.target]
kind = "chessboard"
rows = 6
cols = 9
square_size_m = 0.025
```

Every path in the manifest (images, robot poses, an upstream calibration)
resolves against this data root, so a laser rig project can point
`upstream_calibration` at another project's `runs/run-003/export.json`.

## The Views

`Project::views` lists the views the images pair into, without reading an
image: each view's pairing token and each camera's image in it.
`Project::view_poses` pairs them with the manifest's robot poses (a pose is
named by its position in the pose file and its `pose_id`) and
`Project::view_laser_frames` with each camera's laser frames, by the
manifest's `pose_pairing`. What pairs with nothing is listed too: a view
without a pose or a laser frame, a pose row or a laser frame that pairs
with no view. A run refuses such a dataset; the listing shows which rows
and frames are at fault. Python's `Project.views()` returns all three, with
an `{"error", "field"}` entry in place of a pose or laser listing that
cannot be made.

```rust,no_run
use vision_calibration::project::Project;

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let project = Project::open("my-camera")?;
let views = project.views()?;
if let Some(poses) = project.view_poses(&views)? {
    for (token, pose) in views.tokens.iter().zip(&poses.views) {
        match pose {
            Some(row) => println!("view {token}: pose row {}", row.index),
            None => println!("view {token}: no robot pose"),
        }
    }
    for row in &poses.unpaired {
        println!("pose row {} ({:?}) pairs with no view", row.index, row.id);
    }
}
# Ok(())
# }
```

## Leaving Data Out

`[exclude]` leaves out what it names, by what does not change when the
dataset grows; never by an index.

| List | Names | Leaves out |
|---|---|---|
| `views` | a view, by its pairing token (`{ token = "0012" }`, under `shared_filename_token` pairing) or any one of its images (`{ image = "cam0/0012.png" }`) | every camera's image of the view, its robot pose and its laser frames |
| `images` | one camera's image of a view: `camera` and `image` | that image; the view keeps its other cameras |
| `features` | target `points` (metres, as the detector labels them) of one camera's detection in one `image` | those features; the view is still screened for its feature floor |

Image paths are relative to the data root unless absolute. An excluded
image is not read; the run lists it with the reason `excluded`. An
exclusion that names an image, view or camera the dataset does not have is
an error before any detection, so a stale entry never silently leaves
nothing out. A feature point that matches no detected feature is not an
error, since the detection is only known once it ran: the run's record lists
it under `unmatched_exclusions`.

## Starting from a Seed

A project of a device whose type, or whose own earlier calibration, is
known can start every run from it, the way
[Starting from a previous calibration](manual_init.md#starting-from-a-previous-calibration)
describes for one dataset run. `[seed]` holds one key:

| Key | The seed |
|---|---|
| `run` | a calibrated run: its folder (`runs/run-003`, of this project or another) or its `export.json`. Its record names the cameras and their image sizes. |
| `device_spec` | a device specification (`spec.json`): the datasheet's focal lengths and mount tilts, the drawing's mounts ([Device Specifications](manifest.md#device-specifications)) |
| `file` | a seed file (JSON, TOML by its `.toml` extension), such as `vcal seed` writes: one unit's, or a device type's ([A seed for a device type](manual_init.md#a-seed-for-a-device-type)) |

A path is relative to the project folder unless absolute. The seed is made
and checked against the dataset's cameras and the configuration's camera
model before any image is read (`vcal check` and `Project::check` check it
too), and a camera's image size against the seed's once the image headers
are read, before any detection. A seed that is missing, cannot be read or
does not fit is an error that names it (`field` `seed`), never a run
without it. `ProjectRunOptions::seed` (Python `calibrate(seed=...)`) starts
one run from another seed in place of the project's.

The seed is a starting point: the calibration still estimates everything,
and a seed close to the device ends where the unseeded run ends. The run's
record keeps it under `inputs.seed`: the source as the project named it,
the values the run started from (so a seed file edited later does not
change what an old run says it used), and what the workflow `applied` and
`ignored`, by label (`cam0.lens`, `cam0.cam_se3_rig`, `cam0.laser_plane`,
`handeye`): a planar run takes a seed's lens and leaves its laser plane.

`DeviceSeed::from_run` (Python `DeviceSeed.from_run`) makes the seed of a
run, from its export and its record; `vcal seed from-runs` writes it to a
file.

## Runs

`vcal run my-camera` (or `Project::calibrate`) checks the project, detects
through the project's cache, calibrates and writes the next
`runs/run-NNN/`. A run is written whole or not at all; an interrupted run
leaves nothing behind. Run numbers are claimed so that the desktop app,
`vcal` and Python can calibrate into one project at once without overwriting
each other.

`run.json` is the run's record and never changes: the tool and library
version, the inputs as the run resolved them (the manifest, the exclusions,
the configuration changes and the configuration they made, the quality
thresholds, the seed), what detection left out and why, each image the
calibration used with a hash of its content, and a summary (the verdict,
each camera's RMS reprojection error, features, views and image size). A
run's label,
notes and star live apart, in `notes.json`, the only file that changes.

`export.json` is the workflow's export. Its image manifest's `root` leads
from the run folder to the data, relative when the data lies inside the
project, so a project moved with its data still finds its images, and a run
folder opens on its own in a viewer. An export calibrated elsewhere, such
as a `vcal run -o` folder, joins the project with `vcal project import` or
`Project::import_export`.

## Digests and Reproducibility

The record's `digest` holds a SHA-256 of each input: the manifest (with
the content of its robot pose file and upstream calibration), the
exclusions, the resolved configuration, the quality thresholds, the
detections, the set of detection-cache keys the run read, which cover each
image's bytes, the detector, its configuration and its version, and for a
seeded run the seed's values (`seed`; where they were read from is not part
of it). `inputs` combines them; a run without a seed has no `seed` digest,
and its `inputs` combines the other five. Two runs with the same `inputs` digest, by the same library
version, have byte-identical `export.json` and `quality.json`: the time
stamp and duration in `run.json` are the only clock readings, and no digest
includes them. Moving the data does not change a digest.

`Project::input_digest` (Python `Project.input_digest()`) computes the
same digests of the project as it stands, for everything but the
detections, without reading an image. A run whose `dataset`, `exclusions`,
`config`, `quality` or `seed` digest differs from the project's was made
from other inputs than the project now states; a run's own configuration
changes or seed count as such a difference.

## Comparing Runs

`vcal project compare my-camera run-001 run-004` (`Project::compare_runs`)
puts runs side by side against the first, as
[Comparing Calibrations](comparing_calibrations.md) describes. The relation
of the runs' data, which decides whether a difference has a test, is read
from the records: two runs that used no image of the same content are
independent captures; runs of the same features under one configuration and
dataset (equal `config` and `dataset` digests) are identical, under another
configuration they share data; a run whose features are a strict subset of
the other's under one configuration is nested (an exclusion); anything else
overlaps. A feature is one target point of one camera in one image,
identified by the image's content and the detected pixel, so a moved or
renamed image is still the same data and a re-detected one with another
detector is not. An imported run records no image content, so its relation
is unknown and its differences have no test. Each difference also lists the
images and features the two runs share and the exclusions only one of them
made.

Runs of other projects (another capture of the same device) are compared
by their folder: `vcal project compare my-camera run-004
../my-camera-march/runs/run-002`, or `compare_recorded_runs` on their
`RunInfo`s.

## The Library

The projects under a set of folders are a library. `vcal project scan`
(`scan_library`) walks the folders, reads each project and groups the runs
of every device by `[device]`'s `product` and `id`, oldest first (runs of
the app's earlier format, which have no time, first). There is no index to
fall out of date: the folders are the library.

`DeviceHistory::representative_run` (Python
`project.representative_run(device)`) picks the run that stands for a
device: its latest starred run, else its latest run calibrated in a project
(not imported) whose verdict is `pass`. Its seed starts the device's next
calibration; the seeds of several devices of one type make the type's
([A seed for a device type](manual_init.md#a-seed-for-a-device-type)):

```rust,no_run
use vision_calibration::device_seed::DeviceSeed;
use vision_calibration::project::{LibraryScanOptions, scan_library};

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let scan = scan_library(&["calibrations".into()], &LibraryScanOptions::default());
let mut seeds = Vec::new();
for device in scan.devices() {
    if let Some(entry) = device.representative_run() {
        seeds.push(DeviceSeed::from_run(&entry.run.dir)?);
    }
}
let aggregate = DeviceSeed::aggregate(&seeds)?;
std::fs::write("type-seed.json", serde_json::to_string_pretty(&aggregate.seed)?)?;
# Ok(())
# }
```

## From the Command Line

```bash
vcal project init my-camera --manifest captures/dataset.toml --device-id SN-0042 \
  --seed-run ../SN-0041/runs/run-002
vcal check my-camera
vcal run my-camera --label "after refocus" --report
vcal project runs my-camera
vcal project note my-camera run-002 --star
vcal project compare my-camera run-001 run-002
vcal project scan ~/calibrations
```

[The Command Line](cli.md#projects) lists every option.

## From Rust

A project is created, opened, edited and saved in place; `save` refuses
when the file changed on disk since it was read, so a concurrent edit is
never overwritten:

```rust
use vision_calibration::project::{DatasetSource, DeviceIdentity, Project, ProjectSpec};

# fn main() -> Result<(), Box<dyn std::error::Error>> {
# let dir = std::env::temp_dir().join(format!("vc-book-project-{}", std::process::id()));
# let _ = std::fs::remove_dir_all(&dir);
let mut spec = ProjectSpec::for_manifest("captures/dataset.toml");
spec.name = Some("Bench camera".into());
spec.device = Some(DeviceIdentity::new("SN-0042"));
let created = Project::create(&dir, spec)?;

let mut project = Project::open(&dir)?;
assert_eq!(project.revision(), created.revision());
project.spec_mut().description = Some("left camera of the bench rig".into());
project.save()?;
assert!(project.runs()?.is_empty());
assert!(matches!(project.spec().dataset, DatasetSource::File { .. }));
# std::fs::remove_dir_all(&dir)?;
# Ok(())
# }
```

A run needs the images:

```rust,no_run
use vision_calibration::control::RunControl;
use vision_calibration::project::{Project, ProjectRunOptions};

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let project = Project::open("my-camera")?;
project.check()?; // what a run checks before it reads an image
let mut options = ProjectRunOptions::new("my-tool 1.0");
options.notes.label = Some("after refocus".into());
let run = project.calibrate(&options, None, &RunControl::default())?;
println!("{} -> {}", run.id, run.export_path().display());
for info in project.runs()? {
    if let Some(summary) = info.summary() {
        println!("{}: {:?}", info.id, summary.verdict);
    }
}
# Ok(())
# }
```

## From Python

```python
from vision_calibration.project import Project, scan_library

project = Project.open("my-camera")
run = project.calibrate(label="after refocus", report_html=True)
print(run["id"], run["state"]["summary"]["verdict"])
for info in project.runs():
    print(info["id"], info["notes"])
comparison = project.compare_runs(["run-001", run["id"]])
print(comparison["differences"][0]["relation"])

for device in scan_library(["~/calibrations"])["devices"]:
    print(device["device"]["id"], len(device["runs"]))

# The next unit's project starts from this one's run.
spec = project.spec
spec["seed"] = {"run": run["dir"]}
next_unit = Project.create("my-camera-2", spec)
```
