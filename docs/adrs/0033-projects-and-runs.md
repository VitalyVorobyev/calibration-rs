# ADR 0033: Projects and Runs Belong to the Library

## Context

Four things overlapped: the dataset manifest (`DatasetSpec`), the desktop
app's project (`calibration-project.json` version 1 with `dataset.json` and
`runs/run-NNN/{manifest,config,export,diagnostics}.json`), the output
folders of `vcal run -o` (`calibration.json`, `quality.json`) and the app's
presets (configuration overrides written in TypeScript). Each had its own
writer, so a run the app saved was invisible to `vcal`, a preset's settings
were invisible to Python, and nothing tied a device's calibrations
together.

## Decision

A **project** is one dataset with its run history, defined and written by
the library only:

```text
<project>/
  calibration-project.toml     the definition (version 2)
  .detections/                 detection cache (content-addressed, disposable)
  runs/<run-id>/
    run.json                   immutable record: inputs, digests, summary
    export.json                the export (ADR 0012), unchanged
    quality.json               QualityReport
    notes.json                 label / notes / starred: the only mutable file
    report.html                optional page
```

- **D1 — one module.** Everything is in
  `vision_calibration_pipeline::project` (facade `vision_calibration::project`).
  The project file names pipeline types (the configuration patch validated
  against `CalibrationConfig`, `QualityThresholds`) and `run.json` embeds
  `CalibrationConfig`, `Verdict` and `RejectReason`; putting the file type in
  the dataset crate would force untyped fields there. Layering is unchanged.
- **D2 — the data root is part of the dataset.** `[dataset]` holds the
  manifest either inline (`[dataset.manifest]`, with `root`, the folder its
  paths resolve against) or as a path to a manifest file of its own, which
  the library never writes. Inline lets the app edit the manifest; a
  referenced file keeps one source of truth for a hand-written
  `dataset.toml` (an embedded copy would go stale silently).
- **D3 — formats.** A new project is TOML; a JSON project (every version 1
  project) keeps its file name and is written back as version 2 JSON, the
  version 1 `dataset.json` removed after the write. `open` never writes.
- **D4 — exclusions by identity.** A view is named by its pairing token or
  one of its images, an image by camera and path, a feature by the target
  point the detector labelled. Never an index: indices change meaning when
  an image is added. A stale reference is a typed error before detection.
- **D5 — run ids.** `run-NNN`, one more than the largest number, claimed by
  renaming a staging folder; a rename onto a run another writer claimed
  fails and the next number is tried. No clock and no UUID in a name.
- **D6 — the record is immutable.** `run.json` is written once; the label,
  notes and star live in `notes.json` (VIEW-FEEDBACK extends `RunNotes`).
- **D7 — the cache is the project's.** `<project>/.detections`, the `vcal`
  convention, with a `.gitignore` of `*`. A project folder that is also its
  manifest's folder shares one cache between `vcal run dataset.toml` and
  `vcal run .`.
- **D8 — `[seed]` names the seed every run starts from** (`SeedSource`:
  `run`, `device_spec` or `file`, one key, paths relative to the project).
  It is resolved and checked against the dataset and the configuration in
  `prepare`, before any image; a seed that is missing, unreadable or does
  not fit is a typed error (`ProjectError::SeedUnresolved`, or
  `Calibrate(Seed(_))`), never a run without it. `ProjectRunOptions::seed`
  replaces it for one run. `run.json`'s `inputs.seed` (`RunSeed`) keeps the
  source as written, the resolved values (so a seed file that changes later
  does not change what a run says it started from) and what the workflow
  `applied`/`ignored`. `InputDigest.seed` hashes the values only, and
  `inputs` covers it **only when present**: an unseeded run's digests are
  the ones it had before seeds were recorded. `CameraSummary.image_size`
  lets `DeviceSeed::from_run` make a run's seed without the images (an older
  record reads the image headers). `DeviceHistory::representative_run`
  (latest starred, else latest calibrated `pass` run) is the one stated
  policy for which run stands for a device. An older library refuses a
  project with `[seed]` (`deny_unknown_fields`) rather than run it unseeded.
- **D10 — the record names the content of what a run used** (ADR 0035).
  `CalibratedRun::images` lists one `UsedImage` per frame of the export's
  image manifest, in its order: camera, path and the FNV-1a image hash the
  detection cache key already holds, recorded by the project's cache as the
  detection reads each image (no second read). It is an optional field:
  `RUN_RECORD_VERSION` stays 1, no digest covers it, and a run written
  before it compares with an unknown data relation.
- **D9 — the configuration patch is a library type** (`ConfigPatch` in
  `dataset_runner`), not a `serde_json::Value` on a public field.

**Determinism.** `export.json`, `quality.json`, `report.html` and every
digest are functions of the inputs and the library version. The clock
appears only in `run.json` (`timestamp`, `duration_ms`) and is never hashed.
`InputDigest` covers the manifest with the content of its non-image files
(robot poses, upstream calibration), the exclusions (normalized), the
resolved configuration, the quality thresholds and the sorted set of
detection-cache keys the run read (image bytes, detector, detector
configuration and versions). Equal `inputs` digests and the same library
version give byte-identical `export.json` and `quality.json`. The data
root's location is not hashed.

**Images of a moved project.** `export.json`'s image root is restated for the
run folder: relative when the data lies inside the project, absolute
otherwise, so a run folder opens alone. Project-aware readers resolve frames
against the project's current data root instead, so relinking moved data in
the project file fixes every run.

**The library.** `scan_library` walks root folders (no symbolic links, no
dot-folders, no descent into a project, depth-limited) and reads each
project afresh; `LibraryScan::devices` groups runs by `(product, id)` of
`[device]`. Nothing is indexed.

**`vcal`.** `vcal run <PATH>` runs a project when the path is a project
folder or project file (the marker decides, nothing is guessed) and a
manifest otherwise, exactly as before. `vcal project init|runs|import|note|scan`
cover the rest.

## Consequences

- The app's run path moves onto `Project::calibrate` and its presets become
  sample projects (separate change); until then the app writes version 1
  and cannot open a project a library `save` has written as version 2.
- `save` rewrites the whole file: comments in a hand-written TOML are lost.
  A `toml_edit` save that rewrites only changed tables can come later,
  additively.
- The project file version 2, `run.json` version 1, the `vcal` surface and
  the Rust types are frozen once released.
- Digest completeness relies on the runner reading every image through the
  cache; a test pins the key count to the images read.

## Alternatives rejected

- A database or index file for the library: state that drifts from the
  folders.
- `DatasetSpec` growing project fields: it describes data, and denies
  unknown fields.
- Index-based exclusions: they change meaning when an image is added.
- Timestamp or UUID run ids: unreadable, and a clock in a name.
- The OS cache dir for project detections: lost when the project moves,
  and keyed by a description slug.
- Presets as code: they drift from the library and are invisible to `vcal`.
- Embed-only manifests: a hand-edited `dataset.toml` and its copy diverge.
