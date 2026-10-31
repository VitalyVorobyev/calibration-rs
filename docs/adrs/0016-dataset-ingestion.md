# ADR 0016: Dataset Ingestion

## Context

Datasets arrive in arbitrary layouts: per-camera folders or tiled
multi-camera strips, vendor-specific robot-pose files, target and laser
frames side by side. Users should not copy or rename gigabytes of images to
fit a convention. Robot poses are the most dangerous input: KUKA, ABB, UR
and Fanuc disagree on at least one of transform direction, rotation format
and units, and a wrong guess yields a plausible-looking, wrong calibration.

## Decision

### One manifest is the only on-disk input format

`DatasetSpec` (`dataset.toml`, `vision-calibration-dataset`) describes a
dataset in place: per-camera image sources (glob or list, optional ROI,
optional laser frames), the target (`TargetSpec`, tagged on `kind`), robot
poses with a column mapping onto canonical fields, the topology (which of
the seven problem types), how images pair across cameras and with poses
(`by_index` or a shared filename token), the pose convention, an optional
laser-extraction table and, for laser rigs, the frozen upstream
calibration.

The problem types' `*Input` structs are internal IR, built from a manifest
by per-problem converters in `dataset_runner`; they may change freely.

### Ambiguity fails fast

Nothing that changes a result's meaning is defaulted (AGENTS.md §4):

- The pose convention is three closed enums — transform direction,
  rotation format, translation units — with no defaults, required whenever
  robot poses are given. A new vendor convention is a code change, not a
  manifest option.
- `_unresolved` lists fields a manifest generator could not determine;
  a manifest with entries there is rejected, and the app blocks Run until
  they are filled.
- Ambiguity that survives validation (an inconsistent pairing) is the typed
  `AskUser { field, prompt, suggestions }` error: the app asks, and the
  `generate-manifest` CLI prints the question and exits with status 2.
- Empty globs, too few features per view and laser fields on a non-laser
  topology are typed errors, never silently skipped.
- Algorithm tuning with safe defaults (detector and laser-extraction
  parameters) is not ambiguity and may be omitted.

The manifest generator is heuristic (file names, extensions, vendor
signatures); it fills what it can determine and lists the rest.

### Detection is cached by content

Each detection is cached in one JSON file per entry under a per-dataset
root, written by atomic rename. Its key has three parts:
- the image-content hash;
- the detector name;
- a hash of the detector config serialized with sorted keys, together with
  the detect crate's release, the versions of the detection crates it wraps,
  and the detector's output revision.

The hash is of the bytes, so moving or touching an image keeps its entry,
and upgrading the detector stack recomputes. The cache is disposable: no
versions or migrations, so a format change re-detects. A `force_redetect`
switch bypasses reads, for local detector edits that change no version. The
cache is a trait (`DetectionCache`) with a filesystem default.

### Laser extraction is injected

The laser-pixel extractor lives in a crate that is not on crates.io, so no
published crate can depend on it. The pipeline defines an open
`LaserPixelExtractor` trait and owns everything around the call (cache,
decode, ROI crop, lifting to source coordinates); the app supplies the
implementation, tests a deterministic fake. Laser pixels are cached under
the detector name `laser:<extractor>`, so changing extractors re-detects.
A laser rig's per-view target poses are computed from the laser dataset's
own robot poses through the frozen hand-eye chain, so laser views need not
match the upstream calibration's views.

## Consequences

- One schema means one form, one validator and one reproducible artefact
  per dataset.
- Manifests are more verbose than a folder convention; the price buys out
  the most common silent failure in hand-eye calibration.
- Re-running with unchanged detection is near-free, which makes A/B
  comparisons cheap.
- Library users who drive the laser converters must supply an extractor;
  there is no built-in fallback.

## Alternatives rejected

- **A folder convention** — forces users to copy and reshape their data.
- **`*Input` JSON as the user format** — not designed for hand authoring,
  embeds nalgebra serde shapes and would freeze internal types.
- **Defaulting the pose convention** — wrong on roughly half of real
  datasets, silently.
- **A git dependency on the extractor** — breaks `cargo publish`.
