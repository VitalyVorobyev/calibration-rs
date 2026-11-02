# vision-calibration-dataset

Canonical input-data manifest (`DatasetSpec`) for `calibration-rs`. The
manifest is the single on-disk wire format users author to describe a foreign
dataset's layout without copying or renaming images.

Most users want [`vision-calibration`](https://crates.io/crates/vision-calibration) instead, which re-exports this crate as
`vision_calibration::dataset` next to the runner that calibrates a manifest.

You can write a manifest by hand, or generate a skeleton with `sniff_folder`
(also available as `vcal init <dataset-folder>`, from the
[`vision-calibration-cli`](https://crates.io/crates/vision-calibration-cli)
crate).

The sniffer infers only what is structurally unambiguous (camera folders,
image globs, robot-pose file format, pairing). Everything that needs domain
knowledge (board geometry, target kind, frame convention) is left as a
placeholder and listed under `_unresolved`; the runner refuses the manifest
until that list is cleared.

Per-problem-type converters (`DatasetSpec` to the pipeline `*Input` types) live
in `vision-calibration-pipeline`.

## Sample datasets

`samples::catalogue()` lists real calibration datasets, each with a ready
manifest, its licence and the citation its publisher asks for. With the
`fetch` feature, `samples::fetch(id, cache)` downloads one at a pinned
revision and verifies every file against its SHA-256.
