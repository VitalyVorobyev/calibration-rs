# vision-calibration-dataset

Canonical input-data manifest (`DatasetSpec`) for `calibration-rs`. The
manifest is the single on-disk wire format users author to describe a foreign
dataset's layout without copying or renaming images.

You can write a manifest by hand, or generate a skeleton with `sniff_folder`
(also available as the `generate-manifest` binary, which needs the `cli`
feature):

```text
cargo run -p vision-calibration-dataset --features cli --bin generate-manifest -- <dataset-folder>
```

The sniffer infers only what is structurally unambiguous (camera folders,
image globs, robot-pose file format, pairing). Everything that needs domain
knowledge (board geometry, target kind, frame convention) is left as a
placeholder and listed under `_unresolved`; the runner refuses the manifest
until that list is cleared.

Per-problem-type converters (`DatasetSpec` to the pipeline `*Input` types) live
in `vision-calibration-pipeline`.
