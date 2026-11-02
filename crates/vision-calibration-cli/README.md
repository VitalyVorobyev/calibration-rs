# vision-calibration-cli

`vcal`: calibrate cameras from a folder of images on the command line.

```bash
cargo install vision-calibration-cli
```

```text
vcal init <folder>              draft <folder>/dataset.toml from the file names
vcal check <manifest>           check a manifest and its images, without detecting
vcal run <manifest> [-o <dir>]  detect, calibrate, write calibration.json and quality.json
vcal sample list                list the sample datasets
vcal sample fetch <id>          download a sample dataset; prints its manifest's path
```

Try it on a sample dataset:

```bash
vcal run "$(vcal sample fetch opencv_left)" -o out
```

`run` prints each camera's parameters with their σ and the quality verdict,
and writes the calibration export (`calibration.json`) and the quality report
(`quality.json`). `-c <file>` changes the workflow's default configuration
(TOML or JSON). The exit status is 0 when done, 1 on an error, 2 on a wrong
command line and 3 when `run` finished but the verdict is fail.

`pip install vision-calibration` installs the same program. The commands are
a thin layer over [`vision-calibration`](https://crates.io/crates/vision-calibration),
the library to use from code. The book's
[command-line chapter](https://vitalyvorobyev.github.io/calibration-rs/cli.html)
walks through a calibration.
