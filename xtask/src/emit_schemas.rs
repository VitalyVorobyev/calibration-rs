//! Emit JSON Schemas for the workspace's user-facing config types.
//!
//! Output goes to `app/src/schemas/<name>.json`, plus `<name>.default.json` (the
//! type's `Default` value) for every config that has one. The Tauri app reads the
//! schemas at build time to drive schema-driven forms in the Run workspace; the
//! defaults feed the app's form round-trip tests.
//!
//! With `--check`, the command instead verifies that the on-disk schemas
//! match what would be generated from current source. CI runs this to
//! catch drift between source and committed schemas.

use anyhow::{Context, Result, bail};
use schemars::{JsonSchema, schema_for};
use serde::Serialize;
use serde_json::Value;
use std::path::{Path, PathBuf};

use vision_calibration_dataset::DatasetSpec;
use vision_calibration_pipeline::laserline_device::LaserlineDeviceConfig;
use vision_calibration_pipeline::planar_intrinsics::PlanarIntrinsicsConfig;
use vision_calibration_pipeline::rig_extrinsics::RigExtrinsicsConfig;
use vision_calibration_pipeline::rig_handeye::RigHandeyeConfig;
use vision_calibration_pipeline::rig_handeye_laserline::RigHandeyeLaserlineConfig;
use vision_calibration_pipeline::rig_laserline_device::RigLaserlineDeviceConfig;
use vision_calibration_pipeline::scheimpflug_intrinsics::ScheimpflugIntrinsicsConfig;
use vision_calibration_pipeline::single_cam_handeye::SingleCamHandeyeConfig;

pub fn run(workspace_root: &Path, check: bool) -> Result<()> {
    let out_dir = workspace_root.join("app/src/schemas");
    std::fs::create_dir_all(&out_dir).with_context(|| format!("creating {}", out_dir.display()))?;

    // (name, schema, `Config::default()` serialised — `None` for a type with no `Default`).
    let entries: Vec<(&str, Value, Option<Value>)> = vec![
        ("dataset_spec", schema_value::<DatasetSpec>(), None),
        entry::<PlanarIntrinsicsConfig>("planar_intrinsics_config"),
        entry::<ScheimpflugIntrinsicsConfig>("scheimpflug_intrinsics_config"),
        entry::<SingleCamHandeyeConfig>("single_cam_handeye_config"),
        entry::<LaserlineDeviceConfig>("laserline_device_config"),
        entry::<RigExtrinsicsConfig>("rig_extrinsics_config"),
        entry::<RigHandeyeConfig>("rig_handeye_config"),
        entry::<RigHandeyeLaserlineConfig>("rig_handeye_laserline_config"),
        entry::<RigLaserlineDeviceConfig>("rig_laserline_device_config"),
    ];

    let mut drift = Vec::new();
    let mut files = 0;
    for (name, schema, default) in &entries {
        write_or_check(
            &out_dir.join(format!("{name}.json")),
            schema,
            check,
            &mut drift,
        )?;
        files += 1;
        if let Some(default) = default {
            write_or_check(
                &out_dir.join(format!("{name}.default.json")),
                default,
                check,
                &mut drift,
            )?;
            files += 1;
        }
    }

    if check {
        if drift.is_empty() {
            println!("schemas up to date ({files} files)");
            Ok(())
        } else {
            for entry in &drift {
                entry.report();
            }
            let hint = if drift.iter().any(|d| matches!(d, Drift::LineEndings(_))) {
                "; the CRLF ones need an LF checkout (`git add --renormalize .`), not a re-emit"
            } else {
                ""
            };
            bail!(
                "{} schema(s) out of date; run `cargo xtask emit-schemas` and commit the result{hint}",
                drift.len()
            )
        }
    } else {
        println!("emitted {files} files to {}", out_dir.display());
        Ok(())
    }
}

/// A config type's schema and its `Default` value — the app's form tests round-trip the
/// latter through the former.
fn entry<T: JsonSchema + Default + Serialize>(name: &str) -> (&str, Value, Option<Value>) {
    let default = serde_json::to_value(T::default()).expect("a config serialises to JSON");
    (name, schema_value::<T>(), Some(default))
}

fn schema_value<T: JsonSchema>() -> Value {
    serde_json::to_value(schema_for!(T)).expect("JsonSchema serialization is infallible")
}

/// Why a committed schema no longer matches what the generator produces.
enum Drift {
    /// The file is missing, or its content genuinely differs — a config type
    /// changed and the schema was not re-emitted.
    Stale(PathBuf),
    /// The content matches once CRLF is normalized: the working copy was
    /// checked out with the wrong line endings, not left stale. Re-emitting
    /// fixes nothing — `.gitattributes`' `eol=lf` rule is what prevents this.
    LineEndings(PathBuf),
}

impl Drift {
    fn report(&self) {
        match self {
            Self::Stale(path) => eprintln!("schema drift: {}", path.display()),
            Self::LineEndings(path) => {
                eprintln!("line-ending drift (CRLF on disk): {}", path.display());
            }
        }
    }
}

fn write_or_check(path: &Path, schema: &Value, check: bool, drift: &mut Vec<Drift>) -> Result<()> {
    let mut text =
        serde_json::to_string_pretty(schema).context("rendering schema as pretty JSON")?;
    text.push('\n');

    if check {
        let on_disk = match std::fs::read_to_string(path) {
            Ok(s) => s,
            Err(_) => {
                drift.push(Drift::Stale(path.to_path_buf()));
                return Ok(());
            }
        };
        if on_disk != text {
            drift.push(if on_disk.replace("\r\n", "\n") == text {
                Drift::LineEndings(path.to_path_buf())
            } else {
                Drift::Stale(path.to_path_buf())
            });
        }
        return Ok(());
    }

    std::fs::write(path, &text).with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}
