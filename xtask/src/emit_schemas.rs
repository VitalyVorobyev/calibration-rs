//! Emit JSON Schemas for the workspace's user-facing config types.
//!
//! Output goes to `app/src/schemas/<name>.json`. The Tauri app reads these
//! files at build time to drive schema-driven forms in the Run workspace.
//!
//! With `--check`, the command instead verifies that the on-disk schemas
//! match what would be generated from current source. CI runs this to
//! catch drift between source and committed schemas.

use anyhow::{Context, Result, bail};
use schemars::{JsonSchema, schema_for};
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

    let entries: Vec<(&str, Value)> = vec![
        ("dataset_spec", schema_value::<DatasetSpec>()),
        (
            "planar_intrinsics_config",
            schema_value::<PlanarIntrinsicsConfig>(),
        ),
        (
            "scheimpflug_intrinsics_config",
            schema_value::<ScheimpflugIntrinsicsConfig>(),
        ),
        (
            "single_cam_handeye_config",
            schema_value::<SingleCamHandeyeConfig>(),
        ),
        (
            "laserline_device_config",
            schema_value::<LaserlineDeviceConfig>(),
        ),
        (
            "rig_extrinsics_config",
            schema_value::<RigExtrinsicsConfig>(),
        ),
        ("rig_handeye_config", schema_value::<RigHandeyeConfig>()),
        (
            "rig_handeye_laserline_config",
            schema_value::<RigHandeyeLaserlineConfig>(),
        ),
        (
            "rig_laserline_device_config",
            schema_value::<RigLaserlineDeviceConfig>(),
        ),
    ];

    let mut drift = Vec::new();
    for (name, schema) in &entries {
        let path = out_dir.join(format!("{name}.json"));
        write_or_check(&path, schema, check, &mut drift)?;
    }

    if check {
        if drift.is_empty() {
            println!("schemas up to date ({} files)", entries.len());
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
        println!("emitted {} schemas to {}", entries.len(), out_dir.display());
        Ok(())
    }
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
