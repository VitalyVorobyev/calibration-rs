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
            for path in &drift {
                eprintln!("schema drift: {}", path.display());
            }
            bail!(
                "{} schema(s) out of date; run `cargo xtask emit-schemas` and commit the result",
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

fn write_or_check(
    path: &Path,
    schema: &Value,
    check: bool,
    drift: &mut Vec<PathBuf>,
) -> Result<()> {
    let mut text =
        serde_json::to_string_pretty(schema).context("rendering schema as pretty JSON")?;
    text.push('\n');

    if check {
        let on_disk = match std::fs::read_to_string(path) {
            Ok(s) => s,
            Err(_) => {
                drift.push(path.to_path_buf());
                return Ok(());
            }
        };
        if on_disk != text {
            drift.push(path.to_path_buf());
        }
        return Ok(());
    }

    std::fs::write(path, &text).with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}
