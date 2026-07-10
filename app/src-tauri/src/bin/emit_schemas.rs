//! Emit a JSON Schema for the diagnose app's wire types (B-QUAL2).
//!
//! Single source of truth: the Rust `#[derive(schemars::JsonSchema)]` on the
//! pipeline `*Export` types (ADR 0018) and on this crate's Tauri command
//! payload/response types. `bun run generate:types` runs this emitter and
//! then converts the schema to TypeScript under `app/src/types/generated/`.
//!
//! All wire types are emitted into a single combined schema
//! (`app/schemas-generated/diagnose_wire.json`) so shared nested types
//! (`Camera`, `ImageManifest`, `Iso3Schema`, `PerFeatureResiduals`, …) are
//! deduped into one `$defs` block and become a single TypeScript interface
//! each, importable from one place.
//!
//! With `--check` the command verifies the on-disk schema matches what
//! current source would generate, so a Rust type change that was not
//! regenerated fails CI.
//!
//! Only built with `--features schema-export` (see the `[[bin]]`
//! `required-features` gate in `Cargo.toml`); `tauri dev` never compiles it.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use schemars::generate::SchemaSettings;
use serde_json::{Value, json};

use calibration_diagnose_lib::schema_types::{DisparityResult, EpipolarOverlay, RunProgress};

use vision_calibration::laserline_device::LaserlineDeviceExport;
use vision_calibration::planar_intrinsics::PlanarIntrinsicsExport;
use vision_calibration::rig_extrinsics::RigExtrinsicsExport;
use vision_calibration::rig_handeye::RigHandeyeExport;
use vision_calibration::rig_handeye_laserline::RigHandeyeLaserlineExport;
use vision_calibration::rig_laserline_device::RigLaserlineDeviceExport;
use vision_calibration::scheimpflug_intrinsics::ScheimpflugIntrinsicsExport;
use vision_calibration::single_cam_handeye::SingleCamHandeyeExport;

fn main() -> ExitCode {
    let check = std::env::args().any(|a| a == "--check");

    let schema = build_schema();

    let dir = out_dir();
    if let Err(e) = std::fs::create_dir_all(&dir) {
        eprintln!("error: creating {}: {e}", dir.display());
        return ExitCode::FAILURE;
    }
    let out_path = dir.join("diagnose_wire.json");

    match write_or_check(&out_path, &schema, check) {
        Ok(drifted) if check => {
            if drifted {
                eprintln!("schema drift: {}", out_path.display());
                eprintln!(
                    "run `bun run generate:types` and commit the result (schemas-generated/ + src/types/generated/)"
                );
                ExitCode::FAILURE
            } else {
                println!("schema up to date: {}", out_path.display());
                ExitCode::SUCCESS
            }
        }
        Ok(_) => {
            println!("emitted schema to {}", out_path.display());
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

/// Build one combined schema whose `$defs` holds every wire type (deduped),
/// referenced from a thin wrapper object. The wrapper exists only to force
/// every top-level type into `$defs`; it is ignored on the TypeScript side.
fn build_schema() -> Value {
    // Draft-07 (`definitions`, not 2020-12 `$defs`): json-schema-to-typescript
    // natively names + dedupes interfaces from draft-07 `definitions` keys.
    // With 2020-12 `$defs` it re-inlines shared types once per reference site
    // (`Camera1`, `PerFeatureResiduals7`, …).
    let mut generator = SchemaSettings::draft07().into_generator();
    let mut props = serde_json::Map::new();

    // Deterministic order: a fixed sequence of inserts, never an unordered map.
    // Calibration `*Export` types (facade → pipeline).
    props.insert(
        "planar_intrinsics_export".into(),
        generator
            .subschema_for::<PlanarIntrinsicsExport>()
            .to_value(),
    );
    props.insert(
        "scheimpflug_intrinsics_export".into(),
        generator
            .subschema_for::<ScheimpflugIntrinsicsExport>()
            .to_value(),
    );
    props.insert(
        "single_cam_handeye_export".into(),
        generator
            .subschema_for::<SingleCamHandeyeExport>()
            .to_value(),
    );
    props.insert(
        "laserline_device_export".into(),
        generator
            .subschema_for::<LaserlineDeviceExport>()
            .to_value(),
    );
    props.insert(
        "rig_extrinsics_export".into(),
        generator.subschema_for::<RigExtrinsicsExport>().to_value(),
    );
    props.insert(
        "rig_handeye_export".into(),
        generator.subschema_for::<RigHandeyeExport>().to_value(),
    );
    props.insert(
        "rig_laserline_device_export".into(),
        generator
            .subschema_for::<RigLaserlineDeviceExport>()
            .to_value(),
    );
    props.insert(
        "rig_handeye_laserline_export".into(),
        generator
            .subschema_for::<RigHandeyeLaserlineExport>()
            .to_value(),
    );
    // Tauri command payload/response types (this crate).
    props.insert(
        "epipolar_overlay".into(),
        generator.subschema_for::<EpipolarOverlay>().to_value(),
    );
    props.insert(
        "disparity_result".into(),
        generator.subschema_for::<DisparityResult>().to_value(),
    );
    props.insert(
        "run_progress".into(),
        generator.subschema_for::<RunProgress>().to_value(),
    );

    let defs = generator.take_definitions(true);

    json!({
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "DiagnoseWireTypes",
        "description": "Generated wire types for the diagnose app (B-QUAL2). \
            Do not edit by hand — run `bun run generate:types`. The top-level \
            wrapper only anchors the `definitions`; consumers import the \
            individual interfaces.",
        "type": "object",
        "properties": Value::Object(props),
        "definitions": defs,
    })
}

/// `app/schemas-generated`, resolved from this crate's manifest dir so the
/// output location is independent of the caller's working directory.
fn out_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("schemas-generated")
}

/// Write `schema` to `path` (or, in `--check` mode, compare). Returns
/// `Ok(true)` when the on-disk file drifts from `schema` under `--check`.
fn write_or_check(path: &Path, schema: &Value, check: bool) -> Result<bool, String> {
    let mut text = serde_json::to_string_pretty(schema)
        .map_err(|e| format!("rendering {}: {e}", path.display()))?;
    text.push('\n');

    if check {
        let on_disk = std::fs::read_to_string(path).unwrap_or_default();
        return Ok(on_disk != text);
    }

    std::fs::write(path, &text).map_err(|e| format!("writing {}: {e}", path.display()))?;
    Ok(false)
}
