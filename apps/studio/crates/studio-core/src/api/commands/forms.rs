//! The forms' schemas and values, from the library this back end links: a
//! topology's config schema and default, the dataset manifest's schema, and
//! a config resolved from a patch. The frontend carries no schema copy, so
//! its forms cannot lag the back end it talks to.

use schemars::{JsonSchema, Schema, schema_for};
use serde::{Deserialize, Serialize};
use vision_calibration::dataset::{DatasetSpec, ProblemKind};
use vision_calibration::dataset_runner::{CalibrationConfig, ConfigPatch};
use vision_calibration::laserline_device::LaserlineDeviceConfig;
use vision_calibration::planar_intrinsics::PlanarIntrinsicsConfig;
use vision_calibration::rig_extrinsics::RigExtrinsicsConfig;
use vision_calibration::rig_handeye::RigHandeyeConfig;
use vision_calibration::rig_handeye_laserline::RigHandeyeLaserlineConfig;
use vision_calibration::rig_laserline_device::RigLaserlineDeviceConfig;
use vision_calibration::single_cam_handeye::SingleCamHandeyeConfig;

use crate::api::{Ctx, ErrorKind, StudioError};

/// A form: the JSON Schema it renders and the value it starts from.
#[derive(Debug, Serialize, JsonSchema)]
pub struct FormSchema {
    /// A JSON Schema (draft 2020-12, as the library emits it).
    pub schema: serde_json::Value,
    /// The value a new form starts from: the type's default.
    pub default: serde_json::Value,
}

/// The config form of the workflow `topology`: the schema of its config
/// (`PlanarIntrinsicsConfig`, `RigHandeyeConfig`, …) and the library's
/// default configuration.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ConfigFormArgs {
    pub topology: ProblemKind,
}

pub fn config_form(_: &Ctx, args: ConfigFormArgs) -> Result<FormSchema, StudioError> {
    Ok(FormSchema {
        schema: config_schema(args.topology)?.to_value(),
        default: config_fields(&CalibrationConfig::default_for(args.topology))?,
    })
}

/// The dataset manifest form: the schema of `DatasetSpec` and the manifest
/// a new setup starts from (one chessboard camera, planar intrinsics).
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DatasetFormArgs {}

pub fn dataset_form(_: &Ctx, _: DatasetFormArgs) -> Result<FormSchema, StudioError> {
    Ok(FormSchema {
        schema: schema_for!(DatasetSpec).to_value(),
        default: serde_json::json!({
            "version": 1,
            "topology": "planar_intrinsics",
            "target": {
                "kind": "chessboard",
                "rows": 9,
                "cols": 6,
                "square_size_m": 0.025
            },
            "cameras": [
                { "id": "cam0", "images": { "kind": "glob", "pattern": "*.png" } }
            ]
        }),
    })
}

/// The configuration of the workflow `topology` that `patch` makes from its
/// defaults: tables merge key by key, anything else replaces (the library's
/// rule, as `vcal run --config` applies it), then checked. A field the
/// workflow does not have, or an invalid value, is a `config` error.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ResolveConfigArgs {
    pub topology: ProblemKind,
    #[serde(default)]
    pub patch: ConfigPatch,
}

/// A topology's configuration: the config object the form edits (the
/// workflow's `*Config` JSON).
pub type ResolvedConfig = serde_json::Value;

pub fn resolve_config(_: &Ctx, args: ResolveConfigArgs) -> Result<ResolvedConfig, StudioError> {
    let config = CalibrationConfig::patched(args.topology, &args.patch)
        .map_err(|e| StudioError::new(ErrorKind::Config, e.to_string()))?;
    config_fields(&config)
}

/// The schema of `topology`'s config type, as the library emits it.
fn config_schema(topology: ProblemKind) -> Result<Schema, StudioError> {
    Ok(match topology {
        ProblemKind::PlanarIntrinsics => schema_for!(PlanarIntrinsicsConfig),
        ProblemKind::SingleCamHandeye => schema_for!(SingleCamHandeyeConfig),
        ProblemKind::LaserlineDevice => schema_for!(LaserlineDeviceConfig),
        ProblemKind::RigExtrinsics => schema_for!(RigExtrinsicsConfig),
        ProblemKind::RigHandeye => schema_for!(RigHandeyeConfig),
        ProblemKind::RigLaserlineDevice => schema_for!(RigLaserlineDeviceConfig),
        ProblemKind::RigHandeyeLaserline => schema_for!(RigHandeyeLaserlineConfig),
        other => {
            return Err(StudioError::invalid(format!(
                "no config form for the workflow {other:?}"
            )));
        }
    })
}

/// The problem config's own fields: the `config` of the tagged form.
fn config_fields(config: &CalibrationConfig) -> Result<serde_json::Value, StudioError> {
    let tagged = serde_json::to_value(config)
        .map_err(|e| StudioError::internal(format!("serialize the config: {e}")))?;
    match tagged {
        serde_json::Value::Object(mut fields) => fields
            .remove("config")
            .ok_or_else(|| StudioError::internal("a config serializes with its fields")),
        _ => Err(StudioError::internal("a config serializes as an object")),
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use serde_json::json;

    use crate::api::{Api, ErrorKind};
    use crate::scope::RootScope;

    const TOPOLOGIES: [&str; 7] = [
        "planar_intrinsics",
        "single_cam_handeye",
        "laserline_device",
        "rig_extrinsics",
        "rig_handeye",
        "rig_laserline_device",
        "rig_handeye_laserline",
    ];

    /// The schemas the library publishes (`cargo xtask emit-schemas`).
    fn published(name: &str) -> serde_json::Value {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../../schemas")
            .join(format!("{name}.json"));
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap()
    }

    #[test]
    fn the_forms_are_the_published_schemas_and_defaults() {
        let api = Api::for_tests(RootScope::fixed([]));
        for topology in TOPOLOGIES {
            let form = api
                .invoke("config_form", json!({ "topology": topology }))
                .unwrap();
            let name = format!("{topology}_config");
            assert_eq!(form["schema"], published(&name), "{topology}");
            assert_eq!(
                form["default"],
                published(&format!("{name}.default")),
                "{topology}"
            );
        }
        let form = api.invoke("dataset_form", json!({})).unwrap();
        assert_eq!(form["schema"], published("dataset_spec"));
        assert_eq!(form["default"]["topology"], "planar_intrinsics");
        let err = api
            .invoke("config_form", json!({ "topology": "no_such_workflow" }))
            .unwrap_err();
        assert_eq!(err.kind, ErrorKind::InvalidArgs);
    }

    #[test]
    fn a_patch_resolves_by_the_library_rule() {
        let api = Api::for_tests(RootScope::fixed([]));
        let defaults = api
            .invoke("config_form", json!({ "topology": "planar_intrinsics" }))
            .unwrap()["default"]
            .clone();
        let none = api
            .invoke("resolve_config", json!({ "topology": "planar_intrinsics" }))
            .unwrap();
        assert_eq!(none, defaults);

        let patched = api
            .invoke(
                "resolve_config",
                json!({
                    "topology": "planar_intrinsics",
                    "patch": { "solver": { "max_iters": 7 } }
                }),
            )
            .unwrap();
        assert_eq!(patched["solver"]["max_iters"], 7);
        // A table merges: the solver's other fields keep their defaults.
        let mut expected = defaults.clone();
        expected["solver"]["max_iters"] = json!(7);
        assert_eq!(patched, expected);

        let err = api
            .invoke(
                "resolve_config",
                json!({ "topology": "planar_intrinsics", "patch": { "no_such_field": 1 } }),
            )
            .unwrap_err();
        assert_eq!(err.kind, ErrorKind::Config);
    }
}
