//! `studio-api.json`: the command table as a JSON Schema document, from
//! which `scripts/generate-api.mjs` writes the frontend's wire types and its
//! typed client.
//!
//! One generator, one `definitions` block: every type (the library's
//! manifest, configs and exports, the app's own replies) appears once.
//! Draft-07 `definitions` rather than 2020-12 `$defs`, because
//! json-schema-to-typescript names and dedupes interfaces from
//! `definitions` and re-inlines shared types from `$defs`.

use schemars::generate::SchemaSettings;
use serde_json::{Value, json};
use vision_calibration::laserline_device::LaserlineDeviceExport;
use vision_calibration::planar_intrinsics::PlanarIntrinsicsExport;
use vision_calibration::rig_extrinsics::RigExtrinsicsExport;
use vision_calibration::rig_handeye::RigHandeyeExport;
use vision_calibration::rig_handeye_laserline::RigHandeyeLaserlineExport;
use vision_calibration::rig_laserline_device::RigLaserlineDeviceExport;
use vision_calibration::single_cam_handeye::SingleCamHandeyeExport;

use super::error::StudioError;
use super::events::ServerEvent;
use super::jobs::Lane;
use super::table::{CommandSpec, Kind};
use crate::run_source::ViewerSession;

pub(crate) fn build(specs: &[CommandSpec]) -> Value {
    let mut generator = SchemaSettings::draft07().into_generator();
    let mut commands = Vec::with_capacity(specs.len());
    for spec in specs {
        let schema = (spec.schema)(&mut generator);
        let mut command = json!({
            "name": spec.name,
            "kind": match spec.kind {
                Kind::Plain => "plain",
                Kind::Job { .. } => "job",
                Kind::Resource => "resource",
            },
            "args": schema.args.to_value(),
        });
        if let Kind::Job { lane } = spec.kind {
            command["lane"] = json!(match lane {
                Lane::Heavy => "heavy",
                Lane::Light => "light",
            });
        }
        for (key, part) in [
            ("reply", schema.reply),
            ("event", schema.event),
            ("output", schema.output),
        ] {
            if let Some(part) = part {
                command[key] = part.to_value();
            }
        }
        commands.push(command);
    }

    // Wire types the frontend reads that no command names: the exports
    // (`load_export` answers them as JSON it narrows by `kind`) and the
    // browser viewer's session.
    let mut types = serde_json::Map::new();
    for (name, schema) in [
        (
            "PlanarIntrinsicsExport",
            generator.subschema_for::<PlanarIntrinsicsExport>(),
        ),
        (
            "SingleCamHandeyeExport",
            generator.subschema_for::<SingleCamHandeyeExport>(),
        ),
        (
            "LaserlineDeviceExport",
            generator.subschema_for::<LaserlineDeviceExport>(),
        ),
        (
            "RigExtrinsicsExport",
            generator.subschema_for::<RigExtrinsicsExport>(),
        ),
        (
            "RigHandeyeExport",
            generator.subschema_for::<RigHandeyeExport>(),
        ),
        (
            "RigLaserlineDeviceExport",
            generator.subschema_for::<RigLaserlineDeviceExport>(),
        ),
        (
            "RigHandeyeLaserlineExport",
            generator.subschema_for::<RigHandeyeLaserlineExport>(),
        ),
        ("ViewerSession", generator.subschema_for::<ViewerSession>()),
    ] {
        types.insert(name.into(), schema.to_value());
    }
    let events = generator.subschema_for::<ServerEvent>().to_value();
    let error = generator.subschema_for::<StudioError>().to_value();
    let definitions = generator.take_definitions(true);

    // A command's documentation is its argument object's.
    for command in &mut commands {
        let doc = definition(&definitions, &command["args"])
            .and_then(|def| def.get("description"))
            .cloned();
        if let Some(doc) = doc {
            command["doc"] = doc;
        }
    }

    json!({
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "StudioApi",
        "description": "Calibration Studio's back-end API: every command with its argument, \
            reply, event and output types, the event and error types, and the wire types \
            the frontend reads. Do not edit by hand: `bun run generate:api` regenerates it.",
        "commands": commands,
        "events": events,
        "error": error,
        "types": types,
        "definitions": definitions,
    })
}

/// The definition a `{"$ref": "#/definitions/<name>"}` schema points at.
pub(crate) fn definition<'a>(
    definitions: &'a serde_json::Map<String, Value>,
    schema: &Value,
) -> Option<&'a Value> {
    let name = schema
        .get("$ref")?
        .as_str()?
        .strip_prefix("#/definitions/")?;
    definitions.get(name)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;
    use crate::api::Api;

    fn is_snake_case(name: &str) -> bool {
        name.starts_with(|c: char| c.is_ascii_lowercase())
            && name
                .chars()
                .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
            && !name.contains("__")
            && !name.ends_with('_')
    }

    /// A property name that holds a path, and so must be a `UserPath`.
    fn names_a_path(name: &str) -> bool {
        matches!(name, "path" | "dir" | "folder" | "root" | "file")
            || name.ends_with("_path")
            || name.ends_with("_dir")
    }

    fn is_user_path(schema: &Value) -> bool {
        // A documented field wraps its `$ref` in `allOf`, an optional one in
        // `anyOf` with `null`.
        schema.get("$ref").and_then(Value::as_str) == Some("#/definitions/UserPath")
            || ["allOf", "anyOf"].iter().any(|key| {
                schema
                    .get(key)
                    .and_then(Value::as_array)
                    .is_some_and(|options| options.iter().any(is_user_path))
            })
    }

    /// Every path-named property reachable from `schema` that is not a
    /// `UserPath`, as `Type.field`.
    fn bare_paths(
        definitions: &serde_json::Map<String, Value>,
        schema: &Value,
        seen: &mut BTreeSet<String>,
        found: &mut Vec<String>,
    ) {
        if let Some(name) = schema
            .get("$ref")
            .and_then(Value::as_str)
            .and_then(|r| r.strip_prefix("#/definitions/"))
        {
            if seen.insert(name.to_owned())
                && let Some(def) = definitions.get(name)
            {
                if let Some(props) = def.get("properties").and_then(Value::as_object) {
                    for (field, prop) in props {
                        if names_a_path(field) && !is_user_path(prop) {
                            found.push(format!("{name}.{field}"));
                        }
                    }
                }
                bare_paths(definitions, def, seen, found);
            }
            return;
        }
        match schema {
            Value::Object(map) => {
                for value in map.values() {
                    bare_paths(definitions, value, seen, found);
                }
            }
            Value::Array(items) => {
                for value in items {
                    bare_paths(definitions, value, seen, found);
                }
            }
            _ => {}
        }
    }

    #[test]
    fn the_table_holds_its_invariants() {
        let schema = Api::schema();
        let definitions = schema["definitions"].as_object().unwrap();
        let mut names = BTreeSet::new();
        for command in schema["commands"].as_array().unwrap() {
            let name = command["name"].as_str().unwrap();
            assert!(names.insert(name), "{name} is listed twice");
            assert!(is_snake_case(name), "{name} is not snake_case");
            assert!(command["doc"].is_string(), "{name} has no documentation");
            let args = definition(definitions, &command["args"])
                .unwrap_or_else(|| panic!("{name}: the arguments are not a named type"));
            assert_eq!(args["type"], "object", "{name}: arguments are an object");
            assert_eq!(
                args["additionalProperties"], false,
                "{name}: unknown arguments must be refused"
            );
            let mut found = Vec::new();
            bare_paths(
                definitions,
                &command["args"],
                &mut BTreeSet::new(),
                &mut found,
            );
            assert!(
                found.is_empty(),
                "{name}: paths that are not UserPath: {found:?}"
            );
            let fields = args["properties"].as_object().into_iter().flatten();
            for (field, _) in fields {
                assert!(is_snake_case(field), "{name}.{field} is not snake_case");
            }
        }
    }

    #[test]
    fn every_type_the_frontend_reads_is_defined_once() {
        let schema = Api::schema();
        let definitions = schema["definitions"].as_object().unwrap();
        for name in [
            "ServerEvent",
            "StudioError",
            "JobInfo",
            "RunProgress",
            "RunResponse",
            "ProjectInfo",
            "DatasetSpec",
            "RigExtrinsicsExport",
            "ViewerSession",
            "UserPath",
        ] {
            assert!(definitions.contains_key(name), "{name} is missing");
        }
    }
}
