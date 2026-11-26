//! [`ConfigPatch`]: changes to a workflow's default configuration, and
//! [`CalibrationConfig::patched`], the configuration they make.

use serde::{Deserialize, Serialize, Serializer};
use serde_json::{Map, Value};
use vision_calibration_dataset::ProblemKind;

use super::CalibrationConfig;
use crate::Error;

/// Changes to a workflow's default configuration: what the patch names
/// replaces the default, what it leaves out keeps it.
///
/// A patch is keyed like the problem's own config, a table of its fields
/// (`{"solver": {"max_iters": 20}}`, or in TOML `solver.max_iters = 20`),
/// not the tagged `{"topology": …, "config": …}` form of
/// [`CalibrationConfig`]. It applies by one rule: two tables merge key by
/// key, and any other value (a number, a string, a list, `null`) replaces
/// what it lands on. [`CalibrationConfig::patched`] applies a patch to the
/// defaults of a topology and checks the result; a field the workflow does
/// not have is an error there, so a misspelt name never passes silently.
///
/// TOML has no `null`, so a patch read from TOML cannot set an optional
/// field back to none; JSON can.
///
/// The patch serializes as its table, with the keys of every table sorted.
///
/// ```
/// use vision_calibration_dataset::ProblemKind;
/// use vision_calibration_pipeline::dataset_runner::{CalibrationConfig, ConfigPatch};
///
/// let patch = ConfigPatch::from_toml_str("solver.max_iters = 20")?;
/// let config = CalibrationConfig::patched(ProblemKind::PlanarIntrinsics, &patch)?;
/// let CalibrationConfig::PlanarIntrinsics(planar) = &config else {
///     unreachable!("the patch applies to the topology's own config");
/// };
/// assert_eq!(planar.solver.max_iters, 20);
/// // The smallest patch that makes the configuration from the defaults.
/// assert_eq!(ConfigPatch::from_config(&config), patch);
/// # Ok::<(), vision_calibration_pipeline::Error>(())
/// ```
#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[serde(transparent)]
pub struct ConfigPatch(Map<String, Value>);

impl ConfigPatch {
    /// Whether the patch changes nothing.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// A patch from JSON text, an object of the config's fields.
    ///
    /// # Errors
    ///
    /// [`Error::Serde`] when the text is not JSON, [`Error::InvalidInput`]
    /// when it is not an object.
    pub fn from_json_str(text: &str) -> Result<Self, Error> {
        let value: Value = serde_json::from_str(text).map_err(|e| Error::Serde(e.to_string()))?;
        Self::from_value(value)
    }

    /// A patch from TOML text, a document of the config's fields.
    ///
    /// # Errors
    ///
    /// [`Error::Serde`] when the text is not TOML.
    pub fn from_toml_str(text: &str) -> Result<Self, Error> {
        let value: Value = toml::from_str(text).map_err(|e| Error::Serde(e.to_string()))?;
        Self::from_value(value)
    }

    fn from_value(value: Value) -> Result<Self, Error> {
        match value {
            Value::Object(fields) => Ok(Self(fields)),
            other => Err(Error::InvalidInput {
                reason: format!(
                    "a config patch is a table of the config's fields, not {}",
                    json_kind(&other)
                ),
            }),
        }
    }

    /// The smallest patch that makes `config` from the defaults of its
    /// topology: the fields that differ from the default, a table only
    /// where some of its fields do, and `null` for an optional field the
    /// default sets and `config` leaves unset.
    ///
    /// [`CalibrationConfig::patched`] with the config's topology and this
    /// patch gives `config` back for every configuration a patch can
    /// state. One it cannot is a switch between two variants of a field
    /// that each carry fields (a robust loss from Huber to Cauchy, say)
    /// where the earlier variant's fields would be left over; `patched`
    /// then refuses them as fields the workflow does not have.
    pub fn from_config(config: &CalibrationConfig) -> Self {
        let defaults = config_fields(&CalibrationConfig::default_for(config.topology()));
        Self(difference(&defaults, &config_fields(config)))
    }

    /// This patch, then `later` over it: the patch that applies both, by
    /// the same rule, `later` winning where both name a value.
    pub fn then(&self, later: &ConfigPatch) -> ConfigPatch {
        let mut fields = self.0.clone();
        merge_fields(&mut fields, &later.0);
        Self(fields)
    }
}

impl Serialize for ConfigPatch {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        SortedFields(&self.0).serialize(serializer)
    }
}

/// A JSON table serialized with the keys of every table sorted, whatever
/// order the map keeps (`serde_json`'s `preserve_order` keeps insertion
/// order).
struct SortedFields<'a>(&'a Map<String, Value>);

impl Serialize for SortedFields<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeMap;
        let mut keys: Vec<&String> = self.0.keys().collect();
        keys.sort();
        let mut map = serializer.serialize_map(Some(keys.len()))?;
        for key in keys {
            map.serialize_entry(key, &Sorted(&self.0[key]))?;
        }
        map.end()
    }
}

struct Sorted<'a>(&'a Value);

impl Serialize for Sorted<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeSeq;
        match self.0 {
            Value::Object(fields) => SortedFields(fields).serialize(serializer),
            Value::Array(items) => {
                let mut seq = serializer.serialize_seq(Some(items.len()))?;
                for item in items {
                    seq.serialize_element(&Sorted(item))?;
                }
                seq.end()
            }
            other => other.serialize(serializer),
        }
    }
}

impl CalibrationConfig {
    /// The default configuration of `topology` with `patch` applied, as
    /// [`ConfigPatch`] states the rule, then checked as
    /// [`calibrate_dataset`](super::calibrate_dataset) checks a config.
    ///
    /// # Errors
    ///
    /// [`Error::Serde`] when the patched fields do not form the topology's
    /// config (a field it does not have, a value of the wrong type), and
    /// the problem's own error when the config is invalid
    /// ([`Error::InvalidInput`] naming the field).
    pub fn patched(topology: ProblemKind, patch: &ConfigPatch) -> Result<Self, Error> {
        let mut tagged = serde_json::to_value(Self::default_for(topology))
            .map_err(|e| Error::Serde(e.to_string()))?;
        let fields = tagged
            .get_mut("config")
            .and_then(Value::as_object_mut)
            .ok_or_else(|| Error::Serde("a default config serializes as a table".into()))?;
        merge_fields(fields, &patch.0);
        let config: Self =
            serde_json::from_value(tagged).map_err(|e| Error::Serde(e.to_string()))?;
        config.validate()?;
        Ok(config)
    }
}

/// The problem config's own fields: the `config` table of the tagged form.
fn config_fields(config: &CalibrationConfig) -> Map<String, Value> {
    match serde_json::to_value(config) {
        Ok(Value::Object(mut tagged)) => match tagged.remove("config") {
            Some(Value::Object(fields)) => fields,
            _ => Map::new(),
        },
        _ => Map::new(),
    }
}

/// `changes` merged into `base`: tables key by key, any other value
/// replacing what it lands on.
fn merge_fields(base: &mut Map<String, Value>, changes: &Map<String, Value>) {
    for (key, value) in changes {
        match base.get_mut(key) {
            Some(slot) => merge(slot, value),
            None => {
                base.insert(key.clone(), value.clone());
            }
        }
    }
}

fn merge(base: &mut Value, changes: &Value) {
    match (base, changes) {
        (Value::Object(base), Value::Object(changes)) => merge_fields(base, changes),
        (slot, value) => *slot = value.clone(),
    }
}

/// What turns `base` into `target` under [`merge_fields`]: the keys whose
/// values differ, tables recursed into, `null` for a key `target` drops.
fn difference(base: &Map<String, Value>, target: &Map<String, Value>) -> Map<String, Value> {
    let mut out = Map::new();
    for (key, value) in target {
        match (base.get(key), value) {
            (Some(old), new) if old == new => {}
            (Some(Value::Object(old)), Value::Object(new)) => {
                out.insert(key.clone(), Value::Object(difference(old, new)));
            }
            _ => {
                out.insert(key.clone(), value.clone());
            }
        }
    }
    for key in base.keys() {
        if !target.contains_key(key) {
            out.insert(key.clone(), Value::Null);
        }
    }
    out
}

fn json_kind(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "a boolean",
        Value::Number(_) => "a number",
        Value::String(_) => "a string",
        Value::Array(_) => "a list",
        Value::Object(_) => "a table",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const ALL: [ProblemKind; 7] = [
        ProblemKind::PlanarIntrinsics,
        ProblemKind::SingleCamHandeye,
        ProblemKind::LaserlineDevice,
        ProblemKind::RigExtrinsics,
        ProblemKind::RigHandeye,
        ProblemKind::RigLaserlineDevice,
        ProblemKind::RigHandeyeLaserline,
    ];

    fn patch(value: Value) -> ConfigPatch {
        serde_json::from_value(value).unwrap()
    }

    fn fields(config: &CalibrationConfig) -> Value {
        Value::Object(config_fields(config))
    }

    /// Whether `topology`'s config has the top-level field `key`.
    fn has(topology: ProblemKind, key: &str) -> bool {
        config_fields(&CalibrationConfig::default_for(topology)).contains_key(key)
    }

    #[test]
    fn an_empty_patch_is_the_default() {
        for topology in ALL {
            let config = CalibrationConfig::patched(topology, &ConfigPatch::default()).unwrap();
            assert_eq!(
                fields(&config),
                fields(&CalibrationConfig::default_for(topology))
            );
            assert!(ConfigPatch::from_config(&config).is_empty());
        }
    }

    #[test]
    fn a_patch_replaces_what_it_names_and_keeps_the_rest() {
        let text = r#"
            solver.max_iters = 7
            camera.model = { projection = "pinhole", distortion = "none", sensor = "identity" }
        "#;
        let from_toml = ConfigPatch::from_toml_str(text).unwrap();
        let from_json = ConfigPatch::from_json_str(
            r#"{"camera": {"model": {"projection": "pinhole",
                "distortion": "none", "sensor": "identity"}},
                "solver": {"max_iters": 7}}"#,
        )
        .unwrap();
        assert_eq!(from_toml, from_json);
        // The topologies whose config has a camera (a rig laser device takes
        // its cameras from the upstream calibration).
        for topology in ALL.into_iter().filter(|&t| has(t, "camera")) {
            let config = CalibrationConfig::patched(topology, &from_toml).unwrap();
            let mut expected = fields(&CalibrationConfig::default_for(topology));
            expected["solver"]["max_iters"] = json!(7);
            expected["camera"]["model"] = json!({
                "projection": "pinhole", "distortion": "none", "sensor": "identity"
            });
            assert_eq!(fields(&config), expected, "{topology:?}");
        }
    }

    #[test]
    fn from_config_is_the_smallest_patch_and_gives_the_config_back() {
        let changes = [
            json!({}),
            json!({"solver": {"max_iters": 9, "backend": "factrs"}}),
            json!({"camera": {"fix": {"distortion": ["k3"]}}}),
            json!({"camera": {"model": {"projection": "pinhole",
                "distortion": "none", "sensor": "identity"}}}),
        ];
        for topology in ALL {
            let applies = |change: &&Value| {
                change
                    .as_object()
                    .unwrap()
                    .keys()
                    .all(|key| has(topology, key))
            };
            for change in changes.iter().filter(applies) {
                let config = CalibrationConfig::patched(topology, &patch(change.clone())).unwrap();
                let smallest = ConfigPatch::from_config(&config);
                let back = CalibrationConfig::patched(topology, &smallest).unwrap();
                assert_eq!(fields(&back), fields(&config), "{topology:?} {change}");
                // It names no more than the change did.
                let named = serde_json::to_value(&smallest).unwrap();
                assert!(
                    named.as_object().unwrap().len() <= change.as_object().unwrap().len(),
                    "{topology:?}: {named} for {change}"
                );
            }
        }
        // A variant switch with fewer fields, and back with other values.
        let robot = |kind: Value| json!({"handeye_optimize": {"robot_poses": kind}});
        for change in [
            robot(json!({"kind": "exact"})),
            robot(json!({"kind": "refine", "rot_sigma": 0.02})),
        ] {
            let config =
                CalibrationConfig::patched(ProblemKind::RigHandeye, &patch(change.clone()))
                    .unwrap();
            let back = CalibrationConfig::patched(
                ProblemKind::RigHandeye,
                &ConfigPatch::from_config(&config),
            )
            .unwrap();
            assert_eq!(fields(&back), fields(&config), "{change}");
        }
    }

    #[test]
    fn then_layers_a_later_patch_over_an_earlier_one() {
        let project = patch(json!({"solver": {"max_iters": 9, "verbosity": 1}}));
        let run = patch(json!({"solver": {"max_iters": 3}}));
        assert_eq!(
            project.then(&run),
            patch(json!({"solver": {"max_iters": 3, "verbosity": 1}}))
        );
        assert_eq!(ConfigPatch::default().then(&run), run);
        assert_eq!(run.then(&ConfigPatch::default()), run);
    }

    #[test]
    fn a_patch_is_checked_against_the_config() {
        let misspelt = patch(json!({"solver": {"max_iter": 9}}));
        let err = CalibrationConfig::patched(ProblemKind::PlanarIntrinsics, &misspelt).unwrap_err();
        assert!(
            matches!(err, Error::Serde(ref m) if m.contains("max_iter")),
            "{err:?}"
        );
        let invalid = patch(json!({"solver": {"max_iters": 0}}));
        let err = CalibrationConfig::patched(ProblemKind::PlanarIntrinsics, &invalid).unwrap_err();
        assert!(matches!(err, Error::InvalidInput { .. }), "{err:?}");
    }

    #[test]
    fn a_patch_is_a_table() {
        for text in ["[1, 2]", "3", "\"solver\""] {
            let err = ConfigPatch::from_json_str(text).unwrap_err();
            assert!(matches!(err, Error::InvalidInput { .. }), "{text}: {err:?}");
        }
        assert!(matches!(
            ConfigPatch::from_json_str("{").unwrap_err(),
            Error::Serde(_)
        ));
        assert!(matches!(
            ConfigPatch::from_toml_str("solver = ").unwrap_err(),
            Error::Serde(_)
        ));
    }

    #[test]
    fn a_patch_serializes_with_its_keys_sorted_and_round_trips() {
        let mut fields = Map::new();
        fields.insert("solver".into(), json!({"verbosity": 1, "max_iters": 3}));
        fields.insert("camera".into(), json!({"fix": {"k3": true}}));
        let patch = ConfigPatch(fields);
        let text = serde_json::to_string(&patch).unwrap();
        assert_eq!(
            text,
            r#"{"camera":{"fix":{"k3":true}},"solver":{"max_iters":3,"verbosity":1}}"#
        );
        let back: ConfigPatch = serde_json::from_str(&text).unwrap();
        assert_eq!(back, patch);
        let toml_text = toml::to_string(&patch).unwrap();
        assert_eq!(ConfigPatch::from_toml_str(&toml_text).unwrap(), patch);
    }
}
