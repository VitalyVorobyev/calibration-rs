//! The project file: [`ProjectSpec`], its dataset ([`DatasetSource`]) and
//! the device it calibrates ([`DeviceIdentity`]).

use std::path::PathBuf;

use serde::de::Error as _;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use vision_calibration_dataset::DatasetSpec;

use super::{Exclusions, PROJECT_VERSION, SeedSource};
use crate::analysis::QualityThresholds;
use crate::dataset_runner::ConfigPatch;

/// A calibration project's definition: its dataset, the changes to its
/// workflow's default configuration, the quality thresholds its runs are
/// judged by and what it leaves out of the dataset.
///
/// It is the content of `calibration-project.toml` (or of
/// `calibration-project.json`). An unknown field is an error, so a misspelt
/// section such as `[exlude]` never passes silently.
///
/// ```toml
/// version = 2
/// name = "Bench camera"
///
/// [device]
/// product = "line scanner"
/// id = "SN-0042"
///
/// [seed]
/// run = "../SN-0041/runs/run-002"
///
/// [dataset]
/// manifest = "captures/dataset.toml"
///
/// [config]
/// solver.max_iters = 80
///
/// [exclude]
/// images = [{ camera = "cam0", image = "cam0/0003.png" }]
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct ProjectSpec {
    /// The project file's format version, [`PROJECT_VERSION`].
    pub version: u32,
    /// The project's name, for display; without one, the project's folder
    /// name is shown.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// What the project is about, for the reader.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// The device the project calibrates: the key its runs are grouped by
    /// across projects.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device: Option<DeviceIdentity>,
    /// The seed every run starts from instead of the closed-form
    /// initialization; `None` for none.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<SeedSource>,
    /// The dataset: its manifest, written in the project file or kept in a
    /// file of its own.
    pub dataset: DatasetSource,
    /// Changes to the default configuration of the dataset's workflow.
    #[serde(default, skip_serializing_if = "ConfigPatch::is_empty")]
    pub config: ConfigPatch,
    /// The thresholds the runs' quality is judged by; `None` for the
    /// defaults.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quality: Option<QualityThresholds>,
    /// What the runs leave out of the dataset.
    #[serde(default, skip_serializing_if = "Exclusions::is_empty")]
    pub exclude: Exclusions,
}

impl ProjectSpec {
    /// A project of `dataset` at the current version, with no name,
    /// device, seed, configuration changes, thresholds or exclusions.
    pub fn new(dataset: DatasetSource) -> Self {
        Self {
            version: PROJECT_VERSION,
            name: None,
            description: None,
            device: None,
            seed: None,
            dataset,
            config: ConfigPatch::default(),
            quality: None,
            exclude: Exclusions::default(),
        }
    }

    /// A project of the manifest file at `path` ([`DatasetSource::File`]).
    pub fn for_manifest(path: impl Into<PathBuf>) -> Self {
        Self::new(DatasetSource::file(path))
    }
}

/// The device a project calibrates.
///
/// Projects of one device, in any folder, form its calibration history:
/// the library groups them by `product` and `id` together.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct DeviceIdentity {
    /// The device type or product line it belongs to; `None` when the id
    /// alone names the device.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub product: Option<String>,
    /// The device's own identifier: a serial number, a MAC address. Not
    /// empty, without leading or trailing white space, compared as written.
    pub id: String,
    /// A name for display.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
}

impl DeviceIdentity {
    /// The device `id`, of no stated product and without a label.
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            product: None,
            id: id.into(),
            label: None,
        }
    }
}

/// Where a project's dataset manifest is.
///
/// Either the manifest is written in the project file, with the folder its
/// relative paths resolve against, or it is a manifest file of its own,
/// which the project references and the library never writes. In the
/// project file, `manifest` is a table for the first and a path for the
/// second:
///
/// ```toml
/// [dataset]
/// root = "images"            # optional, default "."
/// [dataset.manifest]
/// version = 1
/// topology = "planar_intrinsics"
/// # ... the manifest's fields
/// ```
///
/// ```toml
/// [dataset]
/// manifest = "../captures/dataset.toml"
/// ```
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum DatasetSource {
    /// The manifest is part of the project file.
    #[non_exhaustive]
    Inline {
        /// The folder the manifest's relative paths resolve against:
        /// relative to the project folder unless absolute.
        root: PathBuf,
        /// The manifest.
        spec: Box<DatasetSpec>,
    },
    /// The manifest is a file of its own, read at every use and never
    /// written by the library.
    #[non_exhaustive]
    File {
        /// The manifest file: relative to the project folder unless
        /// absolute. Its relative paths resolve against its own folder, as
        /// a manifest's do.
        path: PathBuf,
    },
}

impl DatasetSource {
    /// The manifest `spec`, written in the project file, whose relative
    /// paths resolve against `root`.
    pub fn inline(root: impl Into<PathBuf>, spec: DatasetSpec) -> Self {
        Self::Inline {
            root: root.into(),
            spec: Box::new(spec),
        }
    }

    /// The manifest file at `path`.
    pub fn file(path: impl Into<PathBuf>) -> Self {
        Self::File { path: path.into() }
    }
}

/// How [`DatasetSource`] is written: `manifest` a path or a table, `root`
/// beside a table only.
#[derive(Serialize)]
struct SourceOut<'a, M: Serialize> {
    #[serde(skip_serializing_if = "Option::is_none")]
    root: Option<&'a PathBuf>,
    manifest: M,
}

impl Serialize for DatasetSource {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Inline { root, spec } => SourceOut {
                root: (root.as_os_str() != ".").then_some(root),
                manifest: spec.as_ref(),
            }
            .serialize(serializer),
            Self::File { path } => SourceOut {
                root: None,
                manifest: path,
            }
            .serialize(serializer),
        }
    }
}

/// How [`DatasetSource`] is read. `manifest` is read as a value first: a
/// string is a file, a table is read as a manifest directly, so a typo
/// inside it reports the manifest's own field.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SourceIn {
    #[serde(default)]
    root: Option<PathBuf>,
    manifest: serde_json::Value,
}

impl<'de> Deserialize<'de> for DatasetSource {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let SourceIn { root, manifest } = SourceIn::deserialize(deserializer)?;
        match manifest {
            serde_json::Value::String(path) => match root {
                Some(_) => Err(D::Error::custom(
                    "`root` is given with a manifest file; a manifest file's paths resolve \
                     against its own folder, so leave `root` out",
                )),
                None => Ok(Self::file(path)),
            },
            table @ serde_json::Value::Object(_) => {
                let spec: DatasetSpec = serde_json::from_value(table)
                    .map_err(|e| D::Error::custom(format!("manifest: {e}")))?;
                Ok(Self::inline(
                    root.unwrap_or_else(|| PathBuf::from(".")),
                    spec,
                ))
            }
            _ => Err(D::Error::custom(
                "`manifest` is the path of a manifest file or the manifest's table",
            )),
        }
    }
}

/// Where a project's dataset manifest is: `manifest` the path of a
/// manifest file of its own, or the manifest itself, written in the project
/// file, with `root` the folder its relative paths resolve against.
#[cfg(feature = "schemars")]
#[derive(schemars::JsonSchema)]
#[schemars(rename = "DatasetSource")]
#[serde(deny_unknown_fields)]
#[allow(
    dead_code,
    reason = "a schema proxy: only its JsonSchema derive is used"
)]
struct DatasetSourceSchema {
    /// For a manifest written in the project file: the folder its relative
    /// paths resolve against, relative to the project folder unless
    /// absolute. Default `.`. Not allowed beside a manifest file.
    #[serde(default)]
    root: Option<PathBuf>,
    /// The manifest file's path (relative to the project folder unless
    /// absolute), or the manifest.
    manifest: DatasetManifest,
}

#[cfg(feature = "schemars")]
#[derive(schemars::JsonSchema)]
/// A manifest file's path, or the manifest.
#[serde(untagged)]
#[allow(
    dead_code,
    reason = "a schema proxy: only its JsonSchema derive is used"
)]
enum DatasetManifest {
    File(PathBuf),
    Inline(Box<DatasetSpec>),
}

#[cfg(feature = "schemars")]
impl schemars::JsonSchema for DatasetSource {
    fn schema_name() -> std::borrow::Cow<'static, str> {
        "DatasetSource".into()
    }

    fn json_schema(generator: &mut schemars::SchemaGenerator) -> schemars::Schema {
        DatasetSourceSchema::json_schema(generator)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::path::Path;

    fn value(v: &impl Serialize) -> serde_json::Value {
        serde_json::to_value(v).unwrap()
    }

    fn manifest() -> serde_json::Value {
        json!({
            "version": 1,
            "topology": "planar_intrinsics",
            "cameras": [{ "id": "cam0", "images": { "kind": "glob", "pattern": "*.png" } }],
            "target": { "kind": "chessboard", "rows": 6, "cols": 8, "square_size_m": 0.03 },
        })
    }

    #[test]
    fn a_referenced_manifest_is_a_path_and_an_inline_one_a_table() {
        let file: DatasetSource =
            serde_json::from_value(json!({ "manifest": "captures/dataset.toml" })).unwrap();
        assert!(
            matches!(&file, DatasetSource::File { path } if path == Path::new("captures/dataset.toml"))
        );
        let inline: DatasetSource =
            serde_json::from_value(json!({ "root": "images", "manifest": manifest() })).unwrap();
        let DatasetSource::Inline { root, spec } = &inline else {
            panic!("a table is an inline manifest");
        };
        assert_eq!(root, &PathBuf::from("images"));
        assert_eq!(spec.cameras[0].id, "cam0");
        // `root` defaults to the project folder, and is not written then.
        let here: DatasetSource =
            serde_json::from_value(json!({ "manifest": manifest() })).unwrap();
        assert!(matches!(&here, DatasetSource::Inline { root, .. } if root == Path::new(".")));
        assert!(serde_json::to_value(&here).unwrap().get("root").is_none());
        for source in [file, inline, here] {
            let back: DatasetSource = serde_json::from_value(value(&source)).unwrap();
            assert_eq!(value(&back), value(&source));
        }
    }

    #[test]
    fn a_typo_inside_an_inline_manifest_names_the_field() {
        let mut bad = manifest();
        bad["topolgy"] = bad["topology"].take();
        let err = serde_json::from_value::<DatasetSource>(json!({ "manifest": bad }))
            .unwrap_err()
            .to_string();
        assert!(err.contains("topolgy"), "{err}");
    }

    #[test]
    fn root_beside_a_manifest_file_is_refused() {
        let err = serde_json::from_value::<DatasetSource>(
            json!({ "root": "images", "manifest": "dataset.toml" }),
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("root"), "{err}");
        assert!(serde_json::from_value::<DatasetSource>(json!({ "manifest": 3 })).is_err());
        assert!(
            serde_json::from_value::<DatasetSource>(json!({ "manifest": "a.toml", "extra": 1 }))
                .is_err()
        );
    }

    fn full_spec() -> ProjectSpec {
        let mut spec = ProjectSpec::new(DatasetSource::inline(
            "images",
            serde_json::from_value(manifest()).unwrap(),
        ));
        spec.name = Some("Bench".into());
        spec.description = Some("the bench camera".into());
        let mut device = DeviceIdentity::new("SN-0042");
        device.product = Some("line scanner".into());
        device.label = Some("left".into());
        spec.device = Some(device);
        spec.seed = Some(SeedSource::Run("../SN-0041/runs/run-002".into()));
        spec.config = ConfigPatch::from_toml_str("solver.max_iters = 80").unwrap();
        spec.quality = Some(QualityThresholds::default());
        spec.exclude
            .images
            .push(super::super::ImageExclusion::new("cam0", "cam0/0003.png"));
        spec
    }

    #[test]
    fn a_project_spec_round_trips_through_toml_and_json() {
        for spec in [full_spec(), ProjectSpec::for_manifest("dataset.toml")] {
            let toml_text = toml::to_string_pretty(&spec).unwrap();
            let from_toml: ProjectSpec = toml::from_str(&toml_text).unwrap();
            assert_eq!(value(&from_toml), value(&spec), "{toml_text}");
            let json_text = serde_json::to_string_pretty(&spec).unwrap();
            let from_json: ProjectSpec = serde_json::from_str(&json_text).unwrap();
            assert_eq!(value(&from_json), value(&spec));
        }
        // Empty sections are left out.
        assert_eq!(
            value(&ProjectSpec::for_manifest("dataset.toml")),
            json!({ "version": 2, "dataset": { "manifest": "dataset.toml" } })
        );
    }

    #[test]
    fn a_misspelt_section_is_an_error() {
        let text = "version = 2\n[dataset]\nmanifest = \"d.toml\"\n[exlude]\nimages = []\n";
        let err = toml::from_str::<ProjectSpec>(text).unwrap_err().to_string();
        assert!(err.contains("exlude"), "{err}");
        let text =
            "version = 2\n[dataset]\nmanifest = \"d.toml\"\n[device]\nid = \"a\"\ntype = \"x\"\n";
        assert!(toml::from_str::<ProjectSpec>(text).is_err());
    }
}
