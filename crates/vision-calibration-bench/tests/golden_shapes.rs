//! Golden wire-format fixtures.
//!
//! `tests/golden/` holds, per problem type, a serialized export, a serialized
//! session after `run_calibration` and the default config, plus one dataset
//! manifest. Each test re-serializes the current value and compares its
//! *shape* with the fixture's: the set of key paths and the value types, not
//! the numbers. A difference means a wire format changed, and the matching
//! version has to change with it.
//!
//! Regenerate the fixtures after the bump with
//! `GOLDEN_BLESS=1 cargo test -p vision-calibration-bench --test golden_shapes`.
//! Fixtures keep the first four elements of every array: the shape of an
//! array is the union of its elements' shapes, so nothing is lost and the
//! files stay small.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::Value;
use vision_calibration::Error;
use vision_calibration::dataset::DatasetSpec;
use vision_calibration::laserline_device::{
    LaserlineDeviceProblem, run_calibration as run_laserline_device,
};
use vision_calibration::planar_intrinsics::{
    PlanarIntrinsicsProblem, run_calibration as run_planar_intrinsics,
};
use vision_calibration::rig_extrinsics::{
    RigExtrinsicsProblem, run_calibration as run_rig_extrinsics,
};
use vision_calibration::rig_handeye::{RigHandeyeProblem, run_calibration as run_rig_handeye};
use vision_calibration::rig_handeye_laserline::{
    RigHandeyeLaserlineProblem, run_calibration as run_rig_handeye_laserline,
};
use vision_calibration::rig_laserline_device::{
    RigLaserlineDeviceProblem, run_calibration as run_rig_laserline_device,
};
use vision_calibration::session::{CalibrationSession, ProblemType};
use vision_calibration::single_cam_handeye::{
    SingleCamHandeyeProblem, run_calibration as run_single_cam_handeye,
};
use vision_calibration_bench::solver::scenes::{
    CameraKind, Preset, Problem, Scene, SceneData, scene_specs,
};
use vision_calibration_dataset::DATASET_SPEC_VERSION;

/// Arrays in a fixture keep their first this many elements.
const KEPT: usize = 4;

fn golden_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/golden")
}

fn blessing() -> bool {
    std::env::var_os("GOLDEN_BLESS").is_some()
}

/// The key paths of `value` (`a.b`, arrays as `a[]`) with the JSON types seen
/// at each. A `null` is dropped where another type is seen at the path: an
/// optional field is `null` in one value and a number in the next.
fn shape(value: &Value) -> BTreeMap<String, BTreeSet<&'static str>> {
    fn walk(value: &Value, path: &str, out: &mut BTreeMap<String, BTreeSet<&'static str>>) {
        let ty = match value {
            Value::Null => "null",
            Value::Bool(_) => "bool",
            Value::Number(_) => "number",
            Value::String(_) => "string",
            Value::Array(_) => "array",
            Value::Object(_) => "object",
        };
        out.entry(path.to_string()).or_default().insert(ty);
        match value {
            Value::Array(items) => {
                for item in items {
                    walk(item, &format!("{path}[]"), out);
                }
            }
            Value::Object(map) => {
                for (key, item) in map {
                    let child = if path.is_empty() {
                        key.clone()
                    } else {
                        format!("{path}.{key}")
                    };
                    walk(item, &child, out);
                }
            }
            _ => {}
        }
    }
    let mut out = BTreeMap::new();
    walk(value, "", &mut out);
    for types in out.values_mut() {
        if types.len() > 1 {
            types.remove("null");
        }
    }
    out
}

/// `value` with every array cut to at most [`KEPT`] elements.
fn truncated(value: &Value) -> Value {
    match value {
        Value::Array(items) => Value::Array(items.iter().take(KEPT).map(truncated).collect()),
        Value::Object(map) => Value::Object(
            map.iter()
                .map(|(key, item)| (key.clone(), truncated(item)))
                .collect(),
        ),
        other => other.clone(),
    }
}

/// What to bump when a fixture's shape changed.
#[derive(Clone, Copy)]
enum Bump {
    Export(&'static str),
    Session(&'static str),
    Manifest,
}

impl Bump {
    fn hint(self) -> String {
        match self {
            Self::Export(ty) => {
                format!("bump `{ty}::SCHEMA_VERSION` (the export's `schema_version`)")
            }
            Self::Session(ty) => format!(
                "bump `ProblemType::schema_version()` of `{ty}` (the session and its config are \
                 versioned together)"
            ),
            Self::Manifest => "bump `DATASET_SPEC_VERSION` (the manifest's `version`)".to_string(),
        }
    }
}

/// Check `value` against the fixture `name`, or write the fixture when
/// `GOLDEN_BLESS` is set.
fn check(name: &str, value: &Value, bump: Bump) -> Value {
    let path = golden_dir().join(name);
    let fresh = truncated(value);
    if blessing() {
        std::fs::create_dir_all(golden_dir()).unwrap();
        let mut text = serde_json::to_string_pretty(&fresh).unwrap();
        text.push('\n');
        std::fs::write(&path, text).unwrap();
        return fresh;
    }
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "golden fixture {} is missing ({e}); generate it with \
             `GOLDEN_BLESS=1 cargo test -p vision-calibration-bench --test golden_shapes`",
            path.display()
        )
    });
    let golden: Value = serde_json::from_str(&text).unwrap();
    let (want, got) = (shape(&golden), shape(&fresh));
    if want != got {
        let mut diff = Vec::new();
        for (key, types) in &want {
            match got.get(key) {
                None => diff.push(format!("  removed: `{key}`")),
                Some(now) if now != types => {
                    diff.push(format!("  changed: `{key}` {types:?} -> {now:?}"))
                }
                _ => {}
            }
        }
        for key in got.keys().filter(|k| !want.contains_key(*k)) {
            diff.push(format!("  added: `{key}`"));
        }
        panic!(
            "the wire shape of {name} changed:\n{}\n\n\
             A serialized format changed, so its version has to change: {}.\n\
             Then regenerate the fixtures with \
             `GOLDEN_BLESS=1 cargo test -p vision-calibration-bench --test golden_shapes` \
             and commit them with the bump.",
            diff.join("\n"),
            bump.hint()
        );
    }
    golden
}

/// The fixture deserializes into `T` and serializes back to the same shape.
fn check_reads_back<T: DeserializeOwned + Serialize>(name: &str, golden: &Value) {
    let typed: T = serde_json::from_value(golden.clone())
        .unwrap_or_else(|e| panic!("golden fixture {name} no longer deserializes: {e}"));
    let back = serde_json::to_value(&typed).unwrap();
    assert_eq!(
        shape(&back),
        shape(golden),
        "golden fixture {name} changes shape through a read and a write"
    );
}

/// The quick preset's pinhole scene of `problem`, as its input and config.
fn scene(problem: Problem) -> SceneData {
    let spec = scene_specs(Preset::Quick)
        .into_iter()
        .find(|s| s.problem == problem && s.camera == CameraKind::Pinhole)
        .expect("the quick preset has a pinhole scene of every problem");
    Scene::build(&spec).expect("scene builds").data
}

/// Run one problem on its scene and check its session, export and default
/// config against the fixtures.
fn golden<P: ProblemType>(
    name: &str,
    ty: &'static str,
    config: P::Config,
    input: P::Input,
    run: fn(&mut CalibrationSession<P>) -> Result<(), Error>,
) {
    let mut session = CalibrationSession::<P>::new();
    session.set_config(config).unwrap();
    session.set_input(input).unwrap();
    run(&mut session).unwrap_or_else(|e| panic!("{name}: calibration failed: {e}"));
    let export = serde_json::to_value(session.export().unwrap()).unwrap();
    let session_json: Value = serde_json::from_str(&session.to_json().unwrap()).unwrap();
    let config = serde_json::to_value(P::Config::default()).unwrap();

    check(&format!("{name}.export.json"), &export, Bump::Export(ty));
    check(
        &format!("{name}.session.json"),
        &session_json,
        Bump::Session(ty),
    );
    let golden_config = check(&format!("{name}.config.json"), &config, Bump::Session(ty));
    check_reads_back::<P::Config>(&format!("{name}.config.json"), &golden_config);
}

#[test]
fn planar_intrinsics() {
    let SceneData::Planar { input, config } = scene(Problem::PlanarIntrinsics) else {
        panic!("planar scene");
    };
    golden::<PlanarIntrinsicsProblem>(
        "planar_intrinsics",
        "PlanarIntrinsicsExport",
        config,
        input,
        run_planar_intrinsics,
    );
}

#[test]
fn single_cam_handeye() {
    let SceneData::SingleCamHandeye { input, config } = scene(Problem::SingleCamHandeye) else {
        panic!("single-camera hand-eye scene");
    };
    golden::<SingleCamHandeyeProblem>(
        "single_cam_handeye",
        "SingleCamHandeyeExport",
        config,
        input,
        run_single_cam_handeye,
    );
}

#[test]
fn laserline_device() {
    let SceneData::Laserline { input, config } = scene(Problem::LaserlineDevice) else {
        panic!("laserline scene");
    };
    golden::<LaserlineDeviceProblem>(
        "laserline_device",
        "LaserlineDeviceExport",
        config,
        input,
        run_laserline_device,
    );
}

#[test]
fn rig_extrinsics() {
    let SceneData::Rig { input, config, .. } = scene(Problem::RigExtrinsics) else {
        panic!("rig scene");
    };
    golden::<RigExtrinsicsProblem>(
        "rig_extrinsics",
        "RigExtrinsicsExport",
        config,
        input,
        run_rig_extrinsics,
    );
}

#[test]
fn rig_handeye() {
    let SceneData::RigHandeye { input, config } = scene(Problem::RigHandeye) else {
        panic!("rig hand-eye scene");
    };
    golden::<RigHandeyeProblem>(
        "rig_handeye",
        "RigHandeyeExport",
        config,
        input,
        run_rig_handeye,
    );
}

#[test]
fn rig_laserline_device() {
    let SceneData::RigLaserline { input, config } = scene(Problem::RigLaserlineDevice) else {
        panic!("rig laserline scene");
    };
    golden::<RigLaserlineDeviceProblem>(
        "rig_laserline_device",
        "RigLaserlineDeviceExport",
        config,
        input,
        run_rig_laserline_device,
    );
}

#[test]
fn rig_handeye_laserline() {
    let SceneData::RigHandeyeLaserline { input, config } = scene(Problem::RigHandeyeLaserline)
    else {
        panic!("rig hand-eye laserline scene");
    };
    golden::<RigHandeyeLaserlineProblem>(
        "rig_handeye_laserline",
        "RigHandeyeLaserlineExport",
        config,
        input,
        run_rig_handeye_laserline,
    );
}

#[test]
fn dataset_manifest() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../vision-calibration-dataset/samples/kuka_1/dataset.toml");
    let spec = DatasetSpec::from_path(&path).unwrap();
    assert_eq!(spec.version, DATASET_SPEC_VERSION);
    let value = serde_json::to_value(&spec).unwrap();
    let golden = check("dataset_manifest.json", &value, Bump::Manifest);
    check_reads_back::<DatasetSpec>("dataset_manifest.json", &golden);
}
