//! Golden wire-format fixtures.
//!
//! `tests/golden/` holds, per problem type, a serialized export, a serialized
//! session after `run_calibration` and the default config, plus one dataset
//! manifest. Each test re-serializes the current value and compares its
//! *shape* with the fixture's: the set of key paths and the value types, not
//! the numbers. A difference means a wire format changed: an added optional
//! field keeps the version, any other change bumps the matching version.
//!
//! Regenerate the fixtures (after the bump, where one is needed) with
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
    Project,
    RunRecord,
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
            Self::Project => "bump `PROJECT_VERSION` (the project file's `version`)".to_string(),
            Self::RunRecord => "bump `RUN_RECORD_VERSION` (the run record's `version`)".to_string(),
        }
    }
}

/// Check `value` against the fixture `name`, or write the fixture when
/// `GOLDEN_BLESS` is set.
fn check(name: &str, value: &Value, bump: Bump) -> Value {
    let path = golden_dir().join(name);
    // Compare what blessing writes: the truncated value.
    let fresh = truncated(value);
    let got = shape(&fresh);
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
    let want = shape(&golden);
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
             A serialized format changed. An added optional field (serde default, \
             skipped when absent) keeps the version; a removed, renamed or retyped \
             key needs a version bump first: {}.\n\
             Then regenerate the fixtures with \
             `GOLDEN_BLESS=1 cargo test -p vision-calibration-bench --test golden_shapes` \
             and commit them with the change.",
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
    let export_value = serde_json::to_value(session.export().unwrap()).unwrap();
    let session_str = session.to_json().unwrap();
    let session_json: Value = serde_json::from_str(&session_str).unwrap();
    let config = serde_json::to_value(P::Config::default()).unwrap();

    check(
        &format!("{name}.export.json"),
        &export_value,
        Bump::Export(ty),
    );
    let _: P::Export = serde_json::from_value(export_value)
        .unwrap_or_else(|e| panic!("{name}.export.json round-trip deserialize failed: {e}"));
    check(
        &format!("{name}.session.json"),
        &session_json,
        Bump::Session(ty),
    );
    let _ = CalibrationSession::<P>::from_json(&session_str)
        .unwrap_or_else(|e| panic!("{name}.session.json round-trip deserialize failed: {e}"));
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

/// A project file with every optional section, in both dataset forms: the
/// manifest written in it, and a manifest file it references.
#[test]
fn project_spec() {
    use vision_calibration::analysis::QualityThresholds;
    use vision_calibration_pipeline::dataset_runner::ConfigPatch;
    use vision_calibration_pipeline::project::{
        DatasetSource, DeviceIdentity, FeatureExclusion, ImageExclusion, PROJECT_VERSION,
        ProjectSpec, ViewExclusion, ViewRef,
    };

    let manifest = DatasetSpec::from_path(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../vision-calibration-dataset/samples/kuka_1/dataset.toml"),
    )
    .unwrap();
    let mut inline = ProjectSpec::new(DatasetSource::inline("images", manifest));
    assert_eq!(inline.version, PROJECT_VERSION);
    inline.name = Some("kuka_1".into());
    inline.description = Some("hand-eye of the KUKA sample".into());
    let mut device = DeviceIdentity::new("SN-0042");
    device.product = Some("bench camera".into());
    device.label = Some("left".into());
    inline.device = Some(device);
    inline.config = ConfigPatch::from_toml_str("solver.max_iters = 80").unwrap();
    inline.quality = Some(QualityThresholds::default());
    let mut view = ViewExclusion::new(ViewRef::Token("12".into()));
    view.note = Some("robot pose glitch".into());
    inline.exclude.views.push(view);
    inline
        .exclude
        .views
        .push(ViewExclusion::new(ViewRef::Image("13.png".into())));
    let mut image = ImageExclusion::new("cam0", "03.png");
    image.note = Some("blurred".into());
    inline.exclude.images.push(image);
    let mut features = FeatureExclusion::new("cam0", "05.png", vec![[0.02, 0.04, 0.0]]);
    features.note = Some("glare".into());
    inline.exclude.features.push(features);
    let referenced = ProjectSpec::for_manifest("../captures/dataset.toml");

    let value = serde_json::to_value([&inline, &referenced]).unwrap();
    let golden = check("project.spec.json", &value, Bump::Project);
    check_reads_back::<Vec<ProjectSpec>>("project.spec.json", &golden);
}

/// A run record, calibrated (with an excluded image, a left-out detection
/// and a feature exclusion that matched nothing) and imported: a project
/// of eight blank images whose detections its cache holds.
#[test]
fn run_record() {
    use vision_calibration::control::RunControl;
    use vision_calibration::core::{BrownConrady5, FxFyCxCy, make_pinhole_camera};
    use vision_calibration::detect::{
        CacheKey, CachedFeatures, ChessboardConfig, DetectionCache, DetectorConfig, Feature,
        FsDetectionCache,
    };
    use vision_calibration::synthetic::{planar, poses};
    use vision_calibration_pipeline::project::{
        DETECTIONS_DIR, DatasetSource, FeatureExclusion, ImageExclusion, Project,
        ProjectRunOptions, ProjectSpec, RUN_RECORD_FILE, RUN_RECORD_VERSION, RunNotes, RunRecord,
    };

    let dir = std::env::temp_dir().join(format!("golden-run-record-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let data = dir.join("project/data");
    std::fs::create_dir_all(&data).unwrap();
    let cache = FsDetectionCache::new(dir.join("project").join(DETECTIONS_DIR));
    let camera = make_pinhole_camera(
        FxFyCxCy {
            fx: 800.0,
            fy: 800.0,
            cx: 320.0,
            cy: 240.0,
        },
        BrownConrady5::default(),
    );
    let board = planar::grid_points(8, 6, 0.03);
    let specs = poses::seeded_board_pose_specs(8, 0.3, 7);
    let board_poses = poses::centered_board_poses(&specs, (0.105, 0.075), 0.6);
    let views = planar::project_views_all(&camera, &board, &board_poses).unwrap();
    let detector = DetectorConfig::Chessboard(ChessboardConfig::new(6, 8, 0.03));
    let mut paths = Vec::new();
    for (i, view) in views.iter().enumerate() {
        let name = format!("view{i:02}.png");
        let mut image = image::GrayImage::new(640, 480);
        image.put_pixel(i as u32, 0, image::Luma([255]));
        image.save(data.join(&name)).unwrap();
        let mut features: Vec<Feature> = view
            .points_3d
            .iter()
            .zip(&view.points_2d)
            .map(|(p, q)| Feature::new([q.x, q.y], [p.x, p.y, p.z]))
            .collect();
        if i == 3 {
            features.truncate(3);
        }
        let bytes = std::fs::read(data.join(&name)).unwrap();
        let key = CacheKey::for_detection(&bytes, &detector, None);
        cache.put(&key, &CachedFeatures::new(features)).unwrap();
        paths.push(name);
    }
    let manifest: DatasetSpec = serde_json::from_value(serde_json::json!({
        "version": 1,
        "topology": "planar_intrinsics",
        "cameras": [{ "id": "cam0", "images": { "kind": "list", "paths": paths } }],
        "target": { "kind": "chessboard", "rows": 6, "cols": 8, "square_size_m": 0.03 },
    }))
    .unwrap();
    let mut spec = ProjectSpec::new(DatasetSource::inline("data", manifest));
    spec.exclude
        .images
        .push(ImageExclusion::new("cam0", "view01.png"));
    spec.exclude.features.push(FeatureExclusion::new(
        "cam0",
        "view00.png",
        vec![[0.0, 0.0, 0.0], [5.0, 5.0, 0.0]],
    ));
    let project = Project::create(dir.join("project"), spec).unwrap();
    let calibrated = project
        .calibrate(
            &ProjectRunOptions::new("golden"),
            None,
            &RunControl::default(),
        )
        .unwrap();
    let read = |run_dir: &Path| -> Value {
        serde_json::from_slice(&std::fs::read(run_dir.join(RUN_RECORD_FILE)).unwrap()).unwrap()
    };
    let value = read(&calibrated.dir);
    assert_eq!(value["version"], RUN_RECORD_VERSION);
    let golden = check("run.record.json", &value, Bump::RunRecord);
    check_reads_back::<RunRecord>("run.record.json", &golden);

    std::fs::copy(calibrated.export_path(), dir.join("calibration.json")).unwrap();
    let imported = project
        .import_export(dir.join("calibration.json"), RunNotes::new())
        .unwrap();
    let golden = check(
        "run.record.imported.json",
        &read(&imported.dir),
        Bump::RunRecord,
    );
    check_reads_back::<RunRecord>("run.record.imported.json", &golden);
    std::fs::remove_dir_all(&dir).unwrap();
}
