//! Golden wire-format fixtures.
//!
//! `tests/golden/` holds, per problem type, a serialized export, a serialized
//! session after `run_calibration` and the default config, plus one dataset
//! manifest and one device seed. Each test re-serializes the current value and compares its
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
    DetectionDetail,
    DeviceSeed,
    SeedAggregate,
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
            Self::DetectionDetail => "the detection detail carries no version (a cache entry \
                 is keyed on the release): an added optional field is a minor change, any \
                 other one a major release"
                .to_string(),
            Self::DeviceSeed => "bump `DEVICE_SEED_VERSION` (the seed's `version`)".to_string(),
            Self::SeedAggregate => "the seed aggregate carries no version of its own (its seed \
                 has `DEVICE_SEED_VERSION`): an added optional field is a minor change, any other \
                 one a major release"
                .to_string(),
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
        ProjectSpec, SeedSource, ViewExclusion, ViewRef,
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
    inline.seed = Some(SeedSource::Run("../SN-0041/runs/run-002".into()));
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
/// and a feature exclusion that matched nothing), imported, and calibrated
/// from a seed (the first run): a project of eight blank images whose
/// detections its cache holds.
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
        SeedSource,
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

    let mut options = ProjectRunOptions::new("golden");
    options.seed = Some(SeedSource::Run("runs/run-001".into()));
    let seeded = project
        .calibrate(&options, None, &RunControl::default())
        .unwrap();
    let golden = check(
        "run.record.seeded.json",
        &read(&seeded.dir),
        Bump::RunRecord,
    );
    check_reads_back::<RunRecord>("run.record.seeded.json", &golden);
    std::fs::remove_dir_all(&dir).unwrap();
}

/// A device type's seed from three units: two-camera rigs with a hand-eye,
/// one of them without laser planes, so they are omitted.
#[test]
fn seed_aggregate() {
    use vision_calibration::core::{
        BrownConrady5, FxFyCxCy, Iso3, LaserPlane, make_pinhole_camera, pinhole_camera_params,
    };
    use vision_calibration::device_seed::{DeviceSeed, SeedCamera, SeedHandeye, SeedLens};
    use vision_calibration::nalgebra::{Translation3, UnitQuaternion, Vector3};

    let unit = |fx: f64, planes: bool| {
        let k = FxFyCxCy {
            fx,
            fy: fx,
            cx: 640.0,
            cy: 480.0,
        };
        let distortion = BrownConrady5 {
            k1: -0.2,
            iters: 8,
            ..Default::default()
        };
        let lens = || SeedLens::Calibrated {
            camera: pinhole_camera_params(&make_pinhole_camera(k, distortion)),
        };
        let mut cam0 = SeedCamera::new("cam0", [1280, 960], lens());
        cam0.cam_se3_rig = Some(Iso3::identity());
        let mut cam1 = SeedCamera::new("cam1", [1280, 960], lens());
        cam1.cam_se3_rig = Some(Iso3::from_parts(
            Translation3::new(-0.1, 0.0, 0.0),
            UnitQuaternion::from_euler_angles(0.0, 0.1, 0.0),
        ));
        if planes {
            for camera in [&mut cam0, &mut cam1] {
                camera.laser_plane =
                    Some(LaserPlane::new(Vector3::new(0.0, 0.6, 0.8), 0.3).unwrap());
            }
        }
        let mut seed = DeviceSeed::new(vec![cam0, cam1]);
        seed.handeye = Some(SeedHandeye::EyeInHand {
            gripper_se3_rig: Iso3::from_parts(
                Translation3::new(0.0, 0.05, 0.2),
                UnitQuaternion::from_euler_angles(0.2, 0.0, 0.1),
            ),
        });
        seed
    };
    let aggregate =
        DeviceSeed::aggregate(&[unit(1500.0, true), unit(1510.0, false), unit(1490.0, true)])
            .unwrap();
    let value = serde_json::to_value(&aggregate).unwrap();
    let golden = check("seed.aggregate.json", &value, Bump::SeedAggregate);
    let typed: vision_calibration::device_seed::SeedAggregate =
        serde_json::from_value(golden.clone()).unwrap();
    assert_eq!(
        shape(&serde_json::to_value(&typed).unwrap()),
        shape(&golden)
    );
}

/// A detection detail of every target kind and outcome, with every optional
/// field set: the shape the desktop app draws overlays from.
#[test]
fn detection_detail() {
    use serde_json::json;
    use vision_calibration::detect::DetectionDetail;

    let quad = json!([[0.5, 0.5], [10.5, 0.5], [10.5, 10.5], [0.5, 10.5]]);
    let corner = json!({ "grid": [1, 2], "id": 7, "score": 0.5, "axes_rad": [0.1, 1.7] });
    let quality = json!({
        "noise_sigma": 1.5,
        "median_sigma_px": 0.03,
        "local_residual_count": 12,
        "local_residual_rms_px": 0.04,
    });
    let ellipse = json!({ "center_px": [5.0, 6.0], "semi_axes_px": [4.0, 3.5], "angle_rad": -0.3 });
    let feature = json!({
        "image_xy": [10.25, 20.5],
        "world_xyz": [0.02, 0.0, 0.0],
        "cov_px": [0.01, 0.0, 0.02],
        "blur": {
            "psf_px": [1.0, 0.0, 1.0],
            "cov": [[1e-3, 0.0, 0.0], [0.0, 1e-3, 0.0], [0.0, 0.0, 1e-3]],
        },
    });
    let rejected = json!([
        { "image_xy": [1.0, 2.0], "reason": { "kind": "unlabelled", "strength": 30.0 } },
        {
            "image_xy": [3.0, 4.0],
            "reason": { "kind": "outside_board" },
            "world_xyz": [0.1, 0.2, 0.0],
            "grid": [5, 10],
            "id": 3,
        },
        { "image_xy": [5.0, 6.0], "reason": { "kind": "unanchored", "confidence": 0.4 } },
        { "image_xy": [7.0, 8.0], "reason": { "kind": "uncorrected_center", "confidence": 0.2 } },
    ]);
    let targets = [
        json!({ "kind": "chessboard", "corners": [corner], "cell_size_px": 20.0, "quality": quality }),
        json!({
            "kind": "charuco",
            "corners": [corner],
            "quality": quality,
            "markers": [{
                "id": 4, "cell": [1, 2], "quad_px": quad, "rotation": 1, "hamming": 0, "score": 0.99,
            }],
            "rejected_markers": [{
                "quad_px": quad, "decoded_id": 9, "expected_id": 5, "score": -12.5,
            }],
        }),
        json!({
            "kind": "puzzleboard",
            "corners": [corner],
            "quality": quality,
            "edge_bits": [{
                "endpoints_px": [[1.0, 2.0], [3.0, 4.0]],
                "bit": 1,
                "confidence": 0.9,
                "matches_master": false,
            }],
            "decode": {
                "edges_observed": 40,
                "edges_matched": 39,
                "bit_error_rate": 0.025,
                "logical_bit_error_rate": 0.0,
                "dot_dissent_rate": 0.01,
                "master_origin": [100, 200],
            },
        }),
        json!({
            "kind": "ringgrid",
            "anchored": true,
            "board_complete": false,
            "origin_dots_px": [[1.0, 1.0]],
            "rings": [{
                "id": 3,
                "grid": [0, 0],
                "confidence": 0.8,
                "outer": ellipse,
                "inner": ellipse,
                "center": "photometric",
                "center_cov_px": [1e-4, 0.0, 2e-4],
            }],
        }),
    ];
    let mut details: Vec<Value> = targets
        .iter()
        .map(|target| {
            json!({
                "features": [feature],
                "outcome": { "status": "detected" },
                "target": target,
                "rejected": rejected,
            })
        })
        .collect();
    for outcome in [
        json!({ "status": "not_found", "reason": "no board" }),
        json!({ "status": "ambiguous", "repeated_pixels": 2, "repeated_points": 0 }),
        json!({ "status": "filtered" }),
    ] {
        details.push(json!({
            "features": [],
            "outcome": outcome,
            "target": { "kind": "chessboard", "corners": [], "quality": { "local_residual_count": 0 } },
            "rejected": rejected,
        }));
    }
    // Through the Rust type: the fixture holds what it writes.
    let typed: Vec<DetectionDetail> = serde_json::from_value(Value::Array(details)).unwrap();
    let value = serde_json::to_value(&typed).unwrap();
    let golden = check("detection.detail.json", &value, Bump::DetectionDetail);
    check_reads_back::<Vec<DetectionDetail>>("detection.detail.json", &golden);
}

/// A device seed with both lens kinds, rig mounts, a laser plane and a
/// hand-eye.
#[test]
fn device_seed() {
    use vision_calibration::core::{
        BrownConrady5, FxFyCxCy, Iso3, LaserPlane, ScheimpflugParams, SensorParams,
        make_pinhole_camera, pinhole_camera_params,
    };
    use vision_calibration::device_seed::{DeviceSeed, SeedCamera, SeedHandeye, SeedLens};
    use vision_calibration::nalgebra::{Translation3, UnitQuaternion, Vector3};

    let k = FxFyCxCy {
        fx: 1500.0,
        fy: 1500.5,
        cx: 640.0,
        cy: 480.0,
    };
    let distortion = BrownConrady5 {
        k1: -0.3,
        ..Default::default()
    };
    let camera = pinhole_camera_params(&make_pinhole_camera(k, distortion)).with_sensor(
        SensorParams::Scheimpflug {
            params: ScheimpflugParams {
                tilt_x: 0.05,
                tilt_y: 0.0,
            },
        },
    );
    let pose = |t: f64| {
        Iso3::from_parts(
            Translation3::new(t, 0.0, 0.0),
            UnitQuaternion::from_euler_angles(0.0, 0.1, 0.0),
        )
    };
    let mut cam0 = SeedCamera::new("cam0", [1280, 960], SeedLens::Calibrated { camera });
    cam0.cam_se3_rig = Some(Iso3::identity());
    cam0.laser_plane = Some(LaserPlane::new(Vector3::new(0.0, 0.6, 0.8), -0.3).unwrap());
    let mut cam1 = SeedCamera::new(
        "cam1",
        [1280, 960],
        SeedLens::Nominal {
            intrinsics: k,
            sensor: ScheimpflugParams::default(),
        },
    );
    cam1.cam_se3_rig = Some(pose(-0.1));
    let mut seed = DeviceSeed::new(vec![cam0, cam1]);
    seed.handeye = Some(SeedHandeye::EyeInHand {
        gripper_se3_rig: pose(0.05),
    });
    seed.description = Some("a synthetic two-camera device".to_string());
    seed.validate().unwrap();

    let value = serde_json::to_value(&seed).unwrap();
    let golden = check("device_seed.json", &value, Bump::DeviceSeed);
    check_reads_back::<DeviceSeed>("device_seed.json", &golden);
}
