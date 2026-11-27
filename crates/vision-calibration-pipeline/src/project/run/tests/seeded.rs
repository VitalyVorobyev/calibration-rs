//! Seeded project runs: the project's `[seed]`, a run's own seed, what the
//! record keeps of it, its digest, and the seed of a run.

use super::*;
use crate::analysis::Verdict;
use crate::device_seed::{DeviceSeed, DeviceSeedError, SeedLens};
use crate::project::fs::digest_of;
use crate::project::{DeviceHistory, DeviceIdentity, DeviceRun, RunSeed, SeedSource};

fn seed_of(info: &RunInfo) -> &RunSeed {
    calibrated(info)
        .inputs
        .seed
        .as_ref()
        .expect("a seeded run records its seed")
}

fn json_of(value: &impl serde::Serialize) -> serde_json::Value {
    serde_json::to_value(value).unwrap()
}

/// A project whose `[seed]` names its own first run starts from it, as the
/// dataset runner does with the same seed: the export is the runner's, bit
/// for bit, but for the image root restated for the run folder.
#[test]
fn a_seeded_run_is_the_dataset_runner_s_seeded_run() {
    let tmp = tempfile::tempdir().unwrap();
    let mut project = planar_project(tmp.path());
    let first = calibrate(&project);
    // The record states each camera's image size, so the run seeds alone.
    assert_eq!(
        record(&first).summary.cameras[0].image_size,
        Some([640, 480])
    );
    project.spec_mut().seed = Some(SeedSource::Run("runs/run-001".into()));
    project.save().unwrap();
    let seeded = calibrate(&project);

    let seed = DeviceSeed::from_run(&first.dir).unwrap();
    assert_eq!(seed.cameras[0].id, "cam0");
    assert_eq!(seed.cameras[0].image_size, [640, 480]);
    assert_eq!(
        seed.description.as_deref(),
        Some("the calibrated run run-001")
    );
    let cache = FsDetectionCache::new(tmp.path().join(DETECTIONS_DIR));
    let direct = crate::dataset_runner::calibrate_dataset_with_seed(
        &project.dataset().unwrap(),
        &project.data_root().unwrap(),
        None,
        &seed,
        Detection::Cached(&cache),
        None,
        &RunControl::default(),
    )
    .unwrap();
    let mut direct = json_of(&direct.export);
    direct["diagnostics"]["image_manifest"]["root"] = json!("../../data");
    let written: serde_json::Value = serde_json::from_slice(&bytes(&seeded.export_path())).unwrap();
    assert_eq!(written, direct);

    // The record keeps the source as written, the values and what the
    // workflow took; the digest covers the values.
    let kept = seed_of(&seeded);
    assert_eq!(kept.source, SeedSource::Run("runs/run-001".into()));
    assert_eq!(json_of(&kept.value), json_of(&seed));
    assert_eq!(kept.applied, ["cam0.lens"]);
    assert!(kept.ignored.is_empty());
    let digest = &calibrated(&seeded).digest;
    assert_eq!(digest.seed.as_deref(), Some(digest_of(&seed).as_str()));
    assert_eq!(
        digest.inputs,
        digest_of(&[
            &digest.dataset,
            &digest.exclusions,
            &digest.config,
            &digest.quality,
            &digest.detections,
            digest.seed.as_ref().unwrap(),
        ])
    );
    assert_ne!(digest.inputs, calibrated(&first).digest.inputs);
    assert_eq!(project.input_digest().unwrap().seed, digest.seed);

    // Seeded twice from the same values: the same digest and export.
    let again = calibrate(&project);
    assert_eq!(calibrated(&again).digest, *digest);
    assert_eq!(bytes(&again.export_path()), bytes(&seeded.export_path()));
}

/// An unseeded run records no seed, and its `inputs` digest covers the
/// five parts alone, as before seeds were recorded.
#[test]
fn an_unseeded_run_keeps_its_digest() {
    let tmp = tempfile::tempdir().unwrap();
    let project = planar_project(tmp.path());
    let run = calibrate(&project);
    let digest = &calibrated(&run).digest;
    assert_eq!(digest.seed, None);
    assert_eq!(
        digest.inputs,
        digest_of(&[
            &digest.dataset,
            &digest.exclusions,
            &digest.config,
            &digest.quality,
            &digest.detections,
        ])
    );
    let written: serde_json::Value =
        serde_json::from_slice(&bytes(&run.dir.join(RUN_RECORD_FILE))).unwrap();
    let calibrated_json = &written["source"];
    assert!(calibrated_json["inputs"].get("seed").is_none());
    assert!(calibrated_json["digest"].get("seed").is_none());
    assert_eq!(project.input_digest().unwrap().seed, None);
}

/// A seed for this run only takes the place of the project's, even of one
/// that cannot be made, and the record names it.
#[test]
fn a_run_s_own_seed_replaces_the_project_s() {
    let tmp = tempfile::tempdir().unwrap();
    let mut project = planar_project(tmp.path());
    let first = calibrate(&project);
    let file = tmp.path().join("seed.json");
    let seed = DeviceSeed::from_run(&first.dir).unwrap();
    std::fs::write(&file, serde_json::to_string_pretty(&seed).unwrap()).unwrap();
    project.spec_mut().seed = Some(SeedSource::Run("runs/run-404".into()));

    let mut options = options();
    options.seed = Some(SeedSource::File("seed.json".into()));
    let run = project
        .calibrate(&options, None, &RunControl::default())
        .unwrap();
    let kept = seed_of(&run);
    assert_eq!(kept.source, SeedSource::File("seed.json".into()));
    assert_eq!(json_of(&kept.value), json_of(&seed));

    // A seed file in TOML reads the same.
    let toml_file = tmp.path().join("seed.toml");
    std::fs::write(&toml_file, toml::to_string(&seed).unwrap()).unwrap();
    options.seed = Some(SeedSource::File(toml_file));
    let run = project
        .calibrate(&options, None, &RunControl::default())
        .unwrap();
    assert_eq!(json_of(&seed_of(&run).value), json_of(&seed));
}

/// A seed that cannot be made, or does not fit the dataset or the
/// configuration, fails before any image is read and writes no run.
#[test]
fn a_seed_that_cannot_be_used_fails_before_any_image_is_read() {
    let tmp = tempfile::tempdir().unwrap();
    let mut project = planar_project(tmp.path());
    let first = calibrate(&project);
    let seed = DeviceSeed::from_run(&first.dir).unwrap();
    std::fs::remove_dir_all(tmp.path().join(DETECTIONS_DIR)).unwrap();
    let write = |name: &str, seed: &DeviceSeed| {
        std::fs::write(tmp.path().join(name), serde_json::to_string(seed).unwrap()).unwrap();
    };

    let fails = |project: &Project| {
        let checked = project.check().unwrap_err();
        let err = project
            .calibrate(&options(), None, &RunControl::default())
            .unwrap_err();
        assert_eq!(checked.to_string(), err.to_string());
        assert!(!tmp.path().join(DETECTIONS_DIR).exists());
        assert_eq!(project.runs().unwrap().len(), 1);
        err
    };

    for source in [
        SeedSource::Run("runs/run-404".into()),
        SeedSource::File("missing.json".into()),
        SeedSource::DeviceSpec("missing-spec.json".into()),
        // A folder that holds no run.
        SeedSource::Run("data".into()),
    ] {
        project.spec_mut().seed = Some(source);
        let err = fails(&project);
        assert!(matches!(err, ProjectError::SeedUnresolved { .. }), "{err}");
        assert_eq!(err.field().as_deref(), Some("seed"));
    }

    // Another camera id.
    let mut other = seed.clone();
    other.cameras[0].id = "left".into();
    write("other-ids.json", &other);
    project.spec_mut().seed = Some(SeedSource::File("other-ids.json".into()));
    let err = fails(&project);
    assert!(
        matches!(
            &err,
            ProjectError::Calibrate(CalibrateDatasetError::Seed(
                DeviceSeedError::CameraIdsDiffer { .. }
            ))
        ),
        "{err}"
    );

    // Another camera model than the configuration's.
    project.spec_mut().seed = Some(SeedSource::Run("runs/run-001".into()));
    project.spec_mut().config = ConfigPatch::from_toml_str(
        "camera.model = { projection = \"pinhole\", distortion = \"rational8\", sensor = \"identity\" }",
    )
    .unwrap();
    let err = fails(&project);
    assert!(
        matches!(
            &err,
            ProjectError::Calibrate(CalibrateDatasetError::Seed(
                DeviceSeedError::ModelMismatch { .. }
            ))
        ),
        "{err}"
    );
    project.spec_mut().config = ConfigPatch::default();

    // A seed for images of another size fails once the headers are read,
    // before any detection.
    let mut other = seed;
    other.cameras[0].image_size = [1280, 960];
    write("other-size.json", &other);
    project.spec_mut().seed = Some(SeedSource::File("other-size.json".into()));
    project.check().unwrap();
    let err = project
        .calibrate(&options(), None, &RunControl::default())
        .unwrap_err();
    assert!(
        matches!(
            &err,
            ProjectError::Calibrate(CalibrateDatasetError::Seed(
                DeviceSeedError::ImageSizeMismatch { .. }
            ))
        ),
        "{err}"
    );
    assert_eq!(project.runs().unwrap().len(), 1);
    assert!(
        std::fs::read_dir(tmp.path().join(DETECTIONS_DIR)).map_or(true, |entries| entries
            .filter_map(Result::ok)
            .all(|e| e.file_name() == ".gitignore"))
    );
}

/// The seed of a run: from its folder or its export, an imported run's
/// too, a run recorded without image sizes from its images' headers, and
/// nothing from a folder or a run that does not say what a seed needs.
#[test]
fn the_seed_of_a_run() {
    let tmp = tempfile::tempdir().unwrap();
    let project = planar_project(tmp.path());
    let run = calibrate(&project);
    let from_dir = DeviceSeed::from_run(&run.dir).unwrap();
    let from_export = DeviceSeed::from_run(run.export_path()).unwrap();
    assert_eq!(json_of(&from_dir), json_of(&from_export));
    let SeedLens::Calibrated { camera } = &from_dir.cameras[0].lens else {
        panic!("a run seeds a calibrated lens");
    };
    let export =
        crate::common::calibrated_cameras(&std::fs::read_to_string(run.export_path()).unwrap())
            .unwrap();
    assert_eq!(json_of(camera), json_of(&export.camera_params()[0]));

    let imported = project
        .import_export(run.export_path(), RunNotes::new())
        .unwrap();
    let seed = DeviceSeed::from_run(&imported.dir).unwrap();
    assert_eq!(
        seed.description.as_deref(),
        Some("the imported run run-002")
    );
    assert_eq!(json_of(&seed.cameras), json_of(&from_dir.cameras));

    // A record without image sizes: they are read from the images.
    let path = run.dir.join(RUN_RECORD_FILE);
    let mut written: serde_json::Value = serde_json::from_slice(&bytes(&path)).unwrap();
    written["summary"]["cameras"][0]
        .as_object_mut()
        .unwrap()
        .remove("image_size");
    std::fs::write(&path, serde_json::to_string(&written).unwrap()).unwrap();
    assert_eq!(
        json_of(&DeviceSeed::from_run(&run.dir).unwrap()),
        json_of(&from_dir)
    );
    // ... and without the images either, there is no seed.
    std::fs::rename(tmp.path().join("data"), tmp.path().join("moved")).unwrap();
    let err = DeviceSeed::from_run(&run.dir).unwrap_err();
    assert!(
        matches!(&err, ProjectError::SeedUnresolved { message, .. } if message.contains("image sizes")),
        "{err}"
    );

    // A run of the app's earlier format names no cameras.
    let old = tmp.path().join("runs/old");
    std::fs::create_dir_all(&old).unwrap();
    std::fs::copy(run.export_path(), old.join(EXPORT_FILE)).unwrap();
    let err = DeviceSeed::from_run(&old).unwrap_err();
    assert!(
        matches!(&err, ProjectError::SeedUnresolved { message, .. } if message.contains("no record")),
        "{err}"
    );
    let err = DeviceSeed::from_run(tmp.path()).unwrap_err();
    assert!(matches!(err, ProjectError::SeedUnresolved { .. }), "{err}");
}

/// The run that stands for a device: its latest starred run, else its
/// latest calibrated run whose verdict is pass.
#[test]
fn the_representative_run_of_a_device() {
    let tmp = tempfile::tempdir().unwrap();
    let project = planar_project(tmp.path());
    let a = calibrate(&project);
    let b = calibrate(&project);
    let imported = project
        .import_export(b.export_path(), RunNotes::new())
        .unwrap();
    let history = |runs: Vec<RunInfo>| {
        let mut history = DeviceHistory {
            device: DeviceIdentity::new("SN-1"),
            runs: Vec::new(),
        };
        history.runs = runs
            .into_iter()
            .map(|run| DeviceRun {
                project: tmp.path().to_path_buf(),
                run,
            })
            .collect();
        history
    };
    let id = |history: &DeviceHistory| history.representative_run().map(|r| r.run.id.clone());

    let judged = |run: &RunInfo, verdict: Verdict| {
        let mut run = run.clone();
        let RunState::Recorded(record) = &mut run.state else {
            panic!("a recorded run");
        };
        record.summary.verdict = Some(verdict);
        run
    };
    let (a, b, imported) = (
        judged(&a, Verdict::Pass),
        judged(&b, Verdict::Pass),
        judged(&imported, Verdict::Pass),
    );
    // The imported run is newer but not calibrated in a project, and a
    // newer run that does not pass does not count.
    let mut warned = judged(&b, Verdict::Warn);
    warned.id = "run-009".into();
    let runs = vec![a.clone(), b.clone(), imported.clone(), warned];
    assert_eq!(id(&history(runs.clone())), Some(b.id.clone()));
    // A starred run wins, however old.
    let mut starred = runs;
    starred[0].notes.starred = true;
    assert_eq!(id(&history(starred.clone())), Some(a.id.clone()));
    starred[2].notes.starred = true;
    assert_eq!(id(&history(starred)), Some(imported.id.clone()));
    // Nothing passed and nothing starred: none.
    assert_eq!(id(&history(vec![imported])), None);
    assert_eq!(id(&history(Vec::new())), None);
}
