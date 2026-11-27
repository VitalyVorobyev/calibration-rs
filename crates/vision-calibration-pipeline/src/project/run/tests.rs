//! Tests of a project's runs on a synthetic planar dataset: blank PNGs
//! whose detections the project's cache holds, the board's corners
//! projected by an ideal camera.

use super::*;
use crate::dataset_runner::RejectReason;
use crate::project::{
    DETECTIONS_DIR, DatasetSource, ExclusionFault, Exclusions, FeatureExclusion, ImageExclusion,
    ProjectSpec,
};
use serde_json::json;
use vision_calibration_core::{CancelToken, Pt3};
use vision_calibration_detect::{
    CacheKey, CachedFeatures, ChessboardConfig, DetectionCache, DetectorConfig, Feature,
};

/// Eight views of an 8 × 6 board under `data/img` of the folder `data`;
/// each image a distinct blank PNG whose detection `cache` holds. Image 3's
/// detection holds three corners, so a run leaves it out. Returns the
/// manifest.
fn planar_dataset(data: &Path, cache: &dyn DetectionCache) -> serde_json::Value {
    use vision_calibration_core::synthetic::{planar, poses};
    use vision_calibration_core::{BrownConrady5, FxFyCxCy, make_pinhole_camera};

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
    std::fs::create_dir_all(data.join("img")).unwrap();
    let mut paths = Vec::new();
    for (i, view) in views.iter().enumerate() {
        let rel = format!("img/view{i:02}.png");
        write_image(data, &rel, i as u32);
        let mut features: Vec<Feature> = view
            .points_3d
            .iter()
            .zip(&view.points_2d)
            .map(|(p, q): (&Pt3, _)| Feature::new([q.x, q.y], [p.x, p.y, p.z]))
            .collect();
        if i == 3 {
            features.truncate(3);
        }
        store(data, &rel, cache, features);
        paths.push(rel);
    }
    json!({
        "version": 1,
        "topology": "planar_intrinsics",
        "cameras": [{ "id": "cam0", "images": { "kind": "list", "paths": paths } }],
        "target": { "kind": "chessboard", "rows": 6, "cols": 8, "square_size_m": 0.03 },
    })
}

/// A blank 640 × 480 PNG at `data/rel`, told apart by one pixel.
fn write_image(data: &Path, rel: &str, marker: u32) {
    let mut image = image::GrayImage::new(640, 480);
    image.put_pixel(marker % 640, marker / 640, image::Luma([255]));
    image.save(data.join(rel)).unwrap();
}

/// Put `features` in `cache` as the detection of the image `data/rel`.
fn store(data: &Path, rel: &str, cache: &dyn DetectionCache, features: Vec<Feature>) {
    let bytes = std::fs::read(data.join(rel)).unwrap();
    let config = DetectorConfig::Chessboard(ChessboardConfig::new(6, 8, 0.03));
    let key = CacheKey::for_detection(&bytes, &config, None);
    cache.put(&key, &CachedFeatures::new(features)).unwrap();
}

/// A project in `dir` whose dataset, written in the project file, lies in
/// `dir/data`; its cache holds every detection.
fn planar_project(dir: &Path) -> Project {
    let cache = FsDetectionCache::new(dir.join(DETECTIONS_DIR));
    let manifest = planar_dataset(&dir.join("data"), &cache);
    let spec = ProjectSpec::new(DatasetSource::inline(
        "data",
        serde_json::from_value(manifest).unwrap(),
    ));
    Project::create(dir, spec).unwrap()
}

fn options() -> ProjectRunOptions {
    ProjectRunOptions::new("test")
}

fn calibrate(project: &Project) -> RunInfo {
    project
        .calibrate(&options(), None, &RunControl::default())
        .unwrap()
}

fn record(info: &RunInfo) -> &RunRecord {
    info.record().expect("a calibrated run has its record")
}

fn calibrated(info: &RunInfo) -> &CalibratedRun {
    match &record(info).source {
        RunSource::Calibrated(run) => run,
        other => panic!("expected a calibrated run, got {other:?}"),
    }
}

fn bytes(path: &Path) -> Vec<u8> {
    std::fs::read(path).unwrap()
}

#[test]
fn a_run_writes_its_files_and_is_listed() {
    let tmp = tempfile::tempdir().unwrap();
    let project = planar_project(tmp.path());
    let first = calibrate(&project);
    assert_eq!(first.id, "run-001");
    for file in [EXPORT_FILE, QUALITY_FILE, RUN_RECORD_FILE] {
        assert!(first.dir.join(file).is_file(), "{file}");
    }
    assert!(!first.dir.join(REPORT_FILE).exists());
    assert!(!first.dir.join(RUN_NOTES_FILE).exists());
    // The cache folder ignores itself.
    assert_eq!(
        std::fs::read_to_string(tmp.path().join(DETECTIONS_DIR).join(".gitignore")).unwrap(),
        "*\n"
    );

    let run = calibrated(&first);
    assert_eq!((run.usable_views, run.total_views), (7, 8));
    assert_eq!(
        run.left_out,
        [LeftOutImage {
            camera: "cam0".into(),
            image: "img/view03.png".into(),
            reason: RejectReason::TooFewFeatures,
        }]
    );
    let summary = &record(&first).summary;
    assert_eq!(summary.kind, ProblemKind::PlanarIntrinsics);
    assert!(summary.verdict.is_some());
    assert_eq!(summary.cameras.len(), 1);
    assert_eq!(summary.cameras[0].id, "cam0");
    assert_eq!(summary.cameras[0].views, 7);
    assert!(summary.cameras[0].rms_px.unwrap() < 1e-6);
    assert_eq!(record(&first).tool, "test");
    assert_eq!(run.inputs.data_root, Path::new("data"));

    // The export's images resolve from the run folder.
    let export: serde_json::Value = serde_json::from_slice(&bytes(&first.export_path())).unwrap();
    let root = export["diagnostics"]["image_manifest"]["root"]
        .as_str()
        .unwrap();
    assert_eq!(root, "../../data");
    let frame = export["diagnostics"]["image_manifest"]["frames"][0]["path"]
        .as_str()
        .unwrap();
    assert!(first.dir.join(root).join(frame).is_file());

    let second = calibrate(&project);
    assert_eq!(second.id, "run-002");
    let ids: Vec<String> = project.runs().unwrap().into_iter().map(|r| r.id).collect();
    assert_eq!(ids, ["run-001", "run-002"]);
    let listed = project.run("run-001").unwrap();
    assert_eq!(
        serde_json::to_value(listed.record()).unwrap(),
        serde_json::to_value(first.record()).unwrap()
    );
}

/// Equal inputs and the same library version give byte-identical exports
/// and quality reports; the record's clock readings are the only
/// difference between the runs.
#[test]
fn equal_input_digests_give_identical_exports_and_reports() {
    let tmp = tempfile::tempdir().unwrap();
    let project = planar_project(tmp.path());
    let a = calibrate(&project);
    let b = calibrate(&project);
    assert_eq!(calibrated(&a).digest, calibrated(&b).digest);
    assert_eq!(bytes(&a.export_path()), bytes(&b.export_path()));
    assert_eq!(bytes(&a.quality_path()), bytes(&b.quality_path()));
    assert!(calibrated(&a).digest.inputs.starts_with("sha256:"));
    assert_ne!(record(&a).timestamp, 0);
}

/// The project's own digest is the run's, part by part, before any image is
/// read; an edit changes the part it touches, and a run's own configuration
/// changes make its config digest differ from the project's.
#[test]
fn the_project_digest_is_the_run_digest_without_the_detections() {
    let tmp = tempfile::tempdir().unwrap();
    let mut project = planar_project(tmp.path());
    let run = calibrate(&project);
    let recorded = &calibrated(&run).digest;
    let current = project.input_digest().unwrap();
    assert_eq!(current.dataset, recorded.dataset);
    assert_eq!(current.exclusions, recorded.exclusions);
    assert_eq!(current.config, recorded.config);
    assert_eq!(current.quality, recorded.quality);

    project
        .spec_mut()
        .exclude
        .images
        .push(ImageExclusion::new("cam0", "img/view00.png"));
    let edited = project.input_digest().unwrap();
    assert_ne!(edited.exclusions, current.exclusions);
    assert_eq!(
        (&edited.dataset, &edited.config, &edited.quality),
        (&current.dataset, &current.config, &current.quality)
    );

    let mut options = options();
    options.config_patch = ConfigPatch::from_json_str(r#"{"solver": {"max_iters": 7}}"#).unwrap();
    let patched = project
        .calibrate(&options, None, &RunControl::default())
        .unwrap();
    assert_ne!(calibrated(&patched).digest.config, edited.config);
}

/// The recording cache holds one key per image the run reads: every image
/// of the dataset but the excluded one.
#[test]
fn the_detections_digest_covers_every_image_read() {
    let tmp = tempfile::tempdir().unwrap();
    let mut project = planar_project(tmp.path());
    let prepared = project.prepare(&ConfigPatch::default()).unwrap();
    let fs_cache = FsDetectionCache::new(tmp.path().join(DETECTIONS_DIR));
    let recording = RecordingCache::new(&fs_cache);
    calibrate_dataset_with(
        &prepared.dataset,
        &prepared.data_root,
        None,
        Detection::Cached(&recording),
        None,
        &RunControl::default(),
        &prepared.options,
    )
    .unwrap();
    assert_eq!(recording.keys().len(), 8);

    project
        .spec_mut()
        .exclude
        .images
        .push(ImageExclusion::new("cam0", "img/view02.png"));
    let prepared = project.prepare(&ConfigPatch::default()).unwrap();
    let recording = RecordingCache::new(&fs_cache);
    calibrate_dataset_with(
        &prepared.dataset,
        &prepared.data_root,
        None,
        Detection::Cached(&recording),
        None,
        &RunControl::default(),
        &prepared.options,
    )
    .unwrap();
    assert_eq!(recording.keys().len(), 7);
}

/// The recording cache sees one key per image read, and a changed image
/// changes the detections digest only.
#[test]
fn a_changed_image_changes_only_the_detections_digest() {
    let tmp = tempfile::tempdir().unwrap();
    let project = planar_project(tmp.path());
    let before = calibrate(&project);

    let data = tmp.path().join("data");
    let cache = FsDetectionCache::new(tmp.path().join(DETECTIONS_DIR));
    // The same detection, from an image whose bytes differ.
    let old_bytes = bytes(&data.join("img/view05.png"));
    let config = DetectorConfig::Chessboard(ChessboardConfig::new(6, 8, 0.03));
    let features = cache
        .get(&CacheKey::for_detection(&old_bytes, &config, None))
        .unwrap()
        .unwrap()
        .features;
    write_image(&data, "img/view05.png", 1000);
    store(&data, "img/view05.png", &cache, features);
    let after = calibrate(&project);

    let (b, a) = (&calibrated(&before).digest, &calibrated(&after).digest);
    assert_ne!(b.detections, a.detections);
    assert_ne!(b.inputs, a.inputs);
    assert_eq!(
        (&b.dataset, &b.exclusions, &b.config, &b.quality),
        (&a.dataset, &a.exclusions, &a.config, &a.quality)
    );
}

#[test]
fn a_moved_project_keeps_its_digests_and_still_resolves_its_images() {
    let tmp = tempfile::tempdir().unwrap();
    let first = tmp.path().join("first");
    let project = planar_project(&first);
    let run = calibrate(&project);
    let moved = tmp.path().join("elsewhere").join("moved");
    std::fs::create_dir_all(moved.parent().unwrap()).unwrap();
    std::fs::rename(&first, &moved).unwrap();

    let project = Project::open(&moved).unwrap();
    let listed = project.run(&run.id).unwrap();
    let export: serde_json::Value = serde_json::from_slice(&bytes(&listed.export_path())).unwrap();
    let manifest = &export["diagnostics"]["image_manifest"];
    let image = listed
        .dir
        .join(manifest["root"].as_str().unwrap())
        .join(manifest["frames"][0]["path"].as_str().unwrap());
    assert!(image.is_file(), "{}", image.display());
    // A project-aware reader resolves the frames against the data root.
    assert!(
        project
            .data_root()
            .unwrap()
            .join(manifest["frames"][0]["path"].as_str().unwrap())
            .is_file()
    );
    let again = calibrate(&project);
    assert_eq!(calibrated(&again).digest, calibrated(&run).digest);
    assert_eq!(bytes(&again.export_path()), bytes(&listed.export_path()));
}

#[test]
fn concurrent_runs_get_distinct_ids_and_complete_folders() {
    let tmp = tempfile::tempdir().unwrap();
    let project = planar_project(tmp.path());
    let runs: Vec<RunInfo> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..4)
            .map(|_| scope.spawn(|| calibrate(&project)))
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let ids: BTreeSet<&str> = runs.iter().map(|r| r.id.as_str()).collect();
    assert_eq!(ids.len(), 4);
    for run in &runs {
        for file in [EXPORT_FILE, QUALITY_FILE, RUN_RECORD_FILE] {
            assert!(run.dir.join(file).is_file());
        }
    }
    let listed: Vec<String> = project.runs().unwrap().into_iter().map(|r| r.id).collect();
    assert_eq!(listed, ["run-001", "run-002", "run-003", "run-004"]);
    // No staging folder is left behind.
    let names: Vec<String> = std::fs::read_dir(tmp.path().join(RUNS_DIR))
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    assert_eq!(names.len(), 4, "{names:?}");
}

#[test]
fn a_cancelled_run_writes_nothing() {
    let tmp = tempfile::tempdir().unwrap();
    let project = planar_project(tmp.path());
    let token = CancelToken::new();
    token.cancel();
    let control = RunControl::new().with_cancel(token);
    let err = project.calibrate(&options(), None, &control).unwrap_err();
    assert!(matches!(err, ProjectError::Cancelled), "{err}");
    assert_eq!(
        std::fs::read_dir(tmp.path().join(RUNS_DIR))
            .unwrap()
            .count(),
        0
    );
}

/// A stale exclusion fails before detection: the cache is not touched,
/// and no run is written.
#[test]
fn a_stale_exclusion_fails_before_any_image_is_read() {
    let tmp = tempfile::tempdir().unwrap();
    let mut project = planar_project(tmp.path());
    std::fs::remove_dir_all(tmp.path().join(DETECTIONS_DIR)).unwrap();
    project
        .spec_mut()
        .exclude
        .images
        .push(ImageExclusion::new("cam0", "img/view99.png"));
    let err = project
        .calibrate(&options(), None, &RunControl::default())
        .unwrap_err();
    assert!(
        matches!(&err, ProjectError::Exclusion { entry, fault: ExclusionFault::ImageNotFound } if entry == "exclude.images[0]"),
        "{err}"
    );
    assert!(!tmp.path().join(DETECTIONS_DIR).exists());
    assert!(project.runs().unwrap().is_empty());
}

#[test]
fn exclusions_reach_the_run_and_unmatched_points_are_listed() {
    let tmp = tempfile::tempdir().unwrap();
    let mut project = planar_project(tmp.path());
    let plain = calibrate(&project);
    let mut exclusions = Exclusions::default();
    exclusions
        .images
        .push(ImageExclusion::new("cam0", "img/view01.png"));
    // One point of the board, one point off it.
    exclusions.features.push(FeatureExclusion::new(
        "cam0",
        "img/view00.png",
        vec![[0.0, 0.0, 0.0], [5.0, 5.0, 0.0]],
    ));
    project.spec_mut().exclude = exclusions;
    let run = calibrate(&project);
    let calibrated_run = calibrated(&run);
    assert_eq!(calibrated_run.usable_views, 6);
    assert!(
        calibrated_run
            .left_out
            .iter()
            .any(|l| l.image == Path::new("img/view01.png") && l.reason == RejectReason::Excluded)
    );
    assert_eq!(calibrated_run.unmatched_exclusions.len(), 1);
    assert_eq!(
        calibrated_run.unmatched_exclusions[0].points,
        [[5.0, 5.0, 0.0]]
    );
    assert_ne!(
        calibrated_run.digest.exclusions,
        calibrated(&plain).digest.exclusions
    );
    assert_eq!(calibrated_run.inputs.exclusions, project.spec().exclude);
}

#[test]
fn a_run_s_own_patch_layers_over_the_project_s_and_is_recorded() {
    let tmp = tempfile::tempdir().unwrap();
    let mut project = planar_project(tmp.path());
    project.spec_mut().config = ConfigPatch::from_toml_str("solver.max_iters = 40").unwrap();
    let mut opts = options();
    opts.config_patch = ConfigPatch::from_toml_str("solver.max_iters = 30").unwrap();
    opts.report_html = true;
    opts.notes.label = Some("first".into());
    let run = project
        .calibrate(&opts, None, &RunControl::default())
        .unwrap();
    let inputs = &calibrated(&run).inputs;
    let CalibrationConfig::PlanarIntrinsics(config) = &inputs.config else {
        panic!("a planar config");
    };
    assert_eq!(config.solver.max_iters, 30);
    assert_eq!(
        serde_json::to_value(&inputs.config_patch).unwrap(),
        json!({ "solver": { "max_iters": 30 } })
    );
    assert!(run.report_path().is_file());
    assert_eq!(
        project.run(&run.id).unwrap().notes.label.as_deref(),
        Some("first")
    );
}

#[test]
fn notes_change_and_runs_are_deleted() {
    let tmp = tempfile::tempdir().unwrap();
    let project = planar_project(tmp.path());
    let run = calibrate(&project);
    let record_before = bytes(&run.dir.join(RUN_RECORD_FILE));
    let mut notes = RunNotes::new();
    notes.label = Some("kept".into());
    notes.starred = true;
    project.set_run_notes(&run.id, &notes).unwrap();
    assert_eq!(project.run(&run.id).unwrap().notes, notes);
    // The record never changes.
    assert_eq!(bytes(&run.dir.join(RUN_RECORD_FILE)), record_before);
    project.set_run_notes(&run.id, &RunNotes::new()).unwrap();
    assert!(!run.dir.join(RUN_NOTES_FILE).exists());

    for id in ["run-009", "../run-001", ".staging", ""] {
        assert!(matches!(
            project.set_run_notes(id, &notes).unwrap_err(),
            ProjectError::RunNotFound { .. }
        ));
    }
    project.delete_run(&run.id).unwrap();
    assert!(!run.dir.exists());
    assert!(project.runs().unwrap().is_empty());
    // A new run takes the next number after the remaining ones.
    assert_eq!(calibrate(&project).id, "run-001");
}

#[test]
fn unrecorded_and_unreadable_runs_are_listed_and_others_skipped() {
    let tmp = tempfile::tempdir().unwrap();
    let project = planar_project(tmp.path());
    let run = calibrate(&project);
    let runs = tmp.path().join(RUNS_DIR);
    // A run of the app's earlier format: its export and diagnostics.
    let unrecorded = runs.join("before-fix");
    std::fs::create_dir_all(&unrecorded).unwrap();
    std::fs::copy(run.export_path(), unrecorded.join(EXPORT_FILE)).unwrap();
    std::fs::write(
        unrecorded.join("diagnostics.json"),
        r#"{"topology":"planar_intrinsics","usable_views":7,"total_views":8,"duration_ms":5,"cache_used":true}"#,
    )
    .unwrap();
    let broken = runs.join("run-010");
    std::fs::create_dir_all(&broken).unwrap();
    std::fs::write(broken.join(RUN_RECORD_FILE), "{ not json").unwrap();
    std::fs::create_dir_all(runs.join("scratch")).unwrap();
    std::fs::create_dir_all(runs.join(".staging-1-2")).unwrap();
    std::fs::write(runs.join(".staging-1-2").join(EXPORT_FILE), "{}").unwrap();

    let listed = project.runs().unwrap();
    let ids: Vec<&str> = listed.iter().map(|r| r.id.as_str()).collect();
    assert_eq!(ids, ["run-001", "run-010", "before-fix"]);
    assert!(matches!(listed[1].state, RunState::Unreadable { .. }));
    let RunState::Unrecorded(old) = &listed[2].state else {
        panic!("a before run");
    };
    assert_eq!((old.usable_views, old.total_views), (Some(7), Some(8)));
    let summary = old.summary.as_ref().unwrap();
    assert_eq!(summary.kind, ProblemKind::PlanarIntrinsics);
    assert_eq!(summary.verdict, None);
    assert_eq!(summary.cameras[0].views, 7);
    // The next run follows the largest number.
    assert_eq!(calibrate(&project).id, "run-011");
}

#[test]
fn a_bare_export_is_imported_with_its_images_resolving_from_the_run() {
    let tmp = tempfile::tempdir().unwrap();
    let project = planar_project(&tmp.path().join("proj"));
    let run = calibrate(&project);
    // An export written next to the data, as `vcal run -o data` would.
    let data = tmp.path().join("proj/data");
    let mut export: serde_json::Value = serde_json::from_slice(&bytes(&run.export_path())).unwrap();
    export["diagnostics"]["image_manifest"]["root"] = json!(".");
    std::fs::write(
        data.join("calibration.json"),
        serde_json::to_string_pretty(&export).unwrap(),
    )
    .unwrap();

    let mut notes = RunNotes::new();
    notes.label = Some("from vcal".into());
    let imported = project.import_export(&data, notes).unwrap();
    let record = record(&imported);
    assert!(
        matches!(&record.source, RunSource::Imported { file_name } if file_name == "calibration.json")
    );
    assert_eq!(record.summary.cameras[0].views, 7);
    assert!(record.summary.verdict.is_some());
    assert!(imported.quality_path().is_file());
    assert_eq!(imported.notes.label.as_deref(), Some("from vcal"));
    let written: serde_json::Value =
        serde_json::from_slice(&bytes(&imported.export_path())).unwrap();
    assert_eq!(
        written["diagnostics"]["image_manifest"]["root"],
        "../../data"
    );
    // Nothing but the root changed.
    let mut back = written.clone();
    back["diagnostics"]["image_manifest"]["root"] = json!(".");
    assert_eq!(back, export);
    // The quality report is the calibrated run's.
    assert_eq!(bytes(&imported.quality_path()), bytes(&run.quality_path()));
}

#[test]
fn an_export_of_another_workflow_or_camera_count_is_refused() {
    let tmp = tempfile::tempdir().unwrap();
    let project = planar_project(&tmp.path().join("proj"));
    let run = calibrate(&project);

    let mut two_cameras: serde_json::Value = planar_dataset(
        &tmp.path().join("other"),
        &FsDetectionCache::new(tmp.path().join("other-cache")),
    );
    let mut second = two_cameras["cameras"][0].clone();
    second["id"] = json!("cam1");
    two_cameras["cameras"].as_array_mut().unwrap().push(second);
    let other = Project::create(
        tmp.path().join("two"),
        ProjectSpec::new(DatasetSource::inline(
            "../other",
            serde_json::from_value(two_cameras.clone()).unwrap(),
        )),
    )
    .unwrap();
    let err = other
        .import_export(run.export_path(), RunNotes::new())
        .unwrap_err();
    assert!(matches!(err, ProjectError::ImportMismatch { .. }), "{err}");

    let mut handeye = two_cameras;
    handeye["topology"] = json!("single_cam_handeye");
    let other = Project::create(
        tmp.path().join("handeye"),
        ProjectSpec::new(DatasetSource::inline(
            "../other",
            serde_json::from_value(handeye).unwrap(),
        )),
    )
    .unwrap();
    let err = other
        .import_export(run.export_path(), RunNotes::new())
        .unwrap_err();
    assert!(
        matches!(&err, ProjectError::ImportMismatch { message } if message.contains("single_cam_handeye")),
        "{err}"
    );
    assert!(matches!(
        project
            .import_export(tmp.path().join("nothing.json"), RunNotes::new())
            .unwrap_err(),
        ProjectError::Io { .. }
    ));
}

/// With nothing excluded, a project run's export is the dataset runner's,
/// bit for bit, but for the image root restated for the run folder.
#[test]
fn a_project_run_is_the_dataset_runner_s_run() {
    let tmp = tempfile::tempdir().unwrap();
    let project = planar_project(tmp.path());
    let run = calibrate(&project);
    let cache = FsDetectionCache::new(tmp.path().join(DETECTIONS_DIR));
    let direct = crate::dataset_runner::calibrate_dataset(
        &project.dataset().unwrap(),
        &project.data_root().unwrap(),
        None,
        Detection::Cached(&cache),
        None,
        &RunControl::default(),
    )
    .unwrap();
    let mut direct = serde_json::to_value(&direct.export).unwrap();
    direct["diagnostics"]["image_manifest"]["root"] = json!("../../data");
    let written: serde_json::Value = serde_json::from_slice(&bytes(&run.export_path())).unwrap();
    assert_eq!(written, direct);
}
