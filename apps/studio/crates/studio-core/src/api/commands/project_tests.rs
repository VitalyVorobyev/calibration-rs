//! The project and run commands end to end, on a synthetic planar project:
//! blank PNGs whose detections the project's cache already holds (the
//! board's corners projected by an ideal camera), so a calibration runs in
//! a test without a detector.

use std::path::{Path, PathBuf};

use serde_json::{Value, json};
use vision_calibration::core::{BrownConrady5, FxFyCxCy, Pt3, make_pinhole_camera};
use vision_calibration::detect::{
    CacheKey, CachedFeatures, ChessboardConfig, DetectionCache, DetectorConfig, Feature,
    FsDetectionCache,
};
use vision_calibration::synthetic::{planar, poses};

use crate::api::{Api, Conditions, ErrorKind, Setup};
use crate::scope::RootScope;

fn slash(path: &Path) -> String {
    path.to_string_lossy().replace('\\', "/")
}

/// `views` views of an 8 × 6 board in `data/img`, their detections in the
/// project cache of `project`; the manifest.
fn planar_dataset(project: &Path, data: &Path, views: usize) -> Value {
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
    let specs = poses::seeded_board_pose_specs(views, 0.3, 7);
    let board_poses = poses::centered_board_poses(&specs, (0.105, 0.075), 0.6);
    let projected = planar::project_views_all(&camera, &board, &board_poses).unwrap();
    let cache = FsDetectionCache::new(project.join(".detections"));
    std::fs::create_dir_all(data.join("img")).unwrap();
    let config = DetectorConfig::Chessboard(ChessboardConfig::new(6, 8, 0.03));
    let mut paths = Vec::new();
    for (i, view) in projected.iter().enumerate() {
        let rel = format!("img/view{i:02}.png");
        let mut image = image::GrayImage::new(640, 480);
        image.put_pixel(i as u32 % 640, 0, image::Luma([255]));
        image.save(data.join(&rel)).unwrap();
        let features: Vec<Feature> = view
            .points_3d
            .iter()
            .zip(&view.points_2d)
            .map(|(p, q): (&Pt3, _)| Feature::new([q.x, q.y], [p.x, p.y, p.z]))
            .collect();
        let bytes = std::fs::read(data.join(&rel)).unwrap();
        let key = CacheKey::for_detection(&bytes, &config, None);
        cache.put(&key, &CachedFeatures::new(features)).unwrap();
        paths.push(rel);
    }
    json!({
        "version": 1,
        "topology": "planar_intrinsics",
        "cameras": [{ "id": "cam0", "images": { "kind": "list", "paths": paths } }],
        "target": { "kind": "chessboard", "rows": 6, "cols": 8, "square_size_m": 0.03 },
    })
}

/// A root holding the folder `lib/p` (to become a project) beside the
/// folder `outside`, the API over `lib` alone.
struct Library {
    _tmp: tempfile::TempDir,
    api: Api,
    root: PathBuf,
    project: PathBuf,
    outside: PathBuf,
}

fn library() -> Library {
    let tmp = tempfile::tempdir().unwrap();
    let base = dunce::canonicalize(tmp.path()).unwrap();
    let root = base.join("lib");
    let project = root.join("p");
    let outside = base.join("outside");
    std::fs::create_dir_all(&project).unwrap();
    std::fs::create_dir_all(&outside).unwrap();
    let setup = Setup::for_tests(Some(base.join("cache")));
    let api = Api::with_scope(RootScope::fixed([root.clone()]), setup);
    Library {
        _tmp: tmp,
        api,
        root,
        project,
        outside,
    }
}

/// The spec of a project with its dataset inline under `data`.
fn spec(manifest: Value) -> Value {
    json!({ "version": 2, "dataset": { "root": "data", "manifest": manifest } })
}

/// `lib/p` made a project of `views` views through `create_project`.
fn created(lib: &Library, views: usize) -> Value {
    let manifest = planar_dataset(&lib.project, &lib.project.join("data"), views);
    lib.api
        .invoke(
            "create_project",
            json!({ "dir": slash(&lib.project), "spec": spec(manifest) }),
        )
        .unwrap()
}

/// Wait for job `id` to end; its terminal state.
fn finish(api: &Api, id: &Value) -> Value {
    for _ in 0..1200 {
        let info = api.invoke("job_status", json!({ "id": id })).unwrap();
        if !matches!(info["state"]["kind"].as_str(), Some("queued" | "running")) {
            return info["state"].clone();
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    panic!("job {id} did not end");
}

#[test]
fn a_project_is_created_opened_and_saved_but_never_over_a_newer_file() {
    let lib = library();
    let dir = slash(&lib.project);
    let view = created(&lib, 6);
    assert_eq!(view["file_version"], 2);
    assert_eq!(view["dataset"]["kind"], "ok");
    assert_eq!(view["dataset_editable"], true);
    assert_eq!(view["config"]["kind"], "ok");
    assert!(view["config"]["config"]["solver"].is_object(), "{view}");
    assert_eq!(view["data_root"], slash(&lib.project.join("data")));
    assert!(
        view["input_digest"]["config"]
            .as_str()
            .unwrap()
            .starts_with("sha256:")
    );
    assert_eq!(view["runs"], json!([]));
    assert!(lib.project.join("calibration-project.toml").is_file());
    assert_eq!(lib.api.ctx().settings.recent()[0], dir);

    // Saved with the revision it was opened at.
    let opened = lib
        .api
        .invoke("open_project", json!({ "dir": dir }))
        .unwrap();
    let mut edited = opened["spec"].clone();
    edited["name"] = json!("bench camera");
    let saved = lib
        .api
        .invoke(
            "save_project",
            json!({ "dir": dir, "revision": opened["revision"], "spec": edited }),
        )
        .unwrap();
    assert_eq!(saved["name"], "bench camera");
    assert_ne!(saved["revision"], opened["revision"]);
    // The old revision is stale: refused, and the file is left as it is.
    let mut stale = opened["spec"].clone();
    stale["name"] = json!("lost edit");
    let err = lib
        .api
        .invoke(
            "save_project",
            json!({ "dir": dir, "revision": opened["revision"], "spec": stale }),
        )
        .unwrap_err();
    assert_eq!(err.kind, ErrorKind::ChangedOnDisk);
    assert_eq!(err.http_status(), 409);
    let now = lib
        .api
        .invoke("open_project", json!({ "dir": dir }))
        .unwrap();
    assert_eq!(now["name"], "bench camera");

    // Creating over it is refused.
    let err = lib
        .api
        .invoke("create_project", json!({ "dir": dir, "spec": now["spec"] }))
        .unwrap_err();
    assert_eq!(err.kind, ErrorKind::ProjectExists);
}

#[test]
fn a_project_may_name_only_paths_inside_the_roots() {
    let lib = library();
    let view = created(&lib, 4);
    let dir = slash(&lib.project);
    let manifest = view["dataset"]["spec"].clone();
    // A project folder outside the roots.
    let err = lib
        .api
        .invoke(
            "create_project",
            json!({ "dir": slash(&lib.outside.join("q")), "spec": spec(manifest.clone()) }),
        )
        .unwrap_err();
    assert_eq!(err.kind, ErrorKind::ForbiddenPath);
    // A data folder outside them, inline or by a referenced manifest.
    let mut escape = spec(manifest.clone());
    escape["dataset"]["root"] = json!(slash(&lib.outside));
    let err = lib
        .api
        .invoke(
            "save_project",
            json!({ "dir": dir, "revision": view["revision"], "spec": escape }),
        )
        .unwrap_err();
    assert_eq!(err.kind, ErrorKind::ForbiddenPath);
    std::fs::write(lib.outside.join("dataset.json"), manifest.to_string()).unwrap();
    let referenced =
        json!({ "version": 2, "dataset": { "manifest": "../../outside/dataset.json" } });
    let err = lib
        .api
        .invoke("project_views", json!({ "dir": dir, "spec": referenced }))
        .unwrap_err();
    assert_eq!(err.kind, ErrorKind::ForbiddenPath);
    // An image the manifest names outside them.
    let mut images = manifest.clone();
    images["cameras"][0]["images"] = json!({ "kind": "glob", "pattern": "../../../outside/*.png" });
    let err = lib
        .api
        .invoke("project_views", json!({ "dir": dir, "spec": spec(images) }))
        .unwrap_err();
    assert_eq!(err.kind, ErrorKind::ForbiddenPath);
    // A project file edited by hand to point outside does not open.
    std::fs::write(
        lib.project.join("calibration-project.toml"),
        format!(
            "version = 2\n[dataset]\nmanifest = \"{}\"\n",
            slash(&lib.outside.join("dataset.json"))
        ),
    )
    .unwrap();
    let err = lib
        .api
        .invoke("open_project", json!({ "dir": dir }))
        .unwrap_err();
    assert_eq!(err.kind, ErrorKind::ForbiddenPath);
    // A folder that is no project says so.
    let plain = lib.root.join("plain");
    std::fs::create_dir(&plain).unwrap();
    let err = lib
        .api
        .invoke("open_project", json!({ "dir": slash(&plain) }))
        .unwrap_err();
    assert_eq!(err.kind, ErrorKind::NotAProject);
}

#[test]
fn project_views_lists_the_views_and_what_a_draft_lacks() {
    let lib = library();
    let view = created(&lib, 4);
    let dir = slash(&lib.project);
    let views = lib
        .api
        .invoke("project_views", json!({ "dir": dir }))
        .unwrap();
    assert_eq!(views["cameras"], json!(["cam0"]));
    assert_eq!(views["views"].as_array().unwrap().len(), 4);
    assert_eq!(views["views"][2]["images"], json!(["img/view02.png"]));
    assert_eq!(views["messages"], json!([]));

    // A draft (not saved) with a stale exclusion and an open field.
    let mut draft = view["spec"].clone();
    draft["exclude"] = json!({ "images": [{ "camera": "cam0", "image": "img/none.png" }] });
    let checked = lib
        .api
        .invoke("project_views", json!({ "dir": dir, "spec": draft }))
        .unwrap();
    assert_eq!(checked["messages"][0]["kind"], "exclusion");
    assert_eq!(checked["messages"][0]["field"], "exclude.images[0]");
    let mut open = view["spec"].clone();
    open["dataset"]["manifest"]["_unresolved"] = json!(["target"]);
    let checked = lib
        .api
        .invoke("project_views", json!({ "dir": dir, "spec": open }))
        .unwrap();
    assert_eq!(checked["messages"][0]["kind"], "manifest");
    assert_eq!(checked["messages"][0]["field"], "target");
    // Nothing was written.
    let again = lib
        .api
        .invoke("open_project", json!({ "dir": dir }))
        .unwrap();
    assert_eq!(again["revision"], view["revision"]);
}

#[test]
fn project_views_previews_a_draft_in_a_folder_that_is_no_project_yet() {
    let lib = library();
    let data = lib.root.join("captures");
    let manifest = planar_dataset(&lib.project, &data, 3);
    let draft = json!({ "version": 2, "dataset": { "root": ".", "manifest": manifest } });
    let dir = slash(&data);
    let views = lib
        .api
        .invoke("project_views", json!({ "dir": dir, "spec": draft }))
        .unwrap();
    assert_eq!(views["cameras"], json!(["cam0"]));
    assert_eq!(views["views"].as_array().unwrap().len(), 3);
    assert_eq!(views["views"][1]["images"], json!(["img/view01.png"]));
    assert_eq!(views["messages"], json!([]));
    // Nothing was written to the folder, and no scratch project is left.
    assert!(!data.join("calibration-project.toml").exists());
    let drafts = lib.outside.parent().unwrap().join("cache/drafts");
    assert_eq!(std::fs::read_dir(drafts).map_or(0, Iterator::count), 0);

    // The draft's open fields come back as messages with their field.
    let mut open = draft.clone();
    open["dataset"]["manifest"]["_unresolved"] = json!(["target"]);
    let checked = lib
        .api
        .invoke("project_views", json!({ "dir": dir, "spec": open }))
        .unwrap();
    assert_eq!(checked["messages"][0]["field"], "target");

    // Without a draft the folder is no project; outside the roots it is refused.
    let err = lib
        .api
        .invoke("project_views", json!({ "dir": dir }))
        .unwrap_err();
    assert_eq!(err.kind, ErrorKind::NotAProject);
    let err = lib
        .api
        .invoke(
            "project_views",
            json!({ "dir": slash(&lib.outside), "spec": draft }),
        )
        .unwrap_err();
    assert_eq!(err.kind, ErrorKind::ForbiddenPath);
}

#[test]
fn a_calibration_job_writes_a_run_that_is_listed_noted_read_and_deleted() {
    let lib = library();
    let view = created(&lib, 6);
    let dir = slash(&lib.project);
    let events = lib.api.subscribe(None);
    let job = lib
        .api
        .invoke(
            "calibrate",
            json!({ "dir": dir, "notes": { "label": "first" } }),
        )
        .unwrap();
    let state = finish(&lib.api, &job["id"]);
    assert_eq!(state["kind"], "done", "{state}");
    let run = &state["output"];
    assert_eq!(run["id"], "run-001");
    assert_eq!(run["notes"]["label"], "first");
    assert_eq!(run["state"]["kind"], "recorded");
    // Its progress was the library's reports, and the project changed.
    let seen: Vec<_> = events.events.try_iter().collect();
    let body = |e: &crate::api::ServerEvent| serde_json::to_value(&e.body).unwrap();
    assert!(seen.iter().any(|e| body(e)["event"]["kind"] == "detection"));
    assert!(
        seen.iter()
            .any(|e| body(e) == json!({ "kind": "project_changed", "dir": dir }))
    );
    // The project's digest is the run's (the run is current).
    let digest = &run["state"]["source"]["digest"];
    for part in ["dataset", "exclusions", "config", "quality"] {
        assert_eq!(view["input_digest"][part], digest[part], "{part}");
    }

    let named = json!({ "dir": dir, "id": "run-001" });
    let runs = lib.api.invoke("list_runs", json!({ "dir": dir })).unwrap();
    assert_eq!(runs.as_array().unwrap().len(), 1);
    let export = lib
        .api
        .invoke("run_export", json!({ "run": named }))
        .unwrap();
    assert_eq!(export["export"]["kind"], "planar_intrinsics");
    // A calibrated run's frames resolve from the project's data folder.
    assert_eq!(export["image_root"], slash(&lib.project.join("data")));
    let quality = lib
        .api
        .invoke("run_quality", json!({ "run": named }))
        .unwrap();
    assert!(quality["verdict"].is_string(), "{quality}");
    let info = lib.api.invoke("run_info", json!({ "run": named })).unwrap();
    assert_eq!(info["id"], "run-001");
    let report = lib
        .api
        .resource("report", json!({ "run": named }), &Conditions::default())
        .unwrap();
    assert_eq!(report.media_type, "text/html; charset=utf-8");
    assert!(
        report
            .content_security_policy
            .unwrap()
            .starts_with("sandbox")
    );

    let noted = lib
        .api
        .invoke(
            "set_run_notes",
            json!({ "dir": dir, "id": "run-001", "notes": { "notes": "kept", "starred": true } }),
        )
        .unwrap();
    assert_eq!(noted["notes"], json!({ "notes": "kept", "starred": true }));
    let left = lib
        .api
        .invoke("delete_run", json!({ "dir": dir, "id": "run-001" }))
        .unwrap();
    assert_eq!(left, json!([]));
    let err = lib
        .api
        .invoke("run_export", json!({ "run": named }))
        .unwrap_err();
    assert_eq!(err.kind, ErrorKind::RunNotFound);
}

#[test]
fn a_cancelled_calibration_writes_no_run() {
    let lib = library();
    created(&lib, 40);
    let dir = slash(&lib.project);
    let events = lib.api.subscribe(None);
    let job = lib
        .api
        .invoke("calibrate", json!({ "dir": dir, "redetect": false }))
        .unwrap();
    // Cancelled at its first progress report, while it runs.
    while let Ok(event) = events
        .events
        .recv_timeout(std::time::Duration::from_secs(60))
    {
        let body = serde_json::to_value(&event.body).unwrap();
        if body["id"] == job["id"] && body["event"].is_object() {
            lib.api
                .invoke("cancel_job", json!({ "id": job["id"] }))
                .unwrap();
            break;
        }
    }
    let state = finish(&lib.api, &job["id"]);
    assert_eq!(state["kind"], "cancelled", "{state}");
    assert_eq!(
        lib.api.invoke("list_runs", json!({ "dir": dir })).unwrap(),
        json!([])
    );
    // No staging folder is left behind.
    let left: Vec<_> = std::fs::read_dir(lib.project.join("runs"))
        .unwrap()
        .flatten()
        .collect();
    assert!(left.is_empty(), "{left:?}");
}

#[test]
fn a_run_names_an_open_choice_as_needs_input() {
    let lib = library();
    let view = created(&lib, 4);
    let dir = slash(&lib.project);
    let mut spec = view["spec"].clone();
    spec["dataset"]["manifest"]["_unresolved"] = json!(["target"]);
    lib.api
        .invoke(
            "save_project",
            json!({ "dir": dir, "revision": view["revision"], "spec": spec }),
        )
        .unwrap();
    let job = lib.api.invoke("calibrate", json!({ "dir": dir })).unwrap();
    let state = finish(&lib.api, &job["id"]);
    assert_eq!(state["kind"], "failed");
    assert_eq!(state["error"]["kind"], "manifest");
    assert_eq!(state["error"]["field"], "target");
}

/// The committed fixture library: a project of the app's earlier format
/// (version 1) with a run of that format, and a bare export.
fn fixture_library(tmp: &Path) -> (Api, PathBuf) {
    let library = tmp.join("library");
    copy_dir(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("../../e2e/fixtures/library"),
        &library,
    );
    let library = dunce::canonicalize(library).unwrap();
    let setup = Setup::for_tests(Some(tmp.join("cache")));
    (
        Api::with_scope(RootScope::fixed([library.clone()]), setup),
        library,
    )
}

fn copy_dir(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).unwrap();
    for entry in std::fs::read_dir(from).unwrap().flatten() {
        let target = to.join(entry.file_name());
        if entry.file_type().unwrap().is_dir() {
            copy_dir(&entry.path(), &target);
        } else {
            std::fs::copy(entry.path(), target).unwrap();
        }
    }
}

#[test]
fn a_version_1_project_opens_lists_its_run_and_is_migrated_on_save() {
    let tmp = tempfile::tempdir().unwrap();
    let (api, library) = fixture_library(&dunce::canonicalize(tmp.path()).unwrap());
    let dir = slash(&library.join("rig-demo"));
    let view = api.invoke("open_project", json!({ "dir": dir })).unwrap();
    assert_eq!(view["file_version"], 1);
    assert_eq!(view["name"], "rig-demo");
    assert_eq!(view["runs"][0]["id"], "run-001");
    assert_eq!(view["runs"][0]["state"]["kind"], "unrecorded");
    assert_eq!(view["dataset"]["spec"]["topology"], "rig_extrinsics");

    let saved = api
        .invoke(
            "save_project",
            json!({ "dir": dir, "revision": view["revision"], "spec": view["spec"] }),
        )
        .unwrap();
    assert_eq!(saved["file_version"], 2);
    assert!(!library.join("rig-demo/dataset.json").exists());
    // The bare export joins it as its next run.
    let run = api
        .invoke(
            "import_export",
            json!({ "dir": dir, "export": slash(&library.join("vcal-run")) }),
        )
        .unwrap();
    assert_eq!(run["id"], "run-002");
    assert_eq!(run["state"]["source"]["kind"], "imported");
    assert_eq!(run["state"]["source"]["file_name"], "calibration.json");
    // An export outside the roots is not read.
    let err = api
        .invoke(
            "import_export",
            json!({ "dir": dir, "export": slash(&library.join("../outside.json")) }),
        )
        .unwrap_err();
    assert!(
        matches!(err.kind, ErrorKind::ForbiddenPath | ErrorKind::Io),
        "{err:?}"
    );
}

#[test]
fn the_library_scan_finds_the_fixture_projects() {
    let tmp = tempfile::tempdir().unwrap();
    let (api, library) = fixture_library(&dunce::canonicalize(tmp.path()).unwrap());
    let scan = api.invoke("scan_library", json!({})).unwrap();
    let projects = scan["projects"].as_array().unwrap();
    assert!(
        projects
            .iter()
            .any(|p| p["dir"] == json!(library.join("rig-demo"))),
        "{scan}"
    );
}

/// Every committed sample dataset with a project file opens as a project
/// (its referenced manifest read through the scope).
#[test]
fn the_repository_s_sample_projects_open() {
    let data = dunce::canonicalize(Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../../data"))
        .unwrap();
    let api = Api::for_tests(RootScope::fixed([data.clone()]));
    let scan = api.invoke("scan_library", json!({})).unwrap();
    let projects = scan["projects"].as_array().unwrap();
    assert!(projects.len() >= 4, "{scan}");
    assert_eq!(scan["failures"], json!([]));
    for project in projects {
        let dir = project["dir"].as_str().unwrap();
        let view = api.invoke("open_project", json!({ "dir": dir })).unwrap();
        assert_eq!(view["dataset"]["kind"], "ok", "{dir}: {view}");
        assert_eq!(view["dataset_editable"], false, "{dir}");
    }
}
