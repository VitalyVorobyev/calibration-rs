//! The detection commands and exclusions on a copy of the committed
//! `planar-demo` project (six rendered views of a 7 × 5 chessboard):
//! the overview before and after detection, a detection's detail, a
//! preview that writes nothing, the `detect_all` job, exclusions that
//! round-trip through `save_project` and reach a run, and the refusals.

use std::collections::BTreeMap;
use std::path::Path;
use std::time::SystemTime;

use serde_json::{Value, json};

use super::test_support::{Demo, finish, golden, slash};
use crate::api::ErrorKind;

/// Every file under `dir` with its length and modification time.
fn listing(dir: &Path) -> BTreeMap<String, (u64, Option<SystemTime>)> {
    let mut out = BTreeMap::new();
    let Ok(entries) = std::fs::read_dir(dir) else {
        return out;
    };
    for entry in entries.flatten() {
        let meta = entry.metadata().unwrap();
        if meta.is_dir() {
            for (name, value) in listing(&entry.path()) {
                out.insert(
                    format!("{}/{name}", entry.file_name().to_string_lossy()),
                    value,
                );
            }
        } else {
            out.insert(
                entry.file_name().to_string_lossy().into_owned(),
                (meta.len(), meta.modified().ok()),
            );
        }
    }
    out
}

/// The overview's cell of `view` (camera 0).
fn cell(overview: &Value, view: usize) -> &Value {
    &overview["views"][view]["images"][0]
}

#[test]
fn the_overview_follows_the_detection_cache() {
    let demo = Demo::new();
    let api = &demo.api;
    let before = api
        .invoke("detection_overview", json!({ "dir": demo.dir() }))
        .unwrap();
    assert_eq!(before["cameras"], json!(["cam0"]));
    assert_eq!(before["expected"], 35);
    assert_eq!(before["run"], Value::Null);
    assert_eq!(before["views"].as_array().unwrap().len(), 6);
    assert_eq!(cell(&before, 0)["image"], "images/view0.png");
    for view in 0..6 {
        assert_eq!(cell(&before, view)["state"]["kind"], "not_detected");
    }
    // An overview reads the cache; it writes nothing.
    assert!(!demo.project.join(".detections").exists());

    let job = api
        .invoke("detect_all", json!({ "dir": demo.dir() }))
        .unwrap();
    let state = finish(api, &job["id"]);
    assert_eq!(state["kind"], "done", "{state}");
    let info = api
        .invoke("job_status", json!({ "id": job["id"] }))
        .unwrap();
    assert_eq!(info["last_event"]["done"], 6, "{info}");
    assert_eq!(info["last_event"]["total"], 6);
    let output = &state["output"];
    golden("detection_overview", output);
    for view in 0..6 {
        let state = &cell(output, view)["state"];
        assert_eq!(state["kind"], "detected", "{state}");
        assert_eq!(state["features"], 35);
        assert_eq!(state["outcome"]["status"], "detected");
    }
    let after = api
        .invoke("detection_overview", json!({ "dir": demo.dir() }))
        .unwrap();
    assert_eq!(&after, output, "the overview reads what the job cached");
}

#[test]
fn a_calibration_s_cache_and_left_out_images_show_in_the_overview() {
    let demo = Demo::new();
    let api = &demo.api;
    // Leave view 5 out as a whole view, view 4's image of cam0, and two
    // corners of view 0 (by the target points the detection labelled).
    let detail = api
        .invoke(
            "detection_detail",
            json!({ "dir": demo.dir(), "camera": 0, "image": "images/view0.png" }),
        )
        .unwrap();
    let points: Vec<Value> = detail["features"].as_array().unwrap()[..2]
        .iter()
        .map(|f| f["world_xyz"].clone())
        .collect();
    let opened = api
        .invoke("open_project", json!({ "dir": demo.dir() }))
        .unwrap();
    let mut spec = opened["spec"].clone();
    let exclude = json!({
        "views": [{ "view": { "image": "images/view5.png" }, "note": "blurred" }],
        "images": [{ "camera": "cam0", "image": "images/view4.png" }],
        "features": [{ "camera": "cam0", "image": "images/view0.png", "points": points }],
    });
    spec["exclude"] = exclude.clone();
    api.invoke(
        "save_project",
        json!({ "dir": demo.dir(), "revision": opened["revision"], "spec": spec }),
    )
    .unwrap();
    // The exclusions round-trip, and the project would run.
    let reopened = api
        .invoke("open_project", json!({ "dir": demo.dir() }))
        .unwrap();
    assert_eq!(reopened["spec"]["exclude"], exclude);
    let views = api
        .invoke("project_views", json!({ "dir": demo.dir() }))
        .unwrap();
    assert_eq!(views["messages"], json!([]), "{views}");

    let job = api
        .invoke("calibrate", json!({ "dir": demo.dir() }))
        .unwrap();
    let state = finish(api, &job["id"]);
    assert_eq!(state["kind"], "done", "{state}");
    let stats = api
        .invoke("residual_stats", json!({ "run": demo.run("run-001") }))
        .unwrap();
    assert_eq!(stats["cameras"][0]["stats"]["count"], 4 * 35 - 2);
    assert_eq!(stats["cameras"][0]["views"], 4);

    let overview = api
        .invoke("detection_overview", json!({ "dir": demo.dir() }))
        .unwrap();
    assert_eq!(overview["run"], "run-001");
    assert_eq!(cell(&overview, 0)["left_out"], Value::Null);
    assert_eq!(cell(&overview, 4)["left_out"], "excluded");
    assert_eq!(cell(&overview, 5)["left_out"], "excluded");
    // The calibration cached features (no detail) of the images it read.
    assert_eq!(cell(&overview, 1)["state"]["features"], 35);
    assert_eq!(cell(&overview, 1)["state"]["outcome"]["status"], "detected");
    assert_eq!(cell(&overview, 5)["state"]["kind"], "not_detected");

    // A stale exclusion is the library's typed error, with its entry.
    let mut stale = reopened["spec"].clone();
    stale["exclude"]["images"][0]["image"] = json!("images/missing.png");
    api.invoke(
        "save_project",
        json!({ "dir": demo.dir(), "revision": reopened["revision"], "spec": stale }),
    )
    .unwrap();
    let views = api
        .invoke("project_views", json!({ "dir": demo.dir() }))
        .unwrap();
    let message = &views["messages"][0];
    assert_eq!(message["kind"], "exclusion", "{views}");
    assert_eq!(message["field"], "exclude.images[0]");
}

#[test]
fn a_detail_is_cached_and_a_preview_writes_nothing() {
    let demo = Demo::new();
    let api = &demo.api;
    let args = json!({ "dir": demo.dir(), "camera": 0, "image": "images/view2.png" });
    let detail = api.invoke("detection_detail", args.clone()).unwrap();
    golden("detection_detail", &detail);
    assert_eq!(detail["outcome"]["status"], "detected");
    assert_eq!(detail["features"].as_array().unwrap().len(), 35);
    assert_eq!(detail["target"]["kind"], "chessboard");
    assert_eq!(detail["target"]["corners"].as_array().unwrap().len(), 35);
    let cache = demo.project.join(".detections");
    let cached = listing(&cache);
    assert_eq!(cached.len(), 2, "features and detail: {cached:?}");
    assert_eq!(api.invoke("detection_detail", args).unwrap(), detail);

    let project_file = demo.project.join("calibration-project.toml");
    let spec_before = std::fs::read(&project_file).unwrap();
    let preview = |detector: Value| {
        api.invoke(
            "detect_preview",
            json!({ "dir": demo.dir(), "camera": 0, "image": "images/view3.png",
                    "detector": detector }),
        )
        .unwrap()
    };
    let same = preview(json!({}));
    assert_eq!(same["features"].as_array().unwrap().len(), 35);
    golden("detect_preview", &same);
    let strict = preview(json!({ "chess_corners": { "min_corner_strength": 1e9 } }));
    assert_eq!(strict["features"], json!([]), "the edited settings apply");
    assert_ne!(strict["outcome"]["status"], "detected");
    assert_eq!(listing(&cache), cached, "the preview wrote to the cache");
    assert_eq!(std::fs::read(&project_file).unwrap(), spec_before);
    // `detector` is required: absent is not "the project's".
    let err = api
        .invoke(
            "detect_preview",
            json!({ "dir": demo.dir(), "camera": 0, "image": "images/view3.png" }),
        )
        .unwrap_err();
    assert_eq!(err.kind, ErrorKind::InvalidArgs);
}

#[test]
fn unknown_cameras_images_and_paths_outside_the_roots_are_refused() {
    let demo = Demo::new();
    let api = &demo.api;
    std::fs::write(demo.outside.join("x.png"), b"not an image").unwrap();
    let cases = [
        (
            json!({ "camera": 1, "image": "images/view0.png" }),
            ErrorKind::Invalid,
        ),
        (
            json!({ "camera": 0, "image": "images/nope.png" }),
            ErrorKind::Io,
        ),
        (
            json!({ "camera": 0, "image": "calibration-project.toml" }),
            ErrorKind::Invalid,
        ),
        (
            json!({ "camera": 0, "image": "../../outside/x.png" }),
            ErrorKind::ForbiddenPath,
        ),
        (
            json!({ "camera": 0, "image": slash(&demo.outside.join("x.png")) }),
            ErrorKind::ForbiddenPath,
        ),
    ];
    for (args, kind) in cases {
        let mut args = args;
        args["dir"] = json!(demo.dir());
        let err = api.invoke("detection_detail", args.clone()).unwrap_err();
        assert_eq!(err.kind, kind, "{args}: {err:?}");
        args["detector"] = json!({});
        let err = api.invoke("detect_preview", args.clone()).unwrap_err();
        assert_eq!(err.kind, kind, "preview {args}: {err:?}");
    }
    let outside = json!({ "dir": slash(&demo.outside) });
    let err = api
        .invoke("detection_overview", outside.clone())
        .unwrap_err();
    assert_eq!(err.kind, ErrorKind::ForbiddenPath);
    // A job's arguments are checked when it runs.
    let job = api.invoke("detect_all", outside).unwrap();
    let state = finish(api, &job["id"]);
    assert_eq!(state["kind"], "failed", "{state}");
    assert_eq!(state["error"]["kind"], "forbidden_path");
    let err = api
        .invoke(
            "detection_overview",
            json!({ "dir": demo.dir(), "run": "run-404" }),
        )
        .unwrap_err();
    assert_eq!(err.kind, ErrorKind::RunNotFound);
}
