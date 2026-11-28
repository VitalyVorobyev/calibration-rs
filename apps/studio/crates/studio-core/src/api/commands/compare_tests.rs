//! `compare_runs` and `projection_difference_map` on calibrated copies of
//! the committed `planar-demo` project: runs of the same data under another
//! loss and of nested data (one view left out) compared as the library
//! compares them, the difference map on the maps' grid agreeing with the
//! comparison's summary, imported runs whose tests and projections the
//! library refuses with their reasons, and the requests refused.

use serde_json::{Value, json};
use vision_calibration::analysis::CompareRunsOptions;
use vision_calibration::project::Project;

use super::test_support::{Demo, blob, check_plane, f32_at, finish, golden, slash};
use crate::api::ErrorKind;

/// Calibrate the demo again with the one-off `config_patch`.
fn calibrate(demo: &Demo, config_patch: Value) {
    let job = demo
        .api
        .invoke(
            "calibrate",
            json!({ "dir": demo.dir(), "config_patch": config_patch }),
        )
        .unwrap();
    let state = finish(&demo.api, &job["id"]);
    assert_eq!(state["kind"], "done", "{state}");
}

/// The demo with `run-001` (the project as it is), `run-002` (a Huber loss)
/// and `run-003` (the project's configuration, view 5 left out).
fn three_runs() -> Demo {
    let demo = Demo::calibrated();
    calibrate(
        &demo,
        json!({ "optimize": { "reprojection_loss": { "kind": "huber", "scale": 0.05 } } }),
    );
    let api = &demo.api;
    let opened = api
        .invoke("open_project", json!({ "dir": demo.dir() }))
        .unwrap();
    let mut spec = opened["spec"].clone();
    spec["exclude"] = json!({ "views": [{ "view": { "image": "images/view5.png" } }] });
    api.invoke(
        "save_project",
        json!({ "dir": demo.dir(), "revision": opened["revision"], "spec": spec }),
    )
    .unwrap();
    calibrate(&demo, json!({}));
    demo
}

/// Every parameter test of camera difference `camera`, by kind and reason.
fn tests_of(camera: &Value) -> Vec<(String, Value)> {
    camera["parameters"]["parameters"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| {
            let test = &p["test"];
            (
                test["kind"].as_str().unwrap().to_owned(),
                test["reason"].clone(),
            )
        })
        .collect()
}

#[test]
fn runs_of_one_project_are_compared_as_the_library_compares_them() {
    let demo = three_runs();
    let api = &demo.api;
    let reply = api
        .invoke(
            "compare_runs",
            json!({ "dir": demo.dir(), "runs": ["run-001", "run-002", "run-003"] }),
        )
        .unwrap();
    golden("compare_runs", &reply);

    // The library's comparison, passed through as it is.
    let project = Project::open(&demo.project).unwrap();
    let library = project
        .compare_runs(
            &["run-001", "run-002", "run-003"],
            &CompareRunsOptions::new(),
        )
        .unwrap();
    assert_eq!(reply, serde_json::to_value(&library).unwrap());

    let names: Vec<&str> = reply["runs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r["name"].as_str().unwrap())
        .collect();
    assert_eq!(names, ["run-001", "run-002", "run-003"]);
    assert_eq!(reply["runs"][0]["covariance"]["status"], "available");

    // The same data under another loss: no test, and why.
    let huber = &reply["differences"][0];
    assert_eq!(huber["relation"]["kind"], "same_data", "{huber}");
    assert!(
        huber["config_changes"]
            .as_array()
            .unwrap()
            .iter()
            .any(|c| c["path"]
                .as_str()
                .unwrap()
                .starts_with("config.optimize.reprojection_loss")),
        "{}",
        huber["config_changes"]
    );
    let camera = &huber["cameras"][0];
    assert_eq!(camera["id"], "cam0");
    assert_eq!(camera["parameters"]["kind"], "same_model");
    let shared = json!("shared_data");
    assert!(
        tests_of(camera)
            .iter()
            .all(|(kind, reason)| kind == "not_available" && *reason == shared)
    );
    assert_eq!(camera["parameters"]["joint"]["reason"], shared);
    assert_eq!(camera["projection"]["status"], "compared");

    // A view left out under one configuration: nested, each parameter's
    // own test, no joint test of the camera.
    let nested = &reply["differences"][1];
    assert_eq!(
        nested["relation"],
        json!({ "kind": "nested", "smaller": "other" })
    );
    assert_eq!(nested["data"]["reference_only_images"], 1);
    assert_eq!(nested["data"]["other_only_images"], 0);
    let camera = &nested["cameras"][0];
    assert!(
        tests_of(camera).iter().any(|(kind, _)| kind == "nested"),
        "{camera}"
    );
    assert_eq!(camera["parameters"]["joint"]["reason"], "nested_joint");
    assert_eq!(
        nested["exclusion_changes"]["other_only"]["views"][0]["view"]["image"],
        "images/view5.png"
    );

    // The library's options, as it reads them.
    let none = api
        .invoke(
            "compare_runs",
            json!({ "dir": demo.dir(), "runs": ["run-001", "run-002"],
                    "options": { "compensation": "none", "grid_columns": 16 } }),
        )
        .unwrap();
    let projection = &none["differences"][0]["cameras"][0]["projection"];
    assert_eq!(projection["compensation"], "none");
    assert_eq!(projection["rotation_deg"], 0.0);
}

#[test]
fn the_difference_map_is_the_comparisons_projection_on_the_maps_grid() {
    let demo = three_runs();
    let api = &demo.api;
    let args = json!({ "dir": demo.dir(), "reference": "run-001", "other": "run-002",
                       "camera": "cam0" });
    let map = api
        .invoke("projection_difference_map", args.clone())
        .unwrap();
    golden("projection_difference_map", &map);
    assert_eq!(map["status"], "compared", "{map}");
    assert_eq!(map["camera"], "cam0");
    assert_eq!(map["compensation"], "rotation");

    // The grid of the reference run's other maps.
    let distortion = api
        .invoke(
            "distortion_map",
            json!({ "run": demo.run("run-001"), "camera": 0 }),
        )
        .unwrap();
    assert_eq!(map["grid"], distortion["grid"]);
    let names: Vec<(&str, &str)> = map["plane"]["channels"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| (c["name"].as_str().unwrap(), c["unit"].as_str().unwrap()))
        .collect();
    assert_eq!(
        names,
        [
            ("dx", "px"),
            ("dy", "px"),
            ("magnitude", "px"),
            ("fitted", "flag")
        ]
    );
    let bytes = check_plane(api, &map["grid"], &map["plane"]);
    let nodes = 64 * 48;
    let fitted = (0..nodes)
        .filter(|&i| f32_at(&bytes, 3 * nodes + i) == 1.0)
        .count();
    assert_eq!(fitted as u64, map["fit_nodes"].as_u64().unwrap());

    // Its numbers are the comparison's summary of the camera.
    let comparison = api
        .invoke(
            "compare_runs",
            json!({ "dir": demo.dir(), "runs": ["run-001", "run-002"] }),
        )
        .unwrap();
    let summary = &comparison["differences"][0]["cameras"][0]["projection"];
    for field in [
        "compensation",
        "rotation_deg",
        "fit_nodes",
        "valid_nodes",
        "rms_fit_px",
        "max_fit_px",
        "rms_valid_px",
        "max_valid_px",
    ] {
        assert_eq!(map[field], summary[field], "{field}");
    }

    // Memoized; another compensation is another plane.
    assert_eq!(
        api.invoke("projection_difference_map", args.clone())
            .unwrap(),
        map
    );
    let mut raw = args.clone();
    raw["compensation"] = json!("none");
    let raw = api.invoke("projection_difference_map", raw).unwrap();
    assert_eq!(raw["rotation_deg"], 0.0);
    assert_ne!(raw["plane"]["blob"]["id"], map["plane"]["blob"]["id"]);

    // A camera neither run has, an unknown run, a folder outside the roots,
    // unknown arguments and a grid out of range.
    let refused = |patch: Value| {
        let mut request = args.clone();
        for (key, value) in patch.as_object().unwrap() {
            request[key] = value.clone();
        }
        api.invoke("projection_difference_map", request)
            .unwrap_err()
    };
    let err = refused(json!({ "camera": "cam9" }));
    assert_eq!(
        (err.kind, err.field.as_deref()),
        (ErrorKind::Invalid, Some("camera"))
    );
    assert_eq!(
        refused(json!({ "other": "run-404" })).kind,
        ErrorKind::RunNotFound
    );
    assert_eq!(
        refused(json!({ "dir": slash(&demo.outside) })).kind,
        ErrorKind::ForbiddenPath
    );
    assert_eq!(refused(json!({ "grid": 64 })).kind, ErrorKind::InvalidArgs);
    assert_eq!(
        refused(json!({ "columns": 4096 })).kind,
        ErrorKind::InvalidArgs
    );
}

/// Import run-001's export, with `edit` applied, as the next run. The file
/// sits as deep in the project as the run's, so its image manifest names
/// the project's images.
fn import(demo: &Demo, name: &str, edit: impl Fn(&mut Value)) -> String {
    let export = demo.project.join("runs/run-001/export.json");
    let mut value: Value = serde_json::from_str(&std::fs::read_to_string(export).unwrap()).unwrap();
    edit(&mut value);
    let folder = demo.project.join("exports").join(name);
    std::fs::create_dir_all(&folder).unwrap();
    let file = folder.join("export.json");
    std::fs::write(&file, value.to_string()).unwrap();
    let run = demo
        .api
        .invoke(
            "import_export",
            json!({ "dir": demo.dir(), "export": slash(&file) }),
        )
        .unwrap();
    run["id"].as_str().unwrap().to_owned()
}

#[test]
fn what_the_library_cannot_test_or_compare_is_answered_with_its_reason() {
    let demo = Demo::calibrated();
    let api = &demo.api;
    // An imported copy records no image content: the relation is unknown.
    let copy = import(&demo, "copy", |_| {});
    let reply = api
        .invoke(
            "compare_runs",
            json!({ "dir": demo.dir(), "runs": ["run-001", copy] }),
        )
        .unwrap();
    let difference = &reply["differences"][0];
    assert_eq!(difference["relation"]["kind"], "unknown");
    let camera = &difference["cameras"][0];
    assert!(
        tests_of(camera)
            .iter()
            .all(|(kind, reason)| kind == "not_available" && reason == "unknown_relation"),
        "{camera}"
    );
    // The same calibration projects the same rays.
    assert_eq!(camera["projection"]["status"], "compared");
    assert!(camera["projection"]["max_valid_px"].as_f64().unwrap() < 1e-9);

    // Without per-feature residuals no grid block is covered: no projection
    // difference, in the comparison and on the map alike.
    let strip = |value: &mut Value| {
        value["diagnostics"]["per_feature_residuals"]["target"] = json!([]);
        value["uncertainty"] =
            json!({ "status": "unavailable", "reason": { "kind": "not_computed" } });
    };
    let a = import(&demo, "bare-a", strip);
    let b = import(&demo, "bare-b", strip);
    let reply = api
        .invoke("compare_runs", json!({ "dir": demo.dir(), "runs": [a, b] }))
        .unwrap();
    assert_eq!(
        reply["runs"][0]["covariance"],
        json!({ "status": "unavailable", "reason": { "kind": "not_computed" } })
    );
    assert_eq!(
        reply["differences"][0]["cameras"][0]["projection"],
        json!({ "status": "unavailable", "reason": "too_few_nodes" })
    );
    let map = api
        .invoke(
            "projection_difference_map",
            json!({ "dir": demo.dir(), "reference": a, "other": b, "camera": "cam0" }),
        )
        .unwrap();
    assert_eq!(
        map,
        json!({ "status": "unavailable", "camera": "cam0", "reason": "too_few_nodes" })
    );
}

#[test]
fn a_comparison_needs_two_known_runs_and_the_librarys_options() {
    let demo = Demo::calibrated();
    let api = &demo.api;
    let compare = |args: Value| api.invoke("compare_runs", args).unwrap_err();
    let err = compare(json!({ "dir": demo.dir(), "runs": ["run-001"] }));
    assert_eq!(
        (err.kind, err.field.as_deref()),
        (ErrorKind::InvalidArgs, Some("runs"))
    );
    let err = compare(json!({ "dir": demo.dir(), "runs": ["run-001", "run-404"] }));
    assert_eq!(err.kind, ErrorKind::RunNotFound);
    let err = compare(json!({ "dir": slash(&demo.outside), "runs": ["run-001", "run-001"] }));
    assert_eq!(err.kind, ErrorKind::ForbiddenPath);
    for options in [
        json!({ "grid_columns": 0 }),
        json!({ "grid_columns": 4096 }),
        json!({ "columns": 16 }),
    ] {
        let err = compare(json!({ "dir": demo.dir(), "runs": ["run-001", "run-001"],
                                  "options": options }));
        assert_eq!(err.kind, ErrorKind::InvalidArgs, "{options}");
    }
    // A run against itself projects the same rays, on the grid asked for.
    let map = api
        .invoke(
            "projection_difference_map",
            json!({ "dir": demo.dir(), "reference": "run-001", "other": "run-001",
                    "camera": "cam0", "columns": 16 }),
        )
        .unwrap();
    assert_eq!(map["status"], "compared");
    assert_eq!(map["grid"]["columns"], 16);
    assert!(map["max_valid_px"].as_f64().unwrap() < 1e-9);
    let bytes = blob(api, &map["plane"]);
    assert_eq!(&bytes[..4], b"VAM1");
}
