//! The device and seed commands on the committed fixture library: two
//! projects of the device `Stereo-X1 / SN-0042` (three imported runs, the
//! second starred), one of `SN-0057` (one failed run), the earlier-format
//! `rig-demo`; and a seeded calibration of `planar-demo`.

use std::path::PathBuf;

use serde_json::{Value, json};
use vision_calibration::device_seed::DeviceSeed;

use super::test_support::{Demo, copy_dir, finish, fixture, golden, slash};
use crate::api::{Api, ErrorKind, Setup};
use crate::scope::RootScope;

struct Fixtures {
    _tmp: tempfile::TempDir,
    api: Api,
    root: PathBuf,
    outside: PathBuf,
}

impl Fixtures {
    fn new() -> Self {
        let tmp = tempfile::tempdir().unwrap();
        let base = dunce::canonicalize(tmp.path()).unwrap();
        let root = base.join("lib");
        for name in [
            "stereo-sn0042",
            "stereo-sn0042-field",
            "stereo-sn0057",
            "rig-demo",
        ] {
            copy_dir(&fixture(name), &root.join(name));
        }
        let outside = base.join("outside");
        std::fs::create_dir_all(&outside).unwrap();
        let api = Api::with_scope(
            RootScope::fixed([root.clone()]),
            Setup::for_tests(Some(base.join("cache"))),
        );
        Self {
            _tmp: tmp,
            api,
            root,
            outside,
        }
    }

    fn dir(&self, project: &str) -> String {
        slash(&self.root.join(project))
    }

    fn run(&self, project: &str, id: &str) -> Value {
        json!({ "dir": self.dir(project), "id": id })
    }

    fn call(&self, name: &str, args: Value) -> Value {
        self.api
            .invoke(name, args)
            .unwrap_or_else(|e| panic!("{name}: {e:?}"))
    }

    fn fail(&self, name: &str, args: Value) -> crate::api::StudioError {
        self.api.invoke(name, args).expect_err(name)
    }
}

/// The selector of the device `Stereo-X1 / SN-0042`.
fn sn42() -> Value {
    json!({ "kind": "device", "product": "Stereo-X1", "id": "SN-0042" })
}

#[test]
fn a_device_history_spans_its_projects_in_the_librarys_order() {
    let lib = Fixtures::new();
    let reply = lib.call("device_history", json!({ "device": sn42() }));
    golden("device_history", &reply);
    let devices = reply["devices"].as_array().unwrap();
    assert_eq!(devices.len(), 1);
    let runs = devices[0]["runs"].as_array().unwrap();
    let rows: Vec<(&str, &str, &str, bool, bool)> = runs
        .iter()
        .map(|r| {
            (
                r["project_name"].as_str().unwrap(),
                r["id"].as_str().unwrap(),
                r["verdict"].as_str().unwrap(),
                r["starred"].as_bool().unwrap(),
                r["representative"].as_bool().unwrap(),
            )
        })
        .collect();
    assert_eq!(
        rows,
        [
            ("Bench calibration", "run-001", "warn", false, false),
            ("Bench calibration", "run-002", "pass", true, true),
            ("Field recheck", "run-001", "pass", false, false),
        ]
    );
    assert_eq!(runs[1]["label"], "tight loss");
    assert_eq!(runs[1]["origin"]["kind"], "imported");
    assert_eq!(runs[1]["timestamp"], 1_793_000_000u64);
    assert_eq!(
        runs[1]["cameras"][1],
        json!({ "id": "cam1", "rms_px": 0.247 })
    );
    assert_eq!(runs[0]["project"], lib.dir("stereo-sn0042"));

    // A device type: each of its devices; a device no project names: none.
    let reply = lib.call(
        "device_history",
        json!({ "device": { "kind": "type", "product": "Stereo-X1" } }),
    );
    let ids: Vec<&str> = reply["devices"]
        .as_array()
        .unwrap()
        .iter()
        .map(|d| d["device"]["id"].as_str().unwrap())
        .collect();
    assert_eq!(ids, ["SN-0042", "SN-0057"]);
    let sn57 = &reply["devices"][1]["runs"][0];
    assert_eq!(sn57["representative"], false, "a failed, unstarred import");
    let none = lib.call(
        "device_history",
        json!({ "device": { "kind": "device", "id": "SN-0042" } }),
    );
    assert_eq!(
        none["devices"],
        json!([]),
        "SN-0042 of no product is another device"
    );
    let unknown = lib.api.invoke(
        "device_history",
        json!({ "device": { "kind": "device", "id": "x", "serial": 1 } }),
    );
    assert_eq!(unknown.unwrap_err().kind, ErrorKind::InvalidArgs);
}

#[test]
fn a_device_series_has_each_parameter_over_its_runs_with_the_librarys_spread() {
    let lib = Fixtures::new();
    let reply = lib.call("device_series", json!({ "device": sn42() }));
    golden("device_series", &reply);
    let series = &reply["devices"][0];
    let orders: Vec<u64> = series["runs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r["order"].as_u64().unwrap())
        .collect();
    assert_eq!(orders, [0, 1, 2]);
    let parameter = |key: &str| -> Value {
        series["parameters"]
            .as_array()
            .unwrap()
            .iter()
            .find(|p| p["key"] == key)
            .unwrap_or_else(|| panic!("no series {key}"))
            .clone()
    };
    let fx = parameter("cam0.fx");
    assert_eq!(fx["group"], "intrinsics");
    assert_eq!(fx["camera"], "cam0");
    assert_eq!(fx["unit"], "px");
    assert_eq!(fx["sigma_of"], "value");
    let values: Vec<f64> = fx["points"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| p["value"].as_f64().unwrap())
        .collect();
    assert_eq!(values, [200.0, 200.6, 199.8]);
    assert_eq!(
        fx["points"][0]["sigma"],
        json!({ "kind": "estimated", "sigma": 0.5 })
    );
    assert_eq!(fx["spread"], "cam0.fx");
    let baseline = parameter("cam1.cam_se3_rig.tx");
    assert_eq!(baseline["group"], "mount");
    assert_eq!(baseline["sigma_of"], "perturbation");
    assert_eq!(baseline["points"][2]["value"], -0.03);
    assert!(baseline.get("spread").is_none());

    // The statistics are the library's aggregate of the runs' seeds.
    let statistics = &series["statistics"];
    assert_eq!(statistics["status"], "available", "{statistics}");
    assert_eq!(statistics["runs"], json!([0, 1, 2]));
    let spread = statistics["spread"]
        .as_array()
        .unwrap()
        .iter()
        .find(|s| s["label"] == "cam0.fx")
        .unwrap();
    assert_eq!(spread["values"], json!([200.0, 200.6, 199.8]));
    assert_eq!(spread["median"], 200.0);
    let mad = spread["mad_sigma"].as_f64().unwrap();
    assert!((mad - 1.482_602_218_505_602 * 0.2).abs() < 1e-9, "{mad}");

    let starred = lib.call(
        "device_series",
        json!({ "device": sn42(), "starred_only": true }),
    );
    let runs = starred["devices"][0]["runs"].as_array().unwrap();
    assert_eq!(runs.len(), 1);
    assert_eq!(runs[0]["order"], 1);
    assert_eq!(
        starred["devices"][0]["parameters"][0]["points"][0]["value"],
        200.6
    );
}

#[test]
fn a_parameter_only_some_runs_have_is_a_gap_not_dropped() {
    let lib = Fixtures::new();
    // The field run's first camera becomes a Brown-Conrady one: its
    // coefficients exist in that run alone.
    let export = lib
        .root
        .join("stereo-sn0042-field/runs/run-001/export.json");
    let mut value: Value =
        serde_json::from_str(&std::fs::read_to_string(&export).unwrap()).unwrap();
    value["cameras"][0]["distortion"] = json!({
        "type": "brown_conrady5", "k1": -0.1, "k2": 0.0, "k3": 0.0, "p1": 0.0, "p2": 0.0, "iters": 8
    });
    std::fs::write(&export, serde_json::to_string(&value).unwrap()).unwrap();

    let reply = lib.call("device_series", json!({ "device": sn42() }));
    let series = &reply["devices"][0];
    let k1 = series["parameters"]
        .as_array()
        .unwrap()
        .iter()
        .find(|p| p["key"] == "cam0.k1")
        .unwrap();
    assert_eq!(k1["group"], "distortion");
    assert_eq!(k1["points"][0], Value::Null);
    assert_eq!(k1["points"][1], Value::Null);
    assert_eq!(k1["points"][2]["value"], -0.1);
    assert_eq!(k1["points"][2]["sigma"]["kind"], "unavailable");
    // The runs' cameras are of two models: the library aggregates none.
    let statistics = &series["statistics"];
    assert_eq!(statistics["status"], "unavailable");
    assert_eq!(statistics["reason"]["kind"], "seed");
    assert!(k1.get("spread").is_none());
}

#[test]
fn a_seed_preview_is_one_runs_seed_or_the_librarys_aggregate() {
    let lib = Fixtures::new();
    let one = lib.call(
        "seed_preview",
        json!({ "from": { "kind": "run", "run": lib.run("stereo-sn0042", "run-002") } }),
    );
    let seed: DeviceSeed = serde_json::from_value(one["seed"].clone()).unwrap();
    assert_eq!(seed.cameras.len(), 2);
    assert_eq!(seed.cameras[0].id, "cam0");
    assert_eq!(seed.cameras[0].image_size, [96, 72]);
    assert_eq!(
        seed.description.as_deref(),
        Some("the imported run run-002")
    );
    assert_eq!(one["spread"], json!([]));
    assert_eq!(
        one["runs"],
        json!([{ "project": lib.dir("stereo-sn0042"), "id": "run-002" }])
    );

    let two = lib.call(
        "seed_preview",
        json!({ "from": { "kind": "runs", "runs": [
            lib.run("stereo-sn0042", "run-002"),
            lib.run("stereo-sn0057", "run-001"),
        ] } }),
    );
    golden("seed_preview", &two);
    let fx = two["spread"]
        .as_array()
        .unwrap()
        .iter()
        .find(|s| s["label"] == "cam0.fx")
        .unwrap();
    assert_eq!(fx["values"], json!([200.6, 200.0]));
    assert!((fx["median"].as_f64().unwrap() - 200.3).abs() < 1e-12);
    assert_eq!(two["omitted"], json!([]));

    // Units: each device's representative run. SN-0057 has none.
    let units = lib.fail(
        "seed_preview",
        json!({ "from": { "kind": "units", "devices": [
            { "product": "Stereo-X1", "id": "SN-0042" },
            { "product": "Stereo-X1", "id": "SN-0057" },
        ] } }),
    );
    assert_eq!(units.kind, ErrorKind::Invalid);
    assert_eq!(units.field.as_deref(), Some("devices[1]"));
    let unit = lib.call(
        "seed_preview",
        json!({ "from": { "kind": "units", "devices": [
            { "product": "Stereo-X1", "id": "SN-0042" },
        ] } }),
    );
    assert_eq!(unit["runs"][0]["id"], "run-002");

    // A run of the earlier format names no cameras: the library's error.
    let legacy = lib.fail(
        "seed_preview",
        json!({ "from": { "kind": "run", "run": lib.run("rig-demo", "run-001") } }),
    );
    assert_eq!(legacy.kind, ErrorKind::Seed);
    assert_eq!(legacy.field.as_deref(), Some("seed"));
}

#[test]
fn a_seed_is_exported_as_json_or_toml_inside_the_roots() {
    let lib = Fixtures::new();
    let from = json!({ "kind": "run", "run": lib.run("stereo-sn0042", "run-002") });
    let json_file = lib.root.join("seed.json");
    let written = lib.call(
        "export_seed",
        json!({ "from": from, "dest": slash(&json_file) }),
    );
    assert_eq!(written["path"], slash(&json_file));
    let back: DeviceSeed =
        serde_json::from_str(&std::fs::read_to_string(&json_file).unwrap()).unwrap();
    assert_eq!(
        serde_json::to_value(&back).unwrap(),
        written["preview"]["seed"]
    );

    let toml_file = lib.root.join("seed.toml");
    lib.call(
        "export_seed",
        json!({ "from": from, "dest": slash(&toml_file) }),
    );
    let back: DeviceSeed = toml::from_str(&std::fs::read_to_string(&toml_file).unwrap()).unwrap();
    back.validate().unwrap();
    assert_eq!(back.cameras[1].id, "cam1");

    let exists = lib.fail(
        "export_seed",
        json!({ "from": from, "dest": slash(&json_file) }),
    );
    assert_eq!(exists.kind, ErrorKind::FileExists);
    assert_eq!(exists.http_status(), 409);
    assert_eq!(exists.field.as_deref(), Some("dest"));
    lib.call(
        "export_seed",
        json!({ "from": from, "dest": slash(&json_file), "overwrite": true }),
    );
    let text = lib.fail(
        "export_seed",
        json!({ "from": from, "dest": slash(&lib.root.join("seed.txt")) }),
    );
    assert_eq!(text.kind, ErrorKind::Invalid);
    assert_eq!(text.field.as_deref(), Some("dest"));
    let outside = lib.fail(
        "export_seed",
        json!({ "from": from, "dest": slash(&lib.outside.join("seed.json")) }),
    );
    assert_eq!(outside.kind, ErrorKind::ForbiddenPath);
    assert!(!lib.outside.join("seed.json").exists());
}

#[test]
fn a_project_seed_is_checked_by_the_library_before_it_is_saved() {
    let lib = Fixtures::new();
    let dir = lib.dir("stereo-sn0057");
    let open = |lib: &Fixtures| lib.call("open_project", json!({ "dir": dir }));
    let seed = json!({ "run": "../stereo-sn0042/runs/run-002" });

    // The seed's lenses are undistorted; the project's model is
    // Brown-Conrady: the library refuses it and nothing is written.
    let before = open(&lib);
    let refused = lib.fail(
        "set_project_seed",
        json!({ "dir": dir, "revision": before["revision"], "seed": seed }),
    );
    assert_eq!(refused.kind, ErrorKind::Seed, "{refused:?}");
    assert_eq!(refused.field.as_deref(), Some("seed"));
    assert_eq!(open(&lib)["revision"], before["revision"]);

    // With the project's model undistorted, it fits.
    let mut spec = before["spec"].clone();
    spec["config"] = json!({ "camera": { "model": { "distortion": "none" } } });
    let saved = lib.call(
        "save_project",
        json!({ "dir": dir, "revision": before["revision"], "spec": spec }),
    );
    let set = lib.call(
        "set_project_seed",
        json!({ "dir": dir, "revision": saved["revision"], "seed": seed }),
    );
    assert_eq!(set["spec"]["seed"], seed);
    assert!(
        set["input_digest"]["seed"].is_string(),
        "{}",
        set["input_digest"]
    );

    // A stale revision, a run that is not one, a path outside the roots.
    let stale = lib.fail(
        "set_project_seed",
        json!({ "dir": dir, "revision": saved["revision"], "seed": null }),
    );
    assert_eq!(stale.kind, ErrorKind::ChangedOnDisk);
    let missing = lib.fail(
        "set_project_seed",
        json!({ "dir": dir, "revision": set["revision"], "seed": { "run": "runs/run-999" } }),
    );
    assert_eq!(missing.kind, ErrorKind::Seed);
    assert!(missing.path.unwrap().ends_with("runs/run-999"));
    let outside = slash(&lib.outside.join("seed.json"));
    let forbidden = lib.fail(
        "set_project_seed",
        json!({ "dir": dir, "revision": set["revision"], "seed": { "file": outside } }),
    );
    assert_eq!(forbidden.kind, ErrorKind::ForbiddenPath);
    assert_eq!(forbidden.field.as_deref(), Some("seed"));
    let mut spec = set["spec"].clone();
    spec["seed"] = json!({ "device_spec": outside });
    let forbidden = lib.fail(
        "save_project",
        json!({ "dir": dir, "revision": set["revision"], "spec": spec }),
    );
    assert_eq!(forbidden.kind, ErrorKind::ForbiddenPath);

    // Cleared.
    let cleared = lib.call(
        "set_project_seed",
        json!({ "dir": dir, "revision": set["revision"], "seed": null }),
    );
    assert!(cleared["spec"].get("seed").is_none());
    assert!(cleared["input_digest"].get("seed").is_none());
}

#[test]
fn a_project_file_naming_a_seed_outside_the_roots_does_not_open() {
    let lib = Fixtures::new();
    let file = lib.root.join("stereo-sn0057/calibration-project.toml");
    let text = std::fs::read_to_string(&file).unwrap();
    let outside = slash(&lib.outside.join("seed.json"));
    std::fs::write(&file, format!("{text}\n[seed]\nfile = \"{outside}\"\n")).unwrap();
    let error = lib.fail("open_project", json!({ "dir": lib.dir("stereo-sn0057") }));
    assert_eq!(error.kind, ErrorKind::ForbiddenPath);
    assert_eq!(error.field.as_deref(), Some("seed"));
}

#[test]
fn a_seeded_run_says_what_it_started_from_and_what_it_took() {
    let demo = Demo::calibrated();
    let dir = demo.dir();
    let open = demo
        .api
        .invoke("open_project", json!({ "dir": dir }))
        .unwrap();
    let mut spec = open["spec"].clone();
    spec["device"] = json!({ "product": "Demo", "id": "PD-1" });
    spec["seed"] = json!({ "run": "runs/run-001" });
    demo.api
        .invoke(
            "save_project",
            json!({ "dir": dir, "revision": open["revision"], "spec": spec }),
        )
        .unwrap();
    let job = demo.api.invoke("calibrate", json!({ "dir": dir })).unwrap();
    let state = finish(&demo.api, &job["id"]);
    assert_eq!(state["kind"], "done", "{state}");

    let info = demo
        .api
        .invoke("run_info", json!({ "run": demo.run("run-002") }))
        .unwrap();
    let seed = &info["state"]["source"]["inputs"]["seed"];
    assert_eq!(seed["source"], json!({ "run": "runs/run-001" }));
    assert_eq!(seed["applied"], json!(["cam0.lens"]));
    assert_eq!(seed["value"]["cameras"][0]["id"], "cam0");

    let history = demo
        .api
        .invoke(
            "device_history",
            json!({ "device": { "kind": "device", "product": "Demo", "id": "PD-1" } }),
        )
        .unwrap();
    let runs = history["devices"][0]["runs"].as_array().unwrap();
    assert_eq!(runs.len(), 2);
    assert!(runs[0].get("seed").is_none());
    assert_eq!(runs[1]["origin"]["kind"], "calibrated");
    assert_eq!(runs[1]["seed"]["applied"], json!(["cam0.lens"]));
    assert_eq!(runs[1]["seed"]["description"], "the calibrated run run-001");
}
