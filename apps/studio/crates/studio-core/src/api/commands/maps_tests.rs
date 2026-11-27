//! The map, parameter and residual commands on a calibrated copy of the
//! committed `planar-demo` project: replies and their golden shapes, the
//! shared grid, memoization and blob expiry, a views subset, a run without
//! per-feature residuals or covariance, and the refusals.

use serde_json::{Value, json};

use super::test_support::{Demo, golden, slash};
use crate::api::{Api, Conditions, ErrorKind};

/// A plane's blob, read through the `blob` resource.
fn blob(api: &Api, plane: &Value) -> Vec<u8> {
    api.resource(
        "blob",
        json!({ "id": plane["blob"]["id"] }),
        &Conditions::default(),
    )
    .unwrap()
    .read_body()
    .unwrap()
}

fn u32_at(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap())
}

fn f32_at(bytes: &[u8], index: usize) -> f32 {
    let at = 24 + 4 * index;
    f32::from_le_bytes(bytes[at..at + 4].try_into().unwrap())
}

/// A plane's header matches the reply's grid and channel list.
fn check_plane(api: &Api, grid: &Value, plane: &Value) -> Vec<u8> {
    let bytes = blob(api, plane);
    assert_eq!(&bytes[..4], b"VAM1");
    assert_eq!(
        u64::from(u32_at(&bytes, 4)),
        grid["columns"].as_u64().unwrap()
    );
    assert_eq!(u64::from(u32_at(&bytes, 8)), grid["rows"].as_u64().unwrap());
    assert_eq!(
        u64::from(u32_at(&bytes, 12)),
        grid["stride"].as_u64().unwrap()
    );
    let channels = plane["channels"].as_array().unwrap().len();
    assert_eq!(u32_at(&bytes, 16) as usize, channels);
    let nodes = (grid["columns"].as_u64().unwrap() * grid["rows"].as_u64().unwrap()) as usize;
    assert_eq!(bytes.len(), 24 + nodes * channels * 4);
    assert_eq!(plane["blob"]["len"].as_u64().unwrap() as usize, bytes.len());
    bytes
}

#[test]
fn the_maps_of_a_camera_share_one_grid_and_answer_their_blobs() {
    let demo = Demo::calibrated();
    let api = &demo.api;
    let run = demo.run("run-001");

    let distortion = api
        .invoke("distortion_map", json!({ "run": run, "camera": 0 }))
        .unwrap();
    golden("distortion_map", &distortion);
    let grid = distortion["grid"].clone();
    assert_eq!(grid["image_size"], json!([320, 240]));
    assert_eq!(grid["columns"], 64);
    assert_eq!(grid["stride"], 5);
    assert_eq!(distortion["has_sensor_layer"], false);
    for layer in ["distortion", "sensor", "total"] {
        let plane = &distortion[layer];
        let names: Vec<&str> = plane["channels"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| c["name"].as_str().unwrap())
            .collect();
        assert_eq!(names, ["dx", "dy", "magnitude"], "{layer}");
        assert!(
            plane["channels"]
                .as_array()
                .unwrap()
                .iter()
                .all(|c| c["unit"] == "px")
        );
        check_plane(api, &grid, plane);
    }
    // Without a sensor tilt the sensor layer is zero and the total is the
    // lens alone.
    let sensor = blob(api, &distortion["sensor"]);
    let nodes = 64 * 48;
    assert!((0..nodes).all(|i| f32_at(&sensor, i).is_nan() || f32_at(&sensor, i) == 0.0));
    assert_eq!(
        blob(api, &distortion["total"]),
        blob(api, &distortion["distortion"])
    );

    let residual = api
        .invoke("residual_map", json!({ "run": run, "camera": 0 }))
        .unwrap();
    golden("residual_map", &residual);
    assert_eq!(
        residual["grid"], grid,
        "one grid for every map of the camera"
    );
    assert_eq!(residual["chain_level"], false);
    assert_eq!(residual["views"], 6);
    assert_eq!(residual["options"]["min_effective_samples"], 8.0);
    let names: Vec<&str> = residual["plane"]["channels"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| c["name"].as_str().unwrap())
        .collect();
    assert_eq!(
        names,
        [
            "mean_dx",
            "mean_dy",
            "mean_magnitude",
            "rms",
            "std_error",
            "significance",
            "effective_samples",
            "effective_views"
        ]
    );
    check_plane(api, &grid, &residual["plane"]);

    let sigma = api
        .invoke("projection_sigma_map", json!({ "run": run, "camera": 0 }))
        .unwrap();
    golden("projection_sigma_map", &sigma);
    assert_eq!(sigma["status"], "available", "{sigma}");
    assert_eq!(sigma["grid"], grid);
    let bytes = check_plane(api, &grid, &sigma["plane"]);
    let centre = f32_at(&bytes, 24 * 64 + 32);
    assert!(centre.is_finite() && centre > 0.0, "{centre}");

    // A coarser grid is asked for by columns, the same for every map.
    let coarse = api
        .invoke(
            "distortion_map",
            json!({ "run": run, "camera": 0, "columns": 16 }),
        )
        .unwrap();
    assert_eq!(coarse["grid"]["columns"], 16);
    let err = api
        .invoke(
            "residual_map",
            json!({ "run": run, "camera": 0, "columns": 4096 }),
        )
        .unwrap_err();
    assert_eq!(err.kind, ErrorKind::InvalidArgs);
}

#[test]
fn a_map_is_memoized_and_made_again_once_its_blob_expires() {
    let demo = Demo::calibrated();
    let api = &demo.api;
    let args = json!({ "run": demo.run("run-001"), "camera": 0 });
    let first = api.invoke("residual_map", args.clone()).unwrap();
    let again = api.invoke("residual_map", args.clone()).unwrap();
    assert_eq!(first, again);
    let id = first["plane"]["blob"]["id"].as_str().unwrap().to_owned();
    let cached = api
        .resource("blob", json!({ "id": id }), &Conditions::default())
        .unwrap();
    assert_eq!(cached.cache_control, "private, max-age=31536000, immutable");
    assert_eq!(cached.etag, format!("\"{id}\""));

    api.ctx().blobs.forget(&id);
    let err = api
        .resource("blob", json!({ "id": id }), &Conditions::default())
        .unwrap_err();
    assert_eq!(err.kind, ErrorKind::BlobExpired);
    // The command again makes the same blob, under the same id.
    let remade = api.invoke("residual_map", args).unwrap();
    assert_eq!(remade, first);
    assert!(
        api.resource("blob", json!({ "id": id }), &Conditions::default())
            .is_ok()
    );
}

#[test]
fn a_views_subset_maps_those_views_only() {
    let demo = Demo::calibrated();
    let api = &demo.api;
    let run = demo.run("run-001");
    let all = api
        .invoke("residual_map", json!({ "run": run, "camera": 0 }))
        .unwrap();
    let subset = api
        .invoke(
            "residual_map",
            json!({ "run": run, "camera": 0, "options": { "views": [3, 1, 1] } }),
        )
        .unwrap();
    assert_eq!(subset["views"], 2);
    assert_eq!(subset["options"]["views"], json!([1, 3]));
    assert!(subset["used"].as_u64() < all["used"].as_u64());
    assert_ne!(subset["plane"]["blob"]["id"], all["plane"]["blob"]["id"]);
    assert_eq!(subset["bandwidth_selection"]["kind"], "few_views");
    // A view the run does not have, and an option out of range.
    for options in [json!({ "views": [99] }), json!({ "outlier_factor": -1.0 })] {
        let err = api
            .invoke(
                "residual_map",
                json!({ "run": run, "camera": 0, "options": options }),
            )
            .unwrap_err();
        assert_eq!(err.kind, ErrorKind::Invalid, "{options}");
    }
    let err = api
        .invoke(
            "residual_map",
            json!({ "run": run, "camera": 0, "options": { "bandwidth": "wide" } }),
        )
        .unwrap_err();
    assert_eq!(err.kind, ErrorKind::InvalidArgs);
}

#[test]
fn parameters_records_and_statistics_of_a_run() {
    let demo = Demo::calibrated();
    let api = &demo.api;
    let run = demo.run("run-001");

    let parameters = api
        .invoke("camera_parameter_values", json!({ "run": run }))
        .unwrap();
    golden("camera_parameter_values", &parameters);
    let camera = &parameters["cameras"][0];
    let names: Vec<&str> = camera["parameters"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| p["name"].as_str().unwrap())
        .collect();
    assert_eq!(&names[..4], ["fx", "fy", "cx", "cy"]);
    let fx = &camera["parameters"][0];
    assert_eq!(fx["label"], "fx");
    assert_eq!(fx["group"], "intrinsics");
    assert_eq!(fx["sigma"]["kind"], "estimated", "{fx}");
    assert!(fx["sigma"]["sigma"].as_f64().unwrap() > 0.0);
    assert!((fx["value"].as_f64().unwrap() - 300.0).abs() < 10.0, "{fx}");

    let records = api
        .invoke("residual_records", json!({ "run": run, "camera": 0 }))
        .unwrap();
    golden("residual_records", &records);
    let n = records["index"].as_array().unwrap().len();
    assert_eq!(n, 6 * 35);
    for column in [
        "view",
        "feature",
        "target_xyz_m",
        "observed_px",
        "projected_px",
        "residual_px",
    ] {
        assert_eq!(records[column].as_array().unwrap().len(), n, "{column}");
    }
    // observed − projected.
    let o = &records["observed_px"][0];
    let p = &records["projected_px"][0];
    let r = &records["residual_px"][0];
    assert!(
        (r[0].as_f64().unwrap() - (o[0].as_f64().unwrap() - p[0].as_f64().unwrap())).abs() < 1e-12
    );
    let one = api
        .invoke(
            "residual_records",
            json!({ "run": run, "camera": 0, "view": 2 }),
        )
        .unwrap();
    assert_eq!(one["index"].as_array().unwrap().len(), 35);
    assert!(one["view"].as_array().unwrap().iter().all(|v| v == 2));

    let stats = api.invoke("residual_stats", json!({ "run": run })).unwrap();
    golden("residual_stats", &stats);
    assert_eq!(stats["cameras"][0]["stats"]["count"], 6 * 35);
    assert_eq!(stats["cameras"][0]["views"], 6);
    assert_eq!(stats["views"].as_array().unwrap().len(), 6);
    assert_eq!(stats["cells"].as_array().unwrap().len(), 6);
    let rms = stats["cameras"][0]["stats"]["rms"].as_f64().unwrap();
    assert!(rms > 0.0 && rms < 0.5, "{rms}");
}

/// Run `run-002`: run-001's export imported with neither per-feature
/// residuals nor a covariance.
fn bare_run(demo: &Demo) -> Value {
    let export = demo.project.join("runs/run-001/export.json");
    let mut value: Value = serde_json::from_str(&std::fs::read_to_string(export).unwrap()).unwrap();
    value["diagnostics"]["per_feature_residuals"]["target"] = json!([]);
    // Its image manifest names the data from the run folder.
    value["diagnostics"]
        .as_object_mut()
        .unwrap()
        .remove("image_manifest");
    value["uncertainty"] = json!({ "status": "unavailable", "reason": { "kind": "not_computed" } });
    let file = demo.project.join("bare.json");
    std::fs::write(&file, value.to_string()).unwrap();
    let run = demo
        .api
        .invoke(
            "import_export",
            json!({ "dir": demo.dir(), "export": slash(&file) }),
        )
        .unwrap();
    assert_eq!(run["id"], "run-002");
    demo.run("run-002")
}

#[test]
fn a_run_without_residuals_or_covariance_says_what_it_lacks() {
    let demo = Demo::calibrated();
    let api = &demo.api;
    let run = bare_run(&demo);
    // The lens needs neither.
    api.invoke("distortion_map", json!({ "run": run, "camera": 0 }))
        .unwrap();
    let err = api
        .invoke("residual_map", json!({ "run": run, "camera": 0 }))
        .unwrap_err();
    assert_eq!(err.kind, ErrorKind::Invalid);
    assert!(err.message.contains("per-feature residuals"), "{err}");
    let sigma = api
        .invoke("projection_sigma_map", json!({ "run": run, "camera": 0 }))
        .unwrap();
    assert_eq!(
        sigma,
        json!({ "status": "unavailable", "camera": 0,
                "reason": { "kind": "no_covariance", "reason": { "kind": "not_computed" } } })
    );
    let parameters = api
        .invoke("camera_parameter_values", json!({ "run": run }))
        .unwrap();
    assert_eq!(
        parameters["cameras"][0]["parameters"][0]["sigma"]["reason"]["kind"],
        "no_covariance"
    );
    let stats = api.invoke("residual_stats", json!({ "run": run })).unwrap();
    assert_eq!(stats, json!({ "cameras": [], "views": [], "cells": [] }));
}

#[test]
fn unknown_cameras_runs_and_paths_outside_the_roots_are_refused() {
    let demo = Demo::calibrated();
    let api = &demo.api;
    let run = demo.run("run-001");
    for command in [
        "distortion_map",
        "residual_map",
        "projection_sigma_map",
        "residual_records",
    ] {
        let err = api
            .invoke(command, json!({ "run": run, "camera": 1 }))
            .unwrap_err();
        assert_eq!(err.kind, ErrorKind::Invalid, "{command}");
        let err = api
            .invoke(
                command,
                json!({ "run": { "dir": demo.dir(), "id": "run-404" }, "camera": 0 }),
            )
            .unwrap_err();
        assert_eq!(err.kind, ErrorKind::RunNotFound, "{command}");
        let err = api
            .invoke(
                command,
                json!({ "run": { "dir": slash(&demo.outside), "id": "run-001" }, "camera": 0 }),
            )
            .unwrap_err();
        assert_eq!(err.kind, ErrorKind::ForbiddenPath, "{command}");
    }
    // A view the camera has no records in is an empty selection.
    let none = api
        .invoke(
            "residual_records",
            json!({ "run": run, "camera": 0, "view": 99 }),
        )
        .unwrap();
    assert_eq!(none["index"], json!([]));
    // Unknown arguments are refused, as the old `labels`.
    let err = api
        .invoke(
            "camera_parameter_values",
            json!({ "run": run, "labels": ["fx"] }),
        )
        .unwrap_err();
    assert_eq!(err.kind, ErrorKind::InvalidArgs);
}
