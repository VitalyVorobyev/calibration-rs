//! `vcal` end to end, in process: each test runs the commands on files in a
//! temporary directory and reads what they wrote.

use std::path::Path;

use image::GrayImage;
use vision_calibration::control::{CancelToken, RunControl};
use vision_calibration::nalgebra::{Matrix3, Rotation3, Vector3};
use vision_calibration_cli::{INTERRUPTED, run};

/// Interior corners of the rendered chessboard, across and down.
const COLS: u32 = 8;
const ROWS: u32 = 6;
const SQUARE_M: f64 = 0.03;
const WIDTH: u32 = 640;
const HEIGHT: u32 = 480;
const FOCAL_PX: f64 = 600.0;

fn vcal(args: &[&str]) -> u8 {
    vcal_with(args, &RunControl::default())
}

fn vcal_with(args: &[&str], control: &RunControl) -> u8 {
    run(std::iter::once("vcal").chain(args.iter().copied()), control)
}

fn path(p: &Path) -> &str {
    p.to_str().expect("temporary paths are UTF-8")
}

/// A pinhole view of the board, tilted by `rx`, `ry` (radians) about the
/// board's x and y axes and centred at `offset` metres in front of the
/// camera, rendered with 4 × 4 supersampling on a grey background.
fn render(rx: f64, ry: f64, offset: [f64; 3]) -> GrayImage {
    let k = Matrix3::new(
        FOCAL_PX,
        0.0,
        f64::from(WIDTH) / 2.0,
        0.0,
        FOCAL_PX,
        f64::from(HEIGHT) / 2.0,
        0.0,
        0.0,
        1.0,
    );
    let r = Rotation3::from_euler_angles(rx, ry, 0.0);
    let t = Vector3::from(offset);
    // Board plane (z = 0, origin at its centre) to pixels.
    let h =
        k * Matrix3::from_columns(&[r.matrix().column(0).into(), r.matrix().column(1).into(), t]);
    let h_inv = h.try_inverse().expect("a board in front of the camera");
    let (squares_x, squares_y) = (COLS + 1, ROWS + 1);
    let (half_w, half_h) = (
        f64::from(squares_x) * SQUARE_M / 2.0,
        f64::from(squares_y) * SQUARE_M / 2.0,
    );
    let shade = |u: f64, v: f64| -> f64 {
        let p = h_inv * Vector3::new(u, v, 1.0);
        let (x, y) = (p.x / p.z + half_w, p.y / p.z + half_h);
        let (i, j) = ((x / SQUARE_M).floor(), (y / SQUARE_M).floor());
        let on_board =
            (0.0..f64::from(squares_x)).contains(&i) && (0.0..f64::from(squares_y)).contains(&j);
        let on_margin = (-1.0..=f64::from(squares_x)).contains(&i)
            && (-1.0..=f64::from(squares_y)).contains(&j);
        match (on_board, on_margin) {
            (true, _) if (i + j) % 2.0 == 0.0 => 20.0,
            (_, true) => 235.0,
            _ => 128.0,
        }
    };
    GrayImage::from_fn(WIDTH, HEIGHT, |px, py| {
        let mut sum = 0.0;
        for sy in 0..4 {
            for sx in 0..4 {
                let u = f64::from(px) + (f64::from(sx) + 0.5) / 4.0;
                let v = f64::from(py) + (f64::from(sy) + 0.5) / 4.0;
                sum += shade(u, v);
            }
        }
        image::Luma([(sum / 16.0).round() as u8])
    })
}

/// Eight tilted views of the board, in `dir`.
fn write_views(dir: &Path) {
    let views = [
        (0.3, 0.0, [0.03, 0.02, 0.55]),
        (-0.3, 0.0, [-0.03, -0.02, 0.6]),
        (0.0, 0.3, [0.04, -0.02, 0.55]),
        (0.0, -0.3, [-0.04, 0.02, 0.6]),
        (0.25, 0.25, [0.0, 0.03, 0.6]),
        (-0.25, 0.25, [0.03, 0.0, 0.55]),
        (0.25, -0.25, [-0.03, 0.0, 0.6]),
        (-0.25, -0.25, [0.0, -0.03, 0.55]),
    ];
    for (n, (rx, ry, offset)) in views.into_iter().enumerate() {
        render(rx, ry, offset)
            .save(dir.join(format!("view{n:02}.png")))
            .unwrap();
    }
}

/// `vcal init`'s draft with the board filled in and `_unresolved` cleared.
fn complete_manifest(dir: &Path) {
    let manifest = dir.join("dataset.toml");
    let draft = std::fs::read_to_string(&manifest).unwrap();
    let mut spec: toml::Table = toml::from_str(&draft).unwrap();
    spec.remove("_unresolved");
    spec.insert(
        "target".into(),
        toml::toml! {
            kind = "chessboard"
            rows = 6
            cols = 8
            square_size_m = 0.03
        }
        .into(),
    );
    std::fs::write(&manifest, toml::to_string(&spec).unwrap()).unwrap();
}

#[test]
fn init_check_run_calibrates_a_folder() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    write_views(dir);
    let manifest = dir.join("dataset.toml");

    assert_eq!(vcal(&["init", path(dir)]), 0);
    assert!(manifest.exists());
    // The draft leaves the board open, and a second init keeps it.
    assert_eq!(vcal(&["check", path(&manifest)]), 1);
    assert_eq!(vcal(&["init", path(dir)]), 1);
    assert_eq!(vcal(&["init", path(dir), "--force"]), 0);

    complete_manifest(dir);
    assert_eq!(vcal(&["check", path(&manifest)]), 0);

    let out = dir.join("out");
    assert_eq!(vcal(&["run", path(&manifest), "-o", path(&out)]), 0);
    let export: serde_json::Value =
        serde_json::from_slice(&std::fs::read(out.join("calibration.json")).unwrap()).unwrap();
    let intrinsics = &export["params"]["camera"]["intrinsics"];
    for (name, truth) in [
        ("fx", FOCAL_PX),
        ("fy", FOCAL_PX),
        ("cx", f64::from(WIDTH) / 2.0),
        ("cy", f64::from(HEIGHT) / 2.0),
    ] {
        let value = intrinsics[name].as_f64().unwrap();
        assert!(
            (value - truth).abs() < 2.0,
            "{name} = {value}, truth {truth}"
        );
    }
    let report: serde_json::Value =
        serde_json::from_slice(&std::fs::read(out.join("quality.json")).unwrap()).unwrap();
    assert_ne!(report["verdict"], "fail");
    // The detection cache sits next to the manifest.
    assert!(dir.join(".detections").is_dir());

    // A configuration file changes the defaults it names: here the camera
    // model, so the export carries no distortion.
    let config = dir.join("config.toml");
    std::fs::write(
        &config,
        "camera.model = { projection = \"pinhole\", distortion = \"none\", sensor = \"identity\" }\n",
    )
    .unwrap();
    let out = dir.join("out_config");
    assert_eq!(
        vcal(&[
            "run",
            path(&manifest),
            "-o",
            path(&out),
            "-c",
            path(&config)
        ]),
        0
    );
    let export: serde_json::Value =
        serde_json::from_slice(&std::fs::read(out.join("calibration.json")).unwrap()).unwrap();
    assert_eq!(export["params"]["camera"]["distortion"]["type"], "none");
}

/// A cancelled run (the binary's Ctrl-C handler cancels the token) exits
/// with the interrupt status and writes nothing, whether it stops in
/// detection or in a solve.
#[test]
fn a_cancelled_run_exits_130_and_writes_nothing() {
    use vision_calibration::control::Progress;

    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    write_views(dir);
    assert_eq!(vcal(&["init", path(dir)]), 0);
    complete_manifest(dir);
    let manifest = dir.join("dataset.toml");

    for stop_at_iteration in [false, true] {
        let token = CancelToken::new();
        let trigger = token.clone();
        let control = RunControl::new()
            .with_cancel(token)
            .with_observer(move |progress| match progress {
                Progress::Detection { done: 2, .. } if !stop_at_iteration => trigger.cancel(),
                Progress::Iteration { iter: 2, .. } if stop_at_iteration => trigger.cancel(),
                _ => {}
            });
        let out = dir.join(format!("out_{stop_at_iteration}"));
        assert_eq!(
            vcal_with(&["run", path(&manifest), "-o", path(&out)], &control),
            INTERRUPTED
        );
        assert!(!out.join("calibration.json").exists());
    }
}

#[test]
fn check_rejects_a_camera_with_mixed_image_sizes() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    write_views(dir);
    GrayImage::new(320, 240)
        .save(dir.join("view99.png"))
        .unwrap();
    assert_eq!(vcal(&["init", path(dir)]), 0);
    complete_manifest(dir);
    assert_eq!(vcal(&["check", path(&dir.join("dataset.toml"))]), 1);
}

#[test]
fn a_config_that_names_an_unknown_field_is_an_error() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    write_views(dir);
    assert_eq!(vcal(&["init", path(dir)]), 0);
    complete_manifest(dir);
    let config = dir.join("config.json");
    std::fs::write(&config, r#"{"camera": {"modle": {}}}"#).unwrap();
    let manifest = dir.join("dataset.toml");
    assert_eq!(vcal(&["run", path(&manifest), "-c", path(&config)]), 1);
    assert!(!dir.join("calibration.json").exists());
}

#[test]
fn usage_errors_and_listings() {
    assert_eq!(vcal(&[]), 2);
    assert_eq!(vcal(&["calibrate"]), 2);
    assert_eq!(vcal(&["--version"]), 0);
    assert_eq!(vcal(&["sample", "list"]), 0);
    let tmp = tempfile::tempdir().unwrap();
    assert_eq!(
        vcal(&[
            "sample",
            "fetch",
            "no_such_sample",
            "--cache",
            path(tmp.path())
        ]),
        1
    );
}
