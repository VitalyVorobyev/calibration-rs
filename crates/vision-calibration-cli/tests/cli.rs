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
    render_pose(
        &Rotation3::from_euler_angles(rx, ry, 0.0),
        Vector3::from(offset),
    )
}

/// A pinhole view of the board at `cam_se3_board = (r, t)`, rendered as
/// [`render`] does.
fn render_pose(r: &Rotation3<f64>, t: Vector3<f64>) -> GrayImage {
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

/// The eight tilted views `write_views` renders: `(rx, ry, offset)`.
const VIEWS: [(f64, f64, [f64; 3]); 8] = [
    (0.3, 0.0, [0.03, 0.02, 0.55]),
    (-0.3, 0.0, [-0.03, -0.02, 0.6]),
    (0.0, 0.3, [0.04, -0.02, 0.55]),
    (0.0, -0.3, [-0.04, 0.02, 0.6]),
    (0.25, 0.25, [0.0, 0.03, 0.6]),
    (-0.25, 0.25, [0.03, 0.0, 0.55]),
    (0.25, -0.25, [-0.03, 0.0, 0.6]),
    (-0.25, -0.25, [0.0, -0.03, 0.55]),
];

/// Eight tilted views of the board, in `dir`.
fn write_views(dir: &Path) {
    for (n, (rx, ry, offset)) in VIEWS.into_iter().enumerate() {
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
    let intrinsics = &export["camera"]["intrinsics"];
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
    assert_eq!(export["camera"]["distortion"]["type"], "none");
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

// ─────────────────────────────────────────────────────────────────────────────
// The other workflows
// ─────────────────────────────────────────────────────────────────────────────

/// Baseline of the two-camera rig: camera 1 sits this far along camera 0's x.
const BASELINE_M: f64 = 0.06;

fn read_json(path: &Path) -> serde_json::Value {
    serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
}

/// A rig manifest over `cam0/` and `cam1/` in `dir`, each holding the eight
/// views; camera 1 is camera 0 moved `BASELINE_M` along x.
fn write_rig(dir: &Path) -> std::path::PathBuf {
    for (camera, shift) in [("cam0", 0.0), ("cam1", BASELINE_M)] {
        std::fs::create_dir_all(dir.join(camera)).unwrap();
        for (n, (rx, ry, [x, y, z])) in VIEWS.into_iter().enumerate() {
            render(rx, ry, [x - shift, y, z])
                .save(dir.join(camera).join(format!("view{n:02}.png")))
                .unwrap();
        }
    }
    let manifest = dir.join("rig.toml");
    std::fs::write(
        &manifest,
        r#"version = 1
topology = "rig_extrinsics"
description = "rendered stereo pair"

[[cameras]]
id = "left"
images = { kind = "glob", pattern = "cam0/*.png" }

[[cameras]]
id = "right"
images = { kind = "glob", pattern = "cam1/*.png" }

[target]
kind = "chessboard"
rows = 6
cols = 8
square_size_m = 0.03

[pose_pairing]
kind = "by_index"
"#,
    )
    .unwrap();
    manifest
}

#[test]
fn a_rig_run_recovers_the_baseline_and_writes_the_report() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    let manifest = write_rig(dir);
    assert_eq!(vcal(&["check", path(&manifest)]), 0);

    let out = dir.join("out");
    assert_eq!(
        vcal(&["run", path(&manifest), "-o", path(&out), "--report"]),
        0
    );
    let export = read_json(&out.join("calibration.json"));
    assert_eq!(export["cameras"].as_array().unwrap().len(), 2);
    for camera in export["cameras"].as_array().unwrap() {
        let fx = camera["intrinsics"]["fx"].as_f64().unwrap();
        assert!((fx - FOCAL_PX).abs() < 3.0, "fx = {fx}");
    }
    // cam_se3_rig[1] moves camera 0's frame to camera 1's: BASELINE_M along -x.
    let t = &export["cam_se3_rig"][1]["translation"];
    let t: Vec<f64> = t
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    assert!(
        (t[0] + BASELINE_M).abs() < 0.003,
        "baseline {t:?}, truth {BASELINE_M}"
    );
    assert!(t[1].abs() < 0.003 && t[2].abs() < 0.003, "{t:?}");

    // The report names both cameras by their manifest ids.
    let page = std::fs::read_to_string(out.join("report.html")).unwrap();
    assert!(page.starts_with("<!doctype html>"));
    assert!(page.contains("Camera 0 (left)") && page.contains("Camera 1 (right)"));
    assert!(page.contains("rig_extrinsics: rendered stereo pair"));
    assert_eq!(page.matches("<section class=\"camera\">").count(), 2);
}

/// Ground truth of the hand-eye run: the camera in the gripper frame.
fn tcp_se3_cam() -> vision_calibration::nalgebra::Isometry3<f64> {
    vision_calibration::nalgebra::Isometry3::from_parts(
        Vector3::new(0.02, 0.03, 0.05).into(),
        vision_calibration::nalgebra::UnitQuaternion::from_euler_angles(0.1, -0.2, 0.15),
    )
}

/// A single-camera hand-eye manifest in `dir`: the eight views and one
/// `T_base_tcp` per view, for a camera on the gripper that looks at a board
/// fixed in the base frame (the base frame is the board's).
fn write_handeye(dir: &Path) -> std::path::PathBuf {
    use vision_calibration::nalgebra::{Isometry3, UnitQuaternion};

    let mut poses = String::new();
    for (n, (rx, ry, offset)) in VIEWS.into_iter().enumerate() {
        let r = Rotation3::from_euler_angles(rx, ry, 0.0);
        let t = Vector3::from(offset);
        render_pose(&r, t)
            .save(dir.join(format!("view{n:02}.png")))
            .unwrap();
        let cam_se3_board =
            Isometry3::from_parts(t.into(), UnitQuaternion::from_rotation_matrix(&r));
        // base = board, so base_se3_tcp = board_se3_cam * cam_se3_tcp.
        let base_se3_tcp = cam_se3_board.inverse() * tcp_se3_cam().inverse();
        let m = base_se3_tcp.to_homogeneous();
        let row: Vec<String> = (0..4)
            .flat_map(|i| (0..4).map(move |j| (i, j)))
            .map(|(i, j)| format!("{:.12}", m[(i, j)]))
            .collect();
        poses.push_str(&row.join(" "));
        poses.push('\n');
    }
    std::fs::write(dir.join("poses.txt"), poses).unwrap();
    let manifest = dir.join("handeye.toml");
    std::fs::write(
        &manifest,
        r#"version = 1
topology = "single_cam_handeye"

[[cameras]]
id = "cam0"
images = { kind = "glob", pattern = "view*.png" }

[target]
kind = "chessboard"
rows = 6
cols = 8
square_size_m = 0.03

[robot_poses]
path = "poses.txt"
format = "rowmajor4x4"

[pose_pairing]
kind = "by_index"

[pose_convention]
transform = "t_base_tcp"
rotation_format = "matrix4x4_row_major"
translation_units = "m"
"#,
    )
    .unwrap();
    manifest
}

#[test]
fn a_hand_eye_run_goes_from_a_manifest_to_the_export() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    let manifest = write_handeye(dir);
    assert_eq!(vcal(&["check", path(&manifest)]), 0);

    let out = dir.join("out");
    assert_eq!(vcal(&["run", path(&manifest), "-o", path(&out)]), 0);
    let export = read_json(&out.join("calibration.json"));
    assert_eq!(export["kind"], "single_cam_handeye");
    let fx = export["camera"]["intrinsics"]["fx"].as_f64().unwrap();
    assert!((fx - FOCAL_PX).abs() < 3.0, "fx = {fx}");
    let handeye = &export["handeye"];
    assert_eq!(handeye["mode"], "eye_in_hand");
    let pose = &handeye["gripper_se3_cam"];
    let t: Vec<f64> = pose["translation"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    let truth = tcp_se3_cam().translation.vector;
    for (value, truth) in t.iter().zip(truth.iter()) {
        assert!((value - truth).abs() < 0.005, "{t:?} vs {truth:?}");
    }
    let q: Vec<f64> = pose["rotation"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    let estimated = vision_calibration::nalgebra::UnitQuaternion::from_quaternion(
        vision_calibration::nalgebra::Quaternion::new(q[3], q[0], q[1], q[2]),
    );
    let error = estimated.angle_to(&tcp_se3_cam().rotation);
    assert!(error < 0.05, "hand-eye rotation off by {error} rad");
}

/// A run whose verdict is fail still calibrates and writes its files, and
/// exits with status 3. Views that cover a small part of the image do that.
#[test]
fn a_failed_verdict_exits_3_after_writing_the_files() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    // The board 1.6 m away covers a few percent of the image.
    for (n, (rx, ry, [x, y, _])) in VIEWS.into_iter().enumerate() {
        render(rx, ry, [x, y, 1.6])
            .save(dir.join(format!("view{n:02}.png")))
            .unwrap();
    }
    assert_eq!(vcal(&["init", path(dir)]), 0);
    complete_manifest(dir);
    let out = dir.join("out");
    let manifest = dir.join("dataset.toml");
    assert_eq!(vcal(&["run", path(&manifest), "-o", path(&out)]), 3);
    assert!(out.join("calibration.json").exists());
    let report = read_json(&out.join("quality.json"));
    assert_eq!(report["verdict"], "fail");
}

/// The command line has no laser-line extractor: a laser manifest is an
/// error before any image is read, and nothing is written.
#[test]
fn a_laser_manifest_is_refused_before_detection() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    write_views(dir);
    for n in 0..VIEWS.len() {
        std::fs::copy(
            dir.join(format!("view{n:02}.png")),
            dir.join(format!("laser{n:02}.png")),
        )
        .unwrap();
    }
    let manifest = dir.join("laser.toml");
    std::fs::write(
        &manifest,
        r#"version = 1
topology = "laserline_device"

[[cameras]]
id = "cam0"
images = { kind = "glob", pattern = "view*.png" }
laser_images = { kind = "glob", pattern = "laser*.png" }

[target]
kind = "chessboard"
rows = 6
cols = 8
square_size_m = 0.03

[pose_pairing]
kind = "by_index"
"#,
    )
    .unwrap();
    let out = dir.join("out");
    assert_eq!(vcal(&["run", path(&manifest), "-o", path(&out)]), 1);
    // The error came before detection: no cache, no output.
    assert!(!dir.join(".detections").exists());
    assert!(!out.exists());
}

/// A report that cannot be written leaves none of the three files.
#[test]
fn a_report_that_cannot_be_written_leaves_no_files() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    write_views(dir);
    assert_eq!(vcal(&["init", path(dir)]), 0);
    complete_manifest(dir);
    let manifest = dir.join("dataset.toml");
    let out = dir.join("out");
    // A directory where the report goes.
    std::fs::create_dir_all(out.join("report.html")).unwrap();
    assert_eq!(
        vcal(&["run", path(&manifest), "-o", path(&out), "--report"]),
        1
    );
    let left: Vec<_> = std::fs::read_dir(&out)
        .unwrap()
        .map(|e| e.unwrap().file_name())
        .collect();
    assert_eq!(left, ["report.html"], "{left:?}");
}

/// The extension of a configuration file names its format whatever its case.
#[test]
fn a_config_file_extension_is_matched_ignoring_case() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    write_views(dir);
    assert_eq!(vcal(&["init", path(dir)]), 0);
    complete_manifest(dir);
    let manifest = dir.join("dataset.toml");
    let config = dir.join("CONFIG.TOML");
    std::fs::write(
        &config,
        "camera.model = { projection = \"pinhole\", distortion = \"none\", sensor = \"identity\" }\n",
    )
    .unwrap();
    let out = dir.join("out");
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
    let export = read_json(&out.join("calibration.json"));
    assert_eq!(export["camera"]["distortion"]["type"], "none");
}
