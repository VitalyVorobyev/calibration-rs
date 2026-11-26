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
    // The export names each view's image, from the folder it is written to:
    // `out` lies inside the dataset folder, so the root climbs out of it.
    let images = &export["diagnostics"]["image_manifest"];
    assert_eq!(images["root"], "..");
    let frames = images["frames"].as_array().unwrap();
    assert_eq!(frames.len(), VIEWS.len());
    for frame in frames {
        let image = out.join("..").join(frame["path"].as_str().unwrap());
        assert!(image.is_file(), "{}", image.display());
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

/// The rule `vcal run -c` merged a configuration file with before the
/// library's `ConfigPatch` held it: tables merge key by key, any other value
/// replaces the default.
fn former_merge(base: &mut serde_json::Value, changes: &serde_json::Value) {
    match (base, changes) {
        (serde_json::Value::Object(base), serde_json::Value::Object(changes)) => {
            for (key, value) in changes {
                match base.get_mut(key) {
                    Some(slot) => former_merge(slot, value),
                    None => {
                        base.insert(key.clone(), value.clone());
                    }
                }
            }
        }
        (slot, value) => *slot = value.clone(),
    }
}

/// `vcal run -c` writes, byte for byte, the export it wrote when it merged
/// the file itself: the library's patch makes the same configuration.
#[test]
fn a_config_file_gives_the_export_the_former_merge_gave() {
    use vision_calibration::dataset::DatasetSpec;
    use vision_calibration::dataset_runner::{CalibrationConfig, Detection, calibrate_dataset};
    use vision_calibration::detect::FsDetectionCache;

    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    write_views(dir);
    assert_eq!(vcal(&["init", path(dir)]), 0);
    complete_manifest(dir);
    let manifest = dir.join("dataset.toml");
    let text = "solver.max_iters = 30\n\
                camera.model = { projection = \"pinhole\", distortion = \"none\", sensor = \"identity\" }\n\
                [intrinsics_init]\niterations = 3\n";
    let config_path = dir.join("config.toml");
    std::fs::write(&config_path, text).unwrap();
    let out = dir.join("out");
    assert_eq!(
        vcal(&[
            "run",
            path(&manifest),
            "-o",
            path(&out),
            "-c",
            path(&config_path)
        ]),
        0
    );

    // The former route: the defaults as JSON, the file's table merged in.
    let spec = DatasetSpec::from_path(&manifest).unwrap();
    let mut tagged = serde_json::to_value(CalibrationConfig::default_for(spec.topology)).unwrap();
    let changes: serde_json::Value = toml::from_str(text).unwrap();
    former_merge(&mut tagged["config"], &changes);
    let config: CalibrationConfig = serde_json::from_value(tagged).unwrap();
    let cache = FsDetectionCache::new(dir.join(".detections"));
    let mut run = calibrate_dataset(
        &spec,
        dir,
        Some(config),
        Detection::Cached(&cache),
        None,
        &RunControl::default(),
    )
    .unwrap();
    let images = run
        .export
        .diagnostics_mut()
        .image_manifest
        .as_mut()
        .unwrap();
    *images = images.rebased(dir, &out, dir);
    let expected = serde_json::to_string_pretty(&run.export).unwrap() + "\n";
    let written = std::fs::read_to_string(out.join("calibration.json")).unwrap();
    assert!(
        written == expected,
        "calibration.json differs from the former route's export"
    );
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

    // Written outside the dataset folder, the export names the images by
    // their absolute folder.
    let elsewhere = tempfile::tempdir().unwrap();
    assert_eq!(
        vcal(&["run", path(&manifest), "-o", path(elsewhere.path())]),
        0
    );
    let export = read_json(&elsewhere.path().join("calibration.json"));
    let images = &export["diagnostics"]["image_manifest"];
    let root = Path::new(images["root"].as_str().unwrap());
    assert!(root.is_absolute(), "{}", root.display());
    for frame in images["frames"].as_array().unwrap() {
        assert!(root.join(frame["path"].as_str().unwrap()).is_file());
    }
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

// ─────────────────────────────────────────────────────────────────────────────
// Projects
// ─────────────────────────────────────────────────────────────────────────────

/// The eight views and their completed manifest, in `dir`.
fn manifest_folder(dir: &Path) -> std::path::PathBuf {
    std::fs::create_dir_all(dir).unwrap();
    write_views(dir);
    assert_eq!(vcal(&["init", path(dir)]), 0);
    complete_manifest(dir);
    dir.join("dataset.toml")
}

/// A project over a manifest gives the manifest's run: the same export but
/// for the image root restated for the run folder, and the same quality
/// report byte for byte. The manifest's own run is unchanged.
#[test]
fn a_project_run_is_the_manifest_s_run_kept_in_the_project() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    let manifest = manifest_folder(dir);
    let out = dir.join("out");
    assert_eq!(vcal(&["run", path(&manifest), "-o", path(&out)]), 0);

    // A folder without a project file is read as a manifest: an error.
    assert_eq!(vcal(&["run", path(dir)]), 1);
    assert_eq!(
        vcal(&["project", "init", path(dir), "--manifest", path(&manifest)]),
        0
    );
    let project_file = read_toml(&dir.join("calibration-project.toml"));
    assert_eq!(project_file["dataset"]["manifest"], "dataset.toml".into());
    assert_eq!(vcal(&["check", path(dir)]), 0);
    assert_eq!(vcal(&["run", path(dir), "--label", "first"]), 0);
    let run = dir.join("runs/run-001");
    let mut export = read_json(&run.join("export.json"));
    assert_eq!(export["diagnostics"]["image_manifest"]["root"], "../..");
    export["diagnostics"]["image_manifest"]["root"] = "..".into();
    assert_eq!(export, read_json(&out.join("calibration.json")));
    assert_eq!(
        std::fs::read(run.join("quality.json")).unwrap(),
        std::fs::read(out.join("quality.json")).unwrap()
    );
    let record = read_json(&run.join("run.json"));
    assert!(record["tool"].as_str().unwrap().starts_with("vcal "));
    assert_eq!(read_json(&run.join("notes.json"))["label"], "first");

    // The project file decides: its path works as well as its folder.
    assert_eq!(
        vcal(&[
            "run",
            path(&dir.join("calibration-project.toml")),
            "--report"
        ]),
        0
    );
    assert!(dir.join("runs/run-002/report.html").is_file());

    // `-o` is a manifest's, `--label` a project's: usage errors.
    assert_eq!(vcal(&["run", path(dir), "-o", path(&out)]), 2);
    assert_eq!(vcal(&["run", path(&manifest), "--label", "x"]), 2);
    assert!(!dir.join("runs/run-003").exists());
}

fn read_toml(path: &Path) -> toml::Table {
    toml::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

#[test]
fn project_init_drafts_references_or_embeds_and_refuses_a_fork() {
    let tmp = tempfile::tempdir().unwrap();
    let data = tmp.path().join("data");
    std::fs::create_dir_all(&data).unwrap();
    write_views(&data);

    // No manifest: one is drafted into the project file.
    let drafted = tmp.path().join("drafted");
    assert_eq!(
        vcal(&[
            "project",
            "init",
            path(&drafted),
            "--data",
            path(&data),
            "--name",
            "bench",
            "--device-id",
            "SN-1",
            "--device-product",
            "rig"
        ]),
        0
    );
    let file = read_toml(&drafted.join("calibration-project.toml"));
    assert_eq!(file["version"], 2.into());
    assert_eq!(file["name"], "bench".into());
    assert_eq!(file["device"]["id"], "SN-1".into());
    assert_eq!(file["device"]["product"], "rig".into());
    // The draft is outside the project: its root is absolute.
    let root = file["dataset"]["root"].as_str().unwrap();
    assert!(Path::new(root).is_absolute(), "{root}");
    assert!(file["dataset"]["manifest"]["_unresolved"].is_array());
    // The draft leaves the board open: the check fails until it is filled in.
    assert_eq!(vcal(&["check", path(&drafted)]), 1);
    // A second init is refused.
    assert_eq!(
        vcal(&["project", "init", path(&drafted), "--data", path(&data)]),
        1
    );

    // A folder holding a manifest is not drafted over: pass the manifest.
    let manifest = manifest_folder(&tmp.path().join("withmanifest"));
    let folder = manifest.parent().unwrap();
    assert_eq!(vcal(&["project", "init", path(folder)]), 1);
    assert!(!folder.join("calibration-project.toml").exists());

    // Embedded: the manifest's table, its folder as the root.
    let embedded = folder.join("embedded");
    assert_eq!(
        vcal(&[
            "project",
            "init",
            path(&embedded),
            "--manifest",
            path(&manifest),
            "--embed"
        ]),
        0
    );
    let file = read_toml(&embedded.join("calibration-project.toml"));
    assert_eq!(
        file["dataset"]["manifest"]["topology"],
        "planar_intrinsics".into()
    );
    assert_eq!(vcal(&["check", path(&embedded)]), 0);

    // Device options need the id; --embed needs a manifest.
    let other = tmp.path().join("other");
    assert_eq!(
        vcal(&["project", "init", path(&other), "--device-product", "rig"]),
        2
    );
    assert_eq!(vcal(&["project", "init", path(&other), "--embed"]), 2);
}

#[test]
fn project_runs_import_note_and_scan() {
    let tmp = tempfile::tempdir().unwrap();
    let library = tmp.path().join("library");
    let dir = library.join("bench");
    std::fs::create_dir_all(&dir).unwrap();
    let manifest = manifest_folder(&dir);
    assert_eq!(
        vcal(&[
            "project",
            "init",
            path(&dir),
            "--manifest",
            path(&manifest),
            "--device-id",
            "SN-9"
        ]),
        0
    );
    assert_eq!(vcal(&["run", path(&dir)]), 0);
    assert_eq!(vcal(&["project", "runs", path(&dir)]), 0);
    assert_eq!(vcal(&["project", "runs", path(&dir), "--json"]), 0);

    // An export calibrated with `vcal run -o` joins the project.
    let out = tmp.path().join("out");
    assert_eq!(vcal(&["run", path(&manifest), "-o", path(&out)]), 0);
    assert_eq!(
        vcal(&[
            "project",
            "import",
            path(&dir),
            path(&out),
            "--label",
            "outside"
        ]),
        0
    );
    let imported = dir.join("runs/run-002");
    assert_eq!(
        read_json(&imported.join("run.json"))["source"]["kind"],
        "imported"
    );
    assert_eq!(read_json(&imported.join("notes.json"))["label"], "outside");

    assert_eq!(
        vcal(&[
            "project",
            "note",
            path(&dir),
            "run-001",
            "--label",
            "kept",
            "--star"
        ]),
        0
    );
    let notes = read_json(&dir.join("runs/run-001/notes.json"));
    assert_eq!(
        (notes["label"].as_str(), notes["starred"].as_bool()),
        (Some("kept"), Some(true))
    );
    assert_eq!(
        vcal(&[
            "project",
            "note",
            path(&dir),
            "run-001",
            "--unstar",
            "--label",
            ""
        ]),
        0
    );
    assert!(!dir.join("runs/run-001/notes.json").exists());
    assert_eq!(
        vcal(&["project", "note", path(&dir), "run-404", "--star"]),
        1
    );
    assert_eq!(
        vcal(&[
            "project",
            "note",
            path(&dir),
            "run-001",
            "--star",
            "--unstar"
        ]),
        2
    );

    // A scan finishes, a missing root among them, and lists the device.
    let missing = tmp.path().join("missing");
    assert_eq!(
        vcal(&["project", "scan", path(&library), path(&missing)]),
        0
    );
    assert_eq!(
        vcal(&[
            "project",
            "scan",
            path(&library),
            "--json",
            "--max-depth",
            "2"
        ]),
        0
    );
    assert_eq!(vcal(&["project", "scan"]), 2);
    assert_eq!(vcal(&["project", "runs", path(tmp.path())]), 1);
}

/// A project's quality thresholds judge its runs: a stated feature noise far
/// below the residuals fails the variance check, and the run is written
/// with exit status 3. A stale exclusion fails before any detection.
#[test]
fn a_project_s_failed_verdict_exits_3_and_a_stale_exclusion_1() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    let manifest = manifest_folder(dir);
    assert_eq!(
        vcal(&["project", "init", path(dir), "--manifest", path(&manifest)]),
        0
    );
    let file = dir.join("calibration-project.toml");
    let mut text = std::fs::read_to_string(&file).unwrap();
    text.push_str("\n[quality]\nexpected_feature_noise_px = 0.000001\n");
    std::fs::write(&file, &text).unwrap();
    assert_eq!(vcal(&["run", path(dir)]), 3);
    assert_eq!(
        read_json(&dir.join("runs/run-001/quality.json"))["verdict"],
        "fail"
    );

    text.push_str("\n[exclude]\nimages = [{ camera = \"cam0\", image = \"view99.png\" }]\n");
    std::fs::write(&file, &text).unwrap();
    assert_eq!(vcal(&["check", path(dir)]), 1);
    assert_eq!(vcal(&["run", path(dir)]), 1);
    assert!(!dir.join("runs/run-002").exists());
}

/// The repository's datasets each have a project that opens and reads its
/// manifest: `vcal run data/kuka_1` and the rest work from a checkout.
#[test]
fn the_committed_projects_open() {
    use vision_calibration::project::Project;

    let data = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data");
    for dir in [
        "kuka_1",
        "DS8",
        "opencv_stereo",
        "stereo/left",
        "stereo/right",
        "stereo/rig",
    ] {
        let project = Project::open(data.join(dir)).unwrap_or_else(|e| panic!("{dir}: {e}"));
        assert!(project.spec().name.is_some(), "{dir}");
        project.dataset().unwrap_or_else(|e| panic!("{dir}: {e}"));
        assert!(project.data_root().unwrap().is_dir(), "{dir}");
    }
}
