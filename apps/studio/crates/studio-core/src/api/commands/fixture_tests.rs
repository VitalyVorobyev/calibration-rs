//! The e2e fixture project `planar-demo` (`e2e/fixtures/library`): six
//! rendered views of a chessboard by an ideal 320 × 240 camera, as a
//! version 2 project whose manifest is in its file. `write_planar_demo`
//! (ignored) writes it; the other test calibrates the committed copy, so a
//! change that breaks the Playwright run fails here first.

use std::path::{Path, PathBuf};

use serde_json::json;

use crate::api::{Api, Setup};
use crate::scope::RootScope;

/// Inner corners of the board, and its square, metres.
const ROWS: usize = 5;
const COLS: usize = 7;
const SQUARE: f64 = 0.02;
/// The camera: 320 × 240, focal length 300 px, no distortion.
const WIDTH: u32 = 320;
const HEIGHT: u32 = 240;
const F: f64 = 300.0;
/// Each view's board tilt about x and y and turn about z (radians), and
/// its distance (metres).
const VIEWS: [[f64; 4]; 6] = [
    [0.0, 0.0, 0.0, 0.40],
    [0.35, 0.0, 0.05, 0.42],
    [-0.3, 0.1, -0.05, 0.40],
    [0.05, 0.4, 0.1, 0.43],
    [0.1, -0.35, -0.1, 0.41],
    [-0.25, -0.25, 0.15, 0.44],
];

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../e2e/fixtures/library/planar-demo")
}

/// `R = Rz · Ry · Rx`.
fn rotation(rx: f64, ry: f64, rz: f64) -> [[f64; 3]; 3] {
    let (sx, cx) = rx.sin_cos();
    let (sy, cy) = ry.sin_cos();
    let (sz, cz) = rz.sin_cos();
    [
        [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
        [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
        [-sy, cy * sx, cy * cx],
    ]
}

/// The grey level of the board (or the background) seen through pixel
/// `(u, v)`: the ray intersected with the board's plane.
fn shade(r: &[[f64; 3]; 3], t: [f64; 3], u: f64, v: f64) -> f64 {
    let cx = (f64::from(WIDTH) - 1.0) / 2.0;
    let cy = (f64::from(HEIGHT) - 1.0) / 2.0;
    let d = [(u - cx) / F, (v - cy) / F, 1.0];
    // Board point p (z = 0) maps to R·p + t; solve s·d = R·p + t for p.
    // In the board frame: p = Rᵀ(s·d − t), its z = 0 fixes s.
    let rt = |w: [f64; 3]| {
        [
            r[0][0] * w[0] + r[1][0] * w[1] + r[2][0] * w[2],
            r[0][1] * w[0] + r[1][1] * w[1] + r[2][1] * w[2],
            r[0][2] * w[0] + r[1][2] * w[1] + r[2][2] * w[2],
        ]
    };
    let dr = rt(d);
    let tr = rt(t);
    let s = tr[2] / dr[2];
    let x = s * dr[0] - tr[0];
    let y = s * dr[1] - tr[1];
    // Squares run from one square before the first inner corner to one
    // after the last; a white margin of one square around them.
    let (i, j) = ((x / SQUARE).floor() + 1.0, (y / SQUARE).floor() + 1.0);
    let squares = (COLS as f64 + 1.0, ROWS as f64 + 1.0);
    if i < -1.0 || j < -1.0 || i > squares.0 || j > squares.1 {
        return 120.0;
    }
    if i < 0.0 || j < 0.0 || i >= squares.0 || j >= squares.1 {
        return 235.0;
    }
    if (i + j) as i64 % 2 == 0 { 25.0 } else { 235.0 }
}

/// One view, 4 × 4 samples a pixel.
fn render(view: [f64; 4]) -> image::GrayImage {
    let r = rotation(view[0], view[1], view[2]);
    // The board's centre in front of the camera's centre.
    let centre = [
        (COLS as f64 - 1.0) * SQUARE / 2.0,
        (ROWS as f64 - 1.0) * SQUARE / 2.0,
        0.0,
    ];
    let rc = [
        r[0][0] * centre[0] + r[0][1] * centre[1],
        r[1][0] * centre[0] + r[1][1] * centre[1],
        r[2][0] * centre[0] + r[2][1] * centre[1],
    ];
    let t = [-rc[0], -rc[1], view[3] - rc[2]];
    image::GrayImage::from_fn(WIDTH, HEIGHT, |u, v| {
        let mut sum = 0.0;
        for a in 0..4 {
            for b in 0..4 {
                let du = (f64::from(a) + 0.5) / 4.0 - 0.5;
                let dv = (f64::from(b) + 0.5) / 4.0 - 0.5;
                sum += shade(&r, t, f64::from(u) + du, f64::from(v) + dv);
            }
        }
        image::Luma([(sum / 16.0).round() as u8])
    })
}

/// Write `e2e/fixtures/library/planar-demo/` (`cargo test -p studio-core
/// write_planar_demo -- --ignored`).
#[test]
#[ignore = "writes the committed e2e fixture"]
fn write_planar_demo() {
    let dir = fixture_dir();
    std::fs::create_dir_all(dir.join("images")).unwrap();
    for (i, view) in VIEWS.iter().enumerate() {
        render(*view)
            .save(dir.join(format!("images/view{i}.png")))
            .unwrap();
    }
    std::fs::write(
        dir.join("calibration-project.toml"),
        format!(
            "# The e2e fixture: six rendered views of a {COLS} x {ROWS} chessboard\n\
             # (written by studio-core's `write_planar_demo`).\n\n\
             version = 2\n\
             name = \"planar-demo\"\n\n\
             [dataset]\n\
             root = \".\"\n\n\
             [dataset.manifest]\n\
             version = 1\n\
             topology = \"planar_intrinsics\"\n\n\
             [dataset.manifest.target]\n\
             kind = \"chessboard\"\n\
             rows = {ROWS}\n\
             cols = {COLS}\n\
             square_size_m = {SQUARE}\n\n\
             [[dataset.manifest.cameras]]\n\
             id = \"cam0\"\n\n\
             [dataset.manifest.cameras.images]\n\
             kind = \"glob\"\n\
             pattern = \"images/*.png\"\n"
        ),
    )
    .unwrap();
}

/// The committed fixture calibrates: every view detected, a run written
/// with its verdict.
#[test]
fn the_e2e_planar_fixture_calibrates() {
    let tmp = tempfile::tempdir().unwrap();
    let root = dunce::canonicalize(tmp.path()).unwrap();
    let project = root.join("planar-demo");
    std::fs::create_dir_all(project.join("images")).unwrap();
    for entry in std::fs::read_dir(fixture_dir().join("images")).unwrap() {
        let entry = entry.unwrap();
        std::fs::copy(entry.path(), project.join("images").join(entry.file_name())).unwrap();
    }
    std::fs::copy(
        fixture_dir().join("calibration-project.toml"),
        project.join("calibration-project.toml"),
    )
    .unwrap();
    let api = Api::with_scope(
        RootScope::fixed([root.clone()]),
        Setup::for_tests(Some(root.join("cache"))),
    );
    let job = api
        .invoke("calibrate", json!({ "dir": project.to_string_lossy() }))
        .unwrap();
    let mut state = serde_json::Value::Null;
    for _ in 0..1200 {
        state = api
            .invoke("job_status", json!({ "id": job["id"] }))
            .unwrap()["state"]
            .clone();
        if !matches!(state["kind"].as_str(), Some("queued" | "running")) {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    assert_eq!(state["kind"], "done", "{state}");
    let source = &state["output"]["state"]["source"];
    assert_eq!(source["usable_views"], VIEWS.len(), "{state}");
    assert!(state["output"]["state"]["summary"]["verdict"].is_string());
}
