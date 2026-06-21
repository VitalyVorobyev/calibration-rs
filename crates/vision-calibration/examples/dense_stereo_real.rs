//! Real-data evidence for the C5 pure-Rust dense matcher.
//!
//! Runs the full pipeline on the committed `data/stereo` chessboard rig:
//!
//! 1. read the frozen `RigExtrinsics` calibration (`viewer_export.json`) — two
//!    `K` matrices, Brown-Conrady distortion, and the inter-camera pose;
//! 2. undistort + Scheimpflug-aware **rectify** one synchronized pair
//!    (`Im_L_1` / `Im_R_1`) so rows align (Track C4);
//! 3. **dense-match** the rectified pair ([`mvg::dense::match_block`]);
//! 4. write inspectable PNGs and a planarity-fit metric.
//!
//! The target is a planar board, so the *true* disparity over it is an affine
//! function of pixel position. The matcher knows nothing about that — so the
//! residual of a least-squares plane fit to the recovered disparities is a
//! ground-truth-free measure of correctness (a coherent, low-residual planar
//! surface ⇒ the matcher works on real data).
//!
//! ```bash
//! cargo run -p vision-calibration --example dense_stereo_real --release
//! # → target/fixtures/dense_stereo_real/{rectified_pair,disparity,overlay}.png
//! ```

use std::path::{Path, PathBuf};

use nalgebra::{Matrix3, Quaternion, Translation3, UnitQuaternion};
use serde::Deserialize;
use vision_calibration::core::Iso3;
use vision_calibration::mvg::dense::{BlockMatchOptions, DisparityMap, GrayImage, match_block};
use vision_calibration::mvg::rectification::{RectifyCamera, RectifyOptions, rectify_stereo_pair};
use vision_calibration_core::{
    BrownConrady5, Mat3, Pt2, Vec3, distort_to_pixel, pixel_to_normalized,
};

/// Integer downscale factor: the board sits ~0.54 m away with a 0.19 m baseline,
/// so full-res disparities run to ~250 px. Matching at a third of the resolution
/// keeps the disparity search (and the cost volume) modest.
const DOWNSCALE: usize = 3;

fn main() {
    let data_dir = workspace_root().join("data/stereo");
    let export_path = data_dir.join("viewer_export.json");
    if !export_path.exists() {
        eprintln!(
            "data/stereo/viewer_export.json not found at {} — skipping (committed dataset absent)",
            export_path.display()
        );
        return;
    }

    let export: Export =
        serde_json::from_reader(std::fs::File::open(&export_path).expect("open export"))
            .expect("parse RigExtrinsics export");
    assert!(
        export.cameras.len() == 2 && export.cam_se3_rig.len() == 2,
        "expected a two-camera rig export"
    );

    let s = DOWNSCALE as f64;
    let k0 = export.cameras[0].k.matrix(s);
    let k1 = export.cameras[1].k.matrix(s);
    let d0 = export.cameras[0].dist.model();
    let d1 = export.cameras[1].dist.model();
    // cam0 is the rig reference (identity pose), so cam_se3_rig[1] = T_C1_C0.
    let cam1_se3_cam0: Iso3 = export.cam_se3_rig[1].iso();
    // Nominal board depth in cam0 from the first view's target pose.
    let z_board = export.rig_se3_target[0].translation[2];

    // --- load + downscale the synchronized pair ---
    let left_src = load_gray_downscaled(&data_dir.join("imgs/leftcamera/Im_L_1.png"), DOWNSCALE);
    let right_src = load_gray_downscaled(&data_dir.join("imgs/rightcamera/Im_R_1.png"), DOWNSCALE);
    let (w, h) = (left_src.width, left_src.height);

    // --- rectify (Track C4) ---
    let rect = rectify_stereo_pair(
        &RectifyCamera::pinhole(k0),
        &RectifyCamera::pinhole(k1),
        &cam1_se3_cam0,
        &RectifyOptions::default(),
    )
    .expect("rectify stereo pair");

    let rect_left = remap(&left_src, &rect.h_left, &k0, &d0, w, h);
    let rect_right = remap(&right_src, &rect.h_right, &k1, &d1, w, h);

    // --- disparity search window from geometry: d ≈ f_rect · B / Z ---
    // The board is near fronto-parallel, so bracket the nominal disparity
    // tightly: a window much wider than the surface only invites weakly-textured
    // background to false-match at the search edges.
    let f_rect = rect.k_rect[(0, 0)];
    let disp_nominal = (f_rect * rect.baseline / z_board) as f32;
    let min_d = (disp_nominal * 0.6).round().max(0.0) as i32;
    let num_d = (disp_nominal * 0.85).round().max(16.0) as i32;

    let opts = BlockMatchOptions {
        min_disparity: min_d,
        num_disparities: num_d,
        block_size: 11,
        // Real imagery: the high-contrast board boundaries correlate strongly;
        // require a solid peak so low-texture background is rejected.
        min_correlation: 0.5,
        uniqueness_ratio: 0.05,
        ..Default::default()
    };
    let disp = match_block(&rect_left, &rect_right, &opts).expect("dense match");

    // --- metrics ---
    let valid = disp.data.iter().filter(|v| v.is_finite()).count();
    let density = valid as f64 / (w * h) as f64;
    let (lo, hi) = valid_range(&disp).unwrap_or((0.0, 1.0));
    // Robust planar fit: the board dominates, so refitting on inliers reports
    // the board's own planarity, undistracted by any stray background match.
    let (plane_rms, plane_n) = planarity_rms(&disp);

    // --- outputs ---
    let out_dir = workspace_root().join("target/fixtures/dense_stereo_real");
    std::fs::create_dir_all(&out_dir).expect("create output dir");
    save_pair_with_epilines(&out_dir.join("rectified_pair.png"), &rect_left, &rect_right);
    save_disparity(&out_dir.join("disparity.png"), &disp, lo, hi);
    save_overlay(&out_dir.join("overlay.png"), &rect_left, &disp, lo, hi);

    println!("C5 dense matcher — real chessboard rig evidence (data/stereo)");
    println!("  source 1024x576 → matched {w}x{h} (downscale {DOWNSCALE}x)");
    println!(
        "  baseline {:.3} m, board depth {:.3} m, f_rect {:.1} px",
        rect.baseline, z_board, f_rect
    );
    println!("  disparity search {min_d}..{}", min_d + num_d);
    println!(
        "  recovered: {valid} valid px ({:.1}% density), disparity range [{lo:.1}, {hi:.1}] px",
        density * 100.0
    );
    println!(
        "  board plane: {plane_n} inlier px, planarity-fit RMS {plane_rms:.3} px \
         (planar target ⇒ low = coherent surface)"
    );
    println!("  wrote {}", out_dir.display());

    // Evidence the matcher works on real data: it recovers a large, coherent
    // PLANAR surface (the board) — low planarity RMS over many inliers. The raw
    // range legitimately extends below the board: textured background sits at a
    // greater depth (smaller disparity). The board's NEAR edge (max disparity)
    // must not be clipped by the search ceiling.
    let near_edge_clipped = hi >= (min_d + num_d) as f32 - 0.5;
    let pass = plane_n > 3000 && plane_rms < 1.0 && !near_edge_clipped;
    if pass {
        println!(
            "\n  PASS — coherent board plane: {plane_n} inlier px at {plane_rms:.3} px planarity RMS"
        );
    } else {
        eprintln!(
            "\n  CHECK — planarity RMS {plane_rms:.3} px over {plane_n} px, near-edge clipped: {near_edge_clipped} \
             (range [{lo:.1},{hi:.1}] vs search {min_d}..{})",
            min_d + num_d
        );
        std::process::exit(1);
    }
}

// ---------------------------------------------------------------------------
// Export JSON (minimal view — serde ignores the rest)
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct Export {
    cameras: Vec<CamJson>,
    cam_se3_rig: Vec<PoseJson>,
    rig_se3_target: Vec<PoseJson>,
}

#[derive(Deserialize)]
struct CamJson {
    k: KJson,
    dist: DistJson,
}

#[derive(Deserialize)]
struct KJson {
    fx: f64,
    fy: f64,
    cx: f64,
    cy: f64,
    #[serde(default)]
    skew: f64,
}

impl KJson {
    /// Intrinsic matrix scaled for a `scale`× image downscale.
    fn matrix(&self, scale: f64) -> Mat3 {
        Matrix3::new(
            self.fx / scale,
            self.skew / scale,
            self.cx / scale,
            0.0,
            self.fy / scale,
            self.cy / scale,
            0.0,
            0.0,
            1.0,
        )
    }
}

#[derive(Deserialize)]
struct DistJson {
    k1: f64,
    k2: f64,
    k3: f64,
    p1: f64,
    p2: f64,
    #[serde(default = "default_iters")]
    iters: u32,
}

fn default_iters() -> u32 {
    8
}

impl DistJson {
    fn model(&self) -> BrownConrady5<f64> {
        BrownConrady5 {
            k1: self.k1,
            k2: self.k2,
            k3: self.k3,
            p1: self.p1,
            p2: self.p2,
            iters: self.iters,
        }
    }
}

#[derive(Deserialize)]
struct PoseJson {
    /// Quaternion `[x, y, z, w]`.
    rotation: [f64; 4],
    translation: [f64; 3],
}

impl PoseJson {
    fn iso(&self) -> Iso3 {
        let [x, y, z, w] = self.rotation;
        let q = UnitQuaternion::from_quaternion(Quaternion::new(w, x, y, z));
        let t = Translation3::new(
            self.translation[0],
            self.translation[1],
            self.translation[2],
        );
        Iso3::from_parts(t, q)
    }
}

// ---------------------------------------------------------------------------
// Image loading / rectifying remap
// ---------------------------------------------------------------------------

/// Load a PNG as grayscale and box-average downscale by an integer `factor`.
fn load_gray_downscaled(path: &Path, factor: usize) -> GrayImage {
    let img = image::ImageReader::open(path)
        .unwrap_or_else(|e| panic!("open {}: {e}", path.display()))
        .decode()
        .unwrap_or_else(|e| panic!("decode {}: {e}", path.display()))
        .to_luma8();
    let (fw, fh) = (img.width() as usize, img.height() as usize);
    let (w, h) = (fw / factor, fh / factor);
    let mut data = vec![0u8; w * h];
    for oy in 0..h {
        for ox in 0..w {
            let mut acc = 0u32;
            for dy in 0..factor {
                for dx in 0..factor {
                    acc += img
                        .get_pixel((ox * factor + dx) as u32, (oy * factor + dy) as u32)
                        .0[0] as u32;
                }
            }
            data[oy * w + ox] = (acc / (factor * factor) as u32) as u8;
        }
    }
    GrayImage::new(w, h, data)
}

/// Resample a source image into the rectified frame.
///
/// For each rectified output pixel: invert the rectifying homography to the
/// undistorted source pixel, re-apply the lens distortion to find the true
/// (distorted) source sample location, and bilinear-sample. This is the
/// `initUndistortRectifyMap` + `remap` step OpenCV keeps separate from the
/// rectifying rotation.
fn remap(
    src: &GrayImage,
    h_rect: &Mat3,
    k: &Mat3,
    dist: &BrownConrady5<f64>,
    out_w: usize,
    out_h: usize,
) -> GrayImage {
    let h_inv = h_rect
        .try_inverse()
        .expect("rectifying homography invertible");
    let mut data = vec![0u8; out_w * out_h];
    for yr in 0..out_h {
        for xr in 0..out_w {
            // rectified → undistorted source pixel
            let p = h_inv * Vec3::new(xr as f64, yr as f64, 1.0);
            if p.z.abs() < 1e-12 {
                continue;
            }
            let undist = Pt2::new(p.x / p.z, p.y / p.z);
            // undistorted pixel → normalized → distorted source pixel
            let n = pixel_to_normalized(undist, k);
            let src_px = distort_to_pixel(n, k, dist);
            data[yr * out_w + xr] = src_px_sample(src, src_px.x, src_px.y);
        }
    }
    GrayImage::new(out_w, out_h, data)
}

/// Bilinear sample, returning 0 (black) for out-of-frame coordinates.
fn src_px_sample(img: &GrayImage, x: f64, y: f64) -> u8 {
    if x < 0.0 || y < 0.0 || x > (img.width - 1) as f64 || y > (img.height - 1) as f64 {
        return 0;
    }
    img.bilinear(x, y).round().clamp(0.0, 255.0) as u8
}

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

fn valid_range(d: &DisparityMap) -> Option<(f32, f32)> {
    let mut lo = f32::INFINITY;
    let mut hi = f32::NEG_INFINITY;
    let mut any = false;
    for &v in &d.data {
        if v.is_finite() {
            any = true;
            lo = lo.min(v);
            hi = hi.max(v);
        }
    }
    any.then_some((lo, hi))
}

/// Robust least-squares fit `d ≈ a·x + b·y + c`.
///
/// Fits over all valid pixels, then refits once over the inliers (residual
/// within 3 px) so a handful of stray background matches cannot distort the
/// board's reported planarity. Returns `(inlier_rms_px, inlier_count)`.
fn planarity_rms(d: &DisparityMap) -> (f64, usize) {
    let samples: Vec<(f64, f64, f64)> = (0..d.height)
        .flat_map(|y| (0..d.width).map(move |x| (x, y)))
        .filter_map(|(x, y)| {
            let v = d.get(x, y);
            v.is_finite().then_some((x as f64, y as f64, v as f64))
        })
        .collect();
    if samples.len() < 3 {
        return (f64::INFINITY, samples.len());
    }

    let fit = |pts: &[(f64, f64, f64)]| -> Option<Vec3> {
        let mut ata = Matrix3::<f64>::zeros();
        let mut atb = Vec3::zeros();
        for &(x, y, v) in pts {
            let f = Vec3::new(x, y, 1.0);
            ata += f * f.transpose();
            atb += f * v;
        }
        ata.try_inverse().map(|inv| inv * atb)
    };

    let Some(coef) = fit(&samples) else {
        return (f64::INFINITY, samples.len());
    };
    let resid = |c: &Vec3, x: f64, y: f64, v: f64| (v - (c.x * x + c.y * y + c.z)).abs();

    // Refit on inliers (within 3 px of the first fit).
    let inliers: Vec<(f64, f64, f64)> = samples
        .iter()
        .copied()
        .filter(|&(x, y, v)| resid(&coef, x, y, v) < 3.0)
        .collect();
    let pts = if inliers.len() >= 3 {
        &inliers
    } else {
        &samples
    };
    let Some(coef) = fit(pts) else {
        return (f64::INFINITY, pts.len());
    };

    let sum_sq: f64 = pts
        .iter()
        .map(|&(x, y, v)| resid(&coef, x, y, v).powi(2))
        .sum();
    ((sum_sq / pts.len() as f64).sqrt(), pts.len())
}

// ---------------------------------------------------------------------------
// Visualization
// ---------------------------------------------------------------------------

/// Jet colormap, `t` in `[0, 1]` → `[r, g, b]`.
fn jet(t: f32) -> [u8; 3] {
    let t = t.clamp(0.0, 1.0);
    let r = ((1.5 - (4.0 * t - 3.0).abs()).clamp(0.0, 1.0) * 255.0) as u8;
    let g = ((1.5 - (4.0 * t - 2.0).abs()).clamp(0.0, 1.0) * 255.0) as u8;
    let b = ((1.5 - (4.0 * t - 1.0).abs()).clamp(0.0, 1.0) * 255.0) as u8;
    [r, g, b]
}

/// Save the rectified pair side-by-side with shared horizontal epipolar lines:
/// a feature at row `y` on the left must sit at row `y` on the right.
fn save_pair_with_epilines(path: &Path, left: &GrayImage, right: &GrayImage) {
    let (w, h) = (left.width, left.height);
    let gap = 8usize;
    let total_w = w * 2 + gap;
    let mut rgb = vec![[255u8, 255, 255]; total_w * h];
    for y in 0..h {
        for x in 0..w {
            let l = left.get(x, y);
            rgb[y * total_w + x] = [l, l, l];
            let r = right.get(x, y);
            rgb[y * total_w + w + gap + x] = [r, r, r];
        }
    }
    // Epipolar lines every ~h/12 rows.
    let step = (h / 12).max(1);
    let mut yy = step;
    while yy < h {
        for x in 0..total_w {
            rgb[yy * total_w + x] = [40, 220, 90];
        }
        yy += step;
    }
    save_rgb(path, total_w, h, &rgb);
}

/// Save the disparity map as a jet colormap (invalid → black).
fn save_disparity(path: &Path, d: &DisparityMap, lo: f32, hi: f32) {
    let span = (hi - lo).max(1e-6);
    let rgb: Vec<[u8; 3]> = d
        .data
        .iter()
        .map(|&v| {
            if v.is_finite() {
                jet((v - lo) / span)
            } else {
                [0, 0, 0]
            }
        })
        .collect();
    save_rgb(path, d.width, d.height, &rgb);
}

/// Save the disparity colormap alpha-blended over the rectified left image.
fn save_overlay(path: &Path, left: &GrayImage, d: &DisparityMap, lo: f32, hi: f32) {
    let span = (hi - lo).max(1e-6);
    let rgb: Vec<[u8; 3]> = left
        .data
        .iter()
        .zip(&d.data)
        .map(|(&g8, &v)| {
            let g = g8 as f32;
            if v.is_finite() {
                let c = jet((v - lo) / span);
                // 65% disparity color, 35% underlying grayscale.
                [
                    (0.65 * c[0] as f32 + 0.35 * g) as u8,
                    (0.65 * c[1] as f32 + 0.35 * g) as u8,
                    (0.65 * c[2] as f32 + 0.35 * g) as u8,
                ]
            } else {
                [g8, g8, g8]
            }
        })
        .collect();
    save_rgb(path, left.width, left.height, &rgb);
}

fn save_rgb(path: &Path, w: usize, h: usize, rgb: &[[u8; 3]]) {
    let flat: Vec<u8> = rgb.iter().flat_map(|p| p.iter().copied()).collect();
    image::RgbImage::from_raw(w as u32, h as u32, flat)
        .expect("rgb buffer size")
        .save(path)
        .expect("write png");
}

fn workspace_root() -> PathBuf {
    // CARGO_MANIFEST_DIR = <root>/crates/vision-calibration
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("workspace root above crates/<crate>")
        .to_path_buf()
}
