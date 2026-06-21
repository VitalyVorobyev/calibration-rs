//! Server-side dense stereo matching for the Depth workspace.
//!
//! Given a synchronized stereo pair and the loaded rig calibration, this
//! rectifies the pair (Track C4), dense-matches it
//! ([`vision_calibration::mvg::dense::match_block`], block or semi-global), and
//! returns colormapped PNG data URLs plus quality metrics. Math lives here, not
//! in the frontend, so rectification + distortion use the canonical
//! [`vision_calibration`] models — the same reason the epipolar overlay is
//! server-side.

use std::io::Cursor;
use std::path::{Path, PathBuf};

use base64::Engine;
use nalgebra::Matrix3;
use serde::{Deserialize, Serialize};
use tauri::State;
use vision_calibration::core::{BrownConrady5, FxFyCxCySkew, Iso3};
use vision_calibration::mvg::dense::{BlockMatchOptions, DisparityMap, GrayImage, match_block};
use vision_calibration::mvg::rectification::{RectifyCamera, RectifyOptions, rectify_stereo_pair};
use vision_calibration_core::{Mat3, Pt2, Vec3, distort_to_pixel, pixel_to_normalized};

use crate::export_cache::ExportCache;

/// Frontend-supplied matching controls.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DisparityParams {
    /// Matching-window side length (odd). 11 is a good default for real imagery.
    pub block_size: usize,
    /// Use semi-global aggregation (fills low-texture regions) vs block matching.
    pub semi_global: bool,
    /// Integer image downscale before matching (keeps the search tractable).
    pub downscale: usize,
}

/// Result returned to the Depth workspace. All images are `data:image/png`
/// base64 URLs the webview can drop straight into `<img>`.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DisparityResult {
    /// Rectified left | right with shared epipolar rows (rectification sanity).
    pub rectified_pair_png: String,
    /// Disparity colormap (jet; invalid pixels black).
    pub disparity_png: String,
    /// Disparity colormap blended over the rectified left image.
    pub overlay_png: String,
    /// Matched (downscaled) image width.
    pub width: usize,
    /// Matched (downscaled) image height.
    pub height: usize,
    /// Fraction of pixels with a valid disparity.
    pub density: f64,
    /// Min / max recovered disparity (px).
    pub disp_min: f32,
    /// See [`DisparityResult::disp_min`].
    pub disp_max: f32,
    /// Robust planar-fit RMS over the dominant surface (px) — low ⇒ coherent.
    pub plane_rms: f64,
    /// Inlier pixel count of that planar fit.
    pub plane_inliers: usize,
    /// Stereo baseline in calibration units (m).
    pub baseline_m: f64,
    /// Whether semi-global aggregation was used.
    pub semi_global: bool,
}

/// Tauri command: compute a disparity map for the `(cam_a, cam_b)` pair at
/// `pose` and return colormapped PNGs + metrics.
#[tauri::command]
pub async fn compute_disparity(
    path_a: String,
    path_b: String,
    cam_a: usize,
    cam_b: usize,
    pose: usize,
    params: DisparityParams,
    cache: State<'_, ExportCache>,
) -> Result<DisparityResult, String> {
    let (pa, pb) = (PathBuf::from(&path_a), PathBuf::from(&path_b));
    cache
        .read(|cached| compute(&cached.value, &pa, &pb, cam_a, cam_b, pose, &params))
        .ok_or_else(|| "no export loaded yet".to_string())?
}

/// Core pipeline: parse calibration → rectify → match → render.
fn compute(
    export: &serde_json::Value,
    path_a: &Path,
    path_b: &Path,
    cam_a: usize,
    cam_b: usize,
    pose: usize,
    params: &DisparityParams,
) -> Result<DisparityResult, String> {
    if params.block_size == 0 || params.block_size.is_multiple_of(2) {
        return Err("block_size must be odd".to_string());
    }
    let s = params.downscale.max(1);

    let cam_se3_rig: Vec<Iso3> = serde_json::from_value(
        export
            .get("cam_se3_rig")
            .ok_or("export has no `cam_se3_rig` (rig exports only)")?
            .clone(),
    )
    .map_err(|e| format!("cam_se3_rig decode: {e}"))?;
    let n_cams = export
        .get("cameras")
        .and_then(|v| v.as_array())
        .map(|a| a.len())
        .ok_or("export has no `cameras` array")?;
    if cam_a >= n_cams || cam_b >= n_cams || cam_a == cam_b {
        return Err(format!(
            "invalid camera pair (cam_a={cam_a}, cam_b={cam_b}, num_cameras={n_cams})"
        ));
    }

    let (k_a, dist_a) = parse_camera(export, cam_a, s as f64)?;
    let (k_b, dist_b) = parse_camera(export, cam_b, s as f64)?;
    // Left = cam_a (reference); right = cam_b. T_Cb_Ca composes the two extrinsics.
    let cam1_se3_cam0 = cam_se3_rig[cam_b] * cam_se3_rig[cam_a].inverse();

    // Nominal board depth in the reference frame (for the disparity search).
    let z_board = export
        .get("rig_se3_target")
        .and_then(|v| serde_json::from_value::<Vec<Iso3>>(v.clone()).ok())
        .and_then(|poses| poses.get(pose).map(|p| p.translation.z))
        .filter(|z| z.is_finite() && *z > 0.05)
        .unwrap_or(1.0);

    let left = load_gray_downscaled(path_a, s)?;
    let right = load_gray_downscaled(path_b, s)?;
    let (w, h) = (left.width, left.height);

    let rect = rectify_stereo_pair(
        &RectifyCamera::pinhole(k_a),
        &RectifyCamera::pinhole(k_b),
        &cam1_se3_cam0,
        &RectifyOptions::default(),
    )
    .map_err(|e| format!("rectify: {e}"))?;

    let rect_left = remap(&left, &rect.h_left, &k_a, &dist_a, w, h);
    let rect_right = remap(&right, &rect.h_right, &k_b, &dist_b, w, h);

    let f_rect = rect.k_rect[(0, 0)];
    let disp_nominal = (f_rect * rect.baseline / z_board) as f32;
    let min_d = (disp_nominal * 0.6).round().max(0.0) as i32;
    let num_d = (disp_nominal * 0.85).round().max(16.0) as i32;

    let mut opts = BlockMatchOptions {
        min_disparity: min_d,
        num_disparities: num_d,
        block_size: params.block_size,
        ..Default::default()
    };
    if params.semi_global {
        opts.semi_global = true;
    } else {
        opts.min_correlation = 0.5;
        opts.uniqueness_ratio = 0.05;
    }
    let disp = match_block(&rect_left, &rect_right, &opts).map_err(|e| format!("match: {e}"))?;

    let valid = disp.data.iter().filter(|v| v.is_finite()).count();
    let density = valid as f64 / (w * h) as f64;
    let (lo, hi) = valid_range(&disp).unwrap_or((0.0, 1.0));
    let (plane_rms, plane_inliers) = planarity_rms(&disp);

    Ok(DisparityResult {
        rectified_pair_png: pair_with_epilines_url(&rect_left, &rect_right)?,
        disparity_png: disparity_url(&disp, lo, hi)?,
        overlay_png: overlay_url(&rect_left, &disp, lo, hi)?,
        width: w,
        height: h,
        density,
        disp_min: lo,
        disp_max: hi,
        plane_rms,
        plane_inliers,
        baseline_m: rect.baseline,
        semi_global: params.semi_global,
    })
}

// ---------------------------------------------------------------------------
// Calibration parsing
// ---------------------------------------------------------------------------

/// Parse camera `idx`'s intrinsics (scaled for a `scale`× downscale) + distortion.
fn parse_camera(
    export: &serde_json::Value,
    idx: usize,
    scale: f64,
) -> Result<(Mat3, BrownConrady5<f64>), String> {
    let cam = export
        .get("cameras")
        .and_then(|v| v.as_array())
        .and_then(|a| a.get(idx))
        .ok_or("camera index out of range")?;
    let k: FxFyCxCySkew<f64> =
        serde_json::from_value(cam.get("k").ok_or("camera has no `k`")?.clone())
            .map_err(|e| format!("camera.k decode: {e}"))?;
    let dist: BrownConrady5<f64> =
        serde_json::from_value(cam.get("dist").ok_or("camera has no `dist`")?.clone())
            .map_err(|e| format!("camera.dist decode: {e}"))?;
    // Distortion coefficients are in normalized coordinates, so they survive the
    // image downscale unchanged; only K scales.
    let mat = Matrix3::new(
        k.fx / scale,
        k.skew / scale,
        k.cx / scale,
        0.0,
        k.fy / scale,
        k.cy / scale,
        0.0,
        0.0,
        1.0,
    );
    Ok((mat, dist))
}

// ---------------------------------------------------------------------------
// Image loading / rectifying remap
// ---------------------------------------------------------------------------

fn load_gray_downscaled(path: &Path, factor: usize) -> Result<GrayImage, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let img = image::load_from_memory(&bytes)
        .map_err(|e| format!("decode {}: {e}", path.display()))?
        .to_luma8();
    let (fw, fh) = (img.width() as usize, img.height() as usize);
    let (w, h) = (fw / factor, fh / factor);
    if w == 0 || h == 0 {
        return Err("image too small for the chosen downscale".to_string());
    }
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
    Ok(GrayImage::new(w, h, data))
}

/// Resample a source image into the rectified frame (undistort + rectify).
fn remap(
    src: &GrayImage,
    h_rect: &Mat3,
    k: &Mat3,
    dist: &BrownConrady5<f64>,
    out_w: usize,
    out_h: usize,
) -> GrayImage {
    let h_inv = h_rect.try_inverse().unwrap_or_else(Matrix3::identity);
    let mut data = vec![0u8; out_w * out_h];
    for yr in 0..out_h {
        for xr in 0..out_w {
            let p = h_inv * Vec3::new(xr as f64, yr as f64, 1.0);
            if p.z.abs() < 1e-12 {
                continue;
            }
            let undist = Pt2::new(p.x / p.z, p.y / p.z);
            let nrm = pixel_to_normalized(undist, k);
            let src_px = distort_to_pixel(nrm, k, dist);
            data[yr * out_w + xr] = sample(src, src_px.x, src_px.y);
        }
    }
    GrayImage::new(out_w, out_h, data)
}

fn sample(img: &GrayImage, x: f64, y: f64) -> u8 {
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

/// Robust planar fit `d ≈ a·x + b·y + c`, refit once on inliers (≤3 px).
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
    let resid = |c: &Vec3, x: f64, y: f64, v: f64| (v - (c.x * x + c.y * y + c.z)).abs();
    let Some(c0) = fit(&samples) else {
        return (f64::INFINITY, samples.len());
    };
    let inliers: Vec<(f64, f64, f64)> = samples
        .iter()
        .copied()
        .filter(|&(x, y, v)| resid(&c0, x, y, v) < 3.0)
        .collect();
    let pts = if inliers.len() >= 3 {
        &inliers
    } else {
        &samples
    };
    let Some(c) = fit(pts) else {
        return (f64::INFINITY, pts.len());
    };
    let sum_sq: f64 = pts
        .iter()
        .map(|&(x, y, v)| resid(&c, x, y, v).powi(2))
        .sum();
    ((sum_sq / pts.len() as f64).sqrt(), pts.len())
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

/// Jet colormap, `t` in `[0, 1]` → `[r, g, b]`.
fn jet(t: f32) -> [u8; 3] {
    let t = t.clamp(0.0, 1.0);
    let r = ((1.5 - (4.0 * t - 3.0).abs()).clamp(0.0, 1.0) * 255.0) as u8;
    let g = ((1.5 - (4.0 * t - 2.0).abs()).clamp(0.0, 1.0) * 255.0) as u8;
    let b = ((1.5 - (4.0 * t - 1.0).abs()).clamp(0.0, 1.0) * 255.0) as u8;
    [r, g, b]
}

fn disparity_url(d: &DisparityMap, lo: f32, hi: f32) -> Result<String, String> {
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
    rgb_to_url(d.width, d.height, &rgb)
}

fn overlay_url(left: &GrayImage, d: &DisparityMap, lo: f32, hi: f32) -> Result<String, String> {
    let span = (hi - lo).max(1e-6);
    let rgb: Vec<[u8; 3]> = left
        .data
        .iter()
        .zip(&d.data)
        .map(|(&g8, &v)| {
            let g = g8 as f32;
            if v.is_finite() {
                let c = jet((v - lo) / span);
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
    rgb_to_url(left.width, left.height, &rgb)
}

/// Rectified left | right side-by-side with shared horizontal epipolar lines.
fn pair_with_epilines_url(left: &GrayImage, right: &GrayImage) -> Result<String, String> {
    let (w, h) = (left.width, left.height);
    let gap = 6usize;
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
    let step = (h / 12).max(1);
    let mut yy = step;
    while yy < h {
        for x in 0..total_w {
            rgb[yy * total_w + x] = [40, 220, 90];
        }
        yy += step;
    }
    rgb_to_url(total_w, h, &rgb)
}

fn rgb_to_url(w: usize, h: usize, rgb: &[[u8; 3]]) -> Result<String, String> {
    let flat: Vec<u8> = rgb.iter().flat_map(|p| p.iter().copied()).collect();
    let img =
        image::RgbImage::from_raw(w as u32, h as u32, flat).ok_or("rgb buffer size mismatch")?;
    let mut cursor = Cursor::new(Vec::new());
    image::DynamicImage::ImageRgb8(img)
        .write_to(&mut cursor, image::ImageFormat::Png)
        .map_err(|e| format!("encode PNG: {e}"))?;
    let b64 = base64::engine::general_purpose::STANDARD.encode(cursor.into_inner());
    Ok(format!("data:image/png;base64,{b64}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// End-to-end: the actual command pipeline on the committed `data/stereo`
    /// rig must rectify, dense-match, and recover a coherent planar board.
    /// Skips gracefully if the committed dataset is absent.
    #[test]
    fn compute_disparity_recovers_board_on_committed_rig() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/stereo");
        let export_path = dir.join("viewer_export.json");
        if !export_path.exists() {
            eprintln!("skipping: {} absent", export_path.display());
            return;
        }
        let export: serde_json::Value =
            serde_json::from_reader(std::fs::File::open(&export_path).unwrap()).unwrap();
        let pa = dir.join("imgs/leftcamera/Im_L_1.png");
        let pb = dir.join("imgs/rightcamera/Im_R_1.png");
        let params = DisparityParams {
            block_size: 11,
            semi_global: true,
            downscale: 4,
        };
        let res = compute(&export, &pa, &pb, 0, 1, 0, &params).expect("compute disparity");

        assert!(res.density > 0.1, "density too low: {}", res.density);
        assert!(
            res.plane_inliers > 1000,
            "few inliers: {}",
            res.plane_inliers
        );
        assert!(
            res.plane_rms < 2.0,
            "planarity RMS too high: {}",
            res.plane_rms
        );
        assert!(res.baseline_m > 0.0);
        assert!(res.disparity_png.starts_with("data:image/png;base64,"));
        assert!(res.overlay_png.starts_with("data:image/png;base64,"));
        assert!(res.rectified_pair_png.starts_with("data:image/png;base64,"));
        assert!(res.semi_global);

        // Optional visual dump for manual inspection (DUMP_DISPARITY=1).
        if std::env::var("DUMP_DISPARITY").is_ok() {
            for (name, url) in [
                ("overlay", &res.overlay_png),
                ("disparity", &res.disparity_png),
                ("pair", &res.rectified_pair_png),
            ] {
                let b64 = url.strip_prefix("data:image/png;base64,").unwrap();
                let bytes = base64::engine::general_purpose::STANDARD
                    .decode(b64)
                    .unwrap();
                std::fs::write(format!("/tmp/depth_{name}.png"), bytes).unwrap();
            }
        }
    }
}
