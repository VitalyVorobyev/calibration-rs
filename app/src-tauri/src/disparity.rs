//! Server-side dense stereo matching for the Depth workspace.
//!
//! Given a synchronized stereo pair and the loaded rig calibration, this
//! rectifies the pair from the calibration
//! ([`vision_calibration::mvg::rectification::rectify`], any camera model),
//! dense-matches it ([`vision_calibration::mvg::dense::match_block`], block or
//! semi-global), and returns colormapped PNG data URLs plus quality metrics.
//! Math lives here, not in the frontend, so rectification uses the canonical
//! [`vision_calibration`] models — the same reason the epipolar overlay is
//! server-side.

use std::io::Cursor;
use std::path::{Path, PathBuf};

use base64::Engine;
use nalgebra::Matrix3;
use serde::{Deserialize, Serialize};
use tauri::State;
use vision_calibration::common::calibrated_cameras;
use vision_calibration::core::{Iso3, Pt2, Vec3};
use vision_calibration::mvg::dense::{BlockMatchOptions, DisparityMap, GrayImage, match_block};
use vision_calibration::mvg::rectification::{RectifyOptions, StereoRectification, rectify};

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
#[cfg_attr(feature = "schema-export", derive(schemars::JsonSchema))]
#[serde(rename_all = "camelCase")]
pub struct DisparityResult {
    /// Rectified left | right with shared epipolar rows (rectification sanity).
    pub rectified_pair_png: String,
    /// Disparity colormap (jet; invalid pixels black).
    pub disparity_png: String,
    /// Disparity colormap blended over the rectified left image.
    pub overlay_png: String,
    /// Depth colormap (jet over metric depth `Z = f·B / d`; invalid black).
    pub depth_png: String,
    /// Reprojected 3D point cloud (reference-camera frame, metres).
    pub point_cloud: PointCloud,
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

/// A reprojected 3D point cloud for the frontend's WebGL renderer.
#[derive(Debug, Clone, Serialize)]
#[cfg_attr(feature = "schema-export", derive(schemars::JsonSchema))]
#[serde(rename_all = "camelCase")]
pub struct PointCloud {
    /// Flat `[x, y, z, …]` positions in the reference-camera frame (metres).
    pub positions: Vec<f32>,
    /// Flat `[r, g, b, …]` per-point grayscale colours in `[0, 1]`.
    pub colors: Vec<f32>,
    /// Number of points (`positions.len() / 3`).
    pub count: usize,
}

/// Upper bound on rendered points; the cloud is grid-subsampled past this.
const MAX_POINTS: usize = 60_000;

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

    let rig = calibrated_cameras(export.clone())
        .map_err(|e| format!("calibration decode: {e}"))?
        .calibrated_rig()
        .map_err(|e| format!("calibration decode: {e}"))?;
    let n_cams = rig.num_cameras();
    if cam_a >= n_cams || cam_b >= n_cams || cam_a == cam_b {
        return Err(format!(
            "invalid camera pair (cam_a={cam_a}, cam_b={cam_b}, num_cameras={n_cams})"
        ));
    }

    // Nominal board depth in the reference frame (for the disparity search).
    let z_board = export
        .get("rig_se3_target")
        .and_then(|v| serde_json::from_value::<Vec<Iso3>>(v.clone()).ok())
        .and_then(|poses| poses.get(pose).map(|p| p.translation.z))
        .filter(|z| z.is_finite() && *z > 0.05)
        .unwrap_or(1.0);

    // Rectify the full-resolution images straight into the matching size:
    // `scale` shrinks the rectified intrinsics, so the output is 1/s of the
    // source. The source is prefiltered by the same factor to avoid aliasing.
    let left_src = load_gray(path_a)?;
    let right_src = load_gray(path_b)?;
    let (w, h) = (left_src.width / s, left_src.height / s);
    if w == 0 || h == 0 {
        return Err("image too small for the chosen downscale".to_string());
    }
    let mut rect_opts = RectifyOptions::default();
    rect_opts.scale = 1.0 / s as f64;
    let rect = rectify(&rig, [cam_a, cam_b], &rect_opts).map_err(|e| format!("rectify: {e}"))?;
    let rect_left = rect.rectify_image(0, &box_filter(&left_src, s), w, h);
    let rect_right = rect.rectify_image(1, &box_filter(&right_src, s), w, h);

    let f_rect = rect.k_rect().fx;
    let disp_nominal = (f_rect * rect.baseline() / z_board) as f32;
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

    // Reproject to metric depth + a 3D point cloud (reference-camera frame).
    let (point_cloud, depth) = reproject(&disp, &rect_left, &rect);

    Ok(DisparityResult {
        rectified_pair_png: pair_with_epilines_url(&rect_left, &rect_right)?,
        disparity_png: disparity_url(&disp, lo, hi)?,
        overlay_png: overlay_url(&rect_left, &disp, lo, hi)?,
        depth_png: depth_url(&depth, w, h)?,
        point_cloud,
        width: w,
        height: h,
        density,
        disp_min: lo,
        disp_max: hi,
        plane_rms,
        plane_inliers,
        baseline_m: rect.baseline(),
        semi_global: params.semi_global,
    })
}

// ---------------------------------------------------------------------------
// Image loading
// ---------------------------------------------------------------------------

fn load_gray(path: &Path) -> Result<GrayImage, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let img = image::load_from_memory(&bytes)
        .map_err(|e| format!("decode {}: {e}", path.display()))?
        .to_luma8();
    let (w, h) = (img.width() as usize, img.height() as usize);
    Ok(GrayImage::new(w, h, img.into_raw()))
}

/// A separable box filter of side `factor` (the anti-aliasing prefilter of a
/// `factor`× reduction); the image keeps its size. `factor == 1` is a copy.
fn box_filter(src: &GrayImage, factor: usize) -> GrayImage {
    if factor <= 1 {
        return src.clone();
    }
    let (w, h) = (src.width, src.height);
    // The window of pixel `x` is `[x - lo, x - lo + factor)`, clipped to the image.
    let lo = factor / 2;
    let pass = |input: &[f32], horizontal: bool| -> Vec<f32> {
        let mut out = vec![0f32; w * h];
        for y in 0..h {
            for x in 0..w {
                let (mut acc, mut n) = (0f32, 0f32);
                for k in 0..factor {
                    let (xx, yy) = if horizontal {
                        ((x + k).checked_sub(lo), Some(y))
                    } else {
                        (Some(x), (y + k).checked_sub(lo))
                    };
                    if let (Some(xx), Some(yy)) = (xx, yy)
                        && xx < w
                        && yy < h
                    {
                        acc += input[yy * w + xx];
                        n += 1.0;
                    }
                }
                out[y * w + x] = acc / n;
            }
        }
        out
    };
    let gray: Vec<f32> = src.data.iter().map(|&v| f32::from(v)).collect();
    let blurred = pass(&pass(&gray, true), false);
    GrayImage::new(
        w,
        h,
        blurred
            .iter()
            .map(|v| v.round().clamp(0.0, 255.0) as u8)
            .collect(),
    )
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
// Reprojection (disparity → metric depth + point cloud)
// ---------------------------------------------------------------------------

/// Reproject the disparity map to metric depth and a 3D point cloud through
/// the rectification: the depth map holds the rectified-frame `Z = f·B / d`
/// (it is indexed by rectified pixels), the cloud the points of
/// [`StereoRectification::point`] in the reference (left) camera's frame,
/// matching the 3D view's axes gizmo. Returns the (grid-subsampled) point
/// cloud and a per-pixel depth map (`NaN` where disparity is invalid).
fn reproject(
    disp: &DisparityMap,
    rect_left: &GrayImage,
    rect: &StereoRectification,
) -> (PointCloud, Vec<f32>) {
    let (w, h) = (disp.width, disp.height);
    let f_b = (rect.k_rect().fx * rect.baseline()) as f32;

    let mut depth = vec![f32::NAN; w * h];
    // Grid subsample so the rendered cloud stays under MAX_POINTS.
    let step = (((w * h) as f64 / MAX_POINTS as f64).sqrt().ceil() as usize).max(1);
    let mut positions = Vec::new();
    let mut colors = Vec::new();
    for y in 0..h {
        for x in 0..w {
            let d = disp.get(x, y);
            if !(d.is_finite() && d > 0.0) {
                continue;
            }
            depth[y * w + x] = f_b / d;
            if x % step == 0 && y % step == 0 {
                let Some(p) = rect.point(&Pt2::new(x as f64, y as f64), f64::from(d)) else {
                    continue;
                };
                positions.extend([p.x as f32, p.y as f32, p.z as f32]);
                let g = rect_left.get(x, y) as f32 / 255.0;
                colors.extend([g, g, g]);
            }
        }
    }
    let count = positions.len() / 3;
    (
        PointCloud {
            positions,
            colors,
            count,
        },
        depth,
    )
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

/// Depth colormap (jet over the metric depth range; near = blue, far = red).
fn depth_url(depth: &[f32], w: usize, h: usize) -> Result<String, String> {
    let mut lo = f32::INFINITY;
    let mut hi = f32::NEG_INFINITY;
    for &z in depth {
        if z.is_finite() {
            lo = lo.min(z);
            hi = hi.max(z);
        }
    }
    let span = (hi - lo).max(1e-6);
    let rgb: Vec<[u8; 3]> = depth
        .iter()
        .map(|&z| {
            if z.is_finite() {
                jet((z - lo) / span)
            } else {
                [0, 0, 0]
            }
        })
        .collect();
    rgb_to_url(w, h, &rgb)
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
        assert!(res.depth_png.starts_with("data:image/png;base64,"));
        assert!(res.semi_global);

        // Point cloud: reprojected board points cluster near the calibrated
        // board depth (~0.5 m) in the reference frame.
        let pc = &res.point_cloud;
        assert!(pc.count > 1000, "few cloud points: {}", pc.count);
        assert_eq!(pc.positions.len(), pc.count * 3);
        assert_eq!(pc.colors.len(), pc.count * 3);
        let mean_z: f32 = pc.positions.iter().skip(2).step_by(3).sum::<f32>() / pc.count as f32;
        assert!(
            (0.3..0.9).contains(&mean_z),
            "mean reprojected depth off: {mean_z} m"
        );

        // Optional visual dump for manual inspection (DUMP_DISPARITY=1).
        if std::env::var("DUMP_DISPARITY").is_ok() {
            for (name, url) in [
                ("overlay", &res.overlay_png),
                ("disparity", &res.disparity_png),
                ("depth", &res.depth_png),
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
