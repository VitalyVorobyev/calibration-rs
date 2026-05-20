//! Server-side epipolar overlay: given a clicked pixel in pane A, return
//! the polyline of its epipolar line in pane B (in distorted pane-B
//! pixel coordinates) plus the optional epipole.
//!
//! Math lives here, not in the React frontend, so distortion +
//! Scheimpflug projection use the canonical [`vision_calibration::core`]
//! camera models. Drift between Rust and a re-implemented TS projection
//! would be the worst possible bug class for a diagnostic tool — every
//! "is the calibration wrong or is the viz wrong" moment becomes a
//! science fair.

use image::{DynamicImage, Rgba, RgbaImage};
use nalgebra::{Point2, Point3, Vector3};
use serde::Serialize;
use std::{io::Cursor, path::Path};
use vision_calibration::core::{
    BrownConrady5, CameraParams, DistortionParams, FxFyCxCySkew, IntrinsicsParams, Iso3, PixelRect,
    ProjectionParams, ScheimpflugParams, SensorParams,
};
use vision_calibration_core::CameraModel;

/// Number of depth samples along the ray. 256 keeps the rendered SVG
/// smooth even when only a short subset of samples lands inside the
/// pane-B image (which is where the engineer actually reads the
/// epipolar curve). 64 was enough for the in-test scenario but left a
/// visibly piecewise polyline on real datasets where most samples
/// project outside the image and only a handful land within bounds.
const DEPTH_SAMPLES: usize = 256;
/// Min/max depths along the back-projected ray (meters). Logarithmic
/// spacing inside this window. Picked to comfortably cover the puzzle
/// 130×130 working volume and shorter benchtop calibrations.
const DEPTH_MIN_M: f64 = 0.05;
const DEPTH_MAX_M: f64 = 5.0;
/// Cutoff for cam-B-frame z below which we drop the sample. Even
/// though `project_point_c` already filters strictly behind-camera
/// points (`z <= 0`), depth samples that land *just* in front of cam
/// B's image plane produce near-singular projections that fly off
/// across the frame between consecutive samples — visually a fold /
/// "parabola" in pane B. 5 cm is past the lens minimum focus on
/// every dataset we ship; anything closer is meaningless geometry.
const CAM_B_Z_MIN_M: f64 = 0.05;

/// Result of a single epipolar overlay computation.
#[derive(Debug, Clone, Serialize)]
pub struct EpipolarOverlay {
    /// Pane-B distorted pixel coordinates of each depth sample that
    /// projected successfully. `[]` when the ray never crosses pane B's
    /// image plane in front of cam B.
    pub line_b: Vec<[f64; 2]>,
    /// Pane-B pixel coordinate of cam A's optical center. `None` when
    /// the cameras are arranged so the epipole is at infinity (or
    /// behind cam B).
    pub epipole_b: Option<[f64; 2]>,
    /// Number of depth samples whose projection diverged (point behind
    /// cam B, distortion fixed-point failure, …). For diagnostic UI.
    pub samples_clipped: usize,
}

/// Compute the overlay for a click at `point_px` on cam-A.
///
/// `export` is the most-recently-loaded calibration export JSON, expected
/// to carry `cameras: Vec<PinholeCamera-shaped>`, optional
/// `sensors: Vec<ScheimpflugParams>`, and `cam_se3_rig: Vec<Iso3>` — this
/// is true for `RigExtrinsicsExport`, `RigHandeyeExport`, and
/// `RigLaserlineDeviceExport`.
pub fn compute_overlay(
    export: &serde_json::Value,
    cam_a: usize,
    cam_b: usize,
    point_px: [f64; 2],
) -> Result<EpipolarOverlay, String> {
    let cameras = export
        .get("cameras")
        .and_then(|v| v.as_array())
        .ok_or("export has no `cameras` array (rig exports only)")?;
    let cam_se3_rig: Vec<Iso3> = serde_json::from_value(
        export
            .get("cam_se3_rig")
            .ok_or("export has no `cam_se3_rig`")?
            .clone(),
    )
    .map_err(|e| format!("cam_se3_rig decode: {e}"))?;
    let sensors = parse_sensors(export)?;

    let n = cameras.len();
    if cam_a >= n || cam_b >= n {
        return Err(format!(
            "camera index out of range (cam_a={cam_a}, cam_b={cam_b}, num_cameras={n})"
        ));
    }
    if cam_se3_rig.len() != n {
        return Err(format!(
            "cam_se3_rig length {} does not match cameras length {n}",
            cam_se3_rig.len()
        ));
    }

    let sensor_for =
        |idx: usize| -> Option<&ScheimpflugParams> { sensors.as_ref()?.get(idx)?.as_ref() };
    let cam_a_model = build_camera(&cameras[cam_a], sensor_for(cam_a))?;
    let cam_b_model = build_camera(&cameras[cam_b], sensor_for(cam_b))?;

    // Ray in cam-A frame from the clicked pixel. `Camera::backproject_pixel`
    // returns the point on the z = 1 plane, so scaling by `depth` gives a
    // 3D point at that depth without re-undistorting.
    let ray_a = cam_a_model.backproject_pixel(&Point2::new(point_px[0], point_px[1]));

    // T_B_A composes the rig→cam transforms: T_B_A = T_B_R · T_R_A.
    // cam_se3_rig[i] is T_C_R (rig→camera); inverting cam_a gives T_A_R
    // → T_R_A; combined with T_C_R for cam_b yields T_B_A.
    let t_b_a = cam_se3_rig[cam_b] * cam_se3_rig[cam_a].inverse();

    let mut line_b = Vec::with_capacity(DEPTH_SAMPLES);
    let mut samples_clipped = 0usize;
    for d in log_depths() {
        let p_a = Point3::from(ray_a.point * d);
        let p_b = t_b_a * p_a;
        // Pre-filter near-singular samples: in-front of cam B but so
        // close to its image plane that the projection effectively
        // jumps across the frame between this sample and the next.
        if p_b.z < CAM_B_Z_MIN_M {
            samples_clipped += 1;
            continue;
        }
        match cam_b_model.project_point_c(&p_b.coords) {
            Some(px) => line_b.push([px.x, px.y]),
            None => samples_clipped += 1,
        }
    }

    // Epipole = projection of cam-A's optical center (origin in A frame)
    // into cam B. None when behind cam B (parallel-cam configuration in
    // particular puts the epipole at infinity).
    let origin_b = t_b_a * Point3::<f64>::origin();
    let epipole_b = cam_b_model
        .project_point_c(&origin_b.coords)
        .map(|p| [p.x, p.y]);

    Ok(EpipolarOverlay {
        line_b,
        epipole_b,
        samples_clipped,
    })
}

/// Convert raw/distorted pixel coordinates into the undistorted pixel
/// frame used by the epipolar workspace. The output keeps the camera's
/// original `K` scale/center but removes distortion and sensor warping.
pub fn undistort_points(
    export: &serde_json::Value,
    camera: usize,
    points_px: Vec<[f64; 2]>,
) -> Result<Vec<[f64; 2]>, String> {
    let geometry = camera_geometry(export, camera)?;
    Ok(points_px
        .iter()
        .map(|p| raw_pixel_to_undistorted_pixel(&geometry, *p))
        .collect())
}

/// Decode an image and render it into the undistorted pixel frame for
/// `camera`. If `roi` is present, the source image is sampled from that
/// source rectangle while the output dimensions remain ROI-local.
pub fn undistort_image_png(
    export: &serde_json::Value,
    path: &Path,
    camera: usize,
    roi: Option<PixelRect>,
) -> Result<Vec<u8>, String> {
    let geometry = camera_geometry(export, camera)?;
    let bytes = std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let src = image::load_from_memory(&bytes)
        .map_err(|e| format!("decode {}: {e}", path.display()))?
        .to_rgba8();
    let (out_w, out_h) = match roi {
        Some(r) => (r.w, r.h),
        None => (src.width(), src.height()),
    };
    if out_w == 0 || out_h == 0 {
        return Err("undistorted image dimensions must be non-zero".to_string());
    }

    let mut out = RgbaImage::new(out_w, out_h);
    let offset_x = roi.map_or(0.0, |r| r.x as f64);
    let offset_y = roi.map_or(0.0, |r| r.y as f64);
    for y in 0..out_h {
        for x in 0..out_w {
            let raw = undistorted_pixel_to_raw_pixel(&geometry, [x as f64, y as f64]);
            let Some(raw) = raw else {
                out.put_pixel(x, y, Rgba([0, 0, 0, 255]));
                continue;
            };
            let sample = sample_bilinear(&src, raw[0] + offset_x, raw[1] + offset_y);
            out.put_pixel(x, y, sample);
        }
    }

    let mut cursor = Cursor::new(Vec::new());
    DynamicImage::ImageRgba8(out)
        .write_to(&mut cursor, image::ImageFormat::Png)
        .map_err(|e| format!("encode undistorted PNG: {e}"))?;
    Ok(cursor.into_inner())
}

/// Compute a straight epipolar line in cam-B's undistorted pixel frame.
///
/// `point_px` is already in cam-A's undistorted pixel frame. Returned
/// `line_b` contains zero or two clipped endpoints in cam-B's
/// undistorted pixel frame.
pub fn compute_overlay_undistorted(
    export: &serde_json::Value,
    cam_a: usize,
    cam_b: usize,
    point_px: [f64; 2],
    image_width_b: f64,
    image_height_b: f64,
) -> Result<EpipolarOverlay, String> {
    if !(image_width_b.is_finite() && image_height_b.is_finite())
        || image_width_b <= 0.0
        || image_height_b <= 0.0
    {
        return Err("cam-B image dimensions must be finite and positive".to_string());
    }

    let cameras = export
        .get("cameras")
        .and_then(|v| v.as_array())
        .ok_or("export has no `cameras` array (rig exports only)")?;
    let cam_se3_rig: Vec<Iso3> = serde_json::from_value(
        export
            .get("cam_se3_rig")
            .ok_or("export has no `cam_se3_rig`")?
            .clone(),
    )
    .map_err(|e| format!("cam_se3_rig decode: {e}"))?;
    let n = cameras.len();
    if cam_a >= n || cam_b >= n {
        return Err(format!(
            "camera index out of range (cam_a={cam_a}, cam_b={cam_b}, num_cameras={n})"
        ));
    }
    if cam_se3_rig.len() != n {
        return Err(format!(
            "cam_se3_rig length {} does not match cameras length {n}",
            cam_se3_rig.len()
        ));
    }

    let cam_a_geometry = camera_geometry(export, cam_a)?;
    let cam_b_geometry = camera_geometry(export, cam_b)?;
    let ray_a = undistorted_pixel_to_ray(&cam_a_geometry.k, point_px);
    let t_b_a = cam_se3_rig[cam_b] * cam_se3_rig[cam_a].inverse();

    let mut samples_clipped = 0usize;
    let mut first: Option<[f64; 2]> = None;
    let mut last: Option<[f64; 2]> = None;
    for d in log_depths() {
        let p_a = Point3::from(ray_a * d);
        let p_b = t_b_a * p_a;
        if p_b.z < CAM_B_Z_MIN_M {
            samples_clipped += 1;
            continue;
        }
        let Some(px) = project_undistorted_camera_point(&cam_b_geometry.k, &p_b.coords) else {
            samples_clipped += 1;
            continue;
        };
        if first.is_none() {
            first = Some(px);
        }
        last = Some(px);
    }

    let line_b = match (first, last) {
        (Some(a), Some(b)) => clip_line_to_image(a, b, image_width_b, image_height_b),
        _ => Vec::new(),
    };

    let origin_b = t_b_a * Point3::<f64>::origin();
    let epipole_b = project_undistorted_camera_point(&cam_b_geometry.k, &origin_b.coords);

    Ok(EpipolarOverlay {
        line_b,
        epipole_b,
        samples_clipped,
    })
}

/// Logarithmically-spaced depths from `DEPTH_MIN_M` to `DEPTH_MAX_M`.
fn log_depths() -> impl Iterator<Item = f64> {
    let ratio = DEPTH_MAX_M / DEPTH_MIN_M;
    (0..DEPTH_SAMPLES).map(move |i| {
        let t = i as f64 / (DEPTH_SAMPLES - 1) as f64;
        DEPTH_MIN_M * ratio.powf(t)
    })
}

/// Build a runtime camera from one entry of the export's `cameras` array
/// + optional Scheimpflug sensor params from the parallel `sensors` array.
fn build_camera(
    camera_json: &serde_json::Value,
    sensor: Option<&ScheimpflugParams>,
) -> Result<CameraModel, String> {
    Ok(build_camera_geometry(camera_json, sensor)?.model)
}

#[derive(Clone, Debug)]
struct CameraGeometry {
    model: CameraModel,
    k: FxFyCxCySkew<f64>,
}

fn camera_geometry(export: &serde_json::Value, camera: usize) -> Result<CameraGeometry, String> {
    let cameras = export
        .get("cameras")
        .and_then(|v| v.as_array())
        .ok_or("export has no `cameras` array (rig exports only)")?;
    let n = cameras.len();
    if camera >= n {
        return Err(format!(
            "camera index out of range (camera={camera}, num_cameras={n})"
        ));
    }
    let sensors = parse_sensors(export)?;
    let sensor_for =
        |idx: usize| -> Option<&ScheimpflugParams> { sensors.as_ref()?.get(idx)?.as_ref() };
    build_camera_geometry(&cameras[camera], sensor_for(camera))
}

fn build_camera_geometry(
    camera_json: &serde_json::Value,
    sensor: Option<&ScheimpflugParams>,
) -> Result<CameraGeometry, String> {
    let k: FxFyCxCySkew<f64> = serde_json::from_value(
        camera_json
            .get("k")
            .ok_or("camera entry has no `k` field")?
            .clone(),
    )
    .map_err(|e| format!("camera.k decode: {e}"))?;
    let dist: BrownConrady5<f64> = serde_json::from_value(
        camera_json
            .get("dist")
            .ok_or("camera entry has no `dist` field")?
            .clone(),
    )
    .map_err(|e| format!("camera.dist decode: {e}"))?;

    let sensor_params = match sensor {
        None => SensorParams::Identity,
        Some(s) => SensorParams::Scheimpflug { params: *s },
    };

    let params = CameraParams {
        projection: ProjectionParams::Pinhole,
        distortion: DistortionParams::BrownConrady5 { params: dist },
        sensor: sensor_params,
        intrinsics: IntrinsicsParams::FxFyCxCySkew { params: k },
    };
    Ok(CameraGeometry {
        model: params.build(),
        k,
    })
}

/// Parse the optional `sensors` array. None when the export is a pinhole
/// rig; Some(_) when Scheimpflug. Each entry is `null` for pinhole
/// cameras inside a mixed rig (none today, but future-proof).
fn parse_sensors(
    export: &serde_json::Value,
) -> Result<Option<Vec<Option<ScheimpflugParams>>>, String> {
    let Some(arr) = export.get("sensors") else {
        return Ok(None);
    };
    if arr.is_null() {
        return Ok(None);
    }
    let arr = arr
        .as_array()
        .ok_or("`sensors` is neither null nor an array")?;
    let mut out = Vec::with_capacity(arr.len());
    for entry in arr {
        if entry.is_null() {
            out.push(None);
        } else {
            let s: ScheimpflugParams = serde_json::from_value(entry.clone())
                .map_err(|e| format!("sensors[..] decode: {e}"))?;
            out.push(Some(s));
        }
    }
    Ok(Some(out))
}

/// Vector3 alias to keep imports tidy if/when the algorithm grows beyond
/// the current single-call shape.
#[allow(dead_code)]
type V3 = Vector3<f64>;

fn raw_pixel_to_undistorted_pixel(geometry: &CameraGeometry, px: [f64; 2]) -> [f64; 2] {
    let ray = geometry.model.backproject_pixel(&Point2::new(px[0], px[1]));
    normalized_to_pixel(&geometry.k, ray.point.x, ray.point.y)
}

fn undistorted_pixel_to_raw_pixel(geometry: &CameraGeometry, px: [f64; 2]) -> Option<[f64; 2]> {
    let ray = undistorted_pixel_to_ray(&geometry.k, px);
    geometry
        .model
        .project_point_c(&ray)
        .map(|p| [p.x, p.y])
        .filter(|p| p[0].is_finite() && p[1].is_finite())
}

fn undistorted_pixel_to_ray(k: &FxFyCxCySkew<f64>, px: [f64; 2]) -> Vector3<f64> {
    let y = (px[1] - k.cy) / k.fy;
    let x = (px[0] - k.cx - k.skew * y) / k.fx;
    Vector3::new(x, y, 1.0)
}

fn project_undistorted_camera_point(k: &FxFyCxCySkew<f64>, p_c: &Vector3<f64>) -> Option<[f64; 2]> {
    if p_c.z <= 0.0 {
        return None;
    }
    let x = p_c.x / p_c.z;
    let y = p_c.y / p_c.z;
    let px = normalized_to_pixel(k, x, y);
    if px[0].is_finite() && px[1].is_finite() {
        Some(px)
    } else {
        None
    }
}

fn normalized_to_pixel(k: &FxFyCxCySkew<f64>, x: f64, y: f64) -> [f64; 2] {
    [k.fx * x + k.skew * y + k.cx, k.fy * y + k.cy]
}

fn clip_line_to_image(a: [f64; 2], b: [f64; 2], w: f64, h: f64) -> Vec<[f64; 2]> {
    let dx = b[0] - a[0];
    let dy = b[1] - a[1];
    let mut pts: Vec<[f64; 2]> = Vec::with_capacity(4);
    let eps = 1e-9;

    if dx.abs() > eps {
        for x in [0.0, w] {
            let t = (x - a[0]) / dx;
            let y = a[1] + t * dy;
            if y >= -eps && y <= h + eps {
                push_unique(&mut pts, [x, y.clamp(0.0, h)]);
            }
        }
    }
    if dy.abs() > eps {
        for y in [0.0, h] {
            let t = (y - a[1]) / dy;
            let x = a[0] + t * dx;
            if x >= -eps && x <= w + eps {
                push_unique(&mut pts, [x.clamp(0.0, w), y]);
            }
        }
    }

    if pts.len() <= 2 {
        return pts;
    }

    let mut best = (0usize, 1usize, -1.0f64);
    for i in 0..pts.len() {
        for j in (i + 1)..pts.len() {
            let d2 = (pts[i][0] - pts[j][0]).powi(2) + (pts[i][1] - pts[j][1]).powi(2);
            if d2 > best.2 {
                best = (i, j, d2);
            }
        }
    }
    vec![pts[best.0], pts[best.1]]
}

fn push_unique(pts: &mut Vec<[f64; 2]>, p: [f64; 2]) {
    if pts
        .iter()
        .any(|q| (q[0] - p[0]).abs() < 1e-7 && (q[1] - p[1]).abs() < 1e-7)
    {
        return;
    }
    pts.push(p);
}

fn sample_bilinear(src: &RgbaImage, x: f64, y: f64) -> Rgba<u8> {
    if !x.is_finite()
        || !y.is_finite()
        || x < 0.0
        || y < 0.0
        || x > (src.width().saturating_sub(1)) as f64
        || y > (src.height().saturating_sub(1)) as f64
    {
        return Rgba([0, 0, 0, 255]);
    }
    let x0 = x.floor() as u32;
    let y0 = y.floor() as u32;
    let x1 = (x0 + 1).min(src.width() - 1);
    let y1 = (y0 + 1).min(src.height() - 1);
    let tx = x - x0 as f64;
    let ty = y - y0 as f64;
    let p00 = src.get_pixel(x0, y0).0;
    let p10 = src.get_pixel(x1, y0).0;
    let p01 = src.get_pixel(x0, y1).0;
    let p11 = src.get_pixel(x1, y1).0;
    let mut out = [0u8; 4];
    for c in 0..4 {
        let top = p00[c] as f64 * (1.0 - tx) + p10[c] as f64 * tx;
        let bottom = p01[c] as f64 * (1.0 - tx) + p11[c] as f64 * tx;
        out[c] = (top * (1.0 - ty) + bottom * ty).round().clamp(0.0, 255.0) as u8;
    }
    Rgba(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use vision_calibration::core::{BrownConrady5 as BC5, FxFyCxCySkew as K, make_pinhole_camera};

    /// Synthetic 2-camera pinhole rig: project a known 3D point through
    /// cam A, ask for the overlay, assert pane-B's projected_pixel of
    /// the same 3D point lies within 0.5 px of *some* polyline sample.
    /// Tight bound because the depth sweep is dense (64 log samples).
    #[test]
    fn overlay_polyline_passes_through_corresponding_pixel_pinhole() {
        let k = K {
            fx: 800.0,
            fy: 800.0,
            cx: 320.0,
            cy: 240.0,
            skew: 0.0,
        };
        let cam_a = make_pinhole_camera(k, BC5::default());
        let cam_b = make_pinhole_camera(k, BC5::default());
        // T_C_R: identity for cam A, +0.2 m baseline along X for cam B
        // (cam B is to the right of cam A in rig frame).
        let t_a_r = Iso3::identity();
        let t_b_r = Iso3::from_parts(
            nalgebra::Translation3::new(0.2, 0.0, 0.0),
            nalgebra::UnitQuaternion::identity(),
        );
        // 3D point in rig frame, ~1 m in front of both cameras.
        let p_r = Point3::new(0.0, 0.0, 1.0);
        let p_a = t_a_r * p_r;
        let p_b = t_b_r * p_r;
        let pixel_a = cam_a.project_point_c(&p_a.coords).unwrap();
        let pixel_b = cam_b.project_point_c(&p_b.coords).unwrap();

        // Hand-build the export-shaped JSON the production code reads.
        let export = serde_json::json!({
            "cameras": [serde_json::to_value(&cam_a).unwrap(), serde_json::to_value(&cam_b).unwrap()],
            "cam_se3_rig": [t_a_r, t_b_r],
        });

        let overlay = compute_overlay(&export, 0, 1, [pixel_a.x, pixel_a.y]).unwrap();
        assert!(!overlay.line_b.is_empty(), "polyline must have samples");

        let dist_min = overlay
            .line_b
            .iter()
            .map(|p| {
                let dx = p[0] - pixel_b.x;
                let dy = p[1] - pixel_b.y;
                (dx * dx + dy * dy).sqrt()
            })
            .fold(f64::INFINITY, f64::min);
        assert!(
            dist_min < 0.5,
            "polyline closest sample {dist_min:.4} px away from \
             corresponding pane-B projection (pixel_b={pixel_b:?})"
        );
    }

    #[test]
    fn overlay_rejects_out_of_range_camera() {
        let k = K {
            fx: 800.0,
            fy: 800.0,
            cx: 320.0,
            cy: 240.0,
            skew: 0.0,
        };
        let cam = make_pinhole_camera(k, BC5::default());
        let export = serde_json::json!({
            "cameras": [serde_json::to_value(&cam).unwrap()],
            "cam_se3_rig": [Iso3::identity()],
        });
        let err = compute_overlay(&export, 0, 5, [100.0, 100.0]).unwrap_err();
        assert!(err.contains("out of range"), "got: {err}");
    }

    #[test]
    fn log_depth_endpoints_match_constants() {
        let depths: Vec<f64> = log_depths().collect();
        assert_eq!(depths.len(), DEPTH_SAMPLES);
        assert!((depths[0] - DEPTH_MIN_M).abs() < 1e-12);
        assert!((depths[DEPTH_SAMPLES - 1] - DEPTH_MAX_M).abs() < 1e-9);
    }

    /// Regression guard for the "parabola in pane B" fold bug. When
    /// cam B is positioned so that early depth samples on cam A's ray
    /// land less than `CAM_B_Z_MIN_M` in front of cam B's image plane,
    /// `compute_overlay` must drop those samples (counted in
    /// `samples_clipped`) so the polyline doesn't fork through the
    /// near-singular projection zone.
    ///
    /// Construction: cam A at rig origin (`T_A_R = I`); cam B has the
    /// same orientation as cam A but is offset so that a forward ray
    /// from cam A's principal pixel maps to `p_b = (0, 0, d - 0.04)`
    /// in cam B's frame. With the depth sweep starting at `d = 0.05`,
    /// the smallest sample produces `p_b.z = 0.01 m`, which is well
    /// below the `CAM_B_Z_MIN_M = 0.05` cutoff and must be dropped.
    /// Samples at `d > 0.09` survive and form a usable polyline.
    #[test]
    fn overlay_drops_near_image_plane_samples() {
        let k = K {
            fx: 800.0,
            fy: 800.0,
            cx: 320.0,
            cy: 240.0,
            skew: 0.0,
        };
        let cam = make_pinhole_camera(k, BC5::default());
        let t_a_r = Iso3::identity();
        // Translation only — rotation is identity so cam B's forward
        // axis aligns with cam A's. Translating (0, 0, -0.04) in
        // T_B_R means cam B sits 4 cm further along Z than the rig
        // origin (i.e. than cam A): a forward ray's `p_b.z` equals
        // `d - 0.04`.
        let t_b_r = Iso3::from_parts(
            nalgebra::Translation3::new(0.0, 0.0, -0.04),
            nalgebra::UnitQuaternion::identity(),
        );
        let export = serde_json::json!({
            "cameras": [serde_json::to_value(&cam).unwrap(), serde_json::to_value(&cam).unwrap()],
            "cam_se3_rig": [t_a_r, t_b_r],
        });
        // Click at cam A's principal point → back-projected ray is
        // along the optical axis.
        let overlay = compute_overlay(&export, 0, 1, [320.0, 240.0]).unwrap();
        assert!(
            overlay.samples_clipped > 0,
            "expected the near-image-plane cutoff to drop at least one sample; got 0 clipped"
        );
        // Polyline should still have a non-trivial run after the cutoff.
        assert!(
            overlay.line_b.len() > 4,
            "expected a usable polyline after the cutoff; got {}",
            overlay.line_b.len()
        );
    }

    #[test]
    fn undistorted_pixel_roundtrip_matches_raw_camera_frame() {
        let export = stereo_like_export();
        let geometry = camera_geometry(&export, 1).unwrap();

        for raw in [[575.3, 108.2], [345.0, 500.0], [900.0, 220.0]] {
            let undistorted = raw_pixel_to_undistorted_pixel(&geometry, raw);
            let raw_roundtrip = undistorted_pixel_to_raw_pixel(&geometry, undistorted).unwrap();
            let err =
                ((raw_roundtrip[0] - raw[0]).powi(2) + (raw_roundtrip[1] - raw[1]).powi(2)).sqrt();
            assert!(
                err < 0.1,
                "raw→undistorted→raw roundtrip drifted by {err:.4} px for {raw:?}"
            );
        }
    }

    #[test]
    fn raw_distorted_overlay_can_fold_but_undistorted_overlay_is_clipped_line() {
        let export = stereo_like_export();
        let raw_left_px = [358.4, 126.9];
        let raw_right_px = [575.3, 108.2];

        let raw = compute_overlay(&export, 0, 1, raw_left_px).unwrap();
        let raw_runs = in_bounds_runs(&raw.line_b, 1024.0, 768.0, 192.0);
        assert!(
            raw_runs >= 2,
            "expected raw distorted depth samples to split into multiple visible runs"
        );

        let point_a = undistort_points(&export, 0, vec![raw_left_px]).unwrap()[0];
        let point_b = undistort_points(&export, 1, vec![raw_right_px]).unwrap()[0];
        let undistorted =
            compute_overlay_undistorted(&export, 0, 1, point_a, 1024.0, 768.0).unwrap();
        assert_eq!(
            undistorted.line_b.len(),
            2,
            "undistorted overlay should be represented by clipped endpoints"
        );
        for p in &undistorted.line_b {
            assert!(
                p[0] >= 0.0 && p[0] <= 1024.0 && p[1] >= 0.0 && p[1] <= 768.0,
                "endpoint outside image bounds: {p:?}"
            );
        }
        let dist = distance_to_segment(point_b, undistorted.line_b[0], undistorted.line_b[1]);
        assert!(
            dist < 3.0,
            "corresponding feature should remain close to the undistorted epipolar line; got {dist:.3} px"
        );
    }

    fn stereo_like_export() -> serde_json::Value {
        serde_json::json!({
            "cameras": [
                {
                    "k": {
                        "fx": 713.8456995546412,
                        "fy": 724.6690645729575,
                        "cx": 523.4129016186437,
                        "cy": 285.9522601918888,
                        "skew": 0.0
                    },
                    "dist": {
                        "k1": 0.015021094581498156,
                        "k2": -0.07151318797378012,
                        "k3": 0.0,
                        "p1": 0.0006183511614000294,
                        "p2": 0.0002526916716399372,
                        "iters": 8
                    }
                },
                {
                    "k": {
                        "fx": 724.7966090646895,
                        "fy": 735.5984809398797,
                        "cx": 513.9329313241342,
                        "cy": 292.95508592356225,
                        "skew": 0.0
                    },
                    "dist": {
                        "k1": 0.00113846407979313,
                        "k2": -0.0525607289860634,
                        "k3": 0.0,
                        "p1": -0.0014292429046042455,
                        "p2": 0.0014269862041345035,
                        "iters": 8
                    }
                }
            ],
            "cam_se3_rig": [
                {
                    "rotation": [0.0, 0.0, 0.0, 1.0],
                    "translation": [0.0, 0.0, 0.0]
                },
                {
                    "rotation": [
                        0.013175949007657478,
                        -0.01753553695143165,
                        -0.004069965919389522,
                        0.9997511363779427
                    ],
                    "translation": [
                        0.1892859214228441,
                        -0.004507939830404787,
                        0.010062154737396994
                    ]
                }
            ]
        })
    }

    fn in_bounds_runs(pts: &[[f64; 2]], w: f64, h: f64, jump_px: f64) -> usize {
        let mut runs = 0usize;
        let mut prev: Option<[f64; 2]> = None;
        for p in pts {
            let in_bounds = p[0] >= 0.0 && p[0] <= w && p[1] >= 0.0 && p[1] <= h;
            if !in_bounds {
                prev = None;
                continue;
            }
            if let Some(prev_p) = prev {
                let jump = ((p[0] - prev_p[0]).powi(2) + (p[1] - prev_p[1]).powi(2)).sqrt();
                if jump > jump_px {
                    runs += 1;
                }
            } else {
                runs += 1;
            }
            prev = Some(*p);
        }
        runs
    }

    fn distance_to_segment(p: [f64; 2], a: [f64; 2], b: [f64; 2]) -> f64 {
        let abx = b[0] - a[0];
        let aby = b[1] - a[1];
        let len_sq = abx * abx + aby * aby;
        if len_sq == 0.0 {
            return ((p[0] - a[0]).powi(2) + (p[1] - a[1]).powi(2)).sqrt();
        }
        let t = (((p[0] - a[0]) * abx + (p[1] - a[1]) * aby) / len_sq).clamp(0.0, 1.0);
        let q = [a[0] + t * abx, a[1] + t * aby];
        ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2)).sqrt()
    }
}
