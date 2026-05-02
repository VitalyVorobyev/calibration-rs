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

use nalgebra::{Point2, Point3, Vector3};
use serde::Serialize;
use vision_calibration::core::{
    BrownConrady5, CameraParams, DistortionParams, FxFyCxCySkew, IntrinsicsParams, Iso3,
    ProjectionParams, ScheimpflugParams, SensorParams,
};
use vision_calibration_core::CameraModel;

/// Number of depth samples along the ray. 64 is enough that even when
/// the polyline crosses the far plane at a high angle the rendered SVG
/// looks smooth.
const DEPTH_SAMPLES: usize = 64;
/// Min/max depths along the back-projected ray (meters). Logarithmic
/// spacing inside this window. Picked to comfortably cover the puzzle
/// 130×130 working volume and shorter benchtop calibrations.
const DEPTH_MIN_M: f64 = 0.05;
const DEPTH_MAX_M: f64 = 5.0;

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
    Ok(params.build())
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
}
