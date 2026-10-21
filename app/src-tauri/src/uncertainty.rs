//! Parameter values and projection uncertainty for the Diagnose
//! uncertainty panel.
//!
//! The export JSON already carries the covariance (labels, matrix, fixed
//! parameters, noise), which the frontend reads directly. What it cannot
//! read without mirroring every export's camera layout is a label's value
//! and the projection map, so both come from the library's
//! [`CalibratedCameras`] view of the export.

use vision_calibration::common::{CalibratedCameras, calibrated_cameras};
use vision_calibration::optim::ProjectionUncertainty;

fn cameras(export: &serde_json::Value) -> Result<Box<dyn CalibratedCameras>, String> {
    calibrated_cameras(export.clone()).map_err(|e| format!("export is unreadable: {e}"))
}

/// The value of each labelled parameter that is a camera parameter
/// (intrinsics, distortion, sensor tilt), aligned with `labels`; `None` for
/// poses and planes, whose covariance is over a perturbation, not a value.
pub fn camera_parameter_values(
    export: &serde_json::Value,
    labels: &[String],
) -> Result<Vec<Option<f64>>, String> {
    let cameras = cameras(export)?;
    Ok(labels
        .iter()
        .map(|label| cameras.camera_parameter(label))
        .collect())
}

/// The projection uncertainty of `camera` over an image of `image_size`.
pub fn projection_uncertainty(
    export: &serde_json::Value,
    camera: usize,
    image_size: [u32; 2],
) -> Result<ProjectionUncertainty, String> {
    cameras(export)?
        .projection_uncertainty(camera, image_size)
        .map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn identity() -> serde_json::Value {
        json!({"rotation": [0.0, 0.0, 0.0, 1.0], "translation": [0.0, 0.0, 0.5]})
    }

    /// A planar-intrinsics export with a covariance over `cx` and `cy`.
    fn planar_export() -> serde_json::Value {
        json!({
            "kind": "planar_intrinsics",
            "params": {
                "camera": {
                    "projection": {"type": "pinhole"},
                    "distortion": {"type": "none"},
                    "sensor": {"type": "identity"},
                    "intrinsics": {
                        "fx": 800.0, "fy": 800.0, "cx": 320.0, "cy": 240.0
                    }
                },
                "camera_se3_target": [identity()]
            },
            "report": {"final_cost": 0.0, "num_iters": 1},
            "mean_reproj_error": 0.1,
            "per_cam_reproj_errors": [0.1],
            "per_feature_residuals": {"target": []},
            "uncertainty": {
                "status": "available",
                "labels": ["cx", "cy", "camera_se3_target.rx"],
                "covariance": [[0.25, 0.0, 0.0], [0.0, 0.25, 0.0], [0.0, 0.0, 1e-6]],
                "noise": [{"kind": "reprojection", "sigma": 0.1, "count": 100}],
                "num_residuals": 100,
                "dof": 90
            }
        })
    }

    #[test]
    fn values_cover_camera_parameters_only() {
        let labels = ["cx", "camera_se3_target.rx", "fx"].map(String::from);
        let values = camera_parameter_values(&planar_export(), &labels).unwrap();
        assert_eq!(values, [Some(320.0), None, Some(800.0)]);
    }

    #[test]
    fn principal_point_sigma_is_the_whole_map() {
        let map = projection_uncertainty(&planar_export(), 0, [640, 480]).unwrap();
        assert!((map.center - 0.5).abs() < 1e-9);
        assert!((map.max - 0.5).abs() < 1e-9);
        assert!(projection_uncertainty(&planar_export(), 1, [640, 480]).is_err());
    }

    #[test]
    fn an_unknown_export_is_an_error() {
        assert!(camera_parameter_values(&json!({"kind": "nope"}), &[]).is_err());
    }
}
