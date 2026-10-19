//! Export fixtures for the suite's unit tests.

use serde_json::{Value, json};
use vision_calibration_core::{BrownConrady5, FxFyCxCySkew, Iso3, make_pinhole_camera};

/// A two-camera rig hand-eye export whose covariance names camera 1's `fx`,
/// its `cam_se3_rig` (`tx`, `rz`) and view 1's robot-pose correction `rx`.
pub(super) fn rig_export(fx1: f64, cam1: &Iso3, delta_rx: f64) -> Value {
    let camera = |fx: f64| {
        make_pinhole_camera(
            FxFyCxCySkew {
                fx,
                fy: fx,
                cx: 640.0,
                cy: 480.0,
                skew: 0.0,
            },
            BrownConrady5::default(),
        )
    };
    json!({
        "kind": "rig_handeye",
        "cameras": [camera(800.0), camera(fx1)],
        "sensors": null,
        "cam_se3_rig": [Iso3::identity(), cam1],
        "rig_se3_target": [Iso3::identity(), Iso3::identity()],
        "handeye_mode": "EyeInHand",
        "gripper_se3_rig": Iso3::identity(),
        "base_se3_target": Iso3::identity(),
        "robot_deltas": [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [delta_rx, 0.0, 0.0, 0.0, 0.0, 0.0]],
        "mean_reproj_error": 0.1,
        "per_cam_reproj_errors": [0.1, 0.1],
        "per_feature_residuals": {},
        "uncertainty": {
            "status": "available",
            "labels": ["cam1.fx", "cam1.cam_se3_rig.tx", "cam1.cam_se3_rig.rz"],
            "covariance": [[4.0, 0.0, 0.0], [0.0, 1e-6, 0.0], [0.0, 0.0, 1e-6]],
            "views": [{"view": 1, "labels": ["robot_delta.rx"], "covariance": [[1e-6]]}],
            "noise": [],
            "num_residuals": 100,
            "dof": 90
        }
    })
}
