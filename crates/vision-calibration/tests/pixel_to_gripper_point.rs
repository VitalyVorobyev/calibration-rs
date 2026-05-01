//! Unit tests for `pixel_to_gripper_point`.
//!
//! Covers: happy path (EyeInHand, known synthetic geometry), cam_idx
//! out-of-range, ray-plane miss (parallel ray), and missing base_se3_gripper
//! in EyeToHand mode.

use nalgebra::{Isometry3, Rotation3, Translation3, Vector3};
use vision_calibration::{
    core::{BrownConrady5, FxFyCxCySkew, Iso3, Pt2, Pt3, ScheimpflugParams, make_pinhole_camera},
    optim::LaserPlane,
    pixel_to_gripper_point,
    rig_handeye::RigHandeyeExport,
};

/// Build a minimal `RigHandeyeExport` (Scheimpflug variant) for EyeInHand mode.
///
/// Uses `serde_json` round-trip to construct the `#[non_exhaustive]` struct
/// without relying on struct literal syntax.
fn make_eye_in_hand_export() -> RigHandeyeExport {
    let intr = FxFyCxCySkew {
        fx: 600.0,
        fy: 600.0,
        cx: 320.0,
        cy: 240.0,
        skew: 0.0,
    };
    let dist = BrownConrady5 {
        k1: 0.0,
        k2: 0.0,
        k3: 0.0,
        p1: 0.0,
        p2: 0.0,
        iters: 5,
    };
    let cam = make_pinhole_camera(intr, dist);
    let sensor = ScheimpflugParams::default();
    let identity: Iso3 = Isometry3::identity();

    let camera_json = serde_json::to_string(&cam).unwrap();
    let sensor_json = serde_json::to_string(&sensor).unwrap();
    let identity_json = serde_json::to_string(&identity).unwrap();

    let json = format!(
        r#"{{
            "cameras": [{camera_json}],
            "sensors": [{sensor_json}],
            "cam_se3_rig": [{identity_json}],
            "handeye_mode": "EyeInHand",
            "gripper_se3_rig": {identity_json},
            "rig_se3_base": null,
            "base_se3_target": null,
            "gripper_se3_target": null,
            "robot_deltas": null,
            "mean_reproj_error": 0.0,
            "per_cam_reproj_errors": [0.0]
        }}"#
    );
    serde_json::from_str(&json).expect("failed to parse eye-in-hand export")
}

/// Build a minimal `RigHandeyeExport` (Scheimpflug variant) for EyeToHand mode.
fn make_eye_to_hand_export() -> RigHandeyeExport {
    let intr = FxFyCxCySkew {
        fx: 600.0,
        fy: 600.0,
        cx: 320.0,
        cy: 240.0,
        skew: 0.0,
    };
    let dist = BrownConrady5 {
        k1: 0.0,
        k2: 0.0,
        k3: 0.0,
        p1: 0.0,
        p2: 0.0,
        iters: 5,
    };
    let cam = make_pinhole_camera(intr, dist);
    let sensor = ScheimpflugParams::default();
    let identity: Iso3 = Isometry3::identity();

    let camera_json = serde_json::to_string(&cam).unwrap();
    let sensor_json = serde_json::to_string(&sensor).unwrap();
    let identity_json = serde_json::to_string(&identity).unwrap();

    let json = format!(
        r#"{{
            "cameras": [{camera_json}],
            "sensors": [{sensor_json}],
            "cam_se3_rig": [{identity_json}],
            "handeye_mode": "EyeToHand",
            "gripper_se3_rig": null,
            "rig_se3_base": {identity_json},
            "base_se3_target": null,
            "gripper_se3_target": null,
            "robot_deltas": null,
            "mean_reproj_error": 0.0,
            "per_cam_reproj_errors": [0.0]
        }}"#
    );
    serde_json::from_str(&json).expect("failed to parse eye-to-hand export")
}

// ─── Test 1: Happy path (EyeInHand) ──────────────────────────────────────────

#[test]
fn happy_path_eye_in_hand() {
    // Set up a simple geometry:
    // - Camera at rig origin, identity cam_se3_rig.
    // - Laser plane in rig frame: y = 0.1 (normal = [0,1,0], distance = -0.1).
    // - A 3D point P_rig on the plane: (0.05, 0.1, 0.3).
    //   Verify: n·p + d = 0*0.05 + 1*0.1 + 0*0.3 + (-0.1) = 0. ✓
    // - gripper_se3_rig = identity → P_gripper = P_rig.
    // - Project P_rig through the camera to get the pixel.
    //
    // Note: we must NOT use a point with y=0 projected through y=0 plane — the
    // resulting ray would have zero y-component and be parallel to the plane.
    let rig_cal = make_eye_in_hand_export();
    let plane = LaserPlane::new(Vector3::new(0.0, 1.0, 0.0), -0.1);

    // Known 3D point on the plane (in rig = cam frame, since identity).
    let p_rig = Pt3::new(0.05, 0.1, 0.3);
    // Project to pixel using simple pinhole (K * P / Z).
    let fx = 600.0_f64;
    let fy = 600.0_f64;
    let cx = 320.0_f64;
    let cy = 240.0_f64;
    let u = fx * p_rig.x / p_rig.z + cx;
    let v = fy * p_rig.y / p_rig.z + cy;
    let pixel = Pt2::new(u, v);

    let result = pixel_to_gripper_point(0, pixel, &rig_cal, &[plane], None);
    assert!(result.is_ok(), "expected Ok, got: {:?}", result);

    let p_out = result.unwrap();
    // With identity transforms, the recovered point should match P_rig.
    assert!(
        (p_out.x - p_rig.x).abs() < 1e-9,
        "x mismatch: {} vs {}",
        p_out.x,
        p_rig.x
    );
    assert!(
        (p_out.y - p_rig.y).abs() < 1e-9,
        "y mismatch: {} vs {}",
        p_out.y,
        p_rig.y
    );
    assert!(
        (p_out.z - p_rig.z).abs() < 1e-9,
        "z mismatch: {} vs {}",
        p_out.z,
        p_rig.z
    );
}

// ─── Test 2: cam_idx out of range ────────────────────────────────────────────

#[test]
fn cam_idx_out_of_range_returns_invalid_input() {
    let rig_cal = make_eye_in_hand_export(); // 1 camera
    let plane = LaserPlane::new(Vector3::new(0.0, 1.0, 0.0), 0.0);
    let pixel = Pt2::new(320.0, 240.0);

    let result = pixel_to_gripper_point(1, pixel, &rig_cal, &[plane], None);
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("out of range") || msg.contains("invalid input"),
        "unexpected error message: {msg}"
    );
}

// ─── Test 3: Ray parallel to laser plane ─────────────────────────────────────

#[test]
fn ray_parallel_to_plane_returns_numerical_error() {
    let rig_cal = make_eye_in_hand_export();
    // Plane normal = [1, 0, 0]; a pixel at the principal point maps to a ray
    // direction [0, 0, 1] in camera frame. dot([1,0,0], [0,0,1]) = 0 → parallel.
    let plane = LaserPlane::new(Vector3::new(1.0, 0.0, 0.0), 0.0);
    let pixel = Pt2::new(320.0, 240.0); // principal point

    let result = pixel_to_gripper_point(0, pixel, &rig_cal, &[plane], None);
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("parallel") || msg.contains("numerical"),
        "unexpected error message: {msg}"
    );
}

// ─── Test 4: EyeToHand with missing base_se3_gripper ─────────────────────────

#[test]
fn eye_to_hand_missing_pose_returns_invalid_input() {
    let rig_cal = make_eye_to_hand_export();
    let plane = LaserPlane::new(Vector3::new(0.0, 1.0, 0.0), 0.0);
    let pixel = Pt2::new(320.0, 220.0); // slightly above principal point

    // base_se3_gripper = None in EyeToHand mode must return an error.
    let result = pixel_to_gripper_point(0, pixel, &rig_cal, &[plane], None);
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("EyeToHand") || msg.contains("base_se3_gripper") || msg.contains("invalid"),
        "unexpected error message: {msg}"
    );
}

// ─── Test 5: EyeToHand happy path ────────────────────────────────────────────

#[test]
fn eye_to_hand_happy_path() {
    let rig_cal = make_eye_to_hand_export();
    // rig_se3_base = identity, base_se3_gripper = identity → p_gripper = p_rig.
    // Use the same y=0.1 plane to avoid degenerate parallel-ray geometry.
    let plane = LaserPlane::new(Vector3::new(0.0, 1.0, 0.0), -0.1);
    let p_rig = Pt3::new(0.05, 0.1, 0.3);
    let fx = 600.0_f64;
    let fy = 600.0_f64;
    let cx = 320.0_f64;
    let cy = 240.0_f64;
    let u = fx * p_rig.x / p_rig.z + cx;
    let v = fy * p_rig.y / p_rig.z + cy;
    let pixel = Pt2::new(u, v);

    // With identity rig_se3_base and identity base_se3_gripper:
    // p_base = rig_se3_base.inverse() * p_rig = p_rig
    // p_gripper = base_se3_gripper.inverse() * p_base = p_rig
    let base_se3_gripper: Iso3 = Isometry3::from_parts(
        Translation3::new(0.0, 0.0, 0.0),
        Rotation3::identity().into(),
    );

    let result = pixel_to_gripper_point(0, pixel, &rig_cal, &[plane], Some(base_se3_gripper));
    assert!(result.is_ok(), "expected Ok, got: {:?}", result);

    let p_out = result.unwrap();
    assert!(
        (p_out.x - p_rig.x).abs() < 1e-9,
        "x mismatch: {} vs {}",
        p_out.x,
        p_rig.x
    );
    assert!(
        (p_out.z - p_rig.z).abs() < 1e-9,
        "z mismatch: {} vs {}",
        p_out.z,
        p_rig.z
    );
}
