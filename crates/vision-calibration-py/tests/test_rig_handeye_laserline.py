"""Runtime test for the R5 ``run_rig_handeye_laserline`` binding.

The synthetic rig mirrors the Rust ``setup_synthetic_rig`` fixture in
``vision-calibration-optim`` (two cameras fixed in the robot base, a target
riding the gripper — EyeToHand — and one laser plane per camera). Building it in
Python exercises the full input-marshaling path (typed dataset →
``to_payload()`` → Rust ``serde`` → three-stage joint solve → export →
``RigHandeyeLaserlineResult``). Because the observations are noise-free, the
joint solve recovers the geometry to numerical precision, so the assertions can
be tight.
"""

from __future__ import annotations

import math
import unittest

import vision_calibration as vc

FX = FY = 900.0
CX, CY = 640.0, 360.0


# --- minimal pose algebra: pose = (R as 3x3 list-of-rows, t as length-3 list) -


def _mat_vec(R: list[list[float]], v: list[float]) -> list[float]:
    return [sum(R[i][k] * v[k] for k in range(3)) for i in range(3)]


def _mat_mul(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    return [[sum(A[i][k] * B[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def _transpose(R: list[list[float]]) -> list[list[float]]:
    return [[R[j][i] for j in range(3)] for i in range(3)]


def _rot_axis_angle(axis: list[float], ang: float) -> list[list[float]]:
    n = math.sqrt(sum(a * a for a in axis))
    ax, ay, az = (a / n for a in axis)
    c, s = math.cos(ang), math.sin(ang)
    d = 1.0 - c
    return [
        [c + ax * ax * d, ax * ay * d - az * s, ax * az * d + ay * s],
        [ay * ax * d + az * s, c + ay * ay * d, ay * az * d - ax * s],
        [az * ax * d - ay * s, az * ay * d + ax * s, c + az * az * d],
    ]


def _euler_rot(ax: float, ay: float, az: float) -> list[list[float]]:
    """Rz(az) · Ry(ay) · Rx(ax) — used to give the robot poses rotation about
    all three axes (a single-axis ramp leaves the hand-eye solve degenerate)."""
    return _mat_mul(
        _mat_mul(_rot_axis_angle([0, 0, 1], az), _rot_axis_angle([0, 1, 0], ay)),
        _rot_axis_angle([1, 0, 0], ax),
    )


def _pose(R: list[list[float]], t: list[float]) -> tuple[list[list[float]], list[float]]:
    return (R, list(t))


def _pose_inv(p):
    R, t = p
    Rt = _transpose(R)
    return (Rt, [-x for x in _mat_vec(Rt, t)])


def _pose_apply(p, v: list[float]) -> list[float]:
    R, t = p
    return [a + b for a, b in zip(_mat_vec(R, v), t)]


def _rot_to_quat_xyzw(R: list[list[float]]) -> tuple[float, float, float, float]:
    tr = R[0][0] + R[1][1] + R[2][2]
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        return ((R[2][1] - R[1][2]) / s, (R[0][2] - R[2][0]) / s, (R[1][0] - R[0][1]) / s, 0.25 * s)
    if R[0][0] > R[1][1] and R[0][0] > R[2][2]:
        s = math.sqrt(1.0 + R[0][0] - R[1][1] - R[2][2]) * 2.0
        return (0.25 * s, (R[0][1] + R[1][0]) / s, (R[0][2] + R[2][0]) / s, (R[2][1] - R[1][2]) / s)
    if R[1][1] > R[2][2]:
        s = math.sqrt(1.0 + R[1][1] - R[0][0] - R[2][2]) * 2.0
        return ((R[0][1] + R[1][0]) / s, 0.25 * s, (R[1][2] + R[2][1]) / s, (R[0][2] - R[2][0]) / s)
    s = math.sqrt(1.0 + R[2][2] - R[0][0] - R[1][1]) * 2.0
    return ((R[0][2] + R[2][0]) / s, (R[1][2] + R[2][1]) / s, 0.25 * s, (R[1][0] - R[0][1]) / s)


def _project(p_cam: list[float]) -> tuple[float, float] | None:
    x, y, z = p_cam
    if z <= 1e-6:
        return None
    return (CX + FX * x / z, CY + FY * y / z)


def _cross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _normalize(v):
    n = math.sqrt(_dot(v, v))
    return [x / n for x in v]


def _synthetic_dataset() -> vc.RigHandeyeLaserlineDataset:
    ident = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    cam_to_rig = [
        _pose(ident, [-0.10, 0.0, 0.0]),
        _pose(_rot_axis_angle([0, 0, 1], 0.05), [0.10, 0.0, 0.0]),
    ]
    cam_se3_rig = [_pose_inv(p) for p in cam_to_rig]
    handeye = _pose(_rot_axis_angle([0, 1, 0], 0.1), [0.15, -0.02, 0.30])  # rig_se3_base
    target_ref = _pose(_rot_axis_angle([1, 0, 0], -0.05), [0.02, 0.0, -0.03])  # gripper_se3_target
    planes_cam = [(_normalize([0.1, 0.0, 1.0]), -0.30), (_normalize([-0.05, 0.05, 1.0]), -0.35)]
    target_pts = [(x * 0.02, y * 0.02, 0.0) for y in range(-2, 3) for x in range(-3, 4)]

    views: list[vc.RigHandeyeLaserlineView] = []
    for i in range(8):
        # Rotate about all three axes across views. Hand-eye calibration
        # (Tsai-Lenz linear init) is ill-conditioned when every relative motion
        # shares one rotation axis — the SVD then diverges across BLAS/LAPACK
        # builds, which surfaced as a macOS-passes / ubuntu-fails flake.
        robot = _pose(
            _euler_rot(
                0.35 * math.sin(0.9 * i + 0.3),
                0.30 * math.cos(0.7 * i) - 0.1,
                0.28 * i - 0.4,
            ),
            [0.10 * math.cos(0.15 * i), 0.05 * math.sin(0.3 * i), 0.20 + 0.01 * i],
        )
        cams: list[vc.Observation | None] = []
        laser: list[list[tuple[float, float]] | None] = []
        for c in range(2):
            pts2d = []
            for pt in target_pts:
                p = _pose_apply(
                    cam_se3_rig[c],
                    _pose_apply(handeye, _pose_apply(robot, _pose_apply(target_ref, list(pt)))),
                )
                uv = _project(p)
                pts2d.append(uv if uv else (0.0, 0.0))
            cams.append(vc.Observation(points_3d=list(target_pts), points_2d=pts2d))

            # Laser pixels: intersect the per-camera laser plane with the target
            # plane in the camera frame and project a few points on the line.
            rot = _mat_mul(_mat_mul(cam_se3_rig[c][0], handeye[0]), _mat_mul(robot[0], target_ref[0]))
            n_target = _mat_vec(rot, [0.0, 0.0, 1.0])
            p_org = _pose_apply(
                cam_se3_rig[c],
                _pose_apply(handeye, _pose_apply(robot, _pose_apply(target_ref, [0.0, 0.0, 0.0]))),
            )
            d_target = -_dot(n_target, p_org)
            n_laser, d_laser = planes_cam[c]
            v = _cross(n_laser, n_target)
            vn = math.sqrt(_dot(v, v))
            laser_px: list[tuple[float, float]] = []
            if vn > 1e-9:
                vu = [x / vn for x in v]
                az, ay, ax = abs(vu[2]), abs(vu[1]), abs(vu[0])
                if az >= ax and az >= ay:
                    det = n_laser[0] * n_target[1] - n_laser[1] * n_target[0]
                    x = (-d_laser * n_target[1] - (-d_target) * n_laser[1]) / det
                    y = (n_laser[0] * (-d_target) - n_target[0] * (-d_laser)) / det
                    p0 = [x, y, 0.0]
                elif ay >= ax:
                    det = n_laser[0] * n_target[2] - n_laser[2] * n_target[0]
                    x = (-d_laser * n_target[2] - (-d_target) * n_laser[2]) / det
                    z = (n_laser[0] * (-d_target) - n_target[0] * (-d_laser)) / det
                    p0 = [x, 0.0, z]
                else:
                    det = n_laser[1] * n_target[2] - n_laser[2] * n_target[1]
                    y = (-d_laser * n_target[2] - (-d_target) * n_laser[2]) / det
                    z = (n_laser[1] * (-d_target) - n_target[1] * (-d_laser)) / det
                    p0 = [0.0, y, z]
                for si in range(-2, 3):
                    p = [p0[k] + vu[k] * (si * 0.01) for k in range(3)]
                    uv = _project(p)
                    if uv:
                        laser_px.append(uv)
            laser.append(laser_px if laser_px else None)

        qx, qy, qz, qw = _rot_to_quat_xyzw(robot[0])
        views.append(
            vc.RigHandeyeLaserlineView(
                cameras=cams,
                laser_pixels=laser,
                base_se3_gripper=vc.Pose(
                    rotation_xyzw=(qx, qy, qz, qw), translation_xyz=tuple(robot[1])
                ),
            )
        )
    return vc.RigHandeyeLaserlineDataset(num_cameras=2, views=views)


def _eye_to_hand_config() -> vc.RigHandeyeLaserlineCalibrationConfig:
    cfg = vc.RigHandeyeLaserlineCalibrationConfig()
    cfg.handeye.handeye_init.handeye_mode = "EyeToHand"
    return cfg


class RigHandeyeLaserlineRuntimeTest(unittest.TestCase):
    def test_joint_solve_recovers_synthetic_geometry(self) -> None:
        result = vc.run_rig_handeye_laserline(_synthetic_dataset(), _eye_to_hand_config())

        self.assertIsInstance(result, vc.RigHandeyeLaserlineResult)
        # Two cameras, one laser plane each, one per-camera stats block each.
        self.assertEqual(len(result.cameras), 2)
        self.assertEqual(len(result.sensors), 2)
        self.assertEqual(len(result.laser_planes_cam), 2)
        self.assertEqual(len(result.laser_planes_rig), 2)
        self.assertEqual(len(result.per_camera_stats), 2)
        self.assertEqual(len(result.per_cam_reproj_errors), 2)

        # EyeToHand mode is echoed back and populates the mode-dependent poses.
        self.assertEqual(result.handeye_mode, "EyeToHand")
        self.assertIsNotNone(result.rig_se3_base)
        self.assertIsNotNone(result.gripper_se3_target)
        self.assertIsNone(result.gripper_se3_rig)

        # Well-conditioned noise-free data → the joint solve is near-exact on
        # every platform. These bounds sit ~6+ orders above the observed optimum
        # (reproj ~5e-13 px, laser ~2e-15 m) — tight enough to catch a real
        # regression, loose enough to absorb cross-platform LM/SVD tail.
        self.assertLess(result.mean_reproj_error, 1e-6)
        for cam in result.cameras:
            # fx recovers to the ground-truth 900 only because the robot
            # rotations span all three axes; a single-axis fixture leaves the
            # hand-eye solve under-determined and fx drifts along a scale
            # ambiguity. This assertion is the guard on fixture conditioning.
            self.assertAlmostEqual(cam.intrinsics.fx, 900.0, delta=1.0)
        for stats in result.per_camera_stats:
            self.assertGreater(stats.laser_count, 0)
            self.assertLess(stats.mean_laser_err_m, 1e-6)

    def test_dict_input_is_rejected_with_type_error(self) -> None:
        with self.assertRaises(TypeError):
            vc.run_rig_handeye_laserline({"num_cameras": 2, "views": []}, _eye_to_hand_config())

    def test_dict_config_is_rejected_with_type_error(self) -> None:
        with self.assertRaises(TypeError):
            vc.run_rig_handeye_laserline(_synthetic_dataset(), {"joint_ba": {}})

    def test_invalid_input_surfaces_value_error(self) -> None:
        # A typed-but-empty dataset still takes the runtime path; Rust
        # `validate_input` rejects it and the binding surfaces a ValueError
        # (not TypeError, which is reserved for wrong Python types).
        empty = vc.RigHandeyeLaserlineDataset(num_cameras=2, views=[])
        with self.assertRaises(ValueError):
            vc.run_rig_handeye_laserline(empty, _eye_to_hand_config())


if __name__ == "__main__":
    unittest.main()
