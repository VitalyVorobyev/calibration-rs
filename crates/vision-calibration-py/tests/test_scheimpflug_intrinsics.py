from __future__ import annotations

import math
import unittest

import vision_calibration as vc


def _rotation_from_euler_xyz(ax: float, ay: float, az: float) -> list[list[float]]:
    sx, cx = math.sin(ax), math.cos(ax)
    sy, cy = math.sin(ay), math.cos(ay)
    sz, cz = math.sin(az), math.cos(az)
    return [
        [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
        [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
        [-sy, cy * sx, cy * cx],
    ]


def _quat_xyzw_from_rot(r: list[list[float]]) -> list[float]:
    trace = r[0][0] + r[1][1] + r[2][2]
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        return [
            (r[2][1] - r[1][2]) / s,
            (r[0][2] - r[2][0]) / s,
            (r[1][0] - r[0][1]) / s,
            0.25 * s,
        ]
    raise ValueError("unexpected non-positive trace in test rotation")


def _make_pose(ax: float, ay: float, az: float, tx: float, ty: float, tz: float) -> dict:
    return {
        "rotation": _quat_xyzw_from_rot(_rotation_from_euler_xyz(ax, ay, az)),
        "translation": [tx, ty, tz],
    }


def _pose_to_rt(pose: dict) -> tuple[list[list[float]], list[float]]:
    x, y, z, w = pose["rotation"]
    tx, ty, tz = pose["translation"]
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z
    return (
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        [tx, ty, tz],
    )


def _transform_point(pose: dict, point: list[float]) -> list[float]:
    r, t = _pose_to_rt(pose)
    return [
        r[0][0] * point[0] + r[0][1] * point[1] + r[0][2] * point[2] + t[0],
        r[1][0] * point[0] + r[1][1] * point[1] + r[1][2] * point[2] + t[1],
        r[2][0] * point[0] + r[2][1] * point[1] + r[2][2] * point[2] + t[2],
    ]


def _project_point_pinhole(point_cam: list[float]) -> list[float] | None:
    x, y, z = point_cam
    if z <= 0.0:
        return None
    return [640.0 + 800.0 * x / z, 360.0 + 780.0 * y / z]


def _make_dataset() -> vc.PlanarDataset:
    board = [(i * 0.04, j * 0.04, 0.0) for i in range(8) for j in range(6)]
    poses = [
        _make_pose(0.02 * i, -0.18 + 0.04 * i, -0.20 + 0.06 * i, -0.04 + 0.02 * i, 0.02 - 0.01 * i, 0.55 + 0.04 * i)
        for i in range(8)
    ]
    views: list[vc.PlanarView] = []
    for pose in poses:
        points_3d = []
        points_2d = []
        for point in board:
            point_cam = _transform_point(pose, list(point))
            uv = _project_point_pinhole(point_cam)
            if uv is None:
                continue
            points_3d.append(point)
            points_2d.append(tuple(uv))
        views.append(vc.PlanarView(observation=vc.Observation(points_3d=points_3d, points_2d=points_2d)))
    return vc.PlanarDataset(views=views)


class ScheimpflugIntrinsicsTest(unittest.TestCase):
    def test_public_bindings_run(self) -> None:
        result = vc.run_scheimpflug_intrinsics(
            _make_dataset(),
            vc.ScheimpflugIntrinsicsCalibrationConfig(
                fix_scheimpflug={"tilt_x": False, "tilt_y": False}
            ),
        )
        self.assertGreaterEqual(result.mean_reproj_error, 0.0)
        self.assertIsInstance(result.camera, vc.PinholeBrownConradyScheimpflugCamera)
        self.assertIsInstance(result.camera.sensor, vc.ScheimpflugSensor)

    def test_invalid_config_maps_to_value_error(self) -> None:
        # Per R-07: invalid config from Python surfaces as ValueError, not
        # RuntimeError. RuntimeError is reserved for genuine runtime failures
        # (solver divergence, export/pythonize errors).
        with self.assertRaises(ValueError) as ctx:
            vc.run_scheimpflug_intrinsics(
                _make_dataset(),
                vc.ScheimpflugIntrinsicsCalibrationConfig(max_iters=0),
            )
        message = str(ctx.exception)
        self.assertIn("invalid config", message)
        self.assertIn("max_iters must be positive", message)

    def test_invalid_input_maps_to_value_error(self) -> None:
        # Per R-07: invalid input (too few views) surfaces as ValueError.
        dataset = _make_dataset()
        dataset.views = dataset.views[:2]
        with self.assertRaises(ValueError) as ctx:
            vc.run_scheimpflug_intrinsics(
                dataset,
                vc.ScheimpflugIntrinsicsCalibrationConfig(),
            )
        message = str(ctx.exception)
        self.assertIn("invalid input", message)
        self.assertIn("insufficient data", message)

    def test_high_level_api_rejects_mapping_inputs(self) -> None:
        with self.assertRaises(TypeError) as cfg_ctx:
            vc.run_scheimpflug_intrinsics(
                _make_dataset(),
                {"max_iters": 50},
            )
        self.assertIn("config must be ScheimpflugIntrinsicsCalibrationConfig", str(cfg_ctx.exception))

        with self.assertRaises(TypeError) as input_ctx:
            vc.run_scheimpflug_intrinsics(
                {"views": []},
                vc.ScheimpflugIntrinsicsCalibrationConfig(),
            )
        self.assertIn("input must be PlanarDataset", str(input_ctx.exception))


if __name__ == "__main__":
    unittest.main()
