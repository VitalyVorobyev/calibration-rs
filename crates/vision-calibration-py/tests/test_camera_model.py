"""CameraModel: vectorized projection with any camera model."""

from __future__ import annotations

import math
import unittest

import numpy as np

import vision_calibration as vc

from _fixtures import planar_calibration_dataset


def brown_conrady() -> vc.PinholeCamera:
    return vc.PinholeCamera(
        intrinsics=vc.PinholeIntrinsics(fx=800.0, fy=780.0, cx=640.0, cy=360.0),
        distortion=vc.BrownConradyDistortion(
            k1=-0.21, k2=0.08, k3=0.0, p1=0.001, p2=-0.0005, iters=8
        ),
    )


def scheimpflug() -> vc.PinholeScheimpflugCamera:
    return vc.PinholeScheimpflugCamera(
        intrinsics=vc.PinholeIntrinsics(fx=1200.0, fy=1190.0, cx=512.0, cy=384.0),
        distortion=vc.Division1Distortion(lambda_=-0.05),
        sensor=vc.ScheimpflugSensor(tilt_x=0.05, tilt_y=-0.03),
    )


def points(n: int, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-0.3, 0.3, size=(n, 2))
    z = rng.uniform(0.8, 2.0, size=(n, 1))
    return np.hstack([xy * z, z])


class CameraModelTest(unittest.TestCase):
    def test_pinhole_projection_is_k_times_the_normalized_point(self) -> None:
        cam = vc.CameraModel(
            vc.PinholeCamera(
                intrinsics=vc.PinholeIntrinsics(fx=800.0, fy=780.0, cx=640.0, cy=360.0),
                distortion=vc.NoDistortion(),
            )
        )
        pts = points(50)
        expected = np.stack(
            [800.0 * pts[:, 0] / pts[:, 2] + 640.0, 780.0 * pts[:, 1] / pts[:, 2] + 360.0], axis=1
        )
        np.testing.assert_allclose(cam.project(pts), expected, rtol=0, atol=1e-9)

    def test_unproject_and_undistort_invert_project(self) -> None:
        for camera in (brown_conrady(), scheimpflug()):
            cam = vc.CameraModel(camera)
            pts = points(200)
            pixels = cam.project(pts)
            self.assertEqual(pixels.shape, (200, 2))
            origins, directions = cam.unproject(pixels)
            self.assertEqual(origins.shape, (200, 3))
            np.testing.assert_allclose(origins, 0.0)
            # A pinhole ray's direction has z = 1: it points at the point.
            np.testing.assert_allclose(directions * pts[:, 2:3], pts, atol=1e-8)
            k = camera.intrinsics
            ideal = np.stack(
                [k.fx * pts[:, 0] / pts[:, 2] + k.cx, k.fy * pts[:, 1] / pts[:, 2] + k.cy], axis=1
            )
            np.testing.assert_allclose(cam.undistort(pixels), ideal, atol=1e-6)

    def test_a_point_behind_the_camera_projects_to_nan(self) -> None:
        cam = vc.CameraModel(brown_conrady())
        out = cam.project([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
        self.assertFalse(np.isnan(out[0]).any())
        self.assertTrue(np.isnan(out[1]).all())

    def test_single_vectors_and_shapes(self) -> None:
        cam = vc.CameraModel(brown_conrady())
        self.assertEqual(cam.project([0.1, 0.2, 1.0]).shape, (2,))
        origin, direction = cam.unproject((640.0, 360.0))
        self.assertEqual(origin.shape, (3,))
        np.testing.assert_allclose(direction, [0.0, 0.0, 1.0], atol=1e-12)
        self.assertEqual(cam.project(np.zeros((0, 3))).shape, (0, 2))
        with self.assertRaises(ValueError):
            cam.project([[1.0, 2.0]])
        with self.assertRaises(ValueError):
            cam.undistort(np.zeros((3, 3)))

    def test_large_arrays_release_the_gil_and_agree(self) -> None:
        cam = vc.CameraModel(brown_conrady())
        pts = points(5000)
        big = cam.project(pts)
        small = np.vstack([cam.project(pts[i : i + 100]) for i in range(0, 5000, 100)])
        np.testing.assert_array_equal(big, small)

    def test_built_from_payloads_and_results(self) -> None:
        payload = brown_conrady().to_payload()
        cam = vc.CameraModel(payload)
        self.assertEqual(cam.params["intrinsics"]["fx"], 800.0)
        result = vc.run_planar_intrinsics(planar_calibration_dataset())
        cam = vc.CameraModel(result.camera)
        # The result's residual table agrees with projecting its own poses.
        residuals = result.per_feature_residuals.target
        first = residuals.pose == 0
        pose = result.camera_se3_target[0]
        x, y, z, w = pose.rotation_xyzw
        rot = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ]
        )
        cam_points = residuals.target_xyz_m[first] @ rot.T + np.array(pose.translation_xyz)
        np.testing.assert_allclose(cam.project(cam_points), residuals.projected_px[first], atol=1e-9)

    def test_invalid_parameters_are_value_errors(self) -> None:
        payload = brown_conrady().to_payload()
        payload["intrinsics"]["fx"] = math.nan
        with self.assertRaises(ValueError):
            vc.CameraModel(payload)
        singular = {
            "projection": {"type": "pinhole"},
            "distortion": {"type": "none"},
            "sensor": {"type": "homography", "h": [[0.0] * 3] * 3},
            "intrinsics": {"fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0},
        }
        with self.assertRaises(ValueError):
            vc.CameraModel(singular)


if __name__ == "__main__":
    unittest.main()
