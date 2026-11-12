"""Runtime tests of ``vision_calibration.mvg``.

The scenes are built from the committed stereo and laser observations: a real
calibration (with covariance) is run once, a synthetic scene is projected
through the calibrated cameras, and the measurements are checked against the
scene's truth.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

import vision_calibration as vc
from vision_calibration import mvg

OBSERVATIONS = Path(__file__).resolve().parents[3] / "regression/observations"


def _pose_matrix(pose: vc.Pose) -> np.ndarray:
    x, y, z, w = pose.rotation_xyzw
    out = np.eye(4)
    out[:3, :3] = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
    out[:3, 3] = pose.translation_xyz
    return out


def _pose_payload(matrix: np.ndarray) -> dict[str, list[float]]:
    """A ``(4, 4)`` pose as the serde ``Iso3`` payload."""
    r = matrix[:3, :3]
    w = 0.5 * np.sqrt(max(1.0 + np.trace(r), 1e-12))
    x = (r[2, 1] - r[1, 2]) / (4 * w)
    y = (r[0, 2] - r[2, 0]) / (4 * w)
    z = (r[1, 0] - r[0, 1]) / (4 * w)
    return {"rotation": [x, y, z, w], "translation": matrix[:3, 3].tolist()}


def _rotation(rx: float, ry: float, rz: float) -> np.ndarray:
    cx, sx, cy, sy, cz, sz = np.cos(rx), np.sin(rx), np.cos(ry), np.sin(ry), np.cos(rz), np.sin(rz)
    rot_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    rot_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rot_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rot_z @ rot_y @ rot_x


def _transform(rotation: np.ndarray, translation: tuple[float, float, float]) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = rotation
    out[:3, 3] = translation
    return out


def _apply(pose: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ pose[:3, :3].T + pose[:3, 3]


def _stereo_result() -> vc.RigExtrinsicsExport:
    payload = json.loads((OBSERVATIONS / "stereo_rig.json").read_text())["input"]
    dataset = vc.RigExtrinsicsInput(
        num_cameras=payload["num_cameras"],
        views=[
            vc.RigExtrinsicsView(
                cameras=[
                    None
                    if cam is None
                    else vc.Observation(points_3d=cam["points_3d"], points_2d=cam["points_2d"])
                    for cam in view["obs"]["cameras"]
                ]
            )
            for view in payload["views"]
        ],
    )
    return vc.rig_extrinsics.run_calibration(dataset)


def _laser_result() -> vc.LaserlineDeviceExport:
    payload = json.loads((OBSERVATIONS / "synth_laserline.json").read_text())["input"]
    return vc.laserline_device.run_calibration(
        vc.LaserlineDeviceInput(
            views=[
                vc.LaserlineView(
                    observation=vc.Observation(
                        points_3d=view["obs"]["points_3d"], points_2d=view["obs"]["points_2d"]
                    ),
                    laser_pixels=view["meta"]["laser_pixels"],
                )
                for view in payload
            ]
        )
    )


class _Scene:
    """A calibrated stereo rig and a synthetic scene seen through it."""

    def __init__(self) -> None:
        self.result = _stereo_result()
        self.models = [vc.CameraModel(c) for c in self.result.cameras]
        self.cam_se3_rig = [_pose_matrix(p) for p in self.result.cam_se3_rig]
        self.noise = mvg.reprojection_noise(self.result, 0)
        self.sigma = self.noise.xx**0.5
        self.rng = np.random.default_rng(7)

    def pixel(self, camera: int, rig_se3_world: np.ndarray, points: np.ndarray) -> np.ndarray:
        """Noisy pixels of world ``points`` in rig camera ``camera``."""
        in_camera = _apply(self.cam_se3_rig[camera] @ rig_se3_world, points)
        pixels = self.models[camera].project(in_camera)
        return pixels + self.rng.normal(0.0, self.sigma, pixels.shape)

    def tracks(
        self, rig_se3_world: np.ndarray, points: np.ndarray
    ) -> list[list[mvg.RigObservation]]:
        """One track per point: every camera at every capture."""
        out: list[list[mvg.RigObservation]] = []
        pixels = {
            (k, c): self.pixel(c, pose, points)
            for k, pose in enumerate(rig_se3_world)
            for c in range(2)
        }
        for i in range(len(points)):
            out.append(
                [
                    mvg.RigObservation(k, c, tuple(pixels[k, c][i]), self.noise)
                    for k in range(len(rig_se3_world))
                    for c in range(2)
                ]
            )
        return out


def _board(rows: int = 6, cols: int = 8, pitch: float = 0.03) -> np.ndarray:
    """A planar board of corners in its own frame, z = 0."""
    grid = np.array([(c * pitch, r * pitch, 0.0) for r in range(rows) for c in range(cols)])
    return grid - grid.mean(axis=0)


def _scatter(n: int = 24) -> np.ndarray:
    """Points spread in depth in front of the rig, in the rig frame."""
    k = np.arange(n)
    return np.stack(
        [0.35 * np.sin(0.9 * k), 0.25 * np.cos(1.7 * k), 1.0 + 0.4 * np.sin(2.3 * k) ** 2], axis=1
    )


class MvgTest(unittest.TestCase):
    scene: _Scene

    @classmethod
    def setUpClass(cls) -> None:
        cls.scene = _Scene()

    # --- noise and errors --------------------------------------------------

    def test_reprojection_noise_is_the_calibrations_level(self) -> None:
        noise = self.scene.noise
        self.assertGreater(noise.xx, 0.0)
        self.assertEqual(noise.xx, noise.yy)
        self.assertEqual(noise.xy, 0.0)
        # The export dict gives the same.
        self.assertEqual(mvg.reprojection_noise(self.scene.result.export, 1).xx, noise.xx)

    def test_errors_map_to_value_and_runtime_errors(self) -> None:
        with self.assertRaises(ValueError):
            mvg.reprojection_noise(self.scene.result, camera=5)
        track = [mvg.RigObservation(0, 5, (10.0, 10.0), self.scene.noise)] * 2
        with self.assertRaises(ValueError):
            mvg.triangulate(self.scene.result, np.eye(4), [track])
        with self.assertRaises(ValueError):
            mvg.triangulate(self.scene.result, np.eye(4), [[track[0]]])  # one observation
        with self.assertRaises(ValueError):  # not a rotation
            mvg.triangulate(self.scene.result, 2.0 * np.eye(4), [track])
        with self.assertRaises(TypeError):
            mvg.triangulate({"kind": "nope"}, np.eye(4), [track])
        with self.assertRaises(ValueError):
            mvg.rectify(self.scene.result, (0, 0))
        # Zero parallax: both observations on one ray from a single camera.
        same = mvg.RigObservation(0, 0, (500.0, 300.0), self.scene.noise)
        with self.assertRaises((ValueError, RuntimeError)):
            mvg.triangulate(self.scene.result, np.eye(4), [[same, same]])

    # --- triangulation and metrology ---------------------------------------

    def test_a_board_length_is_measured_within_its_sigma(self) -> None:
        scene = self.scene
        board = _board()
        # The board sits 1 m ahead, tilted, in the rig frame.
        board_pose = _transform(_rotation(0.3, -0.2, 0.1), (0.02, -0.01, 1.0))
        points = _apply(board_pose, board)
        tracks = scene.tracks(np.eye(4)[None], points)
        found = mvg.triangulate(scene.result, np.eye(4), tracks)

        self.assertEqual(len(found.points), len(points))
        self.assertTrue(all(p.in_front and p.parallax_deg > 0.0 for p in found.points))
        self.assertTrue(found.uncertainty.available, found.uncertainty.reason)
        positions = np.array([p.point for p in found.points])
        self.assertEqual(positions.shape, (len(points), 3))
        self.assertLess(np.abs(positions - points).max(), 5e-3)

        # point_covariance and cross_covariance are the uncertainty's blocks.
        u = found.uncertainty
        full = np.array(u.covariance)
        i, j = 3, 17
        block = lambda a, b: full[3 * a : 3 * a + 3, 3 * b : 3 * b + 3]  # noqa: E731
        self.assertEqual(u.labels[3 * i : 3 * i + 3], ["p3.x", "p3.y", "p3.z"])
        np.testing.assert_array_equal(found.point_covariance(i), block(i, i))
        np.testing.assert_array_equal(found.cross_covariance(i, j), block(i, j))
        np.testing.assert_array_equal(found.cross_covariance(j, i), block(j, i))
        np.testing.assert_allclose(found.cross_covariance(i, j), found.cross_covariance(j, i).T)
        # The calibration is shared, so the points are correlated.
        self.assertGreater(np.abs(found.cross_covariance(i, j)).max(), 0.0)
        self.assertIsNone(found.point_covariance(len(points)))

        # Two corners one board-width apart: the length and its σ.
        measured = mvg.PointSet.from_triangulation(found)
        self.assertEqual(len(measured), len(points))
        a, b = 0, 7
        true_length = float(np.linalg.norm(points[b] - points[a]))
        length = mvg.distance(measured, a, b)
        self.assertGreater(length.sigma, 0.0)
        self.assertLess(abs(length.value - true_length), 3.0 * length.sigma)
        self.assertAlmostEqual(
            length.value, float(np.linalg.norm(positions[b] - positions[a])), places=12
        )
        # The σ is the first-order propagation of the joint covariance.
        d = (positions[b] - positions[a]) / np.linalg.norm(positions[b] - positions[a])
        grad = np.zeros(3 * len(points))
        grad[3 * b : 3 * b + 3], grad[3 * a : 3 * a + 3] = d, -d
        self.assertAlmostEqual(length.sigma, float(np.sqrt(grad @ full @ grad)), places=12)

        # An angle at a corner of the board.
        right_angle = mvg.angle(measured, 1, 0, 8)
        self.assertAlmostEqual(right_angle.value, np.pi / 2, delta=3.0 * right_angle.sigma + 1e-3)
        self.assertGreater(right_angle.sigma, 0.0)

        # The corners are coplanar: the fit's rms is small and the plane's own
        # points are on it.
        plane = mvg.fit_plane(measured, list(range(len(points))))
        true_normal = board_pose[:3, 2]
        self.assertAlmostEqual(abs(float(plane.normal @ true_normal)), 1.0, places=2)
        self.assertEqual(plane.covariance.shape, (3, 3))
        self.assertEqual(plane.tangent_basis.shape, (2, 3))
        self.assertLess(plane.rms, 3e-3)
        off_plane = mvg.point_to_plane(measured, plane, 0)
        self.assertLess(abs(off_plane.value), 3.0 * max(off_plane.sigma, 1e-6) + 3e-3)
        with self.assertRaises(ValueError):
            mvg.distance(measured, 0, len(points))
        with self.assertRaises(ValueError):
            mvg.fit_plane(measured, [0, 1])

    def test_a_point_set_checks_its_shape(self) -> None:
        with self.assertRaises(ValueError):
            mvg.PointSet(np.zeros((2, 3)), np.zeros((3, 3)))
        independent = mvg.PointSet(
            np.array([[0.0, 0.0, 0.7], [0.1, 0.0, 0.7]]), 4e-8 * np.eye(6)
        )
        d = mvg.distance(independent, 0, 1)
        self.assertAlmostEqual(d.value, 0.1, places=12)
        self.assertAlmostEqual(d.sigma, 2.0**0.5 * 2e-4, places=9)

    def test_a_triangulation_without_covariance_gives_no_point_set(self) -> None:
        # A calibration whose covariance is dropped still triangulates.
        export = {k: v for k, v in self.scene.result.export.items() if k != "uncertainty"}
        export["uncertainty"] = {
            "status": "unavailable",
            "reason": {"kind": "not_computed"},
        }
        points = _scatter(4)
        tracks = self.scene.tracks(np.eye(4)[None], points)
        found = mvg.triangulate(export, np.eye(4), tracks)
        self.assertFalse(found.uncertainty.available)
        self.assertIsNone(found.point_covariance(0))
        with self.assertRaises(ValueError):
            mvg.PointSet.from_triangulation(found)
        with self.assertRaises(ValueError):
            mvg.reprojection_noise(export, 0)

    # --- object pose -------------------------------------------------------

    def test_object_pose_with_covariance(self) -> None:
        scene = self.scene
        board = _board()
        truth = _transform(_rotation(-0.2, 0.25, 0.05), (0.03, 0.02, 0.9))
        in_rig = _apply(truth, board)
        observations = [
            mvg.PoseObservation(
                camera=c,
                object=tuple(obj),
                pixel=tuple(px),
                noise=scene.noise,
            )
            for c in range(2)
            for obj, px in zip(board, scene.pixel(c, np.eye(4), in_rig))
        ]
        pose = mvg.estimate_pose(scene.result, observations)
        self.assertEqual(pose.rig_se3_object.shape, (4, 4))
        np.testing.assert_allclose(pose.rig_se3_object, truth, atol=5e-3)
        self.assertEqual(len(pose.inliers), len(observations))
        covariance = pose.pose_covariance()
        self.assertEqual(covariance.shape, (6, 6))
        np.testing.assert_allclose(covariance, covariance.T, atol=1e-18)
        self.assertTrue(np.all(np.diag(covariance) > 0.0))
        self.assertGreater(pose.reprojection_rms_px, 0.0)
        self.assertLess(pose.reprojection_rms_px, 3.0 * self.scene.sigma * 2**0.5)

        # RANSAC drops a gross outlier, and a robust loss runs.
        bad = list(observations)
        bad[5] = mvg.PoseObservation(0, bad[5].object, (900.0, 50.0), scene.noise)
        robust = mvg.estimate_pose(
            scene.result,
            bad,
            ransac=mvg.RansacOptions(thresh=3.0),
            loss=vc.robust_huber(2.0),
        )
        self.assertNotIn(5, robust.inliers)
        np.testing.assert_allclose(robust.rig_se3_object, truth, atol=5e-3)
        with self.assertRaises(ValueError):
            mvg.estimate_pose(scene.result, observations, ransac=mvg.RansacOptions(thresh=-1.0))

    # --- two views ---------------------------------------------------------

    def test_relative_pose_of_the_two_cameras(self) -> None:
        scene = self.scene
        points = _scatter(40)
        pixels = [scene.pixel(c, np.eye(4), points) for c in range(2)]
        matches = [
            mvg.PixelMatch(tuple(p0), scene.noise, tuple(p1), scene.noise)
            for p0, p1 in zip(*pixels)
        ]
        truth = scene.cam_se3_rig[1] @ np.linalg.inv(scene.cam_se3_rig[0])

        pose = mvg.recover_relative_pose(scene.result, (0, 1), matches)
        self.assertEqual(pose.second_se3_first.shape, (4, 4))
        np.testing.assert_allclose(
            pose.second_se3_first[:3, :3], truth[:3, :3], atol=1e-2
        )
        direction = truth[:3, 3] / np.linalg.norm(truth[:3, 3])
        self.assertGreater(float(pose.second_se3_first[:3, 3] @ direction), 0.999)
        self.assertEqual(len(pose.points), len(matches))
        self.assertFalse(pose.scene.is_pure_rotation)
        self.assertGreater(pose.scene.baseline_ratio, 0.0)
        self.assertEqual(pose.essential.shape, (3, 3))

        # Gross outliers are rejected by RANSAC.
        bad = list(matches)
        for k in (2, 9, 20):
            bad[k] = mvg.PixelMatch(matches[k].pt1, scene.noise, (40.0, 500.0), scene.noise)
        ransac = mvg.RansacOptions(thresh=4.0, min_inliers=20)
        estimate = mvg.estimate_essential(scene.result, (0, 1), bad, ransac)
        self.assertEqual(estimate.essential.shape, (3, 3))
        for k in (2, 9, 20):
            self.assertNotIn(k, estimate.inliers)
        self.assertGreaterEqual(len(estimate.inliers), 35)
        robust = mvg.recover_relative_pose_robust(scene.result, (0, 1), bad, ransac)
        self.assertGreater(float(robust.second_se3_first[:3, 3] @ direction), 0.99)
        self.assertEqual(robust.inliers, estimate.inliers)

        # The refined pose carries its covariance in the gauge |t| = 1: the
        # direction block has no variance along the translation.
        self.assertEqual(robust.uncertainty.labels[0], "rotation.rx")
        pose_cov = robust.pose_covariance()
        self.assertEqual(pose_cov.shape, (6, 6))
        np.testing.assert_allclose(pose_cov, pose_cov.T, atol=1e-18)
        self.assertTrue(np.all(np.diag(pose_cov) > 0.0))
        t = robust.second_se3_first[:3, 3]
        self.assertLess(abs(t @ pose_cov[3:, 3:] @ t), 1e-9 * np.trace(pose_cov[3:, 3:]))
        # The covariance of E's nine entries, rank 5.
        e_cov = estimate.essential_covariance()
        self.assertEqual(e_cov.shape, (9, 9))
        self.assertEqual(estimate.uncertainty.labels[0], "essential.e00")
        eig = np.linalg.eigvalsh(e_cov)
        self.assertEqual(int(np.sum(eig > 1e-9 * eig.max())), 5)

        # progress= reports the refinement's iterations, does not change the
        # result, and raising from it stops the call.
        reports: list[dict] = []
        observed = mvg.recover_relative_pose_robust(
            scene.result, (0, 1), bad, ransac, progress=reports.append
        )
        self.assertTrue(reports)
        self.assertTrue(all(r["kind"] == "iteration" for r in reports))
        np.testing.assert_array_equal(observed.second_se3_first, robust.second_se3_first)

        class Stop(Exception):
            pass

        def stop(_report: dict) -> None:
            raise Stop

        with self.assertRaises(Stop):
            mvg.estimate_essential(scene.result, (0, 1), bad, ransac, progress=stop)
        with self.assertRaises(Stop):
            mvg.recover_relative_pose_robust(scene.result, (0, 1), bad, ransac, progress=stop)

        with self.assertRaises(ValueError):
            mvg.recover_relative_pose(scene.result, (0, 1), matches[:3])
        with self.assertRaises(ValueError):
            mvg.recover_relative_pose(scene.result, (0, 7), matches)

    def test_robust_homography_recovers_the_inliers(self) -> None:
        rng = np.random.default_rng(11)
        truth = np.array([[1.05, 0.02, 12.0], [-0.03, 0.97, -7.0], [1e-5, -2e-5, 1.0]])
        first = rng.uniform(0.0, 600.0, (100, 2))
        homogeneous = np.c_[first, np.ones(len(first))] @ truth.T
        second = homogeneous[:, :2] / homogeneous[:, 2:]
        second += rng.normal(0.0, 0.2, second.shape)
        outliers = rng.choice(100, 20, replace=False)
        second[outliers] = rng.uniform(0.0, 600.0, (20, 2))
        pairs = np.stack([first, second], axis=1)
        estimate = mvg.estimate_homography(pairs, mvg.RansacOptions(thresh=2.0, min_inliers=20))
        self.assertEqual(estimate.homography.shape, (3, 3))
        self.assertTrue(set(outliers.tolist()).isdisjoint(estimate.inliers))
        self.assertGreaterEqual(len(estimate.inliers), 78)
        self.assertLess(estimate.inlier_rms, 1.0)
        h = estimate.homography / estimate.homography[2, 2]
        np.testing.assert_allclose(h, truth, rtol=5e-3, atol=0.2)
        as_list = [((a[0], a[1]), (b[0], b[1])) for a, b in pairs]
        again = mvg.estimate_homography(as_list, mvg.RansacOptions(thresh=2.0, min_inliers=20))
        self.assertEqual(again.inliers, estimate.inliers)
        with self.assertRaises(ValueError):
            mvg.estimate_homography(pairs[:3], mvg.RansacOptions(thresh=2.0))
        with self.assertRaises(ValueError):
            mvg.estimate_homography(first, mvg.RansacOptions(thresh=2.0))

    def test_option_defaults_match_the_rust_defaults(self) -> None:
        from dataclasses import fields

        from vision_calibration import _vision_calibration as native

        defaults = native.mvg_defaults()
        for name, cls in (("ransac", mvg.RansacOptions), ("block_match", mvg.BlockMatchOptions)):
            for f in fields(cls):
                if name == "ransac" and f.name == "thresh":
                    self.assertIs(f.default, __import__("dataclasses").MISSING)
                    continue
                self.assertAlmostEqual(f.default, defaults[name][f.name], places=6, msg=f.name)
                self.assertIs(type(f.default), type(defaults[name][f.name]), f.name)

    # --- bundle adjustment -------------------------------------------------

    def test_progress_reports_iterations_and_raising_stops_bundle_adjust(self) -> None:
        scene = self.scene
        points = _scatter(20)
        captures = np.stack(
            [np.eye(4), _transform(_rotation(0.02, -0.08, 0.01), (-0.10, 0.02, 0.05))]
        )
        tracks = scene.tracks(captures, points)
        start = points + scene.rng.normal(0.0, 0.02, points.shape)

        reports: list[dict] = []
        observed = mvg.bundle_adjust(
            scene.result, captures, start, tracks, scale="rig", progress=reports.append
        )
        self.assertTrue(reports)
        self.assertTrue(all(r["kind"] == "iteration" for r in reports))
        plain = mvg.bundle_adjust(scene.result, captures, start, tracks, scale="rig")
        np.testing.assert_array_equal(observed.points, plain.points)
        np.testing.assert_array_equal(observed.rig_se3_world, plain.rig_se3_world)

        class Stop(Exception):
            pass

        def stop(_report: dict) -> None:
            raise Stop

        with self.assertRaises(Stop):
            mvg.bundle_adjust(scene.result, captures, start, tracks, scale="rig", progress=stop)
        with self.assertRaises(Stop):
            mvg.triangulate(scene.result, captures, tracks, progress=stop)

    def test_bundle_adjustment_in_a_stated_gauge(self) -> None:
        scene = self.scene
        points = _scatter(20)
        captures = np.stack(
            [
                np.eye(4),
                _transform(_rotation(0.02, -0.08, 0.01), (-0.10, 0.02, 0.05)),
                _transform(_rotation(-0.03, 0.10, -0.02), (0.12, -0.03, 0.02)),
            ]
        )
        tracks = scene.tracks(captures, points)
        start = points + scene.rng.normal(0.0, 0.02, points.shape)
        start_captures = captures.copy()
        start_captures[1:, :3, 3] += 0.01

        adjusted = mvg.bundle_adjust(scene.result, start_captures, start, tracks, scale="rig")
        self.assertEqual(adjusted.rig_se3_world.shape, (3, 4, 4))
        self.assertEqual(adjusted.points.shape, points.shape)
        self.assertLess(np.abs(adjusted.points - points).max(), 1e-2)
        np.testing.assert_allclose(adjusted.rig_se3_world[0], np.eye(4), atol=1e-12)
        self.assertLess(adjusted.reprojection_rms_px, 3.0 * scene.sigma * 2**0.5)
        self.assertGreater(adjusted.num_iters, 0)
        self.assertEqual(len(adjusted.cameras), 2)
        self.assertEqual(adjusted.cam_se3_rig.shape, (2, 4, 4))
        self.assertTrue(adjusted.uncertainty.available, adjusted.uncertainty.reason)
        self.assertEqual(adjusted.point_covariance(0).shape, (3, 3))
        self.assertIsNone(adjusted.pose_covariance(0))  # the gauge
        self.assertEqual(adjusted.pose_covariance(1).shape, (6, 6))
        measured = mvg.PointSet.from_bundle_adjustment(adjusted)
        self.assertGreater(mvg.distance(measured, 0, 1).sigma, 0.0)

        # A stated distance fixes the scale instead of the rig's baseline.
        true_length = float(np.linalg.norm(points[0] - points[1]))
        by_distance = mvg.bundle_adjust(
            scene.result,
            start_captures,
            start,
            tracks,
            scale=("distance", (0, 1), true_length, 1e-4),
            calibration="consider",
            loss=vc.robust_cauchy(3.0),
            max_iters=50,
        )
        self.assertLess(np.abs(by_distance.points - points).max(), 2e-2)

        with self.assertRaises(ValueError):
            mvg.bundle_adjust(scene.result, start_captures, start, tracks, scale="meters")
        with self.assertRaises(ValueError):
            mvg.bundle_adjust(
                scene.result, start_captures, start, tracks, scale="rig", calibration="free"
            )

    # --- laser -------------------------------------------------------------

    def test_laser_points_on_the_calibrated_plane(self) -> None:
        result = _laser_result()
        camera = vc.CameraModel(result.camera)
        plane = result.laser_plane
        normal = np.array(plane.normal_xyz)
        noise = mvg.reprojection_noise(result, 0)
        # Points on the plane n·x + d = 0, seen across the image.
        u = np.cross(normal, [0.0, 1.0, 0.0])
        u /= np.linalg.norm(u)
        v = np.cross(normal, u)
        centre = -plane.distance * normal
        offsets = np.linspace(-0.05, 0.05, 9)
        points = np.array([centre + a * u + b * v for a in offsets for b in offsets[::4]])
        rays = camera.project(points)
        self.assertTrue(np.all(np.isfinite(rays)))
        observations = [mvg.LaserObservation(0, tuple(px), noise) for px in rays]

        found = mvg.triangulate_laser(result, observations)
        self.assertEqual(len(found.points), len(points))
        positions = np.array([p.point for p in found.points])
        np.testing.assert_allclose(positions, points, atol=1e-6)
        self.assertTrue(all(p.camera == 0 and p.incidence_deg > 0.0 for p in found.points))
        self.assertTrue(found.uncertainty.available, found.uncertainty.reason)
        covariance = found.point_covariance(0)
        self.assertEqual(covariance.shape, (3, 3))
        self.assertTrue(np.all(np.diag(covariance) > 0.0))
        measured = mvg.PointSet.from_laser_triangulation(found)
        length = mvg.distance(measured, 0, 1)
        self.assertAlmostEqual(length.value, float(np.linalg.norm(points[0] - points[1])), places=6)
        self.assertGreater(length.sigma, 0.0)

        with self.assertRaises(ValueError):
            mvg.triangulate_laser(result, [mvg.LaserObservation(3, (10.0, 10.0), noise)])
        with self.assertRaises(ValueError):  # a stereo rig without a laser plane
            mvg.triangulate_laser(
                self.scene.result, [mvg.LaserObservation(0, (500.0, 300.0), noise)]
            )

    def test_laser_points_in_the_robot_frames(self) -> None:
        result = _laser_result()
        export = dict(result.export)
        # The laser calibration as a rig on a robot's gripper: its hand-eye is
        # not in the covariance (held exact), the robot's pose has a σ.
        gripper_se3_rig = _transform(_rotation(0.1, -0.2, 0.3), (0.02, -0.01, 0.15))
        robot_pose = _transform(_rotation(0.3, 0.1, -0.5), (0.5, 0.1, 0.4))
        on_robot = {
            "kind": "rig_handeye_laserline",
            "schema_version": 1,
            "laser_planes_rig": [export["laser_plane"]],
            "laser_planes_cam": [export["laser_plane"]],
            "per_cam_stats": [],
            "cameras": [export["camera"]],
            "cam_se3_rig": [{"rotation": [0.0, 0.0, 0.0, 1.0], "translation": [0.0, 0.0, 0.0]}],
            "rig_se3_target": [],
            "handeye": {
                "mode": "eye_in_hand",
                "gripper_se3_rig": _pose_payload(gripper_se3_rig),
                "base_se3_target": _pose_payload(_transform(_rotation(0.0, 0.0, 0.0), (0.0, 0.0, 1.0))),
            },
            "mean_reproj_error": 0.1,
            "per_cam_reproj_errors": [0.1],
            "uncertainty": export["uncertainty"],
        }
        camera = vc.CameraModel(result.camera)
        plane = result.laser_plane
        normal = np.array(plane.normal_xyz)
        u = np.cross(normal, [0.0, 1.0, 0.0])
        u /= np.linalg.norm(u)
        centre = -plane.distance * normal
        points = np.array([centre + a * u for a in (-0.03, 0.0, 0.03)])
        noise = mvg.reprojection_noise(result, 0)
        observations = [
            mvg.LaserObservation(0, tuple(px), noise) for px in camera.project(points)
        ]

        in_rig = mvg.triangulate_laser(on_robot, observations)
        in_gripper = mvg.triangulate_laser(on_robot, observations, mvg.RobotCaptures("gripper"))
        robot = [mvg.RobotPose.isotropic(robot_pose, 1e-3, 0.5e-3)]
        in_base = mvg.triangulate_laser(on_robot, observations, mvg.RobotCaptures("base", robot))
        rig_points = np.array([p.point for p in in_rig.points])
        np.testing.assert_allclose(
            np.array([p.point for p in in_gripper.points]), _apply(gripper_se3_rig, rig_points),
            atol=1e-9,
        )
        np.testing.assert_allclose(
            np.array([p.point for p in in_base.points]),
            _apply(robot_pose @ gripper_se3_rig, rig_points),
            atol=1e-9,
        )
        # The robot's σ adds to the points'; an exact robot pose adds nothing.
        exact = mvg.triangulate_laser(
            on_robot, observations, mvg.RobotCaptures("base", [mvg.RobotPose(robot_pose)])
        )
        trace = lambda found: float(np.trace(found.point_covariance(0)))  # noqa: E731
        self.assertGreater(trace(in_base), 1.5 * trace(exact))
        self.assertAlmostEqual(trace(exact), trace(in_gripper), delta=1e-6 * trace(exact))
        # Two captures of a scan: each observation names its robot pose.
        second = [mvg.LaserObservation(0, o.pixel, noise, capture=1) for o in observations]
        scan = mvg.triangulate_laser(
            on_robot, observations + second, mvg.RobotCaptures("base", robot * 2)
        )
        self.assertEqual(len(scan.points), 6)

        with self.assertRaises(ValueError):  # the base frame needs the robot's pose
            mvg.triangulate_laser(on_robot, observations, mvg.RobotCaptures("base"))
        with self.assertRaises(ValueError):  # the rig is fixed in the gripper frame
            mvg.triangulate_laser(on_robot, observations, mvg.RobotCaptures("gripper", robot))
        with self.assertRaises(ValueError):  # a calibration without a hand-eye
            mvg.triangulate_laser(result, observations, mvg.RobotCaptures("gripper"))
        with self.assertRaises(ValueError):
            mvg.RobotCaptures("flange").to_payload()
        with self.assertRaises(ValueError):  # not positive semi-definite
            mvg.triangulate_laser(
                on_robot,
                observations,
                mvg.RobotCaptures("base", [mvg.RobotPose(robot_pose, -np.eye(6))]),
            )

    # --- stereo ------------------------------------------------------------

    def test_rectification(self) -> None:
        scene = self.scene
        rect = mvg.rectify(scene.result, (0, 1))
        self.assertEqual(rect.cameras, (0, 1))
        baseline = float(np.linalg.norm(scene.cam_se3_rig[1][:3, 3]))
        self.assertAlmostEqual(rect.baseline, baseline, places=9)
        k = rect.k_rect
        self.assertGreater(k.fx, 0.0)
        half = mvg.rectify(scene.result, (0, 1), scale=0.5)
        self.assertAlmostEqual(half.k_rect.fx, 0.5 * k.fx)
        explicit = mvg.rectify(scene.result, k_rect=vc.PinholeIntrinsics(600.0, 600.0, 500.0, 300.0))
        self.assertEqual(explicit.k_rect.fx, 600.0)
        self.assertEqual(rect.rect_r_cam(0).shape, (3, 3))
        np.testing.assert_allclose(rect.rect_r_cam(1) @ rect.rect_r_cam(1).T, np.eye(3), atol=1e-12)

        # Rectified pixels of one point share a row; the disparity gives its depth.
        point = np.array([0.05, -0.04, 1.3])
        pixels = [
            scene.models[c].project(_apply(scene.cam_se3_rig[c], point[None]))[0] for c in range(2)
        ]
        left = rect.rectified_pixel(0, pixels[0])
        right = rect.rectified_pixel(1, pixels[1])
        self.assertAlmostEqual(left[1], right[1], places=6)
        disparity = left[0] - right[0]
        self.assertGreater(disparity, 0.0)
        seen = rect.point(left, disparity)
        np.testing.assert_allclose(seen, point, atol=1e-6)
        homogeneous = rect.disparity_to_depth() @ np.array([left[0], left[1], disparity, 1.0])
        np.testing.assert_allclose(homogeneous[:3] / homogeneous[3], seen @ rect.rect_r_cam(0).T)
        sigma_d = 0.2
        covariance = rect.disparity_covariance(left, disparity, sigma_d)
        self.assertEqual(covariance.shape, (3, 3))
        np.testing.assert_allclose(
            covariance, np.outer(seen, seen) * (sigma_d / disparity) ** 2, rtol=1e-9
        )
        self.assertIsNone(rect.point(left, -1.0))
        # raw_pixel inverts rectified_pixel.
        raw = rect.raw_pixel(0, left)
        np.testing.assert_allclose(raw, pixels[0], atol=1e-6)

        # Triangulating the rectified match returns the point with a covariance.
        found = rect.triangulate(scene.result, [(left, disparity)], scene.noise)
        np.testing.assert_allclose(found.points[0].point, point, atol=1e-4)
        self.assertTrue(found.uncertainty.available)
        self.assertGreater(found.point_covariance(0)[2, 2], 0.0)

        with self.assertRaises(ValueError):
            rect.rectified_pixel(2, pixels[0])
        with self.assertRaises(ValueError):
            rect.triangulate(_laser_result(), [(left, disparity)], scene.noise)

    def test_dense_cloud_with_the_calibrations_share(self) -> None:
        scene = self.scene
        rect = mvg.rectify(scene.result, (0, 1))
        point = np.array([0.05, -0.04, 1.3])
        pixels = [
            scene.models[c].project(_apply(scene.cam_se3_rig[c], point[None]))[0] for c in range(2)
        ]
        left = rect.rectified_pixel(0, pixels[0])
        right = rect.rectified_pixel(1, pixels[1])
        x, y = int(round(left[0])), int(round(left[1]))
        disparity = np.full((y + 10, x + 10), np.nan, dtype=np.float32)
        disparity[y, x] = left[0] - right[0]
        disparity[y, x + 5] = left[0] - right[0] - 1.0
        disparity[0, 0] = -2.0  # not a point
        cloud = rect.dense_cloud(scene.result, disparity, sigma_d=0.2)
        self.assertEqual(len(cloud), 2)
        self.assertIsNone(cloud.covariance_unavailable)
        self.assertEqual(cloud.sigma_d, 0.2)
        np.testing.assert_array_equal(cloud.pixels, [[x, y], [x + 5, y]])
        self.assertEqual(cloud.points.shape, (2, 3))
        # The left camera is the rig frame here: the point is `point` at the grid pixel.
        np.testing.assert_allclose(
            cloud.points[0], rect.point((x, y), float(disparity[y, x])), atol=1e-12
        )
        covariance = cloud.point_covariance(0)
        disparity_only = rect.disparity_covariance((x, y), float(disparity[y, x]), 0.2)
        self.assertGreater(covariance[2, 2], disparity_only[2, 2])
        joint = cloud.covariance([1, 0])
        self.assertEqual(joint.shape, (6, 6))
        np.testing.assert_allclose(joint[3:, 3:], covariance, rtol=1e-12)
        self.assertGreater(abs(joint[2, 5]), 0.0)  # the shared calibration correlates them
        points = mvg.PointSet.from_dense_cloud(cloud, [0, 1])
        length = mvg.distance(points, 0, 1)
        self.assertGreater(length.sigma, 0.0)
        with self.assertRaises(ValueError):
            cloud.covariance([0, 2])
        with self.assertRaises(ValueError):
            rect.dense_cloud(scene.result, disparity, sigma_d=-1.0)
        with self.assertRaises(ValueError):
            rect.dense_cloud(_laser_result(), disparity, sigma_d=0.2)

    def test_rectify_image_keeps_shape_and_dtype(self) -> None:
        rect = mvg.rectify(self.scene.result, (0, 1))
        raw = (np.arange(576 * 1024) % 251).astype(np.uint8).reshape(576, 1024)
        out = rect.rectify_image(0, raw, 320, 200)
        self.assertEqual(out.shape, (200, 320))
        self.assertEqual(out.dtype, np.uint8)
        self.assertGreater(int(out.max()), 0)
        with self.assertRaises(ValueError):
            rect.rectify_image(0, raw.astype(np.float32), 320, 200)
        with self.assertRaises(ValueError):
            rect.rectify_image(0, raw, 0, 200)

    def test_block_matching_finds_a_shifted_texture(self) -> None:
        rng = np.random.default_rng(3)
        left = rng.integers(0, 256, (60, 120), dtype=np.uint8)
        shift = 6
        right = np.roll(left, -shift, axis=1)  # right(x) = left(x + shift): disparity +shift
        options = mvg.BlockMatchOptions(num_disparities=16, block_size=9)
        disparity = mvg.match_block(left, right, options)
        self.assertEqual(disparity.shape, left.shape)
        self.assertEqual(disparity.dtype, np.float32)
        valid = np.isfinite(disparity)
        self.assertGreater(valid.mean(), 0.3)
        self.assertLess(np.abs(disparity[valid] - shift).max(), 0.2)
        self.assertTrue(np.isnan(disparity[:, :4]).all())
        with self.assertRaises(ValueError):
            mvg.match_block(left, right[:, :100])
        with self.assertRaises(ValueError):
            mvg.match_block(left, right, mvg.BlockMatchOptions(block_size=4))
        with self.assertRaises(ValueError):
            mvg.match_block(left.astype(np.float32), right)


if __name__ == "__main__":
    unittest.main()
