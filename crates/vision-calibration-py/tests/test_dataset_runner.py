"""Manifest-driven runs, target detection and the detection cache.

The image runs use synthetic images rendered by ``_images``; the kuka_1 runs
use the committed images, and skip when the checkout holds Git LFS pointers
instead of the images.
"""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np

import vision_calibration as vc
from vision_calibration import dataset_runner

import _images

KUKA = Path(__file__).resolve().parents[3] / "data/kuka_1"


def kuka_images_present() -> bool:
    """Whether the kuka_1 images are real PNGs, not Git LFS pointers."""
    first = KUKA / "01.png"
    return first.exists() and first.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


class DetectTargetTest(unittest.TestCase):
    def test_finds_every_inner_corner(self) -> None:
        r, t = _images.board_poses(1)[0]
        features = vc.detect_target(_images.render_board(r, t), _images.TARGET)
        self.assertEqual(len(features), 35)
        self.assertEqual(features.image_xy.shape, (35, 2))
        np.testing.assert_allclose(features.world_xyz[:, 2], 0.0)
        # Target points lie on the square grid; pixels inside the image.
        steps = features.world_xyz[:, :2] / _images.SQUARE
        np.testing.assert_allclose(steps, np.round(steps), atol=1e-9)
        self.assertTrue((features.image_xy >= 0).all())
        self.assertTrue((features.image_xy[:, 0] < _images.WIDTH).all())

    def test_a_blank_image_has_no_features(self) -> None:
        features = vc.detect_target(np.full((120, 160), 128, np.uint8), _images.TARGET)
        self.assertEqual(len(features), 0)
        self.assertEqual(features.to_observation().points_2d, [])

    def test_rejects_non_grayscale_images(self) -> None:
        with self.assertRaises(TypeError):
            vc.detect_target(np.zeros((10, 10, 3), np.uint8), _images.TARGET)
        with self.assertRaises(TypeError):
            vc.detect_target(np.zeros((10, 10), np.float32), _images.TARGET)

    def test_an_impossible_target_is_a_run_error(self) -> None:
        target = {
            "kind": "charuco",
            "rows": 12,
            "cols": 12,
            "square_size_m": 0.02,
            "marker_size_m": 0.015,
            "dictionary": "DICT_TYPO_99",
        }
        with self.assertRaises(vc.RunError) as ctx:
            vc.detect_target(np.zeros((10, 10), np.uint8), target)
        self.assertEqual(ctx.exception.kind, "invalid_target_config")

    def test_features_feed_a_session(self) -> None:
        views = []
        for r, t in _images.board_poses(6):
            features = vc.detect_target(_images.render_board(r, t), _images.TARGET)
            views.append(vc.PlanarView(observation=features.to_observation()))
        result = vc.run_planar_intrinsics(vc.PlanarIntrinsicsInput(views=views))
        self.assertAlmostEqual(result.camera.intrinsics.fx, _images.FX, delta=0.03 * _images.FX)

    def test_corner_covariance_is_opt_in(self) -> None:
        r, t = _images.board_poses(1)[0]
        image = _images.render_board(r, t)
        plain = vc.detect_target(image, _images.TARGET)
        self.assertIsNone(plain.cov_px)
        detector = {"chess_corners": {"covariance": {"image_noise": {"known": 2.0}}}}
        features = vc.detect_target(image, _images.TARGET, detector)
        assert features.cov_px is not None
        self.assertEqual(features.cov_px.shape, (len(features), 3))
        np.testing.assert_array_equal(features.image_xy, plain.image_xy)
        obs = features.to_observation()
        assert obs.cov_px is not None
        self.assertEqual(len(obs.cov_px), len(features))
        self.assertIsNone(plain.to_observation().cov_px)
        with self.assertRaises(ValueError):
            vc.Features(image_xy=np.zeros((2, 2)), world_xyz=np.zeros((2, 3)), cov_px=np.zeros((3, 3)))

    def test_ambiguous_detections_are_dropped(self) -> None:
        features = vc.Features(
            image_xy=np.array([[1.0, 2.0], [3.0, 4.0]]),
            world_xyz=np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]),
        )
        self.assertEqual(len(vc.reject_ambiguous_detection(features)), 2)
        duplicated = vc.Features(
            image_xy=np.array([[1.0, 2.0], [1.0, 2.0]]), world_xyz=features.world_xyz
        )
        self.assertEqual(len(vc.reject_ambiguous_detection(duplicated)), 0)
        with self.assertRaises(ValueError):
            vc.Features(image_xy=np.zeros((2, 2)), world_xyz=np.zeros((3, 3)))


class DetectionCacheTest(unittest.TestCase):
    def test_entries_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = vc.FsDetectionCache(Path(tmp) / "cache")
            self.assertEqual(cache.root, Path(tmp) / "cache")
            key = vc.CacheKey.from_inputs(b"image bytes", "chessboard", {"rows": 5, "cols": 7})
            same = vc.CacheKey.from_inputs(b"image bytes", "chessboard", {"cols": 7, "rows": 5})
            self.assertEqual(key, same)
            self.assertNotEqual(key, vc.CacheKey.from_inputs(b"other", "chessboard", {}))
            self.assertIsNone(cache.get(key))
            features = vc.Features(
                image_xy=np.array([[10.5, 20.25]]), world_xyz=np.array([[0.0, 0.03, 0.0]])
            )
            cache.put(key, features)
            self.assertTrue((cache.root / key.file_name()).exists())
            cached = cache.get(key)
            assert cached is not None
            np.testing.assert_array_equal(cached.image_xy, features.image_xy)
            np.testing.assert_array_equal(cached.world_xyz, features.world_xyz)
            self.assertIsNone(cached.cov_px)

    def test_entries_keep_corner_covariances(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = vc.FsDetectionCache(tmp)
            key = vc.CacheKey.from_inputs(b"image bytes", "chessboard", {})
            features = vc.Features(
                image_xy=np.array([[10.5, 20.25]]),
                world_xyz=np.array([[0.0, 0.03, 0.0]]),
                cov_px=np.array([[0.04, 0.001, 0.09]]),
            )
            cache.put(key, features)
            cached = cache.get(key)
            assert cached is not None
            np.testing.assert_array_equal(cached.cov_px, features.cov_px)
            self.assertEqual(len(vc.reject_ambiguous_detection(features).cov_px), 1)


class SniffFolderTest(unittest.TestCase):
    def test_drafts_a_manifest(self) -> None:
        spec = vc.sniff_folder(KUKA)
        self.assertEqual(len(spec["cameras"]), 1)
        self.assertIn("target", spec["_unresolved"])

    def test_a_missing_folder_is_a_sniff_error(self) -> None:
        with self.assertRaises(vc.SniffError) as ctx:
            vc.sniff_folder(KUKA / "no_such_folder")
        self.assertEqual(ctx.exception.kind, "not_a_directory")


class ManifestRunTest(unittest.TestCase):
    def test_planar_images_calibrate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = _images.write_planar_dataset(root, views=8)
            cache = vc.FsDetectionCache(root / "cache")
            run = dataset_runner.build_planar_input(spec, base_dir=root, cache=cache)
            self.assertEqual((run.usable_views, run.total_views), (8, 8))
            self.assertEqual([p.name for p in run.view_paths][:2], ["view_00.png", "view_01.png"])
            result = vc.calibrate_dataset(spec, base_dir=root, cache=cache)
            assert isinstance(result, vc.PlanarIntrinsicsExport)
            k = result.camera.intrinsics
            self.assertAlmostEqual(k.fx, _images.FX, delta=0.03 * _images.FX)
            self.assertAlmostEqual(k.fy, _images.FY, delta=0.03 * _images.FY)
            self.assertLess(result.mean_reproj_error, 0.5)

    def test_left_out_views_carry_their_reason(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = _images.write_planar_dataset(root, views=4)
            _images.write_png(root / "view_99.png", np.full((480, 640), 128, dtype=np.uint8))
            run = dataset_runner.build_planar_input(spec, base_dir=root)
            self.assertEqual((run.usable_views, run.total_views), (4, 5))
            self.assertEqual(len(run.rejected), 1)
            left_out = run.rejected[0]
            self.assertEqual((left_out.camera, left_out.path.name), (0, "view_99.png"))
            self.assertEqual(left_out.reason, "too_few_features")

    def test_a_manifest_file_resolves_against_its_folder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = _images.write_planar_dataset(root, views=4)
            manifest = root / "dataset.toml"
            manifest.write_text(
                "version = 1\n"
                'topology = "planar_intrinsics"\n'
                "[[cameras]]\n"
                'id = "cam0"\n'
                'images = { kind = "glob", pattern = "view_*.png" }\n'
                "[target]\n"
                'kind = "chessboard"\nrows = 5\ncols = 7\nsquare_size_m = 0.03\n'
            )
            run = dataset_runner.build_planar_input(manifest)
            self.assertEqual(run.usable_views, 4)
            self.assertEqual(len(run.input["views"]), len(spec["cameras"]) * 4)

    def test_laser_images_calibrate_with_a_python_extractor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = _images.write_laserline_dataset(root, views=6)
            extractor = _images.ColumnPeakExtractor()
            with self.assertRaises(ValueError):  # the laser topologies need an extractor
                vc.calibrate_dataset(spec, base_dir=root)
            result = vc.calibrate_dataset(spec, base_dir=root, laser_extractor=extractor)
            assert isinstance(result, vc.LaserlineDeviceExport)
            self.assertEqual(extractor.calls, 6)
            plane = result.params.plane
            normal = np.array(plane.normal_xyz)
            self.assertGreater(abs(normal @ _images.LASER_NORMAL), 0.999)
            self.assertAlmostEqual(
                abs(plane.distance), abs(_images.LASER_D), delta=0.01 * abs(_images.LASER_D)
            )
            residuals = result.per_feature_residuals
            self.assertGreater(len(residuals.laser), 100)
            self.assertEqual(len(residuals.laser_lines), 6)

    def test_extractor_failures_are_run_errors(self) -> None:
        class Broken:
            name = "broken"

            def extract(self, image: np.ndarray, spec: dict) -> np.ndarray:
                raise RuntimeError("no stripe")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = _images.write_laserline_dataset(root, views=3)
            with self.assertRaises(vc.RunError) as ctx:
                dataset_runner.build_laserline_device_input(spec, Broken(), base_dir=root)
            self.assertEqual(ctx.exception.kind, "laser_extraction")
            self.assertIn("no stripe", str(ctx.exception))

    def test_typed_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = _images.write_planar_dataset(root, views=1)
            empty = copy.deepcopy(spec)
            empty["cameras"][0]["images"]["pattern"] = "nothing_*.png"
            with self.assertRaises(vc.RunError) as ctx:
                dataset_runner.build_planar_input(empty, base_dir=root)
            self.assertEqual(ctx.exception.kind, "empty_image_match")
            with self.assertRaises(vc.RunError) as ctx:
                dataset_runner.build_single_cam_handeye_input(spec, base_dir=root)
            self.assertEqual(ctx.exception.kind, "unsupported_topology")
            with self.assertRaises(ValueError):  # a payload needs its base directory
                dataset_runner.build_planar_input(spec)
            with self.assertRaises(ValueError):
                vc.calibrate_dataset({**spec, "topology": "stereo"}, base_dir=root)

    def test_an_open_convention_asks_the_user(self) -> None:
        # Checked before any image is read: no images needed.
        spec = {
            "version": 1,
            "topology": "single_cam_handeye",
            "cameras": [{"id": "cam0", "images": {"kind": "glob", "pattern": "*.png"}}],
            "target": {"kind": "chessboard", "rows": 17, "cols": 28, "square_size_m": 0.02},
            "robot_poses": {"path": "RobotPosesVec.txt", "format": "rowmajor4x4"},
            "pose_convention": {
                "transform": "t_base_tcp",
                "rotation_format": "matrix4x4_row_major",
                "translation_units": "m",
            },
        }
        with self.assertRaises(vc.AskUserError) as ctx:
            vc.calibrate_dataset(spec, base_dir=KUKA)
        error = ctx.exception
        self.assertIsInstance(error, vc.RunError)
        self.assertEqual(error.kind, "ask_user")
        self.assertEqual(error.field, "pose_pairing")
        self.assertIn("by_index", error.suggestions)
        self.assertTrue(error.prompt)

    @unittest.skipUnless(kuka_images_present(), "kuka_1 images are Git LFS pointers")
    def test_kuka_hand_eye_from_its_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = vc.FsDetectionCache(tmp)
            run = dataset_runner.build_single_cam_handeye_input(KUKA / "dataset.toml", cache=cache)
            self.assertEqual(run.total_views, 30)
            result = vc.calibrate_dataset(
                KUKA / "dataset.toml", vc.SingleCamHandeyeConfig(), cache=cache
            )
            assert isinstance(result, vc.SingleCamHandeyeExport)
            self.assertEqual(result.handeye_mode, "EyeInHand")
            self.assertLess(result.mean_reproj_error, 0.5)
            self.assertEqual(len(result.per_feature_residuals.target), 30 * 17 * 28)


if __name__ == "__main__":
    unittest.main()
