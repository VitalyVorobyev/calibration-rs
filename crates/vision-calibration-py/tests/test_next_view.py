from __future__ import annotations

import json
import random
import unittest
from pathlib import Path
from typing import Any

import vision_calibration as vc
from vision_calibration._api import _run_rig_handeye_raw
from _fixtures import planar_calibration_dataset

IMAGE = (1280, 720)
#: The fixture's 8 × 6 board of 40 mm cells.
BOARD = [(i * 0.04, j * 0.04, 0.0) for i in range(8) for j in range(6)]


def _noisy_result(seed: int = 7, sigma: float = 0.2) -> vc.PlanarIntrinsicsExport:
    """The fixture's calibration with seeded Gaussian corner noise, so the
    covariance has a scale."""
    rng = random.Random(seed)
    dataset = planar_calibration_dataset(8)
    for view in dataset.views:
        obs = view.observation
        obs.points_2d = [
            (x + rng.gauss(0.0, sigma), y + rng.gauss(0.0, sigma)) for x, y in obs.points_2d
        ]
    return vc.run_planar_intrinsics(dataset, vc.PlanarIntrinsicsConfig())


class NextViewTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = _noisy_result()
        cls.candidates = vc.candidate_poses(
            cls.result.camera, BOARD, IMAGE, vc.CandidateGrid(0.6, 0.4, (3, 2))
        )

    def test_candidates_cover_the_grid(self) -> None:
        # 3 × 2 regions, each fronto-parallel and tilted four ways.
        self.assertEqual(len(self.candidates), 6 * 5)
        self.assertIsInstance(self.candidates[0], vc.Pose)
        again = vc.candidate_poses(
            self.result.camera.to_payload(), BOARD, IMAGE, vc.CandidateGrid(0.6, 0.4, (3, 2))
        )
        self.assertEqual(again, self.candidates)

    def test_prediction_shrinks_the_covariance(self) -> None:
        predicted = vc.predict_with_views(self.result, self.candidates[:2], BOARD, IMAGE)
        self.assertIsInstance(predicted, vc.PredictedUncertainty)
        current = self.result.uncertainty
        self.assertTrue(predicted.covariance.available)
        self.assertEqual(predicted.covariance.labels, current.labels)
        for label in current.labels:
            self.assertLessEqual(
                predicted.covariance.std_dev(label), current.std_dev(label) * (1 + 1e-9)
            )
        now = vc.projection_uncertainty(self.result, IMAGE)
        self.assertLessEqual(predicted.projection.max, now.max)
        self.assertEqual(len(predicted.visible_points), 2)
        self.assertGreater(predicted.effective_noise_px, 0.1)
        # The export dict gives the same prediction.
        again = vc.predict_with_views(self.result.export, self.candidates[:2], BOARD, IMAGE)
        self.assertEqual(again, predicted)

    def test_ranking_is_best_first(self) -> None:
        ranked = vc.rank_candidates(self.result, self.candidates, BOARD, IMAGE)
        self.assertEqual(sorted(r.index for r in ranked), list(range(len(self.candidates))))
        sigmas = [r.max_projection_sigma_px for r in ranked]
        self.assertEqual(sigmas, sorted(sigmas))
        best = ranked[0]
        self.assertIsInstance(best, vc.RankedCandidate)
        self.assertGreater(best.projection_sigma_reduction_px, 0.0)
        self.assertGreaterEqual(best.log_det_reduction, 0.0)

    def test_invalid_inputs_raise(self) -> None:
        with self.assertRaises(ValueError):
            vc.predict_with_views(self.result, [], BOARD, IMAGE)
        with self.assertRaises(ValueError):
            vc.rank_candidates(self.result, self.candidates[:1], [], IMAGE)
        with self.assertRaises(ValueError):
            vc.candidate_poses(self.result.camera, BOARD, IMAGE, vc.CandidateGrid(0.0, 0.4, (3, 2)))
        not_planar = dict(self.result.export, kind="rig_extrinsics")
        with self.assertRaises(TypeError):
            vc.predict_with_views(not_planar, self.candidates[:1], BOARD, IMAGE)


OBSERVATIONS = Path(__file__).resolve().parents[3] / "regression/observations"


def _rig_handeye() -> tuple[dict[str, Any], list[dict[str, Any]], list[list[float]], list[tuple[int, int]]]:
    """The rig hand-eye part of the synthetic joint laser scene: its input (the
    first 12 views), the remaining views' robot poses as candidates, the
    target's points and the cameras' image sizes."""
    scene = json.loads(
        (OBSERVATIONS / "synth_rig_handeye_laserline.json").read_text(encoding="utf-8")
    )
    views = [
        {"obs": {"cameras": v["obs"]["cameras"]}, "meta": v["meta"]}
        for v in scene["input"]["views"]
    ]
    points = sorted(
        {
            tuple(p)
            for v in views
            for cam in v["obs"]["cameras"]
            if cam is not None
            for p in cam["points_3d"]
        }
    )
    input = {"num_cameras": scene["input"]["num_cameras"], "views": views[:12]}
    candidates = [v["meta"]["base_se3_gripper"] for v in views[12:]]
    sizes = [tuple(c["image_size"]) for c in scene["cameras"]]
    return input, candidates, [list(p) for p in points], sizes


class RobotPoseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        input, cls.candidates, cls.points, cls.sizes = _rig_handeye()
        cls.result = _run_rig_handeye_raw(input)

    def test_prediction_shrinks_the_covariance(self) -> None:
        predicted = vc.predict_with_robot_poses(
            self.result, self.candidates[:2], self.points, self.sizes
        )
        self.assertIsInstance(predicted, vc.PredictedRigUncertainty)
        current = self.result.uncertainty
        self.assertEqual(predicted.covariance.labels, current.labels)
        for label in current.labels:
            self.assertLessEqual(
                predicted.covariance.std_dev(label), current.std_dev(label) * (1 + 1e-9)
            )
        self.assertEqual(len(predicted.projection), 2)
        self.assertEqual(len(predicted.visible_points), 2)
        self.assertEqual(len(predicted.visible_points[0]), 2)
        self.assertGreater(predicted.effective_noise_px, 0.0)
        again = vc.predict_with_robot_poses(
            self.result.export, self.candidates[:2], self.points, self.sizes
        )
        self.assertEqual(again, predicted)

    def test_ranking_is_best_first(self) -> None:
        ranked = vc.rank_robot_poses(self.result, self.candidates, self.points, self.sizes)
        self.assertEqual(sorted(r.index for r in ranked), list(range(len(self.candidates))))
        reductions = [r.log_det_reduction for r in ranked]
        self.assertEqual(reductions, sorted(reductions, reverse=True))
        self.assertIsInstance(ranked[0], vc.RankedRobotPose)
        self.assertGreater(ranked[0].log_det_reduction, 0.0)

    def test_invalid_inputs_raise(self) -> None:
        with self.assertRaises(ValueError):
            vc.predict_with_robot_poses(self.result, [], self.points, self.sizes)
        with self.assertRaises(ValueError):
            vc.rank_robot_poses(self.result, self.candidates[:1], self.points, self.sizes[:1])
        not_rig = dict(self.result.export, kind="planar_intrinsics")
        with self.assertRaises(TypeError):
            vc.predict_with_robot_poses(not_rig, self.candidates[:1], self.points, self.sizes)


if __name__ == "__main__":
    unittest.main()
