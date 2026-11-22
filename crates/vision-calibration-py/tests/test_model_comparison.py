from __future__ import annotations

import random
import unittest
from typing import Any

import vision_calibration as vc
from _fixtures import planar_calibration_dataset


def _noisy_dataset(seed: int = 3, sigma: float = 0.2) -> vc.PlanarIntrinsicsInput:
    """The fixture's ten views with seeded Gaussian corner noise."""
    rng = random.Random(seed)
    dataset = planar_calibration_dataset(10)
    for view in dataset.views:
        obs = view.observation
        obs.points_2d = [
            (x + rng.gauss(0.0, sigma), y + rng.gauss(0.0, sigma)) for x, y in obs.points_2d
        ]
    return dataset


def _pinhole() -> vc.CameraConfig:
    return vc.CameraConfig(model=vc.CameraModelSpec(distortion="none"))


class ModelComparisonTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = _noisy_dataset()
        cls.comparison = vc.compare_camera_models(cls.dataset, [_pinhole()], folds=3, seed=1)

    def test_every_view_is_held_out_once(self) -> None:
        folds = self.comparison["folds"]
        self.assertEqual(len(folds), 3)
        self.assertEqual(sorted(v for fold in folds for v in fold), list(range(10)))

    def test_candidates_are_scored_and_ranked(self) -> None:
        candidates = self.comparison["candidates"]
        self.assertEqual(len(candidates), 2)
        # The default Brown-Conrady (k3 fixed) and the plain pinhole.
        self.assertEqual([c["free_parameters"] for c in candidates], [8, 4])
        for candidate in candidates:
            outcome: Any = candidate["outcome"]
            self.assertEqual(outcome["status"], "scored")
            self.assertEqual(len(outcome["folds"]), 3)
            self.assertEqual(len(outcome["per_view_rms_px"]), 10)
            self.assertEqual(outcome["unprojected_points"], 0)
            self.assertGreater(outcome["held_out"]["rms"], 0.0)
            self.assertGreaterEqual(outcome["delta_vs_best_px"], 0.0)
        self.assertEqual(sorted(self.comparison["ranking"]), [0, 1])
        best: Any = candidates[self.comparison["ranking"][0]]["outcome"]
        self.assertEqual(best["delta_vs_best_px"], 0.0)

    def test_the_seed_fixes_the_folds(self) -> None:
        again = vc.compare_camera_models(
            self.dataset.to_payload(),
            [_pinhole().to_payload()],
            vc.PlanarIntrinsicsConfig(),
            folds=3,
            seed=1,
        )
        self.assertEqual(again, self.comparison)
        other = vc.compare_camera_models(self.dataset, [], folds=3, seed=2)
        self.assertNotEqual(other["folds"], self.comparison["folds"])

    def test_progress_is_reported_and_can_stop_the_run(self) -> None:
        reports: list[Any] = []
        vc.compare_camera_models(self.dataset, [], folds=3, progress=reports.append)
        self.assertTrue(any(r["kind"] == "step" for r in reports), reports[:3])

        def stop(_: Any) -> None:
            raise RuntimeError("stop here")

        with self.assertRaisesRegex(RuntimeError, "stop here"):
            vc.compare_camera_models(self.dataset, [], folds=3, progress=stop)

    def test_invalid_inputs_raise(self) -> None:
        with self.assertRaises(ValueError):
            vc.compare_camera_models(self.dataset, [], folds=1)
        with self.assertRaises(ValueError):
            vc.compare_camera_models(self.dataset, [], folds=11)
        with self.assertRaises(ValueError):
            vc.compare_camera_models(self.dataset, [vc.CameraConfig()])
        with self.assertRaises(TypeError):
            vc.compare_camera_models(self.dataset, [{"model": "pinhole"}])


if __name__ == "__main__":
    unittest.main()
