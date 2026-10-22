from __future__ import annotations

import random
import unittest

import vision_calibration as vc
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


if __name__ == "__main__":
    unittest.main()
