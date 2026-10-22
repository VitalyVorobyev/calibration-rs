from __future__ import annotations

import math
import random
import unittest

import vision_calibration as vc
from _fixtures import planar_calibration_dataset


def _noisy(
    dataset: vc.PlanarIntrinsicsInput, seed: int, stated: bool
) -> vc.PlanarIntrinsicsInput:
    """``dataset`` with Gaussian corner noise whose σ varies per corner (0.1 to
    0.5 px along u, a third of that along v), stating each corner's covariance
    when ``stated``."""
    rng = random.Random(seed)
    views = []
    for v, view in enumerate(dataset.views):
        obs = view.observation
        points, covs = [], []
        for i, (u, w) in enumerate(obs.points_2d):
            su = 0.1 + 0.4 * ((7 * v + 3 * i) % 11) / 10.0
            sv = su / 3.0
            points.append((u + rng.gauss(0.0, su), w + rng.gauss(0.0, sv)))
            covs.append(vc.PointCovariance(su * su, 0.0, sv * sv))
        views.append(
            vc.PlanarView(
                observation=vc.Observation(
                    points_3d=obs.points_3d,
                    points_2d=points,
                    cov_px=covs if stated else None,
                )
            )
        )
    return vc.PlanarIntrinsicsInput(views=views)


class PointCovarianceTest(unittest.TestCase):
    def test_rejects_a_covariance_that_is_not_positive_definite(self) -> None:
        self.assertEqual(vc.PointCovariance.isotropic(0.5).xx, 0.25)
        for bad in [(1.0, 2.0, 1.0), (0.0, 0.0, 1.0), (1.0, 0.0, math.nan)]:
            with self.assertRaises(ValueError):
                vc.PointCovariance(*bad)

    def test_observation_round_trips_its_covariances(self) -> None:
        cov = [vc.PointCovariance(0.09, -0.01, 0.04), vc.PointCovariance.isotropic(0.2)]
        obs = vc.Observation(points_3d=[(0, 0, 0), (1, 0, 0)], points_2d=[(1, 2), (3, 4)], cov_px=cov)
        payload = obs.to_payload()
        self.assertEqual(payload["cov_px"][0], {"xx": 0.09, "xy": -0.01, "yy": 0.04})
        self.assertEqual(vc.Observation.from_payload(payload), obs)
        plain = vc.Observation(points_3d=[(0, 0, 0)], points_2d=[(1, 2)])
        self.assertNotIn("cov_px", plain.to_payload())
        with self.assertRaises(ValueError):
            vc.Observation(points_3d=[(0, 0, 0)], points_2d=[(1, 2)], cov_px=cov)

    def test_stated_covariances_whiten_the_solve(self) -> None:
        base = planar_calibration_dataset()
        result = vc.run_planar_intrinsics(_noisy(base, seed=3, stated=True))
        noise = result.uncertainty.noise[0]
        self.assertEqual(noise.kind, "reprojection")
        self.assertTrue(noise.whitened)
        self.assertAlmostEqual(noise.sigma, 1.0, delta=0.2)
        target = result.per_feature_residuals.target
        self.assertTrue(all(math.isfinite(e) for e in target.error_whitened))
        # Reported errors stay in pixels.
        self.assertAlmostEqual(
            result.mean_reproj_error, float(target.error_px.mean()), places=9
        )

        plain = vc.run_planar_intrinsics(_noisy(base, seed=3, stated=False))
        self.assertFalse(plain.uncertainty.noise[0].whitened)
        self.assertTrue(all(math.isnan(e) for e in plain.per_feature_residuals.target.error_whitened))

    def test_a_mixed_input_is_rejected(self) -> None:
        stated = _noisy(planar_calibration_dataset(), seed=1, stated=True)
        stated.views[2] = vc.PlanarView(
            observation=vc.Observation(
                points_3d=stated.views[2].observation.points_3d,
                points_2d=stated.views[2].observation.points_2d,
            )
        )
        with self.assertRaisesRegex(Exception, "per-point covariances or none"):
            vc.run_planar_intrinsics(stated)


if __name__ == "__main__":
    unittest.main()
