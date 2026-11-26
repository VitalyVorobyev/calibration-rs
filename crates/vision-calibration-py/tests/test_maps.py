from __future__ import annotations

import math
import unittest

import numpy as np

import vision_calibration as vc
from _fixtures import planar_calibration_dataset


def _camera(k1: float) -> dict:
    return {
        "projection": {"type": "pinhole"},
        "distortion": {
            "type": "brown_conrady5",
            "k1": k1,
            "k2": 0.0,
            "k3": 0.0,
            "p1": 0.0,
            "p2": 0.0,
            "iters": 8,
        },
        "sensor": {"type": "identity"},
        "intrinsics": {"fx": 800.0, "fy": 800.0, "cx": 639.5, "cy": 479.5},
    }


class DistortionMapTest(unittest.TestCase):
    def test_layers_are_numpy_planes_on_the_grid(self) -> None:
        m = vc.distortion_map(_camera(-0.1), (1280, 960), columns=64)
        self.assertIsInstance(m, vc.DistortionMap)
        self.assertEqual((m.grid.columns, m.grid.rows, m.grid.stride), (64, 48, 20))
        self.assertEqual(m.total.dx.shape, (48, 64))
        self.assertEqual(m.valid_nodes, 64 * 48)
        np.testing.assert_array_equal(m.total.dx, m.distortion.dx + m.sensor.dx)
        np.testing.assert_array_equal(m.sensor.dx, np.zeros((48, 64)))
        nodes = m.grid.nodes()
        self.assertEqual(nodes.shape, (48, 64, 2))
        self.assertEqual(tuple(nodes[0, 0]), (9.5, 9.5))
        # Barrel distortion points inward at the top-left corner.
        self.assertGreater(m.total.dx[0, 0], 0.0)
        self.assertGreater(m.total.dy[0, 0], 0.0)
        self.assertGreater(float(np.nanmax(m.total.magnitude())), 1.0)

    def test_rejects_an_empty_grid(self) -> None:
        with self.assertRaises(ValueError):
            vc.distortion_map(_camera(0.0), (1280, 960), columns=0)


class ResidualMapTest(unittest.TestCase):
    def test_a_result_maps_its_residuals(self) -> None:
        result = vc.planar_intrinsics.run_calibration(
            planar_calibration_dataset(8), vc.PlanarIntrinsicsConfig()
        )
        m = vc.residual_map(result, (1280, 720), bandwidth=120.0)
        self.assertIsInstance(m, vc.ResidualMap)
        self.assertEqual(m.bandwidth_selection["kind"], "given")
        self.assertEqual(m.bandwidth_px, 120.0)
        self.assertEqual(m.mean_dx.shape, (m.grid.rows, m.grid.columns))
        self.assertEqual(m.views, 8)
        self.assertGreater(m.used, 0)
        self.assertTrue(np.isfinite(m.mean_dx).any())
        self.assertEqual(m.mean_magnitude().shape, m.mean_dx.shape)
        # The export JSON gives the same map.
        again = vc.residual_map(result.export, (1280, 720), bandwidth=120.0)
        np.testing.assert_array_equal(m.mean_dx, again.mean_dx)
        auto = vc.residual_map(result, (1280, 720))
        self.assertEqual(auto.bandwidth_selection["kind"], "cross_validated")
        self.assertTrue(math.isfinite(auto.bandwidth_px))

    def test_rejects_a_camera_without_records(self) -> None:
        result = vc.planar_intrinsics.run_calibration(
            planar_calibration_dataset(8), vc.PlanarIntrinsicsConfig()
        )
        with self.assertRaises(ValueError):
            vc.residual_map(result, (1280, 720), camera=3)
        with self.assertRaises(ValueError):
            vc.residual_map(result, (1280, 720), outlier_factor=0.0)


if __name__ == "__main__":
    unittest.main()
