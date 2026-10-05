"""Solver backend selection through ``SolverConfig.backend``.

``SolverBackend`` is ``#[serde(rename_all = "snake_case")]`` in
``vision-calibration-optim``; ``SolverConfig.backend`` defaults to
``tiny_solver`` (``#[serde(default)]``).
"""

from __future__ import annotations

import typing
import unittest

from _fixtures import planar_calibration_dataset

import vision_calibration as vc
import vision_calibration.types as vc_types


class SolverBackendTest(unittest.TestCase):
    def test_backend_literal_matches_serde_snake_case(self) -> None:
        self.assertEqual(set(typing.get_args(vc_types.SolverBackend)), {"tiny_solver", "factrs"})

    def test_solver_config_payload_carries_the_backend(self) -> None:
        self.assertEqual(vc.SolverConfig().to_payload()["backend"], "tiny_solver")
        cfg = vc.SolverConfig(backend="factrs")
        restored = vc.SolverConfig.from_mapping(cfg.to_payload())
        self.assertEqual(restored.backend, "factrs")

    def test_both_backends_recover_the_same_intrinsics(self) -> None:
        dataset = planar_calibration_dataset()
        results = {}
        for backend in ("tiny_solver", "factrs"):
            cfg = vc.PlanarCalibrationConfig(solver=vc.SolverConfig(backend=backend))
            results[backend] = vc.run_planar_intrinsics(dataset, cfg)
        for backend, result in results.items():
            k = result.camera.intrinsics
            self.assertAlmostEqual(k.fx, 800.0, delta=1e-3, msg=backend)
            self.assertAlmostEqual(k.fy, 780.0, delta=1e-3, msg=backend)
            self.assertAlmostEqual(k.cx, 640.0, delta=1e-3, msg=backend)
            self.assertAlmostEqual(k.cy, 360.0, delta=1e-3, msg=backend)


if __name__ == "__main__":
    unittest.main()
