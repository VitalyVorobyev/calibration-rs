"""Typed per-feature residuals of a result."""

from __future__ import annotations

import json
import math
import unittest
from pathlib import Path

import numpy as np

import vision_calibration as vc
from vision_calibration._api import _run_laserline_device_raw

from _fixtures import planar_calibration_dataset

OBSERVATIONS = Path(__file__).resolve().parents[3] / "regression/observations"


class PerFeatureResidualsTest(unittest.TestCase):
    def test_planar_result_has_a_target_table(self) -> None:
        result = vc.run_planar_intrinsics(planar_calibration_dataset(4))
        res = result.per_feature_residuals
        self.assertIsInstance(res, vc.PerFeatureResiduals)
        self.assertEqual(len(res.target), 4 * 48)
        self.assertEqual(res.target.observed_px.shape, (4 * 48, 2))
        self.assertEqual(res.target.target_xyz_m.shape, (4 * 48, 3))
        self.assertEqual(res.target.pose.dtype, np.int64)
        np.testing.assert_array_equal(np.unique(res.target.pose), np.arange(4))
        np.testing.assert_allclose(
            res.target.error_px,
            np.linalg.norm(res.target.projected_px - res.target.observed_px, axis=1),
        )
        self.assertEqual(len(res.laser), 0)
        self.assertEqual(res.laser.residual_m.shape, (0,))
        assert res.target_hist_per_camera is not None
        self.assertEqual(res.target_hist_per_camera[0].count, 4 * 48)
        self.assertEqual(res.target_hist_per_camera[0].bucket_edges_px, (1.0, 2.0, 5.0, 10.0))

    def test_laser_result_has_laser_tables(self) -> None:
        payload = json.loads((OBSERVATIONS / "synth_laserline.json").read_text())["input"]
        res = _run_laserline_device_raw(payload).per_feature_residuals
        pixels = sum(len(view["meta"]["laser_pixels"]) for view in payload)
        self.assertEqual(len(res.laser), pixels)
        self.assertFalse(np.isnan(res.laser.residual_px).any())
        self.assertLess(float(np.nanmean(np.abs(res.laser.residual_px))), 2.0)
        self.assertEqual(len(res.laser_lines), len(payload))
        self.assertEqual(res.laser_lines[0].points_px.shape[1], 2)

    def test_unset_values_are_nan(self) -> None:
        res = vc.PerFeatureResiduals.from_payload(
            {
                "target": [
                    {
                        "pose": 2,
                        "camera": 1,
                        "feature": 5,
                        "target_xyz_m": [0.1, 0.2, 0.0],
                        "observed_px": [10.0, 20.0],
                        "projected_px": None,
                        "error_px": None,
                    }
                ],
                "laser": [
                    {
                        "pose": 0,
                        "camera": 0,
                        "feature": 0,
                        "observed_px": [1.0, 2.0],
                        "residual_m": 0.001,
                        "residual_px": None,
                    }
                ],
            }
        )
        self.assertTrue(np.isnan(res.target.projected_px).all())
        self.assertTrue(math.isnan(res.target.error_px[0]))
        self.assertEqual(int(res.target.camera[0]), 1)
        self.assertEqual(float(res.laser.residual_m[0]), 0.001)
        self.assertTrue(math.isnan(res.laser.residual_px[0]))
        self.assertIsNone(res.target_hist_per_camera)
        self.assertEqual(len(vc.PerFeatureResiduals.from_payload(None).target), 0)


if __name__ == "__main__":
    unittest.main()
