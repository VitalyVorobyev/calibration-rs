from __future__ import annotations

import unittest

import vision_calibration as vc
from _fixtures import planar_calibration_dataset


def _make_dataset() -> vc.PlanarDataset:
    return planar_calibration_dataset(8)


class ScheimpflugIntrinsicsTest(unittest.TestCase):
    def test_public_bindings_run(self) -> None:
        result = vc.run_scheimpflug_intrinsics(
            _make_dataset(),
            vc.ScheimpflugIntrinsicsCalibrationConfig(
                fix_scheimpflug={"tilt_x": False, "tilt_y": False}
            ),
        )
        self.assertGreaterEqual(result.mean_reproj_error, 0.0)
        self.assertIsInstance(result.camera, vc.PinholeScheimpflugCamera)
        self.assertIsInstance(result.camera.sensor, vc.ScheimpflugSensor)

    def test_invalid_config_maps_to_value_error(self) -> None:
        # Per R-07: invalid config from Python surfaces as ValueError, not
        # RuntimeError. RuntimeError is reserved for genuine runtime failures
        # (solver divergence, export/pythonize errors).
        with self.assertRaises(ValueError) as ctx:
            vc.run_scheimpflug_intrinsics(
                _make_dataset(),
                vc.ScheimpflugIntrinsicsCalibrationConfig(
                    solver=vc.SolverConfig(max_iters=0)
                ),
            )
        message = str(ctx.exception)
        self.assertIn("invalid config", message)
        self.assertIn("max_iters must be positive", message)

    def test_invalid_input_maps_to_value_error(self) -> None:
        # Per R-07: invalid input (too few views) surfaces as ValueError.
        dataset = _make_dataset()
        dataset.views = dataset.views[:2]
        with self.assertRaises(ValueError) as ctx:
            vc.run_scheimpflug_intrinsics(
                dataset,
                vc.ScheimpflugIntrinsicsCalibrationConfig(),
            )
        message = str(ctx.exception)
        self.assertIn("invalid input", message)
        self.assertIn("insufficient data", message)

    def test_high_level_api_rejects_mapping_inputs(self) -> None:
        with self.assertRaises(TypeError) as cfg_ctx:
            vc.run_scheimpflug_intrinsics(
                _make_dataset(),
                {"max_iters": 50},
            )
        self.assertIn("config must be ScheimpflugIntrinsicsCalibrationConfig", str(cfg_ctx.exception))

        with self.assertRaises(TypeError) as input_ctx:
            vc.run_scheimpflug_intrinsics(
                {"views": []},
                vc.ScheimpflugIntrinsicsCalibrationConfig(),
            )
        self.assertIn("input must be PlanarDataset", str(input_ctx.exception))


if __name__ == "__main__":
    unittest.main()
