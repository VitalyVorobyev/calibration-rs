from __future__ import annotations

import unittest

import vision_calibration as vc
from _fixtures import planar_calibration_dataset


def _make_dataset() -> vc.PlanarIntrinsicsInput:
    return planar_calibration_dataset(8)


def _tilted_config(**kwargs: object) -> vc.PlanarIntrinsicsConfig:
    return vc.PlanarIntrinsicsConfig(
        camera=vc.CameraConfig(model=vc.CameraModelSpec(sensor="scheimpflug")),
        **kwargs,  # type: ignore[arg-type]
    )


class PlanarScheimpflugTest(unittest.TestCase):
    def test_public_bindings_run(self) -> None:
        result = vc.run_planar_intrinsics(_make_dataset(), _tilted_config())
        self.assertGreaterEqual(result.mean_reproj_error, 0.0)
        self.assertIsInstance(result.camera.sensor, vc.ScheimpflugSensor)
        self.assertIsInstance(result.camera.sensor, vc.ScheimpflugSensor)
        self.assertEqual(result.export["params"]["camera"]["sensor"]["type"], "scheimpflug")

    def test_untilted_camera_has_identity_sensor(self) -> None:
        result = vc.run_planar_intrinsics(_make_dataset())
        self.assertIsInstance(result.camera.sensor, vc.IdentitySensor)

    def test_explicit_fix_lists_reach_the_solver(self) -> None:
        cfg = _tilted_config()
        cfg.camera.fix = vc.CameraFix(sensor=["tilt_x", "tilt_y"])
        result = vc.run_planar_intrinsics(_make_dataset(), cfg)
        self.assertAlmostEqual(result.camera.sensor.tilt_x, 0.0, places=9)
        self.assertAlmostEqual(result.camera.sensor.tilt_y, 0.0, places=9)

    def test_unknown_fix_name_maps_to_value_error(self) -> None:
        cfg = _tilted_config()
        cfg.camera.fix = vc.CameraFix(sensor=["tilt_z"])
        with self.assertRaises(ValueError):
            vc.run_planar_intrinsics(_make_dataset(), cfg)

    def test_config_payload_roundtrip(self) -> None:
        cfg = _tilted_config()
        cfg.camera.model.distortion = "division1"
        cfg.camera.fix = vc.CameraFix(intrinsics=["cx", "cy"], sensor=[])
        payload = cfg.to_payload()
        self.assertEqual(payload["camera"]["model"]["sensor"], "scheimpflug")
        self.assertEqual(payload["camera"]["fix"], {"intrinsics": ["cx", "cy"], "sensor": []})
        restored = vc.PlanarIntrinsicsConfig.from_mapping(payload)
        self.assertEqual(restored.to_payload(), payload)

    def test_invalid_config_maps_to_value_error(self) -> None:
        # Per R-07: invalid config from Python surfaces as ValueError, not
        # RuntimeError. RuntimeError is reserved for genuine runtime failures
        # (solver divergence, export/pythonize errors).
        with self.assertRaises(ValueError) as ctx:
            vc.run_planar_intrinsics(
                _make_dataset(),
                _tilted_config(solver=vc.SolverConfig(max_iters=0)),
            )
        message = str(ctx.exception)
        self.assertIn("invalid config", message)
        self.assertIn("max_iters must be positive", message)

    def test_invalid_input_maps_to_value_error(self) -> None:
        # Per R-07: invalid input (too few views) surfaces as ValueError.
        dataset = _make_dataset()
        dataset.views = dataset.views[:2]
        with self.assertRaises(ValueError) as ctx:
            vc.run_planar_intrinsics(dataset, _tilted_config())
        message = str(ctx.exception)
        self.assertIn("invalid input", message)
        self.assertIn("insufficient data", message)

    def test_high_level_api_rejects_mapping_inputs(self) -> None:
        with self.assertRaises(TypeError) as cfg_ctx:
            vc.run_planar_intrinsics(_make_dataset(), {"max_iters": 50})
        self.assertIn("config must be PlanarIntrinsicsConfig", str(cfg_ctx.exception))

        with self.assertRaises(TypeError) as input_ctx:
            vc.run_planar_intrinsics({"views": []}, _tilted_config())
        self.assertIn("input must be PlanarIntrinsicsInput", str(input_ctx.exception))


if __name__ == "__main__":
    unittest.main()
