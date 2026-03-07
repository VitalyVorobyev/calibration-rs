from __future__ import annotations

import unittest

import vision_calibration as vc
import vision_calibration.types as vc_types


class TypedModelParsingTest(unittest.TestCase):
    def test_types_module_is_low_level_and_not_reexported_top_level(self) -> None:
        self.assertFalse(hasattr(vc, "RobustLoss"))
        self.assertFalse(hasattr(vc, "HandEyeMode"))
        self.assertTrue(hasattr(vc_types, "RobustLoss"))

    def test_pinhole_camera_from_runtime_payload(self) -> None:
        payload = {
            "proj": "Pinhole",
            "k": {
                "fx": 800.0,
                "fy": 780.0,
                "cx": 640.0,
                "cy": 360.0,
                "skew": 0.0,
            },
            "dist": {
                "k1": 0.1,
                "k2": -0.01,
                "k3": 0.001,
                "p1": 0.0,
                "p2": 0.0,
                "iters": 8,
            },
        }

        cam = vc.PinholeBrownConradyCamera.from_payload(payload)
        self.assertAlmostEqual(cam.intrinsics.fx, 800.0)
        self.assertAlmostEqual(cam.intrinsics.fy, 780.0)
        self.assertAlmostEqual(cam.distortion.k1, 0.1)
        self.assertEqual(cam.distortion.iters, 8)

    def test_scheimpflug_sensor_parses_legacy_aliases(self) -> None:
        sensor = vc.ScheimpflugSensor.from_payload({"tau_x": 0.01, "tau_y": -0.02})
        self.assertAlmostEqual(sensor.tilt_x, 0.01)
        self.assertAlmostEqual(sensor.tilt_y, -0.02)

    def test_scheimpflug_camera_from_camera_params_payload(self) -> None:
        payload = {
            "projection": {"type": "pinhole"},
            "intrinsics": {
                "type": "fx_fy_cx_cy_skew",
                "fx": 800.0,
                "fy": 780.0,
                "cx": 640.0,
                "cy": 360.0,
                "skew": 0.0,
            },
            "distortion": {
                "type": "brown_conrady5",
                "k1": 0.01,
                "k2": -0.002,
                "k3": 0.0,
                "p1": 0.0,
                "p2": 0.0,
                "iters": 8,
            },
            "sensor": {
                "type": "scheimpflug",
                "tilt_x": 0.015,
                "tilt_y": -0.012,
            },
        }

        cam = vc.PinholeBrownConradyScheimpflugCamera.from_payload(payload)
        self.assertAlmostEqual(cam.sensor.tilt_x, 0.015)
        self.assertAlmostEqual(cam.sensor.tilt_y, -0.012)


if __name__ == "__main__":
    unittest.main()
