"""Runtime proof that the single-camera hand-eye and laserline-device configs'
``camera`` reaches the solver: the result's camera carries the requested
distortion model, and a tilted sensor reports a :class:`PinholeScheimpflugCamera`.

The data come from the synthetic Python examples.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import vision_calibration as vc

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

import handeye_synthetic  # noqa: E402
import laserline_device_session  # noqa: E402


def _camera(distortion: str, sensor: str = "identity") -> vc.CameraConfig:
    return vc.CameraConfig(
        model=vc.CameraModelSpec(distortion=distortion, sensor=sensor)  # type: ignore[arg-type]
    )


class SingleCamHandeyeCameraTest(unittest.TestCase):
    def test_distortion_model_reaches_the_result(self) -> None:
        cfg = vc.SingleCamHandeyeConfig(camera=_camera("division1"))
        result = vc.run_single_cam_handeye(handeye_synthetic.make_input(), cfg)
        self.assertIsInstance(result.camera, vc.PinholeCamera)
        self.assertIsInstance(result.camera.distortion, vc.Division1Distortion)
        self.assertLess(result.mean_reproj_error, 1e-3)

    def test_bad_fix_name_is_rejected(self) -> None:
        cfg = vc.SingleCamHandeyeConfig(camera=_camera("brown_conrady5"))
        cfg.camera.fix = vc.CameraFix(sensor=["tilt_x"])
        with self.assertRaises(Exception) as ctx:
            vc.run_single_cam_handeye(handeye_synthetic.make_input(), cfg)
        self.assertIn("tilt_x", str(ctx.exception))


class LaserlineDeviceCameraTest(unittest.TestCase):
    def test_distortion_model_reaches_the_result(self) -> None:
        cfg = vc.LaserlineDeviceConfig(camera=_camera("rational8"))
        result = vc.run_laserline_device(laserline_device_session.make_input(), cfg)
        camera = result.estimate.params.camera
        self.assertIsInstance(camera, vc.PinholeCamera)
        self.assertIsInstance(camera.distortion, vc.Rational8Distortion)
        self.assertLess(result.mean_reproj_error, 1e-3)

    def test_tilted_sensor_reports_a_scheimpflug_camera(self) -> None:
        cfg = vc.LaserlineDeviceConfig(
            camera=_camera("brown_conrady5", sensor="scheimpflug")
        )
        result = vc.run_laserline_device(laserline_device_session.make_input(), cfg)
        self.assertIsInstance(result.estimate.params.camera, vc.PinholeScheimpflugCamera)


if __name__ == "__main__":
    unittest.main()
