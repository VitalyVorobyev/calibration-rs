"""Runtime proof that ``distortion_model`` and rig ``sensor`` reach the solver.

The config-schema tests pin the payload *shape*; these prove the two new knobs
are actually *consumed* on the Rust side and surface correctly on the typed
result (not silently defaulted):

* ``distortion_model`` — the typed ``run_planar_intrinsics`` /
  ``run_scheimpflug_intrinsics`` result now carries a polymorphic
  ``camera.distortion`` whose dataclass matches the requested model
  (``brown_conrady5`` → :class:`BrownConradyDistortion`, ``division1`` →
  :class:`Division1Distortion`, ``rational8`` → :class:`Rational8Distortion`,
  ``thin_prism9`` → :class:`ThinPrism9Distortion`). Selecting a model both
  changes the fitted coefficients *and* round-trips through the typed result.
* rig ``sensor`` — a ``ScheimpflugSensorMode`` carrying a non-Brown-Conrady
  ``distortion_model`` is rejected up front by the rig ``validate_config`` (ADR
  0019). That rejection only fires when the internally-tagged ``sensor`` payload
  (``kind: "Scheimpflug"``) *and* its nested ``distortion_model`` both
  deserialize, so it is a fast, distinguishing consumption check. The pinhole
  ``sensor`` path is exercised by ``test_rig_handeye_laserline``.
"""

from __future__ import annotations

import math
import unittest

import vision_calibration as vc
from _fixtures import euler_rot_xyz, planar_calibration_dataset, transform_point


def _rig_extrinsics_dataset() -> vc.RigExtrinsicsDataset:
    # Two cameras viewing the same board — enough for validate_config to run
    # before any solve. Geometry need not be well-conditioned here.
    board = [(i * 0.04, j * 0.04, 0.0) for i in range(6) for j in range(5)]
    views: list[vc.RigExtrinsicsView] = []
    for i in range(6):
        R = euler_rot_xyz(0.03 * i, -0.1 + 0.03 * i, 0.05 * i)
        t = [-0.03 + 0.02 * i, 0.02, 0.5 + 0.03 * i]
        cams: list[vc.Observation | None] = []
        for _ in range(2):
            p3, p2 = [], []
            for pt in board:
                pc = transform_point(R, t, list(pt))
                p3.append(pt)
                p2.append((640.0 + 800.0 * pc[0] / pc[2], 360.0 + 780.0 * pc[1] / pc[2]))
            cams.append(vc.Observation(points_3d=p3, points_2d=p2))
        views.append(vc.RigExtrinsicsView(cameras=cams))
    return vc.RigExtrinsicsDataset(num_cameras=2, views=views)


class PlanarDistortionModelConsumptionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = planar_calibration_dataset()

    def _run(self, model: str) -> vc.PlanarCalibrationResult:
        cfg = vc.PlanarCalibrationConfig(distortion_model=model)  # type: ignore[arg-type]
        return vc.run_planar_intrinsics(self.dataset, cfg)

    def test_none_model_yields_no_distortion_dataclass(self) -> None:
        # `DistortionKind::None` solves fine and exports `{"type": "none"}`; the
        # typed result must parse it (regression: it used to raise after a
        # successful solve because the dispatcher had no "none" entry).
        result = self._run("none")
        self.assertIsInstance(result.camera, vc.PinholeCamera)
        self.assertIsInstance(result.camera.distortion, vc.NoDistortion)

    def test_default_model_yields_brown_conrady_dataclass(self) -> None:
        result = self._run("brown_conrady5")
        self.assertIsInstance(result.camera, vc.PinholeCamera)
        self.assertIsInstance(result.camera.distortion, vc.BrownConradyDistortion)

    def test_division_model_yields_division_dataclass(self) -> None:
        result = self._run("division1")
        self.assertIsInstance(result.camera.distortion, vc.Division1Distortion)
        # The single division parameter is present and finite (keyword-safe name).
        self.assertTrue(math.isfinite(result.camera.distortion.lambda_))

    def test_rational_model_yields_rational_dataclass(self) -> None:
        result = self._run("rational8")
        dist = result.camera.distortion
        self.assertIsInstance(dist, vc.Rational8Distortion)
        # Rational8 exposes the k4..k6 radial block Brown-Conrady lacks.
        for coeff in (dist.k4, dist.k5, dist.k6):
            self.assertTrue(math.isfinite(coeff))

    def test_thin_prism_model_yields_thin_prism_dataclass(self) -> None:
        result = self._run("thin_prism9")
        dist = result.camera.distortion
        self.assertIsInstance(dist, vc.ThinPrism9Distortion)
        # ThinPrism9 exposes the s1..s4 prism block.
        for coeff in (dist.s1, dist.s2, dist.s3, dist.s4):
            self.assertTrue(math.isfinite(coeff))


class ScheimpflugDistortionModelConsumptionTest(unittest.TestCase):
    def test_scheimpflug_division_model_yields_division_dataclass(self) -> None:
        cfg = vc.ScheimpflugIntrinsicsCalibrationConfig(distortion_model="division1")
        result = vc.run_scheimpflug_intrinsics(planar_calibration_dataset(), cfg)
        # The Scheimpflug result camera is polymorphic on distortion too, while
        # still carrying the Scheimpflug sensor tilt.
        self.assertIsInstance(result.camera, vc.PinholeScheimpflugCamera)
        self.assertIsInstance(result.camera.distortion, vc.Division1Distortion)
        self.assertIsInstance(result.camera.sensor, vc.ScheimpflugSensor)


class RigSensorConsumptionTest(unittest.TestCase):
    def test_rig_extrinsics_rejects_non_bc5_scheimpflug_distortion(self) -> None:
        cfg = vc.RigExtrinsicsCalibrationConfig(
            sensor=vc.ScheimpflugSensorMode(distortion_model="rational8")
        )
        with self.assertRaises(ValueError) as ctx:
            vc.run_rig_extrinsics(_rig_extrinsics_dataset(), cfg)
        # The message proves both sensor.kind and sensor.distortion_model were
        # read: a silently-defaulted Pinhole sensor would never reach this arm.
        self.assertIn("BrownConrady5", str(ctx.exception))

    def test_rig_extrinsics_pinhole_sensor_does_not_reject(self) -> None:
        # A pinhole sensor exposes no distortion_model, so validate_config must
        # not raise the Scheimpflug-only rejection (the run may still fail later
        # on ill-conditioned geometry — we only assert the message is absent).
        cfg = vc.RigExtrinsicsCalibrationConfig(sensor=vc.PinholeSensorMode())
        try:
            vc.run_rig_extrinsics(_rig_extrinsics_dataset(), cfg)
        except ValueError as exc:
            self.assertNotIn("BrownConrady5", str(exc))


if __name__ == "__main__":
    unittest.main()
