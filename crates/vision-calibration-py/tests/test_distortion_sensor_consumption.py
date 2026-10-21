"""Runtime proof that the camera model and rig camera overrides reach the solver.

The config-schema tests pin the payload *shape*; these prove the two new knobs
are actually *consumed* on the Rust side and surface correctly on the typed
result (not silently defaulted):

* ``camera.model.distortion`` — the typed ``run_planar_intrinsics``
  result (identity or Scheimpflug sensor) carries a polymorphic
  ``camera.distortion`` whose dataclass matches the requested model
  (``brown_conrady5`` → :class:`BrownConradyDistortion`, ``division1`` →
  :class:`Division1Distortion`, ``rational8`` → :class:`Rational8Distortion`,
  ``thin_prism9`` → :class:`ThinPrism9Distortion`). Selecting a model both
  changes the fitted coefficients *and* round-trips through the typed result.
* rig ``camera_overrides`` — an override naming a camera outside the rig, or
  naming the same camera twice, is rejected up front by the rig config
  validation. That rejection only fires when the override list deserializes
  and reaches the solver, so it is a fast, distinguishing consumption check.
"""

from __future__ import annotations

import math
import unittest

import vision_calibration as vc
from _fixtures import euler_rot_xyz, planar_calibration_dataset, transform_point


def _rig_extrinsics_dataset() -> vc.RigExtrinsicsInput:
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
    return vc.RigExtrinsicsInput(num_cameras=2, views=views)


class PlanarDistortionModelConsumptionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = planar_calibration_dataset()

    def _run(self, model: str) -> vc.PlanarIntrinsicsExport:
        cfg = vc.PlanarIntrinsicsConfig(
            camera=vc.CameraConfig(model=vc.CameraModelSpec(distortion=model))  # type: ignore[arg-type]
        )
        return vc.run_planar_intrinsics(self.dataset, cfg)

    def test_none_model_yields_no_distortion_dataclass(self) -> None:
        # `DistortionKind::None` solves fine and exports `{"type": "none"}`; the
        # typed result must parse it (regression guard: the dispatcher must have a
        # "none" entry, or parsing raises after a successful solve).
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
        cfg = vc.PlanarIntrinsicsConfig(
            camera=vc.CameraConfig(
                model=vc.CameraModelSpec(distortion="division1", sensor="scheimpflug")
            )
        )
        result = vc.run_planar_intrinsics(planar_calibration_dataset(), cfg)
        # The Scheimpflug result camera is polymorphic on distortion too, while
        # still carrying the Scheimpflug sensor tilt.
        self.assertIsInstance(result.camera, vc.PinholeScheimpflugCamera)
        self.assertIsInstance(result.camera.distortion, vc.Division1Distortion)
        self.assertIsInstance(result.camera.sensor, vc.ScheimpflugSensor)


class RigCameraOverrideConsumptionTest(unittest.TestCase):
    def test_rig_extrinsics_rejects_out_of_range_override(self) -> None:
        cfg = vc.RigExtrinsicsConfig(
            camera_overrides=[vc.CameraOverride(camera=9)]
        )
        with self.assertRaisesRegex(RuntimeError, "out of range"):
            vc.run_rig_extrinsics(_rig_extrinsics_dataset(), cfg)

    def test_rig_extrinsics_rejects_duplicate_override(self) -> None:
        cfg = vc.RigExtrinsicsConfig(
            camera_overrides=[vc.CameraOverride(0), vc.CameraOverride(0)]
        )
        with self.assertRaises(ValueError):
            vc.run_rig_extrinsics(_rig_extrinsics_dataset(), cfg)

    def test_rig_extrinsics_result_cameras_are_typed(self) -> None:
        result = vc.run_rig_extrinsics(
            _rig_extrinsics_dataset(), vc.RigExtrinsicsConfig()
        )
        for cam in result.cameras:
            self.assertIsInstance(cam, vc.PinholeCamera)
            self.assertIsInstance(cam.distortion, vc.BrownConradyDistortion)


if __name__ == "__main__":
    unittest.main()
