"""Config-payload schema tests for the calibration configs (no solver).

These pin the ``to_payload()`` shape of the new/changed calibration configs to
what the Rust ``serde`` derives expect. The failure mode they guard against is
silent drift: a mis-named Python payload key must not be ignored so that the
field falls back to its Rust default — a wrong result with no error. Each key name and default value asserted here was read off the Rust
source of truth:

* ``PlanarIntrinsicsConfig`` — grouped ``init`` / ``solver`` / ``camera``
  (model + named fix lists) / ``fix_poses``; ``max_iters`` defaults to 120.
* ``vision-calibration-optim`` ``DistortionKind`` — ``#[serde(rename_all =
  "snake_case")]``.
* ``RigExtrinsicsConfig`` / ``RigHandeyeConfig`` — ``camera`` (the planar
  ``CameraConfig`` wrapper) plus ``camera_overrides`` (a list of
  ``{camera, model, fix}``); ``rig.refine_sensor`` and
  ``handeye_ba.refine_sensor``.
* ``rig_handeye_laserline/problem.rs`` ``RigHandeyeLaserlineBaConfig`` defaults
  (``calib_weight = 1.0``, ``laser_weight = 1e4``, ``fix_scheimpflug`` tilt
  frozen, ``default_camera_fix.distortion`` = ``radial_only``).
"""

from __future__ import annotations

import typing
import unittest

import vision_calibration as vc
import vision_calibration.types as vc_types
from vision_calibration.models import _RADIAL_ONLY_DISTORTION_FIX_MASK as _RADIAL_ONLY


class DistortionModelSchemaTest(unittest.TestCase):
    def test_distortion_model_literal_matches_serde_snake_case(self) -> None:
        # `DistortionKind` is `#[serde(rename_all = "snake_case")]`; the Python
        # Literal must carry exactly the serde spellings.
        self.assertEqual(
            set(typing.get_args(vc_types.DistortionModel)),
            {"none", "brown_conrady5", "rational8", "thin_prism9", "division1"},
        )

    def test_planar_default_payload_shape(self) -> None:
        payload = vc.PlanarCalibrationConfig().to_payload()
        self.assertEqual(set(payload), {"init", "solver", "camera", "fix_poses"})
        self.assertEqual(
            payload["camera"],
            {
                "model": {
                    "projection": "pinhole",
                    "distortion": "brown_conrady5",
                    "sensor": "identity",
                },
                "fix": {},
            },
        )
        # Planar solves default to 120 iterations (the tilt valley needs headroom).
        self.assertEqual(payload["solver"]["max_iters"], 120)

    def test_init_omits_unset_fix_tangential(self) -> None:
        self.assertNotIn("fix_tangential", vc.IntrinsicsInitConfig().to_payload())
        explicit = vc.IntrinsicsInitConfig(fix_tangential=False).to_payload()
        self.assertIs(explicit["fix_tangential"], False)

    def test_planar_distortion_roundtrips_each_model(self) -> None:
        for model in ("none", "brown_conrady5", "rational8", "thin_prism9", "division1"):
            cfg = vc.PlanarCalibrationConfig(
                camera=vc.CameraConfig(model=vc.CameraModelSpec(distortion=model))  # type: ignore[arg-type]
            )
            payload = cfg.to_payload()
            self.assertEqual(payload["camera"]["model"]["distortion"], model)
            restored = vc.PlanarCalibrationConfig.from_mapping(payload)
            self.assertEqual(restored.camera.model.distortion, model)

    def test_camera_fix_payload_omits_unset_slots(self) -> None:
        fix = vc.CameraFix(distortion=["k1"], sensor=[])
        self.assertEqual(fix.to_payload(), {"distortion": ["k1"], "sensor": []})
        self.assertEqual(vc.CameraFix.from_payload(fix.to_payload()).to_payload(), fix.to_payload())

    def test_camera_config_rejects_unknown_fields(self) -> None:
        with self.assertRaises(ValueError):
            vc.CameraConfig.from_payload({"bogus": 1})
        with self.assertRaises(ValueError):
            vc.CameraModelSpec.from_payload({"bogus": 1})
        with self.assertRaises(ValueError):
            vc.CameraFix.from_payload({"bogus": []})
        with self.assertRaises(ValueError):
            vc.PlanarCalibrationConfig.from_mapping({"distortion_model": "none"})


class RigCameraConfigSchemaTest(unittest.TestCase):
    def test_rig_extrinsics_camera_key_and_defaults(self) -> None:
        payload = vc.RigExtrinsicsCalibrationConfig().to_payload()
        # A mis-named key would silently fall back to the Rust default.
        self.assertNotIn("sensor", payload)
        self.assertEqual(
            payload["camera"],
            {
                "model": {
                    "projection": "pinhole",
                    "distortion": "brown_conrady5",
                    "sensor": "identity",
                },
                "fix": {},
            },
        )
        self.assertEqual(payload["camera_overrides"], [])
        self.assertFalse(payload["rig"]["refine_sensor"])

    def test_rig_handeye_camera_key_and_defaults(self) -> None:
        payload = vc.RigHandeyeCalibrationConfig().to_payload()
        self.assertNotIn("sensor", payload)
        self.assertIn("camera", payload)
        self.assertEqual(payload["camera_overrides"], [])
        self.assertEqual(
            payload["handeye_ba"],
            {
                "robot_poses": payload["handeye_ba"]["robot_poses"],
                "refine_cam_se3_rig": False,
                "refine_sensor": False,
            },
        )

    def test_camera_override_payload_shape(self) -> None:
        override = vc.CameraOverride(
            camera=1,
            model=vc.CameraModelSpec(sensor="scheimpflug"),
            fix=vc.CameraFix(sensor=["tilt_y"]),
        )
        self.assertEqual(
            override.to_payload(),
            {
                "camera": 1,
                "model": {
                    "projection": "pinhole",
                    "distortion": "brown_conrady5",
                    "sensor": "scheimpflug",
                },
                "fix": {"sensor": ["tilt_y"]},
            },
        )

    def test_rig_extrinsics_overrides_roundtrip(self) -> None:
        cfg = vc.RigExtrinsicsCalibrationConfig(
            camera=vc.CameraConfig(model=vc.CameraModelSpec(distortion="none")),
            camera_overrides=[
                vc.CameraOverride(1, model=vc.CameraModelSpec(sensor="scheimpflug"))
            ],
        )
        cfg.rig.refine_sensor = True
        restored = vc.RigExtrinsicsCalibrationConfig.from_mapping(cfg.to_payload())
        self.assertEqual(restored.to_payload(), cfg.to_payload())
        self.assertIsInstance(restored.camera_overrides[0], vc.CameraOverride)
        self.assertEqual(restored.camera_overrides[0].model.sensor, "scheimpflug")
        self.assertTrue(restored.rig.refine_sensor)

    def test_rig_handeye_overrides_roundtrip(self) -> None:
        cfg = vc.RigHandeyeCalibrationConfig(
            camera_overrides=[vc.CameraOverride(0, fix=vc.CameraFix(distortion=["k1"]))]
        )
        cfg.handeye_ba.refine_sensor = True
        restored = vc.RigHandeyeCalibrationConfig.from_mapping(cfg.to_payload())
        self.assertEqual(restored.to_payload(), cfg.to_payload())
        self.assertTrue(restored.handeye_ba.refine_sensor)

    def test_removed_sensor_field_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            vc.RigExtrinsicsCalibrationConfig.from_mapping({"sensor": {"kind": "Pinhole"}})
        with self.assertRaises(ValueError):
            vc.HandeyeBaConfig.from_mapping({"refine_scheimpflug": True})

    def test_camera_override_rejects_unknown_and_missing_fields(self) -> None:
        with self.assertRaises(ValueError):
            vc.CameraOverride.from_payload({"camera": 0, "bogus": 1})
        with self.assertRaises(ValueError):
            vc.CameraOverride.from_payload({"model": {}})


class RigHandeyeLaserlineConfigSchemaTest(unittest.TestCase):
    def test_default_payload_has_three_warm_started_stages(self) -> None:
        payload = vc.RigHandeyeLaserlineCalibrationConfig().to_payload()
        self.assertEqual(set(payload), {"handeye", "laserline_init", "joint_ba"})

    def test_laserline_init_defaults_to_point_to_plane(self) -> None:
        payload = vc.RigHandeyeLaserlineCalibrationConfig().to_payload()
        init = payload["laserline_init"]
        self.assertEqual(init["laser_residual_type"], "PointToPlane")
        # Rust default seeds the frozen-geometry stage at 200 iterations.
        self.assertEqual(init["solver"]["max_iters"], 200)

    def test_joint_ba_defaults_match_rust(self) -> None:
        ba = vc.RigHandeyeLaserlineCalibrationConfig().to_payload()["joint_ba"]
        self.assertEqual(ba["solver"]["max_iters"], 30)
        self.assertEqual(ba["laser_residual_type"], "PointToPlane")
        self.assertEqual(ba["calib_loss"], "None")
        self.assertEqual(ba["laser_loss"], "None")
        self.assertEqual(ba["calib_weight"], 1.0)
        self.assertEqual(ba["laser_weight"], 1.0e4)
        # Joint stage freezes tilt and uses the radial-only camera fix mask.
        self.assertEqual(ba["fix_scheimpflug"], {"tilt_x": True, "tilt_y": True})
        self.assertEqual(ba["default_camera_fix"]["distortion"], _RADIAL_ONLY)
        self.assertFalse(ba["fix_handeye"])
        self.assertFalse(ba["fix_target_ref"])
        self.assertEqual(
            set(ba),
            {
                "solver",
                "laser_residual_type",
                "calib_loss",
                "laser_loss",
                "calib_weight",
                "laser_weight",
                "default_camera_fix",
                "fix_scheimpflug",
                "fix_handeye",
                "fix_target_ref",
                "robot_poses",
            },
        )

    def test_config_roundtrips_through_from_mapping(self) -> None:
        cfg = vc.RigHandeyeLaserlineCalibrationConfig()
        restored = vc.RigHandeyeLaserlineCalibrationConfig.from_mapping(cfg.to_payload())
        self.assertEqual(restored.to_payload(), cfg.to_payload())

    def test_ba_config_roundtrips_and_rejects_unknown_field(self) -> None:
        ba = vc.RigHandeyeLaserlineBaConfig(calib_weight=2.0, laser_weight=5.0e3)
        restored = vc.RigHandeyeLaserlineBaConfig.from_mapping(ba.to_payload())
        self.assertEqual(restored.calib_weight, 2.0)
        self.assertEqual(restored.laser_weight, 5.0e3)
        with self.assertRaises(ValueError):
            vc.RigHandeyeLaserlineBaConfig.from_mapping({"bogus": 1})

    def test_config_rejects_unknown_field(self) -> None:
        with self.assertRaises(ValueError):
            vc.RigHandeyeLaserlineCalibrationConfig.from_mapping({"bogus": 1})


if __name__ == "__main__":
    unittest.main()
