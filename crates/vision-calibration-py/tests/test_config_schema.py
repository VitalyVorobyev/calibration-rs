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
* ``rig_family.rs`` ``SensorMode`` — ``#[serde(tag = "kind")]`` (internally
  tagged ``Pinhole`` / ``Scheimpflug``); ``distortion_mask_in_percam_ba``
  defaults to ``DistortionFixMask::radial_only()``.
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


class SensorModeSchemaTest(unittest.TestCase):
    def test_pinhole_sensor_payload_is_internally_tagged(self) -> None:
        self.assertEqual(vc.PinholeSensorMode().to_payload(), {"kind": "Pinhole"})

    def test_scheimpflug_sensor_payload_matches_rust_defaults(self) -> None:
        payload = vc.ScheimpflugSensorMode().to_payload()
        self.assertEqual(payload["kind"], "Scheimpflug")
        self.assertEqual(payload["init_tilt_x"], 0.0)
        self.assertEqual(payload["init_tilt_y"], 0.0)
        # `#[serde(default)]` ScheimpflugFixMask — both axes free.
        self.assertEqual(payload["fix_scheimpflug"], {"tilt_x": False, "tilt_y": False})
        # `distortion_mask_in_percam_ba` defaults to DistortionFixMask::radial_only().
        self.assertEqual(payload["distortion_mask_in_percam_ba"], _RADIAL_ONLY)
        self.assertFalse(payload["refine_scheimpflug_in_rig_ba"])
        self.assertEqual(payload["distortion_model"], "brown_conrady5")
        self.assertEqual(
            set(payload),
            {
                "kind",
                "init_tilt_x",
                "init_tilt_y",
                "fix_scheimpflug",
                "distortion_mask_in_percam_ba",
                "refine_scheimpflug_in_rig_ba",
                "distortion_model",
            },
        )

    def test_rig_extrinsics_sensor_key_and_default(self) -> None:
        payload = vc.RigExtrinsicsCalibrationConfig().to_payload()
        # The Rust field is named `sensor` — a mis-name would silently default.
        self.assertIn("sensor", payload)
        self.assertEqual(payload["sensor"], {"kind": "Pinhole"})

    def test_rig_handeye_sensor_key_and_default(self) -> None:
        payload = vc.RigHandeyeCalibrationConfig().to_payload()
        self.assertIn("sensor", payload)
        self.assertEqual(payload["sensor"], {"kind": "Pinhole"})

    def test_rig_extrinsics_sensor_roundtrip_preserves_variant(self) -> None:
        for sensor, expected in (
            (vc.PinholeSensorMode(), vc.PinholeSensorMode),
            (vc.ScheimpflugSensorMode(init_tilt_x=0.02), vc.ScheimpflugSensorMode),
        ):
            cfg = vc.RigExtrinsicsCalibrationConfig(sensor=sensor)
            restored = vc.RigExtrinsicsCalibrationConfig.from_mapping(cfg.to_payload())
            self.assertIsInstance(restored.sensor, expected)
            self.assertEqual(restored.sensor.to_payload(), sensor.to_payload())

    def test_rig_handeye_sensor_roundtrip_preserves_variant(self) -> None:
        cfg = vc.RigHandeyeCalibrationConfig(
            sensor=vc.ScheimpflugSensorMode(init_tilt_y=-0.03)
        )
        restored = vc.RigHandeyeCalibrationConfig.from_mapping(cfg.to_payload())
        self.assertIsInstance(restored.sensor, vc.ScheimpflugSensorMode)
        self.assertEqual(restored.sensor.init_tilt_y, -0.03)

    def test_scheimpflug_sensor_from_mapping_rejects_unknown_field(self) -> None:
        with self.assertRaises(ValueError):
            vc.ScheimpflugSensorMode.from_mapping({"kind": "Scheimpflug", "bogus": 1})

    def test_sensor_payload_missing_kind_is_rejected(self) -> None:
        # Rust's internally-tagged SensorMode errors on a missing tag; the Python
        # parser must not silently default to Pinhole and drop Scheimpflug fields.
        with self.assertRaises(ValueError):
            vc.RigExtrinsicsCalibrationConfig.from_mapping({"sensor": {"init_tilt_x": 0.02}})


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
