"""Config-payload schema tests for the R5 parity additions (no solver).

These pin the ``to_payload()`` shape of the new/changed calibration configs to
what the Rust ``serde`` derives expect. The failure mode they guard against is
silent drift: because the Rust config structs do **not** use
``#[serde(deny_unknown_fields)]``, a mis-named Python payload key would be
ignored and the field would fall back to its Rust default — a wrong result with
no error. Each key name and default value asserted here was read off the Rust
source of truth:

* ``common/config.rs`` — ``default_distortion_kind`` (``BrownConrady5``).
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

    def test_planar_default_distortion_model_is_brown_conrady5(self) -> None:
        payload = vc.PlanarCalibrationConfig().to_payload()
        self.assertEqual(payload["distortion_model"], "brown_conrady5")
        # The new key sits at top level next to the pre-existing grouped keys.
        self.assertEqual(
            set(payload),
            {"init", "solver", "distortion_model", "fix_camera", "fix_poses"},
        )

    def test_planar_distortion_model_roundtrips_each_model(self) -> None:
        for model in ("none", "brown_conrady5", "rational8", "thin_prism9", "division1"):
            cfg = vc.PlanarCalibrationConfig(distortion_model=model)  # type: ignore[arg-type]
            payload = cfg.to_payload()
            self.assertEqual(payload["distortion_model"], model)
            restored = vc.PlanarCalibrationConfig.from_mapping(payload)
            self.assertEqual(restored.distortion_model, model)

    def test_scheimpflug_default_payload_shape(self) -> None:
        payload = vc.ScheimpflugIntrinsicsCalibrationConfig().to_payload()
        self.assertEqual(payload["distortion_model"], "brown_conrady5")
        # Scheimpflug overrides the shared camera-fix default to `radial_only`.
        self.assertEqual(payload["fix_camera"]["distortion"], _RADIAL_ONLY)
        # `ScheimpflugFixMask::default()` frees both tilt axes.
        self.assertEqual(payload["fix_scheimpflug"], {"tilt_x": False, "tilt_y": False})

    def test_scheimpflug_distortion_model_roundtrips(self) -> None:
        cfg = vc.ScheimpflugIntrinsicsCalibrationConfig(distortion_model="thin_prism9")
        payload = cfg.to_payload()
        self.assertEqual(payload["distortion_model"], "thin_prism9")
        restored = vc.ScheimpflugIntrinsicsCalibrationConfig.from_mapping(payload)
        self.assertEqual(restored.distortion_model, "thin_prism9")


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
