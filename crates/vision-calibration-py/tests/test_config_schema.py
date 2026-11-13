"""Config-payload schema tests for the calibration configs (no solver).

These pin the ``to_payload()`` shape of non-default configs to what the Rust
``serde`` derives expect, and check that ``from_mapping`` rejects a key Rust
does not have. The failure mode they guard against is silent drift: a
mis-named Python payload key must not be ignored so that the field falls back
to its Rust default — a wrong result with no error.

The default payload of each problem config is compared whole with the Rust
default in ``test_config_defaults.py``.
"""

from __future__ import annotations

import typing
import unittest

import vision_calibration as vc
import vision_calibration.types as vc_types


class DistortionModelSchemaTest(unittest.TestCase):
    def test_distortion_model_literal_matches_serde_snake_case(self) -> None:
        # `DistortionKind` is `#[serde(rename_all = "snake_case")]`; the Python
        # Literal must carry exactly the serde spellings.
        self.assertEqual(
            set(typing.get_args(vc_types.DistortionKind)),
            {
                "none",
                "brown_conrady5",
                "rational8",
                "thin_prism9",
                "division1",
                "kannala_brandt4",
            },
        )

    def test_init_has_no_tangential_setting(self) -> None:
        # The camera's fix lists decide the linear stage's tangential terms.
        with self.assertRaises(ValueError):
            vc.IntrinsicsInitConfig.from_mapping({"fix_tangential": True})

    def test_planar_distortion_roundtrips_each_model(self) -> None:
        for model in ("none", "brown_conrady5", "rational8", "thin_prism9", "division1"):
            cfg = vc.PlanarIntrinsicsConfig(
                camera=vc.CameraConfig(model=vc.CameraModelSpec(distortion=model))  # type: ignore[arg-type]
            )
            payload = cfg.to_payload()
            self.assertEqual(payload["camera"]["model"]["distortion"], model)
            restored = vc.PlanarIntrinsicsConfig.from_mapping(payload)
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
            vc.PlanarIntrinsicsConfig.from_mapping({"distortion_model": "none"})


class RigCameraConfigSchemaTest(unittest.TestCase):
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
        cfg = vc.RigExtrinsicsConfig(
            camera=vc.CameraConfig(model=vc.CameraModelSpec(distortion="none")),
            camera_overrides=[
                vc.CameraOverride(1, model=vc.CameraModelSpec(sensor="scheimpflug"))
            ],
        )
        cfg.rig.refine_sensor = False
        restored = vc.RigExtrinsicsConfig.from_mapping(cfg.to_payload())
        self.assertEqual(restored.to_payload(), cfg.to_payload())
        self.assertIsInstance(restored.camera_overrides[0], vc.CameraOverride)
        self.assertEqual(restored.camera_overrides[0].model.sensor, "scheimpflug")
        self.assertFalse(restored.rig.refine_sensor)

    def test_rig_handeye_overrides_roundtrip(self) -> None:
        cfg = vc.RigHandeyeConfig(
            camera_overrides=[vc.CameraOverride(0, fix=vc.CameraFix(distortion=["k1"]))]
        )
        cfg.handeye_optimize.refine_sensor = False
        restored = vc.RigHandeyeConfig.from_mapping(cfg.to_payload())
        self.assertEqual(restored.to_payload(), cfg.to_payload())
        self.assertFalse(restored.handeye_optimize.refine_sensor)

    def test_removed_sensor_field_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            vc.RigExtrinsicsConfig.from_mapping({"sensor": {"kind": "Pinhole"}})
        with self.assertRaises(ValueError):
            vc.HandeyeOptimizeConfig.from_mapping({"refine_scheimpflug": True})

    def test_camera_override_rejects_unknown_and_missing_fields(self) -> None:
        with self.assertRaises(ValueError):
            vc.CameraOverride.from_payload({"camera": 0, "bogus": 1})
        with self.assertRaises(ValueError):
            vc.CameraOverride.from_payload({"model": {}})


class SingleCameraConfigSchemaTest(unittest.TestCase):
    def test_single_cam_handeye_camera_roundtrips(self) -> None:
        cfg = vc.SingleCamHandeyeConfig(
            camera=vc.CameraConfig(
                model=vc.CameraModelSpec(distortion="division1", sensor="scheimpflug"),
                fix=vc.CameraFix(sensor=["tilt_y"]),
            )
        )
        restored = vc.SingleCamHandeyeConfig.from_mapping(cfg.to_payload())
        self.assertEqual(restored.to_payload(), cfg.to_payload())

    def test_laserline_device_camera_roundtrips_and_rejects_removed_fields(self) -> None:
        with self.assertRaises(ValueError):
            vc.LaserlineDeviceConfig.from_mapping({"sensor_init": {}})
        with self.assertRaises(ValueError):
            vc.DeviceOptimizeConfig.from_mapping({"fix_sensor": True})
        cfg = vc.LaserlineDeviceConfig(
            camera=vc.CameraConfig(model=vc.CameraModelSpec(sensor="scheimpflug"))
        )
        restored = vc.LaserlineDeviceConfig.from_mapping(cfg.to_payload())
        self.assertEqual(restored.to_payload(), cfg.to_payload())


class RigHandeyeLaserlineConfigSchemaTest(unittest.TestCase):
    def test_config_roundtrips_through_from_mapping(self) -> None:
        cfg = vc.RigHandeyeLaserlineConfig()
        restored = vc.RigHandeyeLaserlineConfig.from_mapping(cfg.to_payload())
        self.assertEqual(restored.to_payload(), cfg.to_payload())

    def test_ba_config_roundtrips_and_rejects_unknown_field(self) -> None:
        ba = vc.JointOptimizeConfig(
            reprojection_weight=2.0,
            laser_weight=5.0e3,
            fix=[vc.CameraFix(), vc.CameraFix(sensor=["tilt_x"])],
        )
        payload = ba.to_payload()
        self.assertEqual(payload["fix"], [{}, {"sensor": ["tilt_x"]}])
        restored = vc.JointOptimizeConfig.from_mapping(payload)
        self.assertEqual(restored.reprojection_weight, 2.0)
        self.assertEqual(restored.laser_weight, 5.0e3)
        self.assertEqual(restored.to_payload(), payload)
        with self.assertRaises(ValueError):
            vc.JointOptimizeConfig.from_mapping({"bogus": 1})
        with self.assertRaises(ValueError):
            vc.JointOptimizeConfig.from_mapping({"default_camera_fix": {}})

    def test_config_rejects_unknown_field(self) -> None:
        with self.assertRaises(ValueError):
            vc.RigHandeyeLaserlineConfig.from_mapping({"bogus": 1})


class RigLaserlineDeviceConfigSchemaTest(unittest.TestCase):
    def test_default_payload_roundtrips_and_rejects_unknown_fields(self) -> None:
        payload = vc.RigLaserlineDeviceConfig().to_payload()
        restored = vc.RigLaserlineDeviceConfig.from_mapping(payload)
        self.assertEqual(restored.to_payload(), payload)
        with self.assertRaises(ValueError):
            vc.RigLaserlineDeviceConfig.from_mapping({"laser_residual_type": "point_to_plane"})
        with self.assertRaises(ValueError):
            vc.PlaneOptimizeConfig.from_mapping({"bogus": 1})


class SolverConfigSchemaTest(unittest.TestCase):
    def test_solver_has_no_robust_loss(self) -> None:
        with self.assertRaises(ValueError):
            vc.SolverConfig.from_mapping({"robust_loss": {"kind": "none"}})


if __name__ == "__main__":
    unittest.main()
