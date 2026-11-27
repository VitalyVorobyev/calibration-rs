"""A device seed from a result or a device spec, and its round trip."""

import json
import tempfile
import unittest
from pathlib import Path

import vision_calibration as vc


def _camera(fx: float) -> dict:
    return {
        "projection": {"type": "pinhole"},
        "distortion": {
            "type": "brown_conrady5",
            "k1": -0.2,
            "k2": 0.0,
            "k3": 0.0,
            "p1": 0.0,
            "p2": 0.0,
            "iters": 8,
        },
        "sensor": {"type": "scheimpflug", "tilt_x": 0.05, "tilt_y": 0.0},
        "intrinsics": {"fx": fx, "fy": fx, "cx": 640.0, "cy": 480.0},
    }


def _pose(tx: float) -> dict:
    return {"rotation": [0.0, 0.0, 0.0, 1.0], "translation": [tx, 0.0, 0.0]}


RIG_EXPORT = {
    "kind": "rig_handeye",
    "schema_version": 1,
    "cameras": [_camera(1500.0), _camera(1510.0)],
    "cam_se3_rig": [_pose(0.0), _pose(-0.1)],
    "rig_se3_target": [_pose(0.0)],
    "handeye": {
        "mode": "eye_in_hand",
        "gripper_se3_rig": _pose(0.05),
        "base_se3_target": _pose(1.0),
    },
    "mean_reproj_error": 0.1,
    "per_cam_reproj_errors": [0.1, 0.1],
}

SPEC = {
    "version": 1,
    "cameras": [
        {
            "id": "cam0",
            "focal_mm": 16.0,
            "pixel_pitch_um": 4.8,
            "resolution_px": [1280, 960],
            "scheimpflug": {"tilt_x_deg": -5.0},
        }
    ],
}


class DeviceSeedTest(unittest.TestCase):
    def test_from_export_carries_the_device(self) -> None:
        seed = vc.DeviceSeed.from_export(RIG_EXPORT, ["left", "right"], [(1280, 960)] * 2)
        self.assertEqual(seed.camera_ids, ["left", "right"])
        cameras = seed.payload["cameras"]
        self.assertEqual(cameras[0]["lens"]["kind"], "calibrated")
        self.assertEqual(cameras[1]["lens"]["camera"]["intrinsics"]["fx"], 1510.0)
        self.assertEqual(list(cameras[1]["cam_se3_rig"]["translation"]), [-0.1, 0.0, 0.0])
        self.assertEqual(seed.payload["handeye"]["mode"], "eye_in_hand")
        self.assertNotIn("rig_se3_target", json.dumps(seed.to_payload()))

    def test_nominal_select_and_round_trip(self) -> None:
        seed = vc.DeviceSeed.from_export(RIG_EXPORT, ["left", "right"], [(1280, 960)] * 2)
        nominal = seed.nominal()
        lens = nominal.payload["cameras"][0]["lens"]
        self.assertEqual(lens["kind"], "nominal")
        self.assertEqual(lens["sensor"]["tilt_x"], 0.05)
        right = seed.select(["right"])
        self.assertEqual(right.camera_ids, ["right"])
        back = vc.DeviceSeed.from_payload(json.loads(json.dumps(seed.to_payload())))
        self.assertEqual(back.to_payload(), seed.to_payload())

    def test_from_spec_path_or_dict(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "spec.json"
            path.write_text(json.dumps(SPEC))
            seed = vc.DeviceSeed.from_spec(path)
        self.assertEqual(seed.to_payload(), vc.DeviceSeed.from_spec(SPEC).to_payload())
        lens = seed.payload["cameras"][0]["lens"]
        self.assertEqual(lens["kind"], "nominal")
        self.assertAlmostEqual(lens["intrinsics"]["fx"], 16.0e-3 / 4.8e-6, places=6)

    def test_aggregate_takes_the_median_of_each_value(self) -> None:
        def unit(fx: float) -> vc.DeviceSeed:
            export = json.loads(json.dumps(RIG_EXPORT))
            export["cameras"][0]["intrinsics"]["fx"] = fx
            return vc.DeviceSeed.from_export(export, ["left", "right"], [(1280, 960)] * 2)

        seeds = [unit(1500.0), unit(1502.0), unit(1600.0)]
        aggregate = vc.DeviceSeed.aggregate(seeds)
        self.assertIsInstance(aggregate, vc.SeedAggregate)
        self.assertEqual(aggregate.units, 3)
        self.assertEqual(aggregate.omitted, [])
        self.assertEqual(aggregate.seed.camera_ids, ["left", "right"])
        lens = aggregate.seed.payload["cameras"][0]["lens"]
        self.assertEqual(lens["camera"]["intrinsics"]["fx"], 1502.0)
        fx = next(s for s in aggregate.spread if s["label"] == "left.fx")
        self.assertEqual(fx["values"], [1500.0, 1502.0, 1600.0])
        self.assertEqual((fx["median"], fx["min"], fx["max"]), (1502.0, 1500.0, 1600.0))
        # |fx - 1502| = 2, 0, 98: the middle one is 2.
        self.assertAlmostEqual(fx["mad_sigma"], 2 * 1.482602218505602, places=9)
        # The order of the seeds does not matter; payloads work as well.
        again = vc.DeviceSeed.aggregate([s.to_payload() for s in reversed(seeds)])
        self.assertEqual(again.seed.to_payload(), aggregate.seed.to_payload())
        self.assertEqual(json.loads(json.dumps(aggregate.to_payload()))["units"], 3)

        with self.assertRaisesRegex(ValueError, "nominal lens is not aggregated"):
            vc.DeviceSeed.aggregate([seeds[0], seeds[1].nominal()])
        with self.assertRaisesRegex(ValueError, "no seed"):
            vc.DeviceSeed.aggregate([])

    def test_errors(self) -> None:
        with self.assertRaisesRegex(ValueError, "2 cameras but 1 camera ids"):
            vc.DeviceSeed.from_export(RIG_EXPORT, ["left"], [(1280, 960)] * 2)
        seed = vc.DeviceSeed.from_export(RIG_EXPORT, ["left", "right"], [(1280, 960)] * 2)
        with self.assertRaisesRegex(ValueError, "no camera with id `middle`"):
            seed.select(["middle"])
        payload = seed.to_payload()
        payload["version"] = 2
        with self.assertRaisesRegex(ValueError, "unsupported device-seed version 2"):
            vc.DeviceSeed.from_payload(payload)


if __name__ == "__main__":
    unittest.main()
