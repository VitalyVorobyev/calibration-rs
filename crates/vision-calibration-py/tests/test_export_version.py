"""Every export carries its format version, and a reader rejects another one."""

import unittest

import vision_calibration as vc

EXPORT_CLASSES = (
    vc.PlanarIntrinsicsExport,
    vc.SingleCamHandeyeExport,
    vc.RigExtrinsicsExport,
    vc.RigHandeyeExport,
    vc.LaserlineDeviceExport,
    vc.RigLaserlineDeviceExport,
    vc.RigHandeyeLaserlineExport,
)


class ExportVersionTest(unittest.TestCase):
    def test_every_export_class_rejects_a_foreign_or_missing_version(self) -> None:
        for cls in EXPORT_CLASSES:
            for payload in ({"schema_version": 2}, {}):
                with self.subTest(cls=cls.__name__, payload=payload):
                    with self.assertRaisesRegex(ValueError, r"schema_version.*reads 1"):
                        cls.from_payload(payload)

    def test_a_result_carries_the_version_it_was_written_with(self) -> None:
        result = vc.RigLaserlineDeviceExport.from_payload(
            {"schema_version": 1, "laser_planes_rig": [], "laser_planes_cam": [], "per_cam_stats": []}
        )
        self.assertEqual(result.export["schema_version"], 1)

    def test_the_native_readers_reject_a_foreign_version_naming_both(self) -> None:
        export = {"kind": "planar_intrinsics", "schema_version": 9}
        with self.assertRaisesRegex(Exception, r"schema_version 9.*reads 1"):
            vc.projection_uncertainty(export, (640, 480))


if __name__ == "__main__":
    unittest.main()
