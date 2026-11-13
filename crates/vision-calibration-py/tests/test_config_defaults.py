"""Each Python config's defaults are the Rust config's defaults.

``cargo xtask emit-schemas`` writes ``schemas/<name>.default.json``, the JSON
of each problem config's Rust ``Default``. The Python dataclasses restate those
defaults by hand; this test builds each with no arguments and compares its
``to_payload()`` with the file, so a Rust default that changes without the
Python one fails here instead of silently giving Python users another
calibration.

The comparison is exact, floats included: every default is a literal or, for
``robot_poses.rot_sigma``, half a degree in radians, which both sides compute
to the same ``f64``.

The schemas live in the source checkout, not in the wheel; outside a checkout
the test skips.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from typing import Any

import vision_calibration as vc

SCHEMAS = Path(__file__).resolve().parents[3] / "schemas"

# The Python config of each `schemas/<name>.default.json`.
CONFIGS: dict[str, type[Any]] = {
    "planar_intrinsics_config": vc.PlanarIntrinsicsConfig,
    "single_cam_handeye_config": vc.SingleCamHandeyeConfig,
    "rig_extrinsics_config": vc.RigExtrinsicsConfig,
    "rig_handeye_config": vc.RigHandeyeConfig,
    "laserline_device_config": vc.LaserlineDeviceConfig,
    "rig_laserline_device_config": vc.RigLaserlineDeviceConfig,
    "rig_handeye_laserline_config": vc.RigHandeyeLaserlineConfig,
}


@unittest.skipUnless(SCHEMAS.is_dir(), "the schemas directory is only in a source checkout")
class ConfigDefaultsFollowSchemasTest(unittest.TestCase):
    def test_every_default_file_has_a_python_config(self) -> None:
        names = {p.name.removesuffix(".default.json") for p in SCHEMAS.glob("*.default.json")}
        self.assertEqual(names, set(CONFIGS))

    def test_python_defaults_equal_the_rust_defaults(self) -> None:
        for name, cls in CONFIGS.items():
            with self.subTest(config=cls.__name__):
                expected = json.loads((SCHEMAS / f"{name}.default.json").read_text())
                self.assertEqual(cls().to_payload(), expected)


if __name__ == "__main__":
    unittest.main()
