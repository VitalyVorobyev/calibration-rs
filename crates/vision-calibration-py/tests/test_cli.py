"""The ``vcal`` console script, run in process, on synthetic images."""

from __future__ import annotations

import json
import signal
import tempfile
import unittest
from collections.abc import Sequence
from pathlib import Path

from vision_calibration import cli, dataset_runner

import _images

MANIFEST = (
    "version = 1\n"
    'topology = "planar_intrinsics"\n'
    "[[cameras]]\n"
    'id = "cam0"\n'
    'images = { kind = "glob", pattern = "view_*.png" }\n'
    "[target]\n"
    'kind = "chessboard"\nrows = 5\ncols = 7\nsquare_size_m = 0.03\n'
)


def vcal(args: Sequence[str]) -> int:
    """Run ``vcal`` and return its exit status, keeping the test process's
    Ctrl-C handler."""
    handler = signal.getsignal(signal.SIGINT)
    try:
        cli.main(args)
    except SystemExit as exit:
        return int(exit.code or 0)
    finally:
        signal.signal(signal.SIGINT, handler)
    raise AssertionError("vcal returned without exiting")


class VcalTest(unittest.TestCase):
    def test_run_calibrates_and_writes_its_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _images.write_planar_dataset(root, views=8)
            manifest = root / "dataset.toml"
            manifest.write_text(MANIFEST)
            self.assertEqual(vcal(["check", str(manifest)]), 0)
            out = root / "out"
            status = vcal(["run", str(manifest), "-o", str(out)])
            export = json.loads((out / "calibration.json").read_text())
            fx = export["params"]["camera"]["intrinsics"]["fx"]
            self.assertAlmostEqual(fx, _images.FX, delta=0.03 * _images.FX)
            # These small boards leave most of the image uncovered: the
            # verdict fails, and the status says so after writing the files.
            report = json.loads((out / "quality.json").read_text())
            self.assertEqual(status, 3 if report["verdict"] == "fail" else 0)

    def test_camera_image_sizes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = _images.write_planar_dataset(root, views=2)
            sizes = dataset_runner.camera_image_sizes(spec, base_dir=root)
            self.assertEqual(sizes, [(_images.WIDTH, _images.HEIGHT)])

    def test_errors_and_usage(self) -> None:
        self.assertEqual(vcal(["calibrate"]), 2)
        self.assertEqual(vcal(["check", "no/such/dataset.toml"]), 1)


if __name__ == "__main__":
    unittest.main()
