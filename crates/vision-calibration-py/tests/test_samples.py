"""The sample catalogue and fetching a sample from a filled cache.

Nothing here downloads: the fetch test fills the cache from the committed
kuka_1 images and skips when the checkout holds Git LFS pointers instead.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import vision_calibration as vc

KUKA = Path(__file__).resolve().parents[3] / "data/kuka_1"


def kuka_images_present() -> bool:
    first = KUKA / "01.png"
    return first.is_file() and first.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


class CatalogueTest(unittest.TestCase):
    def test_lists_every_sample_with_its_provenance(self) -> None:
        samples = vc.samples.catalogue()
        ids = [s.id for s in samples]
        self.assertIn("opencv_left", ids)
        self.assertIn("kuka_1", ids)
        for sample in samples:
            self.assertTrue(sample.licence and sample.citation and sample.source_url)

    def test_unknown_sample_is_a_value_error(self) -> None:
        with tempfile.TemporaryDirectory() as cache:
            with self.assertRaisesRegex(ValueError, "opencv_left"):
                vc.samples.fetch("no_such_sample", cache)

    def test_fetch_errors_are_os_errors(self) -> None:
        self.assertTrue(issubclass(vc.samples.SampleFetchError, OSError))


@unittest.skipUnless(kuka_images_present(), "kuka_1 images are Git LFS pointers")
class FilledCacheTest(unittest.TestCase):
    def test_a_verified_cache_is_used_without_downloading(self) -> None:
        with tempfile.TemporaryDirectory() as cache:
            shutil.copytree(KUKA, Path(cache) / "kuka_1")
            manifest = vc.samples.fetch("kuka_1", cache)
            self.assertEqual(manifest, Path(cache) / "kuka_1" / "dataset.toml")
            self.assertIn('topology = "single_cam_handeye"', manifest.read_text())


if __name__ == "__main__":
    unittest.main()
