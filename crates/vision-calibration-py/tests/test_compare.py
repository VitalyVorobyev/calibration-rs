from __future__ import annotations

import unittest

import vision_calibration as vc
from _fixtures import planar_calibration_dataset

SIZE = (1280, 720)


def _run(name: str, views: int) -> dict:
    result = vc.planar_intrinsics.run_calibration(
        planar_calibration_dataset(views), vc.PlanarIntrinsicsConfig()
    )
    return {"name": name, "export": result, "camera_ids": ["cam"], "image_sizes": [SIZE]}


class CompareRunsTest(unittest.TestCase):
    def test_two_runs_side_by_side(self) -> None:
        full, part = _run("full", 10), _run("part", 8)
        part["relation"] = {"kind": "nested", "smaller": "other"}
        comparison = vc.compare_runs([full, part])
        self.assertEqual([r["name"] for r in comparison["runs"]], ["full", "part"])
        evidence = comparison["runs"][0]
        self.assertEqual(evidence["kind"], "planar_intrinsics")
        self.assertIn(evidence["verdict"], {"pass", "warn", "fail"})
        self.assertEqual(evidence["cameras"][0]["id"], "cam")
        self.assertEqual(evidence["cameras"][0]["views"], 10)
        difference = comparison["differences"][0]
        # The 8 views are the first 8 of the 10: the stated relation holds.
        self.assertEqual(difference["relation"], {"kind": "nested", "smaller": "other"})
        self.assertNotIn("relation_refused", difference)
        camera = difference["cameras"][0]
        self.assertEqual(camera["parameters"]["kind"], "same_model")
        self.assertIn("joint", camera["parameters"])
        fx = next(p for p in camera["parameters"]["parameters"] if p["label"] == "fx")
        # Noise-free views: both runs recover the same camera.
        self.assertAlmostEqual(fx["delta"], 0.0, places=4)
        self.assertEqual(fx["test"]["kind"], "nested")
        self.assertEqual(
            camera["parameters"]["joint"], {"kind": "not_available", "reason": "nested_joint"}
        )
        self.assertEqual(camera["projection"]["status"], "compared")
        self.assertLess(camera["projection"]["rms_fit_px"], 1e-3)

    def test_a_contradicted_relation_is_refused(self) -> None:
        a, b = _run("a", 10), _run("b", 9)
        b["relation"] = {"kind": "independent"}
        difference = vc.compare_runs([a, b])["differences"][0]
        self.assertEqual(difference["relation"], {"kind": "overlapping"})
        self.assertEqual(
            difference["relation_refused"],
            {"stated": {"kind": "independent"}, "reason": "shared_features"},
        )

    def test_without_a_relation_there_is_no_test(self) -> None:
        a, b = _run("a", 10), _run("b", 9)
        a["config"] = vc.PlanarIntrinsicsConfig()
        b["config"] = vc.PlanarIntrinsicsConfig()
        comparison = vc.compare_runs([a, b])
        difference = comparison["differences"][0]
        self.assertEqual(difference["relation"], {"kind": "unknown"})
        self.assertEqual(difference["config_changes"], [])
        for p in difference["cameras"][0]["parameters"]["parameters"]:
            self.assertEqual(p["test"], {"kind": "not_available", "reason": "unknown_relation"})

    def test_rejects_bad_input(self) -> None:
        a = _run("a", 8)
        with self.assertRaises(ValueError):
            vc.compare_runs([a])
        with self.assertRaises(ValueError):
            vc.compare_runs([a, dict(a, camera_ids=["other"])])
        with self.assertRaises(ValueError):
            vc.compare_runs([dict(a, relation={"kind": "independent"}), a])


if __name__ == "__main__":
    unittest.main()
