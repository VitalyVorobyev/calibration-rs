from __future__ import annotations

import json
import unittest
from pathlib import Path

import vision_calibration as vc
from _fixtures import planar_calibration_dataset

KINDS = {
    "observability",
    "convergence",
    "variance_factor",
    "projection_sigma",
    "feature_coverage",
    "view_outliers",
    "parameter_correlation",
}


def _planar_result() -> vc.PlanarIntrinsicsExport:
    return vc.run_planar_intrinsics(planar_calibration_dataset(8), vc.PlanarIntrinsicsConfig())


def _check(report: vc.QualityReport, kind: str) -> vc.QualityCheck:
    return next(c for c in report.checks if c.kind == kind)


class AssessTest(unittest.TestCase):
    def test_planar_result_is_judged_by_every_check(self) -> None:
        result = _planar_result()
        report = vc.assess(result, image_sizes=[(1280, 720)])
        self.assertIn(report.verdict, {"pass", "warn", "fail"})
        self.assertEqual({c.kind for c in report.checks}, KINDS)
        self.assertEqual(_check(report, "observability").status, "pass")
        self.assertEqual(_check(report, "convergence").status, "pass")
        self.assertIsNone(_check(report, "convergence").camera)
        coverage = _check(report, "feature_coverage")
        self.assertEqual(coverage.camera, 0)
        self.assertIsNotNone(coverage.measurement)
        assert coverage.measurement is not None
        self.assertEqual(coverage.measurement.unit, "fraction")
        self.assertTrue(0.0 < coverage.measurement.value <= 1.0)
        # No stated feature noise: the variance test does not apply.
        variance = _check(report, "variance_factor")
        self.assertEqual(variance.status, "not_applicable")
        self.assertIsNone(variance.measurement)
        # The export dict gives the same report.
        self.assertEqual(vc.assess(result.export, image_sizes=[(1280, 720)]), report)

    def test_without_an_image_size_the_image_checks_do_not_apply(self) -> None:
        report = vc.assess(_planar_result())
        for kind in ("projection_sigma", "feature_coverage"):
            check = _check(report, kind)
            self.assertEqual(check.status, "not_applicable")
            self.assertIn("image size", check.message)

    def test_stated_noise_enables_the_variance_test(self) -> None:
        thresholds = vc.QualityThresholds(expected_feature_noise_px=0.5)
        report = vc.assess(_planar_result(), thresholds, [(1280, 720)])
        variance = _check(report, "variance_factor")
        # The fixture is noise-free, far below the stated 0.5 px.
        self.assertEqual(variance.status, "warn")
        assert variance.measurement is not None
        self.assertEqual(variance.measurement.unit, "ratio")
        self.assertLess(variance.measurement.value, variance.measurement.threshold)

    def test_invalid_inputs_are_value_errors(self) -> None:
        result = _planar_result()
        with self.assertRaises(ValueError):
            vc.assess(result, vc.QualityThresholds(coverage_fail_below=0.9))
        with self.assertRaises(ValueError):
            vc.assess(result, image_sizes=[(1280, 720), (1280, 720)])

    def test_unobservable_parameters_fail_and_are_named(self) -> None:
        # The fixture's views do not separate a sensor tilt from a
        # principal-point shift.
        result = vc.run_planar_intrinsics(
            planar_calibration_dataset(8),
            vc.PlanarIntrinsicsConfig(
                camera=vc.CameraConfig(model=vc.CameraModelSpec(sensor="scheimpflug"))
            ),
        )
        report = vc.assess(result, image_sizes=[(1280, 720)])
        self.assertEqual(report.verdict, "fail")
        observability = _check(report, "observability")
        self.assertEqual(observability.status, "fail")
        self.assertIn("tilt_y", observability.subjects)

    def test_rig_cameras_held_fixed_have_no_projection_check(self) -> None:
        path = Path(__file__).resolve().parents[3] / "regression/observations/stereo_rig.json"
        payload = json.loads(path.read_text(encoding="utf-8"))["input"]
        dataset = vc.RigExtrinsicsInput(
            num_cameras=payload["num_cameras"],
            views=[
                vc.RigExtrinsicsView(
                    cameras=[
                        None
                        if cam is None
                        else vc.Observation(points_3d=cam["points_3d"], points_2d=cam["points_2d"])
                        for cam in view["obs"]["cameras"]
                    ]
                )
                for view in payload["views"]
            ],
        )
        report = vc.assess(vc.run_rig_extrinsics(dataset), image_sizes=[(1024, 576)] * 2)
        for camera in (0, 1):
            check = next(
                c for c in report.checks if c.kind == "projection_sigma" and c.camera == camera
            )
            self.assertEqual(check.status, "not_applicable")
            self.assertIn("held", check.message)

    def test_thresholds_payload_names_every_field(self) -> None:
        payload = vc.QualityThresholds().to_payload()
        self.assertIsNone(payload["expected_feature_noise_px"])
        self.assertEqual(payload["coverage_fail_below"], 0.3)
        self.assertEqual(len(payload), 7)


if __name__ == "__main__":
    unittest.main()
