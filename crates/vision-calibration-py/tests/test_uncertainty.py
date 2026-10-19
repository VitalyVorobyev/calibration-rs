from __future__ import annotations

import json
import math
import unittest
from pathlib import Path

import vision_calibration as vc
from _fixtures import planar_calibration_dataset


def _available_payload() -> dict:
    return {
        "status": "available",
        "labels": ["fx", "cx"],
        "covariance": [[4.0, -1.2], [-1.2, 1.0]],
        "views": [{"view": 0, "labels": ["camera_se3_target.tz"], "covariance": [[1e-8]]}],
        "fixed": ["k3"],
        "noise": [{"kind": "reprojection", "sigma": 0.2, "count": 400}],
        "num_residuals": 400,
        "dof": 380,
    }


class ParameterUncertaintyParsingTest(unittest.TestCase):
    def test_available_payload_reads_std_dev_and_correlation(self) -> None:
        u = vc.ParameterUncertainty.from_payload(_available_payload())
        self.assertTrue(u.available)
        self.assertAlmostEqual(u.std_dev("fx"), 2.0)
        self.assertAlmostEqual(u.correlation("fx", "cx"), -0.6)
        self.assertEqual(u.fixed, ["k3"])
        self.assertEqual(u.views[0].labels, ["camera_se3_target.tz"])
        self.assertEqual(u.noise[0].kind, "reprojection")
        with self.assertRaises(KeyError):
            u.std_dev("k3")

    def test_unavailable_payload_keeps_the_reason(self) -> None:
        u = vc.ParameterUncertainty.from_payload(
            {
                "status": "unavailable",
                "reason": {"kind": "rank_deficient", "unobservable": ["cx", "cy"]},
            }
        )
        self.assertFalse(u.available)
        self.assertEqual(u.reason, "rank_deficient")
        self.assertEqual(u.reason_labels, ["cx", "cy"])
        with self.assertRaises(ValueError):
            u.std_dev("fx")

    def test_missing_payload_is_not_computed(self) -> None:
        u = vc.ParameterUncertainty.from_payload(None)
        self.assertFalse(u.available)
        self.assertEqual(u.reason, "not_computed")


class LaserResultUncertaintyTest(unittest.TestCase):
    def test_rig_laserline_noise_names_its_camera(self) -> None:
        plane = {"normal": [0.0, 0.0, 1.0], "distance": -0.3}
        stats = {
            "mean_reproj_error": 0.1,
            "mean_laser_error": 0.2,
            "per_view_reproj_errors": [0.1],
            "per_view_laser_errors": [0.2],
        }
        uncertainty = {
            "status": "available",
            "labels": ["cam0.plane.d", "cam1.plane.d"],
            "covariance": [[1e-8, 0.0], [0.0, 4e-8]],
            "views": [],
            "fixed": ["cam0.fx", "cam1.fx"],
            "noise": [
                {"kind": "laser_line", "sigma": 0.2, "count": 60, "camera": 0},
                {"kind": "laser_line", "sigma": 0.3, "count": 60, "camera": 1},
            ],
            "num_residuals": 120,
            "dof": 118,
        }
        result = vc.RigLaserlineDeviceResult.from_payload(
            {
                "laser_planes_rig": [plane, plane],
                "laser_planes_cam": [plane, plane],
                "per_camera_stats": [stats, stats],
                "uncertainty": uncertainty,
            }
        )
        self.assertAlmostEqual(result.uncertainty.std_dev("cam1.plane.d"), 2e-4)
        self.assertEqual([n.camera for n in result.uncertainty.noise], [0, 1])
        shared = vc.ParameterUncertainty.from_payload(_available_payload())
        self.assertIsNone(shared.noise[0].camera)


class ResultUncertaintyTest(unittest.TestCase):
    def test_planar_result_carries_the_covariance(self) -> None:
        result = vc.run_planar_intrinsics(
            planar_calibration_dataset(8), vc.PlanarCalibrationConfig()
        )
        u = result.uncertainty
        self.assertTrue(u.available, u.reason)
        self.assertEqual(u.labels[:4], ["fx", "fy", "cx", "cy"])
        self.assertIn("k3", u.fixed)
        self.assertTrue(math.isfinite(u.std_dev("fx")))
        self.assertLessEqual(abs(u.correlation("fx", "fy")), 1.0)
        self.assertEqual(len(u.views), 8)

    def test_scheimpflug_reports_the_tilt_principal_point_degeneracy(self) -> None:
        # The fixture camera has no tilt, and its views do not separate a
        # sensor tilt from a principal-point shift: the result names them
        # instead of reporting a meaningless covariance.
        result = vc.run_scheimpflug_intrinsics(
            planar_calibration_dataset(8),
            vc.ScheimpflugIntrinsicsCalibrationConfig(
                fix_scheimpflug={"tilt_x": False, "tilt_y": False}
            ),
        )
        u = result.uncertainty
        self.assertFalse(u.available)
        self.assertEqual(u.reason, "rank_deficient")
        self.assertIn("tilt_y", u.reason_labels)
        self.assertIn("cx", u.reason_labels)


    def test_rig_result_names_each_camera(self) -> None:
        # The committed stereo observations (two cameras, 20 views).
        path = Path(__file__).resolve().parents[3] / "regression/observations/stereo_rig.json"
        payload = json.loads(path.read_text(encoding="utf-8"))["input"]
        dataset = vc.RigExtrinsicsDataset(
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
        u = vc.run_rig_extrinsics(dataset).uncertainty
        self.assertTrue(u.available, u.reason)
        self.assertIn("cam1.cam_se3_rig.tx", u.labels)
        self.assertIn("cam0.cam_se3_rig", u.fixed)
        self.assertLess(u.std_dev("cam1.cam_se3_rig.tx"), 1e-3)
        self.assertEqual(len(u.views), 20)
        self.assertEqual(u.views[0].labels[0], "rig_se3_target.rx")


if __name__ == "__main__":
    unittest.main()
