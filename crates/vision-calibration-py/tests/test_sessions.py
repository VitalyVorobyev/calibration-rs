"""The session API: every problem type driven step by step.

The inputs are the committed, image-free observations of the regression
suite (``regression/observations``) and the synthetic planar fixture.
"""

from __future__ import annotations

import json
import math
import unittest
from pathlib import Path
from typing import Any

import vision_calibration as vc
from vision_calibration import (
    laserline_device,
    planar_intrinsics,
    rig_extrinsics,
    rig_handeye,
    rig_laserline_device,
    single_cam_handeye,
)
from vision_calibration._api import (
    _run_laserline_device_raw,
    _run_rig_handeye_laserline_raw,
    _run_rig_extrinsics_raw,
    _run_single_cam_handeye_raw,
)

from _fixtures import planar_calibration_dataset

OBSERVATIONS = Path(__file__).resolve().parents[3] / "regression/observations"


def observation(name: str) -> Any:
    """The input payload of a committed regression observation."""
    return json.loads((OBSERVATIONS / f"{name}.json").read_text(encoding="utf-8"))["input"]


def rig_handeye_input() -> dict[str, Any]:
    """The rig hand-eye part of the synthetic joint laser scene."""
    joint = observation("synth_rig_handeye_laserline")
    return {
        "num_cameras": joint["num_cameras"],
        "views": [
            {"obs": {"cameras": v["obs"]["cameras"]}, "meta": v["meta"]}
            for v in joint["views"]
        ],
    }


class PlanarIntrinsicsSessionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = planar_calibration_dataset()

    def test_steps_match_the_one_shot_run(self) -> None:
        session = vc.PlanarIntrinsicsSession()
        session.set_input(self.dataset)
        init = session.step_init()
        self.assertIsInstance(init, planar_intrinsics.IntrinsicsInitResult)
        self.assertEqual(len(init.poses), 10)
        self.assertIsNone(init.sensor)
        self.assertAlmostEqual(init.intrinsics.fx, 800.0, delta=8.0)
        optimized = session.step_optimize()
        self.assertLess(optimized.mean_reproj_error, 1e-6)
        result = session.export()
        reference = vc.run_planar_intrinsics(self.dataset)
        self.assertEqual(result.camera, reference.camera)
        self.assertEqual([e.operation for e in session.log], ["init", "optimize"])
        self.assertTrue(all(e.success for e in session.log))

    def test_step_options_override_the_config(self) -> None:
        session = vc.PlanarIntrinsicsSession()
        session.set_input(self.dataset)
        session.step_init(vc.IntrinsicsInitOptions(iterations=1))
        capped = session.step_optimize(vc.IntrinsicsOptimizeOptions(max_iters=1))
        self.assertLessEqual(capped.iterations, 1)

    def test_seeded_init_takes_the_seed(self) -> None:
        session = vc.PlanarIntrinsicsSession()
        session.set_input(self.dataset)
        seed = planar_intrinsics.IntrinsicsManualInit(
            intrinsics=vc.PinholeIntrinsics(fx=790.0, fy=770.0, cx=630.0, cy=350.0)
        )
        init = session.step_init_with_seed(seed)
        self.assertEqual(init.intrinsics, vc.PinholeIntrinsics(fx=790.0, fy=770.0, cx=630.0, cy=350.0))
        notes = session.log[-1].notes or ""
        self.assertIn("manual", notes)
        session.step_optimize()
        self.assertAlmostEqual(session.export().camera.intrinsics.fx, 800.0, delta=1e-3)

    def test_filtering_clears_the_estimate(self) -> None:
        session = vc.PlanarIntrinsicsSession()
        session.set_input(self.dataset)
        session.run_calibration()
        self.assertTrue(session.has_output)
        session.step_filter(vc.IntrinsicsFilterOptions(max_reproj_error=1.0))
        self.assertFalse(session.has_output)
        session.run_calibration_with_filtering()
        self.assertTrue(session.has_output)

    def test_session_state_round_trips(self) -> None:
        session = vc.PlanarIntrinsicsSession("bench A")
        self.assertIsNone(session.input)
        self.assertFalse(session.has_input)
        session.set_config(vc.PlanarIntrinsicsConfig(fix_poses=[0]))
        self.assertEqual(session.config.fix_poses, [0])
        session.set_input(self.dataset)
        session.run_calibration()
        first = session.export(notes="first")
        restored = vc.PlanarIntrinsicsSession.from_json(session.to_json())
        self.assertEqual(restored.metadata.description, "bench A")
        self.assertEqual(restored.metadata.problem_type, session.metadata.problem_type)
        self.assertEqual(restored.config.fix_poses, [0])
        self.assertTrue(restored.has_output)
        # JSON keeps every float to within an ulp.
        self.assertAlmostEqual(
            restored.export().camera.intrinsics.fx, first.camera.intrinsics.fx, places=9
        )
        self.assertEqual([r.notes for r in session.exports], ["first"])
        self.assertIsInstance(session.exports[0].export, vc.PlanarIntrinsicsExport)
        self.assertIsNotNone(session.output)
        session.reset()
        self.assertFalse(session.has_input)
        self.assertEqual(session.log, [])

    def test_errors_are_value_errors(self) -> None:
        session = vc.PlanarIntrinsicsSession()
        with self.assertRaises(ValueError):
            session.step_init()  # no input
        session.set_input(self.dataset)
        with self.assertRaises(ValueError):
            session.step_optimize()  # not initialized
        with self.assertRaises(ValueError):
            session.export()  # nothing computed
        with self.assertRaises(ValueError):
            session.step_init_with_seed(
                planar_intrinsics.IntrinsicsManualInit(
                    intrinsics=vc.PinholeIntrinsics(fx=math.nan, fy=1.0, cx=0.0, cy=0.0)
                )
            )
        with self.assertRaises(TypeError):
            session.step_init({"iteration": 3})  # misspelled option
        with self.assertRaises(ValueError):
            vc.SingleCamHandeyeSession.from_json(session.to_json())

    def test_a_scheimpflug_seed_on_an_untilted_camera_is_rejected(self) -> None:
        session = vc.PlanarIntrinsicsSession()
        session.set_input(self.dataset)
        seed = planar_intrinsics.IntrinsicsManualInit(
            intrinsics=vc.PinholeIntrinsics(fx=800.0, fy=780.0, cx=640.0, cy=360.0),
            sensor=vc.ScheimpflugSensor(tilt_x=0.01, tilt_y=0.0),
        )
        with self.assertRaises(ValueError):
            session.step_init_with_seed(seed)


class SingleCamHandeyeSessionTest(unittest.TestCase):
    def test_stages_match_the_one_shot_run(self) -> None:
        payload = observation("synth_handeye")
        session = vc.SingleCamHandeyeSession()
        session.set_input(payload)
        init = session.step_intrinsics_init()
        self.assertEqual(len(init.target_poses), len(payload["views"]))
        optimized = session.step_intrinsics_optimize()
        self.assertLess(optimized.mean_reproj_error, 1.0)
        handeye = session.step_handeye_init(vc.HandeyeInitOptions(min_motion_angle_deg=5.0))
        self.assertIsNotNone(handeye.gripper_se3_camera)
        self.assertIsNone(handeye.camera_se3_base)
        refined = session.step_handeye_optimize()
        result = session.export()
        self.assertAlmostEqual(result.mean_reproj_error, refined.mean_reproj_error, places=9)
        reference = _run_single_cam_handeye_raw(payload)
        self.assertEqual(result.gripper_se3_camera, reference.gripper_se3_camera)

    def test_seeding_the_hand_eye_with_its_estimate_keeps_it(self) -> None:
        session = vc.SingleCamHandeyeSession()
        session.set_input(observation("synth_handeye"))
        session.step_intrinsics_init()
        session.step_intrinsics_optimize()
        estimate = session.step_handeye_init()
        seeded = session.step_handeye_init_with_seed(
            single_cam_handeye.HandeyeManualInit(
                gripper_se3_camera=estimate.gripper_se3_camera,
                base_se3_target=estimate.base_se3_target,
            )
        )
        self.assertEqual(seeded, estimate)


class RigExtrinsicsSessionTest(unittest.TestCase):
    def test_stages_match_the_one_shot_run(self) -> None:
        payload = observation("synth_rig")
        config = vc.RigExtrinsicsConfig(rig=vc.RigConfig(refine_intrinsics=True))
        session = vc.RigExtrinsicsSession()
        session.set_input(payload)
        session.set_config(config)
        init = session.step_intrinsics_init_all()
        self.assertEqual(len(init.cameras), 2)
        self.assertEqual(len(init.per_cam_target_poses), len(payload["views"]))
        session.step_intrinsics_optimize_all()
        rig = session.step_rig_init()
        self.assertEqual(len(rig.initial_cam_se3_rig), 2)
        refined = session.step_rig_optimize(vc.RigOptimizeOptions(max_iters=100))
        self.assertLess(refined.mean_reproj_error, 1.0)
        reference = _run_rig_extrinsics_raw(payload, config.to_payload())
        self.assertEqual(session.export().cam_se3_rig, reference.cam_se3_rig)

    def test_seeded_rig_init_is_the_seed(self) -> None:
        session = vc.RigExtrinsicsSession()
        session.set_input(observation("synth_rig"))
        intrinsics = session.step_intrinsics_init_all()
        seeded = session.step_intrinsics_init_all_with_seed(
            rig_extrinsics.IntrinsicsManualInit(
                per_cam_intrinsics=[c.intrinsics for c in intrinsics.cameras]
            )
        )
        self.assertEqual(
            [c.intrinsics for c in seeded.cameras], [c.intrinsics for c in intrinsics.cameras]
        )
        session.step_intrinsics_optimize_all()
        rig = session.step_rig_init()
        again = session.step_rig_init_with_seed(
            rig_extrinsics.RigManualInit(
                cam_se3_rig=rig.initial_cam_se3_rig, rig_se3_target=rig.initial_rig_se3_target
            )
        )
        self.assertEqual(again, rig)
        with self.assertRaises(ValueError):  # the two seeds go together
            session.step_rig_init_with_seed(
                rig_extrinsics.RigManualInit(cam_se3_rig=rig.initial_cam_se3_rig)
            )


class RigHandeyeSessionTest(unittest.TestCase):
    def test_three_stages_run(self) -> None:
        session = vc.RigHandeyeSession()
        session.set_input(rig_handeye_input())
        session.step_intrinsics_init_all()
        intrinsics = session.step_intrinsics_optimize_all()
        self.assertEqual(len(intrinsics.per_cam_reproj_errors), 2)
        session.step_rig_init()
        rig = session.step_rig_optimize()
        self.assertEqual(len(rig.cam_se3_rig), 2)
        handeye = session.step_handeye_init()
        self.assertIsInstance(handeye.initial_handeye, vc.Pose)
        reseeded = session.step_handeye_init_with_seed(
            rig_handeye.HandeyeManualInit(
                handeye=handeye.initial_handeye, mode_target_pose=handeye.initial_mode_target_pose
            )
        )
        self.assertEqual(reseeded, handeye)
        refined = session.step_handeye_optimize()
        self.assertGreater(refined.final_cost, 0.0)
        result = session.export()
        self.assertIsNotNone(result.gripper_se3_rig)
        self.assertLess(result.mean_reproj_error, 1.0)


class LaserlineDeviceSessionTest(unittest.TestCase):
    def test_steps_match_the_one_shot_run(self) -> None:
        payload = observation("synth_laserline")
        session = vc.LaserlineDeviceSession()
        session.set_input(payload)
        init = session.step_init(vc.DeviceInitOptions(iterations=2))
        self.assertIsNotNone(init.plane_rmse)
        optimized = session.step_optimize(vc.DeviceOptimizeOptions(max_iters=60))
        self.assertLess(optimized.mean_reproj_error, 1.0)
        self.assertEqual(len(optimized.per_view_laser_errors), len(payload))
        result = session.export()
        reference = _run_laserline_device_raw(payload)
        self.assertEqual(result.estimate.params.plane, reference.estimate.params.plane)

    def test_a_seeded_plane_has_no_fit_rmse(self) -> None:
        session = vc.LaserlineDeviceSession()
        session.set_input(observation("synth_laserline"))
        init = session.step_init()
        seeded = session.step_init_with_seed(laserline_device.DeviceManualInit(plane=init.plane))
        self.assertIsNone(seeded.plane_rmse)
        self.assertEqual(seeded.plane, init.plane)


class RigLaserlineDeviceSessionTest(unittest.TestCase):
    def test_planes_fit_against_an_upstream_rig(self) -> None:
        # The upstream rig: the rig stage of the joint scene's hand-eye part.
        handeye = vc.RigHandeyeSession()
        handeye.set_input(rig_handeye_input())
        handeye.step_intrinsics_init_all()
        cameras = handeye.step_intrinsics_optimize_all().cameras
        handeye.step_rig_init()
        rig = handeye.step_rig_optimize()
        joint = observation("synth_rig_handeye_laserline")
        payload = {
            "dataset": {
                "num_cameras": joint["num_cameras"],
                "views": [
                    {"cameras": v["obs"]["cameras"], "laser_pixels": v["obs"]["laser_pixels"]}
                    for v in joint["views"]
                ],
            },
            "upstream": vc.RigUpstreamCalibration(
                cameras=[c.to_payload() for c in cameras],
                cam_se3_rig=rig.cam_se3_rig,
                rig_se3_target=rig.rig_se3_target,
            ).to_payload(),
        }
        session = vc.RigLaserlineDeviceSession()
        session.set_input(payload)
        session.step_init()
        session.step_optimize()
        planes = session.export().laser_planes_cam
        self.assertEqual(len(planes), 2)
        # Seeded with its own optimum, the solve stays there.
        session.step_init_with_seed(rig_laserline_device.PlaneManualInit(planes_cam=planes))
        session.step_optimize(vc.PlaneOptimizeOptions(max_iters=30))
        for seeded, plane in zip(session.export().laser_planes_cam, planes):
            self.assertAlmostEqual(seeded.distance, plane.distance, places=6)


class RigHandeyeLaserlineSessionTest(unittest.TestCase):
    def test_run_matches_the_one_shot_run(self) -> None:
        payload = observation("synth_rig_handeye_laserline")
        session = vc.RigHandeyeLaserlineSession()
        session.set_input(payload)
        session.run_calibration()
        result = session.export()
        reference = _run_rig_handeye_laserline_raw(payload)
        self.assertEqual(result.cam_se3_rig, reference.cam_se3_rig)
        self.assertEqual(len(result.laser_planes_rig), 2)


if __name__ == "__main__":
    unittest.main()
