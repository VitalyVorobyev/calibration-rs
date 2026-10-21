from __future__ import annotations

import unittest

import vision_calibration as vc


class HighLevelApiContractTest(unittest.TestCase):
    def test_dict_inputs_are_rejected_for_all_high_level_runners(self) -> None:
        cases = [
            (vc.run_planar_intrinsics, {"views": []}, vc.PlanarIntrinsicsConfig()),
            (
                vc.run_single_cam_handeye,
                {"views": []},
                vc.SingleCamHandeyeConfig(),
            ),
            (
                vc.run_rig_extrinsics,
                {"num_cameras": 2, "views": []},
                vc.RigExtrinsicsConfig(),
            ),
            (
                vc.run_rig_handeye,
                {"num_cameras": 2, "views": []},
                vc.RigHandeyeConfig(),
            ),
            (
                vc.run_laserline_device,
                [],
                vc.LaserlineDeviceConfig(),
            ),
        ]

        for fn, raw_input, cfg in cases:
            with self.subTest(runner=fn.__name__, mode="input"):
                with self.assertRaises(TypeError):
                    fn(raw_input, cfg)

    def test_dict_configs_are_rejected_for_all_high_level_runners(self) -> None:
        cases = [
            (vc.run_planar_intrinsics, vc.PlanarIntrinsicsInput(views=[])),
            (vc.run_single_cam_handeye, vc.SingleCamHandeyeInput(views=[])),
            (vc.run_rig_extrinsics, vc.RigExtrinsicsInput(num_cameras=2, views=[])),
            (vc.run_rig_handeye, vc.RigHandeyeInput(num_cameras=2, views=[])),
            (vc.run_laserline_device, vc.LaserlineDeviceInput(views=[])),
        ]

        for fn, dataset in cases:
            with self.subTest(runner=fn.__name__, mode="config"):
                with self.assertRaises(TypeError):
                    fn(dataset, {"max_iters": 10})

    def test_typed_inputs_take_the_runtime_path(self) -> None:
        # Datasets are intentionally invalid (too few views), so the expected
        # failure mode is ValueError from Rust input validation, not TypeError.
        # Per R-07, invalid input/config from Python surfaces as ValueError;
        # TypeError is reserved for wrong Python types (dicts, lists, ...).
        cases = [
            (vc.run_planar_intrinsics, vc.PlanarIntrinsicsInput(views=[]), vc.PlanarIntrinsicsConfig()),
            (
                vc.run_single_cam_handeye,
                vc.SingleCamHandeyeInput(views=[]),
                vc.SingleCamHandeyeConfig(),
            ),
            (
                vc.run_rig_extrinsics,
                vc.RigExtrinsicsInput(num_cameras=2, views=[]),
                vc.RigExtrinsicsConfig(),
            ),
            (
                vc.run_rig_handeye,
                vc.RigHandeyeInput(num_cameras=2, views=[]),
                vc.RigHandeyeConfig(),
            ),
            (
                vc.run_laserline_device,
                vc.LaserlineDeviceInput(views=[]),
                vc.LaserlineDeviceConfig(),
            ),
        ]

        for fn, dataset, cfg in cases:
            with self.subTest(runner=fn.__name__):
                with self.assertRaises(ValueError):
                    fn(dataset, cfg)


if __name__ == "__main__":
    unittest.main()
