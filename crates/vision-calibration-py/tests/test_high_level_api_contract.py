from __future__ import annotations

import unittest

import vision_calibration as vc


class HighLevelApiContractTest(unittest.TestCase):
    def test_dict_inputs_are_rejected_for_all_high_level_runners(self) -> None:
        cases = [
            (vc.run_planar_intrinsics, {"views": []}, vc.PlanarCalibrationConfig()),
            (
                vc.run_scheimpflug_intrinsics,
                {"views": []},
                vc.ScheimpflugIntrinsicsCalibrationConfig(),
            ),
            (
                vc.run_single_cam_handeye,
                {"views": []},
                vc.SingleCamHandeyeCalibrationConfig(),
            ),
            (
                vc.run_rig_extrinsics,
                {"num_cameras": 2, "views": []},
                vc.RigExtrinsicsCalibrationConfig(),
            ),
            (
                vc.run_rig_handeye,
                {"num_cameras": 2, "views": []},
                vc.RigHandeyeCalibrationConfig(),
            ),
            (
                vc.run_laserline_device,
                [],
                vc.LaserlineDeviceCalibrationConfig(),
            ),
        ]

        for fn, raw_input, cfg in cases:
            with self.subTest(runner=fn.__name__, mode="input"):
                with self.assertRaises(TypeError):
                    fn(raw_input, cfg)

    def test_dict_configs_are_rejected_for_all_high_level_runners(self) -> None:
        cases = [
            (vc.run_planar_intrinsics, vc.PlanarDataset(views=[])),
            (vc.run_scheimpflug_intrinsics, vc.PlanarDataset(views=[])),
            (vc.run_single_cam_handeye, vc.SingleCamHandeyeDataset(views=[])),
            (vc.run_rig_extrinsics, vc.RigExtrinsicsDataset(num_cameras=2, views=[])),
            (vc.run_rig_handeye, vc.RigHandeyeDataset(num_cameras=2, views=[])),
            (vc.run_laserline_device, vc.LaserlineDataset(views=[])),
        ]

        for fn, dataset in cases:
            with self.subTest(runner=fn.__name__, mode="config"):
                with self.assertRaises(TypeError):
                    fn(dataset, {"max_iters": 10})

    def test_typed_inputs_take_the_runtime_path(self) -> None:
        # Datasets are intentionally invalid (too few views), so the expected
        # failure mode is RuntimeError from Rust validation, not TypeError.
        cases = [
            (vc.run_planar_intrinsics, vc.PlanarDataset(views=[]), vc.PlanarCalibrationConfig()),
            (
                vc.run_scheimpflug_intrinsics,
                vc.PlanarDataset(views=[]),
                vc.ScheimpflugIntrinsicsCalibrationConfig(),
            ),
            (
                vc.run_single_cam_handeye,
                vc.SingleCamHandeyeDataset(views=[]),
                vc.SingleCamHandeyeCalibrationConfig(),
            ),
            (
                vc.run_rig_extrinsics,
                vc.RigExtrinsicsDataset(num_cameras=2, views=[]),
                vc.RigExtrinsicsCalibrationConfig(),
            ),
            (
                vc.run_rig_handeye,
                vc.RigHandeyeDataset(num_cameras=2, views=[]),
                vc.RigHandeyeCalibrationConfig(),
            ),
            (
                vc.run_laserline_device,
                vc.LaserlineDataset(views=[]),
                vc.LaserlineDeviceCalibrationConfig(),
            ),
        ]

        for fn, dataset, cfg in cases:
            with self.subTest(runner=fn.__name__):
                with self.assertRaises(RuntimeError):
                    fn(dataset, cfg)


if __name__ == "__main__":
    unittest.main()
