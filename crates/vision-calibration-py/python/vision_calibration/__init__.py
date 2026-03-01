"""Python bindings for calibration-rs.

High-level workflow functions accept Python-native dataclasses from
``vision_calibration.models`` and return result dataclasses with docstrings.
Raw serde-style mappings are still accepted for advanced use.
"""

from __future__ import annotations

from . import _vision_calibration as _native
from ._api import (
    robust_arctan,
    robust_cauchy,
    robust_huber,
    robust_none,
    run_laserline_device,
    run_planar_intrinsics,
    run_rig_extrinsics,
    run_rig_handeye,
    run_single_cam_handeye,
)
from .models import (
    LaserlineDataset,
    LaserlineDeviceCalibrationConfig,
    LaserlineDeviceInitConfig,
    LaserlineDeviceOptimizeConfig,
    LaserlineDeviceResult,
    LaserlineDeviceSolverConfig,
    LaserlineView,
    Observation,
    PlanarCalibrationConfig,
    PlanarCalibrationResult,
    PlanarDataset,
    PlanarView,
    Pose,
    RigExtrinsicsCalibrationConfig,
    RigExtrinsicsDataset,
    RigExtrinsicsResult,
    RigExtrinsicsView,
    RigHandeyeBaConfig,
    RigHandeyeCalibrationConfig,
    RigHandeyeDataset,
    RigHandeyeInitConfig,
    RigHandeyeIntrinsicsConfig,
    RigHandeyeResult,
    RigHandeyeRigConfig,
    RigHandeyeSolverConfig,
    RigHandeyeView,
    SingleCamHandeyeCalibrationConfig,
    SingleCamHandeyeDataset,
    SingleCamHandeyeResult,
    SingleCamHandeyeView,
)
from .types import (
    HandEyeMode,
    LaserlineResidualType,
    RobustLoss,
)

__version__ = _native.library_version()

__all__ = [
    "__version__",
    "Pose",
    "Observation",
    "PlanarView",
    "PlanarDataset",
    "SingleCamHandeyeView",
    "SingleCamHandeyeDataset",
    "RigExtrinsicsView",
    "RigExtrinsicsDataset",
    "RigHandeyeView",
    "RigHandeyeDataset",
    "LaserlineView",
    "LaserlineDataset",
    "PlanarCalibrationConfig",
    "SingleCamHandeyeCalibrationConfig",
    "RigExtrinsicsCalibrationConfig",
    "RigHandeyeIntrinsicsConfig",
    "RigHandeyeRigConfig",
    "RigHandeyeInitConfig",
    "RigHandeyeSolverConfig",
    "RigHandeyeBaConfig",
    "RigHandeyeCalibrationConfig",
    "LaserlineDeviceInitConfig",
    "LaserlineDeviceSolverConfig",
    "LaserlineDeviceOptimizeConfig",
    "LaserlineDeviceCalibrationConfig",
    "PlanarCalibrationResult",
    "SingleCamHandeyeResult",
    "RigExtrinsicsResult",
    "RigHandeyeResult",
    "LaserlineDeviceResult",
    "HandEyeMode",
    "LaserlineResidualType",
    "RobustLoss",
    "robust_none",
    "robust_huber",
    "robust_cauchy",
    "robust_arctan",
    "run_planar_intrinsics",
    "run_single_cam_handeye",
    "run_rig_extrinsics",
    "run_rig_handeye",
    "run_laserline_device",
]
