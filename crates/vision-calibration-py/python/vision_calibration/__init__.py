"""Python bindings for calibration-rs.

This package exposes native Rust calibration pipelines through a typed Python API.
All high-level workflow calls return plain Python dictionaries.
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
from .types import (
    HandEyeMode,
    LaserlineResidualType,
    LaserlineDeviceConfig,
    LaserlineDeviceInitConfig,
    LaserlineDeviceOptimizeConfig,
    LaserlineDeviceSolverConfig,
    LaserlineDeviceExport,
    LaserlineDeviceInput,
    PlanarConfig,
    PlanarExport,
    PlanarInput,
    RigExtrinsicsConfig,
    RigExtrinsicsExport,
    RigExtrinsicsInput,
    RigHandeyeBaConfig,
    RigHandeyeConfig,
    RigHandeyeExport,
    RigHandeyeInitConfig,
    RigHandeyeInput,
    RigHandeyeIntrinsicsConfig,
    RigHandeyeRigConfig,
    RigHandeyeSolverConfig,
    RobustLoss,
    SingleCamHandeyeConfig,
    SingleCamHandeyeExport,
    SingleCamHandeyeInput,
)

__version__ = _native.library_version()

__all__ = [
    "__version__",
    "HandEyeMode",
    "LaserlineResidualType",
    "RobustLoss",
    "PlanarInput",
    "PlanarConfig",
    "PlanarExport",
    "SingleCamHandeyeInput",
    "SingleCamHandeyeConfig",
    "SingleCamHandeyeExport",
    "RigExtrinsicsInput",
    "RigExtrinsicsConfig",
    "RigExtrinsicsExport",
    "RigHandeyeInput",
    "RigHandeyeConfig",
    "RigHandeyeIntrinsicsConfig",
    "RigHandeyeRigConfig",
    "RigHandeyeInitConfig",
    "RigHandeyeSolverConfig",
    "RigHandeyeBaConfig",
    "RigHandeyeExport",
    "LaserlineDeviceInput",
    "LaserlineDeviceConfig",
    "LaserlineDeviceInitConfig",
    "LaserlineDeviceSolverConfig",
    "LaserlineDeviceOptimizeConfig",
    "LaserlineDeviceExport",
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
