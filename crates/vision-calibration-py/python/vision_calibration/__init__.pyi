from typing import Any, Mapping

from .models import (
    BrownConradyDistortion,
    LaserlineDataset,
    LaserlineDeviceCalibrationConfig,
    LaserlineEstimate,
    LaserlineEstimateParams,
    LaserlineDeviceInitConfig,
    LaserlineDeviceOptimizeConfig,
    LaserlineDeviceResult,
    LaserlineDeviceSolverConfig,
    LaserlinePlane,
    LaserlineStats,
    LaserlineView,
    Observation,
    PinholeBrownConradyCamera,
    PinholeBrownConradyScheimpflugCamera,
    PinholeIntrinsics,
    PlanarCalibrationConfig,
    PlanarCalibrationResult,
    PlanarDataset,
    PlanarView,
    Pose,
    ScheimpflugIntrinsicsCalibrationConfig,
    ScheimpflugSensor,
    ScheimpflugIntrinsicsResult,
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

__version__: str


def robust_none() -> RobustLoss: ...
def robust_huber(scale: float) -> RobustLoss: ...
def robust_cauchy(scale: float) -> RobustLoss: ...
def robust_arctan(scale: float) -> RobustLoss: ...


def run_planar_intrinsics(
    input: PlanarDataset | Mapping[str, Any],
    config: PlanarCalibrationConfig | Mapping[str, Any] | None = None,
) -> PlanarCalibrationResult: ...


def run_scheimpflug_intrinsics(
    input: PlanarDataset | Mapping[str, Any],
    config: ScheimpflugIntrinsicsCalibrationConfig | Mapping[str, Any] | None = None,
) -> ScheimpflugIntrinsicsResult: ...


def run_single_cam_handeye(
    input: SingleCamHandeyeDataset | Mapping[str, Any],
    config: SingleCamHandeyeCalibrationConfig | Mapping[str, Any] | None = None,
) -> SingleCamHandeyeResult: ...


def run_rig_extrinsics(
    input: RigExtrinsicsDataset | Mapping[str, Any],
    config: RigExtrinsicsCalibrationConfig | Mapping[str, Any] | None = None,
) -> RigExtrinsicsResult: ...


def run_rig_handeye(
    input: RigHandeyeDataset | Mapping[str, Any],
    config: RigHandeyeCalibrationConfig | Mapping[str, Any] | None = None,
) -> RigHandeyeResult: ...


def run_laserline_device(
    input: LaserlineDataset | list[dict[str, Any]],
    config: LaserlineDeviceCalibrationConfig | Mapping[str, Any] | None = None,
) -> LaserlineDeviceResult: ...


__all__: list[str]
