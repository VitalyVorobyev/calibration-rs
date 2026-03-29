import numpy as np

from . import geometry as geometry
from . import mvg as mvg
from .geometry import CameraMatrixDecomposition as CameraMatrixDecomposition
from .models import (
    BrownConradyDistortion as BrownConradyDistortion,
    DistortionFixMask as DistortionFixMask,
    IntrinsicsFixMask as IntrinsicsFixMask,
    LaserlineDataset as LaserlineDataset,
    LaserlineDeviceCalibrationConfig as LaserlineDeviceCalibrationConfig,
    LaserlineEstimate as LaserlineEstimate,
    LaserlineEstimateParams as LaserlineEstimateParams,
    LaserlineDeviceInitConfig as LaserlineDeviceInitConfig,
    LaserlineDeviceOptimizeConfig as LaserlineDeviceOptimizeConfig,
    LaserlineDeviceResult as LaserlineDeviceResult,
    LaserlineDeviceSolverConfig as LaserlineDeviceSolverConfig,
    LaserlinePlane as LaserlinePlane,
    LaserlineStats as LaserlineStats,
    LaserlineView as LaserlineView,
    Observation as Observation,
    PinholeBrownConradyCamera as PinholeBrownConradyCamera,
    PinholeBrownConradyScheimpflugCamera as PinholeBrownConradyScheimpflugCamera,
    PinholeIntrinsics as PinholeIntrinsics,
    PlanarCalibrationConfig as PlanarCalibrationConfig,
    PlanarCalibrationResult as PlanarCalibrationResult,
    PlanarDataset as PlanarDataset,
    PlanarView as PlanarView,
    Pose as Pose,
    RansacOptions as RansacOptions,
    RobustLoss as RobustLoss,
    RobustLossArctan as RobustLossArctan,
    RobustLossCauchy as RobustLossCauchy,
    RobustLossHuber as RobustLossHuber,
    RobustLossNone as RobustLossNone,
    ScheimpflugFixMask as ScheimpflugFixMask,
    ScheimpflugIntrinsicsCalibrationConfig as ScheimpflugIntrinsicsCalibrationConfig,
    ScheimpflugSensor as ScheimpflugSensor,
    ScheimpflugSensorInit as ScheimpflugSensorInit,
    ScheimpflugIntrinsicsResult as ScheimpflugIntrinsicsResult,
    RigExtrinsicsCalibrationConfig as RigExtrinsicsCalibrationConfig,
    RigExtrinsicsDataset as RigExtrinsicsDataset,
    RigExtrinsicsResult as RigExtrinsicsResult,
    RigExtrinsicsView as RigExtrinsicsView,
    RigHandeyeBaConfig as RigHandeyeBaConfig,
    RigHandeyeCalibrationConfig as RigHandeyeCalibrationConfig,
    RigHandeyeDataset as RigHandeyeDataset,
    RigHandeyeInitConfig as RigHandeyeInitConfig,
    RigHandeyeIntrinsicsConfig as RigHandeyeIntrinsicsConfig,
    RigHandeyeResult as RigHandeyeResult,
    RigHandeyeRigConfig as RigHandeyeRigConfig,
    RigHandeyeSolverConfig as RigHandeyeSolverConfig,
    RigHandeyeView as RigHandeyeView,
    SingleCamHandeyeCalibrationConfig as SingleCamHandeyeCalibrationConfig,
    SingleCamHandeyeDataset as SingleCamHandeyeDataset,
    SingleCamHandeyeResult as SingleCamHandeyeResult,
    SingleCamHandeyeView as SingleCamHandeyeView,
)
from .mvg import (
    EssentialEstimate as EssentialEstimate,
    HomographyDecomposition as HomographyDecomposition,
    HomographyEstimate as HomographyEstimate,
    RelativePose as RelativePose,
    RobustRelativePose as RobustRelativePose,
    SceneDiagnostics as SceneDiagnostics,
    TriangulatedPoint as TriangulatedPoint,
)

__version__: str


def robust_none() -> RobustLossNone: ...
def robust_huber(scale: float) -> RobustLossHuber: ...
def robust_cauchy(scale: float) -> RobustLossCauchy: ...
def robust_arctan(scale: float) -> RobustLossArctan: ...


def run_planar_intrinsics(
    input: PlanarDataset,
    config: PlanarCalibrationConfig | None = None,
) -> PlanarCalibrationResult: ...


def run_scheimpflug_intrinsics(
    input: PlanarDataset,
    config: ScheimpflugIntrinsicsCalibrationConfig | None = None,
) -> ScheimpflugIntrinsicsResult: ...


def run_single_cam_handeye(
    input: SingleCamHandeyeDataset,
    config: SingleCamHandeyeCalibrationConfig | None = None,
) -> SingleCamHandeyeResult: ...


def run_rig_extrinsics(
    input: RigExtrinsicsDataset,
    config: RigExtrinsicsCalibrationConfig | None = None,
) -> RigExtrinsicsResult: ...


def run_rig_handeye(
    input: RigHandeyeDataset,
    config: RigHandeyeCalibrationConfig | None = None,
) -> RigHandeyeResult: ...


def run_laserline_device(
    input: LaserlineDataset,
    config: LaserlineDeviceCalibrationConfig | None = None,
) -> LaserlineDeviceResult: ...


__all__: list[str]
