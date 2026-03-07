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
