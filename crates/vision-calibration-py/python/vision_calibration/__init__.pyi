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
    RigLaserlineDataset,
    RigLaserlineDeviceCalibrationConfig,
    RigLaserlineDeviceInput,
    RigLaserlineDeviceResult,
    RigLaserlineUpstreamCalibration,
    RigLaserlineView,
    SingleCamHandeyeCalibrationConfig,
    SingleCamHandeyeDataset,
    SingleCamHandeyeResult,
    SingleCamHandeyeView,
)
from .types import RobustLoss as _RobustLoss

__version__: str


def library_version() -> str: ...


def robust_none() -> _RobustLoss: ...
def robust_huber(scale: float) -> _RobustLoss: ...
def robust_cauchy(scale: float) -> _RobustLoss: ...
def robust_arctan(scale: float) -> _RobustLoss: ...


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


def run_rig_laserline_device(
    input: RigLaserlineDeviceInput,
    config: RigLaserlineDeviceCalibrationConfig | None = None,
) -> RigLaserlineDeviceResult: ...


def pixel_to_gripper_point(
    cam_idx: int,
    pixel: tuple[float, float] | list[float],
    rig_cal: RigHandeyeResult,
    laser_planes_rig: list[LaserlinePlane],
    base_se3_gripper: Pose | None = None,
) -> tuple[float, float, float]: ...


__all__: list[str]
