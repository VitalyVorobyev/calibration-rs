from typing import Any, Mapping

from .models import (
    BrownConradyDistortion,
    CameraConfig,
    CameraFix,
    CameraModelSpec,
    CameraOverride,
    Distortion,
    Division1Distortion,
    HandeyeInitConfig,
    IntrinsicsInitConfig,
    LaserlineDeviceInput,
    LaserlineDeviceConfig,
    DeviceOptimizeConfig,
    LaserlineEstimate,
    LaserlineParams,
    LaserlineDeviceExport,
    LaserPlane,
    LaserlineStats,
    LaserlineView,
    NoDistortion,
    Observation,
    PinholeCamera,
    PinholeIntrinsics,
    PinholeScheimpflugCamera,
    Rational8Distortion,
    ThinPrism9Distortion,
    PlanarIntrinsicsConfig,
    ParameterUncertainty,
    ProjectionUncertainty,
    Measurement,
    QualityCheck,
    QualityReport,
    QualityThresholds,
    PlanarIntrinsicsExport,
    PlanarIntrinsicsInput,
    PlanarView,
    Pose,
    RigConfig,
    RobotPoseConfig,
    ScheimpflugSensor,
    SolverConfig,
    RigExtrinsicsConfig,
    RigExtrinsicsInput,
    RigExtrinsicsExport,
    RigExtrinsicsView,
    ResidualNoise,
    ViewCovariance,
    HandeyeOptimizeConfig,
    RigHandeyeConfig,
    RigHandeyeInput,
    JointOptimizeConfig,
    RigHandeyeLaserlineConfig,
    RigHandeyeLaserlineInput,
    RigHandeyeLaserlinePerCamStats,
    RigHandeyeLaserlineExport,
    RigHandeyeLaserlineView,
    RigHandeyeExport,
    RigHandeyeView,
    RigLaserlineDataset,
    RigLaserlineDeviceConfig,
    RigLaserlineDeviceInput,
    RigLaserlineDeviceExport,
    RigUpstreamCalibration,
    RigLaserlineView,
    SingleCamHandeyeConfig,
    SingleCamHandeyeInput,
    SingleCamHandeyeExport,
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
    input: PlanarIntrinsicsInput,
    config: PlanarIntrinsicsConfig | None = None,
) -> PlanarIntrinsicsExport: ...


def run_single_cam_handeye(
    input: SingleCamHandeyeInput,
    config: SingleCamHandeyeConfig | None = None,
) -> SingleCamHandeyeExport: ...


def run_rig_extrinsics(
    input: RigExtrinsicsInput,
    config: RigExtrinsicsConfig | None = None,
) -> RigExtrinsicsExport: ...


def run_rig_handeye(
    input: RigHandeyeInput,
    config: RigHandeyeConfig | None = None,
) -> RigHandeyeExport: ...


def run_laserline_device(
    input: LaserlineDeviceInput,
    config: LaserlineDeviceConfig | None = None,
) -> LaserlineDeviceExport: ...


def run_rig_laserline_device(
    input: RigLaserlineDeviceInput,
    config: RigLaserlineDeviceConfig | None = None,
) -> RigLaserlineDeviceExport: ...


def run_rig_handeye_laserline(
    input: RigHandeyeLaserlineInput,
    config: RigHandeyeLaserlineConfig | None = None,
) -> RigHandeyeLaserlineExport: ...


def pixel_to_gripper_point(
    cam_idx: int,
    pixel: tuple[float, float] | list[float],
    rig_cal: RigHandeyeExport,
    laser_planes_rig: list[LaserPlane],
    base_se3_gripper: Pose | None = None,
) -> tuple[float, float, float]: ...



def projection_uncertainty(
    result: PlanarIntrinsicsExport
    | SingleCamHandeyeExport
    | LaserlineDeviceExport
    | RigExtrinsicsExport
    | RigHandeyeExport
    | RigLaserlineDeviceExport
    | RigHandeyeLaserlineExport
    | Mapping[str, Any],
    image_size: tuple[int, int],
    camera: int = 0,
) -> ProjectionUncertainty: ...


def assess(
    result: PlanarIntrinsicsExport
    | SingleCamHandeyeExport
    | LaserlineDeviceExport
    | RigExtrinsicsExport
    | RigHandeyeExport
    | RigLaserlineDeviceExport
    | RigHandeyeLaserlineExport
    | Mapping[str, Any],
    thresholds: QualityThresholds | None = None,
    image_sizes: list[tuple[int, int]] | None = None,
) -> QualityReport: ...


__all__: list[str]
