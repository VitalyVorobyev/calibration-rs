from typing import Any, Mapping, Sequence

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
    IntrinsicsOptimizeConfig,
    PlanarOptimizeConfig,
    SingleCamHandeyeOptimizeConfig,
    PlaneOptimizeConfig,
    LaserlineDeviceInput,
    LaserlineDeviceConfig,
    DeviceOptimizeConfig,
    LaserlineDeviceExport,
    LaserPlane,
    LaserlineStats,
    LaserlineView,
    NoDistortion,
    Observation,
    PointBlur,
    PointCovariance,
    CameraParams,
    EquidistantProjection,
    IdentitySensor,
    KannalaBrandt4Distortion,
    PinholeIntrinsics,
    PinholeProjection,
    Projection,
    Rational8Distortion,
    Sensor,
    ThinPrism9Distortion,
    PlanarIntrinsicsConfig,
    ParameterCovariance,
    SecondOrderError,
    CombinationTerm,
    ProjectionUncertainty,
    Measurement,
    QualityCheck,
    QualityReport,
    QualityThresholds,
    CandidateGrid,
    PredictedRigUncertainty,
    PredictedUncertainty,
    RankedCandidate,
    RankedRobotPose,
    PlanarIntrinsicsExport,
    PlanarIntrinsicsInput,
    PlanarView,
    Pose,
    RigConfig,
    RobotPoseConfig,
    RobotPosePrior,
    RobotPoseSigma,
    RobotRefinement,
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
    RigEyeInHand,
    RigEyeToHand,
    HandeyeSolution,
    CameraEyeInHand,
    CameraEyeToHand,
    SingleCamHandeyeSolution,
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
from ._generated_types import FocusDiagnostic as _FocusDiagnostic
from ._generated_types import ModelComparison as _ModelComparison
from ._generated_types import RobustLoss as _RobustLoss
from .types import ProgressCallback as _ProgressCallback
from . import common as common
from . import dataset as dataset
from . import dataset_runner as dataset_runner
from . import detect as detect
from . import device_seed as device_seed
from . import laserline_device as laserline_device
from . import maps as maps
from . import mvg as mvg
from . import planar_intrinsics as planar_intrinsics
from . import project as project
from . import residuals as residuals
from . import rig_extrinsics as rig_extrinsics
from . import rig_handeye as rig_handeye
from . import rig_handeye_laserline as rig_handeye_laserline
from . import rig_laserline_device as rig_laserline_device
from . import samples as samples
from . import session as session
from . import single_cam_handeye as single_cam_handeye
from .camera import CameraModel as CameraModel
from .maps import (
    DisplacementPlane as DisplacementPlane,
    DistortionMap as DistortionMap,
    ImageGrid as ImageGrid,
    ProjectionDifference as ProjectionDifference,
    ResidualMap as ResidualMap,
)
from .common import (
    HandeyeInitOptions as HandeyeInitOptions,
    HandeyeOptimizeOptions as HandeyeOptimizeOptions,
    HandeyeOptimizeResult as HandeyeOptimizeResult,
    IntrinsicsInitOptions as IntrinsicsInitOptions,
    IntrinsicsManualInit as IntrinsicsManualInit,
    IntrinsicsOptimizeOptions as IntrinsicsOptimizeOptions,
    RigInitOptions as RigInitOptions,
    RigInitResult as RigInitResult,
    RigIntrinsicsInitResult as RigIntrinsicsInitResult,
    RigIntrinsicsManualInit as RigIntrinsicsManualInit,
    RigManualInit as RigManualInit,
    RigOptimizeOptions as RigOptimizeOptions,
    RigPoses as RigPoses,
)
from .dataset import SniffError as SniffError, sniff_folder as sniff_folder
from .dataset_runner import (
    AskUserError as AskUserError,
    RunError as RunError,
    calibrate_dataset as calibrate_dataset,
    detect_target as detect_target,
)
from .device_seed import DeviceSeed as DeviceSeed
from .detect import (
    CacheKey as CacheKey,
    DetectionDetail as DetectionDetail,
    Features as Features,
    FsDetectionCache as FsDetectionCache,
    reject_ambiguous_detection as reject_ambiguous_detection,
)
from .laserline_device import (
    DeviceInitOptions as DeviceInitOptions,
    DeviceOptimizeOptions as DeviceOptimizeOptions,
    LaserlineDeviceSession as LaserlineDeviceSession,
)
from .planar_intrinsics import (
    IntrinsicsFilterOptions as IntrinsicsFilterOptions,
    PlanarIntrinsicsSession as PlanarIntrinsicsSession,
)
from .residuals import (
    FeatureResidualHistogram as FeatureResidualHistogram,
    LaserFeatureResiduals as LaserFeatureResiduals,
    LaserLineTrace as LaserLineTrace,
    PerFeatureResiduals as PerFeatureResiduals,
    TargetFeatureResiduals as TargetFeatureResiduals,
)
from .rig_extrinsics import RigExtrinsicsSession as RigExtrinsicsSession
from .rig_handeye import RigHandeyeSession as RigHandeyeSession
from .rig_handeye_laserline import RigHandeyeLaserlineSession as RigHandeyeLaserlineSession
from .rig_laserline_device import (
    PlaneInitOptions as PlaneInitOptions,
    PlaneOptimizeOptions as PlaneOptimizeOptions,
    RigLaserlineDeviceSession as RigLaserlineDeviceSession,
)
from .session import (
    CalibrationSession as CalibrationSession,
    ExportRecord as ExportRecord,
    LogEntry as LogEntry,
    SessionMetadata as SessionMetadata,
)
from .single_cam_handeye import SingleCamHandeyeSession as SingleCamHandeyeSession

__version__: str


def robust_none() -> _RobustLoss: ...
def robust_huber(scale: float) -> _RobustLoss: ...
def robust_cauchy(scale: float) -> _RobustLoss: ...
def robust_arctan(scale: float) -> _RobustLoss: ...


def rig_upstream_calibration(
    export: RigHandeyeExport, rig_se3_target: Sequence[Pose]
) -> RigUpstreamCalibration: ...



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


def html_report(
    result: PlanarIntrinsicsExport
    | SingleCamHandeyeExport
    | LaserlineDeviceExport
    | RigExtrinsicsExport
    | RigHandeyeExport
    | RigLaserlineDeviceExport
    | RigHandeyeLaserlineExport
    | Mapping[str, Any],
    image_sizes: list[tuple[int, int]] | None = None,
    *,
    title: str | None = None,
    camera_names: Sequence[str] | None = None,
    thresholds: QualityThresholds | None = None,
) -> str: ...


def predict_with_views(
    result: PlanarIntrinsicsExport | Mapping[str, Any],
    candidates: Sequence[Pose | Mapping[str, Any]],
    target_points: Sequence[Sequence[float]],
    image_size: tuple[int, int],
) -> PredictedUncertainty: ...


def rank_candidates(
    result: PlanarIntrinsicsExport | Mapping[str, Any],
    candidates: Sequence[Pose | Mapping[str, Any]],
    target_points: Sequence[Sequence[float]],
    image_size: tuple[int, int],
) -> list[RankedCandidate]: ...


def predict_with_robot_poses(
    result: RigHandeyeExport | Mapping[str, Any],
    candidates: Sequence[Pose | Mapping[str, Any]],
    target_points: Sequence[Sequence[float]],
    image_sizes: Sequence[tuple[int, int]],
) -> PredictedRigUncertainty: ...


def rank_robot_poses(
    result: RigHandeyeExport | Mapping[str, Any],
    candidates: Sequence[Pose | Mapping[str, Any]],
    target_points: Sequence[Sequence[float]],
    image_sizes: Sequence[tuple[int, int]],
) -> list[RankedRobotPose]: ...


def candidate_poses(
    camera: CameraParams | Mapping[str, Any],
    target_points: Sequence[Sequence[float]],
    image_size: tuple[int, int],
    grid: CandidateGrid,
) -> list[Pose]: ...


def compare_camera_models(
    input: PlanarIntrinsicsInput | Mapping[str, Any],
    alternatives: Sequence[CameraConfig | Mapping[str, Any]],
    config: PlanarIntrinsicsConfig | Mapping[str, Any] | None = None,
    folds: int = 5,
    seed: int = 0,
    progress: _ProgressCallback | None = None,
) -> _ModelComparison: ...


def fit_focus(
    input: PlanarIntrinsicsInput | Mapping[str, Any],
    camera: CameraParams | Mapping[str, Any],
    cam_se3_target: Sequence[Pose | Mapping[str, Any]],
    blur_limit_px: float | None = None,
    view_jackknife: bool = False,
) -> _FocusDiagnostic: ...


def distortion_map(
    camera: CameraParams | Mapping[str, Any],
    image_size: tuple[int, int],
    columns: int = 256,
) -> DistortionMap: ...


def projection_difference(
    a: CameraParams | Mapping[str, Any],
    b: CameraParams | Mapping[str, Any],
    image_size: tuple[int, int],
    columns: int = 64,
    compensation: str = "rotation",
    fit_points: Sequence[Sequence[float]] | None = None,
) -> ProjectionDifference: ...


def residual_map(
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
    columns: int = 64,
    bandwidth: str | float = "auto",
    views: Sequence[int] | None = None,
    outlier_factor: float = 8.0,
    min_effective_samples: float = 8.0,
    min_effective_views: float = 5.0,
) -> ResidualMap: ...


__all__: list[str]
