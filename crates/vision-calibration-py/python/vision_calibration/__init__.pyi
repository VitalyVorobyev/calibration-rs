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

__version__: str


def robust_none() -> RobustLoss: ...
def robust_huber(scale: float) -> RobustLoss: ...
def robust_cauchy(scale: float) -> RobustLoss: ...
def robust_arctan(scale: float) -> RobustLoss: ...


def run_planar_intrinsics(
    input: PlanarInput,
    config: PlanarConfig | None = None,
) -> PlanarExport: ...


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


__all__: list[str]
