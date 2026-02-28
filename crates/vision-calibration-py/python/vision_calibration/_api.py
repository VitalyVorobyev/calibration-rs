"""High-level typed Python wrappers around the Rust extension."""

from __future__ import annotations

from typing import cast

from . import _vision_calibration as _native
from .types import (
    LaserlineDeviceConfig,
    LaserlineDeviceExport,
    LaserlineDeviceInput,
    PlanarConfig,
    PlanarExport,
    PlanarInput,
    RigExtrinsicsConfig,
    RigExtrinsicsExport,
    RigExtrinsicsInput,
    RigHandeyeConfig,
    RigHandeyeExport,
    RigHandeyeInput,
    RobustLoss,
    SingleCamHandeyeConfig,
    SingleCamHandeyeExport,
    SingleCamHandeyeInput,
)


def robust_none() -> RobustLoss:
    """Build a serde-compatible "no robust loss" value.

    Returns
    -------
    RobustLoss
        Literal ``"None"`` accepted by Rust `RobustLoss`.
    """
    return "None"


def robust_huber(scale: float) -> RobustLoss:
    """Build a serde-compatible Huber robust loss payload.

    Parameters
    ----------
    scale:
        Positive scale parameter for the Huber kernel.
    """
    return {"Huber": {"scale": float(scale)}}


def robust_cauchy(scale: float) -> RobustLoss:
    """Build a serde-compatible Cauchy robust loss payload.

    Parameters
    ----------
    scale:
        Positive scale parameter for the Cauchy kernel.
    """
    return {"Cauchy": {"scale": float(scale)}}


def robust_arctan(scale: float) -> RobustLoss:
    """Build a serde-compatible Arctan robust loss payload.

    Parameters
    ----------
    scale:
        Positive scale parameter for the Arctan kernel.
    """
    return {"Arctan": {"scale": float(scale)}}


def run_planar_intrinsics(
    input: PlanarInput,
    config: PlanarConfig | None = None,
) -> PlanarExport:
    """Run planar intrinsics calibration.

    Parameters
    ----------
    input:
        Serde-compatible payload for Rust `PlanarDataset`.
    config:
        Optional serde-compatible payload for Rust `PlanarConfig`.

    Returns
    -------
    PlanarExport
        Dictionary containing refined camera parameters and summary metrics.
    """
    return cast(PlanarExport, _native.run_planar_intrinsics(input, config))


def run_single_cam_handeye(
    input: SingleCamHandeyeInput,
    config: SingleCamHandeyeConfig | None = None,
) -> SingleCamHandeyeExport:
    """Run single-camera hand-eye calibration.

    Returns
    -------
    SingleCamHandeyeExport
        Mode-explicit hand-eye transforms and reprojection statistics.
    """
    return cast(SingleCamHandeyeExport, _native.run_single_cam_handeye(input, config))


def run_rig_extrinsics(
    input: RigExtrinsicsInput,
    config: RigExtrinsicsConfig | None = None,
) -> RigExtrinsicsExport:
    """Run multi-camera rig extrinsics calibration.

    Returns
    -------
    RigExtrinsicsExport
        Per-camera intrinsics/extrinsics and reprojection statistics.
    """
    return cast(RigExtrinsicsExport, _native.run_rig_extrinsics(input, config))


def run_rig_handeye(
    input: RigHandeyeInput,
    config: RigHandeyeConfig | None = None,
) -> RigHandeyeExport:
    """Run multi-camera rig hand-eye calibration.

    Returns
    -------
    RigHandeyeExport
        Mode-explicit hand-eye transforms plus rig and reprojection outputs.
    """
    return cast(RigHandeyeExport, _native.run_rig_handeye(input, config))


def run_laserline_device(
    input: LaserlineDeviceInput,
    config: LaserlineDeviceConfig | None = None,
) -> LaserlineDeviceExport:
    """Run single-camera laserline-device calibration.

    Returns
    -------
    LaserlineDeviceExport
        Joint camera+laser estimate and residual statistics.
    """
    return cast(LaserlineDeviceExport, _native.run_laserline_device(input, config))
