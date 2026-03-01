"""High-level Python wrappers around the Rust extension.

The public API accepts Python dataclasses from :mod:`vision_calibration.models`
and returns result dataclasses. Raw serde mappings are still accepted for
advanced/interop use.
"""

from __future__ import annotations

from typing import Any, Mapping, cast

from . import _vision_calibration as _native
from .models import (
    LaserlineDataset,
    LaserlineDeviceCalibrationConfig,
    LaserlineDeviceResult,
    PlanarCalibrationConfig,
    PlanarCalibrationResult,
    PlanarDataset,
    RigExtrinsicsCalibrationConfig,
    RigExtrinsicsDataset,
    RigExtrinsicsResult,
    RigHandeyeCalibrationConfig,
    RigHandeyeDataset,
    RigHandeyeResult,
    SingleCamHandeyeCalibrationConfig,
    SingleCamHandeyeDataset,
    SingleCamHandeyeResult,
    normalize_input_payload,
)
from .types import (
    RobustLoss,
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


def _normalize_planar_config(
    config: PlanarCalibrationConfig | Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if config is None:
        return None
    if isinstance(config, PlanarCalibrationConfig):
        return config.to_payload()
    return PlanarCalibrationConfig.from_mapping(config).to_payload()


def _normalize_single_handeye_config(
    config: SingleCamHandeyeCalibrationConfig | Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if config is None:
        return None
    if isinstance(config, SingleCamHandeyeCalibrationConfig):
        return config.to_payload()
    return SingleCamHandeyeCalibrationConfig.from_mapping(config).to_payload()


def _normalize_rig_extrinsics_config(
    config: RigExtrinsicsCalibrationConfig | Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if config is None:
        return None
    if isinstance(config, RigExtrinsicsCalibrationConfig):
        return config.to_payload()
    return RigExtrinsicsCalibrationConfig.from_mapping(config).to_payload()


def _normalize_rig_handeye_config(
    config: RigHandeyeCalibrationConfig | Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if config is None:
        return None
    if isinstance(config, RigHandeyeCalibrationConfig):
        return config.to_payload()
    return RigHandeyeCalibrationConfig.from_mapping(config).to_payload()


def _normalize_laserline_config(
    config: LaserlineDeviceCalibrationConfig | Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if config is None:
        return None
    if isinstance(config, LaserlineDeviceCalibrationConfig):
        return config.to_payload()
    return LaserlineDeviceCalibrationConfig.from_mapping(config).to_payload()


def run_planar_intrinsics(
    input: PlanarDataset | Mapping[str, Any],
    config: PlanarCalibrationConfig | Mapping[str, Any] | None = None,
) -> PlanarCalibrationResult:
    """Run planar intrinsics calibration.

    Parameters
    ----------
    input:
        Planar dataset.
        Preferred type: :class:`vision_calibration.models.PlanarDataset`.
        Raw serde mapping is also accepted.
    config:
        Calibration config.
        Preferred type: :class:`vision_calibration.models.PlanarCalibrationConfig`.
        If omitted, Rust defaults are used.
        Raw serde mapping is also accepted.

    Returns
    -------
    PlanarCalibrationResult
        Python result object with `camera`, `camera_se3_target`, cost, and
        reprojection metrics.
    """
    payload = normalize_input_payload(input)
    cfg = _normalize_planar_config(config)
    raw = cast(dict[str, Any], _native.run_planar_intrinsics(payload, cfg))
    return PlanarCalibrationResult.from_payload(raw)


def run_single_cam_handeye(
    input: SingleCamHandeyeDataset | Mapping[str, Any],
    config: SingleCamHandeyeCalibrationConfig | Mapping[str, Any] | None = None,
) -> SingleCamHandeyeResult:
    """Run single-camera hand-eye calibration.

    Parameters
    ----------
    input:
        Single-camera hand-eye dataset.
        Preferred type: :class:`vision_calibration.models.SingleCamHandeyeDataset`.
        Raw serde mapping is also accepted.
    config:
        Calibration config.
        Preferred type: :class:`vision_calibration.models.SingleCamHandeyeCalibrationConfig`.
        If omitted, Rust defaults are used.

    Returns
    -------
    SingleCamHandeyeResult
        Mode-explicit hand-eye transforms and reprojection statistics.
    """
    payload = normalize_input_payload(input)
    cfg = _normalize_single_handeye_config(config)
    raw = cast(dict[str, Any], _native.run_single_cam_handeye(payload, cfg))
    return SingleCamHandeyeResult.from_payload(raw)


def run_rig_extrinsics(
    input: RigExtrinsicsDataset | Mapping[str, Any],
    config: RigExtrinsicsCalibrationConfig | Mapping[str, Any] | None = None,
) -> RigExtrinsicsResult:
    """Run multi-camera rig extrinsics calibration.

    Parameters
    ----------
    input:
        Rig extrinsics dataset.
        Preferred type: :class:`vision_calibration.models.RigExtrinsicsDataset`.
        Raw serde mapping is also accepted.
    config:
        Calibration config.
        Preferred type: :class:`vision_calibration.models.RigExtrinsicsCalibrationConfig`.
        If omitted, Rust defaults are used.

    Returns
    -------
    RigExtrinsicsResult
        Per-camera intrinsics/extrinsics and reprojection statistics.
    """
    payload = normalize_input_payload(input)
    cfg = _normalize_rig_extrinsics_config(config)
    raw = cast(dict[str, Any], _native.run_rig_extrinsics(payload, cfg))
    return RigExtrinsicsResult.from_payload(raw)


def run_rig_handeye(
    input: RigHandeyeDataset | Mapping[str, Any],
    config: RigHandeyeCalibrationConfig | Mapping[str, Any] | None = None,
) -> RigHandeyeResult:
    """Run multi-camera rig hand-eye calibration.

    Parameters
    ----------
    input:
        Rig hand-eye dataset.
        Preferred type: :class:`vision_calibration.models.RigHandeyeDataset`.
        Raw serde mapping is also accepted.
    config:
        Calibration config.
        Preferred type: :class:`vision_calibration.models.RigHandeyeCalibrationConfig`.
        If omitted, Rust defaults are used.

    Returns
    -------
    RigHandeyeResult
        Mode-explicit hand-eye transforms plus rig and reprojection outputs.
    """
    payload = normalize_input_payload(input)
    cfg = _normalize_rig_handeye_config(config)
    raw = cast(dict[str, Any], _native.run_rig_handeye(payload, cfg))
    return RigHandeyeResult.from_payload(raw)


def run_laserline_device(
    input: LaserlineDataset | list[dict[str, Any]],
    config: LaserlineDeviceCalibrationConfig | Mapping[str, Any] | None = None,
) -> LaserlineDeviceResult:
    """Run single-camera laserline-device calibration.

    Parameters
    ----------
    input:
        Laserline dataset.
        Preferred type: :class:`vision_calibration.models.LaserlineDataset`.
        Raw serde list is also accepted.
    config:
        Calibration config.
        Preferred type: :class:`vision_calibration.models.LaserlineDeviceCalibrationConfig`.
        If omitted, Rust defaults are used.

    Returns
    -------
    LaserlineDeviceResult
        Joint camera+laser estimate and residual statistics.
    """
    payload = normalize_input_payload(input)
    cfg = _normalize_laserline_config(config)
    raw = cast(dict[str, Any], _native.run_laserline_device(payload, cfg))
    return LaserlineDeviceResult.from_payload(raw)
