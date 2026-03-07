"""High-level Python wrappers around the Rust extension.

Public runner functions accept typed dataclasses from :mod:`vision_calibration.models`
and return typed result dataclasses.

Low-level raw helpers (prefixed with ``_run_*_raw``) are kept for interop paths
that need direct serde payload control.
"""

from __future__ import annotations

from typing import Any, Mapping, TypeVar, cast

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
    ScheimpflugIntrinsicsCalibrationConfig,
    ScheimpflugIntrinsicsResult,
    SingleCamHandeyeCalibrationConfig,
    SingleCamHandeyeDataset,
    SingleCamHandeyeResult,
)
from .types import RobustLoss

_T = TypeVar("_T")


def robust_none() -> RobustLoss:
    """Build a serde-compatible "no robust loss" value."""
    return "None"


def robust_huber(scale: float) -> RobustLoss:
    """Build a serde-compatible Huber robust loss payload."""
    return {"Huber": {"scale": float(scale)}}


def robust_cauchy(scale: float) -> RobustLoss:
    """Build a serde-compatible Cauchy robust loss payload."""
    return {"Cauchy": {"scale": float(scale)}}


def robust_arctan(scale: float) -> RobustLoss:
    """Build a serde-compatible Arctan robust loss payload."""
    return {"Arctan": {"scale": float(scale)}}


def _ensure_type(value: Any, expected_type: type[_T], name: str) -> _T:
    if not isinstance(value, expected_type):
        raise TypeError(
            f"{name} must be {expected_type.__name__}, got {type(value).__name__}"
        )
    return value


def _ensure_optional_type(
    value: Any,
    expected_type: type[_T],
    name: str,
) -> _T | None:
    if value is None:
        return None
    return _ensure_type(value, expected_type, name)


# ─────────────────────────────────────────────────────────────────────────────
# Low-level raw helpers
# ─────────────────────────────────────────────────────────────────────────────


def _run_planar_intrinsics_raw(
    input_payload: Mapping[str, Any],
    config_payload: Mapping[str, Any] | None = None,
) -> PlanarCalibrationResult:
    raw = cast(dict[str, Any], _native.run_planar_intrinsics(input_payload, config_payload))
    return PlanarCalibrationResult.from_payload(raw)


def _run_single_cam_handeye_raw(
    input_payload: Mapping[str, Any],
    config_payload: Mapping[str, Any] | None = None,
) -> SingleCamHandeyeResult:
    raw = cast(dict[str, Any], _native.run_single_cam_handeye(input_payload, config_payload))
    return SingleCamHandeyeResult.from_payload(raw)


def _run_rig_extrinsics_raw(
    input_payload: Mapping[str, Any],
    config_payload: Mapping[str, Any] | None = None,
) -> RigExtrinsicsResult:
    raw = cast(dict[str, Any], _native.run_rig_extrinsics(input_payload, config_payload))
    return RigExtrinsicsResult.from_payload(raw)


def _run_rig_handeye_raw(
    input_payload: Mapping[str, Any],
    config_payload: Mapping[str, Any] | None = None,
) -> RigHandeyeResult:
    raw = cast(dict[str, Any], _native.run_rig_handeye(input_payload, config_payload))
    return RigHandeyeResult.from_payload(raw)


def _run_laserline_device_raw(
    input_payload: list[dict[str, Any]],
    config_payload: Mapping[str, Any] | None = None,
) -> LaserlineDeviceResult:
    raw = cast(dict[str, Any], _native.run_laserline_device(input_payload, config_payload))
    return LaserlineDeviceResult.from_payload(raw)


def _run_scheimpflug_intrinsics_raw(
    input_payload: Mapping[str, Any],
    config_payload: Mapping[str, Any] | None = None,
) -> ScheimpflugIntrinsicsResult:
    raw = cast(
        dict[str, Any],
        _native.run_scheimpflug_intrinsics(input_payload, config_payload),
    )
    return ScheimpflugIntrinsicsResult.from_payload(raw)


# ─────────────────────────────────────────────────────────────────────────────
# High-level typed API
# ─────────────────────────────────────────────────────────────────────────────


def run_planar_intrinsics(
    input: PlanarDataset,
    config: PlanarCalibrationConfig | None = None,
) -> PlanarCalibrationResult:
    """Run planar intrinsics calibration with typed input/config objects."""
    dataset = _ensure_type(input, PlanarDataset, "input")
    cfg = _ensure_optional_type(config, PlanarCalibrationConfig, "config")
    return _run_planar_intrinsics_raw(
        dataset.to_payload(),
        None if cfg is None else cfg.to_payload(),
    )


def run_single_cam_handeye(
    input: SingleCamHandeyeDataset,
    config: SingleCamHandeyeCalibrationConfig | None = None,
) -> SingleCamHandeyeResult:
    """Run single-camera hand-eye calibration with typed input/config objects."""
    dataset = _ensure_type(input, SingleCamHandeyeDataset, "input")
    cfg = _ensure_optional_type(config, SingleCamHandeyeCalibrationConfig, "config")
    return _run_single_cam_handeye_raw(
        dataset.to_payload(),
        None if cfg is None else cfg.to_payload(),
    )


def run_rig_extrinsics(
    input: RigExtrinsicsDataset,
    config: RigExtrinsicsCalibrationConfig | None = None,
) -> RigExtrinsicsResult:
    """Run rig extrinsics calibration with typed input/config objects."""
    dataset = _ensure_type(input, RigExtrinsicsDataset, "input")
    cfg = _ensure_optional_type(config, RigExtrinsicsCalibrationConfig, "config")
    return _run_rig_extrinsics_raw(
        dataset.to_payload(),
        None if cfg is None else cfg.to_payload(),
    )


def run_rig_handeye(
    input: RigHandeyeDataset,
    config: RigHandeyeCalibrationConfig | None = None,
) -> RigHandeyeResult:
    """Run rig hand-eye calibration with typed input/config objects."""
    dataset = _ensure_type(input, RigHandeyeDataset, "input")
    cfg = _ensure_optional_type(config, RigHandeyeCalibrationConfig, "config")
    return _run_rig_handeye_raw(
        dataset.to_payload(),
        None if cfg is None else cfg.to_payload(),
    )


def run_laserline_device(
    input: LaserlineDataset,
    config: LaserlineDeviceCalibrationConfig | None = None,
) -> LaserlineDeviceResult:
    """Run laserline-device calibration with typed input/config objects."""
    dataset = _ensure_type(input, LaserlineDataset, "input")
    cfg = _ensure_optional_type(config, LaserlineDeviceCalibrationConfig, "config")
    return _run_laserline_device_raw(
        dataset.to_payload(),
        None if cfg is None else cfg.to_payload(),
    )


def run_scheimpflug_intrinsics(
    input: PlanarDataset,
    config: ScheimpflugIntrinsicsCalibrationConfig | None = None,
) -> ScheimpflugIntrinsicsResult:
    """Run Scheimpflug intrinsics calibration with typed input/config objects."""
    dataset = _ensure_type(input, PlanarDataset, "input")
    cfg = _ensure_optional_type(config, ScheimpflugIntrinsicsCalibrationConfig, "config")
    return _run_scheimpflug_intrinsics_raw(
        dataset.to_payload(),
        None if cfg is None else cfg.to_payload(),
    )
