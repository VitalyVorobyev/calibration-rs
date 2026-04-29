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
    LaserlinePlane,
    PlanarCalibrationConfig,
    PlanarCalibrationResult,
    PlanarDataset,
    Pose,
    RigExtrinsicsCalibrationConfig,
    RigExtrinsicsDataset,
    RigExtrinsicsResult,
    RigHandeyeCalibrationConfig,
    RigHandeyeDataset,
    RigHandeyeResult,
    RigLaserlineDataset,
    RigLaserlineDeviceCalibrationConfig,
    RigLaserlineDeviceInput,
    RigLaserlineDeviceResult,
    RigLaserlineUpstreamCalibration,
    RigScheimpflugExtrinsicsCalibrationConfig,
    RigScheimpflugExtrinsicsDataset,
    RigScheimpflugExtrinsicsResult,
    RigScheimpflugHandeyeCalibrationConfig,
    RigScheimpflugHandeyeDataset,
    RigScheimpflugHandeyeResult,
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


# ─────────────────────────────────────────────────────────────────────────────
# Scheimpflug rig extrinsics
# ─────────────────────────────────────────────────────────────────────────────


def _run_rig_scheimpflug_extrinsics_raw(
    input_payload: Mapping[str, Any],
    config_payload: Mapping[str, Any] | None = None,
) -> RigScheimpflugExtrinsicsResult:
    raw = cast(
        dict[str, Any],
        _native.run_rig_scheimpflug_extrinsics(input_payload, config_payload),
    )
    return RigScheimpflugExtrinsicsResult.from_payload(raw)


def run_rig_scheimpflug_extrinsics(
    input: RigScheimpflugExtrinsicsDataset,
    config: RigScheimpflugExtrinsicsCalibrationConfig | None = None,
) -> RigScheimpflugExtrinsicsResult:
    """Run Scheimpflug rig extrinsics calibration with typed input/config objects."""
    dataset = _ensure_type(input, RigScheimpflugExtrinsicsDataset, "input")
    cfg = _ensure_optional_type(config, RigScheimpflugExtrinsicsCalibrationConfig, "config")
    return _run_rig_scheimpflug_extrinsics_raw(
        dataset.to_payload(),
        None if cfg is None else cfg.to_payload(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Scheimpflug rig hand-eye
# ─────────────────────────────────────────────────────────────────────────────


def _run_rig_scheimpflug_handeye_raw(
    input_payload: Mapping[str, Any],
    config_payload: Mapping[str, Any] | None = None,
) -> RigScheimpflugHandeyeResult:
    raw = cast(
        dict[str, Any],
        _native.run_rig_scheimpflug_handeye(input_payload, config_payload),
    )
    return RigScheimpflugHandeyeResult.from_payload(raw)


def run_rig_scheimpflug_handeye(
    input: RigScheimpflugHandeyeDataset,
    config: RigScheimpflugHandeyeCalibrationConfig | None = None,
) -> RigScheimpflugHandeyeResult:
    """Run Scheimpflug rig hand-eye calibration with typed input/config objects."""
    dataset = _ensure_type(input, RigScheimpflugHandeyeDataset, "input")
    cfg = _ensure_optional_type(config, RigScheimpflugHandeyeCalibrationConfig, "config")
    return _run_rig_scheimpflug_handeye_raw(
        dataset.to_payload(),
        None if cfg is None else cfg.to_payload(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Rig laserline device
# ─────────────────────────────────────────────────────────────────────────────


def _run_rig_laserline_device_raw(
    input_payload: Mapping[str, Any],
    config_payload: Mapping[str, Any] | None = None,
) -> RigLaserlineDeviceResult:
    raw = cast(
        dict[str, Any],
        _native.run_rig_laserline_device(input_payload, config_payload),
    )
    return RigLaserlineDeviceResult.from_payload(raw)


def run_rig_laserline_device(
    input: RigLaserlineDeviceInput,
    config: RigLaserlineDeviceCalibrationConfig | None = None,
) -> RigLaserlineDeviceResult:
    """Run rig laserline device calibration with typed input/config objects."""
    inp = _ensure_type(input, RigLaserlineDeviceInput, "input")
    cfg = _ensure_optional_type(config, RigLaserlineDeviceCalibrationConfig, "config")
    return _run_rig_laserline_device_raw(
        inp.to_payload(),
        None if cfg is None else cfg.to_payload(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# pixel_to_gripper_point
# ─────────────────────────────────────────────────────────────────────────────


def pixel_to_gripper_point(
    cam_idx: int,
    pixel: tuple[float, float] | list[float],
    rig_cal: RigScheimpflugHandeyeResult | Mapping[str, Any],
    laser_planes_rig: list[LaserlinePlane] | list[Mapping[str, Any]],
    base_se3_gripper: Pose | Mapping[str, Any] | None = None,
) -> tuple[float, float, float]:
    """Map a laser pixel in a rig camera to a 3D point in the gripper frame.

    Parameters
    ----------
    cam_idx:
        Camera index.
    pixel:
        Observed pixel as ``(u, v)``.
    rig_cal:
        Scheimpflug rig hand-eye calibration result (typed or raw payload).
    laser_planes_rig:
        Per-camera laser planes in rig frame (typed or raw payloads).
    base_se3_gripper:
        Robot gripper pose (required for EyeToHand; ignored for EyeInHand).

    Returns
    -------
    tuple[float, float, float]
        3D point ``(x, y, z)`` in gripper frame.
    """
    # Normalise pixel
    px_list = list(pixel)

    # Normalise rig_cal
    if isinstance(rig_cal, RigScheimpflugHandeyeResult):
        raise TypeError(
            "rig_cal must be the raw payload dict returned by run_rig_scheimpflug_handeye "
            "or equivalent — not a RigScheimpflugHandeyeResult; "
            "pass the dict from _run_rig_scheimpflug_handeye_raw or the native call"
        )
    rig_cal_payload = dict(rig_cal)

    # Normalise planes
    planes_payload: list[Any] = []
    for p in laser_planes_rig:
        if isinstance(p, LaserlinePlane):
            planes_payload.append({
                "normal": {"coords": list(p.normal_xyz)},
                "distance": p.distance,
            })
        else:
            planes_payload.append(dict(p))

    # Normalise pose
    pose_payload: Any = None
    if base_se3_gripper is not None:
        if isinstance(base_se3_gripper, Pose):
            pose_payload = base_se3_gripper.to_payload()
        else:
            pose_payload = dict(base_se3_gripper)

    result = cast(
        list[float],
        _native.pixel_to_gripper_point(
            cam_idx, px_list, rig_cal_payload, planes_payload, pose_payload
        ),
    )
    return (result[0], result[1], result[2])
