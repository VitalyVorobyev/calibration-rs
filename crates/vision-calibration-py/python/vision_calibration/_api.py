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
    LaserlineDeviceInput,
    LaserlineDeviceConfig,
    LaserlineDeviceExport,
    LaserPlane,
    PlanarIntrinsicsConfig,
    PlanarIntrinsicsExport,
    PlanarIntrinsicsInput,
    Pose,
    ProjectionUncertainty,
    QualityReport,
    QualityThresholds,
    RigExtrinsicsConfig,
    RigExtrinsicsInput,
    RigExtrinsicsExport,
    RigHandeyeConfig,
    RigHandeyeInput,
    RigHandeyeLaserlineConfig,
    RigHandeyeLaserlineInput,
    RigHandeyeLaserlineExport,
    RigHandeyeExport,
    RigLaserlineDataset,
    RigLaserlineDeviceConfig,
    RigLaserlineDeviceInput,
    RigLaserlineDeviceExport,
    RigUpstreamCalibration,
    RigHandeyeExport,
    SingleCamHandeyeConfig,
    SingleCamHandeyeInput,
    SingleCamHandeyeExport,
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
) -> PlanarIntrinsicsExport:
    raw = cast(dict[str, Any], _native.run_planar_intrinsics(input_payload, config_payload))
    return PlanarIntrinsicsExport.from_payload(raw)


def _run_single_cam_handeye_raw(
    input_payload: Mapping[str, Any],
    config_payload: Mapping[str, Any] | None = None,
) -> SingleCamHandeyeExport:
    raw = cast(dict[str, Any], _native.run_single_cam_handeye(input_payload, config_payload))
    return SingleCamHandeyeExport.from_payload(raw)


def _run_rig_extrinsics_raw(
    input_payload: Mapping[str, Any],
    config_payload: Mapping[str, Any] | None = None,
) -> RigExtrinsicsExport:
    raw = cast(dict[str, Any], _native.run_rig_extrinsics(input_payload, config_payload))
    return RigExtrinsicsExport.from_payload(raw)


def _run_rig_handeye_raw(
    input_payload: Mapping[str, Any],
    config_payload: Mapping[str, Any] | None = None,
) -> RigHandeyeExport:
    raw = cast(dict[str, Any], _native.run_rig_handeye(input_payload, config_payload))
    return RigHandeyeExport.from_payload(raw)


def _run_laserline_device_raw(
    input_payload: list[dict[str, Any]],
    config_payload: Mapping[str, Any] | None = None,
) -> LaserlineDeviceExport:
    raw = cast(dict[str, Any], _native.run_laserline_device(input_payload, config_payload))
    return LaserlineDeviceExport.from_payload(raw)


# ─────────────────────────────────────────────────────────────────────────────
# High-level typed API
# ─────────────────────────────────────────────────────────────────────────────


def run_planar_intrinsics(
    input: PlanarIntrinsicsInput,
    config: PlanarIntrinsicsConfig | None = None,
) -> PlanarIntrinsicsExport:
    """Run planar intrinsics calibration with typed input/config objects."""
    dataset = _ensure_type(input, PlanarIntrinsicsInput, "input")
    cfg = _ensure_optional_type(config, PlanarIntrinsicsConfig, "config")
    return _run_planar_intrinsics_raw(
        dataset.to_payload(),
        None if cfg is None else cfg.to_payload(),
    )


def run_single_cam_handeye(
    input: SingleCamHandeyeInput,
    config: SingleCamHandeyeConfig | None = None,
) -> SingleCamHandeyeExport:
    """Run single-camera hand-eye calibration with typed input/config objects."""
    dataset = _ensure_type(input, SingleCamHandeyeInput, "input")
    cfg = _ensure_optional_type(config, SingleCamHandeyeConfig, "config")
    return _run_single_cam_handeye_raw(
        dataset.to_payload(),
        None if cfg is None else cfg.to_payload(),
    )


def run_rig_extrinsics(
    input: RigExtrinsicsInput,
    config: RigExtrinsicsConfig | None = None,
) -> RigExtrinsicsExport:
    """Run rig extrinsics calibration with typed input/config objects."""
    dataset = _ensure_type(input, RigExtrinsicsInput, "input")
    cfg = _ensure_optional_type(config, RigExtrinsicsConfig, "config")
    return _run_rig_extrinsics_raw(
        dataset.to_payload(),
        None if cfg is None else cfg.to_payload(),
    )


def run_rig_handeye(
    input: RigHandeyeInput,
    config: RigHandeyeConfig | None = None,
) -> RigHandeyeExport:
    """Run rig hand-eye calibration with typed input/config objects."""
    dataset = _ensure_type(input, RigHandeyeInput, "input")
    cfg = _ensure_optional_type(config, RigHandeyeConfig, "config")
    return _run_rig_handeye_raw(
        dataset.to_payload(),
        None if cfg is None else cfg.to_payload(),
    )


def run_laserline_device(
    input: LaserlineDeviceInput,
    config: LaserlineDeviceConfig | None = None,
) -> LaserlineDeviceExport:
    """Run laserline-device calibration with typed input/config objects."""
    dataset = _ensure_type(input, LaserlineDeviceInput, "input")
    cfg = _ensure_optional_type(config, LaserlineDeviceConfig, "config")
    return _run_laserline_device_raw(
        dataset.to_payload(),
        None if cfg is None else cfg.to_payload(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Rig laserline device
# ─────────────────────────────────────────────────────────────────────────────


def _run_rig_laserline_device_raw(
    input_payload: Mapping[str, Any],
    config_payload: Mapping[str, Any] | None = None,
) -> RigLaserlineDeviceExport:
    raw = cast(
        dict[str, Any],
        _native.run_rig_laserline_device(input_payload, config_payload),
    )
    return RigLaserlineDeviceExport.from_payload(raw)


def run_rig_laserline_device(
    input: RigLaserlineDeviceInput,
    config: RigLaserlineDeviceConfig | None = None,
) -> RigLaserlineDeviceExport:
    """Run rig laserline device calibration with typed input/config objects."""
    inp = _ensure_type(input, RigLaserlineDeviceInput, "input")
    cfg = _ensure_optional_type(config, RigLaserlineDeviceConfig, "config")
    return _run_rig_laserline_device_raw(
        inp.to_payload(),
        None if cfg is None else cfg.to_payload(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Rig hand-eye laserline (joint)
# ─────────────────────────────────────────────────────────────────────────────


def _run_rig_handeye_laserline_raw(
    input_payload: Mapping[str, Any],
    config_payload: Mapping[str, Any] | None = None,
) -> RigHandeyeLaserlineExport:
    raw = cast(
        dict[str, Any],
        _native.run_rig_handeye_laserline(input_payload, config_payload),
    )
    return RigHandeyeLaserlineExport.from_payload(raw)


def run_rig_handeye_laserline(
    input: RigHandeyeLaserlineInput,
    config: RigHandeyeLaserlineConfig | None = None,
) -> RigHandeyeLaserlineExport:
    """Run joint rig hand-eye laserline calibration with typed input/config objects."""
    dataset = _ensure_type(input, RigHandeyeLaserlineInput, "input")
    cfg = _ensure_optional_type(config, RigHandeyeLaserlineConfig, "config")
    return _run_rig_handeye_laserline_raw(
        dataset.to_payload(),
        None if cfg is None else cfg.to_payload(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# pixel_to_gripper_point
# ─────────────────────────────────────────────────────────────────────────────


def _pixel_to_gripper_point_raw(
    cam_idx: int,
    pixel: list[float],
    rig_cal_payload: Mapping[str, Any],
    laser_planes_payload: list[Mapping[str, Any]],
    base_se3_gripper_payload: Mapping[str, Any] | None = None,
) -> tuple[float, float, float]:
    """Low-level raw helper: all arguments are already serde payloads."""
    result = cast(
        list[float],
        _native.pixel_to_gripper_point(
            cam_idx,
            list(pixel),
            rig_cal_payload,
            list(laser_planes_payload),
            base_se3_gripper_payload,
        ),
    )
    return (result[0], result[1], result[2])


def pixel_to_gripper_point(
    cam_idx: int,
    pixel: tuple[float, float] | list[float],
    rig_cal: RigHandeyeExport | Mapping[str, Any],
    laser_planes_rig: list[LaserPlane] | list[Mapping[str, Any]],
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
        Rig hand-eye calibration result — either a typed
        :class:`~vision_calibration.models.RigHandeyeExport` or
        a raw serde payload dict.
    laser_planes_rig:
        Per-camera laser planes in rig frame — either typed
        :class:`~vision_calibration.models.LaserPlane` instances or
        raw serde payload dicts.
    base_se3_gripper:
        Robot gripper pose (required for EyeToHand; ignored for EyeInHand).
        Either a typed :class:`~vision_calibration.models.Pose` or a raw
        serde payload dict.

    Returns
    -------
    tuple[float, float, float]
        3D point ``(x, y, z)`` in gripper frame.
    """
    # Normalise rig_cal — accept the typed dataclass or a raw dict
    if isinstance(rig_cal, RigHandeyeExport):
        rig_cal_payload: Mapping[str, Any] = rig_cal.to_payload()
    else:
        rig_cal_payload = dict(rig_cal)

    # Normalise planes — use LaserPlane.to_payload() which emits the
    # correct {"normal": [x, y, z], "distance": d} shape expected by Rust
    planes_payload: list[Any] = []
    for p in laser_planes_rig:
        if isinstance(p, LaserPlane):
            planes_payload.append(p.to_payload())
        else:
            planes_payload.append(dict(p))

    # Normalise pose
    pose_payload: Any = None
    if base_se3_gripper is not None:
        if isinstance(base_se3_gripper, Pose):
            pose_payload = base_se3_gripper.to_payload()
        else:
            pose_payload = dict(base_se3_gripper)

    return _pixel_to_gripper_point_raw(
        cam_idx, list(pixel), rig_cal_payload, planes_payload, pose_payload
    )


# ─────────────────────────────────────────────────────────────────────────────
# projection_uncertainty
# ─────────────────────────────────────────────────────────────────────────────

CalibrationResult = (
    PlanarIntrinsicsExport
    | SingleCamHandeyeExport
    | LaserlineDeviceExport
    | RigExtrinsicsExport
    | RigHandeyeExport
    | RigLaserlineDeviceExport
    | RigHandeyeLaserlineExport
)


def projection_uncertainty(
    result: CalibrationResult | Mapping[str, Any],
    image_size: tuple[int, int],
    camera: int = 0,
) -> ProjectionUncertainty:
    """How precisely a calibrated camera places a camera-frame ray in its image.

    The major-axis standard deviation, in pixels, of where the ray through each
    node of a grid across the image lands, from the covariance of the camera's
    estimated parameters (see :class:`~vision_calibration.models.ProjectionUncertainty`).

    Parameters
    ----------
    result:
        A calibration result, or an export JSON loaded as a dict.
    image_size:
        ``(width, height)`` of the camera's images, pixels.
    camera:
        Camera index; 0 for single-camera results.

    Raises
    ------
    ValueError
        If the result has no covariance of the camera's parameters (none was
        computed, or the final solve held them fixed), or ``camera`` is out of
        range.
    """
    export = result if isinstance(result, Mapping) else result.export
    if not export:
        raise ValueError("the result carries no export; it was not produced by this library")
    payload = cast(
        Mapping[str, Any],
        _native.projection_uncertainty(dict(export), camera, list(image_size)),
    )
    return ProjectionUncertainty.from_payload(payload)


# ─────────────────────────────────────────────────────────────────────────────
# assess
# ─────────────────────────────────────────────────────────────────────────────


def assess(
    result: CalibrationResult | Mapping[str, Any],
    thresholds: QualityThresholds | None = None,
    image_sizes: list[tuple[int, int]] | None = None,
) -> QualityReport:
    """Judge a calibration result: a ``"pass"``, ``"warn"`` or ``"fail"`` verdict
    and the checks behind it.

    The checks: observability of the estimated parameters, convergence, the
    noise level against a stated feature noise (the χ² variance-factor test),
    the largest projection σ over each camera's image, the share of each image
    the features reached, views with outlying residuals, and camera parameters
    that correlate almost fully. A check whose inputs the result lacks (no
    image size, no stated noise, intrinsics held fixed) is ``"not_applicable"``
    and does not count toward the verdict.

    Parameters
    ----------
    result:
        A calibration result, or an export JSON loaded as a dict.
    thresholds:
        The checks' limits; ``None`` for the defaults.
    image_sizes:
        ``(width, height)`` of each camera's images, pixels, in camera order,
        e.g. ``[(1280, 1024)]`` for one camera. Needed for the projection and
        coverage checks unless the export's image manifest records the sizes.

    Raises
    ------
    ValueError
        If a threshold is out of range, ``image_sizes`` does not give one size
        per camera, or it disagrees with the image manifest.
    """
    export = result if isinstance(result, Mapping) else result.export
    if not export:
        raise ValueError("the result carries no export; it was not produced by this library")
    payload = cast(
        Mapping[str, Any],
        _native.assess(
            dict(export),
            None if thresholds is None else thresholds.to_payload(),
            None if image_sizes is None else [list(size) for size in image_sizes],
        ),
    )
    return QualityReport.from_payload(payload)
