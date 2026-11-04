"""High-level Python wrappers around the Rust extension.

Public runner functions accept typed dataclasses from :mod:`vision_calibration.models`
and return typed result dataclasses.

Low-level raw helpers (prefixed with ``_run_*_raw``) are kept for interop paths
that need direct serde payload control.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence, TypeVar, cast

from . import _vision_calibration as _native
from .models import (
    Camera,
    CandidateGrid,
    LaserlineDeviceInput,
    LaserlineDeviceConfig,
    LaserlineDeviceExport,
    PlanarIntrinsicsConfig,
    PlanarIntrinsicsExport,
    PlanarIntrinsicsInput,
    Pose,
    PredictedRigUncertainty,
    PredictedUncertainty,
    ProjectionUncertainty,
    QualityReport,
    RankedCandidate,
    RankedRobotPose,
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
from ._generated_types import RobustLoss
from .types import ProgressCallback

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
    config_payload: Mapping[str, Any] | None = None,    progress: ProgressCallback | None = None,
) -> PlanarIntrinsicsExport:
    raw = cast(dict[str, Any], _native.run_planar_intrinsics(input_payload, config_payload, progress))
    return PlanarIntrinsicsExport.from_payload(raw)


def _run_single_cam_handeye_raw(
    input_payload: Mapping[str, Any],
    config_payload: Mapping[str, Any] | None = None,    progress: ProgressCallback | None = None,
) -> SingleCamHandeyeExport:
    raw = cast(dict[str, Any], _native.run_single_cam_handeye(input_payload, config_payload, progress))
    return SingleCamHandeyeExport.from_payload(raw)


def _run_rig_extrinsics_raw(
    input_payload: Mapping[str, Any],
    config_payload: Mapping[str, Any] | None = None,    progress: ProgressCallback | None = None,
) -> RigExtrinsicsExport:
    raw = cast(dict[str, Any], _native.run_rig_extrinsics(input_payload, config_payload, progress))
    return RigExtrinsicsExport.from_payload(raw)


def _run_rig_handeye_raw(
    input_payload: Mapping[str, Any],
    config_payload: Mapping[str, Any] | None = None,    progress: ProgressCallback | None = None,
) -> RigHandeyeExport:
    raw = cast(dict[str, Any], _native.run_rig_handeye(input_payload, config_payload, progress))
    return RigHandeyeExport.from_payload(raw)


def _run_laserline_device_raw(
    input_payload: list[dict[str, Any]],
    config_payload: Mapping[str, Any] | None = None,    progress: ProgressCallback | None = None,
) -> LaserlineDeviceExport:
    raw = cast(dict[str, Any], _native.run_laserline_device(input_payload, config_payload, progress))
    return LaserlineDeviceExport.from_payload(raw)


# ─────────────────────────────────────────────────────────────────────────────
# High-level typed API
# ─────────────────────────────────────────────────────────────────────────────


def run_planar_intrinsics(
    input: PlanarIntrinsicsInput,
    config: PlanarIntrinsicsConfig | None = None,    *,
    progress: ProgressCallback | None = None,
) -> PlanarIntrinsicsExport:
    """Run planar intrinsics calibration with typed input/config objects.

    ``progress``, when given, is called with each progress report (a
    :data:`~vision_calibration.types.Progress` mapping) as the run detects,
    steps and iterates; raising from it stops the run and the exception
    propagates. Ctrl-C stops the run and raises :class:`KeyboardInterrupt`.
    The same holds for every ``run_*`` function.
    """
    dataset = _ensure_type(input, PlanarIntrinsicsInput, "input")
    cfg = _ensure_optional_type(config, PlanarIntrinsicsConfig, "config")
    return _run_planar_intrinsics_raw(
        dataset.to_payload(),
        None if cfg is None else cfg.to_payload(),
        progress,
    )


def run_single_cam_handeye(
    input: SingleCamHandeyeInput,
    config: SingleCamHandeyeConfig | None = None,    *,
    progress: ProgressCallback | None = None,
) -> SingleCamHandeyeExport:
    """Run single-camera hand-eye calibration with typed input/config objects."""
    dataset = _ensure_type(input, SingleCamHandeyeInput, "input")
    cfg = _ensure_optional_type(config, SingleCamHandeyeConfig, "config")
    return _run_single_cam_handeye_raw(
        dataset.to_payload(),
        None if cfg is None else cfg.to_payload(),
        progress,
    )


def run_rig_extrinsics(
    input: RigExtrinsicsInput,
    config: RigExtrinsicsConfig | None = None,    *,
    progress: ProgressCallback | None = None,
) -> RigExtrinsicsExport:
    """Run rig extrinsics calibration with typed input/config objects."""
    dataset = _ensure_type(input, RigExtrinsicsInput, "input")
    cfg = _ensure_optional_type(config, RigExtrinsicsConfig, "config")
    return _run_rig_extrinsics_raw(
        dataset.to_payload(),
        None if cfg is None else cfg.to_payload(),
        progress,
    )


def run_rig_handeye(
    input: RigHandeyeInput,
    config: RigHandeyeConfig | None = None,    *,
    progress: ProgressCallback | None = None,
) -> RigHandeyeExport:
    """Run rig hand-eye calibration with typed input/config objects."""
    dataset = _ensure_type(input, RigHandeyeInput, "input")
    cfg = _ensure_optional_type(config, RigHandeyeConfig, "config")
    return _run_rig_handeye_raw(
        dataset.to_payload(),
        None if cfg is None else cfg.to_payload(),
        progress,
    )


def run_laserline_device(
    input: LaserlineDeviceInput,
    config: LaserlineDeviceConfig | None = None,    *,
    progress: ProgressCallback | None = None,
) -> LaserlineDeviceExport:
    """Run laserline-device calibration with typed input/config objects."""
    dataset = _ensure_type(input, LaserlineDeviceInput, "input")
    cfg = _ensure_optional_type(config, LaserlineDeviceConfig, "config")
    return _run_laserline_device_raw(
        dataset.to_payload(),
        None if cfg is None else cfg.to_payload(),
        progress,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Rig laserline device
# ─────────────────────────────────────────────────────────────────────────────


def _run_rig_laserline_device_raw(
    input_payload: Mapping[str, Any],
    config_payload: Mapping[str, Any] | None = None,    progress: ProgressCallback | None = None,
) -> RigLaserlineDeviceExport:
    raw = cast(
        dict[str, Any],
        _native.run_rig_laserline_device(input_payload, config_payload, progress),
    )
    return RigLaserlineDeviceExport.from_payload(raw)


def run_rig_laserline_device(
    input: RigLaserlineDeviceInput,
    config: RigLaserlineDeviceConfig | None = None,    *,
    progress: ProgressCallback | None = None,
) -> RigLaserlineDeviceExport:
    """Run rig laserline device calibration with typed input/config objects."""
    inp = _ensure_type(input, RigLaserlineDeviceInput, "input")
    cfg = _ensure_optional_type(config, RigLaserlineDeviceConfig, "config")
    return _run_rig_laserline_device_raw(
        inp.to_payload(),
        None if cfg is None else cfg.to_payload(),
        progress,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Rig hand-eye laserline (joint)
# ─────────────────────────────────────────────────────────────────────────────


def _run_rig_handeye_laserline_raw(
    input_payload: Mapping[str, Any],
    config_payload: Mapping[str, Any] | None = None,    progress: ProgressCallback | None = None,
) -> RigHandeyeLaserlineExport:
    raw = cast(
        dict[str, Any],
        _native.run_rig_handeye_laserline(input_payload, config_payload, progress),
    )
    return RigHandeyeLaserlineExport.from_payload(raw)


def run_rig_handeye_laserline(
    input: RigHandeyeLaserlineInput,
    config: RigHandeyeLaserlineConfig | None = None,    *,
    progress: ProgressCallback | None = None,
) -> RigHandeyeLaserlineExport:
    """Run joint rig hand-eye laserline calibration with typed input/config objects."""
    dataset = _ensure_type(input, RigHandeyeLaserlineInput, "input")
    cfg = _ensure_optional_type(config, RigHandeyeLaserlineConfig, "config")
    return _run_rig_handeye_laserline_raw(
        dataset.to_payload(),
        None if cfg is None else cfg.to_payload(),
        progress,
    )


# ─────────────────────────────────────────────────────────────────────────────
# rig_upstream_calibration
# ─────────────────────────────────────────────────────────────────────────────


def rig_upstream_calibration(
    export: RigHandeyeExport, rig_se3_target: Sequence[Pose]
) -> RigUpstreamCalibration:
    """A rig hand-eye result as the upstream of a rig laserline input.

    Parameters
    ----------
    export:
        The rig hand-eye result the laser planes are fitted against.
    rig_se3_target:
        One rig-from-target pose per view of the laserline dataset, through
        the hand-eye chain: ``export.rig_se3_target`` when the laser views
        are its views.

    Returns
    -------
    RigUpstreamCalibration
        The cameras, extrinsics and target poses, with the export's
        covariance in their terms in ``uncertainty``: the cameras and
        extrinsics as they are, the hand-eye and target poses as each view's
        target pose. The robot poses count as exact.

    Raises
    ------
    ValueError
        If the export's covariance lists part of a pose, or its camera and
        extrinsic counts differ.
    """
    payload = _native.rig_upstream_calibration(
        export.export, [p.to_payload() for p in rig_se3_target]
    )
    return RigUpstreamCalibration.from_payload(cast(Mapping[str, Any], payload))


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


def html_report(
    result: CalibrationResult | Mapping[str, Any],
    image_sizes: list[tuple[int, int]] | None = None,
    *,
    title: str | None = None,
    camera_names: Sequence[str] | None = None,
    thresholds: QualityThresholds | None = None,
) -> str:
    """A calibration result as one self-contained HTML page.

    The page holds the :func:`assess` verdict with every check and, per
    camera, its model, each parameter with its σ, the reprojection-error
    statistics, and four figures: the error histogram, the mean error by
    image region, the mean error per view and the projection-σ map. It loads
    nothing from elsewhere, so it opens offline; the same result gives the
    same page.

    Parameters
    ----------
    result:
        A calibration result, or an export JSON loaded as a dict.
    image_sizes:
        ``(width, height)`` of each camera's images, pixels, in camera order,
        as :func:`assess` takes them. Without a camera's size the
        projection-σ map is left out.
    title:
        The page's heading; ``None`` for "Calibration report".
    camera_names:
        Each camera's name, in camera order, e.g. a manifest's camera ids.
    thresholds:
        The quality checks' limits; ``None`` for the defaults.

    Raises
    ------
    ValueError
        If a threshold is out of range or ``image_sizes`` does not give one
        size per camera.

    Example
    -------
    >>> page = vc.html_report(result, image_sizes=[(1280, 960)], title="Left camera")
    >>> Path("report.html").write_text(page, encoding="utf-8")
    """
    export = result if isinstance(result, Mapping) else result.export
    if not export:
        raise ValueError("the result carries no export; it was not produced by this library")
    return cast(
        str,
        _native.html_report(
            dict(export),
            None if image_sizes is None else [list(size) for size in image_sizes],
            title,
            None if camera_names is None else list(camera_names),
            None if thresholds is None else thresholds.to_payload(),
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Next-view planning
# ─────────────────────────────────────────────────────────────────────────────

Point3 = tuple[float, float, float] | Sequence[float]


def _export_payload(result: Any) -> dict[str, Any]:
    export = result if isinstance(result, Mapping) else result.export
    if not export:
        raise ValueError("the result carries no export; it was not produced by this library")
    return dict(export)


def _poses_payload(poses: Sequence[Pose | Mapping[str, Any]]) -> list[Any]:
    return [p.to_payload() if isinstance(p, Pose) else dict(p) for p in poses]


def _points_payload(points: Sequence[Point3]) -> list[list[float]]:
    return [[float(v) for v in p] for p in points]


def predict_with_views(
    result: PlanarIntrinsicsExport | Mapping[str, Any],
    candidates: Sequence[Pose | Mapping[str, Any]],
    target_points: Sequence[Point3],
    image_size: tuple[int, int],
) -> PredictedUncertainty:
    """The covariance a planar calibration would have with more views of its
    target, predicted without a re-solve.

    Each candidate is a ``camera_se3_target`` pose. The target's points are
    projected through the current estimate (those behind the camera or outside
    the image are dropped); the view's information about the camera, with its
    own pose marginalized, is added to the result's, on the pixel noise scale
    the result's covariance implies.

    Parameters
    ----------
    result:
        A planar intrinsics result, or its export JSON loaded as a dict.
    candidates:
        Candidate target poses (``camera_se3_target``), all added together.
    target_points:
        The target's points in its own frame, metres.
    image_size:
        ``(width, height)`` of the camera's images, pixels.

    Raises
    ------
    ValueError
        If the result has no covariance, ``candidates`` or ``target_points`` is
        empty, or the image is smaller than 2×2 pixels.
    TypeError
        If the result is not a planar intrinsics result.
    """
    payload = cast(
        Mapping[str, Any],
        _native.predict_with_views(
            _export_payload(result),
            _poses_payload(candidates),
            _points_payload(target_points),
            list(image_size),
        ),
    )
    return PredictedUncertainty.from_payload(payload)


def rank_candidates(
    result: PlanarIntrinsicsExport | Mapping[str, Any],
    candidates: Sequence[Pose | Mapping[str, Any]],
    target_points: Sequence[Point3],
    image_size: tuple[int, int],
) -> list[RankedCandidate]:
    """Candidate views of a planar calibration, each added alone, ranked from
    the largest reduction of the maximum projection σ over the image to the
    smallest; ties keep the candidates' order.

    Parameters and errors as :func:`predict_with_views`.
    """
    payload = cast(
        list[Any],
        _native.rank_candidates(
            _export_payload(result),
            _poses_payload(candidates),
            _points_payload(target_points),
            list(image_size),
        ),
    )
    return [RankedCandidate.from_payload(cast(Mapping[str, Any], r)) for r in payload]


def predict_with_robot_poses(
    result: RigHandeyeExport | Mapping[str, Any],
    candidates: Sequence[Pose | Mapping[str, Any]],
    target_points: Sequence[Point3],
    image_sizes: Sequence[tuple[int, int]],
) -> PredictedRigUncertainty:
    """The covariance a rig hand-eye calibration would have with views at more
    robot poses, predicted without a re-solve.

    Each candidate is a ``base_se3_gripper`` robot pose. Every camera sees the
    target through the result's chain at that pose (the points behind it or
    outside its image are dropped, and a camera that keeps fewer than four adds
    nothing). The view's information about every estimated parameter (the
    cameras, the rig, the hand-eye and the target) is added to the result's, on
    the pixel noise scale the result's covariance implies. A result that
    corrected its robot poses corrects the candidate's too, under its robot pose
    error.

    Parameters
    ----------
    result:
        A rig hand-eye result, or its export JSON loaded as a dict.
    candidates:
        Candidate robot poses (``base_se3_gripper``), all added together.
    target_points:
        The target's points in its own frame, metres.
    image_sizes:
        ``(width, height)`` of each camera's images, pixels.

    Raises
    ------
    ValueError
        If the result has no covariance, ``candidates`` or ``target_points`` is
        empty, ``image_sizes`` does not give one size per camera, or the result
        cannot be planned from (observations whitened by stated per-point
        covariances, a missing hand-eye or target pose).
    TypeError
        If the result is not a rig hand-eye result.
    """
    payload = cast(
        Mapping[str, Any],
        _native.predict_with_robot_poses(
            _export_payload(result),
            _poses_payload(candidates),
            _points_payload(target_points),
            [list(size) for size in image_sizes],
        ),
    )
    return PredictedRigUncertainty.from_payload(payload)


def rank_robot_poses(
    result: RigHandeyeExport | Mapping[str, Any],
    candidates: Sequence[Pose | Mapping[str, Any]],
    target_points: Sequence[Point3],
    image_sizes: Sequence[tuple[int, int]],
) -> list[RankedRobotPose]:
    """Candidate robot poses of a rig hand-eye calibration, each added alone,
    ranked from the largest reduction of ``ln det Σ`` (the log-volume of the
    whole covariance) to the smallest; ties keep the candidates' order.

    Parameters and errors as :func:`predict_with_robot_poses`.
    """
    payload = cast(
        list[Any],
        _native.rank_robot_poses(
            _export_payload(result),
            _poses_payload(candidates),
            _points_payload(target_points),
            [list(size) for size in image_sizes],
        ),
    )
    return [RankedRobotPose.from_payload(cast(Mapping[str, Any], r)) for r in payload]


def candidate_poses(
    camera: Camera | Mapping[str, Any],
    target_points: Sequence[Point3],
    image_size: tuple[int, int],
    grid: CandidateGrid,
) -> list[Pose]:
    """A deterministic set of candidate target poses over the image (see
    :class:`~vision_calibration.models.CandidateGrid`).

    Parameters
    ----------
    camera:
        The calibrated camera (``result.camera``) or its ``CameraParams`` payload.
    target_points:
        The target's points in its own frame, metres.
    image_size:
        ``(width, height)`` of the camera's images, pixels.
    grid:
        The regions, distance and tilt of the candidates.

    Raises
    ------
    ValueError
        If a grid setting is out of range, ``target_points`` is empty, or the
        image is smaller than 2×2 pixels.
    """
    camera_payload = dict(camera) if isinstance(camera, Mapping) else camera.to_payload()
    payload = cast(
        list[Any],
        _native.candidate_poses(
            camera_payload,
            _points_payload(target_points),
            list(image_size),
            grid.to_payload(),
        ),
    )
    return [Pose.from_payload(cast(Mapping[str, Any], p)) for p in payload]
