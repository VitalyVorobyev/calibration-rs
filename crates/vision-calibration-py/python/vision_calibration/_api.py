"""High-level Python wrappers around the Rust extension.

The analysis functions (``assess``, ``projection_uncertainty``, the next-view
planners) take the exports of :mod:`vision_calibration.models` and return
typed result dataclasses; ``robust_*`` build the robust-loss values.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence, cast

from . import _vision_calibration as _native
from .models import (
    CameraParams,
    CandidateGrid,
    LaserlineDeviceExport,
    PlanarIntrinsicsExport,
    Pose,
    PredictedRigUncertainty,
    PredictedUncertainty,
    ProjectionUncertainty,
    QualityReport,
    RankedCandidate,
    RankedRobotPose,
    QualityThresholds,
    RigExtrinsicsExport,
    RigHandeyeLaserlineExport,
    RigHandeyeExport,
    RigLaserlineDeviceExport,
    RigUpstreamCalibration,
    SingleCamHandeyeExport,
)
from ._generated_types import RobustLoss


def robust_none() -> RobustLoss:
    """Build a serde-compatible "no robust loss" value."""
    return {"kind": "none"}


def robust_huber(scale: float) -> RobustLoss:
    """Build a serde-compatible Huber robust loss payload."""
    return {"kind": "huber", "scale": float(scale)}


def robust_cauchy(scale: float) -> RobustLoss:
    """Build a serde-compatible Cauchy robust loss payload."""
    return {"kind": "cauchy", "scale": float(scale)}


def robust_arctan(scale: float) -> RobustLoss:
    """Build a serde-compatible Arctan robust loss payload."""
    return {"kind": "arctan", "scale": float(scale)}


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

    Each candidate is a ``cam_se3_target`` pose. The target's points are
    projected through the current estimate (those behind the camera or outside
    the image are dropped); the view's information about the camera, with its
    own pose marginalized, is added to the result's, on the pixel noise scale
    the result's covariance implies.

    Parameters
    ----------
    result:
        A planar intrinsics result, or its export JSON loaded as a dict.
    candidates:
        Candidate target poses (``cam_se3_target``), all added together.
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
    camera: CameraParams | Mapping[str, Any],
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
