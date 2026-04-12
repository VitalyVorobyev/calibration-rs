"""Multi-view geometry pipelines from the ``vision-mvg`` Rust crate.

Correspondences are accepted as ``(N, 4)`` numpy arrays with columns
``[x1, y1, x2, y2]``.  All matrix outputs are numpy ndarrays.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from . import _vision_calibration as _native

if TYPE_CHECKING:
    from .models import RansacOptions

_mvg = _native.mvg


# -- Data classes --------------------------------------------------------------


@dataclass(slots=True)
class TriangulatedPoint:
    """A single triangulated 3-D point with quality metrics."""

    point: tuple[float, float, float]
    """(x, y, z) coordinates."""
    reprojection_error: float
    """Reprojection error in pixels."""
    parallax_deg: float
    """Parallax angle in degrees."""
    in_front: bool
    """Whether the point is in front of both cameras."""


@dataclass(slots=True)
class RelativePose:
    """Relative pose recovered from calibrated correspondences."""

    r: np.ndarray
    """(3, 3) rotation matrix."""
    t: np.ndarray
    """(3,) unit translation vector."""
    essential: np.ndarray
    """(3, 3) essential matrix."""
    points: list[TriangulatedPoint]
    """Triangulated points used for cheirality disambiguation."""


@dataclass(slots=True)
class RobustRelativePose:
    """Relative pose recovered with RANSAC."""

    r: np.ndarray
    """(3, 3) rotation matrix."""
    t: np.ndarray
    """(3,) unit translation vector."""
    essential: np.ndarray
    """(3, 3) essential matrix."""
    inliers: list[int]
    """Indices of inlier correspondences."""
    inlier_rms: float
    """RMS Sampson distance over inliers."""


@dataclass(slots=True)
class EssentialEstimate:
    """RANSAC essential matrix estimate."""

    essential: np.ndarray
    """(3, 3) essential matrix."""
    inliers: list[int]
    """Indices of inlier correspondences."""
    inlier_rms: float
    """RMS Sampson distance over inliers."""


@dataclass(slots=True)
class HomographyEstimate:
    """RANSAC homography estimate."""

    homography: np.ndarray
    """(3, 3) homography matrix."""
    inliers: list[int]
    """Indices of inlier correspondences."""
    inlier_rms: float
    """RMS symmetric transfer error over inliers."""


@dataclass(slots=True)
class HomographyDecomposition:
    """One candidate decomposition of a homography."""

    r: np.ndarray
    """(3, 3) rotation matrix."""
    t: np.ndarray
    """(3,) translation vector."""
    normal: np.ndarray
    """(3,) plane normal vector."""


@dataclass(slots=True)
class SceneDiagnostics:
    """Degeneracy analysis of a two-view scene."""

    median_parallax_deg: float
    """Median parallax angle in degrees."""
    is_pure_rotation: bool
    """Whether the motion is a pure rotation (no translation)."""
    is_planar: bool
    """Whether the scene is approximately planar."""
    baseline_ratio: float
    """Ratio of baseline to median scene depth."""


# -- Functions -----------------------------------------------------------------


def recover_relative_pose(corrs: np.ndarray) -> RelativePose:
    """Recover relative pose from calibrated correspondences.

    Parameters
    ----------
    corrs : ndarray, shape (N, 4), N >= 5
        Calibrated correspondences: columns ``[x1, y1, x2, y2]``.

    Returns
    -------
    RelativePose
    """
    d = _mvg.recover_relative_pose(corrs)
    return RelativePose(
        r=d["r"],
        t=d["t"],
        essential=d["essential"],
        points=[
            TriangulatedPoint(
                point=p["point"],
                reprojection_error=p["reprojection_error"],
                parallax_deg=p["parallax_deg"],
                in_front=p["in_front"],
            )
            for p in d["points"]
        ],
    )


def recover_relative_pose_robust(
    corrs: np.ndarray, opts: RansacOptions
) -> RobustRelativePose:
    """Robust relative pose recovery using RANSAC.

    Parameters
    ----------
    corrs : ndarray, shape (N, 4)
        Calibrated correspondences: columns ``[x1, y1, x2, y2]``.
    opts : RansacOptions
        RANSAC configuration.

    Returns
    -------
    RobustRelativePose
    """
    d = _mvg.recover_relative_pose_robust(corrs, opts)
    return RobustRelativePose(
        r=d["r"],
        t=d["t"],
        essential=d["essential"],
        inliers=d["inliers"],
        inlier_rms=d["inlier_rms"],
    )


def estimate_essential(
    corrs: np.ndarray, opts: RansacOptions
) -> EssentialEstimate:
    """RANSAC essential matrix estimation.

    Parameters
    ----------
    corrs : ndarray, shape (N, 4)
        Calibrated correspondences.
    opts : RansacOptions
        RANSAC configuration.

    Returns
    -------
    EssentialEstimate
    """
    d = _mvg.estimate_essential(corrs, opts)
    return EssentialEstimate(
        essential=d["essential"],
        inliers=d["inliers"],
        inlier_rms=d["inlier_rms"],
    )


def estimate_homography(
    corrs: np.ndarray, opts: RansacOptions
) -> HomographyEstimate:
    """RANSAC homography estimation.

    Parameters
    ----------
    corrs : ndarray, shape (N, 4)
        Point correspondences.
    opts : RansacOptions
        RANSAC configuration.

    Returns
    -------
    HomographyEstimate
    """
    d = _mvg.estimate_homography(corrs, opts)
    return HomographyEstimate(
        homography=d["homography"],
        inliers=d["inliers"],
        inlier_rms=d["inlier_rms"],
    )


def decompose_homography(h: np.ndarray) -> list[HomographyDecomposition]:
    """Decompose a homography into candidate (R, t, normal) tuples.

    Parameters
    ----------
    h : ndarray, shape (3, 3)
        Homography matrix.

    Returns
    -------
    list of HomographyDecomposition
    """
    decomps = _mvg.decompose_homography(h)
    return [
        HomographyDecomposition(r=d["r"], t=d["t"], normal=d["normal"])
        for d in decomps
    ]


def homography_transfer(
    h: np.ndarray, pt: tuple[float, float]
) -> tuple[float, float]:
    """Apply a homography to a 2-D point.

    Parameters
    ----------
    h : ndarray, shape (3, 3)
        Homography matrix.
    pt : (x, y)
        Source point.

    Returns
    -------
    (x', y') : tuple of float
        Transformed point.
    """
    return _mvg.homography_transfer(h, pt)


def triangulate_two_view(
    r: np.ndarray,
    t: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
) -> list[TriangulatedPoint]:
    """Triangulate points from two calibrated views.

    Parameters
    ----------
    r : ndarray, shape (3, 3)
        Rotation from view 1 to view 2.
    t : ndarray, shape (3,)
        Translation from view 1 to view 2.
    pts1, pts2 : ndarray, shape (N, 2)
        Calibrated 2-D observations in each view.

    Returns
    -------
    list of TriangulatedPoint
    """
    tps = _mvg.triangulate_two_view(r, t, pts1, pts2)
    return [
        TriangulatedPoint(
            point=p["point"],
            reprojection_error=p["reprojection_error"],
            parallax_deg=p["parallax_deg"],
            in_front=p["in_front"],
        )
        for p in tps
    ]


def analyze_scene(
    corrs: np.ndarray,
    e: np.ndarray,
    r: np.ndarray,
    t: np.ndarray,
) -> SceneDiagnostics:
    """Analyze a two-view scene for degeneracies.

    Parameters
    ----------
    corrs : ndarray, shape (N, 4)
        Correspondences.
    e : ndarray, shape (3, 3)
        Essential matrix.
    r : ndarray, shape (3, 3)
        Rotation matrix.
    t : ndarray, shape (3,)
        Translation vector.

    Returns
    -------
    SceneDiagnostics
    """
    d = _mvg.analyze_scene(corrs, e, r, t)
    return SceneDiagnostics(
        median_parallax_deg=d["median_parallax_deg"],
        is_pure_rotation=d["is_pure_rotation"],
        is_planar=d["is_planar"],
        baseline_ratio=d["baseline_ratio"],
    )


def sampson_distance(
    f: np.ndarray,
    pt1: tuple[float, float],
    pt2: tuple[float, float],
) -> float:
    """Sampson distance for a fundamental/essential matrix and a correspondence.

    Parameters
    ----------
    f : ndarray, shape (3, 3)
        Fundamental or essential matrix.
    pt1, pt2 : (x, y)
        Corresponding points.

    Returns
    -------
    float
        Sampson distance.
    """
    return _mvg.sampson_distance(f, pt1, pt2)


def symmetric_transfer_error(
    h: np.ndarray,
    pt1: tuple[float, float],
    pt2: tuple[float, float],
) -> float:
    """Symmetric transfer error for a homography and a correspondence.

    Parameters
    ----------
    h : ndarray, shape (3, 3)
        Homography matrix.
    pt1, pt2 : (x, y)
        Corresponding points.

    Returns
    -------
    float
        Symmetric transfer error.
    """
    return _mvg.symmetric_transfer_error(h, pt1, pt2)
