"""Low-level geometry solvers from the ``vision-geometry`` Rust crate.

All matrix inputs and outputs are numpy ndarrays.  Point arrays use shape
``(N, 2)`` for 2-D or ``(N, 3)`` for 3-D points.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from . import _vision_calibration as _native

if TYPE_CHECKING:
    from .models import RansacOptions

_geo = _native.geometry


@dataclass(slots=True)
class CameraMatrixDecomposition:
    """Result of decomposing a 3x4 camera matrix P = K [R | t]."""

    k: np.ndarray
    """(3, 3) upper-triangular intrinsic matrix."""
    r: np.ndarray
    """(3, 3) rotation matrix."""
    t: np.ndarray
    """(3,) translation vector."""


# -- Epipolar ------------------------------------------------------------------


def essential_5point(
    pts1: np.ndarray, pts2: np.ndarray
) -> list[np.ndarray]:
    """Nister 5-point essential matrix solver.

    Parameters
    ----------
    pts1, pts2 : ndarray, shape (5, 2)
        Calibrated (normalised) point correspondences.

    Returns
    -------
    list of ndarray, shape (3, 3)
        Up to 10 candidate essential matrices.
    """
    return _geo.essential_5point(pts1, pts2)


def fundamental_7point(
    pts1: np.ndarray, pts2: np.ndarray
) -> list[np.ndarray]:
    """7-point fundamental matrix solver.

    Parameters
    ----------
    pts1, pts2 : ndarray, shape (7, 2)
        Pixel point correspondences.

    Returns
    -------
    list of ndarray, shape (3, 3)
        One or three candidate fundamental matrices.
    """
    return _geo.fundamental_7point(pts1, pts2)


def fundamental_8point(
    pts1: np.ndarray, pts2: np.ndarray
) -> np.ndarray:
    """Normalised 8-point fundamental matrix solver.

    Parameters
    ----------
    pts1, pts2 : ndarray, shape (N, 2), N >= 8
        Pixel point correspondences.

    Returns
    -------
    ndarray, shape (3, 3)
        Fundamental matrix.
    """
    return _geo.fundamental_8point(pts1, pts2)


def fundamental_8point_ransac(
    pts1: np.ndarray,
    pts2: np.ndarray,
    opts: RansacOptions,
) -> tuple[np.ndarray, list[int]]:
    """RANSAC 8-point fundamental matrix estimation.

    Parameters
    ----------
    pts1, pts2 : ndarray, shape (N, 2)
        Pixel point correspondences.
    opts : RansacOptions
        RANSAC configuration.

    Returns
    -------
    F : ndarray, shape (3, 3)
        Fundamental matrix.
    inliers : list of int
        Indices of inlier correspondences.
    """
    return _geo.fundamental_8point_ransac(pts1, pts2, opts)


def decompose_essential(
    e: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Decompose an essential matrix into four candidate (R, t) pairs.

    Parameters
    ----------
    e : ndarray, shape (3, 3)
        Essential matrix.

    Returns
    -------
    list of (R, t)
        R is (3, 3) rotation, t is (3,) translation.
    """
    return _geo.decompose_essential(e)


# -- Homography ----------------------------------------------------------------


def dlt_homography(
    src: np.ndarray, dst: np.ndarray
) -> np.ndarray:
    """Normalised DLT homography estimation.

    Parameters
    ----------
    src, dst : ndarray, shape (N, 2), N >= 4
        Source and destination point correspondences.

    Returns
    -------
    ndarray, shape (3, 3)
        Homography mapping src -> dst.
    """
    return _geo.dlt_homography(src, dst)


def dlt_homography_ransac(
    src: np.ndarray,
    dst: np.ndarray,
    opts: RansacOptions,
) -> tuple[np.ndarray, list[int]]:
    """RANSAC DLT homography estimation.

    Parameters
    ----------
    src, dst : ndarray, shape (N, 2)
        Source and destination point correspondences.
    opts : RansacOptions
        RANSAC configuration.

    Returns
    -------
    H : ndarray, shape (3, 3)
        Homography mapping src -> dst.
    inliers : list of int
        Indices of inlier correspondences.
    """
    return _geo.dlt_homography_ransac(src, dst, opts)


# -- Camera matrix -------------------------------------------------------------


def dlt_camera_matrix(
    world: np.ndarray, image: np.ndarray
) -> np.ndarray:
    """DLT camera matrix estimation.

    Parameters
    ----------
    world : ndarray, shape (N, 3), N >= 6
        3-D world points.
    image : ndarray, shape (N, 2)
        Corresponding 2-D image points.

    Returns
    -------
    ndarray, shape (3, 4)
        Camera projection matrix P.
    """
    return _geo.dlt_camera_matrix(world, image)


def decompose_camera_matrix(
    p: np.ndarray,
) -> CameraMatrixDecomposition:
    """Decompose a 3x4 camera matrix into intrinsics, rotation, translation.

    Parameters
    ----------
    p : ndarray, shape (3, 4)
        Camera projection matrix.

    Returns
    -------
    CameraMatrixDecomposition
        Named fields: k, r, t.
    """
    d = _geo.decompose_camera_matrix(p)
    return CameraMatrixDecomposition(k=d["k"], r=d["r"], t=d["t"])


# -- Triangulation -------------------------------------------------------------


def triangulate_point_linear(
    cameras: list[np.ndarray],
    points: list[tuple[float, float]],
) -> tuple[float, float, float]:
    """Linear DLT triangulation from multiple views.

    Parameters
    ----------
    cameras : list of ndarray, shape (3, 4)
        Camera projection matrices.
    points : list of (x, y)
        Corresponding 2-D image observations.

    Returns
    -------
    (x, y, z) : tuple of float
        Triangulated 3-D point.
    """
    return _geo.triangulate_point_linear(cameras, points)
