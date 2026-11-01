"""Calibrated cameras as objects: project, back-project and undistort numpy
arrays.

A :class:`CameraModel` is built from a camera's parameters — any result's
``camera`` / ``cameras[i]`` or a ``CameraParams`` payload — with any
projection, distortion and sensor model.

.. code-block:: python

    cam = CameraModel(result.camera)
    pixels = cam.project(points_cam)          # (N, 3) -> (N, 2)
    origins, directions = cam.unproject(pixels)
    ideal = cam.undistort(pixels)             # pinhole image, same K
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import _vision_calibration as _native
from ._generated_types import CameraParams
from .models import Camera

__all__ = ["CameraModel"]


def _rows(values: ArrayLike, dim: int, name: str) -> tuple[NDArray[np.float64], bool]:
    """``values`` as a contiguous ``(N, dim)`` float64 array, and whether it
    was a single ``(dim,)`` vector."""
    array = np.asarray(values, dtype=np.float64)
    single = array.ndim == 1
    if single:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[1] != dim:
        raise ValueError(f"{name} must have shape (N, {dim}) or ({dim},), got {np.shape(values)}")
    return np.ascontiguousarray(array), single


class CameraModel:
    """A camera model: projection, distortion, sensor and intrinsics.

    Coordinates follow the camera frame: ``z`` forward, pixels ``(u, v)``.
    Every method takes an ``(N, k)`` array (or one ``(k,)`` vector) and
    returns arrays of the same leading shape. Arrays of a thousand points or
    more are processed with the GIL released.

    Parameters
    ----------
    camera:
        A :class:`~vision_calibration.models.Camera` or a ``CameraParams``
        payload.

    Raises
    ------
    ValueError
        If the parameters are not finite or describe no valid camera.
    """

    __slots__ = ("_native",)

    def __init__(
        self, camera: Camera | CameraParams | Mapping[str, Any]
    ) -> None:
        if isinstance(camera, Camera):
            payload: Any = camera.to_payload()
        else:
            payload = dict(camera)
        self._native: Any = _native.CameraModel(payload)

    @property
    def params(self) -> CameraParams:
        """The ``CameraParams`` payload the model was built from."""
        return cast(CameraParams, self._native.params())

    def project(self, points_3d: ArrayLike) -> NDArray[np.float64]:
        """Project camera-frame points ``(N, 3)`` to pixels ``(N, 2)``.

        A point the camera cannot see (for a pinhole camera: not in front of
        it; for an equidistant one: straight behind it) projects to NaN.
        """
        points, single = _rows(points_3d, 3, "points_3d")
        pixels = cast(NDArray[np.float64], self._native.project(points))
        return pixels[0] if single else pixels

    def unproject(
        self, pixels: ArrayLike
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Back-project pixels ``(N, 2)`` to camera-frame rays.

        Returns ``(origins, directions)``, each ``(N, 3)``. The directions are
        not normalized: a pinhole camera's have ``z = 1``, so a ray's point at
        depth ``z`` is ``origin + z * direction``. An equidistant camera's
        are unit vectors, with ``z ≤ 0`` beyond 90° from the axis.
        """
        rows, single = _rows(pixels, 2, "pixels")
        origins, directions = cast(
            tuple[NDArray[np.float64], NDArray[np.float64]], self._native.unproject(rows)
        )
        return (origins[0], directions[0]) if single else (origins, directions)

    def undistort(self, pixels: ArrayLike) -> NDArray[np.float64]:
        """Map pixels ``(N, 2)`` to where they land in an ideal pinhole image
        with the same intrinsics: distortion and sensor tilt removed.

        A pixel whose ray does not reach the plane in front of the camera
        (a fisheye pixel beyond 90° from the axis) maps to NaN.
        """
        rows, single = _rows(pixels, 2, "pixels")
        out = cast(NDArray[np.float64], self._native.undistort(rows))
        return out[0] if single else out

    def __repr__(self) -> str:
        params = self.params
        return (
            f"CameraModel(projection={params['projection'].get('type')!r}, "
            f"distortion={params['distortion'].get('type')!r}, "
            f"sensor={params['sensor'].get('type')!r}, intrinsics={params['intrinsics']!r})"
        )
