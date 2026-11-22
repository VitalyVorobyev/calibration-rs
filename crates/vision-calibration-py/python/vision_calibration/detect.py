"""Detected target features and the detection cache.

A detection is a :class:`Features` table: the image points of the target's
features and their positions on the target. :func:`detect_target
<vision_calibration.dataset_runner.detect_target>` produces one from a numpy
image; :class:`FsDetectionCache` keeps them on disk so a manifest run skips
images it has already seen.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from . import _vision_calibration as _native
from .models import Observation, PointBlur, PointCovariance

__all__ = [
    "CacheKey",
    "Features",
    "FsDetectionCache",
    "reject_ambiguous_detection",
]


@dataclass(frozen=True, slots=True)
class Features:
    """Detected features: one row per ``Feature`` record.

    ``image_xy`` holds pixels ``(x, y)`` = (column, row); ``world_xyz`` the
    matching target points in metres, on the target's ``z = 0`` plane.
    ``cov_px`` holds each corner's position covariance ``(xx, xy, yy)`` in px²
    when the detector was asked for it (the manifest's
    ``detector.chess_corners.covariance``), else it is ``None``.

    ``blur_psf_px`` and ``blur_cov`` hold the blur the detector measured at
    each feature (chessboard and PuzzleBoard corners, when the target asks
    for it with ``measure_blur``): the PSF
    covariance ``(xx, xy, yy)`` in px², including the pixel's area
    integration, and the 3×3 covariance of those three elements in px⁴ (see
    :class:`PointBlur`). They are given together, ``None`` when no feature
    has a blur, and a feature without one is a row of NaN in both.
    """

    image_xy: NDArray[np.float64]
    """``(N, 2)`` pixels."""
    world_xyz: NDArray[np.float64]
    """``(N, 3)`` target points, metres."""
    cov_px: NDArray[np.float64] | None = None
    """``(N, 3)`` corner covariances ``(xx, xy, yy)``, px², or ``None``."""
    blur_psf_px: NDArray[np.float64] | None = None
    """``(N, 3)`` PSF covariances ``(xx, xy, yy)``, px², NaN rows for features
    without a blur, or ``None``."""
    blur_cov: NDArray[np.float64] | None = None
    """``(N, 3, 3)`` covariances of ``blur_psf_px``, px⁴, NaN for features
    without a blur, or ``None``."""

    def __post_init__(self) -> None:
        image_xy = np.ascontiguousarray(np.asarray(self.image_xy, dtype=np.float64).reshape(-1, 2))
        world_xyz = np.ascontiguousarray(np.asarray(self.world_xyz, dtype=np.float64).reshape(-1, 3))
        if image_xy.shape[0] != world_xyz.shape[0]:
            raise ValueError(
                f"image_xy and world_xyz need the same length, got {image_xy.shape[0]} "
                f"and {world_xyz.shape[0]}"
            )
        if self.cov_px is not None:
            cov_px = np.ascontiguousarray(np.asarray(self.cov_px, dtype=np.float64).reshape(-1, 3))
            if cov_px.shape[0] != image_xy.shape[0]:
                raise ValueError(
                    f"cov_px needs one row per feature, got {cov_px.shape[0]} and {image_xy.shape[0]}"
                )
            object.__setattr__(self, "cov_px", cov_px)
        if (self.blur_psf_px is None) != (self.blur_cov is None):
            raise ValueError("blur_psf_px and blur_cov are given together or not at all")
        if self.blur_psf_px is not None and self.blur_cov is not None:
            psf = np.ascontiguousarray(np.asarray(self.blur_psf_px, dtype=np.float64).reshape(-1, 3))
            cov = np.ascontiguousarray(np.asarray(self.blur_cov, dtype=np.float64).reshape(-1, 3, 3))
            if psf.shape[0] != image_xy.shape[0] or cov.shape[0] != image_xy.shape[0]:
                raise ValueError(
                    f"blur_psf_px and blur_cov need one row per feature, got {psf.shape[0]}, "
                    f"{cov.shape[0]} and {image_xy.shape[0]}"
                )
            missing_psf = np.isnan(psf).all(axis=1)
            if not np.array_equal(missing_psf, np.isnan(cov).all(axis=(1, 2))):
                raise ValueError("a feature without a blur is a row of NaN in both blur arrays")
            object.__setattr__(self, "blur_psf_px", psf)
            object.__setattr__(self, "blur_cov", cov)
        object.__setattr__(self, "image_xy", image_xy)
        object.__setattr__(self, "world_xyz", world_xyz)

    def __len__(self) -> int:
        return int(self.image_xy.shape[0])

    def to_observation(self) -> Observation:
        """The features as one view's 2D-3D correspondences, for a session's
        input, with their corner covariances when ``cov_px`` is set and their
        measured blur when the blur arrays are."""
        cov_px = (
            None
            if self.cov_px is None
            else [PointCovariance(float(xx), float(xy), float(yy)) for xx, xy, yy in self.cov_px]
        )
        blur: list[PointBlur | None] | None = None
        if self.blur_psf_px is not None and self.blur_cov is not None:
            blur = [
                None
                if np.isnan(psf).all()
                else PointBlur(
                    psf_px=(float(psf[0]), float(psf[1]), float(psf[2])),
                    cov=cast(
                        tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]],
                        tuple(tuple(float(v) for v in row) for row in cov),
                    ),
                )
                for psf, cov in zip(self.blur_psf_px, self.blur_cov, strict=True)
            ]
        return Observation(
            points_3d=[(float(x), float(y), float(z)) for x, y, z in self.world_xyz],
            points_2d=[(float(u), float(v)) for u, v in self.image_xy],
            cov_px=cov_px,
            blur=blur,
        )

    def _blur_args(self) -> tuple[NDArray[np.float64] | None, NDArray[np.float64] | None]:
        """The blur arrays as the extension takes them: ``(N, 3)`` and ``(N, 9)``."""
        if self.blur_psf_px is None or self.blur_cov is None:
            return None, None
        return self.blur_psf_px, np.ascontiguousarray(self.blur_cov.reshape(-1, 9))


def _features(arrays: Any) -> Features:
    image_xy, world_xyz, cov_px, blur_psf_px, blur_cov = cast(
        tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64] | None,
            NDArray[np.float64] | None,
            NDArray[np.float64] | None,
        ],
        arrays,
    )
    return Features(
        image_xy=image_xy,
        world_xyz=world_xyz,
        cov_px=cov_px,
        blur_psf_px=blur_psf_px,
        blur_cov=None if blur_cov is None else blur_cov.reshape(-1, 3, 3),
    )


def reject_ambiguous_detection(features: Features) -> Features:
    """The detection unchanged, or empty if two features share an image point.

    A planar target maps to the image one to one, so a repeated image point
    means the detector lost track of its grid; none of the detection's labels
    are then trusted.
    """
    return _features(
        _native.reject_ambiguous_detection(
            features.image_xy, features.world_xyz, features.cov_px, *features._blur_args()
        )
    )


@dataclass(frozen=True, slots=True)
class CacheKey:
    """The key of a cached detection: the image's content hash, the detector
    and the hash of its config."""

    image_hash: int
    detector: str
    config_hash: int

    @classmethod
    def from_inputs(cls, image_bytes: bytes, detector: str, config: Mapping[str, Any]) -> CacheKey:
        """The key of an encoded image file's bytes under a detector name and
        its config (key order of the config does not matter).

        For entries you manage with :meth:`FsDetectionCache.put`; the
        manifest runner's own keys also fold in the detector's release, so a
        release with different detections never reads stale entries.
        """
        return cls.from_payload(
            cast(Mapping[str, Any], _native.cache_key_from_inputs(image_bytes, detector, dict(config)))
        )

    def file_name(self) -> str:
        """The name of the key's entry file in a :class:`FsDetectionCache`."""
        return cast(str, _native.cache_key_file_name(self.to_payload()))

    def to_payload(self) -> dict[str, Any]:
        return {
            "image_hash": self.image_hash,
            "detector": self.detector,
            "config_hash": self.config_hash,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> CacheKey:
        return cls(
            image_hash=int(payload["image_hash"]),
            detector=str(payload["detector"]),
            config_hash=int(payload["config_hash"]),
        )


class FsDetectionCache:
    """A detection cache on disk: one JSON file per entry under ``root``.

    The directory is created on the first write; reading a missing directory
    is a miss. Pass it as ``cache=`` to the manifest runs. I/O failures raise
    :class:`RuntimeError`.
    """

    __slots__ = ("_native",)

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self._native: Any = _native.FsDetectionCache(os.fspath(root))

    @property
    def root(self) -> Path:
        return Path(cast(str, self._native.root))

    def get(self, key: CacheKey) -> Features | None:
        """The features cached under ``key``, or ``None``."""
        arrays = self._native.get(key.to_payload())
        return None if arrays is None else _features(arrays)

    def put(self, key: CacheKey, features: Features) -> None:
        """Store ``features`` under ``key``."""
        self._native.put(
            key.to_payload(), features.image_xy, features.world_xyz, features.cov_px, *features._blur_args()
        )

    def __repr__(self) -> str:
        return f"FsDetectionCache({str(self.root)!r})"

