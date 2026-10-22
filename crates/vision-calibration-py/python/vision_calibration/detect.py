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
from .models import Observation

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
    """

    image_xy: NDArray[np.float64]
    """``(N, 2)`` pixels."""
    world_xyz: NDArray[np.float64]
    """``(N, 3)`` target points, metres."""

    def __post_init__(self) -> None:
        image_xy = np.ascontiguousarray(np.asarray(self.image_xy, dtype=np.float64).reshape(-1, 2))
        world_xyz = np.ascontiguousarray(np.asarray(self.world_xyz, dtype=np.float64).reshape(-1, 3))
        if image_xy.shape[0] != world_xyz.shape[0]:
            raise ValueError(
                f"image_xy and world_xyz need the same length, got {image_xy.shape[0]} "
                f"and {world_xyz.shape[0]}"
            )
        object.__setattr__(self, "image_xy", image_xy)
        object.__setattr__(self, "world_xyz", world_xyz)

    def __len__(self) -> int:
        return int(self.image_xy.shape[0])

    def to_observation(self) -> Observation:
        """The features as one view's 2D-3D correspondences, for a session's
        input."""
        return Observation(
            points_3d=[(float(x), float(y), float(z)) for x, y, z in self.world_xyz],
            points_2d=[(float(u), float(v)) for u, v in self.image_xy],
        )


def _features(arrays: Any) -> Features:
    image_xy, world_xyz = cast(tuple[NDArray[np.float64], NDArray[np.float64]], arrays)
    return Features(image_xy=image_xy, world_xyz=world_xyz)


def reject_ambiguous_detection(features: Features) -> Features:
    """The detection unchanged, or empty if two features share an image point.

    A planar target maps to the image one to one, so a repeated image point
    means the detector lost track of its grid; none of the detection's labels
    are then trusted.
    """
    return _features(_native.reject_ambiguous_detection(features.image_xy, features.world_xyz))


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
        self._native.put(key.to_payload(), features.image_xy, features.world_xyz)

    def __repr__(self) -> str:
        return f"FsDetectionCache({str(self.root)!r})"

