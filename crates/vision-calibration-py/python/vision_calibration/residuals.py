"""Per-feature residuals of a calibration result, as numpy tables.

Every result's ``per_feature_residuals`` is a :class:`PerFeatureResiduals`:
one row per target corner (:class:`TargetFeatureResiduals`) and per laser
pixel (:class:`LaserFeatureResiduals`), in view-major order, plus the
projected laser line of each view and per-camera histograms. Each table
holds the records of the Rust type it is named after, one column per field;
a value the record leaves unset (a point that did not project) is NaN.

.. code-block:: python

    res = result.per_feature_residuals
    worst = res.target.error_px.argmax()
    print(res.target.pose[worst], res.target.observed_px[worst])
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "FeatureResidualHistogram",
    "LaserFeatureResiduals",
    "LaserLineTrace",
    "PerFeatureResiduals",
    "TargetFeatureResiduals",
]


def _ints(records: Sequence[Mapping[str, Any]], key: str) -> NDArray[np.int64]:
    return np.fromiter((int(r[key]) for r in records), dtype=np.int64, count=len(records))


def _floats(records: Sequence[Mapping[str, Any]], key: str) -> NDArray[np.float64]:
    """A column of optional floats; a missing or null value is NaN."""
    values = (r.get(key) for r in records)
    return np.fromiter(
        (np.nan if v is None else float(v) for v in values), dtype=np.float64, count=len(records)
    )


def _points(records: Sequence[Mapping[str, Any]], key: str, dim: int) -> NDArray[np.float64]:
    """A column of optional ``dim``-vectors as an ``(N, dim)`` array; a missing
    or null value is a row of NaN."""
    out = np.full((len(records), dim), np.nan, dtype=np.float64)
    for i, record in enumerate(records):
        value = record.get(key)
        if value is not None:
            out[i] = value
    return out


@dataclass(frozen=True, slots=True)
class TargetFeatureResiduals:
    """The reprojection of every target corner: one row per
    ``TargetFeatureResidual`` record, view-major."""

    pose: NDArray[np.int64]
    """``(N,)`` view index in the input."""
    camera: NDArray[np.int64]
    """``(N,)`` camera index; 0 for a single camera."""
    feature: NDArray[np.int64]
    """``(N,)`` index of the feature in its view's points."""
    target_xyz_m: NDArray[np.float64]
    """``(N, 3)`` target point, metres, target frame."""
    observed_px: NDArray[np.float64]
    """``(N, 2)`` observed pixel."""
    projected_px: NDArray[np.float64]
    """``(N, 2)`` projected pixel; NaN where the point did not project."""
    error_px: NDArray[np.float64]
    """``(N,)`` distance between the two, pixels; NaN where the point did
    not project."""

    def __len__(self) -> int:
        return int(self.pose.shape[0])

    @classmethod
    def from_payload(cls, records: Sequence[Mapping[str, Any]]) -> TargetFeatureResiduals:
        return cls(
            pose=_ints(records, "pose"),
            camera=_ints(records, "camera"),
            feature=_ints(records, "feature"),
            target_xyz_m=_points(records, "target_xyz_m", 3),
            observed_px=_points(records, "observed_px", 2),
            projected_px=_points(records, "projected_px", 2),
            error_px=_floats(records, "error_px"),
        )


@dataclass(frozen=True, slots=True)
class LaserFeatureResiduals:
    """The residual of every laser pixel: one row per ``LaserFeatureResidual``
    record, view-major."""

    pose: NDArray[np.int64]
    """``(N,)`` view index in the input."""
    camera: NDArray[np.int64]
    """``(N,)`` camera index; 0 for a single camera."""
    feature: NDArray[np.int64]
    """``(N,)`` index of the pixel in its view's laser pixels."""
    observed_px: NDArray[np.float64]
    """``(N, 2)`` observed laser pixel."""
    residual_m: NDArray[np.float64]
    """``(N,)`` distance from the pixel's back-projected ray to the laser
    plane, metres; NaN where the ray misses the plane."""
    residual_px: NDArray[np.float64]
    """``(N,)`` distance from the pixel to the projected laser line, pixels;
    NaN where the laser and target planes share no line."""

    def __len__(self) -> int:
        return int(self.pose.shape[0])

    @classmethod
    def from_payload(cls, records: Sequence[Mapping[str, Any]]) -> LaserFeatureResiduals:
        return cls(
            pose=_ints(records, "pose"),
            camera=_ints(records, "camera"),
            feature=_ints(records, "feature"),
            observed_px=_points(records, "observed_px", 2),
            residual_m=_floats(records, "residual_m"),
            residual_px=_floats(records, "residual_px"),
        )


@dataclass(frozen=True, slots=True)
class LaserLineTrace:
    """The calibrated laser line in one view's image."""

    pose: int
    camera: int
    points_px: NDArray[np.float64]
    """``(M, 2)`` points along the line, pixels, spanning the view's laser
    pixels."""

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> LaserLineTrace:
        points = np.asarray(payload.get("points_px", []), dtype=np.float64).reshape(-1, 2)
        return cls(pose=int(payload["pose"]), camera=int(payload["camera"]), points_px=points)


@dataclass(frozen=True, slots=True)
class FeatureResidualHistogram:
    """Error counts in the buckets ``<=1, <=2, <=5, <=10, >10`` pixels."""

    bucket_edges_px: tuple[float, float, float, float]
    counts: tuple[int, int, int, int, int]
    count: int
    mean: float
    """Mean error, pixels; 0 when ``count`` is 0."""
    max: float
    """Largest error, pixels; 0 when ``count`` is 0."""

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> FeatureResidualHistogram:
        edges = [float(v) for v in cast(Sequence[Any], payload["bucket_edges_px"])]
        counts = [int(v) for v in cast(Sequence[Any], payload["counts"])]
        return cls(
            bucket_edges_px=(edges[0], edges[1], edges[2], edges[3]),
            counts=(counts[0], counts[1], counts[2], counts[3], counts[4]),
            count=int(payload["count"]),
            mean=float(payload["mean"]),
            max=float(payload["max"]),
        )


def _histograms(value: Any) -> list[FeatureResidualHistogram] | None:
    if value is None:
        return None
    return [
        FeatureResidualHistogram.from_payload(cast(Mapping[str, Any], h))
        for h in cast(Sequence[Any], value)
    ]


@dataclass(frozen=True, slots=True)
class PerFeatureResiduals:
    """Every feature's residual in a calibration result.

    A problem without laser data has empty laser tables; a histogram list is
    ``None`` when the problem type produces no per-camera aggregate.
    """

    target: TargetFeatureResiduals
    laser: LaserFeatureResiduals
    laser_lines: list[LaserLineTrace]
    target_hist_per_camera: list[FeatureResidualHistogram] | None
    laser_hist_per_camera: list[FeatureResidualHistogram] | None

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any] | None) -> PerFeatureResiduals:
        payload = payload or {}
        return cls(
            target=TargetFeatureResiduals.from_payload(
                cast(Sequence[Mapping[str, Any]], payload.get("target", []))
            ),
            laser=LaserFeatureResiduals.from_payload(
                cast(Sequence[Mapping[str, Any]], payload.get("laser", []))
            ),
            laser_lines=[
                LaserLineTrace.from_payload(cast(Mapping[str, Any], t))
                for t in cast(Sequence[Any], payload.get("laser_lines", []))
            ],
            target_hist_per_camera=_histograms(payload.get("target_hist_per_camera")),
            laser_hist_per_camera=_histograms(payload.get("laser_hist_per_camera")),
        )
