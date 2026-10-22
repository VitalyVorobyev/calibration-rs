"""Planar intrinsics: one camera from views of a planar target.

:class:`PlanarIntrinsicsSession` runs the calibration step by step:
``step_init`` (or ``step_init_with_seed``) fits the intrinsics, distortion,
sensor tilt and per-view poses linearly; ``step_optimize`` refines them;
``step_filter`` drops outlying features. ``run_calibration`` is
``step_init`` + ``step_optimize``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

from . import _vision_calibration as _native
from .common import IntrinsicsInitOptions, IntrinsicsOptimizeOptions, _StepOptions
from .models import (
    BrownConradyDistortion,
    PinholeIntrinsics,
    PlanarIntrinsicsConfig,
    PlanarIntrinsicsExport,
    PlanarIntrinsicsInput,
    Pose,
    ScheimpflugSensor,
)
from .session import CalibrationSession, _payload, _poses, _seed_payload

__all__ = [
    "IntrinsicsFilterOptions",
    "IntrinsicsInitOptions",
    "IntrinsicsInitResult",
    "IntrinsicsManualInit",
    "IntrinsicsOptimizeOptions",
    "IntrinsicsOptimizeResult",
    "PlanarIntrinsicsSession",
]


@dataclass(slots=True)
class IntrinsicsFilterOptions(_StepOptions):
    """Options of ``step_filter``: which features to drop after a solve."""

    max_reproj_error: float = 2.0
    """Features reprojecting farther than this are removed, pixels."""
    min_points_per_view: int = 4
    """Fewest features a view keeps; at least 4."""
    remove_sparse_views: bool = True
    """Remove views left with fewer than ``min_points_per_view`` features;
    ``False`` makes such a view an error instead."""


@dataclass(slots=True)
class IntrinsicsManualInit:
    """Manual seeds for ``step_init_with_seed``; a field left ``None`` is
    estimated.

    A seeded ``intrinsics`` skips the linear intrinsics fit (distortion then
    starts at zero unless seeded too). The distortion seed is Brown-Conrady;
    for another distortion model its coefficients are embedded and the extra
    ones start at zero. ``sensor`` seeds the tilt of a Scheimpflug camera (an
    error for an untilted one); a seeded tilt is refined directly instead of
    swept. ``poses`` are camera-from-target, one per view.
    """

    intrinsics: PinholeIntrinsics | None = None
    distortion: BrownConradyDistortion | None = None
    sensor: ScheimpflugSensor | None = None
    poses: list[Pose] | None = None

    def to_payload(self) -> dict[str, Any]:
        return _seed_payload(
            {
                "intrinsics": self.intrinsics,
                "distortion": self.distortion,
                "sensor": self.sensor,
                "poses": self.poses,
            }
        )


@dataclass(frozen=True, slots=True)
class IntrinsicsInitResult:
    """The initial estimate of ``step_init`` / ``step_init_with_seed``."""

    intrinsics: PinholeIntrinsics
    distortion: BrownConradyDistortion
    sensor: ScheimpflugSensor | None
    """``None`` exactly when the camera has no Scheimpflug sensor."""
    poses: list[Pose]
    """Per-view ``camera_se3_target``."""

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> IntrinsicsInitResult:
        sensor = payload.get("sensor")
        return cls(
            intrinsics=PinholeIntrinsics.from_payload(cast(Mapping[str, Any], payload["intrinsics"])),
            distortion=BrownConradyDistortion.from_payload(
                cast(Mapping[str, Any], payload["distortion"])
            ),
            sensor=None
            if sensor is None
            else ScheimpflugSensor.from_payload(cast(Mapping[str, Any], sensor)),
            poses=_poses(payload["poses"]),
        )


@dataclass(frozen=True, slots=True)
class IntrinsicsOptimizeResult:
    """The solve metrics of ``step_optimize``."""

    final_cost: float
    mean_reproj_error: float
    """Mean reprojection error, pixels."""
    iterations: int

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> IntrinsicsOptimizeResult:
        return cls(
            final_cost=float(payload["final_cost"]),
            mean_reproj_error=float(payload["mean_reproj_error"]),
            iterations=int(payload["iterations"]),
        )


class PlanarIntrinsicsSession(
    CalibrationSession[PlanarIntrinsicsInput, PlanarIntrinsicsConfig, PlanarIntrinsicsExport]
):
    """A planar intrinsics calibration, step by step."""

    _native_type = _native.PlanarIntrinsicsSession
    _config_class = PlanarIntrinsicsConfig
    _export_class = PlanarIntrinsicsExport

    __slots__ = ()

    def step_init(self, options: IntrinsicsInitOptions | None = None) -> IntrinsicsInitResult:
        """Fit the initial intrinsics, distortion, tilt and poses linearly."""
        return IntrinsicsInitResult.from_payload(self._native.step_init(_payload(options)))

    def step_init_with_seed(
        self,
        seed: IntrinsicsManualInit,
        options: IntrinsicsInitOptions | None = None,
    ) -> IntrinsicsInitResult:
        """``step_init`` from manual seeds; the unseeded parts are estimated."""
        return IntrinsicsInitResult.from_payload(
            self._native.step_init_with_seed(_payload(seed), _payload(options))
        )

    def step_optimize(
        self, options: IntrinsicsOptimizeOptions | None = None
    ) -> IntrinsicsOptimizeResult:
        """Refine the initial estimate by non-linear least squares."""
        return IntrinsicsOptimizeResult.from_payload(
            self._native.step_optimize(_payload(options))
        )

    def step_filter(self, options: IntrinsicsFilterOptions | None = None) -> None:
        """Drop features (and sparse views) that reproject poorly after
        ``step_optimize``. The input changes and the estimate is cleared: run
        ``step_init`` and ``step_optimize`` again."""
        self._native.step_filter(_payload(options))

    def run_calibration_with_filtering(
        self, options: IntrinsicsFilterOptions | None = None
    ) -> None:
        """Calibrate, filter outliers, and calibrate again."""
        self._native.run_calibration_with_filtering(_payload(options))
