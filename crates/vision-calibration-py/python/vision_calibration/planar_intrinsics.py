"""Planar intrinsics: one camera from views of a planar target.

:class:`PlanarIntrinsicsSession` runs the calibration step by step:
``step_init`` (or ``step_init_with_seed``) fits the camera and per-view
poses linearly; ``step_optimize`` refines them;
``step_filter`` drops outlying features. ``run_calibration`` is
``step_init`` + ``step_optimize``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from . import _vision_calibration as _native
from .common import (
    IntrinsicsInitOptions,
    IntrinsicsManualInit,
    IntrinsicsOptimizeOptions,
    _StepOptions,
)
from .models import (
    CameraParams,
    PlanarIntrinsicsConfig,
    PlanarIntrinsicsExport,
    PlanarIntrinsicsInput,
    Pose,
)
from .session import CalibrationSession, _run_calibration, _camera, _payload, _poses
from .types import ProgressCallback

__all__ = [
    "IntrinsicsFilterOptions",
    "IntrinsicsInitOptions",
    "IntrinsicsInitResult",
    "IntrinsicsManualInit",
    "IntrinsicsOptimizeOptions",
    "IntrinsicsOptimizeResult",
    "PlanarIntrinsicsSession",
    "run_calibration",
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


@dataclass(frozen=True, slots=True)
class IntrinsicsInitResult:
    """The initial estimate of ``step_init`` / ``step_init_with_seed``."""

    camera: CameraParams
    """The initial camera, in the configured model."""
    cam_se3_target: list[Pose]
    """Per-view ``cam_se3_target``."""

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> IntrinsicsInitResult:
        return cls(camera=_camera(payload["camera"]), cam_se3_target=_poses(payload["cam_se3_target"]))


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


def run_calibration(
    input: PlanarIntrinsicsInput,
    config: PlanarIntrinsicsConfig | None = None,
    *,
    progress: ProgressCallback | None = None,
) -> PlanarIntrinsicsExport:
    """Run the planar intrinsics calibration end to end: the session's ``run_calibration`` and
    ``export`` in one call.

    ``config`` defaults to the problem's defaults. ``progress``, when given,
    is called with each progress report (a
    :data:`~vision_calibration.types.Progress` mapping); raising from it
    stops the run and the exception propagates, as does Ctrl-C
    (:class:`KeyboardInterrupt`).

    Raises
    ------
    TypeError
        ``input`` or ``config`` is not the problem's dataclass.
    ValueError
        The input or config is invalid.
    RuntimeError
        The calibration failed.
    """
    return _run_calibration(PlanarIntrinsicsSession, input, config, progress, PlanarIntrinsicsInput, PlanarIntrinsicsConfig)
