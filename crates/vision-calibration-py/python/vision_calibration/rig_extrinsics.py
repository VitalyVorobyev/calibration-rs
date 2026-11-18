"""Rig extrinsics: several cameras observing a common planar target.

:class:`RigExtrinsicsSession` runs the calibration in two stages: each
camera's intrinsics (``step_intrinsics_init_all``,
``step_intrinsics_optimize_all``), then the rig — each camera's pose in the
rig frame (``step_rig_init``, ``step_rig_optimize``). ``run_calibration``
runs all four.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from . import _vision_calibration as _native
from .common import (
    IntrinsicsInitOptions,
    IntrinsicsOptimizeOptions,
    RigInitOptions,
    RigInitResult,
    RigIntrinsicsInitResult,
    RigIntrinsicsManualInit,
    RigManualInit,
    RigOptimizeOptions,
    RigPoses,
)
from .models import (
    CameraParams,
    RigExtrinsicsConfig,
    RigExtrinsicsExport,
    RigExtrinsicsInput,
)
from .session import (
    _run_calibration,
    CalibrationSession,
    _cameras,
    _floats,
    _payload,
    _termination,
    _terminations,
)
from .types import ProgressCallback, SolveTermination

__all__ = [
    "IntrinsicsInitOptions",
    "IntrinsicsOptimizeOptions",
    "IntrinsicsOptimizeResult",
    "RigExtrinsicsSession",
    "RigInitOptions",
    "RigInitResult",
    "RigIntrinsicsInitResult",
    "RigIntrinsicsManualInit",
    "RigManualInit",
    "RigOptimizeOptions",
    "RigOptimizeResult",
    "RigPoses",
    "run_calibration",
]


@dataclass(frozen=True, slots=True)
class IntrinsicsOptimizeResult:
    """The refined cameras of the intrinsics stage."""

    cameras: list[CameraParams]
    per_cam_reproj_errors: list[float]
    """Mean reprojection error per camera, pixels."""
    per_cam_terminations: list[SolveTermination | None]
    """Per camera, why its solver stopped; ``None`` when that is unknown."""

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> IntrinsicsOptimizeResult:
        return cls(
            cameras=_cameras(payload["cameras"]),
            per_cam_reproj_errors=_floats(payload["per_cam_reproj_errors"]),
            per_cam_terminations=_terminations(payload.get("per_cam_terminations")),
        )


@dataclass(frozen=True, slots=True)
class RigOptimizeResult:
    """The rig bundle adjustment's result."""

    mean_reproj_error: float
    """Mean reprojection error over the rig, pixels."""
    per_cam_reproj_errors: list[float]
    cameras: list[CameraParams]
    termination: SolveTermination | None
    """Why the solver stopped; ``None`` when that is unknown."""

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> RigOptimizeResult:
        return cls(
            mean_reproj_error=float(payload["mean_reproj_error"]),
            per_cam_reproj_errors=_floats(payload["per_cam_reproj_errors"]),
            cameras=_cameras(payload["cameras"]),
            termination=_termination(payload.get("termination")),
        )


class RigExtrinsicsSession(
    CalibrationSession[RigExtrinsicsInput, RigExtrinsicsConfig, RigExtrinsicsExport]
):
    """A rig extrinsics calibration, step by step."""

    _native_type = _native.RigExtrinsicsSession
    _config_class = RigExtrinsicsConfig
    _export_class = RigExtrinsicsExport

    __slots__ = ()

    def step_intrinsics_init_all(
        self, options: IntrinsicsInitOptions | None = None
    ) -> RigIntrinsicsInitResult:
        """Fit every camera's initial intrinsics and target poses linearly."""
        return RigIntrinsicsInitResult.from_payload(
            self._native.step_intrinsics_init_all(_payload(options))
        )

    def step_intrinsics_init_all_with_seed(
        self,
        seed: RigIntrinsicsManualInit,
        options: IntrinsicsInitOptions | None = None,
    ) -> RigIntrinsicsInitResult:
        """``step_intrinsics_init_all`` from manual seeds."""
        return RigIntrinsicsInitResult.from_payload(
            self._native.step_intrinsics_init_all_with_seed(_payload(seed), _payload(options))
        )

    def step_intrinsics_optimize_all(
        self, options: IntrinsicsOptimizeOptions | None = None
    ) -> IntrinsicsOptimizeResult:
        """Refine each camera on its own views."""
        return IntrinsicsOptimizeResult.from_payload(
            self._native.step_intrinsics_optimize_all(_payload(options))
        )

    def step_rig_init(self, options: RigInitOptions | None = None) -> RigInitResult:
        """Fit the camera-from-rig poses linearly."""
        return RigInitResult.from_payload(self._native.step_rig_init(_payload(options)))

    def step_rig_init_with_seed(
        self,
        seed: RigManualInit,
        options: RigInitOptions | None = None,
    ) -> RigInitResult:
        """``step_rig_init`` from manual seeds."""
        return RigInitResult.from_payload(
            self._native.step_rig_init_with_seed(_payload(seed), _payload(options))
        )

    def step_rig_optimize(self, options: RigOptimizeOptions | None = None) -> RigOptimizeResult:
        """Bundle-adjust the rig."""
        return RigOptimizeResult.from_payload(self._native.step_rig_optimize(_payload(options)))


def run_calibration(
    input: RigExtrinsicsInput,
    config: RigExtrinsicsConfig | None = None,
    *,
    progress: ProgressCallback | None = None,
) -> RigExtrinsicsExport:
    """Run the rig extrinsics calibration end to end: the session's ``run_calibration`` and
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
    return _run_calibration(RigExtrinsicsSession, input, config, progress, RigExtrinsicsInput, RigExtrinsicsConfig)
