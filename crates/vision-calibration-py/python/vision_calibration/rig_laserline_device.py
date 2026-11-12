"""Rig laserline device: one laser plane per camera of a calibrated rig.

:class:`RigLaserlineDeviceSession` fits the planes against a frozen upstream
rig calibration: ``step_init`` (or ``step_init_with_seed``) fits each plane
linearly, ``step_optimize`` refines them. ``run_calibration`` runs both.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

from . import _vision_calibration as _native
from .common import _StepOptions
from .models import (
    LaserlineStats,
    LaserPlane,
    RigLaserlineDeviceConfig,
    RigLaserlineDeviceExport,
    RigLaserlineDeviceInput,
)
from .session import CalibrationSession, _run_calibration, _payload, _seed_payload
from .types import ProgressCallback

__all__ = [
    "PlaneInitOptions",
    "PlaneInitResult",
    "PlaneManualInit",
    "PlaneOptimizeOptions",
    "PlaneOptimizeResult",
    "RigLaserlineDeviceSession",
    "run_calibration",
]


@dataclass(slots=True)
class PlaneInitOptions(_StepOptions):
    """Options of ``step_init`` (the linear plane fit has none)."""


@dataclass(slots=True)
class PlaneOptimizeOptions(_StepOptions):
    """Options of ``step_optimize``."""

    max_iters: int | None = None
    """Maximum solver iterations."""
    verbosity: int | None = None
    """Solver verbosity."""


@dataclass(slots=True)
class PlaneManualInit:
    """Manual seeds of ``step_init_with_seed``: one laser plane per camera,
    in that camera's frame. Seeds skip the linear plane fit and override the
    input's initial planes."""

    laser_planes_cam: list[LaserPlane] | None = None

    def to_payload(self) -> dict[str, Any]:
        return _seed_payload({"laser_planes_cam": self.laser_planes_cam})


@dataclass(frozen=True, slots=True)
class PlaneInitResult:
    """The initial laser planes."""

    laser_planes_cam: list[LaserPlane]
    """One plane per camera, in that camera's frame."""
    plane_rmse: list[float | None]
    """Per camera, the RMS distance of the laser points to the fitted plane,
    metres; ``None`` for a seeded plane."""

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> PlaneInitResult:
        return cls(
            laser_planes_cam=_planes(payload["laser_planes_cam"]),
            plane_rmse=[None if v is None else float(v) for v in payload["plane_rmse"]],
        )


@dataclass(frozen=True, slots=True)
class PlaneOptimizeResult:
    """The refined laser planes and their per-camera statistics."""

    laser_planes_rig: list[LaserPlane]
    """One plane per camera, in the rig frame."""
    laser_planes_cam: list[LaserPlane]
    """One plane per camera, in that camera's frame."""
    mean_reproj_error: float
    """Mean target reprojection error over the cameras, pixels."""
    per_cam_stats: list[LaserlineStats]

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> PlaneOptimizeResult:
        return cls(
            laser_planes_rig=_planes(payload["laser_planes_rig"]),
            laser_planes_cam=_planes(payload["laser_planes_cam"]),
            mean_reproj_error=float(payload["mean_reproj_error"]),
            per_cam_stats=[
                LaserlineStats.from_payload(cast(Mapping[str, Any], v))
                for v in payload["per_cam_stats"]
            ],
        )


def _planes(values: Any) -> list[LaserPlane]:
    return [LaserPlane.from_payload(cast(Mapping[str, Any], v)) for v in values]


class RigLaserlineDeviceSession(
    CalibrationSession[RigLaserlineDeviceInput, RigLaserlineDeviceConfig, RigLaserlineDeviceExport]
):
    """A rig laserline device calibration, step by step."""

    _native_type = _native.RigLaserlineDeviceSession
    _config_class = RigLaserlineDeviceConfig
    _export_class = RigLaserlineDeviceExport

    __slots__ = ()

    def step_init(self, options: PlaneInitOptions | None = None) -> PlaneInitResult:
        """Fit each camera's laser plane linearly."""
        return PlaneInitResult.from_payload(self._native.step_init(_payload(options)))

    def step_init_with_seed(
        self,
        seed: PlaneManualInit,
        options: PlaneInitOptions | None = None,
    ) -> PlaneInitResult:
        """``step_init`` from manual seeds."""
        return PlaneInitResult.from_payload(
            self._native.step_init_with_seed(_payload(seed), _payload(options))
        )

    def step_optimize(self, options: PlaneOptimizeOptions | None = None) -> PlaneOptimizeResult:
        """Refine the laser planes."""
        return PlaneOptimizeResult.from_payload(self._native.step_optimize(_payload(options)))


def run_calibration(
    input: RigLaserlineDeviceInput,
    config: RigLaserlineDeviceConfig | None = None,
    *,
    progress: ProgressCallback | None = None,
) -> RigLaserlineDeviceExport:
    """Run the rig laserline device calibration end to end: the session's ``run_calibration`` and
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
    return _run_calibration(RigLaserlineDeviceSession, input, config, progress, RigLaserlineDeviceInput, RigLaserlineDeviceConfig)
