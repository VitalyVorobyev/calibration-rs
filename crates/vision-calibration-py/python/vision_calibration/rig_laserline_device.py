"""Rig laserline device: one laser plane per camera of a calibrated rig.

:class:`RigLaserlineDeviceSession` fits the planes against a frozen upstream
rig calibration: ``step_init`` (or ``step_init_with_seed``) fits each plane
linearly, ``step_optimize`` refines them. ``run_calibration`` runs both.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from . import _vision_calibration as _native
from .common import _StepOptions
from .models import (
    LaserPlane,
    RigLaserlineDeviceConfig,
    RigLaserlineDeviceExport,
    RigLaserlineDeviceInput,
)
from .session import CalibrationSession, _payload, _seed_payload

__all__ = [
    "PlaneInitOptions",
    "PlaneManualInit",
    "PlaneOptimizeOptions",
    "RigLaserlineDeviceSession",
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

    planes_cam: list[LaserPlane] | None = None

    def to_payload(self) -> dict[str, Any]:
        return _seed_payload({"planes_cam": self.planes_cam})


class RigLaserlineDeviceSession(
    CalibrationSession[RigLaserlineDeviceInput, RigLaserlineDeviceConfig, RigLaserlineDeviceExport]
):
    """A rig laserline device calibration, step by step."""

    _native_type = _native.RigLaserlineDeviceSession
    _config_class = RigLaserlineDeviceConfig
    _export_class = RigLaserlineDeviceExport

    __slots__ = ()

    def step_init(self, options: PlaneInitOptions | None = None) -> None:
        """Fit each camera's laser plane linearly."""
        self._native.step_init(_payload(options))

    def step_init_with_seed(
        self,
        seed: PlaneManualInit,
        options: PlaneInitOptions | None = None,
    ) -> None:
        """``step_init`` from manual seeds."""
        self._native.step_init_with_seed(_payload(seed), _payload(options))

    def step_optimize(self, options: PlaneOptimizeOptions | None = None) -> None:
        """Refine the laser planes."""
        self._native.step_optimize(_payload(options))
