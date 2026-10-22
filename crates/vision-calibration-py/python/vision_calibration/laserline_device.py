"""Laserline device: one camera plus the laser plane it observes.

:class:`LaserlineDeviceSession` runs ``step_init`` (or
``step_init_with_seed``) — the camera, target poses and laser plane fitted
linearly — then ``step_optimize``, which refines them jointly.
``run_calibration`` runs both.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

from . import _vision_calibration as _native
from .common import _StepOptions
from .models import (
    BrownConradyDistortion,
    Camera,
    LaserlineDeviceConfig,
    LaserlineDeviceExport,
    LaserlineDeviceInput,
    LaserPlane,
    PinholeIntrinsics,
    Pose,
    ScheimpflugSensor,
)
from .session import CalibrationSession, _camera, _floats, _payload, _poses, _seed_payload

__all__ = [
    "DeviceInitOptions",
    "DeviceInitResult",
    "DeviceManualInit",
    "DeviceOptimizeOptions",
    "DeviceOptimizeResult",
    "LaserlineDeviceSession",
]


@dataclass(slots=True)
class DeviceInitOptions(_StepOptions):
    """Options of ``step_init``."""

    iterations: int | None = None
    """Iterations of the iterative intrinsics estimate."""


@dataclass(slots=True)
class DeviceOptimizeOptions(_StepOptions):
    """Options of ``step_optimize``."""

    max_iters: int | None = None
    """Maximum solver iterations."""
    verbosity: int | None = None
    """Solver verbosity."""


@dataclass(slots=True)
class DeviceManualInit:
    """Manual seeds of ``step_init_with_seed``; a field left ``None`` is
    estimated.

    The camera seeds mean what they mean for planar intrinsics; ``poses``
    are camera-from-target, one per view; ``plane`` is the laser plane in the
    camera frame and skips the linear plane fit.
    """

    intrinsics: PinholeIntrinsics | None = None
    distortion: BrownConradyDistortion | None = None
    sensor: ScheimpflugSensor | None = None
    poses: list[Pose] | None = None
    plane: LaserPlane | None = None

    def to_payload(self) -> dict[str, Any]:
        return _seed_payload(
            {
                "intrinsics": self.intrinsics,
                "distortion": self.distortion,
                "sensor": self.sensor,
                "poses": self.poses,
                "plane": self.plane,
            }
        )


@dataclass(frozen=True, slots=True)
class DeviceInitResult:
    """The initial camera, poses and laser plane."""

    camera: Camera
    poses: list[Pose]
    """Per-view camera-from-target."""
    plane: LaserPlane
    """Laser plane in the camera frame."""
    plane_rmse: float | None
    """RMS distance of the laser points to the fitted plane, metres; ``None``
    for a seeded plane."""

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> DeviceInitResult:
        rmse = payload.get("plane_rmse")
        return cls(
            camera=_camera(payload["camera"]),
            poses=_poses(payload["poses"]),
            plane=LaserPlane.from_payload(cast(Mapping[str, Any], payload["plane"])),
            plane_rmse=None if rmse is None else float(rmse),
        )


@dataclass(frozen=True, slots=True)
class DeviceOptimizeResult:
    """The joint refinement's metrics."""

    final_cost: float
    mean_reproj_error: float
    """Mean target reprojection error, pixels."""
    mean_laser_error: float
    """Mean laser residual, in the configured residual's unit."""
    per_view_reproj_errors: list[float]
    per_view_laser_errors: list[float]

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> DeviceOptimizeResult:
        return cls(
            final_cost=float(payload["final_cost"]),
            mean_reproj_error=float(payload["mean_reproj_error"]),
            mean_laser_error=float(payload["mean_laser_error"]),
            per_view_reproj_errors=_floats(payload["per_view_reproj_errors"]),
            per_view_laser_errors=_floats(payload["per_view_laser_errors"]),
        )


class LaserlineDeviceSession(
    CalibrationSession[LaserlineDeviceInput, LaserlineDeviceConfig, LaserlineDeviceExport]
):
    """A laserline device calibration, step by step."""

    _native_type = _native.LaserlineDeviceSession
    _config_class = LaserlineDeviceConfig
    _export_class = LaserlineDeviceExport

    __slots__ = ()

    def step_init(self, options: DeviceInitOptions | None = None) -> DeviceInitResult:
        """Fit the camera, target poses and laser plane linearly."""
        return DeviceInitResult.from_payload(self._native.step_init(_payload(options)))

    def step_init_with_seed(
        self,
        seed: DeviceManualInit,
        options: DeviceInitOptions | None = None,
    ) -> DeviceInitResult:
        """``step_init`` from manual seeds."""
        return DeviceInitResult.from_payload(
            self._native.step_init_with_seed(_payload(seed), _payload(options))
        )

    def step_optimize(self, options: DeviceOptimizeOptions | None = None) -> DeviceOptimizeResult:
        """Refine the camera, poses and plane jointly."""
        return DeviceOptimizeResult.from_payload(self._native.step_optimize(_payload(options)))
