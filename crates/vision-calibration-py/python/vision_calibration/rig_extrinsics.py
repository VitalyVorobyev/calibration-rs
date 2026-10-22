"""Rig extrinsics: several cameras observing a common planar target.

:class:`RigExtrinsicsSession` runs the calibration in two stages: each
camera's intrinsics (``step_intrinsics_init_all``,
``step_intrinsics_optimize_all``), then the rig — each camera's pose in the
rig frame (``step_rig_init``, ``step_rig_optimize``). ``run_calibration``
runs all four.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from . import _vision_calibration as _native
from .common import (
    IntrinsicsInitOptions,
    IntrinsicsOptimizeOptions,
    RigInitOptions,
    RigOptimizeOptions,
)
from .models import (
    BrownConradyDistortion,
    Camera,
    PinholeIntrinsics,
    Pose,
    RigExtrinsicsConfig,
    RigExtrinsicsExport,
    RigExtrinsicsInput,
    ScheimpflugSensor,
)
from .session import (
    CalibrationSession,
    _cameras,
    _floats,
    _optional_pose,
    _payload,
    _poses,
    _seed_payload,
)

__all__ = [
    "IntrinsicsInitOptions",
    "IntrinsicsInitResult",
    "IntrinsicsManualInit",
    "IntrinsicsOptimizeOptions",
    "IntrinsicsOptimizeResult",
    "RigExtrinsicsSession",
    "RigInitOptions",
    "RigInitResult",
    "RigManualInit",
    "RigOptimizeOptions",
    "RigOptimizeResult",
]


@dataclass(slots=True)
class IntrinsicsManualInit:
    """Manual seeds of the per-camera intrinsics stage, one entry per camera;
    a field left ``None`` is estimated.

    Seeded intrinsics skip the per-camera linear fit (distortion then starts
    at zero unless seeded too). The distortion seed is Brown-Conrady,
    embedded into each camera's model. ``per_cam_sensors`` seeds the tilt of
    each Scheimpflug camera; an untilted camera takes only a zero tilt.
    """

    per_cam_intrinsics: list[PinholeIntrinsics] | None = None
    per_cam_distortion: list[BrownConradyDistortion] | None = None
    per_cam_sensors: list[ScheimpflugSensor] | None = None

    def to_payload(self) -> dict[str, Any]:
        return _seed_payload(
            {
                "per_cam_intrinsics": self.per_cam_intrinsics,
                "per_cam_distortion": self.per_cam_distortion,
                "per_cam_sensors": self.per_cam_sensors,
            }
        )


@dataclass(slots=True)
class RigManualInit:
    """Manual seeds of the rig stage: set both or neither.

    ``cam_se3_rig`` is each camera's camera-from-rig pose; ``rig_se3_target``
    each view's rig-from-target pose.
    """

    cam_se3_rig: list[Pose] | None = None
    rig_se3_target: list[Pose] | None = None

    def to_payload(self) -> dict[str, Any]:
        return _seed_payload(
            {"cam_se3_rig": self.cam_se3_rig, "rig_se3_target": self.rig_se3_target}
        )


def _target_poses(values: Any) -> list[list[Pose | None]]:
    return [
        [_optional_pose(p) for p in cast(Sequence[Any], row)]
        for row in cast(Sequence[Any], values)
    ]


@dataclass(frozen=True, slots=True)
class IntrinsicsInitResult:
    """The initial per-camera estimate."""

    cameras: list[Camera]
    per_cam_target_poses: list[list[Pose | None]]
    """``[view][camera]`` camera-from-target; ``None`` where the camera did
    not see the target."""

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> IntrinsicsInitResult:
        return cls(
            cameras=_cameras(payload["cameras"]),
            per_cam_target_poses=_target_poses(payload["per_cam_target_poses"]),
        )


@dataclass(frozen=True, slots=True)
class IntrinsicsOptimizeResult:
    """The refined cameras of the intrinsics stage."""

    cameras: list[Camera]
    per_cam_reproj_errors: list[float]
    """Mean reprojection error per camera, pixels."""

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> IntrinsicsOptimizeResult:
        return cls(
            cameras=_cameras(payload["cameras"]),
            per_cam_reproj_errors=_floats(payload["per_cam_reproj_errors"]),
        )


@dataclass(frozen=True, slots=True)
class RigInitResult:
    """The linear rig estimate."""

    initial_cam_se3_rig: list[Pose]
    """Camera-from-rig per camera; the reference camera's is the identity."""
    initial_rig_se3_target: list[Pose]
    """Rig-from-target per view."""

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> RigInitResult:
        return cls(
            initial_cam_se3_rig=_poses(payload["initial_cam_se3_rig"]),
            initial_rig_se3_target=_poses(payload["initial_rig_se3_target"]),
        )


@dataclass(frozen=True, slots=True)
class RigOptimizeResult:
    """The rig bundle adjustment's result."""

    mean_reproj_error: float
    """Mean reprojection error over the rig, pixels."""
    per_cam_reproj_errors: list[float]
    cameras: list[Camera]

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> RigOptimizeResult:
        return cls(
            mean_reproj_error=float(payload["mean_reproj_error"]),
            per_cam_reproj_errors=_floats(payload["per_cam_reproj_errors"]),
            cameras=_cameras(payload["cameras"]),
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
    ) -> IntrinsicsInitResult:
        """Fit every camera's initial intrinsics and target poses linearly."""
        return IntrinsicsInitResult.from_payload(
            self._native.step_intrinsics_init_all(_payload(options))
        )

    def step_intrinsics_init_all_with_seed(
        self,
        seed: IntrinsicsManualInit,
        options: IntrinsicsInitOptions | None = None,
    ) -> IntrinsicsInitResult:
        """``step_intrinsics_init_all`` from manual seeds."""
        return IntrinsicsInitResult.from_payload(
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
