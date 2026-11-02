"""Single-camera hand-eye: a camera on (or watching) a robot.

:class:`SingleCamHandeyeSession` runs the calibration in two stages, each an
init and an optimize step: the camera's intrinsics
(``step_intrinsics_init``, ``step_intrinsics_optimize``), then the hand-eye
transform (``step_handeye_init``, ``step_handeye_optimize``).
``run_calibration`` runs all four.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from . import _vision_calibration as _native
from .common import (
    HandeyeInitOptions,
    HandeyeOptimizeOptions,
    IntrinsicsInitOptions,
    IntrinsicsOptimizeOptions,
)
from .models import (
    Camera,
    Distortion,
    PinholeIntrinsics,
    Pose,
    ScheimpflugSensor,
    SingleCamHandeyeConfig,
    SingleCamHandeyeExport,
    SingleCamHandeyeInput,
)
from .session import (
    CalibrationSession,
    _camera,
    _optional_pose,
    _payload,
    _poses,
    _seed_payload,
)

__all__ = [
    "HandeyeInitOptions",
    "HandeyeInitResult",
    "HandeyeManualInit",
    "HandeyeOptimizeOptions",
    "HandeyeOptimizeResult",
    "IntrinsicsInitOptions",
    "IntrinsicsInitResult",
    "IntrinsicsManualInit",
    "IntrinsicsOptimizeOptions",
    "IntrinsicsOptimizeResult",
    "SingleCamHandeyeSession",
]


@dataclass(slots=True)
class IntrinsicsManualInit:
    """Manual seeds of the intrinsics stage; a field left ``None`` is
    estimated.

    The fields mean what they mean for planar intrinsics; ``target_poses``
    are camera-from-target, one per view.
    """

    intrinsics: PinholeIntrinsics | None = None
    distortion: Distortion | None = None
    sensor: ScheimpflugSensor | None = None
    target_poses: list[Pose] | None = None

    def to_payload(self) -> dict[str, Any]:
        return _seed_payload(
            {
                "intrinsics": self.intrinsics,
                "distortion": self.distortion,
                "sensor": self.sensor,
                "target_poses": self.target_poses,
            }
        )


@dataclass(slots=True)
class HandeyeManualInit:
    """Manual seeds of the hand-eye stage; a field left ``None`` is
    estimated.

    Eye-in-hand reads ``gripper_se3_camera`` and ``base_se3_target``;
    eye-to-hand reads ``camera_se3_base`` and ``gripper_se3_target``. The
    other mode's fields are ignored.
    """

    gripper_se3_camera: Pose | None = None
    base_se3_target: Pose | None = None
    camera_se3_base: Pose | None = None
    gripper_se3_target: Pose | None = None

    def to_payload(self) -> dict[str, Any]:
        return _seed_payload(
            {
                "gripper_se3_camera": self.gripper_se3_camera,
                "base_se3_target": self.base_se3_target,
                "camera_se3_base": self.camera_se3_base,
                "gripper_se3_target": self.gripper_se3_target,
            }
        )


@dataclass(frozen=True, slots=True)
class IntrinsicsInitResult:
    """The initial camera and per-view target poses of the intrinsics stage."""

    camera: Camera
    target_poses: list[Pose]

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> IntrinsicsInitResult:
        return cls(camera=_camera(payload["camera"]), target_poses=_poses(payload["target_poses"]))


@dataclass(frozen=True, slots=True)
class IntrinsicsOptimizeResult:
    """The refined camera and target poses of the intrinsics stage."""

    camera: Camera
    target_poses: list[Pose]
    mean_reproj_error: float
    """Mean reprojection error, pixels."""

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> IntrinsicsOptimizeResult:
        return cls(
            camera=_camera(payload["camera"]),
            target_poses=_poses(payload["target_poses"]),
            mean_reproj_error=float(payload["mean_reproj_error"]),
        )


@dataclass(frozen=True, slots=True)
class HandeyeInitResult:
    """The linear hand-eye estimate: the two transforms of the session's mode
    are set, the other two are ``None``."""

    gripper_se3_camera: Pose | None
    base_se3_target: Pose | None
    camera_se3_base: Pose | None
    gripper_se3_target: Pose | None

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> HandeyeInitResult:
        return cls(
            gripper_se3_camera=_optional_pose(payload.get("gripper_se3_camera")),
            base_se3_target=_optional_pose(payload.get("base_se3_target")),
            camera_se3_base=_optional_pose(payload.get("camera_se3_base")),
            gripper_se3_target=_optional_pose(payload.get("gripper_se3_target")),
        )


@dataclass(frozen=True, slots=True)
class HandeyeOptimizeResult:
    """The solve metrics of the hand-eye refinement."""

    mean_reproj_error: float
    """Mean reprojection error, pixels."""
    final_cost: float

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> HandeyeOptimizeResult:
        return cls(
            mean_reproj_error=float(payload["mean_reproj_error"]),
            final_cost=float(payload["final_cost"]),
        )


class SingleCamHandeyeSession(
    CalibrationSession[SingleCamHandeyeInput, SingleCamHandeyeConfig, SingleCamHandeyeExport]
):
    """A single-camera hand-eye calibration, step by step."""

    _native_type = _native.SingleCamHandeyeSession
    _config_class = SingleCamHandeyeConfig
    _export_class = SingleCamHandeyeExport

    __slots__ = ()

    def step_intrinsics_init(
        self, options: IntrinsicsInitOptions | None = None
    ) -> IntrinsicsInitResult:
        """Fit the initial camera and target poses linearly."""
        return IntrinsicsInitResult.from_payload(
            self._native.step_intrinsics_init(_payload(options))
        )

    def step_intrinsics_init_with_seed(
        self,
        seed: IntrinsicsManualInit,
        options: IntrinsicsInitOptions | None = None,
    ) -> IntrinsicsInitResult:
        """``step_intrinsics_init`` from manual seeds."""
        return IntrinsicsInitResult.from_payload(
            self._native.step_intrinsics_init_with_seed(_payload(seed), _payload(options))
        )

    def step_intrinsics_optimize(
        self, options: IntrinsicsOptimizeOptions | None = None
    ) -> IntrinsicsOptimizeResult:
        """Refine the camera and target poses."""
        return IntrinsicsOptimizeResult.from_payload(
            self._native.step_intrinsics_optimize(_payload(options))
        )

    def step_handeye_init(self, options: HandeyeInitOptions | None = None) -> HandeyeInitResult:
        """Solve the hand-eye transform linearly from the robot poses."""
        return HandeyeInitResult.from_payload(self._native.step_handeye_init(_payload(options)))

    def step_handeye_init_with_seed(
        self,
        seed: HandeyeManualInit,
        options: HandeyeInitOptions | None = None,
    ) -> HandeyeInitResult:
        """``step_handeye_init`` from manual seeds."""
        return HandeyeInitResult.from_payload(
            self._native.step_handeye_init_with_seed(_payload(seed), _payload(options))
        )

    def step_handeye_optimize(
        self, options: HandeyeOptimizeOptions | None = None
    ) -> HandeyeOptimizeResult:
        """Refine camera, hand-eye and (if configured) robot poses jointly."""
        return HandeyeOptimizeResult.from_payload(
            self._native.step_handeye_optimize(_payload(options))
        )
