"""Rig hand-eye: a camera rig on (or watching) a robot.

:class:`RigHandeyeSession` runs three stages, each an init and an optimize
step: the cameras' intrinsics (``step_intrinsics_init_all``,
``step_intrinsics_optimize_all``), the rig (``step_rig_init``,
``step_rig_optimize``) and the hand-eye transform (``step_handeye_init``,
``step_handeye_optimize``). ``run_calibration`` runs all six, seeding the
intrinsics stage from the config's ``manual_init``.

The intrinsics and rig seeds and initial results have the shapes of
:mod:`vision_calibration.rig_extrinsics`, whose classes this module
re-exports.
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
    RigInitOptions,
    RigOptimizeOptions,
)
from .models import (
    Camera,
    Pose,
    RigHandeyeConfig,
    RigHandeyeExport,
    RigHandeyeInput,
)
from .rig_extrinsics import (
    IntrinsicsInitResult,
    IntrinsicsManualInit,
    RigInitResult,
    RigManualInit,
    _target_poses,
)
from .session import (
    CalibrationSession,
    _cameras,
    _floats,
    _optional_pose,
    _payload,
    _pose,
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
    "RigHandeyeSession",
    "RigInitOptions",
    "RigInitResult",
    "RigManualInit",
    "RigOptimizeOptions",
    "RigOptimizeResult",
]


@dataclass(slots=True)
class HandeyeManualInit:
    """Manual seeds of the hand-eye stage; a field left ``None`` is
    estimated.

    Eye-in-hand: ``handeye`` is ``gripper_se3_rig`` and
    ``mode_target_pose`` is ``base_se3_target``. Eye-to-hand: ``handeye`` is
    ``rig_se3_base`` and ``mode_target_pose`` is ``gripper_se3_target``.
    """

    handeye: Pose | None = None
    mode_target_pose: Pose | None = None

    def to_payload(self) -> dict[str, Any]:
        return _seed_payload({"handeye": self.handeye, "mode_target_pose": self.mode_target_pose})


@dataclass(frozen=True, slots=True)
class IntrinsicsOptimizeResult:
    """The refined cameras and target poses of the intrinsics stage."""

    cameras: list[Camera]
    per_cam_reproj_errors: list[float]
    """Mean reprojection error per camera, pixels."""
    per_cam_target_poses: list[list[Pose | None]]
    """``[view][camera]`` camera-from-target; ``None`` where the camera did
    not see the target."""

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> IntrinsicsOptimizeResult:
        return cls(
            cameras=_cameras(payload["cameras"]),
            per_cam_reproj_errors=_floats(payload["per_cam_reproj_errors"]),
            per_cam_target_poses=_target_poses(payload["per_cam_target_poses"]),
        )


@dataclass(frozen=True, slots=True)
class RigOptimizeResult:
    """The rig bundle adjustment's result."""

    mean_reproj_error: float
    """Mean reprojection error over the rig, pixels."""
    per_cam_reproj_errors: list[float]
    cam_se3_rig: list[Pose]
    rig_se3_target: list[Pose]

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> RigOptimizeResult:
        return cls(
            mean_reproj_error=float(payload["mean_reproj_error"]),
            per_cam_reproj_errors=_floats(payload["per_cam_reproj_errors"]),
            cam_se3_rig=_poses(payload["cam_se3_rig"]),
            rig_se3_target=_poses(payload["rig_se3_target"]),
        )


@dataclass(frozen=True, slots=True)
class HandeyeInitResult:
    """The linear hand-eye estimate, in the session's mode (see
    :class:`HandeyeManualInit`)."""

    initial_handeye: Pose
    initial_mode_target_pose: Pose | None

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> HandeyeInitResult:
        return cls(
            initial_handeye=_pose(payload["initial_handeye"]),
            initial_mode_target_pose=_optional_pose(payload.get("initial_mode_target_pose")),
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


class RigHandeyeSession(CalibrationSession[RigHandeyeInput, RigHandeyeConfig, RigHandeyeExport]):
    """A rig hand-eye calibration, step by step."""

    _native_type = _native.RigHandeyeSession
    _config_class = RigHandeyeConfig
    _export_class = RigHandeyeExport

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
        """Refine rig, hand-eye and (if configured) robot poses jointly."""
        return HandeyeOptimizeResult.from_payload(
            self._native.step_handeye_optimize(_payload(options))
        )
