"""Rig hand-eye: a camera rig on (or watching) a robot.

:class:`RigHandeyeSession` runs three stages, each an init and an optimize
step: the cameras' intrinsics (``step_intrinsics_init_all``,
``step_intrinsics_optimize_all``), the rig (``step_rig_init``,
``step_rig_optimize``) and the hand-eye transform (``step_handeye_init``,
``step_handeye_optimize``). ``run_calibration`` runs all six, seeding the
intrinsics stage from the config's ``manual_init``.

The rig seeds and initial results have the shapes of
:mod:`vision_calibration.rig_extrinsics`, whose classes this module
re-exports; the intrinsics seed is :class:`RigIntrinsicsManualInit`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar, TypeAlias

from . import _vision_calibration as _native
from .common import (
    HandeyeInitOptions,
    HandeyeOptimizeOptions,
    IntrinsicsInitOptions,
    IntrinsicsOptimizeOptions,
    RigInitOptions,
    RigIntrinsicsManualInit,
    RigOptimizeOptions,
)
from .models import (
    CameraParams,
    HandeyeMode,
    Pose,
    RigHandeyeConfig,
    RigHandeyeExport,
    RigHandeyeInput,
    HandeyeSolution,
    _rig_solution_from_payload,
)
from .rig_extrinsics import (
    IntrinsicsInitResult,
    RigInitResult,
    RigManualInit,
    RigPoses,
    _target_poses,
)
from .session import (
    _run_calibration,
    CalibrationSession,
    _cameras,
    _floats,
    _payload,
    _poses,
    _seed_payload,
)
from .types import ProgressCallback

__all__ = [
    "EyeInHandManualInit",
    "EyeToHandManualInit",
    "HandeyeInitOptions",
    "HandeyeInitResult",
    "HandeyeManualInit",
    "HandeyeOptimizeOptions",
    "HandeyeOptimizeResult",
    "IntrinsicsInitOptions",
    "IntrinsicsInitResult",
    "IntrinsicsOptimizeOptions",
    "IntrinsicsOptimizeResult",
    "RigHandeyeSession",
    "RigInitOptions",
    "RigInitResult",
    "RigIntrinsicsManualInit",
    "RigManualInit",
    "RigOptimizeOptions",
    "RigOptimizeResult",
    "RigPoses",
    "run_calibration",
]


@dataclass(slots=True)
class EyeInHandManualInit:
    """Manual seeds of the hand-eye stage of an eye-in-hand session (the rig
    on the gripper); a pose left ``None`` is estimated.

    The session's ``handeye_init.handeye_mode`` must be ``"eye_in_hand"``; a
    seed of the other mode raises before any work.
    """

    gripper_se3_rig: Pose | None = None
    base_se3_target: Pose | None = None
    mode: ClassVar[HandeyeMode] = "eye_in_hand"

    def to_payload(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            **_seed_payload(
                {
                    "gripper_se3_rig": self.gripper_se3_rig,
                    "base_se3_target": self.base_se3_target,
                }
            ),
        }


@dataclass(slots=True)
class EyeToHandManualInit:
    """Manual seeds of the hand-eye stage of an eye-to-hand session (the rig
    in the base frame); a pose left ``None`` is estimated.

    The session's ``handeye_init.handeye_mode`` must be ``"eye_to_hand"``; a
    seed of the other mode raises before any work.
    """

    rig_se3_base: Pose | None = None
    gripper_se3_target: Pose | None = None
    mode: ClassVar[HandeyeMode] = "eye_to_hand"

    def to_payload(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            **_seed_payload(
                {
                    "rig_se3_base": self.rig_se3_base,
                    "gripper_se3_target": self.gripper_se3_target,
                }
            ),
        }


HandeyeManualInit: TypeAlias = EyeInHandManualInit | EyeToHandManualInit
"""The hand-eye seed of a session: one class per mode, named like the
export's :data:`HandeyeSolution`."""


@dataclass(frozen=True, slots=True)
class IntrinsicsOptimizeResult:
    """The refined cameras and target poses of the intrinsics stage."""

    cameras: list[CameraParams]
    per_cam_reproj_errors: list[float]
    """Mean reprojection error per camera, pixels."""
    cam_se3_target: list[list[Pose | None]]
    """``[view][camera]`` camera-from-target; ``None`` where the camera did
    not see the target."""

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> IntrinsicsOptimizeResult:
        return cls(
            cameras=_cameras(payload["cameras"]),
            per_cam_reproj_errors=_floats(payload["per_cam_reproj_errors"]),
            cam_se3_target=_target_poses(payload["cam_se3_target"]),
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
    """The linear hand-eye estimate: the hand-eye and target pose of the
    session's mode, both always set."""

    handeye: HandeyeSolution

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> HandeyeInitResult:
        return cls(handeye=_rig_solution_from_payload(payload["handeye"]))


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
        seed: RigIntrinsicsManualInit,
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


def run_calibration(
    input: RigHandeyeInput,
    config: RigHandeyeConfig | None = None,
    *,
    progress: ProgressCallback | None = None,
) -> RigHandeyeExport:
    """Run the rig hand-eye calibration end to end: the session's ``run_calibration`` and
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
    return _run_calibration(RigHandeyeSession, input, config, progress, RigHandeyeInput, RigHandeyeConfig)
