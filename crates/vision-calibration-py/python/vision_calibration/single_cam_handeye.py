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
from typing import Any, ClassVar, TypeAlias

from . import _vision_calibration as _native
from .common import (
    HandeyeInitOptions,
    HandeyeOptimizeOptions,
    IntrinsicsInitOptions,
    IntrinsicsOptimizeOptions,
)
from .models import (
    CameraParams,
    Distortion,
    HandeyeMode,
    PinholeIntrinsics,
    Pose,
    ScheimpflugSensor,
    SingleCamHandeyeConfig,
    SingleCamHandeyeExport,
    SingleCamHandeyeInput,
    SingleCamHandeyeSolution,
    _camera_solution_from_payload,
)
from .session import (
    _run_calibration,
    CalibrationSession,
    _camera,
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
    "IntrinsicsManualInit",
    "IntrinsicsOptimizeOptions",
    "IntrinsicsOptimizeResult",
    "SingleCamHandeyeSession",
    "run_calibration",
]


@dataclass(slots=True)
class IntrinsicsManualInit:
    """Manual seeds of the intrinsics stage; a field left ``None`` is
    estimated.

    The fields mean what they mean for planar intrinsics; ``cam_se3_target``
    are camera-from-target, one per view.
    """

    intrinsics: PinholeIntrinsics | None = None
    distortion: Distortion | None = None
    sensor: ScheimpflugSensor | None = None
    cam_se3_target: list[Pose] | None = None

    def to_payload(self) -> dict[str, Any]:
        return _seed_payload(
            {
                "intrinsics": self.intrinsics,
                "distortion": self.distortion,
                "sensor": self.sensor,
                "cam_se3_target": self.cam_se3_target,
            }
        )


@dataclass(slots=True)
class EyeInHandManualInit:
    """Manual seeds of the hand-eye stage of an eye-in-hand session (the
    camera on the gripper); a pose left ``None`` is estimated.

    The session's ``handeye_init.handeye_mode`` must be ``"eye_in_hand"``; a
    seed of the other mode raises before any work.
    """

    gripper_se3_cam: Pose | None = None
    base_se3_target: Pose | None = None
    mode: ClassVar[HandeyeMode] = "eye_in_hand"

    def to_payload(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            **_seed_payload(
                {
                    "gripper_se3_cam": self.gripper_se3_cam,
                    "base_se3_target": self.base_se3_target,
                }
            ),
        }


@dataclass(slots=True)
class EyeToHandManualInit:
    """Manual seeds of the hand-eye stage of an eye-to-hand session (the
    camera in the base frame); a pose left ``None`` is estimated.

    The session's ``handeye_init.handeye_mode`` must be ``"eye_to_hand"``; a
    seed of the other mode raises before any work.
    """

    cam_se3_base: Pose | None = None
    gripper_se3_target: Pose | None = None
    mode: ClassVar[HandeyeMode] = "eye_to_hand"

    def to_payload(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            **_seed_payload(
                {
                    "cam_se3_base": self.cam_se3_base,
                    "gripper_se3_target": self.gripper_se3_target,
                }
            ),
        }


HandeyeManualInit: TypeAlias = EyeInHandManualInit | EyeToHandManualInit
"""The hand-eye seed of a session: one class per mode, named like the
export's :data:`SingleCamHandeyeSolution`."""


@dataclass(frozen=True, slots=True)
class IntrinsicsInitResult:
    """The initial camera and per-view target poses of the intrinsics stage."""

    camera: CameraParams
    cam_se3_target: list[Pose]

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> IntrinsicsInitResult:
        return cls(camera=_camera(payload["camera"]), cam_se3_target=_poses(payload["cam_se3_target"]))


@dataclass(frozen=True, slots=True)
class IntrinsicsOptimizeResult:
    """The refined camera and target poses of the intrinsics stage."""

    camera: CameraParams
    cam_se3_target: list[Pose]
    mean_reproj_error: float
    """Mean reprojection error, pixels."""

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> IntrinsicsOptimizeResult:
        return cls(
            camera=_camera(payload["camera"]),
            cam_se3_target=_poses(payload["cam_se3_target"]),
            mean_reproj_error=float(payload["mean_reproj_error"]),
        )


@dataclass(frozen=True, slots=True)
class HandeyeInitResult:
    """The linear hand-eye estimate: the hand-eye and target pose of the
    session's mode, both always set."""

    handeye: SingleCamHandeyeSolution

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> HandeyeInitResult:
        return cls(handeye=_camera_solution_from_payload(payload["handeye"]))


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


def run_calibration(
    input: SingleCamHandeyeInput,
    config: SingleCamHandeyeConfig | None = None,
    *,
    progress: ProgressCallback | None = None,
) -> SingleCamHandeyeExport:
    """Run the single-camera hand-eye calibration end to end: the session's ``run_calibration`` and
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
    return _run_calibration(SingleCamHandeyeSession, input, config, progress, SingleCamHandeyeInput, SingleCamHandeyeConfig)
