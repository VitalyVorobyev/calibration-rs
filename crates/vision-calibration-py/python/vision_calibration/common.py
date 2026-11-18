"""Step options and manual seeds shared across the problem types.

A step's options override the session config for that one call; a field left
``None`` keeps the config's value. The intrinsics options serve every
intrinsics-bearing problem, the rig options both rig problems and the
hand-eye options both hand-eye problems.

The manual seeds of the intrinsics stages: :class:`IntrinsicsManualInit` for
one camera (planar intrinsics, single-camera hand-eye) and
:class:`RigIntrinsicsManualInit` for each camera of a rig (rig extrinsics,
rig hand-eye).

The seeds and results of the stages both rig problems run:
:class:`RigIntrinsicsInitResult` of the per-camera intrinsics init,
:class:`RigManualInit` (with its :class:`RigPoses`) and
:class:`RigInitResult` of the rig init; and :class:`HandeyeOptimizeResult`
of the hand-eye refinement both hand-eye problems run.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from typing import Any, cast

from ._generated_types import HandeyeMethod
from .models import CameraParams, Distortion, PinholeIntrinsics, Pose, ScheimpflugSensor
from .session import _cameras, _optional_pose, _poses, _seed_payload, _termination
from .types import SolveTermination

__all__ = [
    "HandeyeInitOptions",
    "HandeyeOptimizeOptions",
    "HandeyeOptimizeResult",
    "IntrinsicsInitOptions",
    "IntrinsicsManualInit",
    "IntrinsicsOptimizeOptions",
    "RigInitOptions",
    "RigInitResult",
    "RigIntrinsicsInitResult",
    "RigIntrinsicsManualInit",
    "RigManualInit",
    "RigOptimizeOptions",
    "RigPoses",
]


class _StepOptions:
    """Serde payload of a step-options dataclass: unset fields are left out."""

    __slots__ = ()

    def to_payload(self) -> dict[str, Any]:
        return {
            f.name: getattr(self, f.name)
            for f in fields(self)  # type: ignore[arg-type]
            if getattr(self, f.name) is not None
        }


@dataclass(slots=True)
class IntrinsicsInitOptions(_StepOptions):
    """Options of an intrinsics initialization step."""

    iterations: int | None = None
    """Iterations of the iterative intrinsics estimate."""


@dataclass(slots=True)
class IntrinsicsOptimizeOptions(_StepOptions):
    """Options of an intrinsics optimization step."""

    max_iters: int | None = None
    """Maximum solver iterations."""
    verbosity: int | None = None
    """Solver verbosity."""


@dataclass(slots=True)
class HandeyeInitOptions(_StepOptions):
    """Options of a hand-eye initialization step."""

    method: HandeyeMethod | None = None
    """Closed-form solver: ``"tsai_lenz"``, ``"park_martin"``,
    ``"daniilidis"`` or ``"shah"``."""
    min_motion_angle_deg: float | None = None
    """Minimum rotation between robot poses used by the linear hand-eye
    solve, degrees."""


@dataclass(slots=True)
class HandeyeOptimizeOptions(_StepOptions):
    """Options of a hand-eye optimization step."""

    max_iters: int | None = None
    """Maximum solver iterations."""
    verbosity: int | None = None
    """Solver verbosity."""


@dataclass(slots=True)
class RigInitOptions(_StepOptions):
    """Options of a rig initialization step (the linear rig fit has none)."""


@dataclass(slots=True)
class RigOptimizeOptions(_StepOptions):
    """Options of a rig optimization step."""

    max_iters: int | None = None
    """Maximum solver iterations."""
    verbosity: int | None = None
    """Solver verbosity."""


@dataclass(slots=True)
class IntrinsicsManualInit:
    """Manual seeds of a one-camera intrinsics stage (planar intrinsics'
    ``step_init_with_seed``, single-camera hand-eye's
    ``step_intrinsics_init_with_seed``); a field left ``None`` is estimated.

    A seeded ``intrinsics`` skips the linear intrinsics fit (distortion then
    starts at zero unless seeded too). The distortion seed is a model of the
    camera's own distortion kind (``camera.model.distortion``) and is used
    as given; another kind is an error. ``sensor`` seeds the tilt of a
    Scheimpflug camera; an untilted camera takes only a zero tilt. A seeded
    tilt is used as given and refined directly instead of swept. ``cam_se3_target``
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
class RigIntrinsicsManualInit:
    """Manual seeds of the per-camera intrinsics stage of a rig (rig
    extrinsics, rig hand-eye), one entry per camera; a field left ``None`` is
    estimated.

    Seeded intrinsics skip the per-camera linear fit (distortion then starts
    at zero unless seeded too). Each distortion seed is a model of its
    camera's own distortion kind and is used as given; another kind is an
    error. ``per_cam_sensors`` seeds the tilt of
    each Scheimpflug camera; an untilted camera takes only a zero tilt.
    """

    per_cam_intrinsics: list[PinholeIntrinsics] | None = None
    per_cam_distortion: list[Distortion] | None = None
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
class RigPoses:
    """Seeded rig poses: the two go together.

    ``cam_se3_rig`` is each camera's camera-from-rig pose (one per camera);
    ``rig_se3_target`` each view's rig-from-target pose (one per view).
    """

    cam_se3_rig: list[Pose]
    rig_se3_target: list[Pose]

    def to_payload(self) -> dict[str, Any]:
        return _seed_payload(
            {"cam_se3_rig": self.cam_se3_rig, "rig_se3_target": self.rig_se3_target}
        )


@dataclass(slots=True)
class RigManualInit:
    """Manual seeds of the rig stage (rig extrinsics, rig hand-eye).

    ``poses`` seeds the camera-from-rig and rig-from-target poses together;
    ``None`` runs the linear extrinsics fit.
    """

    poses: RigPoses | None = None

    def to_payload(self) -> dict[str, Any]:
        return _seed_payload({"poses": self.poses})


def _target_poses(values: Any) -> list[list[Pose | None]]:
    return [
        [_optional_pose(p) for p in cast(Sequence[Any], row)]
        for row in cast(Sequence[Any], values)
    ]


@dataclass(frozen=True, slots=True)
class RigIntrinsicsInitResult:
    """The initial per-camera estimate of a rig (rig extrinsics, rig
    hand-eye)."""

    cameras: list[CameraParams]
    cam_se3_target: list[list[Pose | None]]
    """``[view][camera]`` camera-from-target; ``None`` where the camera did
    not see the target."""

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> RigIntrinsicsInitResult:
        return cls(
            cameras=_cameras(payload["cameras"]),
            cam_se3_target=_target_poses(payload["cam_se3_target"]),
        )


@dataclass(frozen=True, slots=True)
class RigInitResult:
    """The linear rig estimate (rig extrinsics, rig hand-eye)."""

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
class HandeyeOptimizeResult:
    """The solve metrics of the hand-eye refinement (single-camera and rig
    hand-eye)."""

    mean_reproj_error: float
    """Mean reprojection error, pixels."""
    final_cost: float
    termination: SolveTermination | None
    """Why the solver stopped; ``None`` when that is unknown."""

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> HandeyeOptimizeResult:
        return cls(
            mean_reproj_error=float(payload["mean_reproj_error"]),
            final_cost=float(payload["final_cost"]),
            termination=_termination(payload.get("termination")),
        )
