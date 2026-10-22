"""Step options shared across the problem types.

A step's options override the session config for that one call; a field left
``None`` keeps the config's value. The intrinsics options serve every
intrinsics-bearing problem, the rig options both rig problems and the
hand-eye options both hand-eye problems.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

__all__ = [
    "HandeyeInitOptions",
    "HandeyeOptimizeOptions",
    "IntrinsicsInitOptions",
    "IntrinsicsOptimizeOptions",
    "RigInitOptions",
    "RigOptimizeOptions",
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
