"""The session framework: a calibration run driven step by step.

A session holds one problem's input, config, intermediate state and result.
Each problem type has its session class — :class:`PlanarIntrinsicsSession`,
:class:`RigHandeyeSession`, … in the problem's module — with the problem's
``step_*`` methods; this module holds their shared base and records.

.. code-block:: python

    from vision_calibration.planar_intrinsics import PlanarIntrinsicsSession

    session = PlanarIntrinsicsSession()
    session.set_input(dataset)
    session.step_init()
    session.step_optimize()
    result = session.export()
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Generic, Protocol, TypeVar, cast

from .models import (
    _DISTORTION_TAG_BY_TYPE,
    Camera,
    Pose,
    _camera_from_payload,
    _distortion_to_payload,
)

__all__ = [
    "CalibrationSession",
    "ExportRecord",
    "LogEntry",
    "SessionMetadata",
]


class _Payload(Protocol):
    def to_payload(self) -> Any: ...


_InputT = TypeVar("_InputT", bound=_Payload)
_ConfigT = TypeVar("_ConfigT", bound=_Payload)
_ExportT = TypeVar("_ExportT")
_SessionT = TypeVar("_SessionT", bound="CalibrationSession[Any, Any, Any]")


@dataclass(frozen=True, slots=True)
class LogEntry:
    """One operation in a session's log: a step that ran, and how it went.

    ``notes`` carries the step's remarks (the init log's seed sources, a
    warning about a seed that does not fit the camera model) or, for a
    failure, the error.
    """

    timestamp: int
    """Unix time of the operation, seconds."""
    operation: str
    success: bool
    notes: str | None

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> LogEntry:
        return cls(
            timestamp=int(payload["timestamp"]),
            operation=str(payload["operation"]),
            success=bool(payload["success"]),
            notes=cast(str | None, payload.get("notes")),
        )


@dataclass(frozen=True, slots=True)
class SessionMetadata:
    """A session's problem type, schema version, timestamps and description."""

    problem_type: str
    schema_version: int
    created_at: int
    """Unix time the session was created, seconds."""
    last_modified: int
    """Unix time of the session's last change, seconds."""
    description: str | None

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> SessionMetadata:
        return cls(
            problem_type=str(payload["problem_type"]),
            schema_version=int(payload["schema_version"]),
            created_at=int(payload["created_at"]),
            last_modified=int(payload["last_modified"]),
            description=cast(str | None, payload.get("description")),
        )


@dataclass(frozen=True, slots=True)
class ExportRecord(Generic[_ExportT]):
    """An export the session produced, with when and the notes given."""

    timestamp: int
    """Unix time of the export, seconds."""
    export: _ExportT
    notes: str | None


def _payload(value: Any) -> Any:
    """A dataclass's serde payload; a payload (mapping or list) or ``None``
    passes through."""
    if value is None or isinstance(value, (Mapping, list)):
        return value
    to_payload = getattr(value, "to_payload", None)
    if to_payload is None:
        raise TypeError(
            f"expected a dataclass with to_payload() or a payload, got {type(value).__name__}"
        )
    return to_payload()


def _seed_payload(fields: Mapping[str, Any]) -> dict[str, Any]:
    """The payload of a manual-init seed: unset (``None``) fields are left out,
    dataclasses become their payloads (a distortion model its type-tagged
    payload), lists convert element-wise."""

    def convert(value: Any) -> Any:
        if isinstance(value, (list, tuple)):
            return [convert(v) for v in value]
        if type(value) in _DISTORTION_TAG_BY_TYPE:
            return _distortion_to_payload(value)
        return _payload(value)

    return {name: convert(value) for name, value in fields.items() if value is not None}


def _pose(value: Any) -> Pose:
    return Pose.from_payload(cast(Mapping[str, Any], value))


def _optional_pose(value: Any) -> Pose | None:
    return None if value is None else _pose(value)


def _poses(values: Any) -> list[Pose]:
    return [_pose(v) for v in cast(Sequence[Any], values)]


def _camera(value: Any) -> Camera:
    return _camera_from_payload(cast(Mapping[str, Any], value))


def _cameras(values: Any) -> list[Camera]:
    return [_camera(v) for v in cast(Sequence[Any], values)]


def _floats(values: Any) -> list[float]:
    return [float(v) for v in cast(Sequence[Any], values)]


class CalibrationSession(Generic[_InputT, _ConfigT, _ExportT]):
    """Shared base of the per-problem session classes.

    ``set_input`` and ``set_config`` take the problem's dataclasses or their
    payload mappings. ``set_input`` validates the input and clears any
    computed state; ``set_config`` keeps it. ``run_calibration`` runs the
    problem's default pipeline; the subclasses add the individual
    ``step_*`` methods. Long steps release the GIL.

    Errors: an input or config the problem rejects, a step run out of order
    and a manual-init seed that does not fit raise :class:`ValueError`; a
    numerical failure raises :class:`RuntimeError`.
    """

    # The native session class, the config dataclass (``from_mapping``) and
    # the export dataclass (``from_payload``) of the problem.
    _native_type: ClassVar[Callable[..., Any]]
    _config_class: ClassVar[type[Any]]
    _export_class: ClassVar[type[Any]]

    __slots__ = ("_native",)

    def __init__(self, description: str | None = None) -> None:
        self._native: Any = type(self)._native_type(description)

    # ── Input and config ──────────────────────────────────────────────────

    def set_input(self, input: _InputT | Mapping[str, Any] | list[Any]) -> None:
        """Set the observations; clears any computed state."""
        self._native.set_input(_payload(input))

    @property
    def input(self) -> dict[str, Any] | None:
        """The input as its serde payload, or ``None`` before ``set_input``."""
        return cast(dict[str, Any] | None, self._native.input())

    @property
    def has_input(self) -> bool:
        return bool(self._native.has_input())

    def set_config(self, config: _ConfigT | Mapping[str, Any]) -> None:
        """Replace the config; computed state is kept."""
        self._native.set_config(_payload(config))

    @property
    def config(self) -> _ConfigT:
        """The session's config (the problem's default until ``set_config``)."""
        return cast(_ConfigT, type(self)._config_class.from_mapping(self._native.config()))

    # ── Result ────────────────────────────────────────────────────────────

    @property
    def has_output(self) -> bool:
        """Whether a final solve has produced an estimate."""
        return bool(self._native.has_output())

    @property
    def output(self) -> dict[str, Any] | None:
        """The final solve's estimate as its serde payload, or ``None``."""
        return cast(dict[str, Any] | None, self._native.output())

    def export(self, notes: str | None = None) -> _ExportT:
        """The result, built from the input, the estimate and the config.

        Each call is also kept in :attr:`exports`, with ``notes``.
        """
        return cast(_ExportT, type(self)._export_class.from_payload(self._native.export(notes)))

    @property
    def exports(self) -> list[ExportRecord[_ExportT]]:
        """Every export made so far, oldest first."""
        records = cast(list[Mapping[str, Any]], self._native.exports())
        return [
            ExportRecord(
                timestamp=int(r["timestamp"]),
                export=cast(_ExportT, type(self)._export_class.from_payload(cast(Mapping[str, Any], r["export"]))),
                notes=cast(str | None, r.get("notes")),
            )
            for r in records
        ]

    # ── Log, metadata, lifecycle ──────────────────────────────────────────

    @property
    def log(self) -> list[LogEntry]:
        """The operation log, oldest first; the init steps record their seed
        sources and warnings here."""
        return [LogEntry.from_payload(e) for e in cast(list[Mapping[str, Any]], self._native.log())]

    @property
    def metadata(self) -> SessionMetadata:
        return SessionMetadata.from_payload(cast(Mapping[str, Any], self._native.metadata()))

    def validate(self) -> None:
        """Check the input and config are set and agree; raise ``ValueError``
        if not."""
        self._native.validate()

    def reset(self) -> None:
        """Clear input, state, result, exports and log; keep the config."""
        self._native.reset()

    def run_calibration(self) -> None:
        """Run the problem's default pipeline of steps."""
        self._native.run_calibration()

    def to_json(self) -> str:
        """The whole session as JSON, to resume later with :meth:`from_json`."""
        return cast(str, self._native.to_json())

    @classmethod
    def from_json(cls: type[_SessionT], json: str) -> _SessionT:
        """Resume a session saved by :meth:`to_json`.

        Raises ``ValueError`` if the JSON is a session of another problem
        type or schema version.
        """
        session = cls.__new__(cls)
        native_type: Any = cls._native_type
        session._native = native_type.from_json(json)
        return session
