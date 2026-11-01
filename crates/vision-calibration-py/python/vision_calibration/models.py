"""Public high-level models for :mod:`vision_calibration`.

These dataclasses provide a Python-native API surface with explicit fields,
defaults, and docstrings. Wrapper functions in :mod:`vision_calibration`
accept these models and convert them to the serde payloads expected by Rust.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, cast

from .residuals import PerFeatureResiduals
from ._generated_types import (
    DistortionKind,
    ProjectionKind,
    SensorKind,
    HandeyeMode,
    LaserlineResidualType,
    RobustLoss,
    SolverBackend,
)

Vec2 = tuple[float, float]
Vec3 = tuple[float, float, float]
QuatXyzw = tuple[float, float, float, float]

@dataclass(slots=True)
class IntrinsicsInitConfig:
    """Shared per-camera linear-initialization stage.

    Zhang's method plus an iterative Brown-Conrady distortion fit. Shared by
    every intrinsics-bearing problem's ``init``/``intrinsics`` config group.
    The fit estimates only the distortion terms the camera's solves estimate
    (its ``camera.fix`` lists), and holds the tangential terms while a sensor
    tilt is estimated.
    """

    iterations: int = 2
    #: Hold ``k3`` at zero in the linear stage even where the camera's solves
    #: estimate it.
    fix_k3: bool = True

    def to_payload(self) -> dict[str, Any]:
        return {
            "iterations": int(self.iterations),
            "fix_k3": bool(self.fix_k3),
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "IntrinsicsInitConfig":
        cfg = cls()
        for key, value in mapping.items():
            if not hasattr(cfg, key):
                raise ValueError(f"unknown IntrinsicsInitConfig field: {key}")
            setattr(cfg, key, value)
        return cfg


@dataclass(slots=True)
class SolverConfig:
    """Shared non-linear solve stage settings."""

    max_iters: int = 50
    verbosity: int = 0
    robust_loss: RobustLoss = "None"
    #: Engine that linearizes the problem: ``"tiny_solver"`` (default) or
    #: ``"factrs"``. Both reach the same minimizer.
    backend: SolverBackend = "tiny_solver"

    def to_payload(self) -> dict[str, Any]:
        return {
            "max_iters": int(self.max_iters),
            "verbosity": int(self.verbosity),
            "robust_loss": cast(Any, self.robust_loss),
            "backend": str(self.backend),
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "SolverConfig":
        cfg = cls()
        for key, value in mapping.items():
            if not hasattr(cfg, key):
                raise ValueError(f"unknown SolverConfig field: {key}")
            setattr(cfg, key, value)
        return cfg


@dataclass(slots=True)
class RobotPoseConfig:
    """Shared robot-pose refinement settings for hand-eye BA.

    With ``refine``, every view's robot pose gets an se(3) correction in
    the gripper frame under a zero-mean prior: ``rot_sigma`` (radians) and
    ``trans_sigma`` (metres) are the standard deviations of the tool's
    orientation and position error per gripper axis. The prior is weighed
    against the corner residuals at their estimated pixel noise. With
    ``estimate_sigma`` (the default) they are starting values: the solve
    estimates the robot's error from the corrections and reports it as the
    export's ``robot_pose_sigma``. Without it the result's covariance takes
    them as the robot's true accuracy.
    """

    refine: bool = True
    rot_sigma: float = 0.5 * 3.141592653589793 / 180.0
    trans_sigma: float = 0.001
    estimate_sigma: bool = True

    def to_payload(self) -> dict[str, Any]:
        return {
            "refine": bool(self.refine),
            "rot_sigma": float(self.rot_sigma),
            "trans_sigma": float(self.trans_sigma),
            "estimate_sigma": bool(self.estimate_sigma),
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "RobotPoseConfig":
        cfg = cls()
        for key, value in mapping.items():
            if not hasattr(cfg, key):
                raise ValueError(f"unknown RobotPoseConfig field: {key}")
            setattr(cfg, key, value)
        return cfg


@dataclass(slots=True)
class RobotPoseSigma:
    """A robot's pose error: standard deviations of the tool's orientation
    error about each gripper axis (``rot_sigma``, radians) and of its
    position error along each (``trans_sigma``, metres)."""

    rot_sigma: float
    trans_sigma: float

    def to_payload(self) -> dict[str, Any]:
        return {"rot_sigma": float(self.rot_sigma), "trans_sigma": float(self.trans_sigma)}

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any] | None) -> "RobotPoseSigma | None":
        if payload is None:
            return None
        return cls(
            rot_sigma=float(payload["rot_sigma"]), trans_sigma=float(payload["trans_sigma"])
        )


@dataclass(slots=True)
class HandeyeInitConfig:
    """Shared hand-eye linear-initialization settings (Tsai-Lenz DLT)."""

    handeye_mode: HandeyeMode = "EyeInHand"
    min_motion_angle_deg: float = 5.0

    def to_payload(self) -> dict[str, Any]:
        return {
            "handeye_mode": self.handeye_mode,
            "min_motion_angle_deg": float(self.min_motion_angle_deg),
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "HandeyeInitConfig":
        cfg = cls()
        for key, value in mapping.items():
            if not hasattr(cfg, key):
                raise ValueError(f"unknown HandeyeInitConfig field: {key}")
            setattr(cfg, key, value)
        return cfg


@dataclass(slots=True)
class RigConfig:
    """Shared multi-camera rig frame options: reference camera and
    rig-BA scope. Mirrors ``vision_calibration_pipeline::common::config::RigConfig``.
    """

    reference_camera_idx: int = 0
    #: Refine each camera's intrinsics jointly with the rig (default), so the
    #: result's covariance includes them.
    refine_intrinsics: bool = True
    #: Also refine each tilted camera's sensor tilt in the rig bundle
    #: adjustment (no effect for untilted cameras).
    refine_sensor: bool = False

    def to_payload(self) -> dict[str, Any]:
        return {
            "reference_camera_idx": int(self.reference_camera_idx),
            "refine_intrinsics": bool(self.refine_intrinsics),
            "refine_sensor": bool(self.refine_sensor),
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "RigConfig":
        cfg = cls()
        for key, value in mapping.items():
            if not hasattr(cfg, key):
                raise ValueError(f"unknown RigConfig field: {key}")
            setattr(cfg, key, value)
        return cfg


@dataclass(slots=True)
class CameraModelSpec:
    """Camera model: one kind per slot.

    ``projection`` is ``"pinhole"`` or ``"equidistant"`` (fisheye);
    ``distortion`` is one of ``"none"``, ``"brown_conrady5"``, ``"rational8"``,
    ``"thin_prism9"``, ``"division1"``, ``"kannala_brandt4"``; ``sensor`` is
    ``"identity"`` or ``"scheimpflug"`` (tilted sensor). The calibration
    workflows take a pinhole projection; an equidistant camera can be built
    and evaluated with :class:`~vision_calibration.CameraModel`.
    """

    projection: ProjectionKind = "pinhole"
    distortion: DistortionKind = "brown_conrady5"
    sensor: SensorKind = "identity"

    def to_payload(self) -> dict[str, Any]:
        return {
            "projection": self.projection,
            "distortion": self.distortion,
            "sensor": self.sensor,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "CameraModelSpec":
        cfg = cls()
        for key, value in payload.items():
            if not hasattr(cfg, key):
                raise ValueError(f"unknown CameraModelSpec field: {key}")
            setattr(cfg, key, value)
        return cfg


@dataclass(slots=True)
class CameraFix:
    """Camera parameters held fixed during refinement, by parameter name.

    Each slot lists the names to fix: ``intrinsics`` (``fx, fy, cx, cy``),
    ``projection``, ``distortion`` (names of the chosen distortion model, e.g.
    ``k1, k2, k3, p1, p2`` for ``brown_conrady5`` or ``lambda`` for
    ``division1``) and ``sensor`` (``tilt_x, tilt_y``). ``None`` keeps the
    model default for that slot (``k3`` fixed for ``brown_conrady5``; under a
    Scheimpflug sensor everything but the leading distortion terms); an empty
    list fixes nothing. Unknown names are rejected by the solver.
    """

    intrinsics: list[str] | None = None
    projection: list[str] | None = None
    distortion: list[str] | None = None
    sensor: list[str] | None = None

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for name in ("intrinsics", "projection", "distortion", "sensor"):
            value = getattr(self, name)
            if value is not None:
                payload[name] = [str(v) for v in value]
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "CameraFix":
        cfg = cls()
        for key, value in payload.items():
            if not hasattr(cfg, key):
                raise ValueError(f"unknown CameraFix field: {key}")
            setattr(cfg, key, None if value is None else [str(v) for v in value])
        return cfg


@dataclass(slots=True)
class CameraConfig:
    """Camera model plus the parameters fixed during refinement."""

    model: CameraModelSpec = field(default_factory=CameraModelSpec)
    fix: CameraFix = field(default_factory=CameraFix)

    def to_payload(self) -> dict[str, Any]:
        return {"model": self.model.to_payload(), "fix": self.fix.to_payload()}

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "CameraConfig":
        cfg = cls()
        for key, value in payload.items():
            if key == "model":
                cfg.model = (
                    value
                    if isinstance(value, CameraModelSpec)
                    else CameraModelSpec.from_payload(cast(Mapping[str, Any], value))
                )
            elif key == "fix":
                cfg.fix = (
                    value
                    if isinstance(value, CameraFix)
                    else CameraFix.from_payload(cast(Mapping[str, Any], value))
                )
            else:
                raise ValueError(f"unknown CameraConfig field: {key}")
        return cfg


@dataclass(slots=True)
class CameraOverride:
    """Per-camera model and fix lists for one camera of a rig.

    ``camera`` is the rig camera index; ``model`` and ``fix`` replace the
    rig-wide :class:`CameraConfig` for that camera. A camera index out of
    range or listed twice is rejected by the solver before any work.
    """

    camera: int
    model: CameraModelSpec = field(default_factory=CameraModelSpec)
    fix: CameraFix = field(default_factory=CameraFix)

    def to_payload(self) -> dict[str, Any]:
        return {
            "camera": int(self.camera),
            "model": self.model.to_payload(),
            "fix": self.fix.to_payload(),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "CameraOverride":
        if "camera" not in payload:
            raise ValueError("CameraOverride payload missing required 'camera'")
        cfg = cls(camera=int(payload["camera"]))
        for key, value in payload.items():
            if key == "camera":
                continue
            if key == "model":
                cfg.model = (
                    value
                    if isinstance(value, CameraModelSpec)
                    else CameraModelSpec.from_payload(cast(Mapping[str, Any], value))
                )
            elif key == "fix":
                cfg.fix = (
                    value
                    if isinstance(value, CameraFix)
                    else CameraFix.from_payload(cast(Mapping[str, Any], value))
                )
            else:
                raise ValueError(f"unknown CameraOverride field: {key}")
        return cfg


def _camera_config_from_value(value: Any) -> CameraConfig:
    return (
        value
        if isinstance(value, CameraConfig)
        else CameraConfig.from_payload(cast(Mapping[str, Any], value))
    )


def _camera_overrides_from_value(value: Any) -> list[CameraOverride]:
    return [
        v if isinstance(v, CameraOverride) else CameraOverride.from_payload(cast(Mapping[str, Any], v))
        for v in cast(list[Any], value)
    ]


def _as_floats(values: tuple[Any, ...] | list[Any], expected: int, name: str) -> tuple[float, ...]:
    if len(values) != expected:
        raise ValueError(f"{name} must have length {expected}, got {len(values)}")
    return tuple(float(v) for v in values)


def _as_vec2(values: tuple[Any, Any] | list[Any]) -> Vec2:
    x, y = _as_floats(values, 2, "Vec2")
    return (x, y)


def _as_vec3(values: tuple[Any, Any, Any] | list[Any]) -> Vec3:
    x, y, z = _as_floats(values, 3, "Vec3")
    return (x, y, z)


def _as_quat(values: tuple[Any, Any, Any, Any] | list[Any]) -> QuatXyzw:
    x, y, z, w = _as_floats(values, 4, "QuatXyzw")
    return (x, y, z, w)


def _payload_from_maybe_model(value: Any) -> Any:
    if hasattr(value, "to_payload"):
        return value.to_payload()
    return value


@dataclass(slots=True)
class Pose:
    """Rigid pose with quaternion rotation and translation.

    Parameters
    ----------
    rotation_xyzw:
        Quaternion as `(x, y, z, w)`.
    translation_xyz:
        Translation as `(x, y, z)` in meters.
    """

    rotation_xyzw: QuatXyzw = (0.0, 0.0, 0.0, 1.0)
    translation_xyz: Vec3 = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        self.rotation_xyzw = _as_quat(list(self.rotation_xyzw))
        self.translation_xyz = _as_vec3(list(self.translation_xyz))

    def to_payload(self) -> dict[str, list[float]]:
        """Convert to Rust/serde shape."""
        return {
            "rotation": [*self.rotation_xyzw],
            "translation": [*self.translation_xyz],
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "Pose":
        """Build pose from Rust/serde shape."""
        return cls(
            rotation_xyzw=cast(tuple[float, float, float, float], tuple(payload["rotation"])),
            translation_xyz=cast(tuple[float, float, float], tuple(payload["translation"])),
        )


@dataclass(frozen=True, slots=True)
class PointCovariance:
    """Covariance of one observed image point, px²: ``[[xx, xy], [xy, yy]]``.

    It states the point's measurement noise. A calibration whose views carry
    covariances weighs each reprojection residual by its inverse (whitens
    it): an uncertain point counts less, and an elongated one is trusted more
    across its long axis than along it. A uniform weight ``w`` on every point
    corresponds to the covariance ``I / w``.

    Raises :class:`ValueError` unless the entries are finite and the matrix
    is positive definite.
    """

    xx: float
    xy: float
    yy: float

    def __post_init__(self) -> None:
        xx, xy, yy = float(self.xx), float(self.xy), float(self.yy)
        finite = all(math.isfinite(v) for v in (xx, xy, yy))
        if not (finite and xx > 0.0 and yy > 0.0 and xx * yy - xy * xy > 0.0):
            raise ValueError(
                "a point covariance must be finite and positive definite, "
                f"got [[{xx}, {xy}], [{xy}, {yy}]] px²"
            )
        object.__setattr__(self, "xx", xx)
        object.__setattr__(self, "xy", xy)
        object.__setattr__(self, "yy", yy)

    @classmethod
    def isotropic(cls, sigma_px: float) -> "PointCovariance":
        """The covariance ``σ² I`` of a point with ``sigma_px`` per axis."""
        return cls(sigma_px * sigma_px, 0.0, sigma_px * sigma_px)

    def to_payload(self) -> dict[str, float]:
        return {"xx": self.xx, "xy": self.xy, "yy": self.yy}

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "PointCovariance":
        return cls(float(payload["xx"]), float(payload["xy"]), float(payload["yy"]))


@dataclass(slots=True)
class Observation:
    """2D-3D correspondences for one camera view.

    Parameters
    ----------
    points_3d:
        Target points as `(x, y, z)` tuples.
    points_2d:
        Pixel points as `(u, v)` tuples.
    cov_px:
        Optional covariance of each pixel point (:class:`PointCovariance`,
        px²), one per point. Within one calibration input either every view
        states them or none does.
    """

    points_3d: list[Vec3]
    points_2d: list[Vec2]
    cov_px: list[PointCovariance] | None = None

    def __post_init__(self) -> None:
        self.points_3d = [_as_vec3(list(p)) for p in self.points_3d]
        self.points_2d = [_as_vec2(list(p)) for p in self.points_2d]
        if len(self.points_3d) != len(self.points_2d):
            raise ValueError(
                "points_3d and points_2d must have identical length "
                f"(got {len(self.points_3d)} vs {len(self.points_2d)})"
            )
        if self.cov_px is not None:
            if len(self.cov_px) != len(self.points_3d):
                raise ValueError(
                    "cov_px must match point count "
                    f"(got {len(self.cov_px)} vs {len(self.points_3d)})"
                )
            if not all(isinstance(c, PointCovariance) for c in self.cov_px):
                raise TypeError("cov_px entries must be PointCovariance")

    def to_payload(self) -> dict[str, Any]:
        """Convert to Rust/serde shape."""
        payload: dict[str, Any] = {
            "points_3d": [[x, y, z] for x, y, z in self.points_3d],
            "points_2d": [[u, v] for u, v in self.points_2d],
        }
        if self.cov_px is not None:
            payload["cov_px"] = [c.to_payload() for c in self.cov_px]
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "Observation":
        """Build observation from Rust/serde shape."""
        cov = cast(list[Mapping[str, Any]], payload.get("cov_px") or [])
        return cls(
            points_3d=[cast(Vec3, tuple(p)) for p in payload["points_3d"]],
            points_2d=[cast(Vec2, tuple(p)) for p in payload["points_2d"]],
            cov_px=[PointCovariance.from_payload(c) for c in cov] or None,
        )


@dataclass(slots=True)
class PlanarView:
    """Planar calibration view."""

    observation: Observation

    def to_payload(self) -> dict[str, Any]:
        return {"obs": self.observation.to_payload(), "meta": None}


@dataclass(slots=True)
class PlanarIntrinsicsInput:
    """Planar calibration dataset."""

    views: list[PlanarView]

    def to_payload(self) -> dict[str, Any]:
        return {"views": [view.to_payload() for view in self.views]}


@dataclass(slots=True)
class SingleCamHandeyeView:
    """Single-camera hand-eye view."""

    observation: Observation
    base_se3_gripper: Pose

    def to_payload(self) -> dict[str, Any]:
        return {
            "obs": self.observation.to_payload(),
            "meta": {"base_se3_gripper": self.base_se3_gripper.to_payload()},
        }


@dataclass(slots=True)
class SingleCamHandeyeInput:
    """Single-camera hand-eye dataset."""

    views: list[SingleCamHandeyeView]

    def to_payload(self) -> dict[str, Any]:
        return {"views": [view.to_payload() for view in self.views]}


@dataclass(slots=True)
class RigExtrinsicsView:
    """One frame in a multi-camera rig dataset.

    Parameters
    ----------
    cameras:
        Per-camera observation for this frame. Use `None` for a missing camera.
    """

    cameras: list[Observation | None]

    def to_payload(self) -> dict[str, Any]:
        return {
            "obs": {
                "cameras": [
                    None if obs is None else obs.to_payload()
                    for obs in self.cameras
                ]
            },
            "meta": None,
        }


@dataclass(slots=True)
class RigExtrinsicsInput:
    """Multi-camera rig dataset for extrinsics calibration."""

    num_cameras: int
    views: list[RigExtrinsicsView]

    def to_payload(self) -> dict[str, Any]:
        return {
            "num_cameras": int(self.num_cameras),
            "views": [view.to_payload() for view in self.views],
        }


@dataclass(slots=True)
class RigHandeyeView:
    """One frame in rig hand-eye calibration."""

    cameras: list[Observation | None]
    base_se3_gripper: Pose

    def to_payload(self) -> dict[str, Any]:
        return {
            "obs": {
                "cameras": [
                    None if obs is None else obs.to_payload()
                    for obs in self.cameras
                ]
            },
            "meta": {"base_se3_gripper": self.base_se3_gripper.to_payload()},
        }


@dataclass(slots=True)
class RigHandeyeInput:
    """Multi-camera rig hand-eye dataset."""

    num_cameras: int
    views: list[RigHandeyeView]

    def to_payload(self) -> dict[str, Any]:
        return {
            "num_cameras": int(self.num_cameras),
            "views": [view.to_payload() for view in self.views],
        }


@dataclass(slots=True)
class LaserlineView:
    """Single view for laserline-device calibration."""

    observation: Observation
    laser_pixels: list[Vec2]
    laser_weights: list[float] | None = None

    def __post_init__(self) -> None:
        self.laser_pixels = [_as_vec2(list(p)) for p in self.laser_pixels]
        if self.laser_weights is not None:
            if len(self.laser_weights) != len(self.laser_pixels):
                raise ValueError(
                    "laser_weights must match laser_pixels length "
                    f"(got {len(self.laser_weights)} vs {len(self.laser_pixels)})"
                )
            self.laser_weights = [float(w) for w in self.laser_weights]

    def to_payload(self) -> dict[str, Any]:
        meta: dict[str, Any] = {"laser_pixels": [[u, v] for u, v in self.laser_pixels]}
        if self.laser_weights is not None:
            meta["laser_weights"] = self.laser_weights
        return {"obs": self.observation.to_payload(), "meta": meta}


@dataclass(slots=True)
class LaserlineDeviceInput:
    """Dataset for laserline-device calibration."""

    views: list[LaserlineView]

    def to_payload(self) -> list[dict[str, Any]]:
        return [view.to_payload() for view in self.views]


@dataclass(slots=True)
class PlanarIntrinsicsConfig:
    """Configuration for planar intrinsics calibration.

    Grouped configuration: linear-init and non-linear-solve settings live in
    the shared :class:`IntrinsicsInitConfig` / :class:`SolverConfig`
    sub-objects, and ``camera`` (a :class:`CameraConfig`) selects the camera
    model and which of its parameters stay fixed. A tilted (Scheimpflug)
    sensor is ``camera.model.sensor = "scheimpflug"``. ``solver.max_iters``
    defaults to 120: the tilt valley needs more headroom than a plain
    intrinsics solve.
    """

    init: IntrinsicsInitConfig = field(default_factory=IntrinsicsInitConfig)
    solver: SolverConfig = field(default_factory=lambda: SolverConfig(max_iters=120))
    camera: CameraConfig = field(default_factory=CameraConfig)
    fix_poses: list[int] = field(default_factory=list)

    def to_payload(self) -> dict[str, Any]:
        return {
            "init": self.init.to_payload(),
            "solver": self.solver.to_payload(),
            "camera": self.camera.to_payload(),
            "fix_poses": [int(i) for i in self.fix_poses],
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "PlanarIntrinsicsConfig":
        cfg = cls()
        for key, value in mapping.items():
            if key == "init":
                cfg.init = (
                    value
                    if isinstance(value, IntrinsicsInitConfig)
                    else IntrinsicsInitConfig.from_mapping(cast(Mapping[str, Any], value))
                )
            elif key == "solver":
                cfg.solver = (
                    value
                    if isinstance(value, SolverConfig)
                    else SolverConfig.from_mapping(cast(Mapping[str, Any], value))
                )
            elif key == "camera":
                cfg.camera = (
                    value
                    if isinstance(value, CameraConfig)
                    else CameraConfig.from_payload(cast(Mapping[str, Any], value))
                )
            elif key == "fix_poses":
                cfg.fix_poses = [int(i) for i in cast(list[Any], value)]
            else:
                raise ValueError(f"unknown PlanarIntrinsicsConfig field: {key}")
        return cfg


@dataclass(slots=True)
class SingleCamHandeyeConfig:
    """Configuration for single-camera hand-eye calibration.

    Grouped configuration: per-camera linear init, the camera model
    (``camera``, a :class:`CameraConfig`: any distortion model, an untilted
    or a Scheimpflug sensor), hand-eye linear init, non-linear solve, and
    robot-pose refinement each live in their own shared sub-object.
    """

    intrinsics: IntrinsicsInitConfig = field(default_factory=IntrinsicsInitConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    handeye_init: HandeyeInitConfig = field(default_factory=HandeyeInitConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    robot_poses: RobotPoseConfig = field(default_factory=RobotPoseConfig)

    def to_payload(self) -> dict[str, Any]:
        return {
            "intrinsics": self.intrinsics.to_payload(),
            "camera": self.camera.to_payload(),
            "handeye_init": self.handeye_init.to_payload(),
            "solver": self.solver.to_payload(),
            "robot_poses": self.robot_poses.to_payload(),
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "SingleCamHandeyeConfig":
        cfg = cls()
        for key, value in mapping.items():
            if key == "intrinsics":
                cfg.intrinsics = (
                    value
                    if isinstance(value, IntrinsicsInitConfig)
                    else IntrinsicsInitConfig.from_mapping(cast(Mapping[str, Any], value))
                )
            elif key == "camera":
                cfg.camera = _camera_config_from_value(value)
            elif key == "handeye_init":
                cfg.handeye_init = (
                    value
                    if isinstance(value, HandeyeInitConfig)
                    else HandeyeInitConfig.from_mapping(cast(Mapping[str, Any], value))
                )
            elif key == "solver":
                cfg.solver = (
                    value
                    if isinstance(value, SolverConfig)
                    else SolverConfig.from_mapping(cast(Mapping[str, Any], value))
                )
            elif key == "robot_poses":
                cfg.robot_poses = (
                    value
                    if isinstance(value, RobotPoseConfig)
                    else RobotPoseConfig.from_mapping(cast(Mapping[str, Any], value))
                )
            else:
                raise ValueError(f"unknown SingleCamHandeyeConfig field: {key}")
        return cfg


@dataclass(slots=True)
class RigExtrinsicsConfig:
    """Configuration for rig extrinsics calibration.

    Grouped configuration. ``camera`` is the camera model shared by every rig
    camera (a tilted sensor is ``camera.model.sensor = "scheimpflug"``);
    ``camera_overrides`` replaces it for individual cameras, so a rig can mix
    models.
    """

    intrinsics: IntrinsicsInitConfig = field(default_factory=IntrinsicsInitConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    camera_overrides: list[CameraOverride] = field(default_factory=list)
    rig: RigConfig = field(default_factory=RigConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)

    def to_payload(self) -> dict[str, Any]:
        return {
            "intrinsics": self.intrinsics.to_payload(),
            "camera": self.camera.to_payload(),
            "camera_overrides": [o.to_payload() for o in self.camera_overrides],
            "rig": self.rig.to_payload(),
            "solver": self.solver.to_payload(),
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "RigExtrinsicsConfig":
        cfg = cls()
        for key, value in mapping.items():
            if key == "intrinsics":
                cfg.intrinsics = (
                    value
                    if isinstance(value, IntrinsicsInitConfig)
                    else IntrinsicsInitConfig.from_mapping(cast(Mapping[str, Any], value))
                )
            elif key == "camera":
                cfg.camera = _camera_config_from_value(value)
            elif key == "camera_overrides":
                cfg.camera_overrides = _camera_overrides_from_value(value)
            elif key == "rig":
                cfg.rig = (
                    value
                    if isinstance(value, RigConfig)
                    else RigConfig.from_mapping(cast(Mapping[str, Any], value))
                )
            elif key == "solver":
                cfg.solver = (
                    value
                    if isinstance(value, SolverConfig)
                    else SolverConfig.from_mapping(cast(Mapping[str, Any], value))
                )
            else:
                raise ValueError(f"unknown RigExtrinsicsConfig field: {key}")
        return cfg


@dataclass(slots=True)
class HandeyeOptimizeConfig:
    """Final hand-eye optimization options (the ``handeye_optimize`` group of
    :class:`RigHandeyeConfig`).

    ``robot_poses`` sets robot-pose refinement; ``refine_intrinsics``
    (default) refines each camera's intrinsics, ``refine_cam_se3_rig`` the rig
    extrinsics, and ``refine_sensor`` each tilted camera's sensor tilt.
    """

    robot_poses: RobotPoseConfig = field(default_factory=RobotPoseConfig)
    refine_intrinsics: bool = True
    refine_cam_se3_rig: bool = False
    refine_sensor: bool = False

    def to_payload(self) -> dict[str, Any]:
        return {
            "robot_poses": self.robot_poses.to_payload(),
            "refine_intrinsics": bool(self.refine_intrinsics),
            "refine_cam_se3_rig": bool(self.refine_cam_se3_rig),
            "refine_sensor": bool(self.refine_sensor),
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "HandeyeOptimizeConfig":
        cfg = cls()
        for key, value in mapping.items():
            if key == "robot_poses":
                cfg.robot_poses = (
                    value
                    if isinstance(value, RobotPoseConfig)
                    else RobotPoseConfig.from_mapping(cast(Mapping[str, Any], value))
                )
            elif key in ("refine_intrinsics", "refine_cam_se3_rig", "refine_sensor"):
                setattr(cfg, key, bool(value))
            else:
                raise ValueError(f"unknown HandeyeOptimizeConfig field: {key}")
        return cfg


@dataclass(slots=True)
class RigHandeyeConfig:
    """Configuration for rig hand-eye calibration.

    Grouped configuration, sharing sub-objects with the other rig/hand-eye
    configs. ``camera`` / ``camera_overrides`` select the camera model of
    every rig camera and of individual cameras, as in
    :class:`RigExtrinsicsConfig`. (``manual_init`` is still
    Rust-only.)
    """

    intrinsics: IntrinsicsInitConfig = field(default_factory=IntrinsicsInitConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    camera_overrides: list[CameraOverride] = field(default_factory=list)
    rig: RigConfig = field(default_factory=RigConfig)
    handeye_init: HandeyeInitConfig = field(default_factory=HandeyeInitConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    handeye_optimize: HandeyeOptimizeConfig = field(default_factory=HandeyeOptimizeConfig)

    def to_payload(self) -> dict[str, Any]:
        return {
            "intrinsics": self.intrinsics.to_payload(),
            "camera": self.camera.to_payload(),
            "camera_overrides": [o.to_payload() for o in self.camera_overrides],
            "rig": self.rig.to_payload(),
            "handeye_init": self.handeye_init.to_payload(),
            "solver": self.solver.to_payload(),
            "handeye_optimize": self.handeye_optimize.to_payload(),
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "RigHandeyeConfig":
        cfg = cls()
        for key, value in mapping.items():
            if key == "intrinsics":
                cfg.intrinsics = (
                    value
                    if isinstance(value, IntrinsicsInitConfig)
                    else IntrinsicsInitConfig.from_mapping(cast(Mapping[str, Any], value))
                )
            elif key == "camera":
                cfg.camera = _camera_config_from_value(value)
            elif key == "camera_overrides":
                cfg.camera_overrides = _camera_overrides_from_value(value)
            elif key == "rig":
                cfg.rig = (
                    value
                    if isinstance(value, RigConfig)
                    else RigConfig.from_mapping(cast(Mapping[str, Any], value))
                )
            elif key == "handeye_init":
                cfg.handeye_init = (
                    value
                    if isinstance(value, HandeyeInitConfig)
                    else HandeyeInitConfig.from_mapping(cast(Mapping[str, Any], value))
                )
            elif key == "solver":
                cfg.solver = (
                    value
                    if isinstance(value, SolverConfig)
                    else SolverConfig.from_mapping(cast(Mapping[str, Any], value))
                )
            elif key == "handeye_optimize":
                cfg.handeye_optimize = (
                    value
                    if isinstance(value, HandeyeOptimizeConfig)
                    else HandeyeOptimizeConfig.from_mapping(cast(Mapping[str, Any], value))
                )
            else:
                raise ValueError(f"unknown RigHandeyeConfig field: {key}")
        return cfg


@dataclass(slots=True)
class DeviceOptimizeConfig:
    """Bundle-adjustment options for laserline-device calibration.

    The camera's fixed parameters are ``camera.fix`` of
    :class:`LaserlineDeviceConfig`.
    """

    calib_loss: RobustLoss = field(default_factory=lambda: {"Huber": {"scale": 1.0}})
    laser_loss: RobustLoss = field(default_factory=lambda: {"Huber": {"scale": 1.0}})
    calib_weight: float = 1.0
    laser_weight: float = 1.0
    fix_poses: list[int] = field(default_factory=list)
    fix_plane: bool = False
    laser_residual_type: LaserlineResidualType = "LineDistNormalized"

    def to_payload(self) -> dict[str, Any]:
        return {
            "calib_loss": cast(Any, self.calib_loss),
            "laser_loss": cast(Any, self.laser_loss),
            "calib_weight": float(self.calib_weight),
            "laser_weight": float(self.laser_weight),
            "fix_poses": [int(i) for i in self.fix_poses],
            "fix_plane": bool(self.fix_plane),
            "laser_residual_type": self.laser_residual_type,
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "DeviceOptimizeConfig":
        cfg = cls()
        for key, value in mapping.items():
            if not hasattr(cfg, key):
                raise ValueError(f"unknown DeviceOptimizeConfig field: {key}")
            else:
                setattr(cfg, key, value)
        return cfg


@dataclass(slots=True)
class LaserlineDeviceConfig:
    """Configuration for laserline-device calibration.

    ``camera`` (a :class:`CameraConfig`) selects the camera model — any
    distortion model, an untilted or a Scheimpflug sensor — and the
    parameters the bundle adjustment holds fixed. A Scheimpflug sensor's
    tilt is held by default at the manual-init seed (zero when unseeded):
    a device's mount tilt is measured separately. An explicit
    ``camera.fix.sensor`` list estimates it instead (an empty list refines
    both angles).
    """

    init: IntrinsicsInitConfig = field(default_factory=IntrinsicsInitConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    optimize: DeviceOptimizeConfig = field(default_factory=DeviceOptimizeConfig)

    def to_payload(self) -> dict[str, Any]:
        return {
            "init": self.init.to_payload(),
            "camera": self.camera.to_payload(),
            "solver": self.solver.to_payload(),
            "optimize": self.optimize.to_payload(),
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "LaserlineDeviceConfig":
        cfg = cls()
        for key, value in mapping.items():
            if key == "init":
                cfg.init = (
                    value
                    if isinstance(value, IntrinsicsInitConfig)
                    else IntrinsicsInitConfig.from_mapping(cast(Mapping[str, Any], value))
                )
            elif key == "camera":
                cfg.camera = _camera_config_from_value(value)
            elif key == "solver":
                cfg.solver = (
                    value
                    if isinstance(value, SolverConfig)
                    else SolverConfig.from_mapping(cast(Mapping[str, Any], value))
                )
            elif key == "optimize":
                cfg.optimize = (
                    value
                    if isinstance(value, DeviceOptimizeConfig)
                    else DeviceOptimizeConfig.from_mapping(cast(Mapping[str, Any], value))
                )
            else:
                raise ValueError(f"unknown LaserlineDeviceConfig field: {key}")
        return cfg


@dataclass(slots=True)
class PinholeIntrinsics:
    """Typed pinhole intrinsics model."""

    fx: float
    fy: float
    cx: float
    cy: float

    def to_payload(self) -> dict[str, float]:
        """Convert to serde payload shape."""
        return {
            "fx": float(self.fx),
            "fy": float(self.fy),
            "cx": float(self.cx),
            "cy": float(self.cy),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "PinholeIntrinsics":
        """Parse intrinsics from serde payload shape."""
        return cls(
            fx=float(payload["fx"]),
            fy=float(payload["fy"]),
            cx=float(payload["cx"]),
            cy=float(payload["cy"]),
        )


@dataclass(slots=True)
class BrownConradyDistortion:
    """Typed Brown-Conrady distortion model."""

    k1: float
    k2: float
    k3: float
    p1: float
    p2: float
    iters: int

    def to_payload(self) -> dict[str, float | int]:
        """Convert to serde payload shape."""
        return {
            "k1": float(self.k1),
            "k2": float(self.k2),
            "k3": float(self.k3),
            "p1": float(self.p1),
            "p2": float(self.p2),
            "iters": int(self.iters),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "BrownConradyDistortion":
        """Parse distortion from serde payload shape."""
        return cls(
            k1=float(payload["k1"]),
            k2=float(payload["k2"]),
            k3=float(payload["k3"]),
            p1=float(payload["p1"]),
            p2=float(payload["p2"]),
            iters=int(payload["iters"]),
        )


@dataclass(slots=True)
class NoDistortion:
    """No distortion (serde tag ``none``): an empty parameter block."""

    def to_payload(self) -> dict[str, Any]:
        """Convert to serde payload shape (no coefficients)."""
        return {}

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "NoDistortion":
        """Parse from serde payload shape."""
        return cls()


@dataclass(slots=True)
class Division1Distortion:
    """Fitzgibbon single-parameter division distortion (serde tag ``division``)."""

    # `lambda` is a Python keyword; the serde field is `lambda`.
    lambda_: float

    def to_payload(self) -> dict[str, float]:
        """Convert to serde payload shape (flat, no ``type`` tag)."""
        return {"lambda": float(self.lambda_)}

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "Division1Distortion":
        """Parse from serde payload shape."""
        return cls(lambda_=float(payload["lambda"]))


@dataclass(slots=True)
class Rational8Distortion:
    """Rational-polynomial distortion (serde tag ``rational``): ``k1..k6, p1, p2``."""

    k1: float
    k2: float
    k3: float
    k4: float
    k5: float
    k6: float
    p1: float
    p2: float
    iters: int

    def to_payload(self) -> dict[str, float | int]:
        """Convert to serde payload shape (flat, no ``type`` tag)."""
        return {
            "k1": float(self.k1),
            "k2": float(self.k2),
            "k3": float(self.k3),
            "k4": float(self.k4),
            "k5": float(self.k5),
            "k6": float(self.k6),
            "p1": float(self.p1),
            "p2": float(self.p2),
            "iters": int(self.iters),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "Rational8Distortion":
        """Parse from serde payload shape."""
        return cls(
            k1=float(payload["k1"]),
            k2=float(payload["k2"]),
            k3=float(payload["k3"]),
            k4=float(payload["k4"]),
            k5=float(payload["k5"]),
            k6=float(payload["k6"]),
            p1=float(payload["p1"]),
            p2=float(payload["p2"]),
            iters=int(payload["iters"]),
        )


@dataclass(slots=True)
class ThinPrism9Distortion:
    """Brown-Conrady + thin-prism distortion (serde tag ``thin_prism``):
    ``k1..k3, p1, p2, s1..s4``."""

    k1: float
    k2: float
    k3: float
    p1: float
    p2: float
    s1: float
    s2: float
    s3: float
    s4: float
    iters: int

    def to_payload(self) -> dict[str, float | int]:
        """Convert to serde payload shape (flat, no ``type`` tag)."""
        return {
            "k1": float(self.k1),
            "k2": float(self.k2),
            "k3": float(self.k3),
            "p1": float(self.p1),
            "p2": float(self.p2),
            "s1": float(self.s1),
            "s2": float(self.s2),
            "s3": float(self.s3),
            "s4": float(self.s4),
            "iters": int(self.iters),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ThinPrism9Distortion":
        """Parse from serde payload shape."""
        return cls(
            k1=float(payload["k1"]),
            k2=float(payload["k2"]),
            k3=float(payload["k3"]),
            p1=float(payload["p1"]),
            p2=float(payload["p2"]),
            s1=float(payload["s1"]),
            s2=float(payload["s2"]),
            s3=float(payload["s3"]),
            s4=float(payload["s4"]),
            iters=int(payload["iters"]),
        )


@dataclass(slots=True)
class KannalaBrandt4Distortion:
    """Kannala-Brandt radial polynomial (serde tag ``kannala_brandt4``):
    ``r_d = r (1 + k1 r² + k2 r⁴ + k3 r⁶ + k4 r⁸)``. With an
    :class:`EquidistantProjection` it is the Kannala-Brandt fisheye model
    (OpenCV ``fisheye``)."""

    k1: float
    k2: float
    k3: float
    k4: float
    iters: int

    def to_payload(self) -> dict[str, float | int]:
        """Convert to serde payload shape (flat, no ``type`` tag)."""
        return {
            "k1": float(self.k1),
            "k2": float(self.k2),
            "k3": float(self.k3),
            "k4": float(self.k4),
            "iters": int(self.iters),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "KannalaBrandt4Distortion":
        """Parse from serde payload shape."""
        return cls(
            k1=float(payload["k1"]),
            k2=float(payload["k2"]),
            k3=float(payload["k3"]),
            k4=float(payload["k4"]),
            iters=int(payload["iters"]),
        )


# Any distortion model a result's camera may carry.
Distortion = (
    NoDistortion
    | BrownConradyDistortion
    | Division1Distortion
    | Rational8Distortion
    | ThinPrism9Distortion
    | KannalaBrandt4Distortion
)

# Single source of truth mapping the serde ``DistortionParams`` tag to its parser
# and back. The tags are the export-side ``DistortionParams`` names (``none``,
# ``division``, ``rational``, ``thin_prism``), which differ from the
# ``DistortionKind`` config spellings (``division1``, ``rational8``,
# ``thin_prism9``).
_DISTORTION_BY_TAG: dict[str, type] = {
    "none": NoDistortion,
    "brown_conrady5": BrownConradyDistortion,
    "division": Division1Distortion,
    "rational": Rational8Distortion,
    "thin_prism": ThinPrism9Distortion,
    "kannala_brandt4": KannalaBrandt4Distortion,
}
_DISTORTION_TAG_BY_TYPE: dict[type, str] = {v: k for k, v in _DISTORTION_BY_TAG.items()}


def _distortion_from_payload(payload: Mapping[str, Any]) -> Distortion:
    """Parse a tagged ``DistortionParams`` payload into the matching dataclass.

    An absent ``type`` tag defaults to Brown-Conrady.
    """
    tag = payload.get("type", "brown_conrady5")
    parser = _DISTORTION_BY_TAG.get(tag)
    if parser is None:
        raise ValueError(f"unsupported distortion payload type: {tag!r}")
    return cast(Distortion, parser.from_payload(payload))


def _distortion_to_payload(distortion: Distortion) -> dict[str, Any]:
    """Serialize a distortion dataclass to its tagged ``DistortionParams`` shape."""
    tag = _DISTORTION_TAG_BY_TYPE[type(distortion)]
    return {"type": tag, **distortion.to_payload()}


@dataclass(slots=True)
class ScheimpflugSensor:
    """Typed Scheimpflug sensor tilt model."""

    tilt_x: float
    tilt_y: float

    def to_payload(self) -> dict[str, float]:
        """Convert to serde payload shape."""
        return {
            "tilt_x": float(self.tilt_x),
            "tilt_y": float(self.tilt_y),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ScheimpflugSensor":
        """Parse Scheimpflug tilt from serde payload shape."""
        tilt_x = payload.get("tilt_x")
        tilt_y = payload.get("tilt_y")
        if tilt_x is None:
            raise ValueError("Scheimpflug sensor payload missing tilt_x")
        if tilt_y is None:
            raise ValueError("Scheimpflug sensor payload missing tilt_y")
        return cls(tilt_x=float(tilt_x), tilt_y=float(tilt_y))


@dataclass(slots=True)
class PinholeProjection:
    """Pinhole (perspective) projection (serde tag ``pinhole``); no parameters."""

    def to_payload(self) -> dict[str, Any]:
        """Convert to serde payload shape (no parameters)."""
        return {}

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "PinholeProjection":
        """Parse from serde payload shape."""
        return cls()


@dataclass(slots=True)
class EquidistantProjection:
    """Equidistant (fisheye) projection (serde tag ``equidistant``): the
    normalized radius is the incidence angle in radians, so rays beyond 90°
    project. No parameters."""

    def to_payload(self) -> dict[str, Any]:
        """Convert to serde payload shape (no parameters)."""
        return {}

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "EquidistantProjection":
        """Parse from serde payload shape."""
        return cls()


# Any projection model a camera may use.
Projection = PinholeProjection | EquidistantProjection

_PROJECTION_BY_TAG: dict[str, type] = {
    "pinhole": PinholeProjection,
    "equidistant": EquidistantProjection,
}
_PROJECTION_TAG_BY_TYPE: dict[type, str] = {v: k for k, v in _PROJECTION_BY_TAG.items()}


@dataclass(slots=True)
class IdentitySensor:
    """Sensor plane perpendicular to the optical axis (serde tag ``identity``);
    no parameters."""

    def to_payload(self) -> dict[str, Any]:
        """Convert to serde payload shape (no parameters)."""
        return {}

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "IdentitySensor":
        """Parse from serde payload shape."""
        return cls()


# Any sensor model a camera may use.
Sensor = IdentitySensor | ScheimpflugSensor

_SENSOR_BY_TAG: dict[str, type] = {
    "identity": IdentitySensor,
    "scheimpflug": ScheimpflugSensor,
}
_SENSOR_TAG_BY_TYPE: dict[type, str] = {v: k for k, v in _SENSOR_BY_TAG.items()}


def _tagged_from_payload(kind: str, table: Mapping[str, type], payload: Mapping[str, Any]) -> Any:
    tag = payload.get("type")
    parser = table.get(cast(str, tag))
    if parser is None:
        raise ValueError(f"unsupported {kind} payload type: {tag!r}")
    return parser.from_payload(payload)


@dataclass(slots=True)
class Camera:
    """A calibrated camera: intrinsics plus one model per slot (projection,
    distortion, sensor), the ``CameraParams`` of a result.

    ``projection`` defaults to :class:`PinholeProjection` and ``sensor`` to
    :class:`IdentitySensor`; ``distortion`` is any :data:`Distortion` model.
    """

    intrinsics: PinholeIntrinsics
    distortion: Distortion
    projection: Projection = field(default_factory=PinholeProjection)
    sensor: Sensor = field(default_factory=IdentitySensor)

    def to_payload(self) -> dict[str, Any]:
        """Convert to the ``CameraParams`` serde payload shape."""
        return {
            "projection": {
                "type": _PROJECTION_TAG_BY_TYPE[type(self.projection)],
                **self.projection.to_payload(),
            },
            "distortion": _distortion_to_payload(self.distortion),
            "sensor": {"type": _SENSOR_TAG_BY_TYPE[type(self.sensor)], **self.sensor.to_payload()},
            "intrinsics": self.intrinsics.to_payload(),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "Camera":
        """Parse from the ``CameraParams`` serde payload shape.

        Raises ``ValueError`` for a projection or sensor this package does
        not know (a general homography sensor included).
        """
        return cls(
            intrinsics=PinholeIntrinsics.from_payload(cast(Mapping[str, Any], payload["intrinsics"])),
            distortion=_distortion_from_payload(cast(Mapping[str, Any], payload["distortion"])),
            projection=_tagged_from_payload(
                "projection", _PROJECTION_BY_TAG, cast(Mapping[str, Any], payload["projection"])
            ),
            sensor=_tagged_from_payload(
                "sensor", _SENSOR_BY_TAG, cast(Mapping[str, Any], payload["sensor"])
            ),
        )


@dataclass(slots=True)
class LaserPlane:
    """Typed laser plane representation."""

    normal_xyz: Vec3
    distance: float

    def __post_init__(self) -> None:
        self.normal_xyz = _as_vec3(list(self.normal_xyz))
        self.distance = float(self.distance)

    def to_payload(self) -> dict[str, Any]:
        """Convert to Rust/serde shape.

        Rust's ``LaserPlane`` (``nalgebra::Unit<Vector3<f64>>``) deserializes
        ``normal`` as a plain array ``[x, y, z]``.
        """
        return {
            "normal": list(self.normal_xyz),
            "distance": float(self.distance),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "LaserPlane":
        """Parse laser plane from serde payload."""
        normal_value = payload["normal"]
        if isinstance(normal_value, Mapping):
            if "coords" in normal_value:
                normal_value = normal_value["coords"]
            elif "data" in normal_value:
                normal_value = normal_value["data"]
            else:
                raise ValueError("unsupported laser plane normal payload shape")
        return cls(
            normal_xyz=cast(Vec3, tuple(normal_value)),
            distance=float(payload["distance"]),
        )


@dataclass(slots=True)
class LaserlineParams:
    """Typed parameter payload for laserline optimization results.

    ``camera`` is a :class:`Camera`.
    """

    camera: Camera
    poses: list[Pose]
    plane: LaserPlane

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "LaserlineParams":
        return cls(
            camera=_camera_from_payload(cast(Mapping[str, Any], payload["camera"])),
            poses=[
                Pose.from_payload(cast(Mapping[str, Any], pose))
                for pose in cast(list[Any], payload["poses"])
            ],
            plane=LaserPlane.from_payload(cast(Mapping[str, Any], payload["plane"])),
        )


@dataclass(slots=True)
class LaserlineStats:
    """Typed summary statistics for laserline results."""

    mean_reproj_error: float
    mean_laser_error: float
    per_view_reproj_errors: list[float]
    per_view_laser_errors: list[float]

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "LaserlineStats":
        return cls(
            mean_reproj_error=float(payload["mean_reproj_error"]),
            mean_laser_error=float(payload["mean_laser_error"]),
            per_view_reproj_errors=[
                float(v) for v in cast(list[Any], payload["per_view_reproj_errors"])
            ],
            per_view_laser_errors=[
                float(v) for v in cast(list[Any], payload["per_view_laser_errors"])
            ],
        )


@dataclass(slots=True)
class ResidualNoise:
    """Estimated noise level of one kind of residual.

    ``sigma`` is the Gaussian-equivalent standard deviation of one scalar
    residual (one pixel axis for reprojection), from the median absolute
    residual, so outliers do not inflate it. Its unit depends on
    :attr:`whitened`: when false, pixels for ``"reprojection"`` and
    ``"laser_line"`` and metres for ``"laser_plane"``; when true (the input
    stated per-point covariances), ``sigma`` is the dimensionless ``σ̂₀`` of
    the whitened reprojection residuals, about 1 when the stated covariances
    describe the noise.
    """

    kind: str
    sigma: float
    count: int
    camera: int | None = None
    """The camera, when the result estimates each camera's noise apart (the
    rig laserline); ``None`` for one level shared by every camera."""
    whitened: bool = False
    """Whether ``sigma`` is the dimensionless ``σ̂₀`` of residuals whitened
    by stated per-point covariances (the payload's ``sigma0``) rather than a
    level in the residual's unit (its ``sigma``)."""

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ResidualNoise":
        camera = payload.get("camera")
        whitened = "sigma0" in payload
        return cls(
            kind=str(payload["kind"]),
            sigma=float(payload["sigma0" if whitened else "sigma"]),
            count=int(payload["count"]),
            camera=None if camera is None else int(camera),
            whitened=whitened,
        )


@dataclass(slots=True)
class ViewCovariance:
    """Covariance of one view's own parameters (e.g. its target pose)."""

    view: int
    labels: list[str]
    covariance: list[list[float]]

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ViewCovariance":
        return cls(
            view=int(payload["view"]),
            labels=[str(v) for v in cast(list[Any], payload["labels"])],
            covariance=[
                [float(v) for v in cast(list[Any], row)]
                for row in cast(list[Any], payload["covariance"])
            ],
        )


@dataclass(slots=True)
class ParameterUncertainty:
    """Covariance of a result's estimated parameters.

    Parameters are named by labels such as ``"fx"``, ``"k1"`` or ``"tilt_x"``;
    a pose's six labels end in ``rx, ry, rz`` (radians) and ``tx, ty, tz``
    (metres). The covariance is first order, at the solution, and conditional
    on the parameters listed in :attr:`fixed`.

    :attr:`view_scatter` is how many times the shared parameters' variance
    across views exceeds what the residual noise predicts (errors shared by
    a view's corners: lens model, target, detector bias), measured by
    leaving out one view at a time; the covariance is already scaled by it.
    ``None`` when the views scatter no more than the noise predicts, and
    below six views.

    When :attr:`available` is false, :attr:`reason` says why
    (``"not_converged"``, ``"rank_deficient"``, ``"at_bound"``,
    ``"no_redundancy"`` or ``"not_computed"``) and :attr:`reason_labels` names
    the parameters involved.
    """

    available: bool
    labels: list[str] = field(default_factory=list)
    covariance: list[list[float]] = field(default_factory=list)
    views: list[ViewCovariance] = field(default_factory=list)
    fixed: list[str] = field(default_factory=list)
    noise: list[ResidualNoise] = field(default_factory=list)
    num_residuals: int = 0
    dof: int = 0
    view_scatter: float | None = None
    reason: str | None = None
    reason_labels: list[str] = field(default_factory=list)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any] | None) -> "ParameterUncertainty":
        if payload is None:
            return cls(available=False, reason="not_computed")
        if payload["status"] == "unavailable":
            reason = cast(Mapping[str, Any], payload["reason"])
            names = reason.get("unobservable", reason.get("labels", []))
            return cls(
                available=False,
                reason=str(reason["kind"]),
                reason_labels=[str(v) for v in cast(list[Any], names)],
            )
        return cls(
            available=True,
            labels=[str(v) for v in cast(list[Any], payload["labels"])],
            covariance=[
                [float(v) for v in cast(list[Any], row)]
                for row in cast(list[Any], payload["covariance"])
            ],
            views=[
                ViewCovariance.from_payload(cast(Mapping[str, Any], v))
                for v in cast(list[Any], payload.get("views", []))
            ],
            fixed=[str(v) for v in cast(list[Any], payload.get("fixed", []))],
            noise=[
                ResidualNoise.from_payload(cast(Mapping[str, Any], n))
                for n in cast(list[Any], payload["noise"])
            ],
            num_residuals=int(payload["num_residuals"]),
            dof=int(payload["dof"]),
            view_scatter=(
                None
                if payload.get("view_scatter") is None
                else float(payload["view_scatter"])
            ),
        )

    def _index(self, label: str) -> int:
        if not self.available:
            raise ValueError(f"no covariance: {self.reason}")
        try:
            return self.labels.index(label)
        except ValueError:
            raise KeyError(f"no estimated parameter {label!r}") from None

    def std_dev(self, label: str) -> float:
        """Standard deviation of the parameter ``label``.

        Raises :class:`KeyError` for a label that is not estimated (see
        :attr:`labels` and :attr:`fixed`) and :class:`ValueError` when the
        covariance is not available.
        """
        i = self._index(label)
        return max(self.covariance[i][i], 0.0) ** 0.5

    def correlation(self, a: str, b: str) -> float:
        """Correlation coefficient of two parameters, in ``[-1, 1]``."""
        i, j = self._index(a), self._index(b)
        denom = (self.covariance[i][i] * self.covariance[j][j]) ** 0.5
        if denom == 0.0:
            return 0.0
        return max(-1.0, min(1.0, self.covariance[i][j] / denom))


@dataclass(slots=True)
class ProjectionUncertainty:
    """How precisely a camera's estimated parameters place a camera-frame ray
    in the image.

    Each value is the major-axis standard deviation, in pixels, of where the
    ray through a grid node lands. Nodes run evenly from pixel ``(0, 0)`` to
    ``(width - 1, height - 1)``, corners included, row by row from the
    top-left; ``None`` marks a node no ray reaches. Only the camera's own
    parameters vary (intrinsics, distortion, sensor tilt), so the figure does
    not depend on range, and it is conservative for uses that re-estimate the
    camera pose, which absorb part of a principal-point error.
    """

    image_size: tuple[int, int]
    columns: int
    rows: int
    sigma: list[float | None]
    center: float
    corner: float | None
    max: float

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ProjectionUncertainty":
        size = cast(list[Any], payload["image_size"])
        corner = payload.get("corner")
        return cls(
            image_size=(int(size[0]), int(size[1])),
            columns=int(payload["columns"]),
            rows=int(payload["rows"]),
            sigma=[None if v is None else float(v) for v in cast(list[Any], payload["sigma"])],
            center=float(payload["center"]),
            corner=None if corner is None else float(corner),
            max=float(payload["max"]),
        )

    def grid(self) -> list[list[float | None]]:
        """The node values as ``rows`` lists of ``columns`` each, top row first."""
        return [self.sigma[r * self.columns : (r + 1) * self.columns] for r in range(self.rows)]


@dataclass(slots=True)
class QualityThresholds:
    """Limits of the quality checks run by :func:`~vision_calibration.assess`.

    The defaults are limits a well-planned calibration of a perspective camera
    clears. ``expected_feature_noise_px`` has none: it depends on the detector,
    the target and the optics, and while it is ``None`` the variance-factor
    check is not applicable.
    """

    #: Expected noise σ₀ of one feature coordinate, pixels per axis (> 0).
    expected_feature_noise_px: float | None = None
    #: Significance level α of the two-sided χ² test of σ̂²/σ₀², in (0, 1).
    variance_test_alpha: float = 0.01
    #: Projection σ anywhere in the image above which the check warns, pixels.
    max_projection_sigma_px: float = 1.0
    #: Share of the image's 8 × 6 coverage cells with a feature below which
    #: the coverage check warns.
    coverage_warn_below: float = 0.5
    #: Share below which the coverage check fails; at most ``coverage_warn_below``.
    coverage_fail_below: float = 0.3
    #: Mean ``|r| / σ̂`` of one view above which it is named as an outlier.
    max_view_residual: float = 3.0
    #: Absolute correlation of two parameters of one camera above which the
    #: pair is named (``fx``/``fy`` excepted).
    max_parameter_correlation: float = 0.99

    def to_payload(self) -> dict[str, Any]:
        return {
            "expected_feature_noise_px": (
                None
                if self.expected_feature_noise_px is None
                else float(self.expected_feature_noise_px)
            ),
            "variance_test_alpha": float(self.variance_test_alpha),
            "max_projection_sigma_px": float(self.max_projection_sigma_px),
            "coverage_warn_below": float(self.coverage_warn_below),
            "coverage_fail_below": float(self.coverage_fail_below),
            "max_view_residual": float(self.max_view_residual),
            "max_parameter_correlation": float(self.max_parameter_correlation),
        }


@dataclass(slots=True)
class Measurement:
    """A quality check's measured value and the limit that decided its status."""

    #: The measured value; the worst one for a check over several items.
    value: float
    #: The limit it was compared to: the one it crossed, or for a pass the nearest.
    threshold: float
    #: ``"pixel"``, ``"fraction"``, ``"ratio"``, ``"sigma"`` or ``"correlation"``.
    unit: str

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "Measurement":
        return cls(
            value=float(payload["value"]),
            threshold=float(payload["threshold"]),
            unit=str(payload["unit"]),
        )


@dataclass(slots=True)
class QualityCheck:
    """One check of a :class:`QualityReport`."""

    #: What it examines: ``"observability"``, ``"convergence"``,
    #: ``"variance_factor"``, ``"projection_sigma"``, ``"feature_coverage"``,
    #: ``"view_outliers"`` or ``"parameter_correlation"``.
    kind: str
    #: ``"pass"``, ``"warn"``, ``"fail"`` or ``"not_applicable"``.
    status: str
    #: One sentence on the outcome, or on why the check was not applicable.
    message: str
    #: The camera, for a per-camera check; ``None`` for a result-wide one.
    camera: int | None = None
    #: The measured value; ``None`` when not applicable, and for the
    #: observability and convergence checks.
    measurement: Measurement | None = None
    #: Named items: parameter labels, parameter pairs (``"fx/cx"``) or views
    #: (``"view 3"``).
    subjects: list[str] = field(default_factory=list)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "QualityCheck":
        camera = payload.get("camera")
        measurement = payload.get("measurement")
        return cls(
            kind=str(payload["kind"]),
            status=str(payload["status"]),
            message=str(payload["message"]),
            camera=None if camera is None else int(camera),
            measurement=(
                None
                if measurement is None
                else Measurement.from_payload(cast(Mapping[str, Any], measurement))
            ),
            subjects=[str(s) for s in cast(list[Any], payload.get("subjects", []))],
        )


@dataclass(slots=True)
class QualityReport:
    """The verdict on a calibration result and the checks behind it.

    ``verdict`` is the worst status of the assessed checks (``"pass"`` <
    ``"warn"`` < ``"fail"``); a check that was not applicable does not count,
    and a report where no check could be assessed is ``"warn"``.
    """

    verdict: str
    checks: list[QualityCheck]

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "QualityReport":
        return cls(
            verdict=str(payload["verdict"]),
            checks=[
                QualityCheck.from_payload(cast(Mapping[str, Any], c))
                for c in cast(list[Any], payload["checks"])
            ],
        )


@dataclass(slots=True)
class CandidateGrid:
    """A deterministic set of candidate target poses for
    :func:`~vision_calibration.candidate_poses`.

    For each of ``regions`` (``(columns, rows)``) of the image, row by row from
    the top-left, the target's centroid is put on the ray through the region's
    centre at ``distance_m`` metres, facing the camera; the poses are the
    fronto-parallel one, then tilts of ``+tilt_rad`` and ``-tilt_rad`` about the
    target's x axis and about its y axis (only the fronto-parallel one when
    ``tilt_rad`` is 0).
    """

    #: Distance from the camera centre to the target's centroid, metres (> 0).
    distance_m: float
    #: Tilt out of the fronto-parallel pose, radians, in ``[0, π/2)``;
    #: 0.3–0.6 is typical.
    tilt_rad: float
    #: Regions of the image grid, ``(columns, rows)``, each at least 1.
    regions: tuple[int, int]

    def to_payload(self) -> dict[str, Any]:
        return {
            "distance_m": float(self.distance_m),
            "tilt_rad": float(self.tilt_rad),
            "regions": [int(self.regions[0]), int(self.regions[1])],
        }


@dataclass(slots=True)
class PredictedUncertainty:
    """The covariance a planar calibration is predicted to have with candidate
    views added, from :func:`~vision_calibration.predict_with_views`.

    ``covariance`` has the result's labels, fixed list and noise levels; it
    predicts no per-view blocks. ``projection`` is the predicted projection
    uncertainty over the image. ``visible_points`` gives, per candidate, the
    target points that land in the image (fewer than four add nothing), and
    ``effective_noise_px`` the pixel σ the result's covariance implies, which
    scales the candidates' information.
    """

    covariance: ParameterUncertainty
    projection: ProjectionUncertainty
    visible_points: list[int]
    effective_noise_px: float

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "PredictedUncertainty":
        covariance = dict(cast(Mapping[str, Any], payload["covariance"]))
        covariance["status"] = "available"
        return cls(
            covariance=ParameterUncertainty.from_payload(covariance),
            projection=ProjectionUncertainty.from_payload(
                cast(Mapping[str, Any], payload["projection"])
            ),
            visible_points=[int(v) for v in cast(list[Any], payload["visible_points"])],
            effective_noise_px=float(payload["effective_noise_px"]),
        )


@dataclass(slots=True)
class RankedCandidate:
    """One candidate view scored by :func:`~vision_calibration.rank_candidates`."""

    #: The candidate's position in the list that was ranked.
    index: int
    #: Target points that land in the image; fewer than four add nothing.
    visible_points: int
    #: The largest projection σ over the image with this view added, pixels.
    max_projection_sigma_px: float
    #: The current largest projection σ minus ``max_projection_sigma_px``, pixels.
    projection_sigma_reduction_px: float
    #: ``ln det Σ − ln det Σ′``: the reduction of the covariance's log-volume.
    log_det_reduction: float

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "RankedCandidate":
        return cls(
            index=int(payload["index"]),
            visible_points=int(payload["visible_points"]),
            max_projection_sigma_px=float(payload["max_projection_sigma_px"]),
            projection_sigma_reduction_px=float(payload["projection_sigma_reduction_px"]),
            log_det_reduction=float(payload["log_det_reduction"]),
        )



class _ExportResiduals:
    """The typed per-feature residuals of a result's export JSON."""

    __slots__ = ()
    export: dict[str, Any]

    @property
    def per_feature_residuals(self) -> PerFeatureResiduals:
        """Every target corner's and laser pixel's residual, as numpy tables
        (see :class:`~vision_calibration.residuals.PerFeatureResiduals`)."""
        diagnostics = cast(Mapping[str, Any], self.export.get("diagnostics") or {})
        return PerFeatureResiduals.from_payload(
            cast(Mapping[str, Any] | None, diagnostics.get("per_feature_residuals"))
        )


def _camera_from_payload(payload: Mapping[str, Any]) -> Camera:
    return Camera.from_payload(payload)


def _cameras_from_payload(payload: Mapping[str, Any]) -> list[Camera]:
    return [
        _camera_from_payload(cast(Mapping[str, Any], c))
        for c in cast(list[Any], payload["cameras"])
    ]


@dataclass(slots=True)
class PlanarIntrinsicsExport(_ExportResiduals):
    """Result from :func:`vision_calibration.run_planar_intrinsics`.

    ``camera.distortion`` is one of the :data:`Distortion` variants, matching
    ``camera.model.distortion`` in the config (Brown-Conrady by default).
    ``camera.sensor`` is a :class:`ScheimpflugSensor` when the config selected a
    Scheimpflug sensor, otherwise an :class:`IdentitySensor`.
    """

    camera: Camera
    camera_se3_target: list[Pose]
    mean_reproj_error: float
    per_cam_reproj_errors: list[float]
    uncertainty: ParameterUncertainty = field(
        default_factory=lambda: ParameterUncertainty(available=False, reason="not_computed")
    )
    export: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)
    """The result as the library's export JSON, as
    :func:`~vision_calibration.projection_uncertainty` reads it."""

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "PlanarIntrinsicsExport":
        params = cast(Mapping[str, Any], payload["params"])
        poses = [
            Pose.from_payload(cast(Mapping[str, Any], p))
            for p in cast(list[Any], params["camera_se3_target"])
        ]
        return cls(
            export=dict(payload),
            camera=_camera_from_payload(cast(Mapping[str, Any], params["camera"])),
            camera_se3_target=poses,
            mean_reproj_error=float(payload["mean_reproj_error"]),
            per_cam_reproj_errors=[
                float(v) for v in cast(list[Any], payload["per_cam_reproj_errors"])
            ],
            uncertainty=ParameterUncertainty.from_payload(
                cast(Mapping[str, Any] | None, payload.get("uncertainty"))
            ),
        )


@dataclass(slots=True)
class SingleCamHandeyeExport(_ExportResiduals):
    """Result from :func:`vision_calibration.run_single_cam_handeye`.

    ``camera`` is a :class:`Camera`.
    """

    camera: Camera
    handeye_mode: HandeyeMode
    gripper_se3_camera: Pose | None
    camera_se3_base: Pose | None
    base_se3_target: Pose | None
    gripper_se3_target: Pose | None
    robot_deltas: list[list[float]] | None
    mean_reproj_error: float
    per_cam_reproj_errors: list[float]
    robot_pose_sigma: RobotPoseSigma | None = None
    """The robot's pose error the corrections were weighed with (estimated
    with ``robot_poses.estimate_sigma``); ``None`` without refinement."""
    uncertainty: ParameterUncertainty = field(
        default_factory=lambda: ParameterUncertainty(available=False, reason="not_computed")
    )
    export: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)
    """The result as the library's export JSON, as
    :func:`~vision_calibration.projection_uncertainty` reads it."""

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "SingleCamHandeyeExport":
        def _pose(name: str) -> Pose | None:
            value = payload.get(name)
            if value is None:
                return None
            return Pose.from_payload(cast(Mapping[str, Any], value))

        return cls(
            export=dict(payload),
            camera=_camera_from_payload(cast(Mapping[str, Any], payload["camera"])),
            handeye_mode=cast(HandeyeMode, payload["handeye_mode"]),
            gripper_se3_camera=_pose("gripper_se3_camera"),
            camera_se3_base=_pose("camera_se3_base"),
            base_se3_target=_pose("base_se3_target"),
            gripper_se3_target=_pose("gripper_se3_target"),
            robot_deltas=cast(list[list[float]] | None, payload.get("robot_deltas")),
            robot_pose_sigma=RobotPoseSigma.from_payload(
                cast(Mapping[str, Any] | None, payload.get("robot_pose_sigma"))
            ),
            mean_reproj_error=float(payload["mean_reproj_error"]),
            per_cam_reproj_errors=[float(v) for v in cast(list[Any], payload["per_cam_reproj_errors"])],
            uncertainty=ParameterUncertainty.from_payload(
                cast(Mapping[str, Any] | None, payload.get("uncertainty"))
            ),
        )


@dataclass(slots=True)
class RigExtrinsicsExport(_ExportResiduals):
    """Result from :func:`vision_calibration.run_rig_extrinsics`."""

    cameras: list[Camera]
    cam_se3_rig: list[Pose]
    mean_reproj_error: float
    per_cam_reproj_errors: list[float]
    uncertainty: ParameterUncertainty = field(
        default_factory=lambda: ParameterUncertainty(available=False, reason="not_computed")
    )
    export: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)
    """The result as the library's export JSON, as
    :func:`~vision_calibration.projection_uncertainty` reads it."""

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "RigExtrinsicsExport":
        return cls(
            export=dict(payload),
            cameras=_cameras_from_payload(payload),
            cam_se3_rig=[Pose.from_payload(cast(Mapping[str, Any], p)) for p in cast(list[Any], payload["cam_se3_rig"])],
            mean_reproj_error=float(payload["mean_reproj_error"]),
            per_cam_reproj_errors=[float(v) for v in cast(list[Any], payload["per_cam_reproj_errors"])],
            uncertainty=ParameterUncertainty.from_payload(
                cast(Mapping[str, Any] | None, payload.get("uncertainty"))
            ),
        )


@dataclass(slots=True)
class RigHandeyeExport(_ExportResiduals):
    """Result from :func:`vision_calibration.run_rig_handeye`.

    Each entry of ``cameras`` is a :class:`Camera`.
    """

    cameras: list[Camera]
    cam_se3_rig: list[Pose]
    handeye_mode: HandeyeMode
    gripper_se3_rig: Pose | None
    rig_se3_base: Pose | None
    base_se3_target: Pose | None
    gripper_se3_target: Pose | None
    robot_deltas: list[list[float]] | None
    mean_reproj_error: float
    per_cam_reproj_errors: list[float]
    robot_pose_sigma: RobotPoseSigma | None = None
    """The robot's pose error the corrections were weighed with (estimated
    with ``robot_poses.estimate_sigma``); ``None`` without refinement."""
    uncertainty: ParameterUncertainty = field(
        default_factory=lambda: ParameterUncertainty(available=False, reason="not_computed")
    )
    export: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)
    """The result as the library's export JSON, as
    :func:`~vision_calibration.projection_uncertainty` reads it."""

    def to_payload(self) -> dict[str, Any]:
        """Convert to Rust/serde shape (``RigHandeyeExport``).

        Used by helpers like :func:`pixel_to_gripper_point` that pass a
        result back across the FFI boundary.
        """

        def _pose(p: Pose | None) -> Any:
            return None if p is None else p.to_payload()

        return {
            "cameras": [c.to_payload() for c in self.cameras],
            "cam_se3_rig": [p.to_payload() for p in self.cam_se3_rig],
            "handeye_mode": self.handeye_mode,
            "gripper_se3_rig": _pose(self.gripper_se3_rig),
            "rig_se3_base": _pose(self.rig_se3_base),
            "base_se3_target": _pose(self.base_se3_target),
            "gripper_se3_target": _pose(self.gripper_se3_target),
            "robot_deltas": self.robot_deltas,
            "robot_pose_sigma": (
                None if self.robot_pose_sigma is None else self.robot_pose_sigma.to_payload()
            ),
            "mean_reproj_error": float(self.mean_reproj_error),
            "per_cam_reproj_errors": [float(v) for v in self.per_cam_reproj_errors],
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "RigHandeyeExport":
        def _pose(name: str) -> Pose | None:
            value = payload.get(name)
            if value is None:
                return None
            return Pose.from_payload(cast(Mapping[str, Any], value))

        return cls(
            export=dict(payload),
            cameras=_cameras_from_payload(payload),
            cam_se3_rig=[
                Pose.from_payload(cast(Mapping[str, Any], p))
                for p in cast(list[Any], payload["cam_se3_rig"])
            ],
            handeye_mode=cast(HandeyeMode, payload["handeye_mode"]),
            gripper_se3_rig=_pose("gripper_se3_rig"),
            rig_se3_base=_pose("rig_se3_base"),
            base_se3_target=_pose("base_se3_target"),
            gripper_se3_target=_pose("gripper_se3_target"),
            robot_deltas=cast(list[list[float]] | None, payload.get("robot_deltas")),
            robot_pose_sigma=RobotPoseSigma.from_payload(
                cast(Mapping[str, Any] | None, payload.get("robot_pose_sigma"))
            ),
            mean_reproj_error=float(payload["mean_reproj_error"]),
            per_cam_reproj_errors=[
                float(v) for v in cast(list[Any], payload["per_cam_reproj_errors"])
            ],
            uncertainty=ParameterUncertainty.from_payload(
                cast(Mapping[str, Any] | None, payload.get("uncertainty"))
            ),
        )


@dataclass(slots=True)
class LaserlineDeviceExport(_ExportResiduals):
    """Result from :func:`vision_calibration.run_laserline_device`."""

    params: LaserlineParams
    stats: LaserlineStats
    mean_reproj_error: float
    per_cam_reproj_errors: list[float]
    uncertainty: ParameterUncertainty = field(
        default_factory=lambda: ParameterUncertainty(available=False, reason="not_computed")
    )
    """Covariance of the camera's free parameters and the laser plane
    (``plane.nx`` … ``plane.d``), with each view's target pose."""
    export: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)
    """The result as the library's export JSON, as
    :func:`~vision_calibration.projection_uncertainty` reads it."""

    @property
    def mean_laser_error(self) -> float:
        """Mean laser residual (units depend on residual type)."""
        return self.stats.mean_laser_error

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "LaserlineDeviceExport":
        return cls(
            export=dict(payload),
            params=LaserlineParams.from_payload(cast(Mapping[str, Any], payload["params"])),
            stats=LaserlineStats.from_payload(cast(Mapping[str, Any], payload["stats"])),
            mean_reproj_error=float(payload["mean_reproj_error"]),
            per_cam_reproj_errors=[float(v) for v in cast(list[Any], payload["per_cam_reproj_errors"])],
            uncertainty=ParameterUncertainty.from_payload(
                cast(Mapping[str, Any] | None, payload.get("uncertainty"))
            ),
        )


# ─── Rig laserline device ─────────────────────────────────────────────────────


@dataclass(slots=True)
class RigUpstreamCalibration:
    """Frozen upstream calibration for rig laserline device.

    Typically derived from a :class:`RigHandeyeExport`; ``cameras``
    are ``CameraParams`` payloads (projection, distortion, sensor,
    intrinsics) as found in the export.
    """

    cameras: list[dict[str, Any]]
    cam_se3_rig: list[Pose]
    rig_se3_target: list[Pose]

    def to_payload(self) -> dict[str, Any]:
        return {
            "cameras": self.cameras,
            "cam_se3_rig": [p.to_payload() for p in self.cam_se3_rig],
            "rig_se3_target": [p.to_payload() for p in self.rig_se3_target],
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "RigUpstreamCalibration":
        return cls(
            cameras=cast(list[dict[str, Any]], payload["cameras"]),
            cam_se3_rig=[
                Pose.from_payload(cast(Mapping[str, Any], p))
                for p in cast(list[Any], payload["cam_se3_rig"])
            ],
            rig_se3_target=[
                Pose.from_payload(cast(Mapping[str, Any], p))
                for p in cast(list[Any], payload["rig_se3_target"])
            ],
        )


@dataclass(slots=True)
class RigLaserlineView:
    """One frame in a rig laserline dataset.

    Parameters
    ----------
    cameras:
        Per-camera target correspondences. Use ``None`` for a missing camera.
    laser_pixels:
        Per-camera laser pixel observations. Use ``None`` for a missing camera.
    """

    cameras: list[Observation | None]
    laser_pixels: list[list[Vec2] | None]

    def to_payload(self) -> dict[str, Any]:
        return {
            "cameras": [None if obs is None else obs.to_payload() for obs in self.cameras],
            "laser_pixels": [
                None if px is None else [[u, v] for u, v in px]
                for px in self.laser_pixels
            ],
        }


@dataclass(slots=True)
class RigLaserlineDataset:
    """Multi-camera rig laserline dataset."""

    num_cameras: int
    views: list[RigLaserlineView]

    def to_payload(self) -> dict[str, Any]:
        return {
            "num_cameras": int(self.num_cameras),
            "views": [view.to_payload() for view in self.views],
        }


@dataclass(slots=True)
class RigLaserlineDeviceInput:
    """Input for rig laserline device calibration.

    The ``initial_planes_cam`` warm-start field accepts the same
    [`LaserPlane`] dataclass returned by
    [`RigLaserlineDeviceExport.laser_planes_cam`], so the canonical
    restart pattern ``initial_planes_cam=previous.laser_planes_cam``
    works directly without manual conversion.
    """

    dataset: RigLaserlineDataset
    upstream: RigUpstreamCalibration
    initial_planes_cam: list[LaserPlane] | None = None

    def to_payload(self) -> dict[str, Any]:
        initial: list[dict[str, Any]] | None = None
        if self.initial_planes_cam is not None:
            initial = [plane.to_payload() for plane in self.initial_planes_cam]
        return {
            "dataset": self.dataset.to_payload(),
            "upstream": self.upstream.to_payload(),
            "initial_planes_cam": initial,
        }


@dataclass(slots=True)
class RigLaserlineDeviceConfig:
    """Configuration for rig laserline device calibration.

    Grouped configuration. ``solver.max_iters`` defaults to 200: this stage
    refines only per-camera laser-plane parameters against an
    already-frozen rig geometry (1 DOF per view per camera), so iterations
    are nearly free.
    """

    solver: SolverConfig = field(default_factory=lambda: SolverConfig(max_iters=200))
    laser_residual_type: LaserlineResidualType = "LineDistNormalized"

    def to_payload(self) -> dict[str, Any]:
        return {
            "solver": self.solver.to_payload(),
            "laser_residual_type": self.laser_residual_type,
        }

    @classmethod
    def from_mapping(
        cls, mapping: Mapping[str, Any]
    ) -> "RigLaserlineDeviceConfig":
        cfg = cls()
        for key, value in mapping.items():
            if key == "solver":
                cfg.solver = (
                    value
                    if isinstance(value, SolverConfig)
                    else SolverConfig.from_mapping(cast(Mapping[str, Any], value))
                )
            elif key == "laser_residual_type":
                cfg.laser_residual_type = cast(LaserlineResidualType, value)
            else:
                raise ValueError(f"unknown RigLaserlineDeviceConfig field: {key}")
        return cfg


@dataclass(slots=True)
class RigLaserlineDeviceExport(_ExportResiduals):
    """Result from :func:`vision_calibration.run_rig_laserline_device`."""

    laser_planes_rig: list[LaserPlane]
    laser_planes_cam: list[LaserPlane]
    per_camera_stats: list[LaserlineStats]
    uncertainty: ParameterUncertainty = field(
        default_factory=lambda: ParameterUncertainty(available=False, reason="not_computed")
    )
    export: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)
    """The result as the library's export JSON, as
    :func:`~vision_calibration.projection_uncertainty` reads it."""

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "RigLaserlineDeviceExport":
        return cls(
            export=dict(payload),
            laser_planes_rig=[
                LaserPlane.from_payload(cast(Mapping[str, Any], p))
                for p in cast(list[Any], payload["laser_planes_rig"])
            ],
            laser_planes_cam=[
                LaserPlane.from_payload(cast(Mapping[str, Any], p))
                for p in cast(list[Any], payload["laser_planes_cam"])
            ],
            per_camera_stats=[
                LaserlineStats.from_payload(cast(Mapping[str, Any], s))
                for s in cast(list[Any], payload["per_camera_stats"])
            ],
            uncertainty=ParameterUncertainty.from_payload(
                cast(Mapping[str, Any] | None, payload.get("uncertainty"))
            ),
        )


# ─── Rig hand-eye laserline (joint) ───────────────────────────────────────────


@dataclass(slots=True)
class RigHandeyeLaserlineView:
    """One frame in a joint rig hand-eye laserline dataset.

    Parameters
    ----------
    cameras:
        Per-camera target correspondences. Use ``None`` for a missing camera.
    laser_pixels:
        Per-camera laser pixel observations. Use ``None`` for a missing camera.
    base_se3_gripper:
        Robot gripper pose at this frame.
    """

    cameras: list[Observation | None]
    laser_pixels: list[list[Vec2] | None]
    base_se3_gripper: Pose

    def to_payload(self) -> dict[str, Any]:
        return {
            "obs": {
                "cameras": [
                    None if obs is None else obs.to_payload() for obs in self.cameras
                ],
                "laser_pixels": [
                    None if px is None else [[u, v] for u, v in px]
                    for px in self.laser_pixels
                ],
            },
            "meta": {"base_se3_gripper": self.base_se3_gripper.to_payload()},
        }


@dataclass(slots=True)
class RigHandeyeLaserlineInput:
    """Joint rig hand-eye laserline dataset (``RigHandeyeLaserlineInput``)."""

    num_cameras: int
    views: list[RigHandeyeLaserlineView]

    def to_payload(self) -> dict[str, Any]:
        return {
            "num_cameras": int(self.num_cameras),
            "views": [view.to_payload() for view in self.views],
        }


@dataclass(slots=True)
class JointOptimizeConfig:
    """Final joint bundle-adjustment stage settings.

    ``fix`` holds one :class:`CameraFix` per camera, each named in that
    camera's model. Empty (the default) gives every camera the stage
    default: intrinsics and the low-order radial terms (``k1``, ``k2``; the
    division model's ``lambda``) refined, the higher-order distortion terms
    and the sensor tilt held. Laser and target residuals carry independent
    robust losses (``calib_loss`` / ``laser_loss``); the shared
    ``solver.robust_loss`` is not consulted by this stage.
    """

    solver: SolverConfig = field(default_factory=lambda: SolverConfig(max_iters=30))
    laser_residual_type: LaserlineResidualType = "PointToPlane"
    calib_loss: RobustLoss = "None"
    laser_loss: RobustLoss = "None"
    calib_weight: float = 1.0
    laser_weight: float = 1.0e4
    fix: list[CameraFix] = field(default_factory=list)
    fix_handeye: bool = False
    fix_target_ref: bool = False
    robot_poses: RobotPoseConfig = field(default_factory=RobotPoseConfig)

    def to_payload(self) -> dict[str, Any]:
        return {
            "solver": self.solver.to_payload(),
            "laser_residual_type": self.laser_residual_type,
            "calib_loss": cast(Any, self.calib_loss),
            "laser_loss": cast(Any, self.laser_loss),
            "calib_weight": float(self.calib_weight),
            "laser_weight": float(self.laser_weight),
            "fix": [f.to_payload() for f in self.fix],
            "fix_handeye": bool(self.fix_handeye),
            "fix_target_ref": bool(self.fix_target_ref),
            "robot_poses": self.robot_poses.to_payload(),
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "JointOptimizeConfig":
        cfg = cls()
        for key, value in mapping.items():
            if key == "solver":
                cfg.solver = (
                    value
                    if isinstance(value, SolverConfig)
                    else SolverConfig.from_mapping(cast(Mapping[str, Any], value))
                )
            elif key == "fix":
                cfg.fix = [
                    f if isinstance(f, CameraFix) else CameraFix.from_payload(f)
                    for f in cast(list[Any], value)
                ]
            elif key == "robot_poses":
                cfg.robot_poses = (
                    value
                    if isinstance(value, RobotPoseConfig)
                    else RobotPoseConfig.from_mapping(cast(Mapping[str, Any], value))
                )
            elif not hasattr(cfg, key):
                raise ValueError(f"unknown JointOptimizeConfig field: {key}")
            else:
                setattr(cfg, key, value)
        return cfg


@dataclass(slots=True)
class RigHandeyeLaserlineConfig:
    """Configuration for joint rig hand-eye laserline calibration.

    Three warm-started stages: the rig hand-eye stage (``handeye``), the
    frozen-geometry laser plane init (``laserline_init``, defaults to a
    point-to-plane residual), and the final joint bundle adjustment
    (``joint_optimize``).
    """

    handeye: RigHandeyeConfig = field(
        default_factory=RigHandeyeConfig
    )
    laserline_init: RigLaserlineDeviceConfig = field(
        default_factory=lambda: RigLaserlineDeviceConfig(
            laser_residual_type="PointToPlane"
        )
    )
    joint_optimize: JointOptimizeConfig = field(
        default_factory=JointOptimizeConfig
    )

    def to_payload(self) -> dict[str, Any]:
        return {
            "handeye": self.handeye.to_payload(),
            "laserline_init": self.laserline_init.to_payload(),
            "joint_optimize": self.joint_optimize.to_payload(),
        }

    @classmethod
    def from_mapping(
        cls, mapping: Mapping[str, Any]
    ) -> "RigHandeyeLaserlineConfig":
        cfg = cls()
        for key, value in mapping.items():
            if key == "handeye":
                cfg.handeye = (
                    value
                    if isinstance(value, RigHandeyeConfig)
                    else RigHandeyeConfig.from_mapping(
                        cast(Mapping[str, Any], value)
                    )
                )
            elif key == "laserline_init":
                cfg.laserline_init = (
                    value
                    if isinstance(value, RigLaserlineDeviceConfig)
                    else RigLaserlineDeviceConfig.from_mapping(
                        cast(Mapping[str, Any], value)
                    )
                )
            elif key == "joint_optimize":
                cfg.joint_optimize = (
                    value
                    if isinstance(value, JointOptimizeConfig)
                    else JointOptimizeConfig.from_mapping(
                        cast(Mapping[str, Any], value)
                    )
                )
            else:
                raise ValueError(
                    f"unknown RigHandeyeLaserlineConfig field: {key}"
                )
        return cfg


@dataclass(slots=True)
class RigHandeyeLaserlinePerCamStats:
    """Per-camera statistics for a joint rig hand-eye laserline result."""

    mean_reproj_error_px: float
    reproj_count: int
    max_reproj_error_px: float
    reproj_histogram_px: list[int]
    mean_laser_err_m: float
    max_laser_err_m: float
    laser_histogram_m: list[int]
    mean_laser_err_px: float
    max_laser_err_px: float
    laser_count: int

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "RigHandeyeLaserlinePerCamStats":
        return cls(
            mean_reproj_error_px=float(payload["mean_reproj_error_px"]),
            reproj_count=int(payload["reproj_count"]),
            max_reproj_error_px=float(payload["max_reproj_error_px"]),
            reproj_histogram_px=[int(v) for v in cast(list[Any], payload["reproj_histogram_px"])],
            mean_laser_err_m=float(payload["mean_laser_err_m"]),
            max_laser_err_m=float(payload["max_laser_err_m"]),
            laser_histogram_m=[int(v) for v in cast(list[Any], payload["laser_histogram_m"])],
            mean_laser_err_px=float(payload["mean_laser_err_px"]),
            max_laser_err_px=float(payload["max_laser_err_px"]),
            laser_count=int(payload["laser_count"]),
        )


@dataclass(slots=True)
class RigHandeyeLaserlineExport(_ExportResiduals):
    """Result from :func:`vision_calibration.run_rig_handeye_laserline`."""

    laser_planes_rig: list[LaserPlane]
    laser_planes_cam: list[LaserPlane]
    per_camera_stats: list[RigHandeyeLaserlinePerCamStats]
    cameras: list[Camera]
    cam_se3_rig: list[Pose]
    rig_se3_target: list[Pose]
    handeye_mode: HandeyeMode
    gripper_se3_rig: Pose | None
    rig_se3_base: Pose | None
    base_se3_target: Pose | None
    gripper_se3_target: Pose | None
    robot_deltas: list[list[float]] | None
    mean_reproj_error: float
    per_cam_reproj_errors: list[float]
    robot_pose_sigma: RobotPoseSigma | None = None
    """The robot's pose error the corrections were weighed with (estimated
    with ``robot_poses.estimate_sigma``); ``None`` without refinement."""
    uncertainty: ParameterUncertainty = field(
        default_factory=lambda: ParameterUncertainty(available=False, reason="not_computed")
    )
    export: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)
    """The result as the library's export JSON, as
    :func:`~vision_calibration.projection_uncertainty` reads it."""

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "RigHandeyeLaserlineExport":
        def _pose(name: str) -> Pose | None:
            value = payload.get(name)
            if value is None:
                return None
            return Pose.from_payload(cast(Mapping[str, Any], value))

        return cls(
            export=dict(payload),
            laser_planes_rig=[
                LaserPlane.from_payload(cast(Mapping[str, Any], p))
                for p in cast(list[Any], payload["laser_planes_rig"])
            ],
            laser_planes_cam=[
                LaserPlane.from_payload(cast(Mapping[str, Any], p))
                for p in cast(list[Any], payload["laser_planes_cam"])
            ],
            per_camera_stats=[
                RigHandeyeLaserlinePerCamStats.from_payload(cast(Mapping[str, Any], s))
                for s in cast(list[Any], payload["per_camera_stats"])
            ],
            cameras=_cameras_from_payload(payload),
            cam_se3_rig=[
                Pose.from_payload(cast(Mapping[str, Any], p))
                for p in cast(list[Any], payload["cam_se3_rig"])
            ],
            rig_se3_target=[
                Pose.from_payload(cast(Mapping[str, Any], p))
                for p in cast(list[Any], payload["rig_se3_target"])
            ],
            handeye_mode=cast(HandeyeMode, payload["handeye_mode"]),
            gripper_se3_rig=_pose("gripper_se3_rig"),
            rig_se3_base=_pose("rig_se3_base"),
            base_se3_target=_pose("base_se3_target"),
            gripper_se3_target=_pose("gripper_se3_target"),
            robot_deltas=cast(list[list[float]] | None, payload.get("robot_deltas")),
            robot_pose_sigma=RobotPoseSigma.from_payload(
                cast(Mapping[str, Any] | None, payload.get("robot_pose_sigma"))
            ),
            mean_reproj_error=float(payload["mean_reproj_error"]),
            per_cam_reproj_errors=[
                float(v) for v in cast(list[Any], payload["per_cam_reproj_errors"])
            ],
            uncertainty=ParameterUncertainty.from_payload(
                cast(Mapping[str, Any] | None, payload.get("uncertainty"))
            ),
        )


def normalize_input_payload(input_value: Any) -> Any:
    """Convert high-level model inputs to serde payloads."""
    return _payload_from_maybe_model(input_value)
