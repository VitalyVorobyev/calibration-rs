"""Public high-level models for :mod:`vision_calibration`.

These dataclasses provide a Python-native API surface with explicit fields,
defaults, and docstrings. Wrapper functions in :mod:`vision_calibration`
accept these models and convert them to the serde payloads expected by Rust.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, cast

from .types import DistortionModel, HandEyeMode, LaserlineResidualType, RobustLoss

Vec2 = tuple[float, float]
Vec3 = tuple[float, float, float]
QuatXyzw = tuple[float, float, float, float]

_DEFAULT_INTRINSICS_FIX_MASK: dict[str, bool] = {
    "fx": False,
    "fy": False,
    "cx": False,
    "cy": False,
}
_DEFAULT_DISTORTION_FIX_MASK: dict[str, bool] = {
    "k1": False,
    "k2": False,
    "k3": True,
    "p1": False,
    "p2": False,
}
_DEFAULT_SENSOR_INIT: dict[str, float] = {
    "tilt_x": 0.0,
    "tilt_y": 0.0,
}
_DEFAULT_SCHEIMPFLUG_FIX_MASK: dict[str, bool] = {
    "tilt_x": False,
    "tilt_y": False,
}
# `SensorMode::Scheimpflug` and the joint BA both freeze tilt by default.
_FIXED_SCHEIMPFLUG_MASK: dict[str, bool] = {
    "tilt_x": True,
    "tilt_y": True,
}
# `DistortionFixMask::radial_only()` — the per-camera Scheimpflug default and the
# joint-BA / Scheimpflug-intrinsics camera-fix default (k1, k2 free; k3, p1, p2
# fixed).
_RADIAL_ONLY_DISTORTION_FIX_MASK: dict[str, bool] = {
    "k1": False,
    "k2": False,
    "k3": True,
    "p1": True,
    "p2": True,
}


@dataclass(slots=True)
class IntrinsicsInitConfig:
    """Shared per-camera linear-initialization stage.

    Zhang's method plus an iterative Brown-Conrady distortion fit. Shared by
    every intrinsics-bearing problem's ``init``/``intrinsics`` config group.
    """

    init_iterations: int = 2
    fix_k3: bool = True
    fix_tangential: bool = False
    zero_skew: bool = True

    def to_payload(self) -> dict[str, Any]:
        return {
            "init_iterations": int(self.init_iterations),
            "fix_k3": bool(self.fix_k3),
            "fix_tangential": bool(self.fix_tangential),
            "zero_skew": bool(self.zero_skew),
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

    def to_payload(self) -> dict[str, Any]:
        return {
            "max_iters": int(self.max_iters),
            "verbosity": int(self.verbosity),
            "robust_loss": cast(Any, self.robust_loss),
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
    """Shared robot-pose refinement settings for hand-eye BA."""

    refine: bool = True
    rot_sigma: float = 0.5 * 3.141592653589793 / 180.0
    trans_sigma: float = 0.001

    def to_payload(self) -> dict[str, Any]:
        return {
            "refine": bool(self.refine),
            "rot_sigma": float(self.rot_sigma),
            "trans_sigma": float(self.trans_sigma),
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
class HandeyeInitConfig:
    """Shared hand-eye linear-initialization settings (Tsai-Lenz DLT)."""

    handeye_mode: HandEyeMode = "EyeInHand"
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
    refine_intrinsics_in_rig_ba: bool = False

    def to_payload(self) -> dict[str, Any]:
        return {
            "reference_camera_idx": int(self.reference_camera_idx),
            "refine_intrinsics_in_rig_ba": bool(self.refine_intrinsics_in_rig_ba),
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
class CameraFixMask:
    """Combined per-camera intrinsics + distortion fix mask."""

    intrinsics: dict[str, bool] = field(default_factory=lambda: dict(_DEFAULT_INTRINSICS_FIX_MASK))
    distortion: dict[str, bool] = field(default_factory=lambda: dict(_DEFAULT_DISTORTION_FIX_MASK))

    def to_payload(self) -> dict[str, Any]:
        intrinsics = dict(_DEFAULT_INTRINSICS_FIX_MASK)
        intrinsics.update(self.intrinsics)
        distortion = dict(_DEFAULT_DISTORTION_FIX_MASK)
        distortion.update(self.distortion)
        return {"intrinsics": intrinsics, "distortion": distortion}

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "CameraFixMask":
        cfg = cls()
        for key, value in mapping.items():
            if key == "intrinsics":
                cfg.intrinsics = dict(value)
            elif key == "distortion":
                cfg.distortion = dict(value)
            else:
                raise ValueError(f"unknown CameraFixMask field: {key}")
        return cfg


@dataclass(slots=True)
class PinholeSensorMode:
    """Pinhole rig sensor mode — mirrors ``SensorMode::Pinhole``.

    Standard pinhole + Brown-Conrady projection with no sensor tilt. This is
    the default flavour for both rig workflows (``RigExtrinsics`` /
    ``RigHandeye``).
    """

    def to_payload(self) -> dict[str, Any]:
        return {"kind": "Pinhole"}


@dataclass(slots=True)
class ScheimpflugSensorMode:
    """Scheimpflug rig sensor mode — mirrors ``SensorMode::Scheimpflug``.

    Adds per-camera tilt parameters so a rig of Scheimpflug cameras can be
    calibrated. ``distortion_model`` mirrors the Rust field but only
    ``"brown_conrady5"`` (the default) is accepted by the Brown-Conrady-typed
    rig bundle adjustment.
    """

    init_tilt_x: float = 0.0
    init_tilt_y: float = 0.0
    fix_scheimpflug: dict[str, bool] = field(
        default_factory=lambda: dict(_DEFAULT_SCHEIMPFLUG_FIX_MASK)
    )
    distortion_mask_in_percam_ba: dict[str, bool] = field(
        default_factory=lambda: dict(_RADIAL_ONLY_DISTORTION_FIX_MASK)
    )
    refine_scheimpflug_in_rig_ba: bool = False
    distortion_model: DistortionModel = "brown_conrady5"

    def to_payload(self) -> dict[str, Any]:
        fix_scheimpflug = dict(_DEFAULT_SCHEIMPFLUG_FIX_MASK)
        fix_scheimpflug.update(self.fix_scheimpflug)
        distortion_mask = dict(_RADIAL_ONLY_DISTORTION_FIX_MASK)
        distortion_mask.update(self.distortion_mask_in_percam_ba)
        return {
            "kind": "Scheimpflug",
            "init_tilt_x": float(self.init_tilt_x),
            "init_tilt_y": float(self.init_tilt_y),
            "fix_scheimpflug": fix_scheimpflug,
            "distortion_mask_in_percam_ba": distortion_mask,
            "refine_scheimpflug_in_rig_ba": bool(self.refine_scheimpflug_in_rig_ba),
            "distortion_model": self.distortion_model,
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "ScheimpflugSensorMode":
        cfg = cls()
        for key, value in mapping.items():
            if key == "kind":
                continue
            if key == "fix_scheimpflug":
                cfg.fix_scheimpflug = dict(cast(Mapping[str, bool], value))
            elif key == "distortion_mask_in_percam_ba":
                cfg.distortion_mask_in_percam_ba = dict(cast(Mapping[str, bool], value))
            elif not hasattr(cfg, key):
                raise ValueError(f"unknown ScheimpflugSensorMode field: {key}")
            else:
                setattr(cfg, key, value)
        return cfg


# Rig sensor flavour selector (pinhole vs Scheimpflug), mirroring the Rust
# ``SensorMode`` enum. Illegal states are unrepresentable: a ``PinholeSensorMode``
# carries no tilt fields, and only ``ScheimpflugSensorMode`` carries them.
SensorMode = PinholeSensorMode | ScheimpflugSensorMode


def _sensor_mode_from_mapping(value: Any) -> SensorMode:
    """Build a :data:`SensorMode` from a model instance or a serde mapping."""
    if isinstance(value, (PinholeSensorMode, ScheimpflugSensorMode)):
        return value
    mapping = cast(Mapping[str, Any], value)
    # Rust's internally-tagged `SensorMode` errors on a missing tag rather than
    # defaulting; match that so a malformed payload fails loudly instead of
    # silently discarding Scheimpflug fields.
    if "kind" not in mapping:
        raise ValueError("SensorMode payload missing required 'kind' tag")
    kind = mapping["kind"]
    if kind == "Pinhole":
        return PinholeSensorMode()
    if kind == "Scheimpflug":
        return ScheimpflugSensorMode.from_mapping(mapping)
    raise ValueError(f"unknown SensorMode kind: {kind!r}")


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


@dataclass(slots=True)
class Observation:
    """2D-3D correspondences for one camera view.

    Parameters
    ----------
    points_3d:
        Target points as `(x, y, z)` tuples.
    points_2d:
        Pixel points as `(u, v)` tuples.
    weights:
        Optional per-point non-negative weights.
    """

    points_3d: list[Vec3]
    points_2d: list[Vec2]
    weights: list[float] | None = None

    def __post_init__(self) -> None:
        self.points_3d = [_as_vec3(list(p)) for p in self.points_3d]
        self.points_2d = [_as_vec2(list(p)) for p in self.points_2d]
        if len(self.points_3d) != len(self.points_2d):
            raise ValueError(
                "points_3d and points_2d must have identical length "
                f"(got {len(self.points_3d)} vs {len(self.points_2d)})"
            )
        if self.weights is not None:
            if len(self.weights) != len(self.points_3d):
                raise ValueError(
                    "weights must match point count "
                    f"(got {len(self.weights)} vs {len(self.points_3d)})"
                )
            self.weights = [float(w) for w in self.weights]

    def to_payload(self) -> dict[str, Any]:
        """Convert to Rust/serde shape."""
        payload: dict[str, Any] = {
            "points_3d": [[x, y, z] for x, y, z in self.points_3d],
            "points_2d": [[u, v] for u, v in self.points_2d],
        }
        if self.weights is not None:
            payload["weights"] = list(self.weights)
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "Observation":
        """Build observation from Rust/serde shape."""
        return cls(
            points_3d=[cast(Vec3, tuple(p)) for p in payload["points_3d"]],
            points_2d=[cast(Vec2, tuple(p)) for p in payload["points_2d"]],
            weights=cast(list[float] | None, payload.get("weights")),
        )


@dataclass(slots=True)
class PlanarView:
    """Planar calibration view."""

    observation: Observation

    def to_payload(self) -> dict[str, Any]:
        return {"obs": self.observation.to_payload(), "meta": None}


@dataclass(slots=True)
class PlanarDataset:
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
class SingleCamHandeyeDataset:
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
class RigExtrinsicsDataset:
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
class RigHandeyeDataset:
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
        return {
            "obs": self.observation.to_payload(),
            "meta": {
                "laser_pixels": [[u, v] for u, v in self.laser_pixels],
                "laser_weights": self.laser_weights,
            },
        }


@dataclass(slots=True)
class LaserlineDataset:
    """Dataset for laserline-device calibration."""

    views: list[LaserlineView]

    def to_payload(self) -> list[dict[str, Any]]:
        return [view.to_payload() for view in self.views]


@dataclass(slots=True)
class PlanarCalibrationConfig:
    """Configuration for planar intrinsics calibration.

    Grouped configuration: linear-init and non-linear-solve settings live in
    the shared :class:`IntrinsicsInitConfig` / :class:`SolverConfig`
    sub-objects. ``distortion_model`` selects which distortion model is fitted
    (default ``"brown_conrady5"``); the extended models (``"rational8"``,
    ``"thin_prism9"``, ``"division1"``) are supported by the two single-camera
    intrinsics workflows (PlanarIntrinsics and ScheimpflugIntrinsics) but not by
    the rig / hand-eye / laserline consumers, which are Brown-Conrady-typed.
    """

    init: IntrinsicsInitConfig = field(default_factory=IntrinsicsInitConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    distortion_model: DistortionModel = "brown_conrady5"
    fix_camera: CameraFixMask = field(default_factory=CameraFixMask)
    fix_poses: list[int] = field(default_factory=list)

    def to_payload(self) -> dict[str, Any]:
        return {
            "init": self.init.to_payload(),
            "solver": self.solver.to_payload(),
            "distortion_model": self.distortion_model,
            "fix_camera": self.fix_camera.to_payload(),
            "fix_poses": [int(i) for i in self.fix_poses],
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "PlanarCalibrationConfig":
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
            elif key == "distortion_model":
                cfg.distortion_model = cast(DistortionModel, value)
            elif key == "fix_camera":
                cfg.fix_camera = (
                    value
                    if isinstance(value, CameraFixMask)
                    else CameraFixMask.from_mapping(cast(Mapping[str, Any], value))
                )
            elif key == "fix_poses":
                cfg.fix_poses = [int(i) for i in cast(list[Any], value)]
            else:
                raise ValueError(f"unknown PlanarCalibrationConfig field: {key}")
        return cfg


@dataclass(slots=True)
class SingleCamHandeyeCalibrationConfig:
    """Configuration for single-camera hand-eye calibration.

    Grouped configuration: per-camera linear init, hand-eye linear init,
    non-linear solve, and robot-pose refinement each live in their own
    shared sub-object.
    """

    intrinsics: IntrinsicsInitConfig = field(default_factory=IntrinsicsInitConfig)
    handeye_init: HandeyeInitConfig = field(default_factory=HandeyeInitConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    robot_poses: RobotPoseConfig = field(default_factory=RobotPoseConfig)

    def to_payload(self) -> dict[str, Any]:
        return {
            "intrinsics": self.intrinsics.to_payload(),
            "handeye_init": self.handeye_init.to_payload(),
            "solver": self.solver.to_payload(),
            "robot_poses": self.robot_poses.to_payload(),
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "SingleCamHandeyeCalibrationConfig":
        cfg = cls()
        for key, value in mapping.items():
            if key == "intrinsics":
                cfg.intrinsics = (
                    value
                    if isinstance(value, IntrinsicsInitConfig)
                    else IntrinsicsInitConfig.from_mapping(cast(Mapping[str, Any], value))
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
            elif key == "robot_poses":
                cfg.robot_poses = (
                    value
                    if isinstance(value, RobotPoseConfig)
                    else RobotPoseConfig.from_mapping(cast(Mapping[str, Any], value))
                )
            else:
                raise ValueError(f"unknown SingleCamHandeyeCalibrationConfig field: {key}")
        return cfg


@dataclass(slots=True)
class RigExtrinsicsCalibrationConfig:
    """Configuration for rig extrinsics calibration.

    Grouped configuration. ``sensor`` selects the rig sensor flavour
    (:class:`PinholeSensorMode` default, or :class:`ScheimpflugSensorMode` for
    a rig of Scheimpflug cameras).
    """

    intrinsics: IntrinsicsInitConfig = field(default_factory=IntrinsicsInitConfig)
    sensor: SensorMode = field(default_factory=PinholeSensorMode)
    rig: RigConfig = field(default_factory=RigConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)

    def to_payload(self) -> dict[str, Any]:
        return {
            "intrinsics": self.intrinsics.to_payload(),
            "sensor": self.sensor.to_payload(),
            "rig": self.rig.to_payload(),
            "solver": self.solver.to_payload(),
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "RigExtrinsicsCalibrationConfig":
        cfg = cls()
        for key, value in mapping.items():
            if key == "intrinsics":
                cfg.intrinsics = (
                    value
                    if isinstance(value, IntrinsicsInitConfig)
                    else IntrinsicsInitConfig.from_mapping(cast(Mapping[str, Any], value))
                )
            elif key == "sensor":
                cfg.sensor = _sensor_mode_from_mapping(value)
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
                raise ValueError(f"unknown RigExtrinsicsCalibrationConfig field: {key}")
        return cfg


@dataclass(slots=True)
class HandeyeBaConfig:
    """Final hand-eye bundle-adjustment options.

    Renamed and reshaped from ``RigHandeyeBaConfig``: robot-pose refinement
    now lives in the shared ``RobotPoseConfig`` group, and
    ``refine_cam_se3_rig_in_handeye_ba`` / ``refine_scheimpflug_in_handeye_ba``
    shorten to ``refine_cam_se3_rig`` / ``refine_scheimpflug``.
    """

    robot_poses: RobotPoseConfig = field(default_factory=RobotPoseConfig)
    refine_cam_se3_rig: bool = False
    refine_scheimpflug: bool = False

    def to_payload(self) -> dict[str, Any]:
        return {
            "robot_poses": self.robot_poses.to_payload(),
            "refine_cam_se3_rig": bool(self.refine_cam_se3_rig),
            "refine_scheimpflug": bool(self.refine_scheimpflug),
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "HandeyeBaConfig":
        cfg = cls()
        for key, value in mapping.items():
            if key == "robot_poses":
                cfg.robot_poses = (
                    value
                    if isinstance(value, RobotPoseConfig)
                    else RobotPoseConfig.from_mapping(cast(Mapping[str, Any], value))
                )
            elif key in ("refine_cam_se3_rig", "refine_scheimpflug"):
                setattr(cfg, key, bool(value))
            else:
                raise ValueError(f"unknown HandeyeBaConfig field: {key}")
        return cfg


@dataclass(slots=True)
class RigHandeyeCalibrationConfig:
    """Configuration for rig hand-eye calibration.

    Grouped configuration, sharing sub-objects with the other rig/hand-eye
    configs. ``sensor`` selects the rig sensor flavour
    (:class:`PinholeSensorMode` default, or :class:`ScheimpflugSensorMode`).
    (``manual_init`` is still Rust-only.)
    """

    intrinsics: IntrinsicsInitConfig = field(default_factory=IntrinsicsInitConfig)
    sensor: SensorMode = field(default_factory=PinholeSensorMode)
    rig: RigConfig = field(default_factory=RigConfig)
    handeye_init: HandeyeInitConfig = field(default_factory=HandeyeInitConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    handeye_ba: HandeyeBaConfig = field(default_factory=HandeyeBaConfig)

    def to_payload(self) -> dict[str, Any]:
        return {
            "intrinsics": self.intrinsics.to_payload(),
            "sensor": self.sensor.to_payload(),
            "rig": self.rig.to_payload(),
            "handeye_init": self.handeye_init.to_payload(),
            "solver": self.solver.to_payload(),
            "handeye_ba": self.handeye_ba.to_payload(),
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "RigHandeyeCalibrationConfig":
        cfg = cls()
        for key, value in mapping.items():
            if key == "intrinsics":
                cfg.intrinsics = (
                    value
                    if isinstance(value, IntrinsicsInitConfig)
                    else IntrinsicsInitConfig.from_mapping(cast(Mapping[str, Any], value))
                )
            elif key == "sensor":
                cfg.sensor = _sensor_mode_from_mapping(value)
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
            elif key == "handeye_ba":
                cfg.handeye_ba = (
                    value
                    if isinstance(value, HandeyeBaConfig)
                    else HandeyeBaConfig.from_mapping(cast(Mapping[str, Any], value))
                )
            else:
                raise ValueError(f"unknown RigHandeyeCalibrationConfig field: {key}")
        return cfg


@dataclass(slots=True)
class LaserlineDeviceOptimizeConfig:
    """Bundle-adjustment options for laserline-device calibration.

    ``fix_camera`` collapses the old ``fix_intrinsics`` /
    ``fix_distortion`` / ``fix_k3`` boolean trio into one
    [`CameraFixMask`]. It is honored only at the granularity the
    underlying laserline solver supports: ``intrinsics`` is all-or-nothing
    and ``distortion`` collapses to ``{all-fixed, k3-only, all-free}`` —
    see the Rust `LaserlineDeviceConfig::solve_opts` doc comment.
    """

    calib_loss: RobustLoss = field(default_factory=lambda: {"Huber": {"scale": 1.0}})
    laser_loss: RobustLoss = field(default_factory=lambda: {"Huber": {"scale": 0.01}})
    calib_weight: float = 1.0
    laser_weight: float = 1.0
    fix_camera: CameraFixMask = field(default_factory=CameraFixMask)
    fix_sensor: bool = True
    fix_poses: list[int] = field(default_factory=lambda: [0])
    fix_plane: bool = False
    laser_residual_type: LaserlineResidualType = "LineDistNormalized"

    def to_payload(self) -> dict[str, Any]:
        return {
            "calib_loss": cast(Any, self.calib_loss),
            "laser_loss": cast(Any, self.laser_loss),
            "calib_weight": float(self.calib_weight),
            "laser_weight": float(self.laser_weight),
            "fix_camera": self.fix_camera.to_payload(),
            "fix_sensor": bool(self.fix_sensor),
            "fix_poses": [int(i) for i in self.fix_poses],
            "fix_plane": bool(self.fix_plane),
            "laser_residual_type": self.laser_residual_type,
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "LaserlineDeviceOptimizeConfig":
        cfg = cls()
        for key, value in mapping.items():
            if key == "fix_camera":
                cfg.fix_camera = (
                    value
                    if isinstance(value, CameraFixMask)
                    else CameraFixMask.from_mapping(cast(Mapping[str, Any], value))
                )
            elif not hasattr(cfg, key):
                raise ValueError(f"unknown LaserlineDeviceOptimizeConfig field: {key}")
            else:
                setattr(cfg, key, value)
        return cfg


@dataclass(slots=True)
class LaserlineDeviceCalibrationConfig:
    """Configuration for laserline-device calibration."""

    init: IntrinsicsInitConfig = field(default_factory=IntrinsicsInitConfig)
    sensor_init: dict[str, float] = field(default_factory=lambda: dict(_DEFAULT_SENSOR_INIT))
    solver: SolverConfig = field(default_factory=SolverConfig)
    optimize: LaserlineDeviceOptimizeConfig = field(default_factory=LaserlineDeviceOptimizeConfig)

    def to_payload(self) -> dict[str, Any]:
        sensor_init = dict(_DEFAULT_SENSOR_INIT)
        sensor_init.update(self.sensor_init)
        return {
            "init": self.init.to_payload(),
            "sensor_init": sensor_init,
            "solver": self.solver.to_payload(),
            "optimize": self.optimize.to_payload(),
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "LaserlineDeviceCalibrationConfig":
        cfg = cls()
        for key, value in mapping.items():
            if key == "init":
                cfg.init = (
                    value
                    if isinstance(value, IntrinsicsInitConfig)
                    else IntrinsicsInitConfig.from_mapping(cast(Mapping[str, Any], value))
                )
            elif key == "sensor_init":
                cfg.sensor_init = dict(value)
            elif key == "solver":
                cfg.solver = (
                    value
                    if isinstance(value, SolverConfig)
                    else SolverConfig.from_mapping(cast(Mapping[str, Any], value))
                )
            elif key == "optimize":
                cfg.optimize = (
                    value
                    if isinstance(value, LaserlineDeviceOptimizeConfig)
                    else LaserlineDeviceOptimizeConfig.from_mapping(cast(Mapping[str, Any], value))
                )
            else:
                raise ValueError(f"unknown LaserlineDeviceCalibrationConfig field: {key}")
        return cfg


@dataclass(slots=True)
class ScheimpflugIntrinsicsCalibrationConfig:
    """Configuration for planar Scheimpflug intrinsics calibration.

    Grouped configuration. Two defaults deviate from the shared sub-objects'
    own defaults, mirroring the Rust config: ``init.fix_tangential`` is
    ``True`` (tilt and tangential distortion are coupled, so a free
    tangential term is ill-posed during the linear stage), and
    ``solver.max_iters`` is 120 (the tilt valley needs more headroom than a
    plain intrinsics solve). ``fix_camera.distortion`` defaults to
    `radial_only` (k3/p1/p2 fixed) rather than the shared k3-only default.
    """

    init: IntrinsicsInitConfig = field(
        default_factory=lambda: IntrinsicsInitConfig(fix_tangential=True)
    )
    solver: SolverConfig = field(default_factory=lambda: SolverConfig(max_iters=120))
    distortion_model: DistortionModel = "brown_conrady5"
    fix_camera: CameraFixMask = field(
        default_factory=lambda: CameraFixMask(
            distortion=dict(_RADIAL_ONLY_DISTORTION_FIX_MASK)
        )
    )
    fix_scheimpflug: dict[str, bool] = field(default_factory=lambda: dict(_DEFAULT_SCHEIMPFLUG_FIX_MASK))
    fix_poses: list[int] = field(default_factory=lambda: [0])

    def to_payload(self) -> dict[str, Any]:
        fix_scheimpflug = dict(_DEFAULT_SCHEIMPFLUG_FIX_MASK)
        fix_scheimpflug.update(self.fix_scheimpflug)
        return {
            "init": self.init.to_payload(),
            "solver": self.solver.to_payload(),
            "distortion_model": self.distortion_model,
            "fix_camera": self.fix_camera.to_payload(),
            "fix_scheimpflug": fix_scheimpflug,
            "fix_poses": [int(i) for i in self.fix_poses],
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "ScheimpflugIntrinsicsCalibrationConfig":
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
            elif key == "distortion_model":
                cfg.distortion_model = cast(DistortionModel, value)
            elif key == "fix_camera":
                cfg.fix_camera = (
                    value
                    if isinstance(value, CameraFixMask)
                    else CameraFixMask.from_mapping(cast(Mapping[str, Any], value))
                )
            elif key == "fix_scheimpflug":
                cfg.fix_scheimpflug = dict(cast(Mapping[str, bool], value))
            elif key == "fix_poses":
                cfg.fix_poses = [int(i) for i in cast(list[Any], value)]
            else:
                raise ValueError(f"unknown ScheimpflugIntrinsicsCalibrationConfig field: {key}")
        return cfg


@dataclass(slots=True)
class PinholeIntrinsics:
    """Typed pinhole intrinsics model."""

    fx: float
    fy: float
    cx: float
    cy: float
    skew: float

    def to_payload(self) -> dict[str, float]:
        """Convert to serde payload shape."""
        return {
            "fx": float(self.fx),
            "fy": float(self.fy),
            "cx": float(self.cx),
            "cy": float(self.cy),
            "skew": float(self.skew),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "PinholeIntrinsics":
        """Parse intrinsics from serde payload shape."""
        return cls(
            fx=float(payload["fx"]),
            fy=float(payload["fy"]),
            cx=float(payload["cx"]),
            cy=float(payload["cy"]),
            skew=float(payload["skew"]),
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


# Any distortion model a single-camera (Planar / Scheimpflug) result may carry.
# The rig / hand-eye / laserline results stay strictly ``BrownConradyDistortion``.
Distortion = (
    NoDistortion
    | BrownConradyDistortion
    | Division1Distortion
    | Rational8Distortion
    | ThinPrism9Distortion
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
}
_DISTORTION_TAG_BY_TYPE: dict[type, str] = {v: k for k, v in _DISTORTION_BY_TAG.items()}


def _distortion_from_payload(payload: Mapping[str, Any]) -> Distortion:
    """Parse a tagged ``DistortionParams`` payload into the matching dataclass.

    An absent ``type`` tag defaults to Brown-Conrady (the untyped ``dist`` shape
    emitted by the ``PinholeCamera`` serde form used by rig exports).
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
        """Parse Scheimpflug tilt, accepting legacy `tau_x`/`tau_y` aliases."""
        tilt_x = payload.get("tilt_x", payload.get("tau_x"))
        tilt_y = payload.get("tilt_y", payload.get("tau_y"))
        if tilt_x is None:
            raise ValueError("Scheimpflug sensor payload missing tilt_x/tau_x")
        if tilt_y is None:
            raise ValueError("Scheimpflug sensor payload missing tilt_y/tau_y")
        return cls(tilt_x=float(tilt_x), tilt_y=float(tilt_y))


@dataclass(slots=True)
class PinholeBrownConradyCamera:
    """Typed pinhole camera model with Brown-Conrady distortion."""

    intrinsics: PinholeIntrinsics
    distortion: BrownConradyDistortion

    def to_payload(self) -> dict[str, Any]:
        """Convert to the `PinholeCamera` serde payload shape."""
        return {
            "k": self.intrinsics.to_payload(),
            "dist": self.distortion.to_payload(),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "PinholeBrownConradyCamera":
        """Parse from either `PinholeCamera` or `CameraParams`-style payload."""
        if "k" in payload and "dist" in payload:
            return cls(
                intrinsics=PinholeIntrinsics.from_payload(cast(Mapping[str, Any], payload["k"])),
                distortion=BrownConradyDistortion.from_payload(
                    cast(Mapping[str, Any], payload["dist"])
                ),
            )

        if "intrinsics" in payload and "distortion" in payload:
            intrinsics_payload = cast(Mapping[str, Any], payload["intrinsics"])
            distortion_payload = cast(Mapping[str, Any], payload["distortion"])

            intrinsics_type = intrinsics_payload.get("type")
            if intrinsics_type not in (None, "fx_fy_cx_cy_skew"):
                raise ValueError(
                    f"unsupported intrinsics payload type for pinhole camera: {intrinsics_type!r}"
                )

            distortion_type = distortion_payload.get("type")
            if distortion_type not in (None, "brown_conrady5"):
                raise ValueError(
                    f"unsupported distortion payload type for pinhole camera: {distortion_type!r}"
                )

            return cls(
                intrinsics=PinholeIntrinsics.from_payload(intrinsics_payload),
                distortion=BrownConradyDistortion.from_payload(distortion_payload),
            )

        raise ValueError("camera payload missing expected intrinsics/distortion fields")


def _check_intrinsics_type(payload: Mapping[str, Any], where: str) -> None:
    intrinsics_type = payload.get("type")
    if intrinsics_type not in (None, "fx_fy_cx_cy_skew"):
        raise ValueError(
            f"unsupported intrinsics payload type for {where} camera: {intrinsics_type!r}"
        )


@dataclass(slots=True)
class PinholeCamera:
    """Typed pinhole camera whose distortion may be any fitted model.

    Used by the two single-camera results that expose ``distortion_model``
    (``PlanarCalibrationResult``). The rig / hand-eye / laserline results keep
    the strictly Brown-Conrady :class:`PinholeBrownConradyCamera`.
    """

    intrinsics: PinholeIntrinsics
    distortion: Distortion

    def to_payload(self) -> dict[str, Any]:
        """Convert to the `CameraParams` serde payload shape."""
        return {
            "projection": {"type": "pinhole"},
            "distortion": _distortion_to_payload(self.distortion),
            "sensor": {"type": "identity"},
            "intrinsics": {"type": "fx_fy_cx_cy_skew", **self.intrinsics.to_payload()},
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "PinholeCamera":
        """Parse from the `CameraParams` serde payload shape."""
        intrinsics_payload = cast(Mapping[str, Any], payload["intrinsics"])
        _check_intrinsics_type(intrinsics_payload, "pinhole")
        return cls(
            intrinsics=PinholeIntrinsics.from_payload(intrinsics_payload),
            distortion=_distortion_from_payload(cast(Mapping[str, Any], payload["distortion"])),
        )


@dataclass(slots=True)
class PinholeScheimpflugCamera:
    """Typed pinhole + Scheimpflug camera whose distortion may be any fitted model.

    Used by ``ScheimpflugIntrinsicsResult`` (which exposes ``distortion_model``).
    Rig Scheimpflug results instead carry a strictly Brown-Conrady
    :class:`PinholeBrownConradyCamera` alongside a separate
    :class:`ScheimpflugSensor`.
    """

    intrinsics: PinholeIntrinsics
    distortion: Distortion
    sensor: ScheimpflugSensor

    def to_payload(self) -> dict[str, Any]:
        """Convert to `CameraParams` serde payload shape."""
        return {
            "projection": {"type": "pinhole"},
            "distortion": _distortion_to_payload(self.distortion),
            "sensor": {"type": "scheimpflug", **self.sensor.to_payload()},
            "intrinsics": {"type": "fx_fy_cx_cy_skew", **self.intrinsics.to_payload()},
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "PinholeScheimpflugCamera":
        """Parse from `CameraParams` payload with Scheimpflug sensor."""
        intrinsics_payload = cast(Mapping[str, Any], payload["intrinsics"])
        sensor_payload = cast(Mapping[str, Any], payload["sensor"])
        _check_intrinsics_type(intrinsics_payload, "Scheimpflug")

        sensor_type = sensor_payload.get("type")
        if sensor_type not in (None, "scheimpflug"):
            raise ValueError(
                f"unsupported sensor payload type for Scheimpflug camera: {sensor_type!r}"
            )

        return cls(
            intrinsics=PinholeIntrinsics.from_payload(intrinsics_payload),
            distortion=_distortion_from_payload(cast(Mapping[str, Any], payload["distortion"])),
            sensor=ScheimpflugSensor.from_payload(sensor_payload),
        )


@dataclass(slots=True)
class LaserlinePlane:
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
    def from_payload(cls, payload: Mapping[str, Any]) -> "LaserlinePlane":
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
class LaserlineEstimateParams:
    """Typed parameter payload for laserline optimization results."""

    intrinsics: PinholeIntrinsics
    distortion: BrownConradyDistortion
    sensor: ScheimpflugSensor
    poses: list[Pose]
    plane: LaserlinePlane

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "LaserlineEstimateParams":
        return cls(
            intrinsics=PinholeIntrinsics.from_payload(
                cast(Mapping[str, Any], payload["intrinsics"])
            ),
            distortion=BrownConradyDistortion.from_payload(
                cast(Mapping[str, Any], payload["distortion"])
            ),
            sensor=ScheimpflugSensor.from_payload(cast(Mapping[str, Any], payload["sensor"])),
            poses=[
                Pose.from_payload(cast(Mapping[str, Any], pose))
                for pose in cast(list[Any], payload["poses"])
            ],
            plane=LaserlinePlane.from_payload(cast(Mapping[str, Any], payload["plane"])),
        )


@dataclass(slots=True)
class LaserlineEstimate:
    """Typed laserline optimizer estimate payload."""

    params: LaserlineEstimateParams
    final_cost: float

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "LaserlineEstimate":
        report = cast(Mapping[str, Any], payload["report"])
        return cls(
            params=LaserlineEstimateParams.from_payload(
                cast(Mapping[str, Any], payload["params"])
            ),
            final_cost=float(report["final_cost"]),
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
class PlanarCalibrationResult:
    """Result from :func:`vision_calibration.run_planar_intrinsics`.

    ``camera.distortion`` is one of the :data:`Distortion` variants, matching the
    ``distortion_model`` requested in the config (Brown-Conrady by default).
    """

    camera: PinholeCamera
    camera_se3_target: list[Pose]
    final_cost: float
    mean_reproj_error: float
    per_cam_reproj_errors: list[float]

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "PlanarCalibrationResult":
        params = cast(Mapping[str, Any], payload["params"])
        report = cast(Mapping[str, Any], payload["report"])
        poses = [
            Pose.from_payload(cast(Mapping[str, Any], p))
            for p in cast(list[Any], params["camera_se3_target"])
        ]
        return cls(
            camera=PinholeCamera.from_payload(cast(Mapping[str, Any], params["camera"])),
            camera_se3_target=poses,
            final_cost=float(report["final_cost"]),
            mean_reproj_error=float(payload["mean_reproj_error"]),
            per_cam_reproj_errors=[
                float(v) for v in cast(list[Any], payload["per_cam_reproj_errors"])
            ],
        )


@dataclass(slots=True)
class SingleCamHandeyeResult:
    """Result from :func:`vision_calibration.run_single_cam_handeye`."""

    camera: PinholeBrownConradyCamera
    handeye_mode: HandEyeMode
    gripper_se3_camera: Pose | None
    camera_se3_base: Pose | None
    base_se3_target: Pose | None
    gripper_se3_target: Pose | None
    robot_deltas: list[list[float]] | None
    mean_reproj_error: float
    per_cam_reproj_errors: list[float]

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "SingleCamHandeyeResult":
        def _pose(name: str) -> Pose | None:
            value = payload.get(name)
            if value is None:
                return None
            return Pose.from_payload(cast(Mapping[str, Any], value))

        return cls(
            camera=PinholeBrownConradyCamera.from_payload(
                cast(Mapping[str, Any], payload["camera"])
            ),
            handeye_mode=cast(HandEyeMode, payload["handeye_mode"]),
            gripper_se3_camera=_pose("gripper_se3_camera"),
            camera_se3_base=_pose("camera_se3_base"),
            base_se3_target=_pose("base_se3_target"),
            gripper_se3_target=_pose("gripper_se3_target"),
            robot_deltas=cast(list[list[float]] | None, payload.get("robot_deltas")),
            mean_reproj_error=float(payload["mean_reproj_error"]),
            per_cam_reproj_errors=[float(v) for v in cast(list[Any], payload["per_cam_reproj_errors"])],
        )


@dataclass(slots=True)
class RigExtrinsicsResult:
    """Result from :func:`vision_calibration.run_rig_extrinsics`."""

    cameras: list[PinholeBrownConradyCamera]
    cam_se3_rig: list[Pose]
    mean_reproj_error: float
    per_cam_reproj_errors: list[float]

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "RigExtrinsicsResult":
        return cls(
            cameras=[
                PinholeBrownConradyCamera.from_payload(cast(Mapping[str, Any], c))
                for c in cast(list[Any], payload["cameras"])
            ],
            cam_se3_rig=[Pose.from_payload(cast(Mapping[str, Any], p)) for p in cast(list[Any], payload["cam_se3_rig"])],
            mean_reproj_error=float(payload["mean_reproj_error"]),
            per_cam_reproj_errors=[float(v) for v in cast(list[Any], payload["per_cam_reproj_errors"])],
        )


@dataclass(slots=True)
class RigHandeyeResult:
    """Result from :func:`vision_calibration.run_rig_handeye`.

    `sensors` is `None` for pinhole rigs and `Some(_)` for Scheimpflug rigs
    (one entry per camera); matches the Rust ``RigHandeyeExport.sensors``
    field (A6.3 unified hand-eye family).
    """

    cameras: list[PinholeBrownConradyCamera]
    cam_se3_rig: list[Pose]
    handeye_mode: HandEyeMode
    gripper_se3_rig: Pose | None
    rig_se3_base: Pose | None
    base_se3_target: Pose | None
    gripper_se3_target: Pose | None
    robot_deltas: list[list[float]] | None
    mean_reproj_error: float
    per_cam_reproj_errors: list[float]
    sensors: list[ScheimpflugSensor] | None = None

    def to_payload(self) -> dict[str, Any]:
        """Convert to Rust/serde shape (``RigHandeyeExport``).

        Used by helpers like :func:`pixel_to_gripper_point` that pass a
        result back across the FFI boundary.
        """

        def _pose(p: Pose | None) -> Any:
            return None if p is None else p.to_payload()

        return {
            "cameras": [c.to_payload() for c in self.cameras],
            "sensors": (
                None
                if self.sensors is None
                else [s.to_payload() for s in self.sensors]
            ),
            "cam_se3_rig": [p.to_payload() for p in self.cam_se3_rig],
            "handeye_mode": self.handeye_mode,
            "gripper_se3_rig": _pose(self.gripper_se3_rig),
            "rig_se3_base": _pose(self.rig_se3_base),
            "base_se3_target": _pose(self.base_se3_target),
            "gripper_se3_target": _pose(self.gripper_se3_target),
            "robot_deltas": self.robot_deltas,
            "mean_reproj_error": float(self.mean_reproj_error),
            "per_cam_reproj_errors": [float(v) for v in self.per_cam_reproj_errors],
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "RigHandeyeResult":
        def _pose(name: str) -> Pose | None:
            value = payload.get(name)
            if value is None:
                return None
            return Pose.from_payload(cast(Mapping[str, Any], value))

        sensors_raw = payload.get("sensors")
        sensors: list[ScheimpflugSensor] | None = (
            None
            if sensors_raw is None
            else [
                ScheimpflugSensor.from_payload(cast(Mapping[str, Any], s))
                for s in cast(list[Any], sensors_raw)
            ]
        )

        return cls(
            cameras=[
                PinholeBrownConradyCamera.from_payload(cast(Mapping[str, Any], c))
                for c in cast(list[Any], payload["cameras"])
            ],
            cam_se3_rig=[
                Pose.from_payload(cast(Mapping[str, Any], p))
                for p in cast(list[Any], payload["cam_se3_rig"])
            ],
            handeye_mode=cast(HandEyeMode, payload["handeye_mode"]),
            gripper_se3_rig=_pose("gripper_se3_rig"),
            rig_se3_base=_pose("rig_se3_base"),
            base_se3_target=_pose("base_se3_target"),
            gripper_se3_target=_pose("gripper_se3_target"),
            robot_deltas=cast(list[list[float]] | None, payload.get("robot_deltas")),
            mean_reproj_error=float(payload["mean_reproj_error"]),
            per_cam_reproj_errors=[
                float(v) for v in cast(list[Any], payload["per_cam_reproj_errors"])
            ],
            sensors=sensors,
        )


@dataclass(slots=True)
class LaserlineDeviceResult:
    """Result from :func:`vision_calibration.run_laserline_device`."""

    estimate: LaserlineEstimate
    stats: LaserlineStats
    mean_reproj_error: float
    per_cam_reproj_errors: list[float]

    @property
    def mean_laser_error(self) -> float:
        """Mean laser residual (units depend on residual type)."""
        return self.stats.mean_laser_error

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "LaserlineDeviceResult":
        return cls(
            estimate=LaserlineEstimate.from_payload(
                cast(Mapping[str, Any], payload["estimate"])
            ),
            stats=LaserlineStats.from_payload(cast(Mapping[str, Any], payload["stats"])),
            mean_reproj_error=float(payload["mean_reproj_error"]),
            per_cam_reproj_errors=[float(v) for v in cast(list[Any], payload["per_cam_reproj_errors"])],
        )


@dataclass(slots=True)
class ScheimpflugIntrinsicsResult:
    """Result from :func:`vision_calibration.run_scheimpflug_intrinsics`.

    ``camera.distortion`` is one of the :data:`Distortion` variants, matching the
    ``distortion_model`` requested in the config (Brown-Conrady by default).
    """

    camera: PinholeScheimpflugCamera
    camera_se3_target: list[Pose]
    final_cost: float
    mean_reproj_error: float
    per_cam_reproj_errors: list[float]

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ScheimpflugIntrinsicsResult":
        params = cast(Mapping[str, Any], payload["params"])
        report = cast(Mapping[str, Any], payload["report"])
        return cls(
            camera=PinholeScheimpflugCamera.from_payload(
                cast(Mapping[str, Any], params["camera"])
            ),
            camera_se3_target=[
                Pose.from_payload(cast(Mapping[str, Any], pose))
                for pose in cast(list[Any], params["camera_se3_target"])
            ],
            final_cost=float(report["final_cost"]),
            mean_reproj_error=float(payload["mean_reproj_error"]),
            per_cam_reproj_errors=[float(v) for v in cast(list[Any], payload["per_cam_reproj_errors"])],
        )


# ─── Rig laserline device ─────────────────────────────────────────────────────


@dataclass(slots=True)
class RigLaserlineUpstreamCalibration:
    """Frozen upstream calibration for rig laserline device.

    Typically derived from a :class:`RigHandeyeResult` export with
    ``sensors`` populated (Scheimpflug rig).
    """

    intrinsics: list[dict[str, float]]
    distortion: list[dict[str, Any]]
    sensors: list[dict[str, float]]
    cam_se3_rig: list[Pose]
    rig_se3_target: list[Pose]

    def to_payload(self) -> dict[str, Any]:
        return {
            "intrinsics": self.intrinsics,
            "distortion": self.distortion,
            "sensors": self.sensors,
            "cam_se3_rig": [p.to_payload() for p in self.cam_se3_rig],
            "rig_se3_target": [p.to_payload() for p in self.rig_se3_target],
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "RigLaserlineUpstreamCalibration":
        return cls(
            intrinsics=cast(list[dict[str, float]], payload["intrinsics"]),
            distortion=cast(list[dict[str, Any]], payload["distortion"]),
            sensors=cast(list[dict[str, float]], payload["sensors"]),
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
    [`LaserlinePlane`] dataclass returned by
    [`RigLaserlineDeviceResult.laser_planes_cam`], so the canonical
    restart pattern ``initial_planes_cam=previous.laser_planes_cam``
    works directly without manual conversion.
    """

    dataset: RigLaserlineDataset
    upstream: RigLaserlineUpstreamCalibration
    initial_planes_cam: list[LaserlinePlane] | None = None

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
class RigLaserlineDeviceCalibrationConfig:
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
    ) -> "RigLaserlineDeviceCalibrationConfig":
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
                raise ValueError(f"unknown RigLaserlineDeviceCalibrationConfig field: {key}")
        return cfg


@dataclass(slots=True)
class RigLaserlineDeviceResult:
    """Result from :func:`vision_calibration.run_rig_laserline_device`."""

    laser_planes_rig: list[LaserlinePlane]
    laser_planes_cam: list[LaserlinePlane]
    per_camera_stats: list[LaserlineStats]

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "RigLaserlineDeviceResult":
        return cls(
            laser_planes_rig=[
                LaserlinePlane.from_payload(cast(Mapping[str, Any], p))
                for p in cast(list[Any], payload["laser_planes_rig"])
            ],
            laser_planes_cam=[
                LaserlinePlane.from_payload(cast(Mapping[str, Any], p))
                for p in cast(list[Any], payload["laser_planes_cam"])
            ],
            per_camera_stats=[
                LaserlineStats.from_payload(cast(Mapping[str, Any], s))
                for s in cast(list[Any], payload["per_camera_stats"])
            ],
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
class RigHandeyeLaserlineDataset:
    """Joint rig hand-eye laserline dataset (``RigHandeyeLaserlineInput``)."""

    num_cameras: int
    views: list[RigHandeyeLaserlineView]

    def to_payload(self) -> dict[str, Any]:
        return {
            "num_cameras": int(self.num_cameras),
            "views": [view.to_payload() for view in self.views],
        }


@dataclass(slots=True)
class RigHandeyeLaserlineBaConfig:
    """Final joint bundle-adjustment stage settings.

    ``default_camera_fix.distortion`` defaults to `radial_only` (k1, k2 free;
    k3, p1, p2 fixed) and ``fix_scheimpflug`` freezes tilt during the joint
    stage, mirroring the Rust defaults. Laser and target residuals carry
    independent robust losses (``calib_loss`` / ``laser_loss``); the shared
    ``solver.robust_loss`` is not consulted by this stage.
    """

    solver: SolverConfig = field(default_factory=lambda: SolverConfig(max_iters=30))
    laser_residual_type: LaserlineResidualType = "PointToPlane"
    calib_loss: RobustLoss = "None"
    laser_loss: RobustLoss = "None"
    calib_weight: float = 1.0
    laser_weight: float = 1.0e4
    default_camera_fix: CameraFixMask = field(
        default_factory=lambda: CameraFixMask(
            distortion=dict(_RADIAL_ONLY_DISTORTION_FIX_MASK)
        )
    )
    fix_scheimpflug: dict[str, bool] = field(
        default_factory=lambda: dict(_FIXED_SCHEIMPFLUG_MASK)
    )
    fix_handeye: bool = False
    fix_target_ref: bool = False
    robot_poses: RobotPoseConfig = field(default_factory=RobotPoseConfig)

    def to_payload(self) -> dict[str, Any]:
        fix_scheimpflug = dict(_FIXED_SCHEIMPFLUG_MASK)
        fix_scheimpflug.update(self.fix_scheimpflug)
        return {
            "solver": self.solver.to_payload(),
            "laser_residual_type": self.laser_residual_type,
            "calib_loss": cast(Any, self.calib_loss),
            "laser_loss": cast(Any, self.laser_loss),
            "calib_weight": float(self.calib_weight),
            "laser_weight": float(self.laser_weight),
            "default_camera_fix": self.default_camera_fix.to_payload(),
            "fix_scheimpflug": fix_scheimpflug,
            "fix_handeye": bool(self.fix_handeye),
            "fix_target_ref": bool(self.fix_target_ref),
            "robot_poses": self.robot_poses.to_payload(),
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "RigHandeyeLaserlineBaConfig":
        cfg = cls()
        for key, value in mapping.items():
            if key == "solver":
                cfg.solver = (
                    value
                    if isinstance(value, SolverConfig)
                    else SolverConfig.from_mapping(cast(Mapping[str, Any], value))
                )
            elif key == "default_camera_fix":
                cfg.default_camera_fix = (
                    value
                    if isinstance(value, CameraFixMask)
                    else CameraFixMask.from_mapping(cast(Mapping[str, Any], value))
                )
            elif key == "robot_poses":
                cfg.robot_poses = (
                    value
                    if isinstance(value, RobotPoseConfig)
                    else RobotPoseConfig.from_mapping(cast(Mapping[str, Any], value))
                )
            elif key == "fix_scheimpflug":
                cfg.fix_scheimpflug = dict(cast(Mapping[str, bool], value))
            elif not hasattr(cfg, key):
                raise ValueError(f"unknown RigHandeyeLaserlineBaConfig field: {key}")
            else:
                setattr(cfg, key, value)
        return cfg


@dataclass(slots=True)
class RigHandeyeLaserlineCalibrationConfig:
    """Configuration for joint rig hand-eye laserline calibration.

    Three warm-started stages: the rig hand-eye stage (``handeye``), the
    frozen-geometry laser plane init (``laserline_init``, defaults to a
    point-to-plane residual), and the final joint bundle adjustment
    (``joint_ba``).
    """

    handeye: RigHandeyeCalibrationConfig = field(
        default_factory=RigHandeyeCalibrationConfig
    )
    laserline_init: RigLaserlineDeviceCalibrationConfig = field(
        default_factory=lambda: RigLaserlineDeviceCalibrationConfig(
            laser_residual_type="PointToPlane"
        )
    )
    joint_ba: RigHandeyeLaserlineBaConfig = field(
        default_factory=RigHandeyeLaserlineBaConfig
    )

    def to_payload(self) -> dict[str, Any]:
        return {
            "handeye": self.handeye.to_payload(),
            "laserline_init": self.laserline_init.to_payload(),
            "joint_ba": self.joint_ba.to_payload(),
        }

    @classmethod
    def from_mapping(
        cls, mapping: Mapping[str, Any]
    ) -> "RigHandeyeLaserlineCalibrationConfig":
        cfg = cls()
        for key, value in mapping.items():
            if key == "handeye":
                cfg.handeye = (
                    value
                    if isinstance(value, RigHandeyeCalibrationConfig)
                    else RigHandeyeCalibrationConfig.from_mapping(
                        cast(Mapping[str, Any], value)
                    )
                )
            elif key == "laserline_init":
                cfg.laserline_init = (
                    value
                    if isinstance(value, RigLaserlineDeviceCalibrationConfig)
                    else RigLaserlineDeviceCalibrationConfig.from_mapping(
                        cast(Mapping[str, Any], value)
                    )
                )
            elif key == "joint_ba":
                cfg.joint_ba = (
                    value
                    if isinstance(value, RigHandeyeLaserlineBaConfig)
                    else RigHandeyeLaserlineBaConfig.from_mapping(
                        cast(Mapping[str, Any], value)
                    )
                )
            else:
                raise ValueError(
                    f"unknown RigHandeyeLaserlineCalibrationConfig field: {key}"
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
class RigHandeyeLaserlineResult:
    """Result from :func:`vision_calibration.run_rig_handeye_laserline`."""

    laser_planes_rig: list[LaserlinePlane]
    laser_planes_cam: list[LaserlinePlane]
    per_camera_stats: list[RigHandeyeLaserlinePerCamStats]
    cameras: list[PinholeBrownConradyCamera]
    sensors: list[ScheimpflugSensor]
    cam_se3_rig: list[Pose]
    rig_se3_target: list[Pose]
    handeye_mode: HandEyeMode
    gripper_se3_rig: Pose | None
    rig_se3_base: Pose | None
    base_se3_target: Pose | None
    gripper_se3_target: Pose | None
    robot_deltas: list[list[float]] | None
    mean_reproj_error: float
    per_cam_reproj_errors: list[float]

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "RigHandeyeLaserlineResult":
        def _pose(name: str) -> Pose | None:
            value = payload.get(name)
            if value is None:
                return None
            return Pose.from_payload(cast(Mapping[str, Any], value))

        return cls(
            laser_planes_rig=[
                LaserlinePlane.from_payload(cast(Mapping[str, Any], p))
                for p in cast(list[Any], payload["laser_planes_rig"])
            ],
            laser_planes_cam=[
                LaserlinePlane.from_payload(cast(Mapping[str, Any], p))
                for p in cast(list[Any], payload["laser_planes_cam"])
            ],
            per_camera_stats=[
                RigHandeyeLaserlinePerCamStats.from_payload(cast(Mapping[str, Any], s))
                for s in cast(list[Any], payload["per_camera_stats"])
            ],
            cameras=[
                PinholeBrownConradyCamera.from_payload(cast(Mapping[str, Any], c))
                for c in cast(list[Any], payload["cameras"])
            ],
            sensors=[
                ScheimpflugSensor.from_payload(cast(Mapping[str, Any], s))
                for s in cast(list[Any], payload["sensors"])
            ],
            cam_se3_rig=[
                Pose.from_payload(cast(Mapping[str, Any], p))
                for p in cast(list[Any], payload["cam_se3_rig"])
            ],
            rig_se3_target=[
                Pose.from_payload(cast(Mapping[str, Any], p))
                for p in cast(list[Any], payload["rig_se3_target"])
            ],
            handeye_mode=cast(HandEyeMode, payload["handeye_mode"]),
            gripper_se3_rig=_pose("gripper_se3_rig"),
            rig_se3_base=_pose("rig_se3_base"),
            base_se3_target=_pose("base_se3_target"),
            gripper_se3_target=_pose("gripper_se3_target"),
            robot_deltas=cast(list[list[float]] | None, payload.get("robot_deltas")),
            mean_reproj_error=float(payload["mean_reproj_error"]),
            per_cam_reproj_errors=[
                float(v) for v in cast(list[Any], payload["per_cam_reproj_errors"])
            ],
        )


def normalize_input_payload(input_value: Any) -> Any:
    """Convert high-level model inputs to serde payloads."""
    return _payload_from_maybe_model(input_value)
