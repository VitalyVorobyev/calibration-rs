"""Calibrate the regression suite's entries with OpenCV.

Reads each entry's frozen observations (``regression/observations/<id>.json``)
and its config from ``regression/suite.toml``, runs the OpenCV calibration that
matches the problem, and writes ``regression/results/<id>/opencv.json``:

    {
      "producer": "opencv 4.12.0",
      "method": "calibrateCameraExtended (CALIB_FIX_K3)",
      "parameters": {"fx": ..., "k1": ...},      # by calibration-rs label
      "poses": {"gripper_se3_camera": {...}},    # by calibration-rs export name
      "sigma": {"fx": ...},                       # OpenCV's standard deviations
      "mean_reproj_error": ...                    # mean Euclidean distance, px
    }

Labels and pose names follow the calibration-rs result the entry is compared
with, so ``calib-bench suite compare`` reads both side by side. Poses are
``a_se3_b`` (``b`` coordinates to ``a``) as ``{"rotation": [qx, qy, qz, qw],
"translation": [x, y, z]}``.

Run from the repository root:

    uv run --project tools/compare/opencv tools/compare/opencv/compare.py [--only ID ...]
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SUITE = ROOT / "regression"
CRITERIA = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 200, 1e-12)
# calibration-rs fixes k3 by default; OpenCV's distCoeffs are (k1, k2, p1, p2, k3).
DIST_LABELS = ("k1", "k2", "p1", "p2", "k3")


# ── poses ────────────────────────────────────────────────────────────────────


def to_matrix(pose: dict[str, Any]) -> np.ndarray:
    """A 4×4 transform from an exported pose."""
    qx, qy, qz, qw = pose["rotation"]
    rot = np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ]
    )
    t = np.eye(4)
    t[:3, :3] = rot
    t[:3, 3] = pose["translation"]
    return t


def from_rt(rot: np.ndarray, trans: np.ndarray) -> np.ndarray:
    """A 4×4 transform from a rotation (matrix or Rodrigues vector) and a translation."""
    rot = np.asarray(rot, np.float64)
    if rot.size == 3:
        rot, _ = cv2.Rodrigues(rot)
    t = np.eye(4)
    t[:3, :3] = rot
    t[:3, 3] = np.asarray(trans, np.float64).ravel()
    return t


def to_pose(t: np.ndarray) -> dict[str, list[float]]:
    """An exported pose from a 4×4 transform."""
    r = t[:3, :3]
    w = np.sqrt(max(0.0, 1.0 + r[0, 0] + r[1, 1] + r[2, 2])) / 2.0
    x = np.sqrt(max(0.0, 1.0 + r[0, 0] - r[1, 1] - r[2, 2])) / 2.0
    y = np.sqrt(max(0.0, 1.0 - r[0, 0] + r[1, 1] - r[2, 2])) / 2.0
    z = np.sqrt(max(0.0, 1.0 - r[0, 0] - r[1, 1] + r[2, 2])) / 2.0
    x = np.copysign(x, r[2, 1] - r[1, 2])
    y = np.copysign(y, r[0, 2] - r[2, 0])
    z = np.copysign(z, r[1, 0] - r[0, 1])
    q = np.array([x, y, z, w]) / np.linalg.norm([x, y, z, w])
    return {"rotation": q.tolist(), "translation": t[:3, 3].tolist()}


# ── single cameras ───────────────────────────────────────────────────────────


class Camera:
    """One camera calibrated with ``calibrateCameraExtended``."""

    def __init__(
        self,
        object_points: list[np.ndarray],
        image_points: list[np.ndarray],
        size: tuple[int, int],
        flags: int,
    ) -> None:
        (
            _rms,
            self.k,
            self.dist,
            rvecs,
            tvecs,
            sd_intrinsics,
            _sd_extrinsics,
            _per_view,
        ) = cv2.calibrateCameraExtended(
            object_points, image_points, size, None, None, flags=flags, criteria=CRITERIA
        )
        self.object_points = object_points
        self.image_points = image_points
        self.sd = sd_intrinsics.ravel()
        self.flags = flags
        # cam_se3_target per view.
        self.poses = [from_rt(r, t) for r, t in zip(rvecs, tvecs)]

    def parameters(self) -> dict[str, float]:
        k, d = self.k, self.dist.ravel()
        out = {"fx": k[0, 0], "fy": k[1, 1], "cx": k[0, 2], "cy": k[1, 2]}
        out.update(zip(DIST_LABELS, d[:5]))
        if self.flags & cv2.CALIB_TILTED_MODEL:
            out["tilt_x"], out["tilt_y"] = d[12], d[13]
        return {key: float(v) for key, v in out.items()}

    def sigma(self) -> dict[str, float]:
        """OpenCV's standard deviations of the estimated parameters (fixed
        ones, reported as 0, are left out)."""
        # stdDeviationsIntrinsics: fx, fy, cx, cy, k1, k2, p1, p2, k3, k4–k6,
        # s1–s4, τx, τy.
        names = ("fx", "fy", "cx", "cy", *DIST_LABELS)
        out = dict(zip(names, self.sd[:9]))
        if self.flags & cv2.CALIB_TILTED_MODEL:
            out["tilt_x"], out["tilt_y"] = self.sd[16], self.sd[17]
        return {key: float(v) for key, v in out.items() if v > 0}

    def reprojection_errors(self, poses: list[np.ndarray] | None = None) -> np.ndarray:
        """Euclidean reprojection error of every point, px."""
        errors = []
        for obj, img, pose in zip(self.object_points, self.image_points, poses or self.poses):
            rvec, _ = cv2.Rodrigues(pose[:3, :3])
            proj, _ = cv2.projectPoints(obj, rvec, pose[:3, 3], self.k, self.dist)
            errors.append(np.linalg.norm(proj.reshape(-1, 2) - img, axis=1))
        return np.concatenate(errors)


def tilted(entry: dict[str, Any]) -> bool:
    """Whether the entry's config names a Scheimpflug (tilted) sensor."""
    model = entry.get("config", {}).get("camera", {}).get("model", {})
    return model.get("sensor") == "scheimpflug"


def camera_flags(entry: dict[str, Any]) -> int:
    flags = cv2.CALIB_FIX_K3
    if tilted(entry):
        # The tilted-sensor model with calibration-rs's Scheimpflug parameter
        # set: the tangential terms held at zero, as they alias with the tilt,
        # and the extra rational and thin-prism terms too.
        flags |= (
            cv2.CALIB_TILTED_MODEL
            | cv2.CALIB_ZERO_TANGENT_DIST
            | cv2.CALIB_FIX_K4
            | cv2.CALIB_FIX_K5
            | cv2.CALIB_FIX_K6
            | cv2.CALIB_FIX_S1_S2_S3_S4
        )
    return flags


def equidistant(entry: dict[str, Any]) -> bool:
    """Whether the entry's config names an equidistant (fisheye) projection."""
    model = entry.get("config", {}).get("camera", {}).get("model", {})
    return model.get("projection") == "equidistant"


def observations(views: list[dict[str, Any]]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    return (
        [np.array(v["points_3d"], np.float32) for v in views],
        [np.array(v["points_2d"], np.float32) for v in views],
    )


# ── fisheye cameras ──────────────────────────────────────────────────────────

# The flags of OpenCV's own fisheye calibration tests. `cv::fisheye` has no
# tangential terms and fixes nothing else; its distortion is (k1, k2, k3, k4)
# of the Kannala-Brandt model, calibration-rs's `KannalaBrandt4`.
FISHEYE_FLAGS = (
    cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
    | cv2.fisheye.CALIB_CHECK_COND
    | cv2.fisheye.CALIB_FIX_SKEW
)
FISHEYE_FLAG_NAMES = "CALIB_RECOMPUTE_EXTRINSIC, CALIB_CHECK_COND, CALIB_FIX_SKEW"
FISHEYE_LABELS = ("k1", "k2", "k3", "k4")


def fisheye_observations(
    views: list[dict[str, Any]],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Points in the shapes `cv::fisheye` takes: (1, N, 3) and (1, N, 2), float64."""
    return (
        [np.array(v["points_3d"], np.float64).reshape(1, -1, 3) for v in views],
        [np.array(v["points_2d"], np.float64).reshape(1, -1, 2) for v in views],
    )


def fisheye_parameters(k: np.ndarray, d: np.ndarray) -> dict[str, float]:
    out = {"fx": k[0, 0], "fy": k[1, 1], "cx": k[0, 2], "cy": k[1, 2]}
    out.update(zip(FISHEYE_LABELS, np.asarray(d).ravel()))
    return {key: float(v) for key, v in out.items()}


def fisheye_planar(obs: dict[str, Any]) -> dict[str, Any]:
    size = tuple(obs["cameras"][0]["image_size"])
    obj, img = fisheye_observations([v["obs"] for v in obs["input"]["views"]])
    _rms, k, d, rvecs, tvecs = cv2.fisheye.calibrate(
        obj, img, size, None, None, flags=FISHEYE_FLAGS, criteria=CRITERIA
    )
    errors = [
        np.linalg.norm(
            cv2.fisheye.projectPoints(o, r, t, k, d)[0].reshape(-1, 2) - i.reshape(-1, 2), axis=1
        )
        for o, i, r, t in zip(obj, img, rvecs, tvecs)
    ]
    return {
        "method": f"fisheye.calibrate ({FISHEYE_FLAG_NAMES})",
        "parameters": fisheye_parameters(k, d),
        "poses": {},
        "sigma": {},
        "mean_reproj_error": float(np.concatenate(errors).mean()),
    }


def fisheye_rig(obs: dict[str, Any]) -> dict[str, Any]:
    """`fisheye.stereoCalibrate` of a two-camera rig from scratch, intrinsics
    refined jointly with the extrinsics, as OpenCV's own test runs it."""
    views = obs["input"]["views"]
    if len(obs["cameras"]) != 2 or any(None in v["obs"]["cameras"] for v in views):
        raise SystemExit("fisheye.stereoCalibrate needs two cameras that see every view")
    obj, img0 = fisheye_observations([v["obs"]["cameras"][0] for v in views])
    _, img1 = fisheye_observations([v["obs"]["cameras"][1] for v in views])
    _rms, k0, d0, k1, d1, rot, trans, *_ = cv2.fisheye.stereoCalibrate(
        obj,
        img0,
        img1,
        None,
        None,
        None,
        None,
        tuple(obs["cameras"][0]["image_size"]),
        flags=FISHEYE_FLAGS,
        criteria=CRITERIA,
    )
    parameters = {
        **prefixed(0, fisheye_parameters(k0, d0)),
        **prefixed(1, fisheye_parameters(k1, d1)),
    }
    return {
        "method": f"fisheye.stereoCalibrate ({FISHEYE_FLAG_NAMES})",
        "parameters": parameters,
        # R, T map camera-0 coordinates to camera 1: cam1_se3_cam0.
        "poses": {
            "cam0.cam_se3_rig": to_pose(np.eye(4)),
            "cam1.cam_se3_rig": to_pose(from_rt(rot, trans)),
        },
        "sigma": {},
        "mean_reproj_error": None,
    }


# ── problems ─────────────────────────────────────────────────────────────────


def planar(entry: dict[str, Any], obs: dict[str, Any]) -> dict[str, Any]:
    if equidistant(entry):
        return fisheye_planar(obs)
    size = tuple(obs["cameras"][0]["image_size"])
    views = obs["input"]["views"]
    cam = Camera(*observations([v["obs"] for v in views]), size, camera_flags(entry))
    method = "calibrateCameraExtended (CALIB_FIX_K3)"
    if cam.flags & cv2.CALIB_TILTED_MODEL:
        method = "calibrateCameraExtended (CALIB_TILTED_MODEL, CALIB_ZERO_TANGENT_DIST, CALIB_FIX_K3)"
    return {
        "method": method,
        "parameters": cam.parameters(),
        "poses": {},
        "sigma": cam.sigma(),
        "mean_reproj_error": float(cam.reprojection_errors().mean()),
    }


def robot_world_handeye(
    cam: Camera, base_se3_gripper: list[np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """``calibrateRobotWorldHandEye`` (Shah) on one camera's views:
    returns ``(gripper_se3_camera, base_se3_target)``."""
    gripper_se3_base = [np.linalg.inv(t) for t in base_se3_gripper]
    r_b2w, t_b2w, r_g2c, t_g2c = cv2.calibrateRobotWorldHandEye(
        [p[:3, :3] for p in cam.poses],
        [p[:3, 3] for p in cam.poses],
        [t[:3, :3] for t in gripper_se3_base],
        [t[:3, 3] for t in gripper_se3_base],
    )
    # base2world = target_se3_base, gripper2cam = camera_se3_gripper.
    return np.linalg.inv(from_rt(r_g2c, t_g2c)), np.linalg.inv(from_rt(r_b2w, t_b2w))


def single_cam_handeye(entry: dict[str, Any], obs: dict[str, Any]) -> dict[str, Any]:
    size = tuple(obs["cameras"][0]["image_size"])
    views = obs["input"]["views"]
    cam = Camera(*observations([v["obs"] for v in views]), size, cv2.CALIB_FIX_K3)
    robot = [to_matrix(v["meta"]["base_se3_gripper"]) for v in views]
    gripper_se3_camera, base_se3_target = robot_world_handeye(cam, robot)
    # The hand-eye chain's own reprojection: target → base → gripper → camera.
    chain = [
        np.linalg.inv(gripper_se3_camera) @ np.linalg.inv(t_bg) @ base_se3_target
        for t_bg in robot
    ]
    return {
        "method": "calibrateCameraExtended (CALIB_FIX_K3), calibrateRobotWorldHandEye (Shah)",
        "parameters": cam.parameters(),
        "poses": {
            "gripper_se3_camera": to_pose(gripper_se3_camera),
            "base_se3_target": to_pose(base_se3_target),
        },
        "sigma": cam.sigma(),
        "mean_reproj_error": float(cam.reprojection_errors(chain).mean()),
    }


def rig_cameras(obs: dict[str, Any]) -> list[tuple[Camera, list[int]]]:
    """Each camera calibrated alone on the views it saw, with those view indices."""
    out = []
    views = obs["input"]["views"]
    for i, camera in enumerate(obs["cameras"]):
        seen = [v for v, view in enumerate(views) if view["obs"]["cameras"][i] is not None]
        cam = Camera(
            *observations([views[v]["obs"]["cameras"][i] for v in seen]),
            tuple(camera["image_size"]),
            cv2.CALIB_FIX_K3,
        )
        out.append((cam, seen))
    return out


def prefixed(i: int, values: dict[str, float]) -> dict[str, float]:
    return {f"cam{i}.{k}": v for k, v in values.items()}


def rig_extrinsics(entry: dict[str, Any], obs: dict[str, Any]) -> dict[str, Any]:
    if equidistant(entry):
        return fisheye_rig(obs)
    views = obs["input"]["views"]
    cams = rig_cameras(obs)
    # The rig refines intrinsics jointly unless the entry's config says not to.
    refine = entry.get("config", {}).get("rig", {}).get("refine_intrinsics", True) is not False
    parameters: dict[str, float] = {}
    sigma: dict[str, float] = {}
    poses = {"cam0.cam_se3_rig": to_pose(np.eye(4))}
    errors = []
    k0, d0 = cams[0][0].k, cams[0][0].dist
    for i in range(1, len(cams)):
        both = [
            v
            for v, view in enumerate(views)
            if view["obs"]["cameras"][0] is not None and view["obs"]["cameras"][i] is not None
        ]
        obj = [np.array(views[v]["obs"]["cameras"][0]["points_3d"], np.float32) for v in both]
        img0 = [np.array(views[v]["obs"]["cameras"][0]["points_2d"], np.float32) for v in both]
        imgi = [np.array(views[v]["obs"]["cameras"][i]["points_2d"], np.float32) for v in both]
        flags = (
            cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_K3
            if refine
            else cv2.CALIB_FIX_INTRINSIC
        )
        rms, k0, d0, ki, di, rot, trans, _, _ = cv2.stereoCalibrate(
            obj,
            img0,
            imgi,
            k0.copy(),
            d0.copy(),
            cams[i][0].k.copy(),
            cams[i][0].dist.copy(),
            tuple(obs["cameras"][0]["image_size"]),
            flags=flags,
            criteria=CRITERIA,
        )
        cams[i][0].k, cams[i][0].dist = ki, di
        # R, T map camera-0 coordinates to camera i: cam_i_se3_cam0.
        poses[f"cam{i}.cam_se3_rig"] = to_pose(from_rt(rot, trans))
        errors.append(rms)
    cams[0][0].k, cams[0][0].dist = k0, d0
    for i, (cam, _) in enumerate(cams):
        parameters.update(prefixed(i, cam.parameters()))
        if not refine:
            sigma.update(prefixed(i, cam.sigma()))
    method = "calibrateCameraExtended (CALIB_FIX_K3) per camera, stereoCalibrate " + (
        "(CALIB_USE_INTRINSIC_GUESS, CALIB_FIX_K3)" if refine else "(CALIB_FIX_INTRINSIC)"
    )
    return {
        "method": method,
        "parameters": parameters,
        "poses": poses,
        "sigma": sigma,
        "mean_reproj_error": None,
    }


def rig_handeye(entry: dict[str, Any], obs: dict[str, Any]) -> dict[str, Any]:
    """Robot-world hand-eye per camera; the rig frame is camera 0's."""
    views = obs["input"]["views"]
    cams = rig_cameras(obs)
    parameters: dict[str, float] = {}
    sigma: dict[str, float] = {}
    poses: dict[str, Any] = {}
    gripper_se3_cam = []
    for i, (cam, seen) in enumerate(cams):
        robot = [to_matrix(views[v]["meta"]["base_se3_gripper"]) for v in seen]
        g_se3_c, base_se3_target = robot_world_handeye(cam, robot)
        gripper_se3_cam.append(g_se3_c)
        parameters.update(prefixed(i, cam.parameters()))
        sigma.update(prefixed(i, cam.sigma()))
        if i == 0:
            poses["gripper_se3_rig"] = to_pose(g_se3_c)
            poses["base_se3_target"] = to_pose(base_se3_target)
    for i, g_se3_c in enumerate(gripper_se3_cam):
        poses[f"cam{i}.cam_se3_rig"] = to_pose(np.linalg.inv(g_se3_c) @ gripper_se3_cam[0])
    return {
        "method": "calibrateCameraExtended (CALIB_FIX_K3) and calibrateRobotWorldHandEye "
        "(Shah) per camera; rig frame = camera 0",
        "parameters": parameters,
        "poses": poses,
        "sigma": sigma,
        "mean_reproj_error": None,
    }


PROBLEMS = {
    "planar_intrinsics": planar,
    "single_cam_handeye": single_cam_handeye,
    "rig_extrinsics": rig_extrinsics,
    "rig_handeye": rig_handeye,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--only", nargs="*", default=[], help="entry ids")
    args = parser.parse_args()
    suite = tomllib.loads((SUITE / "suite.toml").read_text())
    for entry in suite["entry"]:
        if args.only and entry["id"] not in args.only:
            continue
        obs = json.loads((SUITE / "observations" / f"{entry['id']}.json").read_text())
        run = PROBLEMS.get(obs["problem"])
        if run is None:
            print(f"skip  {entry['id']}: OpenCV has no {obs['problem']} calibration")
            continue
        result = {"producer": f"opencv {cv2.__version__}", **run(entry, obs)}
        out = SUITE / "results" / entry["id"] / "opencv.json"
        out.write_text(json.dumps(result, indent=2) + "\n")
        print(f"wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
