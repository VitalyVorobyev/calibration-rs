"""Shared helpers for vision-calibration Python examples.

These examples intentionally use only Python stdlib so they run in a minimal
virtual environment where `vision_calibration` is installed.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def norm3(v: Iterable[float]) -> float:
    x, y, z = v
    return math.sqrt(x * x + y * y + z * z)


def add3(a: list[float], b: list[float]) -> list[float]:
    return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]


def sub3(a: list[float], b: list[float]) -> list[float]:
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]


def scale3(v: list[float], s: float) -> list[float]:
    return [v[0] * s, v[1] * s, v[2] * s]


def dot3(a: list[float], b: list[float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def normalize3(v: list[float]) -> list[float]:
    n = norm3(v)
    if n == 0.0:
        raise ValueError("cannot normalize zero-length vector")
    return [v[0] / n, v[1] / n, v[2] / n]


def mat3_mul_vec(m: list[list[float]], v: list[float]) -> list[float]:
    return [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]


def mat3_mul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [
        [
            a[0][0] * b[0][0] + a[0][1] * b[1][0] + a[0][2] * b[2][0],
            a[0][0] * b[0][1] + a[0][1] * b[1][1] + a[0][2] * b[2][1],
            a[0][0] * b[0][2] + a[0][1] * b[1][2] + a[0][2] * b[2][2],
        ],
        [
            a[1][0] * b[0][0] + a[1][1] * b[1][0] + a[1][2] * b[2][0],
            a[1][0] * b[0][1] + a[1][1] * b[1][1] + a[1][2] * b[2][1],
            a[1][0] * b[0][2] + a[1][1] * b[1][2] + a[1][2] * b[2][2],
        ],
        [
            a[2][0] * b[0][0] + a[2][1] * b[1][0] + a[2][2] * b[2][0],
            a[2][0] * b[0][1] + a[2][1] * b[1][1] + a[2][2] * b[2][1],
            a[2][0] * b[0][2] + a[2][1] * b[1][2] + a[2][2] * b[2][2],
        ],
    ]


def mat3_transpose(m: list[list[float]]) -> list[list[float]]:
    return [
        [m[0][0], m[1][0], m[2][0]],
        [m[0][1], m[1][1], m[2][1]],
        [m[0][2], m[1][2], m[2][2]],
    ]


def rotation_from_euler_xyz(ax: float, ay: float, az: float) -> list[list[float]]:
    sx, cx = math.sin(ax), math.cos(ax)
    sy, cy = math.sin(ay), math.cos(ay)
    sz, cz = math.sin(az), math.cos(az)
    return [
        [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
        [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
        [-sy, cy * sx, cy * cx],
    ]


def quat_xyzw_from_rot(r: list[list[float]]) -> list[float]:
    trace = r[0][0] + r[1][1] + r[2][2]
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (r[2][1] - r[1][2]) / s
        y = (r[0][2] - r[2][0]) / s
        z = (r[1][0] - r[0][1]) / s
    elif r[0][0] > r[1][1] and r[0][0] > r[2][2]:
        s = math.sqrt(1.0 + r[0][0] - r[1][1] - r[2][2]) * 2.0
        w = (r[2][1] - r[1][2]) / s
        x = 0.25 * s
        y = (r[0][1] + r[1][0]) / s
        z = (r[0][2] + r[2][0]) / s
    elif r[1][1] > r[2][2]:
        s = math.sqrt(1.0 + r[1][1] - r[0][0] - r[2][2]) * 2.0
        w = (r[0][2] - r[2][0]) / s
        x = (r[0][1] + r[1][0]) / s
        y = 0.25 * s
        z = (r[1][2] + r[2][1]) / s
    else:
        s = math.sqrt(1.0 + r[2][2] - r[0][0] - r[1][1]) * 2.0
        w = (r[1][0] - r[0][1]) / s
        x = (r[0][2] + r[2][0]) / s
        y = (r[1][2] + r[2][1]) / s
        z = 0.25 * s
    return [x, y, z, w]


def make_pose(ax: float, ay: float, az: float, tx: float, ty: float, tz: float) -> dict:
    r = rotation_from_euler_xyz(ax, ay, az)
    q = quat_xyzw_from_rot(r)
    return {"rotation": q, "translation": [tx, ty, tz]}


def pose_to_rt(pose: dict) -> tuple[list[list[float]], list[float]]:
    x, y, z, w = pose["rotation"]
    tx, ty, tz = pose["translation"]
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z
    r = [
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
        [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
    ]
    return r, [tx, ty, tz]


def pose_compose(a: dict, b: dict) -> dict:
    ra, ta = pose_to_rt(a)
    rb, tb = pose_to_rt(b)
    r = mat3_mul(ra, rb)
    t = add3(mat3_mul_vec(ra, tb), ta)
    return {"rotation": quat_xyzw_from_rot(r), "translation": t}


def pose_inverse(pose: dict) -> dict:
    r, t = pose_to_rt(pose)
    rt = mat3_transpose(r)
    tinv = scale3(mat3_mul_vec(rt, t), -1.0)
    return {"rotation": quat_xyzw_from_rot(rt), "translation": tinv}


def transform_point(pose: dict, p: list[float]) -> list[float]:
    r, t = pose_to_rt(pose)
    return add3(mat3_mul_vec(r, p), t)


def project_point_pinhole(
    p_cam: list[float],
    intrinsics: dict,
    distortion: dict | None = None,
) -> list[float] | None:
    x, y, z = p_cam
    if z <= 0.0:
        return None
    xn = x / z
    yn = y / z
    if distortion is not None:
        k1 = float(distortion["k1"])
        k2 = float(distortion["k2"])
        k3 = float(distortion["k3"])
        p1 = float(distortion["p1"])
        p2 = float(distortion["p2"])
        r2 = xn * xn + yn * yn
        r4 = r2 * r2
        r6 = r4 * r2
        radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
        x_tan = 2.0 * p1 * xn * yn + p2 * (r2 + 2.0 * xn * xn)
        y_tan = p1 * (r2 + 2.0 * yn * yn) + 2.0 * p2 * xn * yn
        xd = xn * radial + x_tan
        yd = yn * radial + y_tan
    else:
        xd = xn
        yd = yn

    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    cx = float(intrinsics["cx"])
    cy = float(intrinsics["cy"])
    skew = float(intrinsics.get("skew", 0.0))
    u = fx * xd + skew * yd + cx
    v = fy * yd + cy
    return [u, v]


def grid_points(cols: int, rows: int, spacing_m: float) -> list[list[float]]:
    points: list[list[float]] = []
    for i in range(cols):
        for j in range(rows):
            points.append([i * spacing_m, j * spacing_m, 0.0])
    return points


def baseline_m(cam_se3_rig: list[object]) -> float:
    if len(cam_se3_rig) < 2:
        return 0.0
    def _translation(pose: object) -> list[float]:
        if isinstance(pose, dict):
            return list(pose["translation"])
        # vision_calibration.models.Pose
        return list(getattr(pose, "translation_xyz"))

    t0 = _translation(cam_se3_rig[0])
    t1 = _translation(cam_se3_rig[1])
    return norm3(sub3(t1, t0))
