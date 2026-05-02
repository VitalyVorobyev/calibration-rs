#!/usr/bin/env python3
"""Patch privatedata/130x130_puzzle/export.json with rig_se3_target.

The puzzle 130x130 export was generated before RigHandeyeExport carried
rig_se3_target. The 3D viewer needs that field to render the per-pose
target boards. Re-running the full puzzle pipeline takes minutes; the
chain is already deterministic so we recompute rig_se3_target here from
the fields that *are* in the export, plus the source robot poses in
poses.json.

For EyeToHand:  T_R_T = T_R_B * T_B_G_corrected * T_G_T

This script is one-off plumbing; once we re-run the pipeline (with the
new export schema), the rig_se3_target field will be populated natively.
"""

import json
import math
from pathlib import Path

DATA = Path("privatedata/130x130_puzzle")
EXPORT = DATA / "export.json"
POSES = DATA / "poses.json"


def quat_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def quat_rotate(q, v):
    qx, qy, qz, qw = q
    vx, vy, vz = v
    # qv = q * (vx, vy, vz, 0) * q^-1, but use the more efficient form
    # https://gamedev.stackexchange.com/a/50545
    t = (
        2 * (qy * vz - qz * vy),
        2 * (qz * vx - qx * vz),
        2 * (qx * vy - qy * vx),
    )
    return (
        vx + qw * t[0] + (qy * t[2] - qz * t[1]),
        vy + qw * t[1] + (qz * t[0] - qx * t[2]),
        vz + qw * t[2] + (qx * t[1] - qy * t[0]),
    )


def iso_compose(a, b):
    """Compose two iso3 dicts: result = a * b."""
    aq = a["rotation"]
    at = a["translation"]
    bq = b["rotation"]
    bt = b["translation"]
    cq = quat_mul(aq, bq)
    btr = quat_rotate(aq, bt)
    return {
        "rotation": list(cq),
        "translation": [at[0] + btr[0], at[1] + btr[1], at[2] + btr[2]],
    }


def matrix_to_iso(m):
    """Convert a 4x4 row-major homogeneous matrix to an iso3 (translation
    in mm → m, mirroring PoseEntry::base_se3_gripper in the Rust loader)."""
    # Translation in mm; convert to m.
    tx = m[0][3] * 1e-3
    ty = m[1][3] * 1e-3
    tz = m[2][3] * 1e-3
    # Quaternion from 3x3 rotation block (Shepperd's method, scalar-last).
    r00, r01, r02 = m[0][0], m[0][1], m[0][2]
    r10, r11, r12 = m[1][0], m[1][1], m[1][2]
    r20, r21, r22 = m[2][0], m[2][1], m[2][2]
    tr = r00 + r11 + r22
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * s
        qx = (r21 - r12) / s
        qy = (r02 - r20) / s
        qz = (r10 - r01) / s
    elif (r00 > r11) and (r00 > r22):
        s = math.sqrt(1.0 + r00 - r11 - r22) * 2.0
        qw = (r21 - r12) / s
        qx = 0.25 * s
        qy = (r01 + r10) / s
        qz = (r02 + r20) / s
    elif r11 > r22:
        s = math.sqrt(1.0 + r11 - r00 - r22) * 2.0
        qw = (r02 - r20) / s
        qx = (r01 + r10) / s
        qy = 0.25 * s
        qz = (r12 + r21) / s
    else:
        s = math.sqrt(1.0 + r22 - r00 - r11) * 2.0
        qw = (r10 - r01) / s
        qx = (r02 + r20) / s
        qy = (r12 + r21) / s
        qz = 0.25 * s
    return {
        "rotation": [qx, qy, qz, qw],
        "translation": [tx, ty, tz],
    }


def apply_robot_delta(base_se3_gripper, delta):
    """Mirror the Rust apply_robot_delta: T_B_G_corr = exp(delta) * T_B_G."""
    rx, ry, rz, tx, ty, tz = delta
    angle = math.sqrt(rx * rx + ry * ry + rz * rz)
    if angle > 1e-12:
        ax, ay, az = rx / angle, ry / angle, rz / angle
        s = math.sin(angle * 0.5)
        delta_rot = [ax * s, ay * s, az * s, math.cos(angle * 0.5)]
    else:
        delta_rot = [0.0, 0.0, 0.0, 1.0]
    delta_iso = {"rotation": delta_rot, "translation": [tx, ty, tz]}
    return iso_compose(delta_iso, base_se3_gripper)


def main() -> None:
    export = json.loads(EXPORT.read_text())
    poses = json.loads(POSES.read_text())

    if export.get("rig_se3_target"):
        print(
            f"export already has rig_se3_target ({len(export['rig_se3_target'])} entries) — nothing to do"
        )
        return

    mode = export["handeye_mode"]
    deltas = export.get("robot_deltas") or [[0.0] * 6] * len(poses)
    if mode != "EyeToHand":
        raise SystemExit(f"unexpected handeye_mode {mode!r}; only EyeToHand patched")

    rig_se3_base = export["rig_se3_base"]
    gripper_se3_target = export["gripper_se3_target"]

    rig_se3_target = []
    for i, pose in enumerate(poses):
        base_se3_gripper = matrix_to_iso(pose["tcp2base"])
        if i < len(deltas):
            base_se3_gripper = apply_robot_delta(base_se3_gripper, deltas[i])
        # T_R_T = T_R_B * T_B_G * T_G_T
        rig_se3_target.append(
            iso_compose(iso_compose(rig_se3_base, base_se3_gripper), gripper_se3_target)
        )

    # Splice in alphabetical-ish position: after cam_se3_rig.
    new_export = {}
    inserted = False
    for k, v in export.items():
        new_export[k] = v
        if k == "cam_se3_rig":
            new_export["rig_se3_target"] = rig_se3_target
            inserted = True
    if not inserted:
        new_export["rig_se3_target"] = rig_se3_target

    EXPORT.write_text(json.dumps(new_export, indent=2))
    print(f"patched {EXPORT} with {len(rig_se3_target)} rig_se3_target entries")


if __name__ == "__main__":
    main()
