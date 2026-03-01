"""Single laserline-device calibration with synthetic data.

Python counterpart of:
`crates/vision-calibration/examples/laserline_device_session.rs`
"""

from __future__ import annotations

import vision_calibration as vc

from _common import (
    add3,
    dot3,
    grid_points,
    make_pose,
    mat3_mul_vec,
    mat3_transpose,
    norm3,
    normalize3,
    pose_to_rt,
    project_point_pinhole,
    scale3,
    transform_point,
)


def make_poses() -> list[dict]:
    return [
        make_pose(0.0, 0.0, 0.0, 0.00, 0.00, 0.50),
        make_pose(0.15, -0.05, 0.0, 0.05, -0.02, 0.55),
        make_pose(-0.10, 0.08, 0.02, -0.04, 0.03, 0.60),
        make_pose(0.05, 0.12, -0.04, 0.02, 0.06, 0.52),
        make_pose(-0.12, -0.04, 0.06, -0.06, -0.04, 0.58),
    ]


def make_laser_plane(poses: list[dict]) -> dict:
    normal = normalize3([0.1, 0.05, 0.99])
    target_center = [0.075, 0.06, 0.0]
    center_cam = transform_point(poses[0], target_center)
    return {"normal": normal, "distance": -dot3(normal, center_cam)}


def laser_pixels_for_view(
    pose: dict,
    plane: dict,
    intrinsics: dict,
    distortion: dict,
) -> list[list[float]]:
    r, t = pose_to_rt(pose)
    rt = mat3_transpose(r)
    n_c = list(plane["normal"])
    n_t = mat3_mul_vec(rt, n_c)
    d_t = dot3(n_c, t) + float(plane["distance"])

    direction = [n_t[1], -n_t[0], 0.0]
    direction_norm = norm3(direction)
    if direction_norm < 1e-12:
        return []
    direction = scale3(direction, 1.0 / direction_norm)

    if abs(n_t[0]) > abs(n_t[1]):
        x0, y0 = -d_t / n_t[0], 0.0
    else:
        x0, y0 = 0.0, -d_t / n_t[1]

    pixels: list[list[float]] = []
    for i in range(40):
        s = (i / 39.0) * 0.2 - 0.1
        p_target = [x0 + s * direction[0], y0 + s * direction[1], 0.0]
        p_cam = transform_point(pose, p_target)
        uv = project_point_pinhole(p_cam, intrinsics, distortion)
        if uv is None:
            continue
        if not (uv[0] == uv[0] and uv[1] == uv[1]):
            continue
        if p_cam[2] <= 1e-6:
            continue
        ray_dir_cam = normalize3(p_cam)
        ray_dir_target = mat3_mul_vec(rt, ray_dir_cam)
        if abs(ray_dir_target[2]) <= 1e-3:
            continue
        if not (0.0 <= uv[0] < 1280.0 and 0.0 <= uv[1] < 720.0):
            continue
        pixels.append(uv)
    return pixels


def make_input() -> vc.LaserlineDataset:
    intrinsics = {"fx": 800.0, "fy": 780.0, "cx": 640.0, "cy": 360.0, "skew": 0.0}
    distortion = {"k1": 0.0, "k2": 0.0, "k3": 0.0, "p1": 0.0, "p2": 0.0, "iters": 8}
    points_3d = grid_points(cols=6, rows=5, spacing_m=0.03)
    poses = make_poses()
    plane = make_laser_plane(poses)

    views: list[vc.LaserlineView] = []
    for pose in poses:
        obs_3d: list[list[float]] = []
        obs_2d: list[list[float]] = []
        for p in points_3d:
            p_cam = transform_point(pose, p)
            uv = project_point_pinhole(p_cam, intrinsics, distortion)
            if uv is None:
                continue
            obs_3d.append(p)
            obs_2d.append(uv)

        laser_pixels = laser_pixels_for_view(pose, plane, intrinsics, distortion)
        if len(obs_2d) >= 4 and laser_pixels:
            views.append(
                vc.LaserlineView(
                    observation=vc.Observation(points_3d=obs_3d, points_2d=obs_2d),
                    laser_pixels=[tuple(p) for p in laser_pixels],
                    laser_weights=None,
                )
            )

    return vc.LaserlineDataset(views=views)


def main() -> None:
    print("=== Laserline Device Calibration (Synthetic, Python) ===")
    payload = make_input()
    result = vc.run_laserline_device(payload)
    intr = result.estimate["params"]["intrinsics"]
    plane = result.estimate["params"]["plane"]

    print(f"Views: {len(payload.views)}")
    print(f"Final cost: {result.estimate['report']['final_cost']:.3e}")
    print(f"Mean reprojection error: {result.mean_reproj_error:.6f} px")
    print(f"Mean laser error: {result.mean_laser_error:.6f}")
    print(f"Recovered intrinsics: fx={intr['fx']:.3f}, fy={intr['fy']:.3f}, cx={intr['cx']:.3f}, cy={intr['cy']:.3f}")
    print(f"Recovered plane: normal={plane['normal']}, distance={plane['distance']:.6f}")


if __name__ == "__main__":
    main()
