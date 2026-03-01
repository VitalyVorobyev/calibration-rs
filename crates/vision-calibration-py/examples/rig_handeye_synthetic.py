"""Multi-camera rig hand-eye calibration with synthetic data.

Python counterpart of:
`crates/vision-calibration/examples/rig_handeye_synthetic.rs`
"""

from __future__ import annotations

import vision_calibration as vc

from _common import (
    baseline_m,
    grid_points,
    make_pose,
    pose_compose,
    pose_inverse,
    project_point_pinhole,
    transform_point,
)


def project_view(
    camera_intrinsics: dict,
    camera_distortion: dict,
    cam_se3_target: dict,
    board_points: list[list[float]],
) -> vc.Observation | None:
    points_3d = []
    points_2d = []
    for p in board_points:
        p_cam = transform_point(cam_se3_target, p)
        if p_cam[2] <= 0.0:
            continue
        uv = project_point_pinhole(p_cam, camera_intrinsics, camera_distortion)
        if uv is None:
            continue
        if 0.0 <= uv[0] < 1280.0 and 0.0 <= uv[1] < 720.0:
            points_3d.append(p)
            points_2d.append(uv)
    if len(points_2d) < 4:
        return None
    return vc.Observation(points_3d=points_3d, points_2d=points_2d)


def make_input() -> vc.RigHandeyeDataset:
    k0 = {"fx": 800.0, "fy": 780.0, "cx": 640.0, "cy": 360.0, "skew": 0.0}
    k1 = {"fx": 810.0, "fy": 790.0, "cx": 635.0, "cy": 355.0, "skew": 0.0}
    dist = {"k1": 0.0, "k2": 0.0, "k3": 0.0, "p1": 0.0, "p2": 0.0, "iters": 8}

    cam0_se3_rig = make_pose(0.0, 0.0, 0.0, 0.00, 0.00, 0.00)  # T_C0_R
    cam1_se3_rig = make_pose(0.0, 0.0, 0.0, 0.12, 0.00, 0.00)  # T_C1_R
    handeye_gt = make_pose(0.05, -0.03, 0.10, 0.00, -0.05, 0.08)  # T_G_R
    target_in_base_gt = make_pose(0.0, 0.0, 0.0, 0.0, 0.0, 1.0)  # T_B_T

    board = grid_points(cols=6, rows=5, spacing_m=0.05)
    robot_poses = [
        make_pose(0.0, 0.0, 0.0, 0.00, 0.00, 0.00),
        make_pose(0.15, 0.0, 0.0, 0.05, 0.00, 0.00),
        make_pose(0.0, 0.15, 0.0, 0.00, 0.05, 0.00),
        make_pose(0.0, 0.0, 0.15, 0.00, 0.00, 0.05),
        make_pose(0.10, 0.10, 0.0, 0.03, -0.03, 0.00),
        make_pose(-0.10, 0.0, 0.10, -0.03, 0.03, 0.00),
        make_pose(0.08, -0.08, 0.05, 0.00, -0.03, 0.02),
    ]

    views: list[vc.RigHandeyeView] = []
    for robot_pose in robot_poses:
        # Eye-in-hand rig chain: T_R_T = (T_B_G * T_G_R)^-1 * T_B_T
        rig_se3_target = pose_compose(pose_inverse(pose_compose(robot_pose, handeye_gt)), target_in_base_gt)
        cam0_se3_target = pose_compose(cam0_se3_rig, rig_se3_target)
        cam1_se3_target = pose_compose(cam1_se3_rig, rig_se3_target)
        obs0 = project_view(k0, dist, cam0_se3_target, board)
        obs1 = project_view(k1, dist, cam1_se3_target, board)
        if obs0 is None and obs1 is None:
            continue
        views.append(
            vc.RigHandeyeView(
                cameras=[obs0, obs1],
                base_se3_gripper=vc.Pose(
                    rotation_xyzw=tuple(robot_pose["rotation"]),
                    translation_xyz=tuple(robot_pose["translation"]),
                ),
            )
        )

    return vc.RigHandeyeDataset(num_cameras=2, views=views)


def main() -> None:
    print("=== Rig Hand-Eye Calibration (Synthetic, Python) ===")
    payload = make_input()
    result = vc.run_rig_handeye(payload)
    base_mm = baseline_m(result.cam_se3_rig) * 1000.0

    print(f"Views: {len(payload.views)}, cameras: {payload.num_cameras}")
    print(f"Hand-eye mode: {result.handeye_mode}")
    print(f"Mean reprojection error: {result.mean_reproj_error:.6f} px")
    print(f"Per-camera reprojection: {result.per_cam_reproj_errors}")
    print(f"Estimated baseline: {base_mm:.3f} mm")
    if result.gripper_se3_rig is not None:
        t = result.gripper_se3_rig.translation_xyz
        print(f"Recovered gripper->rig translation: [{t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}] m")


if __name__ == "__main__":
    main()
