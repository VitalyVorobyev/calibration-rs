"""Single-camera hand-eye calibration with synthetic data.

Python counterpart of:
`crates/vision-calibration/examples/handeye_synthetic.rs`
"""

from __future__ import annotations

import vision_calibration as vc

from _common import (
    grid_points,
    make_pose,
    pose_compose,
    pose_inverse,
    project_point_pinhole,
    transform_point,
)


def make_input() -> vc.SingleCamHandeyeDataset:
    intrinsics_gt = {"fx": 800.0, "fy": 780.0, "cx": 640.0, "cy": 360.0, "skew": 0.0}
    distortion_gt = {"k1": 0.0, "k2": 0.0, "k3": 0.0, "p1": 0.0, "p2": 0.0, "iters": 8}
    handeye_gt = make_pose(0.05, -0.03, 0.10, 0.10, -0.05, 0.02)  # T_G_C
    target_in_base_gt = make_pose(0.0, 0.0, 0.0, 0.0, 0.0, 1.0)  # T_B_T
    board = grid_points(cols=6, rows=5, spacing_m=0.05)

    robot_poses = [
        make_pose(0.0, 0.0, 0.0, 0.00, 0.00, 0.00),
        make_pose(0.3, 0.0, 0.0, 0.10, 0.00, 0.00),
        make_pose(0.0, 0.3, 0.0, 0.00, 0.10, 0.00),
        make_pose(0.0, 0.0, 0.3, 0.00, 0.00, 0.10),
        make_pose(0.2, 0.2, 0.0, 0.05, -0.05, 0.00),
        make_pose(-0.2, 0.0, 0.2, -0.05, 0.05, 0.00),
        make_pose(0.15, -0.15, 0.1, 0.00, -0.05, 0.05),
    ]

    views: list[vc.SingleCamHandeyeView] = []
    for robot_pose in robot_poses:
        # Eye-in-hand chain: T_C_T = (T_B_G * T_G_C)^-1 * T_B_T
        cam_pose = pose_compose(pose_inverse(pose_compose(robot_pose, handeye_gt)), target_in_base_gt)
        points_3d = []
        points_2d = []
        for p in board:
            p_cam = transform_point(cam_pose, p)
            uv = project_point_pinhole(p_cam, intrinsics_gt, distortion_gt)
            if uv is None:
                continue
            points_3d.append(p)
            points_2d.append(uv)
        if len(points_2d) >= 4:
            views.append(
                vc.SingleCamHandeyeView(
                    observation=vc.Observation(points_3d=points_3d, points_2d=points_2d),
                    base_se3_gripper=vc.Pose(
                        rotation_xyzw=tuple(robot_pose["rotation"]),
                        translation_xyz=tuple(robot_pose["translation"]),
                    ),
                )
            )
    return vc.SingleCamHandeyeDataset(views=views)


def main() -> None:
    print("=== Single-Camera Hand-Eye Calibration (Synthetic, Python) ===")
    payload = make_input()
    result = vc.run_single_cam_handeye(payload)
    k = result.camera["k"]

    print(f"Views: {len(payload.views)}")
    print(f"Hand-eye mode: {result.handeye_mode}")
    print(f"Mean reprojection error: {result.mean_reproj_error:.6f} px")
    print(f"Per-camera reprojection: {result.per_cam_reproj_errors}")
    print(f"Recovered intrinsics: fx={k['fx']:.3f}, fy={k['fy']:.3f}, cx={k['cx']:.3f}, cy={k['cy']:.3f}")
    if result.gripper_se3_camera is not None:
        t = result.gripper_se3_camera.translation_xyz
        print(f"Recovered gripper->camera translation: [{t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}] m")


if __name__ == "__main__":
    main()
