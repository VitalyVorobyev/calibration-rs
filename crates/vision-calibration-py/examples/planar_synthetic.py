"""Planar intrinsics calibration with deterministic synthetic data.

Python counterpart of:
`crates/vision-calibration/examples/planar_synthetic.rs`
"""

from __future__ import annotations

import vision_calibration as vc

from _common import (
    grid_points,
    make_pose,
    project_point_pinhole,
    transform_point,
)


def make_input() -> vc.PlanarDataset:
    intrinsics_gt = {"fx": 800.0, "fy": 780.0, "cx": 640.0, "cy": 360.0, "skew": 0.0}
    distortion_gt = {"k1": 0.05, "k2": -0.02, "k3": 0.0, "p1": 0.001, "p2": -0.001, "iters": 8}
    board = grid_points(cols=8, rows=6, spacing_m=0.04)

    poses = [
        make_pose(0.02 * i, -0.18 + 0.04 * i, -0.20 + 0.06 * i, -0.04 + 0.02 * i, 0.02 - 0.01 * i, 0.55 + 0.04 * i)
        for i in range(8)
    ]

    views: list[vc.PlanarView] = []
    for pose in poses:
        points_3d = []
        points_2d = []
        for p in board:
            p_cam = transform_point(pose, p)
            uv = project_point_pinhole(p_cam, intrinsics_gt, distortion_gt)
            if uv is None:
                continue
            points_3d.append(p)
            points_2d.append(uv)
        if len(points_2d) >= 4:
            views.append(vc.PlanarView(observation=vc.Observation(points_3d=points_3d, points_2d=points_2d)))

    return vc.PlanarDataset(views=views)


def main() -> None:
    print("=== Planar Intrinsics Calibration (Synthetic Data, Python) ===")
    payload = make_input()
    result = vc.run_planar_intrinsics(payload)
    k = result.camera["k"]
    d = result.camera["dist"]

    print(f"Views: {len(payload.views)}")
    print(f"Final cost: {result.final_cost:.3e}")
    print(f"Mean reprojection error: {result.mean_reproj_error:.6f} px")
    print("Recovered intrinsics:")
    print(f"  fx={k['fx']:.3f}, fy={k['fy']:.3f}, cx={k['cx']:.3f}, cy={k['cy']:.3f}")
    print("Recovered distortion:")
    print(f"  k1={d['k1']:.6f}, k2={d['k2']:.6f}, p1={d['p1']:.6f}, p2={d['p2']:.6f}")


if __name__ == "__main__":
    main()
