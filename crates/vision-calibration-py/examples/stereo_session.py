"""Stereo rig extrinsics calibration from detected chessboard corners.

Python counterpart of:
`crates/vision-calibration/examples/stereo_session.rs`

The corners of a 7x11 chessboard with 30 mm squares, detected in 20 image
pairs of a stereo camera, are read from
`regression/observations/stereo_rig.json`.
"""

from __future__ import annotations

import vision_calibration as vc

from _common import baseline_m, load_observation_input


def load_dataset() -> vc.RigExtrinsicsInput:
    payload = load_observation_input("stereo_rig")
    views = [
        vc.RigExtrinsicsView(
            cameras=[
                None
                if cam is None
                else vc.Observation(points_3d=cam["points_3d"], points_2d=cam["points_2d"])
                for cam in view["obs"]["cameras"]
            ]
        )
        for view in payload["views"]
    ]
    print(f"Loaded {len(views)} views")
    return vc.RigExtrinsicsInput(num_cameras=payload["num_cameras"], views=views)


def main() -> None:
    print("=== Stereo Rig Calibration Session (Python) ===")
    dataset = load_dataset()
    result = vc.run_rig_extrinsics(dataset)
    base_mm = baseline_m(result.cam_se3_rig) * 1000.0

    print(f"Mean reprojection error: {result.mean_reproj_error:.6f} px")
    print(f"Per-camera reprojection: {result.per_cam_reproj_errors}")
    print(f"Estimated baseline: {base_mm:.3f} mm")
    for idx, cam in enumerate(result.cameras):
        k = cam.intrinsics
        d = cam.distortion
        print(f"Camera {idx}: fx={k.fx:.3f}, fy={k.fy:.3f}, cx={k.cx:.3f}, cy={k.cy:.3f}")
        print(f"  Distortion: k1={d.k1:.6f}, k2={d.k2:.6f}, p1={d.p1:.6f}, p2={d.p2:.6f}")


if __name__ == "__main__":
    main()
