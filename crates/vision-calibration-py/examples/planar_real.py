"""Planar intrinsics calibration from real images.

Python counterpart of:
`crates/vision-calibration/examples/planar_real.rs`

Uses the 30 images of `data/kuka_1/` (a 17x28 chessboard with 20 mm squares).
The images are stored with Git LFS; run `git lfs pull` first if they are
pointer files in your checkout.
"""

from __future__ import annotations

import vision_calibration as vc

from _common import repo_root
from handeye_session import detect_observation, parse_square_size_m


def load_dataset() -> vc.PlanarIntrinsicsInput:
    data_dir = repo_root() / "data" / "kuka_1"
    square_size_m = parse_square_size_m(data_dir / "squaresize.txt")

    views: list[vc.PlanarView] = []
    skipped = 0
    for image in sorted(data_dir.glob("*.png")):
        obs = detect_observation(image, square_size_m)
        if obs is None:
            skipped += 1
            continue
        views.append(vc.PlanarView(observation=obs))

    if len(views) < 3:
        raise RuntimeError(f"need at least 3 usable views, got {len(views)}")
    print(f"Loaded {len(views)} views ({skipped} skipped)")
    return vc.PlanarIntrinsicsInput(views=views)


def main() -> None:
    print("=== Planar Intrinsics Calibration (Real Images, Python) ===")
    dataset = load_dataset()
    result = vc.run_planar_intrinsics(dataset)
    k = result.camera.intrinsics
    d = result.camera.distortion

    print(f"Mean reprojection error: {result.mean_reproj_error:.6f} px")
    print(f"Intrinsics: fx={k.fx:.3f}, fy={k.fy:.3f}, cx={k.cx:.3f}, cy={k.cy:.3f}")
    print(f"Distortion: k1={d.k1:.6f}, k2={d.k2:.6f}, k3={d.k3:.6f}, p1={d.p1:.6f}, p2={d.p2:.6f}")

    u = result.uncertainty
    if u.available:
        print(f"fx = {k.fx:.2f} ± {u.std_dev('fx'):.2f} px, cx = {k.cx:.2f} ± {u.std_dev('cx'):.2f} px")
        print(f"corr(fx, cx) = {u.correlation('fx', 'cx'):+.3f}")
    else:
        print("No uncertainty:", u.reason, u.reason_labels)


if __name__ == "__main__":
    main()
