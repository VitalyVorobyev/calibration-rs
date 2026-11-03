"""Measure a board of known size with a stereo calibration, error bars included.

Python counterpart of:
`crates/vision-calibration/examples/measure_board.rs`

Calibrates the fisheye stereo pair of OpenCV's own test set on half of its
views, triangulates the chessboard corners of the other half, and measures the
distances between corners of a row or column. The board's squares are 24.4 mm,
so every distance has a known truth, and the example compares the error with
the sigma the library states, with and without the calibration's share in it.

The sample is downloaded from its publisher on the first run (set
`VISION_CALIBRATION_DATA` to choose the cache directory).

    python measure_board.py
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np

import vision_calibration as vc
from vision_calibration import dataset_runner, mvg, samples

MAX_SQUARES = 7  # longest distance measured, in squares


def without_calibration_share(result: vc.RigExtrinsicsExport) -> dict:
    """The export with its covariance shrunk to nothing: the sigma of the
    observations alone, to show what the calibration's share adds."""
    export = copy.deepcopy(result.export)
    n = len(export["uncertainty"]["labels"])
    export["uncertainty"]["covariance"] = (1e-20 * np.eye(n)).tolist()
    return export


def measure_view(rig, view: dict, square: float):
    """Triangulate the corners both cameras saw in `view` and measure the
    distances between corners of a row or a column.

    The view is one capture with the rig at the origin, so the points come out
    in the rig frame. The pixel noise is the calibration's own reprojection
    level of each camera. Returns `(distances, plane_rms)`, each distance a
    `(squares, value, sigma, truth)` tuple in metres, or `None` when the
    cameras share too few corners.
    """
    left, right = view["obs"]["cameras"][:2]
    if left is None or right is None:
        return None
    noise = [mvg.reprojection_noise(rig, camera) for camera in (0, 1)]
    # A corner is identified by its position on the board, the same
    # coordinates in both cameras.
    in_right = {tuple(p): px for p, px in zip(right["points_3d"], right["points_2d"])}
    board, tracks = [], []
    for p, px in zip(left["points_3d"], left["points_2d"]):
        if tuple(p) in in_right:
            board.append(np.array(p))
            tracks.append(
                [
                    mvg.RigObservation(0, 0, px, noise[0]),
                    mvg.RigObservation(0, 1, in_right[tuple(p)], noise[1]),
                ]
            )
    if len(board) < 3:
        return None

    found = mvg.triangulate(rig, np.eye(4), tracks)
    points = mvg.PointSet.from_triangulation(found)
    distances = []
    for a in range(len(board)):
        for b in range(a + 1, len(board)):
            d = board[b] - board[a]
            on_row = abs(d[1]) < 1e-9 * square
            on_column = abs(d[0]) < 1e-9 * square
            squares = round(np.linalg.norm(d) / square)
            if on_row == on_column or not 1 <= squares <= MAX_SQUARES:
                continue
            length = mvg.distance(points, a, b)
            distances.append((squares, length.value, length.sigma, float(np.linalg.norm(d))))
    plane = mvg.fit_plane(points, range(len(board)))
    return distances, plane.rms


def summarize(views):
    """Per number of squares: pairs, mean error, rms error, mean sigma, z rms."""
    rows = {}
    for k in range(1, MAX_SQUARES + 1):
        m = np.array([d for distances, _ in views for d in distances if d[0] == k])
        if len(m) == 0:
            continue
        error, sigma = m[:, 1] - m[:, 3], m[:, 2]
        rows[k] = (
            len(m),
            error.mean(),
            np.sqrt((error**2).mean()),
            sigma.mean(),
            np.sqrt(((error / sigma) ** 2).mean()),
        )
    return rows


def main() -> None:
    # 1. The sample: 34 stereo views of a 6x8-corner chessboard, 24.4 mm squares.
    manifest = samples.fetch("opencv_fisheye_stereo")
    square = 0.0244
    cache = vc.FsDetectionCache(manifest.parent / ".detections")
    run = dataset_runner.build_rig_extrinsics_input(manifest, cache=cache)
    print(f"{run.usable_views} of {run.total_views} views usable")

    # 2. Split the views: even ones calibrate, odd ones are held out. Each view
    #    keeps both cameras' detections, so the pairing stays intact.
    views = run.input["views"]
    calibration, held_out = views[0::2], views[1::2]
    print(f"{len(calibration)} calibration views, {len(held_out)} held out")

    # 3. Calibrate the rig with the fisheye model (equidistant projection,
    #    Kannala-Brandt distortion).
    def observation(camera):
        if camera is None:
            return None
        if camera.get("cov_px") is not None:
            raise ValueError("this example expects views without stated pixel covariances")
        return vc.Observation(points_3d=camera["points_3d"], points_2d=camera["points_2d"])

    dataset = vc.RigExtrinsicsInput(
        num_cameras=run.input["num_cameras"],
        views=[
            vc.RigExtrinsicsView(cameras=[observation(c) for c in v["obs"]["cameras"]])
            for v in calibration
        ],
    )
    config = vc.RigExtrinsicsConfig()
    config.camera.model = vc.CameraModelSpec(projection="equidistant", distortion="kannala_brandt4")
    result = vc.run_rig_extrinsics(dataset, config)
    scatter = result.uncertainty.view_scatter
    print(
        f"calibration: mean reprojection error {result.mean_reproj_error:.3f} px, "
        f"view scatter {'none' if scatter is None else f'{scatter:.2f}'}, "
        f"sigma-hat {mvg.reprojection_noise(result, 0).xx ** 0.5:.3f} / "
        f"{mvg.reprojection_noise(result, 1).xx ** 0.5:.3f} px"
    )

    # 4. Measure every held-out view with the full sigma and with the
    #    observations' share alone.
    bare = without_calibration_share(result)
    with_share, without_share = [], []
    for view in held_out:
        a, b = measure_view(result, view, square), measure_view(bare, view, square)
        if a is not None and b is not None:
            with_share.append(a)
            without_share.append(b)
    if not with_share:
        sys.exit("no held-out view could be measured")
    plane_rms = np.mean([rms for _, rms in with_share])
    print(
        f"{len(with_share)} of {len(held_out)} held-out views measured "
        f"(both cameras need the board); board planarity rms {plane_rms * 1e3:.3f} mm"
    )

    # 5. Error against the sigma the library states, per length.
    full, only = summarize(with_share), summarize(without_share)
    print(
        "\nsquares  pairs  mean err   rms err  mean sigma   z rms | "
        "sigma without calibration: mean sigma   z rms   (mm)"
    )
    for k, (n, mean_err, rms_err, mean_sigma, z) in full.items():
        o = only[k]
        print(
            f"{k:7}  {n:5}  {mean_err * 1e3:+8.3f}  {rms_err * 1e3:8.3f}  "
            f"{mean_sigma * 1e3:10.3f}  {z:6.2f} | {o[3] * 1e3:41.3f}  {o[4]:6.2f}"
        )


if __name__ == "__main__":
    main()
