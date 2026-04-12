"""Tests for vision_calibration.mvg Python bindings."""

import numpy as np
import pytest

from vision_calibration import RansacOptions, mvg


def _rotation_x(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def _project(pts3d: np.ndarray, r: np.ndarray, t: np.ndarray) -> np.ndarray:
    projected = pts3d @ r.T + t.reshape(1, 3)
    return projected[:, :2] / projected[:, 2:3]


def _make_corrs(pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
    """Stack (N,2) + (N,2) into (N,4) correspondence array."""
    return np.hstack([pts1, pts2])


@pytest.fixture
def scene():
    """Synthetic two-view scene with known pose."""
    rng = np.random.default_rng(42)
    pts3d = rng.uniform(-1, 1, (30, 3))
    pts3d[:, 2] += 5

    r = _rotation_x(0.15)
    t = np.array([0.5, 0.0, 0.1], dtype=np.float64)
    t = t / np.linalg.norm(t)

    pts1 = _project(pts3d, np.eye(3), np.zeros(3))
    pts2 = _project(pts3d, r, t)
    return pts1, pts2, r, t, pts3d


class TestPoseRecovery:
    def test_recover_relative_pose(self, scene):
        pts1, pts2, r_true, t_true, _ = scene
        corrs = _make_corrs(pts1, pts2)
        result = mvg.recover_relative_pose(corrs)

        assert isinstance(result, mvg.RelativePose)
        assert result.r.shape == (3, 3)
        assert result.t.shape == (3,)
        assert result.essential.shape == (3, 3)
        assert len(result.points) > 0
        assert isinstance(result.points[0], mvg.TriangulatedPoint)

    def test_recover_relative_pose_robust(self, scene):
        pts1, pts2, r_true, t_true, _ = scene
        corrs = _make_corrs(pts1, pts2)
        opts = RansacOptions(max_iters=500, thresh=0.01, min_inliers=10, seed=42)
        result = mvg.recover_relative_pose_robust(corrs, opts)

        assert isinstance(result, mvg.RobustRelativePose)
        assert result.r.shape == (3, 3)
        assert result.t.shape == (3,)
        assert len(result.inliers) >= 10
        assert result.inlier_rms >= 0.0


class TestEstimation:
    def test_estimate_essential(self, scene):
        pts1, pts2, _, _, _ = scene
        corrs = _make_corrs(pts1, pts2)
        opts = RansacOptions(max_iters=500, thresh=0.01, min_inliers=10, seed=42)
        result = mvg.estimate_essential(corrs, opts)

        assert isinstance(result, mvg.EssentialEstimate)
        assert result.essential.shape == (3, 3)
        assert len(result.inliers) >= 10

    def test_estimate_homography(self):
        rng = np.random.default_rng(99)
        # Planar scene
        src = rng.uniform(-1, 1, (30, 2))
        h = np.array(
            [[1.0, 0.1, 0.5], [-0.1, 1.0, 0.3], [0.01, -0.01, 1.0]],
            dtype=np.float64,
        )
        ones = np.ones((30, 1))
        src_h = np.hstack([src, ones])
        dst_h = (h @ src_h.T).T
        dst = dst_h[:, :2] / dst_h[:, 2:3]

        corrs = _make_corrs(src, dst)
        opts = RansacOptions(max_iters=200, thresh=0.01, min_inliers=4, seed=42)
        result = mvg.estimate_homography(corrs, opts)

        assert isinstance(result, mvg.HomographyEstimate)
        assert result.homography.shape == (3, 3)
        assert len(result.inliers) >= 4


class TestHomography:
    def test_decompose_homography(self):
        h = np.array(
            [[1.0, 0.1, 0.5], [-0.1, 1.0, 0.3], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        decomps = mvg.decompose_homography(h)
        assert len(decomps) > 0
        for d in decomps:
            assert isinstance(d, mvg.HomographyDecomposition)
            assert d.r.shape == (3, 3)
            assert d.t.shape == (3,)
            assert d.normal.shape == (3,)

    def test_homography_transfer(self):
        h = np.eye(3, dtype=np.float64)
        h[0, 2] = 1.0  # translation
        x, y = mvg.homography_transfer(h, (2.0, 3.0))
        np.testing.assert_allclose([x, y], [3.0, 3.0], atol=1e-10)


class TestTriangulation:
    def test_triangulate_two_view(self, scene):
        pts1, pts2, r, t, _ = scene
        result = mvg.triangulate_two_view(r, t, pts1, pts2)

        assert len(result) == len(pts1)
        for tp in result:
            assert isinstance(tp, mvg.TriangulatedPoint)
            assert len(tp.point) == 3


class TestSceneAnalysis:
    def test_analyze_scene(self, scene):
        pts1, pts2, r, t, _ = scene
        corrs = _make_corrs(pts1, pts2)
        tx = np.array(
            [[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]],
            dtype=np.float64,
        )
        e = tx @ r
        result = mvg.analyze_scene(corrs, e, r, t)

        assert isinstance(result, mvg.SceneDiagnostics)
        assert result.median_parallax_deg >= 0.0
        assert isinstance(result.is_pure_rotation, bool)
        assert isinstance(result.is_planar, bool)


class TestResiduals:
    def test_sampson_distance(self, scene):
        pts1, pts2, r, t, _ = scene
        tx = np.array(
            [[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]],
            dtype=np.float64,
        )
        e = tx @ r
        d = mvg.sampson_distance(e, (pts1[0, 0], pts1[0, 1]), (pts2[0, 0], pts2[0, 1]))
        assert d >= 0.0
        # For exact correspondences, Sampson distance should be very small
        assert d < 0.01

    def test_symmetric_transfer_error(self):
        h = np.eye(3, dtype=np.float64)
        err = mvg.symmetric_transfer_error(h, (1.0, 2.0), (1.0, 2.0))
        assert err < 1e-10
