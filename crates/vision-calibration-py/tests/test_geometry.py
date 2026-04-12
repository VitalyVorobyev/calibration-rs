"""Tests for vision_calibration.geometry Python bindings."""

import numpy as np
import pytest

from vision_calibration import RansacOptions, geometry


def _rotation_x(angle: float) -> np.ndarray:
    """Rotation matrix around X axis."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def _make_essential(r: np.ndarray, t: np.ndarray) -> np.ndarray:
    """E = [t]_x R."""
    tx = np.array(
        [[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]],
        dtype=np.float64,
    )
    return tx @ r


def _project(pts3d: np.ndarray, r: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Project 3D points to 2D using [R|t] (identity K)."""
    return (pts3d @ r.T + t.reshape(1, 3))[:, :2] / (
        pts3d @ r.T + t.reshape(1, 3)
    )[:, 2:3]


class TestFundamental:
    """Fundamental matrix solvers."""

    @pytest.fixture
    def scene(self):
        """Generate a synthetic scene with known F."""
        rng = np.random.default_rng(42)
        # Random 3D points in front of both cameras
        pts3d = rng.uniform(-1, 1, (20, 3))
        pts3d[:, 2] += 5  # ensure positive depth

        r = _rotation_x(0.1)
        t = np.array([0.5, 0.0, 0.1], dtype=np.float64)
        t = t / np.linalg.norm(t)

        pts1 = _project(pts3d, np.eye(3), np.zeros(3))
        pts2 = _project(pts3d, r, t)
        e = _make_essential(r, t)
        return pts1, pts2, e, r, t

    def test_fundamental_8point(self, scene):
        pts1, pts2, e, r, t = scene
        f = geometry.fundamental_8point(pts1, pts2)
        assert f.shape == (3, 3)
        # F should satisfy x2^T F x1 ≈ 0 for all correspondences
        for i in range(len(pts1)):
            x1 = np.array([pts1[i, 0], pts1[i, 1], 1.0])
            x2 = np.array([pts2[i, 0], pts2[i, 1], 1.0])
            assert abs(x2 @ f @ x1) < 0.01

    def test_fundamental_7point(self, scene):
        pts1, pts2, e, r, t = scene
        results = geometry.fundamental_7point(pts1[:7], pts2[:7])
        assert 1 <= len(results) <= 3
        for f in results:
            assert f.shape == (3, 3)

    def test_fundamental_8point_ransac(self, scene):
        pts1, pts2, e, r, t = scene
        opts = RansacOptions(max_iters=500, thresh=0.01, min_inliers=8, seed=42)
        f, inliers = geometry.fundamental_8point_ransac(pts1, pts2, opts)
        assert f.shape == (3, 3)
        assert len(inliers) >= 8


class TestEssential:
    """Essential matrix solvers."""

    @pytest.fixture
    def scene(self):
        rng = np.random.default_rng(123)
        pts3d = rng.uniform(-1, 1, (20, 3))
        pts3d[:, 2] += 5

        r = _rotation_x(0.15)
        t = np.array([0.3, -0.1, 0.05], dtype=np.float64)
        t = t / np.linalg.norm(t)

        pts1 = _project(pts3d, np.eye(3), np.zeros(3))
        pts2 = _project(pts3d, r, t)
        return pts1, pts2, r, t

    def test_essential_5point(self, scene):
        pts1, pts2, r, t = scene
        results = geometry.essential_5point(pts1[:5], pts2[:5])
        assert len(results) >= 1
        for e in results:
            assert e.shape == (3, 3)

    def test_decompose_essential(self, scene):
        pts1, pts2, r, t = scene
        e = _make_essential(r, t)
        decomps = geometry.decompose_essential(e)
        assert len(decomps) == 4
        for ri, ti in decomps:
            assert ri.shape == (3, 3)
            assert ti.shape == (3,)


class TestHomography:
    """Homography solvers."""

    @pytest.fixture
    def correspondences(self):
        """Planar correspondences with known H."""
        rng = np.random.default_rng(99)
        src = rng.uniform(-1, 1, (20, 2))
        # Simple homography: translation + slight rotation
        h = np.array(
            [[1.0, 0.1, 0.5], [-0.1, 1.0, 0.3], [0.01, -0.01, 1.0]],
            dtype=np.float64,
        )
        # Apply H to src points
        ones = np.ones((20, 1))
        src_h = np.hstack([src, ones])
        dst_h = (h @ src_h.T).T
        dst = dst_h[:, :2] / dst_h[:, 2:3]
        return src, dst, h

    def test_dlt_homography(self, correspondences):
        src, dst, h_true = correspondences
        h = geometry.dlt_homography(src, dst)
        assert h.shape == (3, 3)
        # Normalise and compare
        h = h / h[2, 2]
        h_true = h_true / h_true[2, 2]
        np.testing.assert_allclose(h, h_true, atol=1e-6)

    def test_dlt_homography_ransac(self, correspondences):
        src, dst, h_true = correspondences
        opts = RansacOptions(max_iters=200, thresh=0.01, min_inliers=4, seed=42)
        h, inliers = geometry.dlt_homography_ransac(src, dst, opts)
        assert h.shape == (3, 3)
        assert len(inliers) >= 4


class TestCameraMatrix:
    """Camera matrix estimation and decomposition."""

    def test_dlt_camera_matrix(self):
        rng = np.random.default_rng(7)
        # Known K, R, t
        k = np.array(
            [[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64
        )
        r = _rotation_x(0.2)
        t = np.array([0.1, -0.2, 5.0], dtype=np.float64)

        # 3D points
        pts3d = rng.uniform(-1, 1, (12, 3))
        pts3d[:, 2] += 3  # positive depth

        # Project
        p_true = k @ np.hstack([r, t.reshape(3, 1)])
        pts_h = np.hstack([pts3d, np.ones((12, 1))])
        proj = (p_true @ pts_h.T).T
        pts2d = proj[:, :2] / proj[:, 2:3]

        p = geometry.dlt_camera_matrix(pts3d, pts2d)
        assert p.shape == (3, 4)

    def test_decompose_camera_matrix(self):
        k = np.array(
            [[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64
        )
        r = _rotation_x(0.2)
        t = np.array([0.1, -0.2, 5.0], dtype=np.float64)
        p = k @ np.hstack([r, t.reshape(3, 1)])

        decomp = geometry.decompose_camera_matrix(p)
        assert isinstance(decomp, geometry.CameraMatrixDecomposition)
        assert decomp.k.shape == (3, 3)
        assert decomp.r.shape == (3, 3)
        assert decomp.t.shape == (3,)


class TestTriangulation:
    """Linear triangulation."""

    def test_triangulate_point_linear(self):
        k = np.eye(3, dtype=np.float64)
        r = _rotation_x(0.1)
        t = np.array([1.0, 0.0, 0.0], dtype=np.float64)

        p1 = k @ np.hstack([np.eye(3), np.zeros((3, 1))])
        p2 = k @ np.hstack([r, t.reshape(3, 1)])

        # A known 3D point
        pt3d = np.array([0.5, -0.3, 5.0])
        proj1 = p1 @ np.append(pt3d, 1.0)
        proj2 = p2 @ np.append(pt3d, 1.0)
        uv1 = (proj1[0] / proj1[2], proj1[1] / proj1[2])
        uv2 = (proj2[0] / proj2[2], proj2[1] / proj2[2])

        x, y, z = geometry.triangulate_point_linear([p1, p2], [uv1, uv2])
        np.testing.assert_allclose([x, y, z], pt3d, atol=0.01)
