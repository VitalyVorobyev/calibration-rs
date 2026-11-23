from __future__ import annotations

import math
import unittest

import vision_calibration as vc
from _fixtures import euler_rot_xyz, planar_calibration_dataset, transform_point

# The focus law the blur is rendered with: the plane of sharp focus π·X = 1
# (1/m), the blur per unit defocus κ (px·m) and the optical blur at focus σ0²
# (px²). The fixture's camera is untilted and without distortion, so the spot
# scale is 1.
PLANE = (0.2, -0.1, 1.0 / 0.75)
KAPPA = 8.0
SIGMA0_SQ = 0.25
COV = ((1e-4, 0.0, 0.0), (0.0, 5e-5, 0.0), (0.0, 0.0, 1e-4))


def _with_blur(dataset: vc.PlanarIntrinsicsInput, cov=COV) -> vc.PlanarIntrinsicsInput:
    """``dataset`` (the fixture's views, in its pose order) with each point's
    blur rendered from the focus law at the fixture's true pose."""
    views = []
    for i, view in enumerate(dataset.views):
        R = euler_rot_xyz(0.02 * i, -0.18 + 0.04 * i, -0.20 + 0.06 * i)
        t = [-0.04 + 0.02 * i, 0.02 - 0.01 * i, 0.55 + 0.04 * i]
        obs = view.observation
        blur = []
        for p in obs.points_3d:
            x = transform_point(R, t, list(p))
            d = (1.0 - sum(a * b for a, b in zip(PLANE, x))) / x[2]
            var = SIGMA0_SQ + 1.0 / 12.0 + (KAPPA * d) ** 2
            blur.append(vc.PointBlur(psf_px=(var, 0.0, var), cov=cov))
        views.append(
            vc.PlanarView(
                observation=vc.Observation(points_3d=obs.points_3d, points_2d=obs.points_2d, blur=blur)
            )
        )
    return vc.PlanarIntrinsicsInput(views=views)


class FitFocusTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = planar_calibration_dataset()
        self.result = vc.planar_intrinsics.run_calibration(self.dataset)

    def test_recovers_the_rendered_plane_of_sharp_focus(self) -> None:
        focus = vc.fit_focus(
            _with_blur(self.dataset),
            self.result.camera,
            self.result.cam_se3_target,
            blur_limit_px=1.5,
            view_jackknife=True,
        )
        self.assertEqual(focus["status"], "fitted")
        for got, want in zip(focus["plane_inv_m"], PLANE):
            self.assertAlmostEqual(got, want, delta=1e-5)
        self.assertAlmostEqual(focus["blur_scale_px_m"], KAPPA, delta=1e-4)
        self.assertAlmostEqual(focus["in_focus_variance_px2"], SIGMA0_SQ, delta=1e-4)
        self.assertTrue(focus["stated_weights"])
        self.assertEqual(len(focus["views"]), len(self.dataset.views))
        self.assertIsNotNone(focus["depth_of_field"])
        self.assertEqual(focus["view_jackknife"]["status"], "estimated")

    def test_reports_why_there_is_no_fit(self) -> None:
        focus = vc.fit_focus(self.dataset, self.result.camera, self.result.cam_se3_target)
        self.assertEqual(focus["status"], "no_blur")
        self.assertIsNone(
            vc.fit_focus(_with_blur(self.dataset), self.result.camera, self.result.cam_se3_target).get(
                "view_jackknife"
            )
        )

    def test_rejects_mixed_blur_weights_and_a_pose_count_mismatch(self) -> None:
        stated = _with_blur(self.dataset)
        unstated = _with_blur(self.dataset, cov=((0.0,) * 3,) * 3)
        mixed = vc.PlanarIntrinsicsInput(views=[*stated.views[:5], *unstated.views[5:]])
        with self.assertRaises(ValueError):
            vc.fit_focus(mixed, self.result.camera, self.result.cam_se3_target)
        with self.assertRaises(ValueError):
            vc.fit_focus(stated, self.result.camera, self.result.cam_se3_target[:-1])
        # All unstated: every point weighs the same.
        focus = vc.fit_focus(unstated, self.result.camera, self.result.cam_se3_target)
        self.assertEqual(focus["status"], "fitted")
        self.assertFalse(focus["stated_weights"])
        self.assertTrue(math.isfinite(focus["noise_scale"]))


if __name__ == "__main__":
    unittest.main()
