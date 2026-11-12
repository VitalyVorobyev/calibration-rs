"""Progress reports and stopping a run: the ``progress`` callable sees each
step and solver iteration without changing the result, and raising from it,
or Ctrl-C, stops the run with that exception."""

from __future__ import annotations

import _thread
import math
import unittest
from typing import Any

import vision_calibration as vc

from _fixtures import planar_calibration_dataset


def noisy_dataset() -> vc.PlanarIntrinsicsInput:
    """The planar fixture with deterministic sub-pixel noise, so the
    refinement takes several iterations."""
    views = []
    for i, view in enumerate(planar_calibration_dataset().views):
        obs = view.observation
        noisy = [
            (u + 0.3 * math.sin(7 * i + 3 * k), v + 0.3 * math.cos(5 * i + 2 * k))
            for k, (u, v) in enumerate(obs.points_2d)
        ]
        views.append(
            vc.PlanarView(observation=vc.Observation(points_3d=obs.points_3d, points_2d=noisy))
        )
    return vc.PlanarIntrinsicsInput(views=views)


class Stop(Exception):
    """Raised by a progress callable to stop the run."""


class ProgressTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = noisy_dataset()

    def test_progress_sees_steps_and_iterations_and_changes_nothing(self) -> None:
        reports: list[dict[str, Any]] = []
        observed = vc.planar_intrinsics.run_calibration(self.dataset, progress=reports.append)
        plain = vc.planar_intrinsics.run_calibration(self.dataset)
        self.assertEqual(observed, plain)

        steps = [r["name"] for r in reports if r["kind"] == "step"]
        self.assertEqual(steps, ["init", "optimize"])
        iterations = [r["iter"] for r in reports if r["kind"] == "iteration"]
        self.assertGreater(len(iterations), 2)
        self.assertEqual(iterations[0], 1)
        self.assertTrue(all(b in (a + 1, 1) for a, b in zip(iterations, iterations[1:])))
        last = [r for r in reports if r["kind"] == "iteration"][-1]
        self.assertEqual(set(last), {"kind", "iter", "max_iters", "cost"})

    def test_raising_from_progress_stops_the_run(self) -> None:
        reports: list[dict[str, Any]] = []

        def stop_at_second_iteration(report: dict[str, Any]) -> None:
            reports.append(report)
            if report["kind"] == "iteration" and report["iter"] == 2:
                raise Stop

        with self.assertRaises(Stop):
            vc.planar_intrinsics.run_calibration(self.dataset, progress=stop_at_second_iteration)
        # The run stopped at the next check: nothing after the raise.
        self.assertEqual(reports[-1]["kind"], "iteration")
        self.assertEqual(reports[-1]["iter"], 2)

    def test_ctrl_c_stops_the_run_with_keyboard_interrupt(self) -> None:
        def interrupt(report: dict[str, Any]) -> None:
            if report["kind"] == "iteration":
                # What Ctrl-C does: a pending SIGINT for the main thread.
                _thread.interrupt_main()

        with self.assertRaises(KeyboardInterrupt):
            vc.planar_intrinsics.run_calibration(self.dataset, progress=interrupt)

    def test_a_session_stopped_in_a_step_keeps_no_result(self) -> None:
        session = vc.PlanarIntrinsicsSession()
        session.set_input(self.dataset)
        steps: list[str] = []

        def stop_in_optimize(report: dict[str, Any]) -> None:
            if report["kind"] == "step":
                steps.append(report["name"])
            if report["kind"] == "iteration" and steps[-1:] == ["optimize"]:
                raise Stop

        session.set_progress(stop_in_optimize)
        with self.assertRaises(Stop):
            session.run_calibration()
        self.assertFalse(session.has_output)

        # Without the callable the same session runs to the end.
        session.set_progress(None)
        session.run_calibration()
        self.assertTrue(session.has_output)
        self.assertEqual(
            session.export(), vc.planar_intrinsics.run_calibration(self.dataset)
        )


if __name__ == "__main__":
    unittest.main()
