// @vitest-environment jsdom
/** Component test for the cross-camera residual matrix (B-UX2): renders a
 * cameras × poses grid for multi-camera exports, jumps to (pose, camera)
 * on cell click, and renders nothing for single-camera exports. The
 * aggregation is covered by `lib/residualStats.test.ts`. */
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { TargetFeatureResidual } from "../../types";
import { CameraResidualMatrix } from "./CameraResidualMatrix";

function res(pose: number, camera: number, error_px: number): TargetFeatureResidual {
  return {
    pose,
    camera,
    feature: 0,
    target_xyz_m: [0, 0, 0],
    observed_px: [0, 0],
    projected_px: [0, 0],
    error_px,
  };
}

// Two cameras, two poses.
const RESIDUALS = [res(0, 0, 1), res(1, 0, 4), res(0, 1, 2), res(1, 1, 8)];

afterEach(cleanup);

describe("CameraResidualMatrix", () => {
  it("renders a cell per observed (camera, pose) slot with the mean px", () => {
    const { container } = render(
      <CameraResidualMatrix
        residuals={RESIDUALS}
        selectedPose={0}
        selectedCamera={0}
        onSelect={() => {}}
      />,
    );
    expect(container.querySelector("table")).toBeTruthy();
    // cam1 · pose1 mean is 8.0 → the button label carries the exact number.
    expect(
      screen.getByRole("button", { name: /cam 1 · pose 1: mean 8\.000 px/ }),
    ).toBeTruthy();
  });

  it("jumps to the clicked (pose, camera)", () => {
    const onSelect = vi.fn();
    render(
      <CameraResidualMatrix
        residuals={RESIDUALS}
        selectedPose={0}
        selectedCamera={0}
        onSelect={onSelect}
      />,
    );
    fireEvent.click(screen.getByRole("button", { name: /cam 1 · pose 0/ }));
    expect(onSelect).toHaveBeenCalledWith(0, 1);
  });

  it("renders nothing for a single-camera export", () => {
    const single = [res(0, 0, 1), res(1, 0, 2)];
    const { container } = render(
      <CameraResidualMatrix
        residuals={single}
        selectedPose={0}
        selectedCamera={0}
        onSelect={() => {}}
      />,
    );
    expect(container.querySelector("table")).toBeNull();
  });
});
