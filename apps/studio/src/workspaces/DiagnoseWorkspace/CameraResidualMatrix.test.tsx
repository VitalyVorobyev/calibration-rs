// @vitest-environment jsdom
/** Component test for the cross-camera residual matrix: renders a
 * cameras × poses grid of the back end's per-(camera, pose) statistics
 * (`residual_stats`' `cells`), jumps to (pose, camera) on cell click, and
 * renders nothing for single-camera exports. */
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { CellStats, ResidualStats } from "../../api/generated/types";
import { CameraResidualMatrix } from "./CameraResidualMatrix";

function cell(pose: number, camera: number, mean: number): CellStats {
  return {
    view: pose,
    camera,
    stats: { count: 1, mean, rms: mean, max: mean },
    unprojected: 0,
  };
}

function stats(cells: CellStats[]): ResidualStats {
  return { cameras: [], views: [], cells };
}

// Two cameras, two poses.
const RESIDUALS = stats([cell(0, 0, 1), cell(1, 0, 4), cell(0, 1, 2), cell(1, 1, 8)]);

afterEach(cleanup);

describe("CameraResidualMatrix", () => {
  it("renders a cell per observed (camera, pose) slot with the mean px", () => {
    const { container } = render(
      <CameraResidualMatrix
        stats={RESIDUALS}
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
        stats={RESIDUALS}
        selectedPose={0}
        selectedCamera={0}
        onSelect={onSelect}
      />,
    );
    fireEvent.click(screen.getByRole("button", { name: /cam 1 · pose 0/ }));
    expect(onSelect).toHaveBeenCalledWith(0, 1);
  });

  it("renders nothing for a single-camera export", () => {
    const single = stats([cell(0, 0, 1), cell(1, 0, 2)]);
    const { container } = render(
      <CameraResidualMatrix
        stats={single}
        selectedPose={0}
        selectedCamera={0}
        onSelect={() => {}}
      />,
    );
    expect(container.querySelector("table")).toBeNull();
  });
});
