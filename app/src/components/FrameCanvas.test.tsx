// @vitest-environment jsdom
import { render } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { EpipolarOverlay } from "../workspaces/EpipolarWorkspace/EpipolarOverlay";
import { FrameCanvas } from "./FrameCanvas";
import type { FrameKey, TargetFeatureResidual } from "../types";

function fakeImage(width: number, height: number): HTMLImageElement {
  const img = new Image();
  Object.defineProperty(img, "naturalWidth", { value: width });
  Object.defineProperty(img, "naturalHeight", { value: height });
  img.src = "data:image/png;base64,AAAA";
  return img;
}

const residual = (feature: number, error: number): TargetFeatureResidual => ({
  pose: 0,
  camera: 0,
  feature,
  target_xyz_m: [0, 0, 0],
  observed_px: [10 + feature, 10],
  projected_px: [11 + feature, 10],
  error_px: error,
});

describe("FrameCanvas", () => {
  it("shows an empty viewer until the image is decoded", () => {
    const frame: FrameKey = { pose: 0, camera: 0, abs_path: "/a.png", label: "a" };
    const { container } = render(
      <FrameCanvas frame={frame} residuals={[]} image={null} />,
    );
    expect(container.querySelector('[role="application"]')).toBeTruthy();
    expect(container.querySelector("[data-stage]")).toBeNull();
  });

  it("sizes the stage to the ROI and batches residuals into one path per colour", () => {
    const frame: FrameKey = {
      pose: 0,
      camera: 0,
      abs_path: "/a.png",
      label: "a",
      roi: { x: 20, y: 10, w: 100, h: 80 },
    };
    const residuals = [residual(0, 0.5), residual(1, 0.4), residual(2, 3)];
    const { container } = render(
      <FrameCanvas frame={frame} residuals={residuals} image={fakeImage(160, 120)}>
        <EpipolarOverlay
          markers={[
            { px: [5, 5], color: "x" },
            { px: [6, 6], color: "x" },
          ]}
          polyline={[
            [0, 0],
            [10, 10],
          ]}
        />
      </FrameCanvas>,
    );
    const stage = container.querySelector<HTMLElement>("[data-stage]")!;
    expect(stage.style.width).toBe("100px");
    expect(stage.style.height).toBe("80px");
    const svgs = stage.querySelectorAll("svg");
    expect(svgs).toHaveLength(2);
    expect(svgs[0]!.getAttribute("viewBox")).toBe("-0.5 -0.5 100 80");
    // 2 error colours + 1 observed-dot path; 3 residuals are not 3 elements.
    expect(svgs[0]!.querySelectorAll("path")).toHaveLength(3);
    // The epipolar layer: 1 batched marker path (+ the polyline element).
    expect(svgs[1]!.querySelectorAll("path")).toHaveLength(1);
    expect(svgs[1]!.querySelectorAll("polyline")).toHaveLength(1);
  });
});
