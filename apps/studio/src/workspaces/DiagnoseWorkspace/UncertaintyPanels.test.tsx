// @vitest-environment jsdom
/** The uncertainty panels against a mocked backend: values beside σ, the
 * projection summary, and the two states without a covariance. */
import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { setTransport } from "../../api";
import { createMockTransport } from "../../api/transport/mock";
import { useStore } from "../../store";
import { UncertaintyPanels } from "./UncertaintyPanels";
import type { ProjectionUncertainty, Uncertainty } from "../../api/generated/types";

const AVAILABLE: Uncertainty = {
  status: "available",
  labels: ["fx", "cx"],
  covariance: [
    [4, 0.6],
    [0.6, 0.25],
  ],
  fixed: ["k3"],
  noise: [{ kind: "reprojection", sigma: 0.12, count: 400 }],
  num_residuals: 400,
  dof: 380,
};

const MAP: ProjectionUncertainty = {
  image_size: [640, 480],
  columns: 2,
  rows: 2,
  sigma: [1.5, 0.9, null, 2.5],
  center: 0.5,
  corner: 2.5,
  max: 2.5,
};

beforeEach(() => {
  // The run the panels' commands name.
  useStore.setState({ run: { dir: "/p", id: "run-001" } });
  setTransport(
    createMockTransport({
      commands: {
        camera_parameter_values: () => [812.25, 319.5],
        projection_uncertainty: () => MAP,
      },
    }),
  );
});

afterEach(() => {
  cleanup();
  setTransport(null);
  useStore.setState({ run: null });
});

describe("UncertaintyPanels", () => {
  it("shows values ± σ, the fixed parameters and the projection summary", async () => {
    render(
      <UncertaintyPanels uncertainty={AVAILABLE} camera={0} imageSize={[640, 480]} />,
    );
    expect(await screen.findByText("812.25")).toBeTruthy();
    expect(screen.getByText("319.5")).toBeTruthy();
    expect(screen.getByText("k3")).toBeTruthy();
    expect(await screen.findByText(/centre 0\.5 · corner 2\.5 · max 2\.5/)).toBeTruthy();
    expect(screen.getByLabelText("Parameter correlation")).toBeTruthy();
  });

  it("explains a covariance the data cannot give", () => {
    render(
      <UncertaintyPanels
        uncertainty={{
          status: "unavailable",
          reason: { kind: "rank_deficient", unobservable: ["tilt_y", "cx"] },
        }}
        camera={0}
        imageSize={null}
      />,
    );
    expect(screen.getByText("The data does not determine: tilt_y, cx.")).toBeTruthy();
  });

  it("says when the export has no uncertainty", () => {
    render(<UncertaintyPanels uncertainty={null} camera={0} imageSize={null} />);
    expect(screen.getByText("This export was written without uncertainty.")).toBeTruthy();
  });
});
