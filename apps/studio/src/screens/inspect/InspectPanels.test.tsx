// @vitest-environment jsdom
/** The parameters and quality tabs of Inspect over a mock back end: what each
 * reads and says, and the words for the states before there is a run. */
import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { StudioError } from "../../api";
import { useProject } from "../../project/projectStore";
import {
  PARAMETERS,
  backend,
  mount,
  show,
  installInspectHooks,
} from "../../test/inspectKit";
import { calibratedRun, projectView } from "../../test/projectFixtures";
import { DistortionTab } from "./maps/DistortionTab";
import { ParametersTab } from "./ParametersTab";
import { QualityTab } from "./QualityTab";

installInspectHooks();

describe("with no run shown", () => {
  it("every tab says what to do", () => {
    backend();
    for (const tab of [
      <ParametersTab key="p" />,
      <QualityTab key="q" />,
      <DistortionTab key="d" />,
    ]) {
      const { unmount } = mount(tab);
      expect(screen.getByText(/No run is shown/)).toBeTruthy();
      unmount();
    }
  });
});

describe("while a run opens", () => {
  it("waits, rather than saying there is none", () => {
    backend();
    useProject.setState({
      project: projectView("/work/cell", [calibratedRun("/work/cell", "run-001")]),
    });
    const { container } = mount(<ParametersTab />);
    expect(container.querySelector('[aria-busy="true"]')).toBeTruthy();
    expect(screen.queryByText(/No run is shown/)).toBeNull();
    useProject.setState({ project: null });
  });
});

describe("Parameters", () => {
  it("lists every parameter in groups: value ± σ, unit, the reason when there is no σ", async () => {
    backend();
    show();
    mount(<ParametersTab />);
    expect(await screen.findByTestId("model-summary")).toHaveProperty(
      "textContent",
      "pinhole · Brown–Conrady (5)",
    );
    expect(screen.getByTestId("value-fx").textContent).toBe("1412.37 ± 0.21");
    expect(screen.getByRole("region", { name: "Intrinsics" })).toBeTruthy();
    const distortion = screen.getByRole("region", { name: "Distortion" });
    expect(within(distortion).getByTestId("value-k3").title).toBe(
      "held fixed, not estimated",
    );
    expect(within(distortion).getByText("held fixed, not estimated")).toBeTruthy();
    expect(within(distortion).getByText("on a bound")).toBeTruthy();
    expect(within(distortion).getByText(/ended on a bound: k2/)).toBeTruthy();
    // The run's export carries no uncertainty at all, and says so.
    expect(screen.getByText("This run was written without uncertainty.")).toBeTruthy();
  });

  it("lists the hand-eye and the camera's laser plane with their values", async () => {
    const unlisted = {
      kind: "unavailable" as const,
      reason: { kind: "not_listed" as const },
    };
    backend({
      camera_parameter_values: () => ({
        ...PARAMETERS,
        poses: [
          {
            label: "gripper_se3_cam",
            camera: null,
            translation_m: [0.01, 0.02, 0.3],
            rotation_deg: [0, 0, 90],
            sigma: Array.from({ length: 6 }, () => unlisted),
          },
        ],
        planes: [
          {
            label: "plane",
            camera: 0,
            normal: [0, 0, 1],
            distance_m: 0.25,
            sigma: Array.from({ length: 4 }, () => unlisted),
          },
        ],
      }),
    });
    show();
    mount(<ParametersTab />);
    expect((await screen.findByTestId("pose-gripper_se3_cam")).textContent).toBe(
      "10.000, 20.000, 300.000",
    );
    const plane = screen.getByRole("region", { name: "Laser plane" });
    expect(within(plane).getByText("250 ·")).toBeTruthy();
  });

  it("copies the camera's parameters as JSON", async () => {
    backend();
    show();
    const writeText = vi.fn(() => Promise.resolve());
    vi.stubGlobal("navigator", { clipboard: { writeText } });
    mount(<ParametersTab />);
    fireEvent.click(await screen.findByRole("button", { name: "Copy as JSON" }));
    await waitFor(() => expect(writeText).toHaveBeenCalledTimes(1));
    const json = JSON.parse((writeText.mock.calls[0] as unknown as [string])[0]) as {
      parameters: { name: string; sigma: number | null }[];
    };
    expect(json.parameters.map((p) => p.name)).toEqual(["fx", "k3", "k2"]);
    expect(await screen.findByRole("button", { name: "Copied" })).toBeTruthy();
  });

  it("says why when the command fails, and tries again", async () => {
    let fail = true;
    backend({
      camera_parameter_values: () => {
        if (fail) throw new StudioError({ kind: "io", message: "no export" });
        return PARAMETERS;
      },
    });
    show();
    mount(<ParametersTab />);
    expect(
      await screen.findByText(/Could not read the parameters: no export/),
    ).toBeTruthy();
    fail = false;
    fireEvent.click(screen.getByRole("button", { name: "Retry" }));
    expect(await screen.findByTestId("value-fx")).toBeTruthy();
  });
});

describe("Quality", () => {
  it("gives the verdict in words and each check's measured value and limit", async () => {
    backend();
    show();
    mount(<QualityTab />);
    const verdict = await screen.findByTestId("verdict");
    expect(verdict.textContent).toContain("Warning");
    const sigma = screen.getByTestId("check-projection_sigma");
    expect(sigma.textContent).toContain("Projection is precise");
    expect(sigma.textContent).toContain("measured 1.4 px, limit 1 px");
    // Another camera's check is not listed for this one.
    expect(screen.queryByTestId("check-feature_coverage")).toBeNull();
    expect(screen.getByTestId("check-convergence").textContent).toContain("Pass");
  });

  it("says an imported run has no report", async () => {
    backend({
      run_quality: () => {
        throw new StudioError({ kind: "io", message: "not found" });
      },
    });
    show();
    mount(<QualityTab />);
    expect(
      await screen.findByText(
        /An imported export whose quality could not be assessed|imported export whose quality/,
      ),
    ).toBeTruthy();
  });
});
