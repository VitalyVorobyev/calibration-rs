// @vitest-environment jsdom
/** The shared map viewer: its legend (unit, clipping, a named limit), the
 * controls it offers for each kind of map, the reading under the pointer and
 * the words under the map. The stage's own drawing needs a layout engine; the
 * Playwright spec covers it in a real browser. */
import { fireEvent, render, screen, within } from "@testing-library/react";
import { TooltipProvider } from "@vitavision/ui";
import type { ReactElement } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  DISTORTION_CHANNELS,
  GRID,
  RESIDUAL_CHANNELS,
  distortionPlane,
  plane,
  residualPlane,
} from "../../../test/planeFixtures";
import { MapReadout } from "./MapReadout";
import { readoutAt } from "./mapModel";
import { MapViewer, type MapViewerProps } from "./MapViewer";

// jsdom has no canvas; the heatmap layers then report an error and draw nothing.
beforeEach(() => {
  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(null);
});

function renderViewer(ui: ReactElement) {
  return render(<TooltipProvider>{ui}</TooltipProvider>);
}

const distortion: MapViewerProps = {
  source: { plane: distortionPlane(), grid: GRID, channels: DISTORTION_CHANNELS },
  quantity: { channel: "magnitude", label: "|d|", kind: "magnitude" },
  readout: [{ channel: "magnitude", label: "|d|" }],
  arrows: { dx: "dx", dy: "dy" },
  isoLines: true,
  underlay: { url: null, reason: "The run lists no image for this camera." },
  name: "Distortion map of camera 0",
  provenance: "run-001 · camera 0 · lens",
  statement: "The lens displaces a pixel by up to 3.61 px.",
};

describe("MapViewer", () => {
  it("draws a legend in the channel's unit that says the scale clips", () => {
    renderViewer(<MapViewer {...distortion} />);
    // 99.5th percentile of |d| is 3.6, the top of the scale is 5: no clip.
    expect(screen.getByText(/^\|d\|/)).toBeTruthy();
    expect(screen.queryByText(/≥ 5/)).toBeNull();
  });

  it("clips and says so when a fixed top is below the data", () => {
    renderViewer(<MapViewer {...distortion} />);
    fireEvent.click(screen.getByRole("combobox", { name: "Range" }));
    fireEvent.click(screen.getByRole("option", { name: "Fixed top" }));
    const top = screen.getByRole("spinbutton", { name: "Top of the range" });
    expect(top).toBeTruthy();
    fireEvent.change(top, { target: { value: "2" } });
    fireEvent.blur(top);
    expect(screen.getByText(/≥ 2/)).toBeTruthy();
  });

  it("states its provenance and what it says", () => {
    renderViewer(<MapViewer {...distortion}>{<p>A caption.</p>}</MapViewer>);
    expect(screen.getByText("run-001 · camera 0 · lens")).toBeTruthy();
    expect(screen.getByText("The lens displaces a pixel by up to 3.61 px.")).toBeTruthy();
    expect(screen.getByText("A caption.")).toBeTruthy();
  });

  it("explains a disabled image and lists the layers it can draw", () => {
    renderViewer(<MapViewer {...distortion} />);
    const image = screen.getByRole("switch", { name: "Image" }) as HTMLButtonElement;
    expect(image.disabled).toBe(true);
    expect(image.title).toBe("The run lists no image for this camera.");
    expect(screen.getByRole("switch", { name: "Iso-lines" })).toBeTruthy();
    expect(screen.getByRole("switch", { name: "Arrows" })).toBeTruthy();
    expect(screen.queryByRole("switch", { name: "Fade not significant" })).toBeNull();
  });

  it("takes the diverging map for a signed quantity and does not let it be changed", () => {
    renderViewer(
      <MapViewer
        {...distortion}
        quantity={{ channel: "dx", label: "dx", kind: "signed" }}
      />,
    );
    const select = screen.getByRole("combobox", {
      name: "Colour map",
    }) as HTMLButtonElement;
    expect(select.disabled).toBe(true);
    expect(select.textContent).toContain("Blue–orange");
  });

  it("offers the significance fade, and not when no node has a test", () => {
    const residual: MapViewerProps = {
      ...distortion,
      source: { plane: residualPlane(), grid: GRID, channels: RESIDUAL_CHANNELS },
      quantity: { channel: "mean_magnitude", label: "|μ|", kind: "magnitude" },
      mask: { channel: "significance", minimum: 3.03 },
      arrows: { dx: "mean_dx", dy: "mean_dy", on: true },
    };
    const { unmount } = renderViewer(<MapViewer {...residual} />);
    const fade = screen.getByRole("switch", { name: "Fade not significant" });
    expect((fade as HTMLButtonElement).disabled).toBe(false);
    expect(fade.getAttribute("aria-checked")).toBe("true");
    expect(screen.getByText("not distinguishable from zero (p ≥ 0.01)")).toBeTruthy();
    unmount();

    const untested = plane(4, 3, 8, (c, col) => (c === 5 ? Number.NaN : 0.1 * (col + 1)));
    renderViewer(
      <MapViewer
        {...residual}
        source={{ plane: untested, grid: GRID, channels: RESIDUAL_CHANNELS }}
      />,
    );
    const none = screen.getByRole("switch", { name: "Fade not significant" });
    expect((none as HTMLButtonElement).disabled).toBe(true);
    expect(screen.queryByText("not distinguishable from zero (p ≥ 0.01)")).toBeNull();
  });

  it("keeps the tab's own controls in its toolbar", () => {
    renderViewer(<MapViewer {...distortion} controls={<button>Layer</button>} />);
    const toolbar = screen.getByRole("toolbar", { name: "Map display" });
    expect(within(toolbar).getByRole("button", { name: "Layer" })).toBeTruthy();
  });

  it("marks what is blank: no data is not zero", () => {
    renderViewer(<MapViewer {...distortion} />);
    expect(screen.getByRole("list", { name: "Map key" }).textContent).toContain(
      "no data",
    );
  });
});

describe("MapReadout", () => {
  it("lists the node's values with units, `no data` where it has none", () => {
    const reading = readoutAt(residualPlane(), GRID, 35, 5)!;
    render(
      <MapReadout
        reading={reading}
        channels={RESIDUAL_CHANNELS}
        readout={[
          { channel: "mean_magnitude", label: "|μ|" },
          { channel: "significance", label: "χ" },
        ]}
      />,
    );
    const box = screen.getByTestId("map-readout");
    expect(box.textContent).toContain("u, v");
    expect(box.textContent).toContain("35, 5 px");
    // The right-hand column has no support.
    expect(box.textContent).toContain("no data");
  });

  it("reads a value with its unit, the significance in χ", () => {
    const reading = readoutAt(residualPlane(), GRID, 5, 25)!;
    render(
      <MapReadout
        reading={reading}
        channels={RESIDUAL_CHANNELS}
        readout={[
          { channel: "mean_magnitude", label: "|μ|" },
          { channel: "significance", label: "χ" },
        ]}
      />,
    );
    const box = screen.getByTestId("map-readout");
    expect(box.textContent).toContain("0.112 px");
    expect(box.textContent).toContain("5 χ");
  });

  it("asks for the pointer before there is a reading", () => {
    render(<MapReadout reading={null} channels={[]} readout={[]} />);
    expect(screen.getByText("Move over the map to read a value.")).toBeTruthy();
  });
});
