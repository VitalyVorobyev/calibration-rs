// @vitest-environment jsdom
/** Component tests for the Run workspace's config forms (`SchemaValueForm`
 * from `@vitavision/forms`, ADR 0018).
 *
 * Renders against the real generated `PlanarIntrinsicsConfig` schema (not a
 * hand-rolled toy schema) so the tests exercise the same `$ref` resolution,
 * nested objects, and both enum shapes (`RobustLoss`'s externally-tagged
 * enum, `DistortionKind`'s plain string enum) that `RunWorkspace` renders —
 * and the same props: the whole value in, the whole next value out,
 * `columns={1}` for the narrow rail.
 *
 * These replace the tests of the app's former `configForm.tsx`; the controls
 * are now found by role and accessible name rather than by raw-key `<label>`
 * text.
 */
import { SchemaValueForm, type JsonSchema } from "@vitavision/forms";
import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { useState } from "react";
import { afterEach, beforeAll, describe, expect, it, vi } from "vitest";
import planarDefault from "../../schemas/planar_intrinsics_config.default.json";
import planarConfigSchemaJson from "../../schemas/planar_intrinsics_config.json";

const planarConfigSchema = planarConfigSchemaJson as unknown as JsonSchema;

// The Rust `PlanarIntrinsicsConfig::default()` (ADR 0024 grouped shape), as
// emitted by `cargo xtask emit-schemas`.
const DEFAULT_PLANAR_CONFIG = planarDefault;

beforeAll(() => {
  // Radix's Select (ui's `Select`) calls these on open; jsdom has no layout
  // or pointer-capture implementation.
  const proto = Element.prototype as unknown as Record<string, unknown>;
  proto["hasPointerCapture"] ??= () => false;
  proto["setPointerCapture"] ??= () => {};
  proto["releasePointerCapture"] ??= () => {};
  proto["scrollIntoView"] ??= () => {};
});

afterEach(() => cleanup());

function renderForm(onValueChange: (next: unknown) => void) {
  return render(
    <SchemaValueForm
      schema={planarConfigSchema}
      value={DEFAULT_PLANAR_CONFIG}
      onValueChange={onValueChange}
      columns={1}
    />,
  );
}

/** Opens a Radix `Select` and picks the option labelled `option`. */
function choose(trigger: HTMLElement, option: string) {
  fireEvent.pointerDown(trigger, { button: 0, ctrlKey: false, pointerType: "mouse" });
  fireEvent.click(screen.getByRole("option", { name: option }));
}

describe("config form (SchemaValueForm)", () => {
  it("renders top-level fields from the schema", () => {
    renderForm(() => {});
    // Nested objects are labelled groups; the plain enum is a labelled picker.
    expect(screen.getByRole("group", { name: "Init" })).toBeTruthy();
    expect(screen.getByRole("group", { name: "Solver" })).toBeTruthy();
    expect(screen.getByRole("group", { name: "Fix camera" })).toBeTruthy();
    expect(screen.getByRole("combobox", { name: "Distortion model" }).textContent).toBe(
      "brown_conrady5",
    );
  });

  it("propagates a number field edit into the produced config object", () => {
    const onValueChange = vi.fn();
    renderForm(onValueChange);
    const solver = screen.getByRole("group", { name: "Solver" });
    const maxIters = within(solver).getByRole("spinbutton", { name: "Max iters" });
    fireEvent.focus(maxIters);
    fireEvent.change(maxIters, { target: { value: "77" } });

    expect(onValueChange).toHaveBeenCalledTimes(1);
    const next = onValueChange.mock.calls[0]![0] as typeof DEFAULT_PLANAR_CONFIG;
    expect(next.solver.max_iters).toBe(77);
    // Sibling fields are untouched by the edit.
    expect(next.solver.verbosity).toBe(0);
    expect(next.distortion_model).toBe("brown_conrady5");
  });

  it("propagates a nested boolean toggle two levels deep", () => {
    const onValueChange = vi.fn();
    renderForm(onValueChange);
    const fixCamera = screen.getByRole("group", { name: "Fix camera" });
    const distortionMask = within(fixCamera).getByRole("group", { name: "Distortion" });
    const k3 = within(distortionMask).getByRole("switch", { name: /K3/ });
    expect(k3.getAttribute("aria-checked")).toBe("true"); // fix_k3 defaults to true

    fireEvent.click(k3);

    expect(onValueChange).toHaveBeenCalledTimes(1);
    const next = onValueChange.mock.calls[0]![0] as typeof DEFAULT_PLANAR_CONFIG;
    expect(next.fix_camera.distortion.k3).toBe(false);
    expect(next.fix_camera.distortion.k1).toBe(false); // untouched sibling
    expect(next.fix_camera.intrinsics).toEqual(
      DEFAULT_PLANAR_CONFIG.fix_camera.intrinsics,
    );
  });

  it("propagates a plain-string enum change", () => {
    const onValueChange = vi.fn();
    renderForm(onValueChange);
    choose(screen.getByRole("combobox", { name: "Distortion model" }), "rational8");

    expect(onValueChange).toHaveBeenCalledTimes(1);
    const next = onValueChange.mock.calls[0]![0] as typeof DEFAULT_PLANAR_CONFIG;
    expect(next.distortion_model).toBe("rational8");
  });

  it("propagates an externally-tagged enum variant switch (RobustLoss)", () => {
    const onValueChange = vi.fn();
    renderForm(onValueChange);
    const solver = screen.getByRole("group", { name: "Solver" });
    const robustLoss = within(solver).getByRole("combobox", { name: "Robust loss" });
    expect(robustLoss.textContent).toBe("None");

    choose(robustLoss, "Huber");

    expect(onValueChange).toHaveBeenCalledTimes(1);
    const next = onValueChange.mock.calls[0]![0] as {
      solver: { robust_loss: { Huber?: { scale: number } } };
    };
    expect(Object.keys(next.solver.robust_loss)).toEqual(["Huber"]);
    expect(typeof next.solver.robust_loss.Huber?.scale).toBe("number");
  });

  it("round-trips through a controlled harness: edits are visible in the DOM", () => {
    function Harness() {
      const [value, setValue] = useState<unknown>(DEFAULT_PLANAR_CONFIG);
      return (
        <SchemaValueForm
          schema={planarConfigSchema}
          value={value}
          onValueChange={setValue}
          columns={1}
        />
      );
    }
    render(<Harness />);
    const maxIters = screen.getByRole<HTMLInputElement>("spinbutton", {
      name: "Max iters",
    });
    expect(maxIters.value).toBe("50");

    fireEvent.focus(maxIters);
    fireEvent.change(maxIters, { target: { value: "120" } });

    expect(maxIters.value).toBe("120");
  });
});
