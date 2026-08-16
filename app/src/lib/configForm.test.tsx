// @vitest-environment jsdom
/** Component tests for the schema-driven `ConfigForm`.
 *
 * Renders against the real generated `PlanarIntrinsicsConfig` schema
 * fixture (not a hand-rolled toy schema) so the tests exercise the same
 * `$ref` resolution, nested objects, and both `oneOf` union shapes
 * (`RobustLoss`'s externally-tagged enum, `DistortionKind`'s plain
 * string enum) that `RunWorkspace` actually renders.
 */
import { useState } from "react";
import { cleanup, fireEvent, render, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ConfigForm, type JsonSchema } from "./configForm";
import planarConfigSchemaJson from "../schemas/planar_intrinsics_config.json";

const planarConfigSchema = planarConfigSchemaJson as unknown as JsonSchema;

// Mirrors RunWorkspace's DEFAULT_PLANAR_CONFIG (ADR 0024 grouped shape).
const DEFAULT_PLANAR_CONFIG = {
  init: {
    init_iterations: 2,
    fix_k3: true,
    fix_tangential: false,
    zero_skew: true,
  },
  solver: {
    max_iters: 50,
    verbosity: 0,
    robust_loss: "None",
  },
  distortion_model: "brown_conrady5",
  fix_camera: {
    intrinsics: { fx: false, fy: false, cx: false, cy: false },
    distortion: { k1: false, k2: false, k3: true, p1: false, p2: false },
  },
  fix_poses: [],
};

afterEach(() => cleanup());

/** Finds the `<input>`/`<select>` sibling of a field's label text. The
 * form doesn't wire `<label htmlFor>` (ADR 0018 keeps the widget ~250
 * LoC), so fields are located by their label text's parent `<div>`
 * rather than `getByLabelText`. */
function fieldControl(root: HTMLElement, label: string): HTMLElement {
  const labelEl = within(root).getByText(label, { selector: "label" });
  const control = labelEl.parentElement?.querySelector("input, select");
  if (!control) throw new Error(`no input/select sibling for label "${label}"`);
  return control as HTMLElement;
}

/** Scopes a query to the `<fieldset>` whose `<legend>` matches `label`
 * (i.e. a nested object field, e.g. "solver" or "fix_camera"). */
function fieldsetFor(root: HTMLElement, label: string): HTMLElement {
  const legend = within(root).getByText(label, { selector: "legend" });
  const fieldset = legend.closest("fieldset");
  if (!fieldset) throw new Error(`no fieldset for legend "${label}"`);
  return fieldset;
}

describe("ConfigForm", () => {
  it("renders top-level fields from the schema fixture", () => {
    const { container } = render(
      <ConfigForm
        schema={planarConfigSchema}
        value={DEFAULT_PLANAR_CONFIG}
        onChange={() => {}}
        rootLabel="config"
      />,
    );
    // Leaf/union fields render a `<label>` (from the parent ObjectField)
    // plus a `<legend>` when their own widget is itself a fieldset (a
    // nested object, or a `oneOf` union like `distortion_model`) — so
    // "distortion_model" legitimately matches twice.
    expect(
      within(container).getByText("distortion_model", { selector: "label" }),
    ).toBeTruthy();
    expect(within(container).getByText("fix_camera", { selector: "label" })).toBeTruthy();
    expect(within(container).getByText("solver", { selector: "label" })).toBeTruthy();
    expect(within(container).getByText("init", { selector: "label" })).toBeTruthy();
  });

  it("propagates a number field edit into the produced config object", () => {
    const onChange = vi.fn();
    const { container } = render(
      <ConfigForm
        schema={planarConfigSchema}
        value={DEFAULT_PLANAR_CONFIG}
        onChange={onChange}
        rootLabel="config"
      />,
    );
    const solverFieldset = fieldsetFor(container, "solver");
    const maxIters = fieldControl(solverFieldset, "max_iters");
    fireEvent.change(maxIters, { target: { value: "77" } });

    expect(onChange).toHaveBeenCalledTimes(1);
    const next = onChange.mock.calls[0][0] as typeof DEFAULT_PLANAR_CONFIG;
    expect(next.solver.max_iters).toBe(77);
    // Sibling fields are untouched by the edit.
    expect(next.solver.verbosity).toBe(0);
    expect(next.distortion_model).toBe("brown_conrady5");
  });

  it("propagates a nested boolean checkbox toggle two levels deep", () => {
    const onChange = vi.fn();
    const { container } = render(
      <ConfigForm
        schema={planarConfigSchema}
        value={DEFAULT_PLANAR_CONFIG}
        onChange={onChange}
        rootLabel="config"
      />,
    );
    const fixCamera = fieldsetFor(container, "fix_camera");
    const distortionMask = fieldsetFor(fixCamera, "distortion");
    const k3 = fieldControl(distortionMask, "k3") as HTMLInputElement;
    expect(k3.checked).toBe(true); // fix_k3 defaults to true

    fireEvent.click(k3);

    expect(onChange).toHaveBeenCalledTimes(1);
    const next = onChange.mock.calls[0][0] as typeof DEFAULT_PLANAR_CONFIG;
    expect(next.fix_camera.distortion.k3).toBe(false);
    expect(next.fix_camera.distortion.k1).toBe(false); // untouched sibling
    expect(next.fix_camera.intrinsics).toEqual(
      DEFAULT_PLANAR_CONFIG.fix_camera.intrinsics,
    );
  });

  it("propagates a plain-string enum change", () => {
    const onChange = vi.fn();
    const { container } = render(
      <ConfigForm
        schema={planarConfigSchema}
        value={DEFAULT_PLANAR_CONFIG}
        onChange={onChange}
        rootLabel="config"
      />,
    );
    const distortionModel = fieldControl(
      container,
      "distortion_model",
    ) as HTMLSelectElement;
    fireEvent.change(distortionModel, { target: { value: "rational8" } });

    expect(onChange).toHaveBeenCalledTimes(1);
    const next = onChange.mock.calls[0][0] as typeof DEFAULT_PLANAR_CONFIG;
    expect(next.distortion_model).toBe("rational8");
  });

  it("propagates an externally-tagged oneOf variant switch (RobustLoss)", () => {
    const onChange = vi.fn();
    const { container } = render(
      <ConfigForm
        schema={planarConfigSchema}
        value={DEFAULT_PLANAR_CONFIG}
        onChange={onChange}
        rootLabel="config"
      />,
    );
    const solverFieldset = fieldsetFor(container, "solver");
    const robustLoss = fieldControl(solverFieldset, "robust_loss") as HTMLSelectElement;
    expect(robustLoss.value).toBe("None");

    fireEvent.change(robustLoss, { target: { value: "Huber" } });

    expect(onChange).toHaveBeenCalledTimes(1);
    const next = onChange.mock.calls[0][0] as {
      solver: { robust_loss: { Huber?: { scale: number } } };
    };
    expect(Object.keys(next.solver.robust_loss)).toEqual(["Huber"]);
    expect(typeof next.solver.robust_loss.Huber?.scale).toBe("number");
  });

  it("round-trips through a controlled harness: edits are visible in the DOM", () => {
    function Harness() {
      const [value, setValue] = useState<unknown>(DEFAULT_PLANAR_CONFIG);
      return <ConfigForm schema={planarConfigSchema} value={value} onChange={setValue} />;
    }
    const { container } = render(<Harness />);
    const solverFieldset = fieldsetFor(container, "solver");
    const maxIters = fieldControl(solverFieldset, "max_iters") as HTMLInputElement;
    expect(maxIters.value).toBe("50");

    fireEvent.change(maxIters, { target: { value: "120" } });

    expect(maxIters.value).toBe("120");
  });
});
