// @vitest-environment jsdom
/** Every schema the Run workspace renders round-trips (lab-ui PLAN L4-1): its
 * default value goes into `SchemaValueForm`, nothing is touched and nothing is
 * reported; each control can then be edited away from its value and back, and
 * the value the form hands out is the default again, key for key.
 *
 * The schemas are `app/src/schemas/<name>.json` and the values
 * `<name>.default.json`, both emitted by `cargo xtask emit-schemas` (the
 * latter is `Config::default()` serialised, so a Rust default the form cannot
 * hold shows up here). `dataset_spec` has no `Default`; a real manifest
 * (`data/kuka_1/dataset.toml`) stands in.
 */
import { SchemaValueForm, type JsonSchema } from "@vitavision/forms";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { useState } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import datasetKuka1 from "./__fixtures__/dataset_kuka_1.json";

const schemaModules = import.meta.glob<{ default: unknown }>(
  "../../schemas/!(*.default).json",
  { eager: true },
);
const defaultModules = import.meta.glob<{ default: unknown }>(
  "../../schemas/*.default.json",
  { eager: true },
);

const nameOf = (path: string) =>
  path.replace(/^.*\//, "").replace(/(\.default)?\.json$/, "");

const CASES: { name: string; schema: JsonSchema; value: unknown }[] = Object.entries(
  schemaModules,
).map(([path, module]) => {
  const name = nameOf(path);
  const defaultModule = Object.entries(defaultModules).find(
    ([p]) => nameOf(p) === name,
  )?.[1];
  return {
    name,
    schema: module.default as JsonSchema,
    value: defaultModule?.default ?? (name === "dataset_spec" ? datasetKuka1 : undefined),
  };
});

afterEach(() => cleanup());

/** A parent that owns the value, the way `RunWorkspace` does, and reports every change. */
function Harness({
  schema,
  initial,
  spy,
}: {
  schema: JsonSchema;
  initial: unknown;
  spy: (next: unknown) => void;
}) {
  const [value, setValue] = useState(initial);
  return (
    <SchemaValueForm
      schema={schema}
      value={value}
      columns={1}
      onValueChange={(next) => {
        spy(next);
        setValue(next);
      }}
    />
  );
}

function setup(schema: JsonSchema, initial: unknown) {
  const spy = vi.fn<(next: unknown) => void>();
  const view = render(<Harness schema={schema} initial={initial} spy={spy} />);
  return { spy, ...view };
}

/** The same JSON, key for key and in order — "serialises back identical". */
const same = (a: unknown, b: unknown) =>
  expect(JSON.stringify(a)).toBe(JSON.stringify(b));

/** `null` members removed. Clearing a text field, or switching a nullable block off,
 * where the value had no such key (serde skips a `None` `skip_serializing_if`) writes
 * `null` rather than removing the key: the same `None` to serde, a different key in the JSON. */
function dropNulls(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(dropNulls);
  if (typeof value !== "object" || value === null) return value;
  return Object.fromEntries(
    Object.entries(value)
      .filter(([, member]) => member !== null)
      .map(([key, member]) => [key, dropNulls(member)]),
  );
}

it("covers every schema in app/src/schemas", () => {
  expect(CASES.map((c) => c.name)).toEqual([
    "dataset_spec",
    "laserline_device_config",
    "planar_intrinsics_config",
    "rig_extrinsics_config",
    "rig_handeye_config",
    "rig_handeye_laserline_config",
    "rig_laserline_device_config",
    "scheimpflug_intrinsics_config",
    "single_cam_handeye_config",
  ]);
  for (const c of CASES) expect(c.value, `${c.name} has a default value`).toBeDefined();
});

describe.each(CASES)("$name", ({ schema, value }) => {
  it("renders the default without reporting a change", () => {
    const { spy, container } = setup(schema, value);
    expect(container.querySelector("input, button, textarea")).not.toBeNull();
    expect(spy).not.toHaveBeenCalled();
  });

  it("reports nothing when a control is focused and left", () => {
    const { spy, container } = setup(schema, value);
    const controls = container.querySelectorAll<HTMLElement>("input, textarea, button");
    for (const control of controls) {
      fireEvent.focus(control);
      fireEvent.blur(control);
    }
    expect(spy).not.toHaveBeenCalled();
  });

  it("reports nothing when the chosen segment of a strip is chosen again", () => {
    const { spy } = setup(schema, value);
    for (const radio of screen.queryAllByRole<HTMLInputElement>("radio")) {
      if (radio.checked) fireEvent.click(radio);
    }
    expect(spy).not.toHaveBeenCalled();
  });

  it("reports nothing when a number is retyped as itself", () => {
    const { spy } = setup(schema, value);
    for (const input of screen.queryAllByRole<HTMLInputElement>("spinbutton")) {
      fireEvent.focus(input);
      fireEvent.change(input, { target: { value: input.value } });
      fireEvent.blur(input);
    }
    expect(spy).not.toHaveBeenCalled();
  });

  it("hands the default back after every number, text and boolean is edited away and back", () => {
    const { spy, container } = setup(schema, value);
    const controlCount = () =>
      container.querySelectorAll("input, textarea, button").length;
    let last = value;
    spy.mockImplementation((next) => {
      last = next;
    });

    for (const input of screen.queryAllByRole<HTMLInputElement>("spinbutton")) {
      const original = input.value;
      if (original === "") continue; // an unset nullable number: set below
      const edited = String(Number(original) + 1);
      fireEvent.focus(input);
      fireEvent.change(input, { target: { value: edited } });
      fireEvent.change(input, { target: { value: original } });
      fireEvent.blur(input);
      same(last, value);
    }

    for (const input of screen.queryAllByRole<HTMLInputElement>("textbox")) {
      if (input.tagName !== "INPUT") continue;
      const original = input.value;
      fireEvent.change(input, { target: { value: `${original}x` } });
      fireEvent.change(input, { target: { value: original } });
      same(dropNulls(last), dropNulls(value));
    }

    // A list the form cannot lay out is a JSON textarea: the same value written
    // differently is an edit that reports the same value.
    for (const area of container.querySelectorAll("textarea")) {
      fireEvent.change(area, {
        target: { value: JSON.stringify(JSON.parse(area.value)) },
      });
      same(dropNulls(last), dropNulls(value));
    }

    for (const toggle of screen.queryAllByRole("switch")) {
      const before = controlCount();
      const valueBefore = last;
      fireEvent.click(toggle);
      const isBlock = controlCount() !== before;
      fireEvent.click(toggle);
      // A boolean comes back as it was. A set nullable block (a switch that shows
      // and hides fields) is reseeded from the schema's defaults when switched back
      // on, by design: it only has to be a block again.
      if (isBlock) expect(controlCount()).toBe(before);
      else same(dropNulls(last), dropNulls(valueBefore));
    }

    // The loops above would pass vacuously if nothing were edited.
    expect(spy.mock.calls.length).toBeGreaterThan(0);
  });
});
