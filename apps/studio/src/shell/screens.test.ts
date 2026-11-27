import { describe, expect, it } from "vitest";
import {
  isEnabled,
  parsePlace,
  placePath,
  screenById,
  screenForShortcut,
  screenPlace,
  SCREENS,
} from "./screens";

const key = (
  k: string,
  mods: Partial<Record<"meta" | "ctrl" | "alt" | "shift", boolean>> = {},
) => ({
  key: k,
  metaKey: mods.meta ?? false,
  ctrlKey: mods.ctrl ?? false,
  altKey: mods.alt ?? false,
  shiftKey: mods.shift ?? false,
});

describe("the screens", () => {
  it("are the seven of the workflow, in order", () => {
    expect(SCREENS.map((s) => s.id)).toEqual([
      "library",
      "dataset",
      "detect",
      "calibrate",
      "inspect",
      "compare",
      "device",
    ]);
  });

  it("are enabled when built, and with a project when they need one", () => {
    expect(isEnabled(screenById("library"), false)).toBe(true);
    expect(isEnabled(screenById("inspect"), false)).toBe(false);
    expect(isEnabled(screenById("inspect"), true)).toBe(true);
    expect(isEnabled(screenById("dataset"), true)).toBe(true);
    expect(isEnabled(screenById("dataset"), false)).toBe(false);
    expect(isEnabled(screenById("detect"), true)).toBe(false);
    expect(isEnabled(screenById("device"), false)).toBe(false);
  });

  it("are chosen by ⌘ or Ctrl and the digit of their position", () => {
    expect(screenForShortcut(key("1", { meta: true }))).toBe("library");
    expect(screenForShortcut(key("5", { ctrl: true }))).toBe("inspect");
    expect(screenForShortcut(key("7", { meta: true }))).toBe("device");
  });

  it("ignore bare digits, other modifiers and digits past the rail", () => {
    expect(screenForShortcut(key("1"))).toBeNull();
    expect(screenForShortcut(key("1", { meta: true, shift: true }))).toBeNull();
    expect(screenForShortcut(key("1", { meta: true, alt: true }))).toBeNull();
    expect(screenForShortcut(key("8", { meta: true }))).toBeNull();
    expect(screenForShortcut(key("0", { meta: true }))).toBeNull();
    expect(screenForShortcut(key("s", { meta: true }))).toBeNull();
  });
});

describe("a place in the app", () => {
  it("round-trips through its address, project and run included", () => {
    const place = {
      screen: "inspect" as const,
      tab: "epipolar" as const,
      project: "/data/my project/rig",
      run: "run-007",
    };
    const path = placePath(place);
    expect(path).toBe("/inspect/epipolar?project=%2Fdata%2Fmy+project%2Frig&run=run-007");
    const [pathname, search] = path.split("?");
    expect(parsePlace(pathname!, `?${search}`)).toEqual(place);
  });

  it("gives Inspect its default tab and other screens none", () => {
    expect(parsePlace("/inspect", "")).toMatchObject({
      screen: "inspect",
      tab: "frames",
    });
    expect(parsePlace("/inspect/nonsense", "").tab).toBe("frames");
    expect(parsePlace("/calibrate", "").tab).toBeNull();
    expect(placePath(parsePlace("/inspect", ""))).toBe("/inspect/frames");
  });

  it("falls back to the Library for an address of no screen", () => {
    expect(parsePlace("/", "").screen).toBe("library");
    expect(parsePlace("/viewer3d", "").screen).toBe("library");
  });

  it("drops a run that names no project", () => {
    expect(
      placePath({ screen: "calibrate", tab: null, project: null, run: "run-001" }),
    ).toBe("/calibrate");
  });

  it("keeps the project and run when the screen changes", () => {
    const from = parsePlace("/inspect/depth", "?project=%2Fp&run=run-002");
    expect(placePath(screenPlace("calibrate", from))).toBe(
      "/calibrate?project=%2Fp&run=run-002",
    );
    expect(placePath(screenPlace("inspect", from))).toBe(
      "/inspect/depth?project=%2Fp&run=run-002",
    );
  });
});
