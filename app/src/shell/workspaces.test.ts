import { describe, expect, it } from "vitest";
import { availableWorkspaces, workspaceForShortcut } from "./workspaces";

const key = (
  k: string,
  mods: Partial<Record<"meta" | "ctrl" | "alt" | "shift", boolean>>,
) => ({
  key: k,
  metaKey: mods.meta ?? false,
  ctrlKey: mods.ctrl ?? false,
  altKey: mods.alt ?? false,
  shiftKey: mods.shift ?? false,
});

describe("workspaceForShortcut", () => {
  it("maps ⌘ or Ctrl with a rail digit to that workspace", () => {
    expect(workspaceForShortcut(key("1", { meta: true }))).toBe("/diagnose");
    expect(workspaceForShortcut(key("5", { ctrl: true }))).toBe("/run");
  });

  it("ignores bare digits, extra modifiers and digits past the rail", () => {
    expect(workspaceForShortcut(key("1", {}))).toBeNull();
    expect(workspaceForShortcut(key("1", { meta: true, shift: true }))).toBeNull();
    expect(workspaceForShortcut(key("1", { meta: true, alt: true }))).toBeNull();
    expect(workspaceForShortcut(key("6", { meta: true }))).toBeNull();
    expect(workspaceForShortcut(key("0", { meta: true }))).toBeNull();
    expect(workspaceForShortcut(key("s", { meta: true }))).toBeNull();
  });
});

describe("availableWorkspaces", () => {
  it("leaves the desktop-only Run workspace out of the browser viewer", () => {
    expect(availableWorkspaces(true).map((w) => w.to)).toContain("/run");
    const viewer = availableWorkspaces(false);
    expect(viewer.map((w) => w.to)).toEqual([
      "/diagnose",
      "/viewer3d",
      "/epipolar",
      "/depth",
    ]);
    expect(workspaceForShortcut(key("5", { meta: true }), viewer)).toBeNull();
  });
});
