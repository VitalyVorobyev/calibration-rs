import { describe, expect, it } from "vitest";
import { keyAction, type KeyLike } from "./keymap";

function key(
  k: string,
  mods: Partial<Record<"meta" | "ctrl" | "alt" | "shift", boolean>> = {},
  code = `Key${k.toUpperCase()}`,
): KeyLike {
  return {
    key: k,
    code,
    metaKey: mods.meta ?? false,
    ctrlKey: mods.ctrl ?? false,
    altKey: mods.alt ?? false,
    shiftKey: mods.shift ?? false,
  };
}

describe("global keys", () => {
  it("switch screens with ⌘ or Ctrl and a digit", () => {
    expect(keyAction(key("5", { meta: true }), false)).toEqual({
      kind: "screen",
      id: "inspect",
    });
    expect(keyAction(key("1", { ctrl: true }), false)).toEqual({
      kind: "screen",
      id: "library",
    });
  });

  it("open the palette and save with ⌘K and ⌘S, in a text field too", () => {
    expect(keyAction(key("k", { meta: true }), false)).toEqual({ kind: "palette" });
    expect(keyAction(key("K", { ctrl: true }), true)).toEqual({ kind: "palette" });
    expect(keyAction(key("s", { meta: true }), true)).toEqual({ kind: "save" });
    expect(keyAction(key("s", { meta: true, shift: true }), false)).toBeNull();
  });

  it("step views with [ and ], cameras with , and ., runs with ⇧[ and ⇧]", () => {
    expect(keyAction(key("[", {}, "BracketLeft"), false)).toEqual({
      kind: "view",
      delta: -1,
    });
    expect(keyAction(key("]", {}, "BracketRight"), false)).toEqual({
      kind: "view",
      delta: 1,
    });
    expect(keyAction(key(",", {}, "Comma"), false)).toEqual({
      kind: "camera",
      delta: -1,
    });
    expect(keyAction(key(".", {}, "Period"), false)).toEqual({
      kind: "camera",
      delta: 1,
    });
    // `⇧[` types "{" on a US layout: the physical key decides.
    expect(keyAction(key("{", { shift: true }, "BracketLeft"), false)).toEqual({
      kind: "run",
      delta: -1,
    });
    expect(keyAction(key("}", { shift: true }, "BracketRight"), false)).toEqual({
      kind: "run",
      delta: 1,
    });
  });

  it("clear with Escape and show the shortcuts with ?", () => {
    expect(keyAction(key("Escape", {}, "Escape"), false)).toEqual({ kind: "escape" });
    expect(keyAction(key("?", { shift: true }, "Slash"), false)).toEqual({
      kind: "help",
    });
  });

  it("never take a bare key while the user types", () => {
    for (const k of [
      key("[", {}, "BracketLeft"),
      key(",", {}, "Comma"),
      key("Escape", {}, "Escape"),
      key("?", { shift: true }, "Slash"),
    ]) {
      expect(keyAction(k, true)).toBeNull();
    }
  });

  it("leave other keys and Alt chords alone", () => {
    expect(keyAction(key("a"), false)).toBeNull();
    expect(keyAction(key("[", { alt: true }, "BracketLeft"), false)).toBeNull();
    expect(keyAction(key("1", { meta: true, alt: true }), false)).toBeNull();
    expect(keyAction(key("z", { meta: true }), false)).toBeNull();
  });
});
