import { describe, expect, it } from "vitest";
import { BUILTIN_PRESETS } from "./presets";

describe("BUILTIN_PRESETS sample ids", () => {
  const enabled = BUILTIN_PRESETS.filter((p) => p.disabled !== true);

  it("has at least one enabled preset to check", () => {
    expect(enabled.length).toBeGreaterThan(0);
  });

  it("names a plain catalogue id, not a path", () => {
    for (const preset of enabled) {
      expect("sample" in preset.source).toBe(true);
      if ("sample" in preset.source) {
        expect(preset.source.sample).toMatch(/^[a-z0-9_]+$/);
      }
    }
  });
});
